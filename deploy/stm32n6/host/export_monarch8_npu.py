"""Export the trained monarch_8 NSNet2 into an NPU-deployable int8 ONNX.

Why this exists
---------------
The stock monarch ONNX (`cp_monarch_8/g_best.onnx`) does NOT compile for the
STM32N6 Neural-ART: the monarch block-matmul exports as `Einsum 'bkp,kqp->bkq'`
wrapped in dynamic-shape `Pad`/`Reshape` plumbing that the compiler's shape
engine rejects (`Error in computation of shapes`). Lowering the `Einsum` and
constant-folding the shapes is not enough — the residual rank-2/`Pad` layout
still fails.

What works (this script)
------------------------
Re-express the model in the exact op vocabulary that the dense NSNet2 baseline
compiles with: **rank-2 `MatMul` + `Add`** (no `Einsum`, no `Pad`, no grouped
`Conv`). Each monarch block-diagonal projection becomes per-block
`Slice` + `MatMul` + `Concat` (the block-diagonal structure made explicit).
The GRU update gate is rewritten `(1-z)*n + z*h == n + z*(h-n)` to drop the
scalar-constant `Sub` that trips the Neural-ART int8 elementwise HW lowering.
States are two flat `[1,400]` tensors in/out (no `[2,B,400]`+`Gather`).

The result is numerically identical to the trained model (mask cosine ~0.999
vs the stock int8) and maps to the NPU with weights resident in on-chip npuRAM
(the `n6-noextmem` profile) — i.e. it can run real-time, unlike the dense
baseline whose 2.70 MB weights overflow on-chip RAM.

Quantization mirrors `nsnet2.quant.quantize_checkpoint` (same VBD calibration,
200 utts, MinMax, per-channel QInt8) but with `skip_optimization=True` so ORT
does not re-fuse anything (see NSNET2_DEPLOYMENT_NOTES.md).

Usage
-----
    python deploy/stm32n6/host/export_monarch8_npu.py \
        --checkpoint_dir cp_monarch_8 --num_utterances 200 \
        --out_fp32 /tmp/monarch8_npu_fp32.onnx --out_int8 /tmp/monarch8_npu_int8.onnx

Then generate/load/profile per ONBOARD_MEASUREMENT.md, e.g.:
    stedgeai generate -m <out_int8> --target stm32n6 \
        --st-neural-art n6-noextmem@user_neuralart.json \
        --fix-parametric-shapes "{'B':1}" -n network -o /tmp/gen
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common.dataset import load_voicebank_demand
from common.env import AttrDict
from nsnet2.calibration import VBDCalibrationReader
from nsnet2.streaming import NSNet2Streaming


# --- conv-native (rank-2) model ------------------------------------------------

class BlockLin(nn.Module):
    """A torch_structured BlockdiagLinear re-expressed as per-block rank-2
    MatMuls (Slice + MatMul + Concat). Numerically identical; emits only the
    ops the Neural-ART maps to HW. The input-pad / output-trim that the stock
    monarch layer does with `F.pad` are folded away: the last input block just
    uses fewer columns, and the output is sliced to `out_features`."""

    def __init__(self, m):
        super().__init__()
        nb, ob, ib = m.weight.shape
        self.nb, self.ob, self.ib = nb, ob, ib
        self.in_f, self.out_f = m.in_features, m.out_features
        self.blocks = nn.ModuleList()
        for k in range(nb):
            nc = min(ib, self.in_f - k * ib)          # last block may be short (drops pad cols)
            lin = nn.Linear(nc, ob, bias=False)
            lin.weight.data = m.weight[k][:, :nc].clone()
            self.blocks.append(lin)
        self.bias = nn.Parameter(m.bias.clone())

    def forward(self, x):                              # x: [1, in_f]
        ys = [lin(x[:, k * self.ib: k * self.ib + lin.in_features])
              for k, lin in enumerate(self.blocks)]
        return torch.cat(ys, dim=-1)[:, :self.out_f] + self.bias


class Monarch8NPU(nn.Module):
    """Streaming monarch_8 with flat [1,400] states and the gate rewritten to
    avoid the constant-Sub. forward(frame_in, h0_in, h1_in) -> mask, h0, h1."""

    def __init__(self, st: NSNet2Streaming):
        super().__init__()
        b = st.base
        assert b.gru_kind == "monarch" and st.num_layers == 2 and st.hidden_size == 400
        self.fc_in = BlockLin(b.fc_in)
        self.x0 = BlockLin(b.gru.cells[0].x_proj); self.h0 = BlockLin(b.gru.cells[0].h_proj)
        self.x1 = BlockLin(b.gru.cells[1].x_proj); self.h1 = BlockLin(b.gru.cells[1].h_proj)
        self.fc1 = BlockLin(b.fc1); self.fc2 = BlockLin(b.fc2); self.fc_out = BlockLin(b.fc_out)

    @staticmethod
    def _cell(xp, hp, x, h):
        gx, gh = xp(x), hp(h)
        xr, xz, xn = gx[:, :400], gx[:, 400:800], gx[:, 800:1200]
        hr, hz, hn = gh[:, :400], gh[:, 400:800], gh[:, 800:1200]
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return n + z * (h - n)                         # == (1-z)*n + z*h

    def forward(self, frame_in, h0_in, h1_in):         # [1,257],[1,400],[1,400]
        h = torch.relu(self.fc_in(frame_in))
        a0 = self._cell(self.x0, self.h0, h, h0_in)
        a1 = self._cell(self.x1, self.h1, a0, h1_in)
        y = torch.relu(self.fc1(a1))
        y = torch.relu(self.fc2(y))
        mask = torch.sigmoid(self.fc_out(y))
        return mask, a0, a1


# --- calibration adapter -------------------------------------------------------

class _Rank2Reader(CalibrationDataReader):
    """Reshape the (frame_in,states_in) VBD calib stream into the flat-state inputs."""
    def __init__(self, inner): self.inner = inner

    def get_next(self):
        d = self.inner.get_next()
        if d is None:
            return None
        fr, st = d["frame_in"], d["states_in"]
        return {"frame_in": fr.reshape(1, 257).astype(np.float32),
                "h0_in": st[0].reshape(1, 400).astype(np.float32),
                "h1_in": st[1].reshape(1, 400).astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="cp_monarch_8")
    ap.add_argument("--num_utterances", type=int, default=200)
    ap.add_argument("--out_fp32", default="/tmp/monarch8_npu_fp32.onnx")
    ap.add_argument("--out_int8", default="/tmp/monarch8_npu_int8.onnx")
    a = ap.parse_args()

    st = NSNet2Streaming.from_checkpoint(str(Path(a.checkpoint_dir) / "g_best")).eval()
    model = Monarch8NPU(st).eval()

    # parity gate
    torch.manual_seed(0)
    sref = torch.zeros(2, 1, 400)
    h0 = torch.zeros(1, 400); h1 = torch.zeros(1, 400)
    err = 0.0
    for _ in range(30):
        fr = torch.randn(1, 257) * 0.5
        with torch.no_grad():
            mref, sref = st.forward_step(fr, sref)
            m, h0, h1 = model(fr, h0, h1)
        err = max(err, (mref - m).abs().max().item(),
                  (sref[0] - h0).abs().max().item(), (sref[1] - h1).abs().max().item())
    print(f"parity max abs err vs trained streaming: {err:.3e}")
    assert err < 1e-4, "conv-native model diverged from the trained model"

    torch.onnx.export(
        model, (torch.randn(1, 257), torch.zeros(1, 400), torch.zeros(1, 400)),
        a.out_fp32, input_names=["frame_in", "h0_in", "h1_in"],
        output_names=["mask", "h0_out", "h1_out"], opset_version=17)
    print(f"wrote FP32 ONNX: {a.out_fp32}")

    # int8 PTQ (same recipe as nsnet2.quant, skip_optimization to avoid re-fusion)
    h = st.base.h
    cal = dict(getattr(h, "calibration", AttrDict({})))
    cal["num_utterances"] = int(a.num_utterances)
    h.calibration = AttrDict(cal)
    reader = _Rank2Reader(VBDCalibrationReader(st, h, load_voicebank_demand()))
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as t:
        pre = Path(t.name)
    try:
        quant_pre_process(a.out_fp32, str(pre), skip_optimization=True)
        quantize_static(
            str(pre), a.out_int8, calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8, per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    finally:
        pre.unlink(missing_ok=True)
    mb = Path(a.out_int8).stat().st_size / 1e6
    print(f"wrote int8 ONNX: {a.out_int8} ({mb:.3f} MB)")


if __name__ == "__main__":
    main()
