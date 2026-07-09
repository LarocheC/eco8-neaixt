"""Export a trained fully-blockdiag NSNet2 into an NPU-deployable int8 ONNX.

Works for any NSNet2 config whose `linear` AND `gru` backends are both
`blockdiag` (single block-diagonal factor; e.g. `blockdiag_8` nblocks=8,
`blockdiag_full` nblocks=4, `wide_blockdiag`) — the architecture dims (n_freq,
hidden, num GRU layers, block count) are all read from the checkpoint, nothing
is hard-coded. Configs with a dense GRU (`blockdiag_fc`) are rejected: their
GRU has no block structure to re-express here.

NOTE: this exporter is specific to the block-diagonal structure — it reads each
projection's single `.weight` factor `[nblocks, out_blksz, in_blksz]` and makes
the block-diagonal structure explicit as per-block MatMuls. The genuine
two-factor Monarch (`kind="monarch"`, MonarchLinear with `w1`/`w2` and a
permutation between them) has NO single `.weight` and needs a different NPU
decomposition (two block-diagonal MatMul stages with a channel-permutation
Gather/Reshape between them); that exporter does not exist yet.

Why this exists
---------------
The stock blockdiag ONNX (e.g. `cp_blockdiag_8/g_best.onnx`) does NOT compile
for the STM32N6 Neural-ART: the block-diagonal block-matmul exports as
`Einsum 'bkp,kqp->bkq'` wrapped in dynamic-shape `Pad`/`Reshape` plumbing that
the compiler's shape engine rejects (`Error in computation of shapes`).
Lowering the `Einsum` and constant-folding the shapes is not enough — the
residual rank-2/`Pad` layout still fails.

What works (this script)
------------------------
Re-express the model in the exact op vocabulary that the dense NSNet2 baseline
compiles with: **rank-2 `MatMul` + `Add`** (no `Einsum`, no `Pad`, no grouped
`Conv`). Each block-diagonal projection becomes per-block
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
    python deploy/stm32n6/host/export_blockdiag_npu.py \
        --checkpoint_dir cp_blockdiag_full --num_utterances 200 \
        --out_fp32 /tmp/blockdiag_npu_fp32.onnx --out_int8 /tmp/blockdiag_npu_int8.onnx

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
    blockdiag layer does with `F.pad` are folded away: the last input block just
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


class BlockdiagNPU(nn.Module):
    """Streaming fully-blockdiag NSNet2 with flat per-layer states and the GRU
    gate rewritten to avoid the constant-Sub. Dims (n_freq, hidden, num layers)
    are read from the trained model. forward(frame_in, h0_in, ..., h{L-1}_in)
    -> (mask, h0_out, ..., h{L-1}_out)."""

    def __init__(self, st: NSNet2Streaming):
        super().__init__()
        b = st.base
        if b.gru_kind != "blockdiag":
            raise ValueError(
                f"gru_kind={b.gru_kind!r}; this exporter needs a fully-blockdiag "
                f"config (both linear and gru = 'blockdiag'). blockdiag_fc (dense GRU) "
                f"is not supported. The genuine two-factor 'monarch' kind is also "
                f"not supported by this block-diagonal exporter.")
        self.L = int(st.num_layers)
        self.H = int(st.hidden_size)
        self.n_freq = int(b.fc_in.in_features)
        self.fc_in = BlockLin(b.fc_in)
        self.cells_x = nn.ModuleList([BlockLin(b.gru.cells[k].x_proj) for k in range(self.L)])
        self.cells_h = nn.ModuleList([BlockLin(b.gru.cells[k].h_proj) for k in range(self.L)])
        self.fc1 = BlockLin(b.fc1)
        self.fc2 = BlockLin(b.fc2)
        self.fc_out = BlockLin(b.fc_out)

    def _cell(self, k, x, h):
        H = self.H
        gx, gh = self.cells_x[k](x), self.cells_h[k](h)
        xr, xz, xn = gx[:, :H], gx[:, H:2 * H], gx[:, 2 * H:3 * H]
        hr, hz, hn = gh[:, :H], gh[:, H:2 * H], gh[:, 2 * H:3 * H]
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return n + z * (h - n)                         # == (1-z)*n + z*h

    def forward(self, frame_in, *states):              # [1,n_freq], L x [1,H]
        x = torch.relu(self.fc_in(frame_in))
        outs = []
        for k in range(self.L):
            x = self._cell(k, x, states[k])
            outs.append(x)
        y = torch.relu(self.fc1(x))
        y = torch.relu(self.fc2(y))
        mask = torch.sigmoid(self.fc_out(y))
        return (mask, *outs)


# --- calibration adapter -------------------------------------------------------

class _Rank2Reader(CalibrationDataReader):
    """Reshape the (frame_in,states_in) VBD calib stream into the flat-state inputs."""
    def __init__(self, inner, n_freq, hidden, num_layers):
        self.inner, self.n_freq, self.H, self.L = inner, n_freq, hidden, num_layers

    def get_next(self):
        d = self.inner.get_next()
        if d is None:
            return None
        fr, st = d["frame_in"], d["states_in"]
        out = {"frame_in": fr.reshape(1, self.n_freq).astype(np.float32)}
        for k in range(self.L):
            out[f"h{k}_in"] = st[k].reshape(1, self.H).astype(np.float32)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="cp_blockdiag_8")
    ap.add_argument("--num_utterances", type=int, default=200)
    ap.add_argument("--out_fp32", default="/tmp/blockdiag_npu_fp32.onnx")
    ap.add_argument("--out_int8", default="/tmp/blockdiag_npu_int8.onnx")
    a = ap.parse_args()

    st = NSNet2Streaming.from_checkpoint(str(Path(a.checkpoint_dir) / "g_best")).eval()
    model = BlockdiagNPU(st).eval()
    L, H, n_freq = model.L, model.H, model.n_freq
    print(f"config: n_freq={n_freq} hidden={H} num_layers={L}")

    # parity gate
    torch.manual_seed(0)
    sref = torch.zeros(L, 1, H)
    states = [torch.zeros(1, H) for _ in range(L)]
    err = 0.0
    for _ in range(30):
        fr = torch.randn(1, n_freq) * 0.5
        with torch.no_grad():
            mref, sref = st.forward_step(fr, sref)
            out = model(fr, *states)
        m, states = out[0], list(out[1:])
        err = max(err, (mref - m).abs().max().item(),
                  max((sref[k] - states[k]).abs().max().item() for k in range(L)))
    print(f"parity max abs err vs trained streaming: {err:.3e}")
    assert err < 1e-4, "conv-native model diverged from the trained model"

    in_names = ["frame_in"] + [f"h{k}_in" for k in range(L)]
    out_names = ["mask"] + [f"h{k}_out" for k in range(L)]
    dummy = (torch.randn(1, n_freq), *[torch.zeros(1, H) for _ in range(L)])
    torch.onnx.export(model, dummy, a.out_fp32, input_names=in_names,
                      output_names=out_names, opset_version=17)
    print(f"wrote FP32 ONNX: {a.out_fp32} (inputs={in_names})")

    # int8 PTQ (same recipe as nsnet2.quant, skip_optimization to avoid re-fusion)
    h = st.base.h
    cal = dict(getattr(h, "calibration", AttrDict({})))
    cal["num_utterances"] = int(a.num_utterances)
    h.calibration = AttrDict(cal)
    reader = _Rank2Reader(VBDCalibrationReader(st, h, load_voicebank_demand()),
                          n_freq, H, L)
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
