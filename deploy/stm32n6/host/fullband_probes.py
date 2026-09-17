"""Probe graphs for the N6Net full-band branch hang (STM32N6570-DK).

The ``pool`` full-band branch (two 4x1 stride-4 convs, a full-height 8x1 conv,
broadcast Add onto the (1, 96, 128, 1) trunk) compiles to a clean HW blob but
hangs the NPU on target. ST's limits give two structural suspects:

* vertical kernels are native only up to height 3; larger ones are decomposed
  by the compiler (the 4x1 stride-4 convs and the 8x1 full-height conv);
* the height-1 result is broadcast onto a height-128 tensor.

Each probe below is the real trunk shape (96 x 128 x 1, int8 QDQ, a 5x1 conv
in front so the input is not the graph input) plus one variant of the branch.
Probes ``s4_only`` / ``fullh_only`` / ``gap_only`` isolate a suspect with no
broadcast; the rest are HW-plausible replacements for the branch as a whole.

    python deploy/stm32n6/host/fullband_probes.py OUT_DIR [probe ...]
    deploy/stm32n6/scripts/compile_n6net.sh OUT_DIR/<probe>.onnx OUT_DIR/<probe>_gen

Probe names, what they test, and whether the result feeds a broadcast Add:

    control      trunk only                                          -
    pool         the hanging branch as trained (4x1 s4, 4x1 s4, 8x1)  broadcast
    s4_only      one 4x1 stride-4 conv, 32-row output                 none
    fullh_only   3x1 s2 pyramid to 8 rows, then the 8x1 conv          none (height-1 output)
    gap_only     GlobalAveragePool + 1x1, height-1 output             none
    gap          GlobalAveragePool + 1x1 + ReLU                       broadcast
    pyr3         seven 3x1 stride-2 convs down to 1 row               broadcast
    avgpyr3      seven 3x1 stride-2 AveragePools + 1x1                broadcast
    pool_resize  the trained branch, then Resize 1 -> 128 rows       same-shape Add
    gap_resize   GAP + 1x1, then Resize 1 -> 128 rows                same-shape Add
    pool_tile    the trained branch, then Tile x128                  same-shape Add
    pool_native  three 4x1 s4 convs + a 2x1 conv (no kernel the        broadcast
                 compiler decomposes)
    pool_native_tile  pool_native, then Tile x128                    same-shape Add

What the compiler does with them (stedgeai 4.0.1, n6-noextmem-ec): the 8x1
full-height conv is the only layer it rewrites -- into eight masked 1x1
sub-convs, one per input row, each producing a height-1 output, summed by an
Add tree. 4x1 stride-4 and 5x1 convs stay single native convs. Resize 1->128
followed by Add is folded back into the same broadcast Add, so
``pool_resize`` / ``gap_resize`` do not avoid it; only Tile does (as one SW
epoch on the M55).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

C, H, CG = 96, 128, 48


def _pool_branch():
    return nn.ModuleList([nn.Conv2d(C, CG, (4, 1), stride=(4, 1)),
                          nn.Conv2d(CG, CG, (4, 1), stride=(4, 1)),
                          nn.Conv2d(CG, C, (H // 16, 1))])


def _pool_native_branch():
    return nn.ModuleList([nn.Conv2d(C, CG, (4, 1), stride=(4, 1)),
                          nn.Conv2d(CG, CG, (4, 1), stride=(4, 1)),
                          nn.Conv2d(CG, CG, (4, 1), stride=(4, 1)),
                          nn.Conv2d(CG, C, (2, 1))])


def _run_pool(mods, y):
    for m in mods:
        y = F.relu(m(y))
    return y


def _pyr3(c_in, c_out, n):
    chans = [c_in] + [CG] * (n - 1) + [c_out]
    return nn.ModuleList([nn.Conv2d(chans[i], chans[i + 1], (3, 1), stride=(2, 1), padding=(1, 0))
                          for i in range(n)])


class Probe(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.trunk = nn.Conv2d(C, C, (5, 1), padding=(2, 0))
        if kind in ("pool", "pool_resize", "pool_tile"):
            self.br = _pool_branch()
        elif kind in ("pool_native", "pool_native_tile"):
            self.br = _pool_native_branch()
        elif kind == "s4_only":
            self.br = nn.Conv2d(C, CG, (4, 1), stride=(4, 1))
        elif kind == "fullh_only":
            self.pre = _pyr3(C, CG, 4)                        # 128 -> 8 rows, kernels <= 3
            self.br = nn.Conv2d(CG, C, (H // 16, 1))
        elif kind in ("gap", "gap_only", "gap_resize"):
            self.br = nn.Conv2d(C, C, 1)
        elif kind == "pyr3":
            self.br = _pyr3(C, C, 7)                          # 128 -> 1 row
        elif kind == "avgpyr3":
            self.br = nn.Conv2d(C, C, 1)
        elif kind != "control":
            raise ValueError(kind)

    def forward(self, x):
        y = F.relu(self.trunk(x))
        k = self.kind
        if k == "control":
            return y
        if k == "s4_only":
            return y, F.relu(self.br(y))
        if k == "fullh_only":
            z = y
            for m in self.pre:
                z = F.relu(m(z))
            return y, F.relu(self.br(z))
        if k in ("gap", "gap_only", "gap_resize"):
            g = F.relu(self.br(y.mean(dim=(2, 3), keepdim=True)))
        elif k == "avgpyr3":
            g = y
            for _ in range(7):
                g = F.avg_pool2d(g, (3, 1), stride=(2, 1), padding=(1, 0))
            g = F.relu(self.br(g))
        else:                                                   # pool*, pyr3
            g = _run_pool(self.br, y)
        if k == "gap_only":
            return y, g
        if k in ("pool_resize", "gap_resize"):
            g = F.interpolate(g, size=(H, 1), mode="nearest")    # asymmetric / floor
        elif k in ("pool_tile", "pool_native_tile"):
            g = g.repeat(1, 1, H, 1)
        return y + g


PROBES = ["control", "pool", "s4_only", "fullh_only", "gap_only", "gap", "pyr3",
          "avgpyr3", "pool_resize", "gap_resize", "pool_tile", "pool_native",
          "pool_native_tile"]


def export(kind, out_dir: Path, n_calib=16):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.preprocess import quant_pre_process

    torch.manual_seed(0)
    m = Probe(kind).eval()
    x = torch.randn(1, C, H, 1)
    outs = ["y", "g"] if kind in ("s4_only", "fullh_only", "gap_only") else ["y"]
    fp32 = out_dir / f"{kind}.fp32.onnx"
    torch.onnx.export(m, x, str(fp32), input_names=["x"], output_names=outs,
                      opset_version=17, dynamo=False, do_constant_folding=True)
    # fold the shape arithmetic Resize/Tile export with, so the compiler sees constants
    folded = out_dir / f"{kind}.folded.onnx"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
    pre = out_dir / f"{kind}.pre.onnx"
    quant_pre_process(str(folded), str(pre), skip_symbolic_shape=True)

    class R(CalibrationDataReader):
        def __init__(self):
            g = torch.Generator().manual_seed(1)
            self.it = iter([{"x": torch.randn(1, C, H, 1, generator=g).numpy()}
                            for _ in range(n_calib)])

        def get_next(self):
            return next(self.it, None)

    q = out_dir / f"{kind}.onnx"
    quantize_static(str(pre), str(q), R(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    ops = sorted({n.op_type for n in onnx.load(str(q)).graph.node}
                 - {"QuantizeLinear", "DequantizeLinear"})
    return q, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("probes", nargs="*", default=PROBES)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for k in a.probes:
        q, ops = export(k, a.out_dir)
        print(f"{k:12s} {q}  ops={ops}")


if __name__ == "__main__":
    main()
