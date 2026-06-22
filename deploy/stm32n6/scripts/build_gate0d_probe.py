#!/usr/bin/env python3
"""Build the Gate-0D 2-D-conv probe ONNX (int8 QDQ) for the STM32N6 deploy box.

Track 3 (leg C) hypothesizes that a 2-D conv over the [F, T] grid lifts MAC-array
utilization. Every conv compiled so far is h:1, and one Conv2d-int8 on degenerate
spatial failed (`lower_arith_set_in_batch`). Before building any 2-D front-end,
the deploy box must confirm a single Conv2d maps to HW at a real [h>1,w>1] extent.

This produces a minimal int8 QDQ ONNX:

    input  [1, 1, 256, 43]  (a [F=256, L+T=43] context grid, 1 channel)
    Conv2d(1 -> 24, kernel=3, stride=(1,2)), valid padding
    output [1, 24, 254, 21]

Run on the deploy box:
    make generate MODEL=deploy/stm32n6/gate0_artifacts/gate0d_conv2d_probe.onnx
then read network_generate_report.txt:
    PASS = compiles int8, maps to the convolutional accelerator (NPU/HW epoch),
           and the conv extent is [h>1, w>1] (NOT h:1).
    FAIL like the NSNet2 `lower_arith_set_in_batch` case => leg C is dead on this
           toolchain and Track 3 is cancelled.

No training: random weights are sufficient — this probes the COMPILER's HW
mapping of an int8 2-D conv, not model quality.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class _Probe(nn.Module):
    def __init__(self, c_out=24):
        super().__init__()
        self.conv = nn.Conv2d(1, c_out, kernel_size=3, stride=(1, 2), padding=0)

    def forward(self, x):  # [1, 1, 256, 43]
        return self.conv(x)


class _RandReader(CalibrationDataReader):
    def __init__(self, shape, n=16, seed=0):
        rng = np.random.default_rng(seed)
        self._data = iter([
            {"grid": np.abs(rng.standard_normal(shape)).astype(np.float32)} for _ in range(n)
        ])

    def get_next(self):
        return next(self._data, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--W", type=int, default=43, help="L+T context width")
    ap.add_argument("--c_out", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "gate0_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else out_dir / "gate0d_conv2d_probe.onnx"
    fp32 = out.with_suffix(".fp32.onnx")

    torch.manual_seed(0)
    model = _Probe(a.c_out).eval()
    shape = (1, 1, a.F, a.W)
    example = torch.zeros(*shape)
    torch.onnx.export(
        model, (example,), str(fp32),
        input_names=["grid"], output_names=["feat"],
        opset_version=17, dynamo=False,
    )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as _t:
        pre = Path(_t.name)
    try:
        quant_pre_process(str(fp32), str(pre))
        quantize_static(
            str(pre), str(out),
            calibration_data_reader=_RandReader(shape),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            per_channel=True, calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    finally:
        pre.unlink(missing_ok=True)

    m = onnx.load(str(out))
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"Wrote Gate-0D probe: {out}")
    print(f"  input grid [1,1,{a.F},{a.W}] -> Conv2d(1->{a.c_out}, k=3, stride=(1,2))")
    print(f"  int8 ops: {ops}")
    print(f"  fp32 reference: {fp32}")
    print("Deploy box: `make generate MODEL=" + str(out) + "`; "
          "PASS = int8 compiles, HW/NPU epoch, conv extent [h>1,w>1].")


if __name__ == "__main__":
    main()
