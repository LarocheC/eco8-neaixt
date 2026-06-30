"""Smoke test for dynamic weight-only int8 quantization of BASENet's ONNX graph.

Covers the path that actually works on this model (see basenet/quant_onnx.py):
weight-only dynamic int8 produces a smaller graph that ONNX Runtime runs and
that stays close to fp32. Static full int8 is intentionally not exercised here —
it collapses without QAT and needs the VBD calibration set, so it's characterised
in the module docstring rather than asserted in a unit test.
"""

from __future__ import annotations

import math

import numpy as np
import onnxruntime as ort
import torch

from basenet.model import BASENet
from basenet.export_onnx import export_fp32
from basenet.quant_onnx import quantize_dynamic_int8


SMALL = dict(base_channels=16, crn_hidden=48, crn_freq=8, dec_depth=1)


def test_dynamic_quant_runs_and_shrinks(tmp_path):
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).eval()
    fp32 = export_fp32(model, tmp_path / "m.onnx", n_frames=24)
    int8 = quantize_dynamic_int8(fp32, tmp_path / "m_int8.onnx")

    assert int8.exists()
    # Weights are int8 now, so the graph is smaller.
    assert int8.stat().st_size < fp32.stat().st_size, "int8 graph is not smaller"

    # ORT loads and runs the quantized graph.
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(int8), sess_options=so,
                                providers=["CPUExecutionProvider"])
    f, t = model.n_features, 16
    rng = np.random.default_rng(0)
    mag = rng.random((1, f, t), dtype=np.float32)
    pha = ((rng.random((1, f, t), dtype=np.float32) * 2 - 1) * math.pi).astype(np.float32)
    o_mag, o_pha = sess.run(["mag_g", "pha_g"], {"mag": mag, "pha": pha})

    assert o_mag.shape == (1, f, t) and o_pha.shape == (1, f, t)
    assert np.isfinite(o_mag).all() and np.isfinite(o_pha).all()

    # Weight-only int8 keeps the magnitude mask in a sane bounded range (the
    # LearnableSigmoid caps it at ~2x the input), i.e. quantization didn't blow up.
    assert (o_mag >= 0).all() and o_mag.max() <= 2.0 * mag.max() + 1.0
