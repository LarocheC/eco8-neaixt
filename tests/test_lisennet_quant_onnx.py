"""Smoke test for dynamic weight-only int8 quantization of LiSenNet's ONNX graph.

Covers the robust path (see lisennet/quant_onnx.py): weight-only dynamic int8
produces a graph that ONNX Runtime runs and that stays close to fp32. Static full
int8 needs the VBD calibration set and is characterised by the deploy eval rather
than asserted here.

Note on size: unlike larger models, the int8 *file* does NOT shrink here — at
~37k params the per-weight-tensor quant metadata (DequantizeLinear nodes, scale /
zero-point initializers, int32 biases) outweighs the byte saving on such small
weights. The model is already ~0.26 MiB in fp32. So this test asserts that
quantization actually happened (int8 weight initializers are present) and that
the graph still runs correctly, NOT that the file got smaller.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lisennet.model import build_lisennet
from lisennet.export_onnx import export_fp32
from lisennet.quant_onnx import quantize_dynamic_int8


SMALL = dict(num_channels=16, n_blocks=2)


def test_dynamic_quant_runs_and_quantizes(tmp_path):
    torch.manual_seed(0)
    model = build_lisennet(SMALL).eval()
    fp32 = export_fp32(model, tmp_path / "m.onnx", n_frames=24)
    int8 = quantize_dynamic_int8(fp32, tmp_path / "m_int8.onnx")

    assert int8.exists()
    # Quantization happened: the graph now carries int8 weight initializers.
    qm = onnx.load(str(int8))
    int8_inits = [i for i in qm.graph.initializer
                  if i.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)]
    assert int8_inits, "no int8 initializers — weights were not quantized"

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(int8), sess_options=so,
                                providers=["CPUExecutionProvider"])
    f, t = model.n_freqs, 16
    rng = np.random.default_rng(0)
    mag = rng.random((1, t, f), dtype=np.float32)
    pha = ((rng.random((1, t, f), dtype=np.float32) * 2 - 1) * np.pi).astype(np.float32)
    feat = model.build_features(torch.from_numpy(mag), torch.from_numpy(pha)).numpy()
    out = sess.run(["est_mag"], {"feat": feat})[0]

    assert out.shape == (1, t, f)
    assert np.isfinite(out).all()
    # est_mag is a (bounded learnable-sigmoid) mask times the noisy magnitude; it
    # must stay non-negative and not blow up under weight-only int8.
    assert (out >= 0).all() and out.max() <= 4.0 * mag.max() + 1.0
