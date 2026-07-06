"""Smoke test for dynamic weight-only int8 quantization of FSPEN's ONNX graph.

Covers the robust path (see fspen/quant_onnx.py): weight-only dynamic int8
produces a graph that ONNX Runtime runs and that stays sane vs fp32. Static full
int8 needs the VBD calibration set and is characterised by the deploy eval
rather than asserted here.

Note on size: as with LiSenNet, at ~35k params the per-weight-tensor quant
metadata outweighs the byte saving, so the int8 *file* does not shrink; the
test asserts that quantization actually happened (int8 weight initializers are
present) and that the graph still runs correctly, NOT that the file got smaller.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import torch

from fspen.model import build_fspen
from fspen.export_onnx import export_fp32
from fspen.quant_onnx import quantize_dynamic_int8


def test_dynamic_quant_runs_and_quantizes(tmp_path):
    torch.manual_seed(0)
    model = build_fspen().eval()
    fp32 = export_fp32(model, tmp_path / "m.onnx", n_frames=24)
    int8 = quantize_dynamic_int8(fp32, tmp_path / "m_int8.onnx")

    assert int8.exists()
    # Quantization happened, and for a meaningful share of the weights: ORT's
    # quantize_dynamic leaves GRU/ConvTranspose nodes fp32 (~43% of this
    # model), so a single-initializer check would be blind to a coverage
    # regression (e.g. an ORT default change dropping MatMul). Pin > 50% of
    # the parameter elements landing in int8 initializers (currently ~57%).
    qm = onnx.load(str(int8))
    int8_inits = [i for i in qm.graph.initializer
                  if i.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)]
    assert int8_inits, "no int8 initializers — weights were not quantized"
    int8_elems = sum(int(np.prod(i.dims)) for i in int8_inits)
    total = sum(p.numel() for p in model.parameters())
    assert int8_elems > 0.5 * total, \
        f"only {int8_elems}/{total} weight elements quantized — coverage regressed"

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(int8), sess_options=so,
                                providers=["CPUExecutionProvider"])
    f, t = model.n_freqs, 16
    g = torch.Generator().manual_seed(0)
    spec = torch.randn(1, t, 2, f, generator=g).numpy()
    out = sess.run(["est_spec"], {"spec": spec})[0]

    assert out.shape == (1, t, 2, f)
    assert np.isfinite(out).all()
    # Weight-only int8 must stay close to the fp32 graph (measured max-abs
    # error ~0.05 on this input; the 0.5 bound leaves ~10x margin while still
    # failing on an all-zero or grossly wrong output).
    sess_fp32 = ort.InferenceSession(str(fp32), sess_options=so,
                                     providers=["CPUExecutionProvider"])
    ref = sess_fp32.run(["est_spec"], {"spec": spec})[0]
    err = float(np.abs(out - ref).max())
    assert err < 0.5, f"int8 diverged from fp32: max abs err {err:.3f}"
