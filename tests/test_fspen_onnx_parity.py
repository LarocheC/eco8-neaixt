"""FP32 ONNX export + ORT parity for FSPEN — the deploy gate.

Proves the enhancer exports to a portable ONNX graph and that ONNX Runtime
reproduces PyTorch numerically. Covers:

  - export smoke: the graph passes the ONNX checker and the dual-path GRUs
    survive export (rather than being silently dropped/decomposed wrong).
  - parity: ORT matches PyTorch on the enhanced spectrum.
  - dynamic axes: a graph traced at one (B, T) runs and stays correct at others.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from fspen.model import build_fspen
from fspen.export_onnx import FSPENONNX, export_fp32


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_fspen().eval()


@pytest.fixture
def onnx_path(model, tmp_path):
    return export_fp32(model, tmp_path / "fspen.onnx", n_frames=32)


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _spec(model, b, t, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, 2, model.n_freqs, generator=g)


def test_export_smoke(onnx_path):
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    op_types = {n.op_type for n in m.graph.node}
    # FSPEN's sequence core is dual-path-recurrent: the GRUs must survive export.
    assert "GRU" in op_types, "DPE GRUs missing from the graph"
    assert "Conv" in op_types and "ConvTranspose" in op_types
    assert [i.name for i in m.graph.input] == ["spec"]
    assert [o.name for o in m.graph.output] == ["est_spec"]


def test_session_runs(onnx_path, model):
    sess = _session(onnx_path)
    f = model.n_freqs
    out = sess.run(["est_spec"], {"spec": np.zeros((1, 16, 2, f), np.float32)})
    assert out[0].shape == (1, 16, 2, f)


def test_parity_pt_vs_onnx(model, onnx_path):
    spec = _spec(model, 1, 24, seed=1)
    with torch.no_grad():
        ref = FSPENONNX(model)(spec).numpy()
    sess = _session(onnx_path)
    out = sess.run(["est_spec"], {"spec": spec.numpy()})[0]
    err = float(np.abs(ref - out).max())
    assert err < 1e-4, f"spectrum parity failed: {err:.3e}"


def test_dynamic_axes(model, onnx_path):
    """A graph traced at (B=1, T=32) must run and stay correct at other sizes."""
    spec = _spec(model, 2, 20, seed=2)             # != trace-time (1, 32)
    with torch.no_grad():
        ref = FSPENONNX(model)(spec).numpy()
    sess = _session(onnx_path)
    out = sess.run(["est_spec"], {"spec": spec.numpy()})[0]
    assert out.shape == (2, 20, 2, model.n_freqs)
    err = float(np.abs(ref - out).max())
    assert err < 1e-4, f"dynamic-axes parity failed: {err:.3e}"
