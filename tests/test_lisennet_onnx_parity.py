"""FP32 ONNX export + ORT parity for LiSenNet's mask sub-network — the deploy gate.

Proves the mask sub-network exports to a portable ONNX graph and that ONNX
Runtime reproduces PyTorch numerically. Covers:

  - export smoke: the graph passes the ONNX checker and the recurrent DPR GRUs
    survive export (rather than being silently dropped/decomposed wrong).
  - parity: ORT matches PyTorch on the enhanced magnitude.
  - dynamic time: a graph traced at one length runs and stays correct at another.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from lisennet.model import build_lisennet
from lisennet.export_onnx import LiSenNetONNX, export_fp32


SMALL = dict(num_channels=16, n_blocks=2)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_lisennet(SMALL).eval()


@pytest.fixture
def onnx_path(model, tmp_path):
    return export_fp32(model, tmp_path / "lisennet.onnx", n_frames=32)


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _feat(model, b, t, seed):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(b, t, model.n_freqs, generator=g)
    pha = (torch.rand(b, t, model.n_freqs, generator=g) * 2 - 1) * torch.pi
    return model.build_features(mag, pha)


def test_export_smoke(onnx_path):
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    op_types = {n.op_type for n in m.graph.node}
    # LiSenNet's bottleneck is dual-path-recurrent: the GRUs must survive export.
    assert "GRU" in op_types, "DPR GRUs missing from the graph"
    assert "Conv" in op_types
    assert [i.name for i in m.graph.input] == ["feat"]
    assert [o.name for o in m.graph.output] == ["est_mag"]


def test_session_runs(onnx_path, model):
    sess = _session(onnx_path)
    f = model.n_freqs
    out = sess.run(["est_mag"], {"feat": np.zeros((1, 3, 16, f), np.float32)})
    assert out[0].shape == (1, 16, f)


def test_parity_pt_vs_onnx(model, onnx_path):
    feat = _feat(model, 1, 24, seed=1)
    with torch.no_grad():
        ref = LiSenNetONNX(model)(feat).numpy()
    sess = _session(onnx_path)
    out = sess.run(["est_mag"], {"feat": feat.numpy()})[0]
    err = float(np.abs(ref - out).max())
    assert err < 1e-4, f"magnitude parity failed: {err:.3e}"


def test_dynamic_time_axis(model, onnx_path):
    """A graph traced at T=32 must run and stay correct at a different T."""
    feat = _feat(model, 1, 20, seed=2)             # != 32 used to trace
    with torch.no_grad():
        ref = LiSenNetONNX(model)(feat).numpy()
    sess = _session(onnx_path)
    out = sess.run(["est_mag"], {"feat": feat.numpy()})[0]
    assert out.shape == (1, 20, model.n_freqs)
    err = float(np.abs(ref - out).max())
    assert err < 1e-4, f"dynamic-time parity failed: {err:.3e}"
