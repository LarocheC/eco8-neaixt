"""FP32 ONNX export + ORT parity for the causal BASENet — the deployment gate.

Proves the spectral network exports to a portable ONNX graph and that ONNX
Runtime reproduces PyTorch numerically. Covers:

  - FreqResample equivalence: the ONNX-friendly frequency pool/upsample matmuls
    are numerically identical to the AdaptiveAvgPool2d / nearest-interpolate ops
    they replace, so a checkpoint trained before the swap deploys unchanged.
  - export smoke: the graph passes the ONNX checker and contains the expected
    ops (the GRU and the cross-band-attention Einsums survive export).
  - parity: ORT matches PyTorch on magnitude and on the reconstructed complex
    spectrogram (the quantity that actually gets iSTFT'd).
  - dynamic time: a graph traced at one length runs at another.
"""

from __future__ import annotations

import math

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F

from basenet.model import BASENet, FreqResample
from basenet.export_onnx import export_fp32


SMALL = dict(base_channels=16, crn_hidden=48, crn_freq=8, dec_depth=1)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return BASENet(causal=True, **SMALL).eval()


@pytest.fixture
def onnx_path(model, tmp_path):
    return export_fp32(model, tmp_path / "basenet.onnx", n_frames=32)


def _session(path):
    """Deterministic single-thread ORT session (matches inference_onnx.py)."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _complex(mag, pha):
    return np.stack([mag * np.cos(pha), mag * np.sin(pha)], axis=-1)


# ---------------------------------------------------------------------------
# FreqResample equivalence — guarantees trained checkpoints are preserved
# ---------------------------------------------------------------------------


def test_freqresample_matches_adaptive_avg_pool():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 201, 13)                      # (B, C, F, N)
    got = FreqResample(201, 16, "avg")(x)
    ref = torch.nn.AdaptiveAvgPool2d((16, None))(x)     # the op it replaces
    assert torch.allclose(got, ref, atol=1e-6), \
        (got - ref).abs().max().item()


def test_freqresample_matches_nearest_interpolate():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 13)
    got = FreqResample(16, 201, "nearest")(x)
    ref = F.interpolate(x, size=(201, x.shape[-1]), mode="nearest")
    assert torch.allclose(got, ref, atol=1e-6), \
        (got - ref).abs().max().item()


# ---------------------------------------------------------------------------
# Export smoke
# ---------------------------------------------------------------------------


def test_export_smoke(onnx_path):
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    op_types = {n.op_type for n in m.graph.node}
    # BASENet is recurrent + attention-based: the GRU and the cross-band Einsums
    # must survive export rather than being silently dropped/decomposed wrong.
    assert "GRU" in op_types, "unidirectional GRU missing from the graph"
    assert "Einsum" in op_types, "cross-band attention / FreqResample Einsum missing"
    assert "Conv" in op_types
    assert [i.name for i in m.graph.input] == ["mag", "pha"]
    assert [o.name for o in m.graph.output] == ["mag_g", "pha_g"]


def test_session_runs(onnx_path, model):
    sess = _session(onnx_path)
    f = model.n_features
    feeds = {"mag": np.zeros((1, f, 16), np.float32),
             "pha": np.zeros((1, f, 16), np.float32)}
    out = sess.run(["mag_g", "pha_g"], feeds)
    assert out[0].shape == (1, f, 16) and out[1].shape == (1, f, 16)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_parity_pt_vs_onnx(model, onnx_path):
    torch.manual_seed(1)
    b, f, t = 1, model.n_features, 24
    mag = torch.rand(b, f, t)
    pha = (torch.rand(b, f, t) * 2 - 1) * math.pi
    with torch.no_grad():
        mag_g, pha_g, _ = model(mag, pha)

    sess = _session(onnx_path)
    o_mag, o_pha = sess.run(["mag_g", "pha_g"],
                            {"mag": mag.numpy(), "pha": pha.numpy()})

    mag_err = float(np.abs(mag_g.numpy() - o_mag).max())
    assert mag_err < 1e-4, f"magnitude parity failed: {mag_err:.3e}"

    # Compare the reconstructed complex spectrogram (what feeds the iSTFT). This
    # is continuous across the atan2 branch cut, so it's the meaningful parity
    # metric for phase — raw-angle diffs spike to ~2pi at the cut for an
    # identical complex value.
    cplx_err = float(np.abs(_complex(mag_g.numpy(), pha_g.numpy())
                            - _complex(o_mag, o_pha)).max())
    assert cplx_err < 1e-4, f"complex-spectrogram parity failed: {cplx_err:.3e}"


def test_dynamic_time_axis(model, onnx_path):
    """A graph traced at T=32 must run and stay correct at a different T."""
    torch.manual_seed(2)
    b, f, t = 1, model.n_features, 20            # != 32 used to trace
    mag = torch.rand(b, f, t)
    pha = (torch.rand(b, f, t) * 2 - 1) * math.pi
    with torch.no_grad():
        mag_g, pha_g, _ = model(mag, pha)
    sess = _session(onnx_path)
    o_mag, o_pha = sess.run(["mag_g", "pha_g"],
                            {"mag": mag.numpy(), "pha": pha.numpy()})
    assert o_mag.shape == (b, f, t)
    cplx_err = float(np.abs(_complex(mag_g.numpy(), pha_g.numpy())
                            - _complex(o_mag, o_pha)).max())
    assert cplx_err < 1e-4, f"dynamic-time parity failed: {cplx_err:.3e}"
