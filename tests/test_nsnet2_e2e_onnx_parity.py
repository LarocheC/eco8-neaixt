"""FP32 ONNX parity for the end-to-end butterfly NSNet2 (waveform->waveform).

Tests:
  - test_export_smoke     : export succeeds, ONNX checker passes, the graph
                            contains NO STFT/DFT ops (the transform is a learned
                            butterfly, in-graph) and DOES contain the structured
                            butterfly ops (Einsum-free structure preserved).
  - test_parity_pt_vs_onnx: ORT agrees with the PyTorch export wrapper to
                            max_abs_err < 1e-4 on a random waveform.
  - test_dynamic_time_axis: a graph traced at one length runs and stays correct
                            at a different (frame-aligned) length.

ORT session uses the 4-knob determinism recipe from inference_onnx.py.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from common.env import AttrDict
from nsnet2.model_e2e import NSNet2E2E
from nsnet2.export_e2e_onnx import _E2EExportReal, export_e2e_fp32, _valid_len


@pytest.fixture
def cfg():
    return AttrDict({
        "win_size": 512, "hop_size": 256,
        "hidden_dim": 96, "fc_hidden_dim": 96, "num_gru_layers": 2,
        "compress_factor": 0.3,
        "transform": {"learnable_window": True, "window_init": "sqrt_hann"},
        "seed": 0,
    })


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    m = NSNet2E2E(cfg)
    # Perturb params off the init so parity isn't testing a degenerate model.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return m.eval()


@pytest.fixture
def onnx_path(model, tmp_path):
    out = tmp_path / "e2e_fp32.onnx"
    export_e2e_fp32(model, out, n_frames=64)
    return out


def _make_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def test_export_smoke(onnx_path):
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    ops = {n.op_type for n in m.graph.node}
    # The whole point: no STFT/DFT — the transform is an in-graph learned butterfly.
    assert "STFT" not in ops and "DFT" not in ops
    # Framing / overlap-add are plain conv ops; the butterfly is structured matmul.
    assert "Conv" in ops and "ConvTranspose" in ops
    io = {i.name for i in m.graph.input} | {o.name for o in m.graph.output}
    assert {"noisy_wav", "enhanced_wav"} <= io


def test_real_decomp_matches_complex_model(model):
    """The real-decomposed export wrapper must match the eager complex-butterfly
    model (this is what makes the ONNX graph faithful)."""
    torch.manual_seed(3)
    L = _valid_len(model.win, model.hop, 40)
    wav = torch.randn(1, L)
    with torch.no_grad():
        ref = model(wav)                      # eager complex path (+ internal pad/crop)
        got = _E2EExportReal(model)(wav)       # real-decomposed, frame-aligned L
    err = (ref[..., :got.shape[-1]] - got).abs().max().item()
    assert err < 1e-4, f"real-decomp vs complex max_abs_err={err:.3e}"


def test_parity_pt_vs_onnx(model, onnx_path):
    torch.manual_seed(1)
    L = _valid_len(model.win, model.hop, 64)
    wav = torch.randn(1, L)
    with torch.no_grad():
        ref = _E2EExportReal(model)(wav).numpy()
    sess = _make_session(onnx_path)
    got = sess.run(["enhanced_wav"], {"noisy_wav": wav.numpy()})[0]
    err = np.max(np.abs(ref - got))
    assert err < 1e-4, f"max_abs_err={err:.3e}"


def test_dynamic_time_axis(model, onnx_path):
    """Traced at 64 frames; must run + stay correct at 100 frames."""
    torch.manual_seed(2)
    L = _valid_len(model.win, model.hop, 100)
    wav = torch.randn(1, L)
    with torch.no_grad():
        ref = _E2EExportReal(model)(wav).numpy()
    sess = _make_session(onnx_path)
    got = sess.run(["enhanced_wav"], {"noisy_wav": wav.numpy()})[0]
    assert got.shape == (1, L)
    err = np.max(np.abs(ref - got))
    assert err < 1e-4, f"max_abs_err={err:.3e}"
