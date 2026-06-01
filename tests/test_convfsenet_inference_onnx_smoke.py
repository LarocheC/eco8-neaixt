"""Smoke tests for inference_onnx_convfsenet.

Avoids the HF VBD download by:
  - Building a random-init model in-process,
  - Exporting to FP32 ONNX + quantizing to int8 (synthetic calib),
  - Driving enhance_one_utterance directly on a synthetic noisy waveform.

Verifies:
  - state_shapes_from_h derives the right per-block buffer widths.
  - _run_session_frame_by_frame produces a (B, F, T)-shape mask in (0, 1).
  - enhance_one_utterance returns finite audio of the right length, both in
    single-session and dual-session modes.
  - RTF is positive and finite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from convfsenet.model import ConvFSENet_QuantFriendly
from common.env import AttrDict
from convfsenet.export_onnx import export_streaming_fp32
from convfsenet.inference_onnx import (
    _make_session,
    _run_session_frame_by_frame,
    enhance_one_utterance,
    state_shapes_from_h,
)
from convfsenet.streaming import (
    ConvFSENetStreamingFast,
    ConvFSENetStreamingONNX,
)
from convfsenet.quant import quantize_fp32_onnx
from tests.test_convfsenet_quant import _SyntheticHFDataset


def _fresh_offline() -> ConvFSENet_QuantFriendly:
    return ConvFSENet_QuantFriendly(
        n_fft=512, win_length=512, n_features=257,
        n_channels_res=128, n_channels_conv=256, kernel_size=3,
        n_blocks=3, n_stacks=3,
        extractor_type="mag", compress_factor=None, causal=True,
        loss=None, preproc=None, postproc=None,
    )


def _randomize_bn(m: nn.Module) -> None:
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                nn.init.normal_(mod.weight, 1.0, 0.1)
                nn.init.normal_(mod.bias, 0.0, 0.1)
                mod.running_mean.normal_(0.0, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
                mod.num_batches_tracked.fill_(1)


@pytest.fixture
def h():
    return AttrDict({
        "n_fft": 512, "win_length": 512, "hop_size": 256, "win_size": 512,
        "n_features": 257,
        "n_channels_res": 128, "n_channels_conv": 256,
        "kernel_size": 3, "n_blocks": 3, "n_stacks": 3,
        "extractor_type": "mag", "compress_factor": 1.0,
        "causal": True,
        "seed": 0,
        "sampling_rate": 16000,
        "calibration": AttrDict({
            "num_utterances": 3, "seed": 0, "frames_per_utterance": 16,
        }),
    })


@pytest.fixture
def offline_model():
    torch.manual_seed(0)
    m = _fresh_offline()
    _randomize_bn(m)
    m.eval()
    return m


@pytest.fixture
def fp32_path(offline_model, tmp_path):
    p = tmp_path / "g_best_fp32.onnx"
    export_streaming_fp32(offline_model, p)
    return p


@pytest.fixture
def int8_path(offline_model, fp32_path, h, tmp_path):
    p = tmp_path / "g_best.onnx"
    fast = ConvFSENetStreamingFast(offline_model).eval()
    onnx_view = ConvFSENetStreamingONNX(fast).eval()
    hf = _SyntheticHFDataset(n_train=4, n_test=2, sr=16000, length_s=0.5)
    quantize_fp32_onnx(fp32_path, p, fast, onnx_view, h, hf)
    return p


# ---- helper-level tests ----------------------------------------------------


def test_state_shapes_from_h(h):
    """Dilations cycle [1, 2, 4] per stack; buffer widths = (K-1)*D = [2, 4, 8]."""
    shapes = state_shapes_from_h(h)
    assert len(shapes) == h.n_blocks * h.n_stacks
    # All blocks have C = n_channels_conv.
    assert all(C == h.n_channels_conv for C, _ in shapes)
    # Per-stack buffer-width sequence.
    expected = [(h.n_channels_conv, w) for _ in range(h.n_stacks) for w in (2, 4, 8)]
    assert shapes == expected


def test_run_session_frame_by_frame_shapes(fp32_path, h):
    sess = _make_session(fp32_path)
    shapes = state_shapes_from_h(h)
    B, F, T = 1, h.n_features, 20
    noisy_mag = np.abs(np.random.default_rng(0).standard_normal((B, F, T))).astype(np.float32)
    mask, rtf_s = _run_session_frame_by_frame(sess, noisy_mag, shapes, measure_rtf=True)
    assert mask.shape == (B, F, T)
    assert mask.dtype == np.float32
    assert (mask >= 0.0).all() and (mask <= 1.0).all()
    assert rtf_s > 0.0 and np.isfinite(rtf_s)


# ---- enhance_one_utterance: single-session ---------------------------------


def _synthetic_noisy_wav(sr=16000, length_s=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * length_s)) / sr
    clean = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1100 * t)
    noise = 0.1 * rng.standard_normal(len(t))
    return (clean + noise).astype(np.float32)


def test_enhance_one_utterance_single_session(fp32_path, h):
    sess = _make_session(fp32_path)
    shapes = state_shapes_from_h(h)
    noisy = _synthetic_noisy_wav(sr=int(h.sampling_rate), length_s=1.0)
    primary, fp32_audio, rtf = enhance_one_utterance(
        noisy, sess, h, shapes, fp32_session=None,
    )
    assert primary.shape == noisy.shape, (
        f"output length should match input: {primary.shape} vs {noisy.shape}"
    )
    assert primary.dtype == np.float32
    assert np.isfinite(primary).all(), "output contains non-finite samples"
    assert fp32_audio is None, "single-session mode should return None for fp32_audio"
    assert rtf > 0.0 and np.isfinite(rtf)


def test_enhance_one_utterance_dual_session(fp32_path, int8_path, h):
    """Dual-session mode populates both audio outputs; both shapes match."""
    primary_sess = _make_session(int8_path)
    fp32_sess = _make_session(fp32_path)
    shapes = state_shapes_from_h(h)
    noisy = _synthetic_noisy_wav(sr=int(h.sampling_rate), length_s=1.0)
    primary, fp32_audio, rtf = enhance_one_utterance(
        noisy, primary_sess, h, shapes, fp32_session=fp32_sess,
    )
    assert primary.shape == noisy.shape
    assert fp32_audio is not None
    assert fp32_audio.shape == noisy.shape
    assert np.isfinite(primary).all() and np.isfinite(fp32_audio).all()
    # int8 mask should be close-ish to fp32 mask, but we don't assert tightly —
    # random-init weights + synthetic calib = loose bound.
    assert np.abs(primary - fp32_audio).mean() < 1.0, (
        "int8 vs fp32 mean abs diff > 1.0 — catastrophic divergence"
    )


def test_enhance_handles_short_utterance(fp32_path, h):
    """A 0.25 s utterance has T=~16 frames — exercise the buffer warmup region."""
    sess = _make_session(fp32_path)
    shapes = state_shapes_from_h(h)
    noisy = _synthetic_noisy_wav(sr=int(h.sampling_rate), length_s=0.25)
    primary, _, rtf = enhance_one_utterance(noisy, sess, h, shapes, fp32_session=None)
    assert primary.shape == noisy.shape
    assert np.isfinite(primary).all()
