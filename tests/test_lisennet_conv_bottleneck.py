"""Tests for the NPU-mappable dual-path CONV bottleneck (bottleneck="conv").

The conv variant replaces LiSenNet's dual-path GRU bottleneck with convolutions
so the model compiles to the STM32N6 Neural-ART NPU. These tests pin the
properties that make it deployable:

  * it builds and forwards end-to-end;
  * it contains no ``nn.LayerNorm`` and no GRU/RNN module (the two Neural-ART
    blockers) — norms are the already-primitive ``CustomLayerNorm``;
  * it is genuinely causal in time (perturbing a future frame leaves earlier
    ``est_mag`` outputs bit-for-bit unchanged);
  * feeding it one STFT frame at a time (FIFO ring buffers on the causal
    time-convs) reproduces the offline whole-utterance mask to fp tolerance;
  * the default ``bottleneck="rnn"`` path is untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lisennet.model import DualPathConv, DualPathRNN, build_lisennet
from lisennet.streaming import (
    disable_streaming, enable_streaming, stream_est_mag,
)


CONV = dict(num_channels=16, n_blocks=2, bottleneck="conv",
            intra_kernel=7, inter_kernel=3, inter_dilations=(1, 2, 4))


def _spec(b=1, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = 257
    mag = torch.rand(b, n, f, generator=g)
    pha = (torch.rand(b, n, f, generator=g) * 2 - 1) * torch.pi
    return mag, pha


def _offline_est_mag(model, mag, pha):
    feat = model.build_features(mag, pha)
    with torch.no_grad():
        return model.apply_mask(model.predict_mask(feat), mag)


def test_conv_variant_builds_and_forwards():
    torch.manual_seed(0)
    model = build_lisennet(CONV).eval()
    assert model.bottleneck == "conv"
    # The conv bottleneck is lighter than the GRU one (~37k); just bound it sanely.
    n = sum(p.numel() for p in model.parameters())
    assert 15_000 < n < 40_000, f"unexpected conv param count: {n:,}"
    x = torch.randn(2, 32000)
    with torch.no_grad():
        r = model(x, x)
    assert r["est"].shape == (2, 32000)
    assert torch.isfinite(r["est"]).all()
    assert (r["est_mag"] >= 0).all()


def test_conv_variant_has_no_layernorm_or_gru():
    """The two Neural-ART blockers must be absent from the conv variant."""
    model = build_lisennet(CONV).eval()
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules()), \
        "nn.LayerNorm present — the 2-axis LayerNorm blocker is back"
    assert not any(isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)) for m in model.modules()), \
        "a recurrent module is present — GRU recurrence does not map to the NPU"
    assert not any(isinstance(m, DualPathRNN) for m in model.modules())
    assert any(isinstance(m, DualPathConv) for m in model.modules())


def test_conv_variant_is_causal():
    torch.manual_seed(0)
    model = build_lisennet(CONV).eval()
    mag, pha = _spec(1, 40, seed=1)
    off = _offline_est_mag(model, mag, pha)

    t0 = 20
    mag2, pha2 = mag.clone(), pha.clone()
    mag2[:, t0:] += 5.0
    pha2[:, t0:] = (torch.rand(1, mag.shape[1] - t0, mag.shape[2]) * 2 - 1) * torch.pi
    off2 = _offline_est_mag(model, mag2, pha2)

    delta = (off[:, :t0] - off2[:, :t0]).abs().max().item()
    assert delta == 0.0, f"future frame leaked into past outputs: max|delta|={delta:.2e}"


def test_conv_variant_streaming_matches_offline():
    torch.manual_seed(0)
    model = build_lisennet(CONV).eval()
    mag, pha = _spec(1, 40, seed=2)
    off = _offline_est_mag(model, mag, pha)
    s = stream_est_mag(model, mag, pha)
    err = (off - s).abs().max().item()
    assert err < 1e-4, f"conv streaming diverged from offline: {err:.3e}"


def test_conv_variant_streaming_batched():
    torch.manual_seed(0)
    model = build_lisennet(CONV).eval()
    mag, pha = _spec(3, 25, seed=3)
    off = _offline_est_mag(model, mag, pha)
    s = stream_est_mag(model, mag, pha)
    assert torch.allclose(off, s, atol=1e-4)


def test_toggling_streaming_leaves_offline_unchanged():
    torch.manual_seed(0)
    model = build_lisennet(CONV).eval()
    mag, pha = _spec(1, 20, seed=4)
    before = _offline_est_mag(model, mag, pha)
    enable_streaming(model)
    disable_streaming(model)
    after = _offline_est_mag(model, mag, pha)
    assert torch.allclose(before, after)


def test_default_bottleneck_is_rnn():
    """The faithful GRU model stays the default so its checkpoints keep loading."""
    model = build_lisennet({"num_channels": 16, "n_blocks": 2}).eval()
    assert model.bottleneck == "rnn"
    assert any(isinstance(m, DualPathRNN) for m in model.modules())
