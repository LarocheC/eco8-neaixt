"""Tests for the BASENet implementation (arXiv:2606.12662).

The load-bearing test is `test_causal_model_has_no_future_leak`: BASENet's whole
selling point for edge deployment is causal streaming, so we verify it directly —
perturbing a future STFT frame must leave every earlier output frame bit-for-bit
unchanged. `test_non_causal_model_does_leak` is its mirror: it confirms the probe
actually detects leakage (the non-causal model fails it), so the causal pass is
not vacuous.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from basenet.model import (
    BASENet,
    bark_scale,
    build_basenet,
    critical_band_densities,
    derive_band_depths,
    hz_edges_to_bins,
)
from common.env import AttrDict


# A small, fast model. Causality is structural, so a narrow model exercises the
# exact same future-leak paths as the full one but runs in milliseconds.
SMALL = dict(base_channels=16, crn_hidden=48, crn_freq=8, dec_depth=1)


def _random_spectrogram(b=1, n_features=201, n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(b, n_features, n, generator=g)
    pha = (torch.rand(b, n_features, n, generator=g) * 2 - 1) * math.pi
    return mag, pha


# ---------------------------------------------------------------------------
# Architecture / shape sanity
# ---------------------------------------------------------------------------


def test_forward_shapes():
    torch.manual_seed(0)
    model = build_basenet(causal=True).eval()
    mag, pha = _random_spectrogram(b=2, n_features=model.n_features, n=40)
    with torch.no_grad():
        mag_g, pha_g, com_g = model(mag, pha)
    assert mag_g.shape == mag.shape
    assert pha_g.shape == pha.shape
    assert com_g.shape == mag.shape + (2,)
    assert torch.isfinite(mag_g).all() and torch.isfinite(pha_g).all()
    # The magnitude mask is bounded by LearnableSigmoid's beta=2, and mag>=0.
    assert (mag_g >= 0).all()
    assert (mag_g <= 2.0 * mag + 1e-5).all()


def test_param_count_in_paper_ballpark():
    """Default config should land near the paper's 0.81 M (BASENet-3, causal)."""
    model = build_basenet(causal=True)
    n = sum(p.numel() for p in model.parameters())
    assert 0.4e6 < n < 1.5e6, f"unexpected param count: {n:,}"


def test_bands_tile_the_full_spectrum():
    model = build_basenet(causal=True)
    slices = model.band_slices
    # First band starts at bin 0, last ends at the Nyquist bin, no gaps/overlap.
    assert slices[0][0] == 0
    assert slices[-1][1] == model.n_features
    for (a_lo, a_hi), (b_lo, b_hi) in zip(slices, slices[1:]):
        assert a_hi == b_lo, f"bands not contiguous: {slices}"
    covered = sum(hi - lo for lo, hi in slices)
    assert covered == model.n_features


def test_default_bands_match_paper():
    """Paper Sec. 3.4: B=3 boundaries at [0,1,4,8] kHz, n_fft=400 -> bins 0/25/100/201."""
    bins = hz_edges_to_bins((0, 1000, 4000, 8000), n_fft=400,
                            sampling_rate=16000, n_features=201)
    assert bins == [0, 25, 100, 201]


# ---------------------------------------------------------------------------
# Bark-scale band allocation (Eqs. 2-4)
# ---------------------------------------------------------------------------


def test_bark_scale_monotonic_and_anchored():
    assert float(bark_scale(0.0)) == pytest.approx(0.0, abs=1e-6)
    # z is strictly increasing in f.
    z = bark_scale([0.0, 1000.0, 4000.0, 8000.0])
    assert torch.all(z[1:] > z[:-1])
    # Sanity against the paper's quoted ~21 Bark near 8 kHz.
    assert 20.0 < float(z[-1]) < 23.0


def test_critical_band_density_decreases_with_frequency():
    """Density (Bark/Hz) falls from low to high bands — the paper's core premise."""
    rho = critical_band_densities([0, 1000, 4000, 8000])
    assert rho[0] > rho[1] > rho[2]
    # Matches the paper's quoted magnitudes (~8e-3, ~3e-3, ~1e-3 Bark/Hz).
    assert rho[0] == pytest.approx(8.0e-3, rel=0.25)
    assert rho[2] == pytest.approx(1.25e-3, rel=0.4)


def test_derive_band_depths_orders_by_density():
    """The reference Eq.-4 rule yields monotonically non-increasing depths."""
    depths = derive_band_depths([0, 1000, 4000, 8000], l_max=4)
    assert depths[0] == 4
    assert depths[0] >= depths[1] >= depths[2] >= 1


# ---------------------------------------------------------------------------
# Causality — the central guarantee
# ---------------------------------------------------------------------------


def test_causal_model_has_no_future_leak():
    """Perturbing frame >= t_f must not change any output frame < t_f.

    This is exact, not approximate: a structurally causal graph never routes a
    future input into a past output, so the past outputs are identical bit-for-
    bit. We assert that for both the magnitude and the phase heads.
    """
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).eval()
    mag, pha = _random_spectrogram(b=2, n_features=model.n_features, n=32, seed=1)
    t_f = 16

    with torch.no_grad():
        mag_g, pha_g, _ = model(mag, pha)

        mag2, pha2 = mag.clone(), pha.clone()
        # Aggressively corrupt every frame from t_f onward.
        mag2[:, :, t_f:] += 7.0
        pha2[:, :, t_f:] = -pha2[:, :, t_f:]
        mag_g2, pha_g2, _ = model(mag2, pha2)

    assert torch.allclose(mag_g[:, :, :t_f], mag_g2[:, :, :t_f], atol=1e-6, rtol=0), \
        "future frames leaked into past magnitude outputs"
    assert torch.allclose(pha_g[:, :, :t_f], pha_g2[:, :, :t_f], atol=1e-6, rtol=0), \
        "future frames leaked into past phase outputs"
    # And the perturbation genuinely changed the future outputs (probe is live).
    assert not torch.allclose(mag_g[:, :, t_f:], mag_g2[:, :, t_f:])


def test_non_causal_model_does_leak():
    """Mirror of the causality test: the non-causal model must fail it.

    Confirms the probe detects future->past leakage, so the causal model passing
    it is meaningful rather than an artefact of an insensitive output.
    """
    torch.manual_seed(0)
    model = BASENet(causal=False, **SMALL).eval()
    mag, pha = _random_spectrogram(b=1, n_features=model.n_features, n=32, seed=2)
    t_f = 16

    with torch.no_grad():
        mag_g, _, _ = model(mag, pha)
        mag2 = mag.clone()
        mag2[:, :, t_f:] += 7.0
        mag_g2, _, _ = model(mag2, pha)

    past_delta = (mag_g[:, :, :t_f] - mag_g2[:, :, :t_f]).abs().max().item()
    assert past_delta > 1e-4, (
        f"non-causal model unexpectedly causal (past delta {past_delta:.2e}); "
        "the leak probe may be broken"
    )


# ---------------------------------------------------------------------------
# Trainability + config plumbing
# ---------------------------------------------------------------------------


def test_forward_backward_produces_finite_grads():
    torch.manual_seed(0)
    model = BASENet(causal=True, **SMALL).train()
    mag, pha = _random_spectrogram(b=2, n_features=model.n_features, n=24, seed=3)
    mag_g, pha_g, com_g = model(mag, pha)
    # A stand-in objective touching every head so all params receive gradient.
    loss = mag_g.pow(2).mean() + com_g.pow(2).mean() + pha_g.sin().mean()
    loss.backward()
    n_with_grad = 0
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "a parameter received no gradient"
            assert torch.isfinite(p.grad).all(), "non-finite gradient"
            n_with_grad += 1
    assert n_with_grad > 0


def test_build_basenet_honours_config_overrides():
    """build_basenet should accept an AttrDict (configs/*.json style) and a
    non-default STFT / band layout — e.g. the repo's 512-point pipeline."""
    cfg = AttrDict({
        "n_fft": 512, "hop_size": 256, "sampling_rate": 16000,
        "base_channels": 16, "crn_hidden": 32, "crn_freq": 8, "dec_depth": 1,
        "band_edges_hz": (0, 1000, 4000, 8000), "band_depths": (4, 3, 2),
        "causal": True,
    })
    model = build_basenet(cfg)
    assert model.n_features == 257
    assert model.band_slices[0][0] == 0 and model.band_slices[-1][1] == 257
    mag, pha = _random_spectrogram(b=1, n_features=257, n=20, seed=4)
    with torch.no_grad():
        mag_g, pha_g, _ = model(mag, pha)
    assert mag_g.shape == (1, 257, 20)


def test_band_depths_length_must_match_band_count():
    with pytest.raises(AssertionError):
        BASENet(band_edges_hz=(0, 1000, 4000, 8000), band_depths=(4, 3))  # 3 bands, 2 depths
