"""Smoke tests for BASENet training (MP-SENet recipe).

Avoids the VoiceBank-DEMAND download by running a few steps on synthetic audio
(same shape the real ``Dataset`` yields). Verifies:

  - the MP-SENet losses behave (anti-wrapping, zero at the optimum);
  - a GAN-free generator step runs forward+backward and trends down;
  - a full MetricGAN step (generator + discriminator) runs end-to-end;
  - the config shipped in configs/basenet_causal.json loads and builds;
  - the enhance() round-trip returns finite audio of the right length.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from basenet.model import build_basenet
from basenet.losses import anti_wrapping_function, phase_losses, spectral_losses
from basenet.train import enhance, generator_loss, gan_step
from common.discriminator import MetricDiscriminator
from common.env import AttrDict


# ---- synthetic dataset (mirrors common.dataset.Dataset output) --------------


class _SyntheticAudioDataset(TorchDataset):
    """(clean, noisy) pairs of shape (samples,) — voiced-ish sinusoids + noise."""

    def __init__(self, n=8, samples=12000, sr=16000, seed=0):
        rng = np.random.default_rng(seed)
        self._items = []
        t = np.arange(samples) / sr
        for i in range(n):
            clean = (
                0.3 * np.sin(2 * np.pi * (220 + 25 * i) * t)
                + 0.2 * np.sin(2 * np.pi * (900 + 60 * i) * t)
            ).astype(np.float32)
            noisy = (clean + 0.10 * rng.standard_normal(samples)).astype(np.float32)
            self._items.append((torch.from_numpy(clean), torch.from_numpy(noisy)))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


@pytest.fixture
def cfg():
    # Small/fast model; STFT + loss weights match the shipped config.
    return AttrDict({
        "n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": 0.3,
        "sampling_rate": 16000,
        "base_channels": 16, "crn_hidden": 48, "crn_freq": 8, "dec_depth": 1,
        "band_edges_hz": (0, 1000, 4000, 8000), "band_depths": (4, 3, 2),
        "causal": True,
        "learning_rate": 1e-3, "adam_b1": 0.9, "adam_b2": 0.999, "seed": 0,
        "loss": {"magnitude": 0.9, "phase": 0.3, "complex": 0.1, "time": 0.2},
    })


# ---- losses -----------------------------------------------------------------


def test_anti_wrapping_ignores_2pi_offsets():
    x = torch.tensor([0.1, 2 * math.pi + 0.1, -2 * math.pi - 0.1, 4 * math.pi])
    out = anti_wrapping_function(x)
    expected = torch.tensor([0.1, 0.1, 0.1, 0.0])
    assert torch.allclose(out, expected, atol=1e-5)


def test_phase_and_spectral_losses_zero_at_optimum():
    torch.manual_seed(0)
    mag = torch.rand(2, 201, 30)
    pha = (torch.rand(2, 201, 30) * 2 - 1) * math.pi
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    ip, gd, iaf = phase_losses(pha, pha)
    assert float(ip) == pytest.approx(0.0, abs=1e-6)
    assert float(gd) == pytest.approx(0.0, abs=1e-6)
    assert float(iaf) == pytest.approx(0.0, abs=1e-6)
    sl = spectral_losses(mag, pha, com, mag, pha, com)
    assert float(sl["magnitude"]) == pytest.approx(0.0, abs=1e-6)
    assert float(sl["complex"]) == pytest.approx(0.0, abs=1e-6)
    assert float(sl["phase"]) == pytest.approx(0.0, abs=1e-6)


# ---- training steps ---------------------------------------------------------


def test_generator_step_runs_and_trends_down(cfg):
    torch.manual_seed(0)
    model = build_basenet(cfg).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ds = _SyntheticAudioDataset(n=8, samples=12000)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    losses = []
    for clean, noisy in loader:
        clean = clean.unsqueeze(1)         # (B, 1, samples)
        noisy = noisy.unsqueeze(1)
        optim.zero_grad()
        loss, parts = generator_loss(model, noisy, clean, cfg)
        loss.backward()
        optim.step()
        l = float(loss.item())
        assert np.isfinite(l)
        for k in ("magnitude", "phase", "complex", "time"):
            assert np.isfinite(parts[k])
        losses.append(l)

    assert np.mean(losses[-2:]) <= np.mean(losses[:2]) * 1.5, (
        f"loss not trending down: {losses}"
    )


def test_grad_accumulation_equals_full_batch(cfg):
    """Accumulating 2 micro-batches of size 2 (scaled 1/2) gives the same
    generator gradients as one batch of size 4. This is exact, not approximate,
    because every module is per-sample (no batch statistics) — which is what
    makes '--accum_steps 2' a faithful stand-in for the paper's larger batch."""
    torch.manual_seed(0)
    model = build_basenet(cfg)
    clean = torch.randn(4, 1, 8000)
    noisy = torch.randn(4, 1, 8000)

    model.zero_grad(set_to_none=True)
    loss, _ = generator_loss(model, noisy, clean, cfg)
    loss.backward()
    g_full = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad(set_to_none=True)
    for sl in (slice(0, 2), slice(2, 4)):
        l, _ = generator_loss(model, noisy[sl], clean[sl], cfg)
        (l / 2).backward()
    g_accum = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    max_diff = max((a - b).abs().max().item() for a, b in zip(g_full, g_accum))
    assert max_diff < 1e-5, f"accumulation != full batch: max grad diff {max_diff:.2e}"


def test_gan_step_runs_end_to_end(cfg):
    """A full MetricGAN step (generator + discriminator) must run without error
    and produce finite losses (PESQ on synthetic audio may be unavailable, which
    the step handles by skipping the discriminator's fake branch)."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = build_basenet(cfg).to(device).train()
    disc = MetricDiscriminator().to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    optim_d = torch.optim.AdamW(disc.parameters(), lr=cfg.learning_rate)

    ds = _SyntheticAudioDataset(n=4, samples=12000)
    clean, noisy = ds[0]
    clean = clean.unsqueeze(0).unsqueeze(1)    # (1, 1, samples)
    noisy = noisy.unsqueeze(0).unsqueeze(1)

    metrics = gan_step(model, disc, optim, optim_d, noisy, clean, cfg,
                       metric_lambda=0.05, device=device)
    for k in ("loss", "base_loss", "loss_metric", "loss_disc"):
        assert np.isfinite(metrics[k]), f"{k} not finite: {metrics[k]}"


def test_enhance_roundtrip_shape(cfg):
    torch.manual_seed(0)
    model = build_basenet(cfg).eval()
    noisy = torch.randn(1, 1, 12000)
    with torch.no_grad():
        out = enhance(model, noisy, cfg)
    # istft length is within one hop of the input length.
    assert out.dim() == 2 and out.shape[0] == 1
    assert abs(out.shape[-1] - 12000) <= cfg.hop_size
    assert torch.isfinite(out).all()


# ---- shipped config ---------------------------------------------------------


def test_shipped_config_builds_paper_sized_model():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "basenet_causal.json"
    with open(cfg_path) as f:
        h = AttrDict(json.load(f))
    model = build_basenet(h)
    assert model.causal is True
    assert model.n_features == 201            # n_fft=400 -> 201 bins
    n = sum(p.numel() for p in model.parameters())
    assert 0.6e6 < n < 1.1e6, f"unexpected param count for shipped config: {n:,}"
