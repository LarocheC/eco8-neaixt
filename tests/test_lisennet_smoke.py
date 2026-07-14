"""Smoke tests for the ported LiSenNet (arXiv:2409.13285) + its CMGAN trainer.

Runs on synthetic audio (no VBD download): the model forwards, the CMGAN
generator/discriminator losses are finite, a full GAN step runs forward+backward,
and the loss trends down over a few steps. The shipped config builds the expected
~37k-param model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from lisennet.model import LiSenNet, build_lisennet
from lisennet.train import _LOGGED_TERMS, forward_spectra, gan_step
from common.discriminator import MetricDiscriminator
from common.env import AttrDict
from common.losses import generator_loss, generator_terms, loss_weights


class _SyntheticAudioDataset(TorchDataset):
    def __init__(self, n=8, samples=32000, sr=16000, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(samples) / sr
        self._items = []
        for i in range(n):
            clean = (0.3 * np.sin(2 * np.pi * (200 + 30 * i) * t)
                     + 0.2 * np.sin(2 * np.pi * (900 + 50 * i) * t)).astype(np.float32)
            noisy = (clean + 0.08 * rng.standard_normal(samples)).astype(np.float32)
            self._items.append((torch.from_numpy(clean), torch.from_numpy(noisy)))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


@pytest.fixture
def cfg():
    return AttrDict({
        "num_channels": 16, "n_blocks": 2, "n_fft": 512, "hop_size": 256,
        "compress_factor": 0.3, "gradient_clip": 5.0, "learning_rate": 5e-4,
        "adam_b1": 0.8, "adam_b2": 0.99, "seed": 0,
        "loss": {"magnitude": 0.9, "complex": 0.1, "consistency": 0.1,
                 "time": 0.2, "metric": 0.05},
    })


def _weights(cfg):
    w = loss_weights(cfg)
    w.pop("phase", None)                # no phase decoder
    return w


def test_model_params_and_forward():
    torch.manual_seed(0)
    model = build_lisennet().eval()
    n = sum(p.numel() for p in model.parameters())
    assert 20_000 < n < 80_000, f"unexpected param count: {n:,}"   # LiSenNet is ~37k
    x = torch.randn(2, 32000)
    with torch.no_grad():
        r = model(x, x)
    assert r["est"].shape == (2, 32000)
    assert torch.isfinite(r["est"]).all()
    assert (r["est_mag"] >= 0).all()


def test_losses_finite(cfg):
    torch.manual_seed(0)
    model = LiSenNet(num_channels=16, n_blocks=2).eval()
    disc = MetricDiscriminator(dim=16).eval()
    x = torch.randn(2, 32000)

    s = forward_spectra(model, x, x)
    metric_g = disc(s["mag_r"], s["mag_g_hat"])
    terms = generator_terms(
        mag_g=s["mag_g"], com_g=s["com_g"], mag_r=s["mag_r"], com_r=s["com_r"],
        com_g_hat=s["com_g_hat"], audio_g=s["audio_g"], audio_r=s["audio_r"],
        metric_g=metric_g,
    )
    g_loss = generator_loss(terms, _weights(cfg))
    assert torch.isfinite(g_loss)
    for k in _LOGGED_TERMS:
        assert np.isfinite(float(terms[k])), k
    # Griffin-Lim phase is derived from a *detached* magnitude, so no phase term.
    assert "phase" not in terms


def test_consistency_term_is_live(cfg):
    """LiSenNet's spectrum is (est_mag, Griffin-Lim phase) — not the STFT of any
    real waveform — so the round trip must actually move it. If this is zero the
    consistency term is doing nothing and the port is wrong."""
    torch.manual_seed(0)
    model = LiSenNet(num_channels=16, n_blocks=2).eval()
    s = forward_spectra(model, torch.randn(2, 32000), torch.randn(2, 32000))
    terms = generator_terms(
        mag_g=s["mag_g"], com_g=s["com_g"], mag_r=s["mag_r"], com_r=s["com_r"],
        com_g_hat=s["com_g_hat"], audio_g=s["audio_g"], audio_r=s["audio_r"],
    )
    assert float(terms["consistency"]) > 1e-6, "consistency term is inert"


def test_time_and_consistency_reach_the_weights(cfg):
    """Both terms must produce gradient at the mask sub-network — they are what
    LiSenNet gained from MP-SENet, and a detach in the wrong place would silence
    them without failing anything else."""
    torch.manual_seed(0)
    model = LiSenNet(num_channels=16, n_blocks=2)
    s = forward_spectra(model, torch.randn(2, 32000), torch.randn(2, 32000))
    terms = generator_terms(
        mag_g=s["mag_g"], com_g=s["com_g"], mag_r=s["mag_r"], com_r=s["com_r"],
        com_g_hat=s["com_g_hat"], audio_g=s["audio_g"], audio_r=s["audio_r"],
    )
    for name in ("time", "consistency"):
        model.zero_grad(set_to_none=True)
        terms[name].backward(retain_graph=True)
        total = sum(p.grad.abs().sum().item()
                    for p in model.parameters() if p.grad is not None)
        assert total > 0, f"{name} term produces no gradient"


def test_gan_step_trends_down(cfg):
    torch.manual_seed(0)
    model = build_lisennet(cfg)
    disc = MetricDiscriminator(dim=16)
    optim_g = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    optim_d = torch.optim.AdamW(disc.parameters(), lr=cfg.learning_rate)
    loader = DataLoader(_SyntheticAudioDataset(n=8, samples=32000), batch_size=2, shuffle=False)

    losses = []
    for clean, noisy in loader:
        m = gan_step(model, disc, optim_g, optim_d, noisy, clean, cfg, _weights(cfg), 5.0)
        assert np.isfinite(m["loss"]) and np.isfinite(m["d_loss"])
        losses.append(m["loss"])
    assert np.mean(losses[-2:]) <= np.mean(losses[:2]) * 1.5, f"loss not trending down: {losses}"


def test_shipped_config_builds():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "lisennet.json"
    with open(cfg_path) as f:
        h = AttrDict(json.load(f))
    model = build_lisennet(h)
    n = sum(p.numel() for p in model.parameters())
    assert 20_000 < n < 80_000
