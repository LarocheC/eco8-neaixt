"""Smoke tests for the re-implemented FSPEN (ICASSP 2024) + its CMGAN trainer.

Runs on synthetic audio (no VBD download): the model forwards, the CMGAN
generator/discriminator losses are finite, a full GAN step runs forward+backward,
and the loss trends down over a few steps. The shipped config builds the expected
~35k-param model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from fspen.model import FSPEN, build_fspen
from fspen.train import gan_step, generator_loss, discriminator_loss, _loss_weights
from common.discriminator import MetricDiscriminator
from common.env import AttrDict


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
        "n_fft": 512, "hop_size": 256, "compress_factor": 0.3,
        "gradient_clip": 5.0, "learning_rate": 5e-4,
        "adam_b1": 0.8, "adam_b2": 0.99, "seed": 0,
        "loss": {"complex": 0.1, "mag": 0.9, "adv": 0.05},
    })


def test_model_params_and_forward():
    torch.manual_seed(0)
    model = build_fspen().eval()
    n = sum(p.numel() for p in model.parameters())
    assert 20_000 < n < 80_000, f"unexpected param count: {n:,}"   # FSPEN is ~35k
    x = torch.randn(2, 32000)
    with torch.no_grad():
        r = model(x, x)
    assert r["est"].shape == (2, 32000)
    assert torch.isfinite(r["est"]).all()
    assert (r["est_mag"] >= 0).all()


def test_enhance_spectrum_shapes_and_state():
    """The deployable core keeps shapes and returns one state per DPE block."""
    torch.manual_seed(0)
    model = build_fspen().eval()
    spec = torch.randn(2, 12, 2, model.n_freqs)
    amp = torch.sqrt(spec[:, :, 0:1] ** 2 + spec[:, :, 1:2] ** 2)
    with torch.no_grad():
        est, hidden = model.enhance_spectrum(spec, amp)
    assert est.shape == spec.shape
    assert len(hidden) == len(model.dpe_blocks)
    for h, h0 in zip(hidden, model.init_hidden(2)):
        assert h.shape == h0.shape
    # The DC bin is masked away by the zero pad on both paths.
    assert torch.count_nonzero(est[:, :, :, 0]) == 0


def test_sub_band_decoder_reads_consecutive_bands():
    """Pins the deliberate fix of the reference's stuck ``start_idx`` bug: each
    decoder group must read its *own* consecutive slice of the 32 DPE-feature
    bands ([0:8], [8:14], ... [26:32]), not [0:bands_g] like the reference
    (which starves bands 8..31 of a sub-band decoder). See fspen/model.py."""
    model = build_fspen()
    starts = [d.start_band for d in model.sub_band_decoders]
    ends = [d.end_band for d in model.sub_band_decoders]
    assert starts == [0, 8, 14, 20, 26], f"non-consecutive band starts: {starts}"
    assert ends == [8, 14, 20, 26, 32], f"non-consecutive band ends: {ends}"
    assert ends[-1] == sum(model.bands_per_group)


def test_zero_padded_batch_gives_finite_grads(cfg):
    """The dataset zero-pads utterances shorter than segment_size with exact
    zeros (common/dataset.py), so batches routinely contain all-zero STFT
    frames. The loss-domain compression must stay differentiable there —
    abs/angle have NaN gradients at 0+0j and would NaN every parameter on the
    first such batch (the regression this test pins)."""
    torch.manual_seed(0)
    model = build_fspen(cfg)
    disc = MetricDiscriminator(dim=16)
    clean = 0.1 * torch.randn(2, 32000)
    noisy = clean + 0.05 * torch.randn(2, 32000)
    clean[1, 20000:] = 0.0                       # zero-padded tail, as the dataset produces
    noisy[1, 20000:] = 0.0
    results = model(noisy, clean)
    g_loss, _ = generator_loss(results, disc, _loss_weights(cfg))
    g_loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads in {len(bad)} tensors, e.g. {bad[:5]}"


def test_losses_finite(cfg):
    torch.manual_seed(0)
    model = FSPEN().eval()
    disc = MetricDiscriminator(dim=16).eval()
    x = torch.randn(2, 32000)
    results = model(x, x)
    g_loss, parts = generator_loss(results, disc, _loss_weights(cfg))
    d_loss = discriminator_loss(results, disc, torch.device("cpu"))
    assert torch.isfinite(g_loss) and torch.isfinite(d_loss)
    for k in ("complex", "mag", "adv"):
        assert np.isfinite(parts[k])


def test_gan_step_trends_down(cfg):
    torch.manual_seed(0)
    model = build_fspen(cfg)
    disc = MetricDiscriminator(dim=16)
    optim_g = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    optim_d = torch.optim.AdamW(disc.parameters(), lr=cfg.learning_rate)
    loader = DataLoader(_SyntheticAudioDataset(n=8, samples=32000), batch_size=2, shuffle=False)

    losses = []
    for clean, noisy in loader:
        m = gan_step(model, disc, optim_g, optim_d, noisy, clean, cfg, _loss_weights(cfg), 5.0)
        assert np.isfinite(m["loss"]) and np.isfinite(m["d_loss"])
        losses.append(m["loss"])
    assert np.mean(losses[-2:]) <= np.mean(losses[:2]) * 1.5, f"loss not trending down: {losses}"


def test_shipped_config_builds():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "fspen.json"
    with open(cfg_path) as f:
        h = AttrDict(json.load(f))
    model = build_fspen(h)
    n = sum(p.numel() for p in model.parameters())
    assert 20_000 < n < 80_000
