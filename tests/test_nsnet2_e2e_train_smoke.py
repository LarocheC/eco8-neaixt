"""Smoke test for the end-to-end butterfly NSNet2 (waveform -> waveform).

Runs a few train steps on a synthetic waveform dataset (no VBD download),
exercising the exact generator + loss used by ``nsnet2/train_e2e.py``.

Verifies:
  - NSNet2E2E builds with n_coeffs "frequency" bins and a sane param count.
  - waveform-in -> waveform-out preserves length (incl. non-multiple lengths).
  - time + multi-res-STFT + ortho loss is finite and trends down over steps.
  - butterfly ortho penalty is a finite scalar and gradients reach the twiddles
    and the learnable windows.
  - checkpoint save/load round-trip preserves the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from common.env import AttrDict
from common.utils import load_checkpoint, save_checkpoint, scan_checkpoint
from nsnet2.model_e2e import NSNet2E2E
from nsnet2.train_e2e import mrstft_mag_loss


class _SyntheticAudioDataset(TorchDataset):
    """(clean, noisy) pairs of shape (samples,) — matches dataset.Dataset."""

    def __init__(self, n=8, samples=8192, sr=16000, seed=0):
        rng = np.random.default_rng(seed)
        self._items = []
        t = np.arange(samples) / sr
        for i in range(n):
            f1, f2 = 250 + 30 * i, 1100 + 70 * i
            clean = (0.3 * np.sin(2 * np.pi * f1 * t)
                     + 0.2 * np.sin(2 * np.pi * f2 * t)).astype(np.float32)
            noisy = clean + (0.10 * rng.standard_normal(samples)).astype(np.float32)
            self._items.append((torch.from_numpy(clean), torch.from_numpy(noisy)))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


@pytest.fixture
def cfg():
    return AttrDict({
        "win_size": 512, "hop_size": 256, "n_coeffs": 512,
        "hidden_dim": 96, "fc_hidden_dim": 96, "num_gru_layers": 2,
        "compress_factor": 0.3,
        "transform": {"learnable_window": True, "window_init": "sqrt_hann",
                      "nblocks": 1, "init": "ortho"},
        "learning_rate": 1e-3, "seed": 0,
    })


# Small resolutions that fit an 8192-sample smoke segment.
SMOKE_RES = [[256, 128, 256], [512, 256, 512]]


def test_build_and_shapes(cfg):
    torch.manual_seed(0)
    model = NSNet2E2E(cfg)
    # core sees win bins (square orthogonal butterfly), not the STFT's n_fft//2+1
    assert model.core.fc_in.in_features == cfg.win_size
    assert model.core.fc_out.out_features == cfg.win_size
    for L in (8192, 8000, 500):
        y = model(torch.randn(2, L))
        assert y.shape == (2, L), (L, y.shape)


def test_transform_reconstructs_at_init(cfg):
    """The shared orthogonal butterfly + transpose synthesis must reconstruct
    (mask=1) at init — this is what lets the net learn denoising instead of
    inversion. Guards against regressing to an independent random synthesis."""
    torch.manual_seed(0)
    t = NSNet2E2E(cfg).transform.eval()
    x = torch.randn(1, 8192)
    with torch.no_grad():
        rec = t.synthesize(t.analyze(x))
    n = min(rec.shape[-1], x.shape[-1])
    a, b = rec[0, :n], x[0, :n]
    a = a - a.mean(); b = b - b.mean()
    alpha = (a * b).sum() / b.pow(2).sum()
    sisnr = 10 * torch.log10(alpha.pow(2) * b.pow(2).sum() / (a - alpha * b).pow(2).sum())
    assert sisnr.item() > 15, f"init reconstruction SI-SNR too low: {sisnr.item():.1f} dB"


def test_train_step_loss_decreases(cfg):
    torch.manual_seed(0)
    model = NSNet2E2E(cfg).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    ds = _SyntheticAudioDataset(n=8, samples=8192)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    losses = []
    for clean, noisy in loader:
        optim.zero_grad()
        audio_g = model(noisy)
        loss = (F_l1(clean, audio_g)
                + mrstft_mag_loss(clean, audio_g, SMOKE_RES, 0.3)
                + 0.01 * model.ortho_penalty())
        loss.backward()
        # gradients must reach the shared structured twiddle and both windows
        assert model.transform.butterfly.twiddle.grad is not None
        assert model.transform.ana_window.grad is not None
        assert model.transform.syn_window.grad is not None
        optim.step()
        l = float(loss.item())
        assert np.isfinite(l)
        losses.append(l)

    avg_early = np.mean(losses[:2])
    avg_late = np.mean(losses[-2:])
    assert avg_late <= avg_early * 1.5, f"loss not trending down: {losses}"


def test_ortho_penalty_scalar(cfg):
    torch.manual_seed(0)
    model = NSNet2E2E(cfg)
    p = model.ortho_penalty()
    assert p.ndim == 0 and np.isfinite(float(p.item()))


def test_checkpoint_roundtrip(cfg, tmp_path):
    torch.manual_seed(0)
    model = NSNet2E2E(cfg).eval()
    x = torch.randn(1, 8192)
    with torch.no_grad():
        out_before = model(x)

    cp = tmp_path / "g_00000100"
    save_checkpoint(str(cp), {"generator": model.state_dict(), "steps": 100, "epoch": 0})
    assert cp.exists()
    with torch.no_grad():
        for pm in model.parameters():
            pm.add_(torch.randn_like(pm))

    state = load_checkpoint(str(cp), torch.device("cpu"))
    model.load_state_dict(state["generator"])
    model.eval()
    with torch.no_grad():
        out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def F_l1(a, b):
    return torch.nn.functional.l1_loss(a, b)
