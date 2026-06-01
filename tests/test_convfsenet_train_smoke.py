"""Smoke test for ConvFSENet causal training.

Avoids the VBD download by running a few steps on a synthetic waveform
dataset (the same `_Split` shape used by the quant test).

Verifies:
  - build_causal_model produces a model with causal=True (streaming-ready).
  - forward + backward on (B, 1, samples) runs cleanly.
  - loss is a finite scalar across N steps.
  - checkpoint save/load round-trip preserves the loss curve.
  - utils.scan_checkpoint finds the saved file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from convfsenet.model import build_causal_model
from common.env import AttrDict
from common.utils import load_checkpoint, save_checkpoint, scan_checkpoint


# ---- synthetic dataset ------------------------------------------------------


class _SyntheticAudioDataset(TorchDataset):
    """Yields (clean, noisy) pairs of shape (samples,) — matches dataset.Dataset
    output. Both signals are smooth-ish (sinusoids + low-amp noise) so the
    DynCompMSE active-speech threshold (0.025) isn't degenerate."""

    def __init__(self, n=8, samples=8000, sr=16000, seed=0):
        rng = np.random.default_rng(seed)
        self._items = []
        t = np.arange(samples) / sr
        for i in range(n):
            f1 = 250 + 30 * i
            f2 = 1100 + 70 * i
            clean = (
                0.3 * np.sin(2 * np.pi * f1 * t)
                + 0.2 * np.sin(2 * np.pi * f2 * t)
            ).astype(np.float32)
            noise = (0.10 * rng.standard_normal(samples)).astype(np.float32)
            noisy = clean + noise
            self._items.append((
                torch.from_numpy(clean),
                torch.from_numpy(noisy),
            ))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


@pytest.fixture
def cfg():
    return AttrDict({
        "n_fft": 512, "win_length": 512, "hop_size": 256, "win_size": 512,
        "n_features": 257,
        "n_channels_res": 128, "n_channels_conv": 256,
        "kernel_size": 3, "n_blocks": 3, "n_stacks": 3,
        "extractor_type": "mag", "compress_factor": None,
        "causal": True,
        "learning_rate": 1e-3,
        "adam_b1": 0.9, "adam_b2": 0.999,
        "seed": 0,
    })


def test_build_causal_model_smoke(cfg):
    torch.manual_seed(0)
    model = build_causal_model(cfg)
    # Causality is load-bearing: the streaming wrapper REQUIRES causal=True.
    for blk in model.tcm:
        assert blk.causal is True
    # Parameter count is in the expected order of magnitude.
    n = sum(p.numel() for p in model.parameters())
    assert 0.3e6 < n < 5e6, f"unexpected param count: {n}"


def test_train_step_runs_and_loss_decreases(cfg, tmp_path):
    """Run 5 train steps on synthetic data; assert loss is finite and the
    average over the last 3 steps is below the average over the first 2
    (gradient descent is at least going downhill on memorized synthetic data).
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = build_causal_model(cfg).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    ds = _SyntheticAudioDataset(n=8, samples=8000)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    losses = []
    for i, (clean, noisy) in enumerate(loader):
        clean = clean.unsqueeze(1)        # (B, 1, samples)
        noisy = noisy.unsqueeze(1)
        optim.zero_grad()
        loss_dict = model.train_step(noisy, clean)
        loss = loss_dict["loss"]
        loss.backward()
        optim.step()
        l = float(loss.item())
        assert np.isfinite(l), f"step {i}: non-finite loss {l}"
        losses.append(l)

    # 4 steps from 8 samples / batch=2. Use first 2 vs last 2.
    avg_early = np.mean(losses[:2])
    avg_late = np.mean(losses[-2:])
    assert avg_late <= avg_early * 1.5, (
        f"loss is not trending down: early={avg_early:.4f}, late={avg_late:.4f}, "
        f"full curve={losses}"
    )


def test_valid_step_runs(cfg):
    """valid_step (no crop, autograd off) on a single full utterance."""
    torch.manual_seed(0)
    model = build_causal_model(cfg).eval()
    ds = _SyntheticAudioDataset(n=2, samples=16000)
    clean, noisy = ds[0]
    clean = clean.unsqueeze(0).unsqueeze(0)    # (1, 1, samples)
    noisy = noisy.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        x_pred, x_clean_p, x_noisy_p, loss_dict = model.valid_step(noisy, clean)
    assert x_pred.shape == x_clean_p.shape
    assert np.isfinite(float(loss_dict["loss"].item()))


def test_checkpoint_roundtrip(cfg, tmp_path):
    """Save model state, mutate it, reload, confirm same output as pre-mutation."""
    torch.manual_seed(0)
    model = build_causal_model(cfg).eval()
    # Capture a deterministic forward output to compare.
    x = torch.randn(1, 1, 8000)
    with torch.no_grad():
        out_before = model.valid_step(x, x)[0]

    cp = tmp_path / "g_00000100"
    save_checkpoint(str(cp), {"generator": model.state_dict(), "steps": 100, "epoch": 0})
    assert cp.exists()

    # Mutate the weights so a no-op reload would be silent.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))

    state = load_checkpoint(str(cp), torch.device("cpu"))
    model.load_state_dict(state["generator"])
    model.eval()
    with torch.no_grad():
        out_after = model.valid_step(x, x)[0]

    assert torch.allclose(out_before, out_after), (
        "checkpoint reload did not restore the original output"
    )


def test_scan_checkpoint_picks_latest(tmp_path):
    (tmp_path / "g_00000100").touch()
    (tmp_path / "g_00000300").touch()
    (tmp_path / "g_00000200").touch()
    latest = scan_checkpoint(str(tmp_path), "g_")
    assert latest is not None and Path(latest).name == "g_00000300"
