"""Per-frequency-bin mean of NSNet2's network input, for ``input_norm``.

Runs the exact training feature pipeline of ``nsnet2.train``:
``common.dataset.Dataset(hf['train'], segment_size, sampling_rate,
split=True, shuffle=True, seed=h.seed)`` (random ``segment_size`` crops of
RMS-normalised utterances, short ones zero-padded) -> ``mag_pha_stft(noisy,
n_fft, hop_size, win_size, compress_factor)`` -> the (B, F, T) magnitude the
generator receives. Accumulates the per-bin mean over every frame in float64
and writes ``{"mean": [F], "n_frames", "config", "provenance"}``.

Convergence is reported as the difference between the means of the first and
second half of the segments (two independent estimates of the same mean).

    python -m nsnet2.input_stats --config configs/sq192_dense.json \\
        --output configs/stats/nsnet2_nfft512_hop256_cf0.3.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.dataset import (HF_DATASET_NAME, Dataset, data_generator,
                            load_voicebank_demand, mag_pha_stft, seed_worker)
from common.env import AttrDict

# The config keys the network input depends on (everything else is model/optim).
FEATURE_KEYS = ("sampling_rate", "segment_size", "n_fft", "hop_size",
                "win_size", "compress_factor", "seed")


def _git_rev():
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


@torch.no_grad()
def compute_input_mean(h, *, epochs=1, batch_size=256, num_workers=5,
                       hf_cache_dir=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    hf = load_voicebank_demand(cache_dir=hf_cache_dir)
    trainset = Dataset(hf["train"], h.segment_size, h.sampling_rate,
                       split=True, shuffle=True, seed=h.seed)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False,
                        worker_init_fn=seed_worker,
                        generator=data_generator(h.seed))
    n_freq = h.n_fft // 2 + 1
    total = len(trainset) * epochs
    # halves[0] accumulates the first half of the segments, halves[1] the rest.
    sums = torch.zeros(2, n_freq, dtype=torch.float64, device=device)
    frames = torch.zeros(2, dtype=torch.float64, device=device)
    seen = 0
    for _ in range(epochs):
        for _clean, noisy in loader:
            noisy = noisy.to(device, non_blocking=True)
            mag, _, _ = mag_pha_stft(noisy, h.n_fft, h.hop_size, h.win_size,
                                     h.compress_factor)          # (B, F, T)
            mag = mag.double()
            idx = torch.arange(seen, seen + mag.shape[0], device=device)
            half = (idx >= total // 2).long()
            for k in (0, 1):
                sel = mag[half == k]
                if sel.numel():
                    sums[k] += sel.sum(dim=(0, 2))
                    frames[k] += sel.shape[0] * sel.shape[2]
            seen += mag.shape[0]
    mean = sums.sum(0) / frames.sum()
    m0, m1 = sums[0] / frames[0], sums[1] / frames[1]
    diff = (m0 - m1).abs()
    conv = {
        "segments": seen,
        "frames_per_half": [int(frames[0].item()), int(frames[1].item())],
        "half_max_abs_diff": diff.max().item(),
        "half_max_rel_diff": (diff / mean.abs()).max().item(),
        "half_mean_abs_diff": diff.mean().item(),
        "mean_range": [mean.min().item(), mean.max().item()],
    }
    return mean.cpu(), int(frames.sum().item()), conv


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1,
                        help="passes over the training split (random crops differ per pass)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--hf_cache_dir", default=None)
    a = parser.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    mean, n_frames, conv = compute_input_mean(
        h, epochs=a.epochs, batch_size=a.batch_size,
        num_workers=a.num_workers, hf_cache_dir=a.hf_cache_dir)

    out = {
        "mean": [float(v) for v in mean.tolist()],
        "n_frames": n_frames,
        "config": {k: h.get(k) for k in FEATURE_KEYS},
        "provenance": {
            "script": "nsnet2/input_stats.py",
            "source_config": a.config,
            "dataset": HF_DATASET_NAME,
            "split": "train",
            "pipeline": ("Dataset(split=True, shuffle=True, seed) -> noisy -> "
                         "mag_pha_stft(n_fft, hop_size, win_size, compress_factor) "
                         "magnitude, per-bin mean over all frames"),
            "epochs": a.epochs,
            "git_rev": _git_rev(),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "convergence": conv,
        },
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(conv, indent=1))
    print(f"wrote {a.output}: {len(out['mean'])} bins, {n_frames} frames")


if __name__ == "__main__":
    main()
