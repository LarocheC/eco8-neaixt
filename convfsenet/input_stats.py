"""Per-frequency-bin mean of ConvFSENet's input features over the TRAINING set.

The mean feeds the block-design `input_norm` option ({"kind": "mean", "path":
<this file's output>}): the model subtracts it from its features right before
the frontend conv. Mean only -- no division by a std.

"The features" are exactly what the frontend sees during training: the train
split is built the way convfsenet/train.py builds it (same Dataset -- RMS
normalisation, random segment_size crops, zero-padding of short utterances --
same DataLoader seeding, batch size and drop_last), each batch goes through
the model's own preproc (torch.stft) and features_extractor (e.g. power-
compressed magnitude). Every (utterance, frame) pair of an epoch counts once.
No model weights are involved; only the feature config matters.

Convergence check: the per-bin mean of the first half of the batches, of the
second half (split at the middle of all --n_epochs passes, default 4; each pass
draws fresh random crops, as training epochs do) are compared, as is the
running mean at 50% vs the final mean, in units of the per-bin feature std.
Both must stay under --tol (default 0.01 std) or the script exits non-zero
without writing.

Usage:
    python -m convfsenet.input_stats --config configs/cfs_c96_dense.json \\
        --output configs/stats/convfsenet_vbd_train_magc0.3_nfft512.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

from common.dataset import HF_DATASET_NAME, Dataset, data_generator, load_voicebank_demand, seed_worker
from common.env import AttrDict
from convfsenet.model import build_causal_model

FEATURE_KEYS = ("n_fft", "win_length", "hop_size", "n_features", "extractor_type", "compress_factor")


def feature_signature(h) -> dict:
    """The config values that determine the features (what load_input_mean checks)."""
    n_fft = int(h.get("n_fft", 512))
    return {
        "n_fft": n_fft,
        "win_length": int(h.get("win_length", n_fft)),
        "hop_size": int(h.get("hop_size", n_fft // 2)),
        "n_features": int(h.get("n_features", n_fft // 2 + 1)),
        "extractor_type": h.get("extractor_type", "mag"),
        "compress_factor": h.get("compress_factor", None),
    }


def build_train_loader(h, hf_cache_dir=None, num_workers=None):
    """The training DataLoader, constructed exactly as convfsenet/train.py does."""
    hf = load_voicebank_demand(cache_dir=hf_cache_dir)
    trainset = Dataset(
        hf["train"], h.segment_size, h.sampling_rate,
        split=True, shuffle=True, seed=h.seed,
    )
    return DataLoader(
        trainset, num_workers=h.num_workers if num_workers is None else num_workers,
        shuffle=False, batch_size=h.batch_size, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=data_generator(h.seed),
    )


@torch.no_grad()
def accumulate(model, loader, device, n_epochs=1, log_every=100):
    """Per-bin float64 sums over every training frame of n_epochs passes.

    Also keeps the sums of the first and second half of all batches (split at
    the middle of the whole run) for the convergence check.
    """
    F = model.n_features
    s1 = torch.zeros(F, dtype=torch.float64, device=device)
    s2 = torch.zeros(F, dtype=torch.float64, device=device)
    halves = [[torch.zeros(F, dtype=torch.float64, device=device), 0] for _ in range(2)]
    n = 0
    n_batches = len(loader)
    total = n_batches * n_epochs
    k = 0
    for ep in range(n_epochs):
        for i, (clean, noisy) in enumerate(loader):
            noisy = noisy.to(device, non_blocking=True).unsqueeze(1)          # (B, 1, S) as in train.py
            stft = model.preproc(noisy).squeeze(1)                             # (B, F, T) complex
            feats = model.features_extractor(stft).double()                    # (B, F, T)
            bsum = feats.sum(dim=(0, 2))
            cnt = feats.shape[0] * feats.shape[2]
            s1 += bsum
            s2 += feats.pow(2).sum(dim=(0, 2))
            n += cnt
            half = halves[0 if k < total // 2 else 1]
            half[0] += bsum; half[1] += cnt
            k += 1
            if log_every and (i + 1) % log_every == 0:
                print(f"  epoch {ep + 1} batch {i + 1}/{n_batches}  frames {n}", flush=True)
    return {"sum": s1, "sumsq": s2, "count": n, "halves": halves, "n_batches": total}


def summarize(acc):
    n = acc["count"]
    mean = acc["sum"] / n
    std = (acc["sumsq"] / n - mean.pow(2)).clamp_min(0).sqrt()
    denom = std.clamp_min(1e-12)
    first = acc["halves"][0][0] / acc["halves"][0][1]           # also the running mean at 50%
    second = acc["halves"][1][0] / acc["halves"][1][1]
    return mean, std, {
        # the full mean's error is ~ half this disagreement
        "max_abs_first_vs_second_half_over_std": float(((first - second).abs() / denom).max()),
        "max_abs_running50_vs_full_over_std": float(((first - mean).abs() / denom).max()),
        "max_abs_first_vs_second_half": float((first - second).abs().max()),
    }


def _git_commit():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL) != 0
        return rev + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001 -- provenance only
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/cfs_c96_dense.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--n_epochs", type=int, default=4,
                    help="passes over the train loader (each pass draws fresh random crops)")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max allowed half-vs-half mean disagreement, in per-bin std units")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    # Stats describe the RAW features: build without any input_norm of the config.
    h_raw = AttrDict({k: v for k, v in h.items() if k not in ("input_norm", "frontend_norm")})
    device = torch.device(a.device)
    torch.manual_seed(h.seed)
    model = build_causal_model(h_raw).to(device).eval()
    loader = build_train_loader(h, a.hf_cache_dir, a.num_workers)
    print(f"train split: {len(loader.dataset)} utterances, {len(loader)} batches/epoch "
          f"(batch {h.batch_size}, segment {h.segment_size}), {a.n_epochs} epoch(s)")

    acc = accumulate(model, loader, device, a.n_epochs)
    mean, std, conv = summarize(acc)
    conv["tol"] = a.tol
    conv["pass"] = (conv["max_abs_first_vs_second_half_over_std"] <= a.tol
                    and conv["max_abs_running50_vs_full_over_std"] <= a.tol)
    print(f"frames: {acc['count']}  mean range [{mean.min():.4f}, {mean.max():.4f}]  "
          f"std range [{std.min():.4f}, {std.max():.4f}]")
    print("convergence: " + json.dumps(conv))
    if not conv["pass"]:
        print("NOT CONVERGED -- not writing; raise --n_epochs", file=sys.stderr)
        return 1

    out = {
        "kind": "mean",
        "description": "per-frequency-bin mean of ConvFSENet frontend input features "
                       "(model.features_extractor(model.preproc(noisy))) over the training split",
        "features": feature_signature(h),
        "mean": [float(x) for x in mean.cpu()],
        "std": [float(x) for x in std.cpu()],
        "convergence": conv,
        "provenance": {
            "script": "convfsenet/input_stats.py",
            "config": a.config,
            "dataset": HF_DATASET_NAME,
            "split": "train",
            "pipeline": "common.dataset.Dataset(split=True, shuffle=True, seed=h.seed) + "
                        "DataLoader(batch_size=h.batch_size, drop_last=True, seed_worker, "
                        "data_generator(h.seed)) as in convfsenet/train.py",
            "segment_size": int(h.segment_size),
            "sampling_rate": int(h.sampling_rate),
            "batch_size": int(h.batch_size),
            "seed": int(h.seed),
            "n_epochs": a.n_epochs,
            "n_batches": acc["n_batches"],
            "n_frames": int(acc["count"]),
            "n_utterances": len(loader.dataset),
            "git_commit": _git_commit(),
            "torch": torch.__version__,
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
