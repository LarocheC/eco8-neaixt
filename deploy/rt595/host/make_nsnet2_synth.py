#!/usr/bin/env python
"""Regenerate cp_nsnet2_plain_synth/ — a canonical plain-GRU NSNet2 with RANDOM weights.

WHY THIS EXISTS: none of the saved NSNet2 checkpoints load on this branch (cp_blockdiag_* use
the pre-rename "blockdiag" linear/gru kind; cp_monarch_*_triton need a torch-structured build;
cp_nsnet2_butterfly_e2e's config has no n_fft). Cycle counts for dense int8 ops are
weight-independent, so a seeded random-init model of the canonical architecture is a valid
subject for a LATENCY measurement.

    IT IS NOT VALID FOR ANY QUALITY MEASUREMENT. Anything derived from this checkpoint must
    carry a "synthetic-weights" taint and must never populate a PESQ/DNSMOS column.

Replace it with nsnet2:baseline from HF (real weights) as soon as that path works.

    ~/.venvs/rt595-export/bin/python deploy/rt595/host/make_nsnet2_synth.py
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Canonical NSNet2 (Braun & Tashev, ICASSP 2021) as this repo parameterizes it.
CONFIG = dict(
    n_fft=512, hidden_dim=400, fc_hidden_dim=600, num_gru_layers=2,
    compress_factor=0.3, sampling_rate=16000, hop_size=256, win_size=512,
    linear={"kind": "linear"}, gru={"kind": "gru"},
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "cp_nsnet2_plain_synth"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", default=str(REPO_ROOT))
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    import torch
    from common.env import AttrDict
    from nsnet2.model import NSNet2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    torch.manual_seed(a.seed)
    model = NSNet2(AttrDict(CONFIG)).eval()
    torch.save({"generator": model.state_dict()}, out / "g_best")

    n = sum(p.numel() for p in model.parameters())
    print(f"Wrote {out}  (seed={a.seed}, {n:,} params)")
    print("  SYNTHETIC WEIGHTS — latency only, never quality.")


if __name__ == "__main__":
    main()
