"""Rebuild the sparsity-study results table from training logs and checkpoints.

The logs (``cp_<run>.log``) and checkpoints (``cp_<run>/``) are gitignored, so the
numbers the Row-Fusion study rests on would otherwise live only in one working
tree. This script is the record of how each number is computed; its output,
``SPARSE_MATMUL_RUNS.csv``, is the committed copy of the numbers themselves.

Per run: study, model, width, mask, init, epochs, seed, learning rate, total and
nonzero parameters, the number of validations, best and last-5 validation PESQ,
and the full PESQ trajectory (``;``-joined, in validation order) so any paired
comparison can be recomputed from the CSV alone.

Definitions used throughout SPARSE_MATMUL_COLLAB.md:

* PESQ is the training loop's validation PESQ on the full 824-utterance
  VoiceBank-DEMAND test split. ``g_best`` is selected on that same split, so
  ``best`` is optimistic; comparisons use ``last5``, the mean of the final five
  validations.
* A paired difference between two runs of one wave (same seed, data order and
  validation set) is the mean of their per-validation differences over the last
  five validations; ``leads`` counts validations where the first run is ahead.
* Nonzero parameters count every floating tensor in ``g_best`` except BatchNorm
  running statistics, so a dense model's count equals its parameter count.

    python -m nsnet2.sparsity_results --csv SPARSE_MATMUL_RUNS.csv
    python -m nsnet2.sparsity_results --markdown      # tables + paired comparisons
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import statistics as st
import sys

import torch

from common.utils import load_checkpoint

# (study, model, run). Order = reading order of SPARSE_MATMUL_COLLAB.md.
RUNS = [
    # NSNet2 257/400/600, 120-epoch masked fine-tunes from the 2.845 baseline.
    *[("nsnet2-full-size", "nsnet2", r) for r in (
        "ov_dense_control", "ov_2to4", "ov_4to8", "ov_1to4", "ov_unstruct_80",
        "ov_block1x4_80", "cb_c1", "cb_c2", "cb_c3")],
    # NSNet2 square / multiple-of-32 trunk and its compute-matched dense control.
    *[("nsnet2-square32", "nsnet2", r) for r in (
        "sq384_dense", "sq384_cb_c1", "sq384_cb_c1_s2345", "sq384_ft", "sq384_ft_s2345",
        "dense_h256f384", "dense_h256f384_ft", "dense_h256f384_ft_s2345")],
    # NSNet2 below the knee: dense 128 vs 2:4@c1 pruned from 192.
    *[("nsnet2-knee", "nsnet2", r) for r in (
        "sq128_dense", "sq192_dense", "sq192_cb_c1", "sq192_cb_c1_s2345",
        "sq192_ft", "sq192_ft_s2345", "sq128_ft", "sq128_ft_s2345")],
    # NSNet2 dense size curve (fc = 1.5 x hidden), 200 epochs from scratch.
    *[("nsnet2-size-curve", "nsnet2", r) for r in (
        "dense_h52", "dense_h68", "dense_h68_s2345", "dense_h68_s3456", "dense_h100",
        "dense_h148", "dense_h168", "dense_h168_s2345", "dense_h168_s3456",
        "dense_h216", "dense_h216_s2345", "dense_h216_s3456")],
    # ConvFSENet 257/192/384, 20-epoch masked fine-tunes from cp_convfsenet_win.
    *[("convfsenet-full-size", "convfsenet", r) for r in (
        "cf_dense_control", "cf_2to4", "cf_c1")],
    # ConvFSENet below the knee: dense 64/128 vs 2:4 pruned from 96/192.
    *[("convfsenet-knee", "convfsenet", r) for r in (
        "cfs_c64_dense", "cfs_c96_dense", "cf96_cb_c1", "cf96_cb_c1_s2345",
        "cf96_2to4", "cf96_2to4_s2345", "cf96_ft", "cf96_ft_s2345", "cf64_ft", "cf64_ft_s2345")],
    # ConvFSENet dense size curve (res = r, conv = 2r), 200 epochs from scratch.
    *[("convfsenet-size-curve", "convfsenet", r) for r in (
        "cfs_dense_r67", "cfs_dense_r83", "cfs_dense_r108", "cfs_dense_r146", "cfs_dense")],
]

# Paired comparisons: (study, label, first run, second run). The Δ is first - second.
PAIRS = [
    ("nsnet2-full-size", "codebook c1 vs free 2:4", "cb_c1", "ov_2to4"),
    ("nsnet2-full-size", "codebook c2 vs free 2:4", "cb_c2", "ov_2to4"),
    ("nsnet2-full-size", "codebook c3 vs free 2:4", "cb_c3", "ov_2to4"),
    ("nsnet2-square32", "mask cost at 384 (C2), seed 1234", "sq384_cb_c1", "sq384_ft"),
    ("nsnet2-square32", "mask cost at 384 (C2), seed 2345", "sq384_cb_c1_s2345", "sq384_ft_s2345"),
    ("nsnet2-square32", "sparse vs matched dense at ~1.2M (C3), seed 1234", "sq384_cb_c1", "dense_h256f384_ft"),
    ("nsnet2-square32", "sparse vs matched dense at ~1.2M (C3), seed 2345", "sq384_cb_c1_s2345", "dense_h256f384_ft_s2345"),
    ("nsnet2-knee", "D: sparse 192 vs dense 128, seed 1234", "sq192_cb_c1", "sq128_ft"),
    ("nsnet2-knee", "D: sparse 192 vs dense 128, seed 2345", "sq192_cb_c1_s2345", "sq128_ft_s2345"),
    ("nsnet2-knee", "U: dense 192 vs dense 128, seed 1234", "sq192_ft", "sq128_ft"),
    ("nsnet2-knee", "U: dense 192 vs dense 128, seed 2345", "sq192_ft_s2345", "sq128_ft_s2345"),
    ("nsnet2-knee", "mask cost at 192, seed 1234", "sq192_cb_c1", "sq192_ft"),
    ("nsnet2-knee", "mask cost at 192, seed 2345", "sq192_cb_c1_s2345", "sq192_ft_s2345"),
    ("convfsenet-full-size", "free 2:4 vs dense", "cf_2to4", "cf_dense_control"),
    ("convfsenet-full-size", "codebook c1 vs free 2:4", "cf_c1", "cf_2to4"),
    ("convfsenet-knee", "D: sparse 96 (c1) vs dense 64, seed 1234", "cf96_cb_c1", "cf64_ft"),
    ("convfsenet-knee", "D: sparse 96 (c1) vs dense 64, seed 2345", "cf96_cb_c1_s2345", "cf64_ft_s2345"),
    ("convfsenet-knee", "U: dense 96 vs dense 64, seed 1234", "cf96_ft", "cf64_ft"),
    ("convfsenet-knee", "U: dense 96 vs dense 64, seed 2345", "cf96_ft_s2345", "cf64_ft_s2345"),
    ("convfsenet-knee", "plain 2:4 96 vs dense 64, seed 1234", "cf96_2to4", "cf64_ft"),
    ("convfsenet-knee", "plain 2:4 96 vs dense 64, seed 2345", "cf96_2to4_s2345", "cf64_ft_s2345"),
    ("convfsenet-knee", "codebook vs free 2:4 at 96, seed 1234", "cf96_cb_c1", "cf96_2to4"),
    ("convfsenet-knee", "codebook vs free 2:4 at 96, seed 2345", "cf96_cb_c1_s2345", "cf96_2to4_s2345"),
    ("convfsenet-knee", "mask cost at 96 (c1), seed 1234", "cf96_cb_c1", "cf96_ft"),
    ("convfsenet-knee", "mask cost at 96 (c1), seed 2345", "cf96_cb_c1_s2345", "cf96_ft_s2345"),
]

_PESQ = re.compile(r"PESQ Score: ([\d.]+)")
_EPOCH = re.compile(r"Time taken for epoch \d+ is")
_INIT = re.compile(r"Warm-started (?:generator|weights) from (\S+)")


def _log_text(run: str, root: str = ".") -> str:
    """The run's own log; the older size-sweep runs live inside a shared log.

    ``root`` is the checkout holding ``cp_<run>`` (runs of one study can live in
    different worktrees)."""
    for path in (f"cp_{run}.log", f"cp_{run}/train.log"):
        if os.path.exists(os.path.join(root, path)):
            return open(os.path.join(root, path)).read()
    shared = open(os.path.join(root, "cp_dense_matched_sweep.log")).read()
    blocks = re.split(r"=== \[[^\]]*\] Run: ", shared)[1:]
    for block in blocks:
        if block.split()[0] == run:
            return block
    raise FileNotFoundError(f"no log for {run}")


def _params(run: str, root: str = ".") -> tuple[int, int]:
    with contextlib.redirect_stdout(sys.stderr):      # load_checkpoint prints a banner
        ckpt = load_checkpoint(os.path.join(root, f"cp_{run}/g_best"), torch.device("cpu"))
    sd = ckpt.get("generator", ckpt)
    total = nonzero = 0
    for name, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point() and "running_" not in name:
            total += v.numel()
            nonzero += int((v != 0).sum())
    return total, nonzero


def collect() -> list[dict]:
    rows = []
    for study, model, run in RUNS:
        text = _log_text(run)
        traj = [float(x) for x in _PESQ.findall(text)]
        cfg = json.load(open(f"cp_{run}/config.json"))
        sp = cfg.get("sparsity") or {}
        width = (f"{cfg.get('hidden_dim')}/{cfg.get('fc_hidden_dim')}" if model == "nsnet2"
                 else f"{cfg.get('n_channels_res')}/{cfg.get('n_channels_conv')}")
        init = _INIT.search(text)
        total, nonzero = _params(run)
        rows.append({
            "study": study, "model": model, "run": run, "width": width,
            "mask": sp.get("pattern", "dense") if sp.get("enabled") else "dense",
            "init": re.sub(r".*/hf_baseline/", "hf:", init.group(1)) if init else "scratch",
            "epochs": len(_EPOCH.findall(text)), "seed": cfg.get("seed"),
            "lr": cfg.get("learning_rate"), "total_params": total, "nonzero_params": nonzero,
            "n_val": len(traj), "best": round(max(traj), 4), "last5": round(st.mean(traj[-5:]), 4),
            "pesq_trajectory": ";".join(f"{v:.3f}" for v in traj),
        })
    return rows


def paired(rows: list[dict]) -> list[dict]:
    by = {r["run"]: [float(v) for v in r["pesq_trajectory"].split(";")] for r in rows}
    out = []
    for study, label, a, b in PAIRS:
        d = [x - y for x, y in zip(by[a], by[b])]
        out.append({"study": study, "comparison": label, "first": a, "second": b,
                    "delta_last5": round(st.mean(d[-5:]), 4),
                    "leads": f"{sum(v > 0 for v in d)}/{len(d)}"})
    return out


def markdown(rows: list[dict]) -> str:
    lines = []
    for study in dict.fromkeys(r["study"] for r in rows):
        lines += [f"### {study}", "",
                  "| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in (r for r in rows if r["study"] == study):
            lines.append(f"| `{r['run']}` | {r['width']} | {r['mask']} | {r['init']} | {r['epochs']} "
                         f"| {r['seed']} | {r['nonzero_params']:,} | {r['best']:.3f} | {r['last5']:.3f} |")
        lines.append("")
    lines += ["### Paired comparisons (Δ = first − second, mean over the last 5 validations)", "",
              "| Study | Comparison | Δ | First run leads |", "| --- | --- | ---: | ---: |"]
    for p in paired(rows):
        lines.append(f"| {p['study']} | {p['comparison']} | {p['delta_last5']:+.3f} | {p['leads']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="", help="write the per-run table here")
    ap.add_argument("--markdown", action="store_true", help="print tables + paired comparisons")
    a = ap.parse_args()
    rows = collect()
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv} ({len(rows)} runs)", file=sys.stderr)
    if a.markdown:
        print(markdown(rows))


if __name__ == "__main__":
    main()
