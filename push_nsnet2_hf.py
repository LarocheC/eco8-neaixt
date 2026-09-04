"""Upload the NSNet2 sweep checkpoints to the HuggingFace model repo.

Mirrors push_convfsenet_hf.py, but for the multi-variant NSNet2 repo
`claroche1/sparse-nsnet2-checkpoints`, whose layout is one subdirectory per
run:

    <run>/config.json
    <run>/g_best            # PyTorch checkpoint ({"generator": state_dict})
    <run>/g_best_fp32.onnx  # streaming FP32 ONNX
    <run>/g_best.onnx       # static int8 ONNX
    README.md               # top-level model card (shared across runs)

Each local run lives in `cp_<run>/` (the layout the sweep scripts produce).
Files are staged into a temp tree mirroring the repo layout and pushed as a
SINGLE commit via `upload_folder`, so a refresh is atomic.

Typical uses
------------
Full mirror of all 9 runs (first upload / full refresh):

    python push_nsnet2_hf.py

Minimal-churn refresh — push ONLY the re-quantized int8 ONNX per run plus an
updated model card (this is the post-calibration-fix refresh):

    python push_nsnet2_hf.py --only g_best.onnx --readme docs/publishing/hf-nsnet2.md \\
        --commit-message "re-quantize int8 after calibration RMS-norm fix"

Preview without touching HF:

    python push_nsnet2_hf.py --only g_best.onnx --dry-run

Requires `huggingface_hub` and a logged-in HF account with WRITE access
(`huggingface-cli login` with a write token, or set HF_TOKEN). A read-only
token fails with 403 at upload time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


# The canonical 9-run sweep (matches run_sweep.sh / docs/models/nsnet2.md).
DEFAULT_RUNS = (
    "baseline", "monarch_fc", "butterfly_fc", "monarch_full", "butterfly_full",
    "monarch_8", "butterfly_ortho", "butterfly_2blocks", "wide_monarch",
)

# Per-run artifacts, in the same order push_convfsenet_hf.py uses.
ALL_FILES = ("config.json", "g_best", "g_best_fp32.onnx", "g_best.onnx")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--source", default=".",
                   help="Base dir holding the cp_<run>/ directories (default: cwd).")
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                   help=f"Run names to upload (default: the {len(DEFAULT_RUNS)}-run sweep).")
    p.add_argument("--repo-id", default="claroche1/sparse-nsnet2-checkpoints",
                   help="Target HF model repo.")
    p.add_argument("--only", nargs="+", default=list(ALL_FILES),
                   choices=ALL_FILES,
                   help="Subset of per-run files to upload (default: all 4). "
                        "Use '--only g_best.onnx' for an int8-only refresh.")
    p.add_argument("--cp-prefix", default="cp_",
                   help="Local dir prefix for each run (default: 'cp_', i.e. cp_<run>).")
    p.add_argument("--readme", default=None,
                   help="Optional path to a top-level README.md (model card) to upload. "
                        "If omitted, the existing repo README is left untouched.")
    p.add_argument("--commit-message", default="upload NSNet2 sweep checkpoints",
                   help="Commit message for the HF upload.")
    p.add_argument("--private", action="store_true",
                   help="Create the HF repo as private (default: public).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen, but don't touch HF.")
    a = p.parse_args()

    base = Path(a.source).resolve()
    if not base.is_dir():
        sys.exit(f"source base dir does not exist: {base}")

    # Resolve and validate every (run, file) before staging anything.
    to_upload = []          # list of (repo_path, local_path)
    missing = []
    for run in a.runs:
        cp_dir = base / f"{a.cp_prefix}{run}"
        for fname in a.only:
            local = cp_dir / fname
            if local.is_file():
                to_upload.append((f"{run}/{fname}", local))
            else:
                missing.append(str(local))
    if missing:
        sys.exit("missing required files (nothing uploaded):\n  " + "\n  ".join(missing))
    if not to_upload:
        sys.exit("no files resolved to upload (check --runs / --only / --source)")

    if a.readme is not None and not Path(a.readme).is_file():
        sys.exit(f"--readme path does not exist: {a.readme}")

    # Sanity-check each run's config so we don't push a mislabeled model.
    if "config.json" in a.only:
        for run in a.runs:
            cfg_path = base / f"{a.cp_prefix}{run}" / "config.json"
            with open(cfg_path) as f:
                cfg = json.load(f)
            gru = (cfg.get("gru") or {}).get("kind", "?")
            lin = (cfg.get("linear") or {}).get("kind", "?")
            print(f"  {run:18s} gru={gru:16s} linear={lin}")

    print(f"source base:   {base}")
    print(f"repo-id:       {a.repo_id}")
    print(f"runs ({len(a.runs)}):     {', '.join(a.runs)}")
    print(f"per-run files: {', '.join(a.only)}")
    print(f"README:        {a.readme or '(left untouched)'}")
    print(f"total objects: {len(to_upload)}" + (" + README.md" if a.readme else ""))
    print(f"private:       {a.private}")

    if a.dry_run:
        print("\n[dry-run] would create repo + upload the paths below in one commit:")
        for repo_path, _ in to_upload:
            print(f"  {repo_path}")
        if a.readme:
            print("  README.md")
        return

    api = HfApi()
    print(f"\nensuring repo exists: {a.repo_id}")
    create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)

    # Stage the chosen files into a temp tree mirroring the repo layout, then
    # upload the whole tree as ONE commit. upload_folder only touches paths
    # present in the staged tree, so '--only g_best.onnx' leaves the PyTorch
    # checkpoints / FP32 ONNX / configs already on the Hub unchanged.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for repo_path, local in to_upload:
            dst = staged / repo_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dst)        # copy (not symlink) for Windows safety
        if a.readme:
            shutil.copy2(a.readme, staged / "README.md")

        print(f"uploading {len(to_upload)} file(s) to {a.repo_id} ...")
        url = api.upload_folder(
            folder_path=str(staged),
            repo_id=a.repo_id,
            repo_type="model",
            commit_message=a.commit_message,
        )
    print(f"\ndone. {url}")
    print(f"model page: https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
