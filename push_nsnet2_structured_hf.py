"""Publish the corrected NSNet2 structured checkpoints to HuggingFace, atomically.

This supersedes the rename-only script: a plain server-side rename would have
carried over the *buggy* int8 ONNX (whose Einsum/structured weights were still
FP32). Instead we upload freshly-corrected artifacts for every affected run.

What this commit does, in ONE `create_commit`:

  1. `blockdiag_*`  <- the old (mislabeled) `monarch_*` block-diagonal models,
     re-exported and re-quantized locally with the Einsum weight-quant fix, and
     with `config.json` corrected to `"kind": "blockdiag"`.
       cp_blockdiag_8/ cp_blockdiag_full/ cp_blockdiag_fc/ cp_wide_blockdiag/

  2. `monarch_*`    <- the NEW genuine two-factor Monarch models. These OVERWRITE
     the old `monarch_*` paths, which is the point: that name now means the real
     thing, and the block-diagonal models it used to hold live at `blockdiag_*`.
       cp_monarch_8_triton/ cp_monarch_full_triton/ cp_monarch_fc_triton/
       cp_wide_monarch_triton/

  3. `README.md`    <- refreshed model card documenting both corrections.

Nothing is deleted: all four old `monarch_*` paths are overwritten by the genuine
Monarch runs, and their previous (block-diagonal) contents are preserved under the
new `blockdiag_*` paths. One atomic commit, so the repo is never half-migrated.

Usage
-----
    python push_nsnet2_structured_hf.py --dry-run     # preview, no writes
    python push_nsnet2_structured_hf.py               # publish

Requires `huggingface_hub` and a WRITE-scoped HF login (`huggingface-cli login`
or HF_TOKEN).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

# repo_subdir -> local checkpoint dir (relative to --source)
RUNS = {
    # block-diagonal (the renamed, re-quantized former "monarch_*")
    "blockdiag_8": "cp_blockdiag_8",
    "blockdiag_full": "cp_blockdiag_full",
    "blockdiag_fc": "cp_blockdiag_fc",
    "wide_blockdiag": "cp_wide_blockdiag",
    # genuine two-factor Monarch (NEW; overwrites the old monarch_* paths)
    "monarch_8": "cp_monarch_8_triton",
    "monarch_full": "cp_monarch_full_triton",
    "monarch_fc": "cp_monarch_fc_triton",
    "wide_monarch": "cp_wide_monarch_triton",
}

FILES = ("config.json", "g_best", "g_best_fp32.onnx", "g_best.onnx")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--repo-id", default="claroche1/sparse-nsnet2-checkpoints")
    p.add_argument("--source", default="/home/clement/eco8-neaixt",
                   help="Base dir holding the cp_<run>/ checkpoint dirs.")
    p.add_argument("--readme", default="docs/publishing/hf-nsnet2.md",
                   help="Model card to upload as README.md (pass '' to skip).")
    p.add_argument("--commit-message",
                   default="rename monarch_*->blockdiag_* (they were single "
                           "block-diagonal factors), publish genuine 2-factor "
                           "Monarch runs, re-quantize int8 with Einsum weight fix")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    base = Path(a.source)
    ops, plan, missing = [], [], []

    for repo_dir, cp_dir in RUNS.items():
        for fname in FILES:
            local = base / cp_dir / fname
            if not local.is_file():
                missing.append(str(local))
                continue
            ops.append(CommitOperationAdd(
                path_in_repo=f"{repo_dir}/{fname}", path_or_fileobj=str(local)
            ))
            mb = local.stat().st_size / 1e6
            plan.append(f"  {repo_dir}/{fname:18s} <- {cp_dir}/{fname}  ({mb:.2f} MB)")

    if missing:
        sys.exit("missing required files (nothing uploaded):\n  " + "\n  ".join(missing))

    if a.readme:
        if not Path(a.readme).is_file():
            sys.exit(f"--readme not found: {a.readme}")
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=a.readme))
        plan.append(f"  README.md          <- {a.readme}")

    print(f"repo:   {a.repo_id}")
    print(f"source: {base}")
    print(f"operations ({len(ops)}):")
    print("\n".join(plan))
    print("\nNOTE: monarch_* paths are OVERWRITTEN with the genuine 2-factor Monarch "
          "models; their previous block-diagonal contents are preserved at blockdiag_*.")

    if a.dry_run:
        print("\n[dry-run] nothing pushed.")
        return

    info = HfApi().create_commit(
        repo_id=a.repo_id, repo_type="model",
        operations=ops, commit_message=a.commit_message,
    )
    print(f"\ndone. {getattr(info, 'commit_url', info)}")
    print(f"model page: https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
