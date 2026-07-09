"""Rename the mislabeled `monarch_*` NSNet2 checkpoints to `blockdiag_*` on the
HuggingFace model repo, in a single atomic commit.

Why
---
The runs published as `monarch_*` are single block-diagonal-factor models (zero
cross-block mixing), NOT genuine two-factor Monarch matrices. This script fixes
the naming on `claroche1/sparse-nsnet2-checkpoints`:

    monarch_8    -> blockdiag_8
    monarch_full -> blockdiag_full
    monarch_fc   -> blockdiag_fc
    wide_monarch -> wide_blockdiag

For each renamed run it:
  - copies every file `monarch_x/<f>` to `blockdiag_y/<f>` (server-side, no
    re-upload of the large LFS blobs — uses CommitOperationCopy),
  - EXCEPT `config.json`, which is downloaded, its `"kind"` values rewritten
    (`monarch -> blockdiag`, `triton_monarch -> triton_blockdiag`), and
    re-added so the on-Hub config matches the corrected loader taxonomy
    (where `kind="monarch"` now means the genuine two-factor layer),
  - deletes the old `monarch_x/<f>` path.
Optionally also replaces the top-level `README.md` model card (`--readme`).

Everything lands in ONE `create_commit`, so the repo is never in a
half-renamed state.

Usage
-----
Preview (no writes):

    python rename_hf_monarch_to_blockdiag.py --dry-run

Execute the rename + push the updated model card:

    python rename_hf_monarch_to_blockdiag.py --readme hf_readme_nsnet2.md

Requires `huggingface_hub` and a logged-in HF account with WRITE access
(`huggingface-cli login` with a write token, or set HF_TOKEN).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

# old run name -> new run name
RENAME = {
    "monarch_8": "blockdiag_8",
    "monarch_full": "blockdiag_full",
    "monarch_fc": "blockdiag_fc",
    "wide_monarch": "wide_blockdiag",
}

# config.json "kind" string rewrites (order matters: triton_ prefix first).
KIND_REWRITES = (
    ("triton_monarch", "triton_blockdiag"),
    ("monarch", "blockdiag"),
)


def _rewrite_config_kinds(cfg: dict) -> dict:
    """Rewrite the block-diagonal 'monarch' kinds to 'blockdiag' in a config."""
    for block in ("linear", "gru"):
        b = cfg.get(block)
        if isinstance(b, dict) and "kind" in b:
            for old, new in KIND_REWRITES:
                if b["kind"] == old:
                    b["kind"] = new
                    break
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--repo-id", default="claroche1/sparse-nsnet2-checkpoints")
    p.add_argument("--readme", default=None,
                   help="Optional path to a new top-level README.md model card to upload "
                        "in the same commit (e.g. hf_readme_nsnet2.md).")
    p.add_argument("--commit-message",
                   default="rename monarch_* -> blockdiag_* (single block-diagonal factor, "
                           "not a true 2-factor Monarch)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned operations without touching HF.")
    a = p.parse_args()

    if a.readme is not None and not Path(a.readme).is_file():
        sys.exit(f"--readme path does not exist: {a.readme}")

    api = HfApi()
    all_files = api.list_repo_files(a.repo_id, repo_type="model")

    operations = []       # list of huggingface_hub CommitOperation*
    plan = []             # human-readable log lines
    tmpdir = tempfile.mkdtemp(prefix="hf_rename_")

    for old, new in RENAME.items():
        run_files = [f for f in all_files if f.startswith(f"{old}/")]
        if not run_files:
            plan.append(f"!! no files found under {old}/ — skipping")
            continue
        for f in run_files:
            rel = f[len(old) + 1:]          # path within the run dir
            new_path = f"{new}/{rel}"
            if rel == "config.json":
                # Download + rewrite kinds + re-add (Copy would keep the stale
                # "monarch" kind bytes verbatim).
                local = hf_hub_download(a.repo_id, f, repo_type="model")
                cfg = _rewrite_config_kinds(json.load(open(local)))
                edited = Path(tmpdir) / f"{new}_config.json"
                edited.write_text(json.dumps(cfg, indent=4))
                operations.append(
                    CommitOperationAdd(path_in_repo=new_path, path_or_fileobj=str(edited))
                )
                plan.append(f"  ADD  {new_path:34s} (config.json, kinds rewritten)")
            else:
                operations.append(
                    CommitOperationCopy(src_path_in_repo=f, path_in_repo=new_path)
                )
                plan.append(f"  COPY {f:34s} -> {new_path}")
            operations.append(CommitOperationDelete(path_in_repo=f))
            plan.append(f"  DEL  {f}")

    if a.readme:
        operations.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=a.readme)
        )
        plan.append(f"  ADD  README.md (model card <- {a.readme})")

    print(f"repo:    {a.repo_id}")
    print(f"renames: " + ", ".join(f"{o}->{n}" for o, n in RENAME.items()))
    print(f"operations ({len(operations)}):")
    print("\n".join(plan))

    if not operations:
        sys.exit("nothing to do (no matching files found)")

    if a.dry_run:
        print("\n[dry-run] no changes pushed. Re-run without --dry-run to execute.")
        return

    print("\npushing single atomic commit ...")
    info = api.create_commit(
        repo_id=a.repo_id,
        repo_type="model",
        operations=operations,
        commit_message=a.commit_message,
    )
    print(f"done. {getattr(info, 'commit_url', info)}")
    print(f"model page: https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
