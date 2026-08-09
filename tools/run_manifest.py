#!/usr/bin/env python3
"""Emit an immutable provenance manifest for an experiment, and optionally run it.

The point of this script is that a result should never be separable from the
conditions that produced it. A PESQ number with no git SHA, no seed, no package
versions and no host is not reproducible and is not evidence.

Stdlib only, on purpose: it must run inside a board-measurement venv or a CI
container as happily as inside the project env.

Usage
-----
Wrap a run (recommended -- records exit code, duration and stdout):

    python tools/run_manifest.py --id nsnet2-monarch8-seed13 \\
        --hypothesis dynse-oracle-001 --config configs/monarch_8.json \\
        --seed 13 --dataset JacobLinCool/VoiceBank-DEMAND-16k --split train \\
        -- python -m nsnet2.train --config configs/monarch_8.json \\
                  --checkpoint_path cp_monarch_8

Record a manifest for something measured elsewhere (a board session):

    python tools/run_manifest.py --id lisennet-nc24-n6-stream-2026-08-09 \\
        --claim lisennet-n6-streaming-deployed --no-run \\
        --toolchain stedgeai=4.0.1 --toolchain cubeprog=2.22 \\
        --toolchain board=STM32N6570-DK --note "n6-noextmem, dev-boot"

Manifests are written to research/experiments/<id>/manifest.json and are
IMMUTABLE. Re-running with an existing id fails unless --force is given; a
re-run should get a new id so the old evidence survives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_guard import is_forbidden_selection_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "research" / "experiments"

# Packages whose version can move a number. Extend when a new one can.
TRACKED_PACKAGES = [
    "torch",
    "torchaudio",
    "numpy",
    "scipy",
    "onnx",
    "onnxruntime",
    "pesq",
    "torchmetrics",
    "scoreq",
    "librosa",
    "soundfile",
    "torch-structured",
    "gru-qat",
    "datasets",
]

# Environment variables that change numerics or device selection.
TRACKED_ENV = [
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
]


def _sh(*args: str, keep_leading_space: bool = False) -> str | None:
    """Run a command in the repo root. Returns None on failure.

    `keep_leading_space` matters for `git status --porcelain`: the first two
    columns are the status code and an unmodified file's index column is a
    space, so stripping the output shifts every path by one character.
    """
    try:
        out = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.rstrip("\n") if keep_leading_space else out.stdout.strip()


def git_state() -> dict:
    sha = _sh("git", "rev-parse", "HEAD")
    status = _sh("git", "status", "--porcelain", keep_leading_space=True)
    return {
        "sha": sha,
        "branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        # The actual diff, not just a flag: a dirty run whose diff is lost is
        # not reproducible, and dirty runs happen.
        "dirty_files": sorted(line[3:] for line in status.splitlines()) if status else [],
        "diff_sha256": (
            hashlib.sha256((_sh("git", "diff", "HEAD") or "").encode()).hexdigest()
            if status
            else None
        ),
    }


def package_versions() -> dict:
    versions: dict[str, str | None] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - py<3.8 only
        return versions
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def file_digest(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path.relative_to(REPO_ROOT)) if _under_repo(path) else str(path),
            "sha256": h.hexdigest(),
            "bytes": path.stat().st_size}


def _under_repo(path: Path) -> bool:
    try:
        path.relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def accelerator() -> dict:
    """Best-effort device identity. Absent torch is not an error."""
    info: dict = {"torch_available": False}
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return info
    info["torch_available"] = True
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["cuda_version"] = torch.version.cuda
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    return info


def build_manifest(a: argparse.Namespace, command: list[str]) -> dict:
    return {
        "schema": "eco8-experiment-manifest/1",
        "id": a.id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": a.note,
        "serves": {
            "hypotheses": a.hypothesis,
            "claims": a.claim,
        },
        "command": command,
        "cwd": str(Path.cwd()),
        "git": git_state(),
        "config": {
            "path": a.config,
            "digest": file_digest(Path(a.config)) if a.config else None,
            "resolved": _load_json(a.config),
        },
        "data": {
            "dataset": a.dataset,
            "revision": a.dataset_revision,
            "split": a.split,
            "calibration_set": a.calibration_set,
        },
        "seeds": a.seed,
        "checkpoint": file_digest(Path(a.checkpoint)) if a.checkpoint else None,
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "packages": package_versions(),
            "lockfile": file_digest(REPO_ROOT / "uv.lock"),
            "env": {k: os.environ.get(k) for k in TRACKED_ENV},
            "accelerator": accelerator(),
        },
        # Compiler / firmware / board / measurement-tool versions. Free-form
        # because the embedded toolchain is not introspectable from Python.
        "toolchain": dict(kv.split("=", 1) for kv in a.toolchain),
        "agent": {
            "model": os.environ.get("ECO8_AGENT_MODEL"),
            "harness": os.environ.get("ECO8_AGENT_HARNESS"),
            "session": os.environ.get("ECO8_AGENT_SESSION"),
        },
        "result": None,  # filled in after the run
    }


def _load_json(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--id", required=True, help="experiment id; becomes the directory name")
    ap.add_argument("--hypothesis", action="append", default=[],
                    help="hypothesis card id this run serves (repeatable)")
    ap.add_argument("--claim", action="append", default=[],
                    help="CLAIMS.yaml id this run is evidence for (repeatable)")
    ap.add_argument("--config", help="path to the run's JSON config")
    ap.add_argument("--dataset", help="e.g. JacobLinCool/VoiceBank-DEMAND-16k")
    ap.add_argument("--dataset-revision", help="HF revision / commit of the dataset")
    ap.add_argument("--split", help="split used; anything that SELECTS must not be 'test'")
    ap.add_argument("--calibration-set", help="identity of the int8 calibration set")
    ap.add_argument("--seed", action="append", type=int, default=[], help="repeatable")
    ap.add_argument("--checkpoint", help="checkpoint or ONNX file this run consumed")
    ap.add_argument("--toolchain", action="append", default=[], metavar="KEY=VALUE",
                    help="e.g. stedgeai=4.0.1, board=STM32N6570-DK (repeatable)")
    ap.add_argument("--note", help="one line: what this run is for")
    ap.add_argument("--no-run", action="store_true",
                    help="record the manifest only (for work measured elsewhere)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest -- avoid; use a new id")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="-- followed by the command to run")
    a = ap.parse_args(argv)

    command = [c for c in a.command if c != "--"]
    if not command and not a.no_run:
        ap.error("give a command after `--`, or pass --no-run")

    if is_forbidden_selection_split(a.split):
        print(
            f"refusing: --split {a.split!r}. The test split selects nothing "
            "(AGENTS.md rule 7). Use a train holdout.",
            file=sys.stderr,
        )
        return 2

    outdir = EXPERIMENTS / a.id
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists() and not a.force:
        print(
            f"refusing: {manifest_path.relative_to(REPO_ROOT)} exists. Manifests are "
            "immutable -- use a new id so the old evidence survives.",
            file=sys.stderr,
        )
        return 2
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(a, command)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"manifest: {manifest_path.relative_to(REPO_ROOT)}")

    if a.no_run:
        return 0

    log_path = outdir / "stdout.log"
    started = time.time()
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=Path.cwd())
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            log.write(line)
        rc = proc.wait()
    elapsed = time.time() - started

    manifest["result"] = {
        "exit_code": rc,
        "wall_seconds": round(elapsed, 3),
        "stdout_log": str(log_path.relative_to(REPO_ROOT)),
        # A failed run is evidence too. Record it; do not delete the directory.
        "outcome": "ok" if rc == 0 else "failed",
        # The post-run git state catches a run that mutated the tree.
        "git_after": git_state(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"manifest: {manifest_path.relative_to(REPO_ROOT)}  exit={rc}  {elapsed:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
