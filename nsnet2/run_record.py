"""Weight snapshots and run records for ``nsnet2.train`` (all opt-in).

* ``g_eNNN``: the generator after NNN finished epochs (``g_e000``: right after
  it is built, before any optimiser step), saved from CPU copies in the format
  of ``g_best`` plus ``epoch`` and ``steps``.
* ``states/s_eNNN``: the full training state at the end of epoch NNN
  (generator, discriminator, both optimisers and schedulers, step, epoch, best
  validation PESQ and the random-number-generator states), so that a run can be
  extended later.
* ``run.json`` (what was run, from which commit, with which libraries and GPU,
  start and end time, fingerprint of ``g_e000``), ``best.json`` (when ``g_best``
  is written) and ``val.jsonl`` (one line per validation).

The fingerprint of a state dict is the sha256 of the float32 (little-endian)
bytes of its tensors in sorted key order. Unlike the hash of a ``torch.save``
file, it does not depend on the device or the PyTorch version.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_epoch_list(spec: str) -> list[int]:
    """Sorted epochs of a list such as ``"0-20,25-200:5"`` (items ``N``, ``A-B`` or ``A-B:STEP``)."""
    out = set()
    for item in (spec or "").replace(" ", "").split(","):
        if not item:
            continue
        rng, _, step = item.partition(":")
        lo, _, hi = rng.partition("-")
        try:
            lo_i = int(lo)
            hi_i = int(hi) if hi else lo_i
            step_i = int(step) if step else 1
        except ValueError:
            raise ValueError(f"bad epoch list item {item!r} in {spec!r}") from None
        if lo_i < 0 or hi_i < lo_i or step_i < 1:
            raise ValueError(f"bad epoch list item {item!r} in {spec!r}")
        out.update(range(lo_i, hi_i + 1, step_i))
    return sorted(out)


def state_fingerprint(state_dict) -> str:
    """sha256 of the float32 little-endian bytes of the tensors, in sorted key order."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().to("cpu", torch.float32).contiguous()
        h.update(t.numpy().astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_copy(obj):
    """The same nested structure with every tensor detached, cloned and moved to the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return type(obj)((k, cpu_copy(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(cpu_copy(v) for v in obj)
    return obj


def check_fresh_run_dir(path) -> None:
    """Refuse a directory that holds rolling checkpoints (train.py would resume from them), snapshots,
    training states or a run.json."""
    p = Path(path)
    if not p.exists():
        return
    rolling = sorted(glob.glob(str(p / "g_????????")) + glob.glob(str(p / "do_????????")))
    if rolling:
        raise SystemExit(f"{p} holds rolling checkpoints ({os.path.basename(rolling[-1])}, ...): train.py "
                         "would resume from them. Start the run in a fresh directory.")
    for sub in ("snapshots", "states"):
        d = p / sub
        if d.is_dir() and any(d.iterdir()):
            raise SystemExit(f"{d} is not empty: start the run in a fresh directory.")
    if (p / "run.json").exists():
        raise SystemExit(f"{p / 'run.json'} exists: start the run in a fresh directory.")


def load_generator_state(path) -> dict:
    """Generator state dict of a snapshot, of ``g_best`` / ``g_XXXXXXXX``, or a bare state dict."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck


def apply_init_from(model: torch.nn.Module, path, prefix: str) -> list[str]:
    """Overwrite the tensors of ``model`` whose keys start with ``prefix`` by those of the file. Draws
    nothing from any random generator. Returns the overwritten keys."""
    src = load_generator_state(path)
    sd = model.state_dict()
    keys = [k for k in sd if k.startswith(prefix)]
    if not keys:
        raise ValueError(f"no tensor of the model has a key starting with {prefix!r}")
    for k in keys:
        if k not in src:
            raise ValueError(f"{path} has no tensor {k!r}")
        if tuple(src[k].shape) != tuple(sd[k].shape):
            raise ValueError(f"{k}: shape {tuple(src[k].shape)} in {path}, {tuple(sd[k].shape)} in the model")
    model.load_state_dict({**sd, **{k: src[k].to(sd[k].dtype) for k in keys}}, strict=True)
    return keys


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


class RunRecord:
    """Snapshots, training states and records of one run directory."""

    def __init__(self, run_dir, config_path, seed, snapshot_epochs, state_epochs):
        self.dir = Path(run_dir)
        self.snapshot_epochs = set(snapshot_epochs) | {0}
        self.state_epochs = set(state_epochs)
        self.dir.mkdir(parents=True, exist_ok=True)
        commit = _git("rev-parse", "HEAD")
        lock = REPO_ROOT / "uv.lock"
        self.info = {
            "run": self.dir.resolve().name,
            "arm": Path(config_path).stem,
            "seed": int(seed),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "commit": commit.strip() if commit else None,
            "git_status_porcelain": _git("status", "--porcelain"),
            "command": [sys.executable, "-m", "nsnet2.train", *sys.argv[1:]],
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch-structured": _version("torch-structured"),
                "gru-qat": _version("gru-qat"),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            },
            "uv_lock_sha256": file_sha256(lock) if lock.exists() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "start_time": _now(),
            "end_time": None,
            "steps_per_epoch": None,
            "snapshot_epochs": sorted(self.snapshot_epochs),
            "state_epochs": sorted(self.state_epochs),
            "init_from": None,
            "g_e000_fingerprint": None,
        }

    # --- run.json -------------------------------------------------------------------------------
    def write(self) -> None:
        _write_json(self.dir / "run.json", self.info)

    def update(self, **fields) -> None:
        self.info.update(fields)
        self.write()

    def finish(self) -> None:
        self.update(end_time=_now())

    # --- weights ----------------------------------------------------------------------------------
    def init_from(self, model, path, prefix) -> None:
        keys = apply_init_from(model, path, prefix)
        self.info["init_from"] = {"file": str(path), "file_sha256": file_sha256(path),
                                  "file_fingerprint": state_fingerprint(load_generator_state(path)),
                                  "prefix": prefix, "keys": keys}

    def snapshot(self, model, epoch: int, steps: int) -> str:
        sd = cpu_copy(model.state_dict())
        (self.dir / "snapshots").mkdir(exist_ok=True)
        torch.save({"generator": sd, "epoch": int(epoch), "steps": int(steps)},
                   self.dir / "snapshots" / f"g_e{epoch:03d}")
        return state_fingerprint(sd)

    def start(self, model) -> str:
        """Write g_e000 and run.json; return the fingerprint of g_e000."""
        fp = self.snapshot(model, 0, 0)
        self.update(g_e000_fingerprint=fp)
        return fp

    def end_of_epoch(self, epoch, steps, generator, discriminator, optim_g, optim_d, scheduler_g,
                     scheduler_d, best_pesq) -> None:
        """Called once epoch ``epoch`` (counted from 1) has finished."""
        if epoch in self.snapshot_epochs:
            self.snapshot(generator, epoch, steps)
        if epoch in self.state_epochs:
            (self.dir / "states").mkdir(exist_ok=True)
            rng = {"torch": torch.get_rng_state(), "python": random.getstate(),
                   "numpy": np.random.get_state()}
            if torch.cuda.is_available():
                rng["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(cpu_copy({
                "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                "optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(), "scheduler_d": scheduler_d.state_dict(),
                "steps": int(steps), "epoch": int(epoch), "best_pesq": float(best_pesq), "rng": rng,
            }), self.dir / "states" / f"s_e{epoch:03d}")

    # --- validation records -----------------------------------------------------------------------
    def validation(self, step, epochs_finished, pesq, loss_mag, loss_com, loss_stft) -> None:
        line = {"step": int(step), "optimizer_steps": int(step) + 1, "epochs_finished": int(epochs_finished),
                "pesq": float(pesq), "loss_mag": float(loss_mag), "loss_com": float(loss_com),
                "loss_stft": float(loss_stft)}
        with open(self.dir / "val.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")

    def best(self, step, epochs_finished, pesq) -> None:
        _write_json(self.dir / "best.json", {"step": int(step), "optimizer_steps": int(step) + 1,
                                             "epochs_finished": int(epochs_finished), "pesq": float(pesq)})
