"""Fine-tune a trained enhancer with the differentiable PESQ loss, and compare.

Study protocol (docs/studies/pesq-finetune.md): from ONE base checkpoint, run two
arms with an identical recipe -- a *control* arm (pesq weight 0) and a *pesq*
arm -- so the PESQ term is the only difference. Fine-tuning itself moves the
numbers (more steps, a fresh discriminator, a lower learning rate), and without
the control that would be credited to the loss.

    # 1. both arms (base = a local g_best with config.json beside it, or a Hub model)
    python finetune_pesq.py train --model lisennet --base hf:conv-hardened \\
        --out cp_ft/conv-hardened__control --pesq_weight 0
    python finetune_pesq.py train --model lisennet --base hf:conv-hardened \\
        --out cp_ft/conv-hardened__pesq0.2 --pesq_weight 0.2

    # 2. paired comparison on the VBD test split, PESQ plus the metrics nobody optimised
    python finetune_pesq.py compare --model lisennet base=hf:conv-hardened \\
        control=cp_ft/conv-hardened__control/g_best pesq=cp_ft/conv-hardened__pesq0.2/g_best \\
        --metrics all --json results/pesq_finetune/conv-hardened.json --md results/pesq_finetune/conv-hardened.md

``run_finetune_pesq.sh`` chains the three commands. The PESQ loss needs triton-pesq
(``uv sync --group pesq-loss``); on CUDA it uses the Triton backend.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from common.env import AttrDict

FAMILIES = ("lisennet", "convfsenet")
HUB_REPOS = {"lisennet": "claroche1/LiSenNet", "convfsenet": "claroche1/convfsenet"}


# --- checkpoints ----------------------------------------------------------------

def resolve_checkpoint(spec: str, family: str, cache_dir: str | None = None) -> Path:
    """A local ``.../g_best`` path, or ``hf:<subfolder>`` for a model published on the Hub
    (``hf:conv-hardened``; ``hf:`` alone = the repo root, as for ConvFSENet)."""
    if not spec.startswith("hf:"):
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint at {path}")
        return path
    from huggingface_hub import hf_hub_download

    sub = spec[3:].strip("/")
    kw = {"subfolder": sub} if sub else {}
    repo = HUB_REPOS[family]
    ckpt = hf_hub_download(repo, "g_best", cache_dir=cache_dir, **kw)
    hf_hub_download(repo, "config.json", cache_dir=cache_dir, **kw)  # lands beside g_best
    return Path(ckpt)


def load_config(checkpoint: Path) -> dict:
    path = checkpoint.parent / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"expected the run's config.json next to the checkpoint: {path}")
    return json.loads(path.read_text())


def finetune_config(base_cfg: dict, *, base: str, pesq_weight: float, backend: str,
                    lr: float, epochs: int) -> dict:
    """The base run's config (architecture, data, loss weights) with the fine-tune recipe on top."""
    cfg = dict(base_cfg)
    cfg["learning_rate"] = lr
    cfg["pesq_loss"] = {"weight": pesq_weight, "backend": backend}
    cfg["_finetune"] = {
        "base": base, "arm": "pesq" if pesq_weight > 0 else "control",
        "pesq_weight": pesq_weight, "learning_rate": lr, "epochs": epochs,
    }
    return cfg


def build_model(family: str, cfg: dict) -> torch.nn.Module:
    h = AttrDict(cfg)
    if family == "lisennet":
        from lisennet.model import build_lisennet
        return build_lisennet(h)
    from convfsenet.model import build_causal_model
    return build_causal_model(h)


# --- train ---------------------------------------------------------------------------

def cmd_train(a) -> int:
    base = resolve_checkpoint(a.base, a.model, a.hf_cache_dir)
    cfg = finetune_config(load_config(base), base=a.base, pesq_weight=a.pesq_weight,
                          backend=a.backend, lr=a.lr, epochs=a.epochs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        previous = json.loads(cfg_path.read_text()).get("_finetune", {})
        if previous and previous != cfg["_finetune"]:
            raise SystemExit(f"{out} already holds a different fine-tune ({previous}); "
                             "use a new --out rather than mixing arms in one directory")
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")

    trainer = importlib.import_module(f"{a.model}.train")
    argv = ["--config", str(cfg_path), "--checkpoint_path", str(out),
            "--training_epochs", str(a.epochs), "--init_from", str(base),
            "--best_checkpoint_start_epoch", "0"]
    if a.validation_interval:
        argv += ["--validation_interval", str(a.validation_interval)]
    if a.hf_cache_dir:
        argv += ["--hf_cache_dir", a.hf_cache_dir]
    targs = trainer.build_parser().parse_args(argv)
    print(f"Fine-tuning {a.model} from {a.base} -> {out} "
          f"({cfg['_finetune']['arm']} arm, pesq weight {a.pesq_weight}, lr {a.lr}, {a.epochs} epochs)")
    trainer.train(targs, AttrDict(cfg))
    return 0


# --- compare ---------------------------------------------------------------------------

@torch.no_grad()
def enhance(family: str, model: torch.nn.Module, noisy: np.ndarray, device) -> np.ndarray:
    """One utterance at the dataset's level, enhanced the way the family's _validate does
    (offline forward; LiSenNet with its Griffin-Lim phase), then returned at the input
    level (the level DNSMOS/NISQA need, see benchmarks/enhance.py)."""
    x = torch.from_numpy(noisy.astype(np.float32))
    norm = torch.sqrt(len(x) / (torch.sum(x ** 2) + 1e-8))
    x = (x * norm).to(device)
    if family == "lisennet":
        est = model(x[None])["est"][0]
    else:
        x3 = x[None, None]
        est, _, _, _ = model.valid_step(x3, x3)
        est = est.reshape(-1)
    est = est[: len(noisy)] / norm.to(device)
    return est.cpu().numpy().astype(np.float32)


def paired_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Mean of (a - b) over utterances both scored, and its 95% half-width (normal approx)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan"), float("nan"), int(d.size)
    return float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(d.size)), int(d.size)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_markdown(result: dict) -> str:
    names = list(result["checkpoints"])
    cols = result["columns"]
    lines = [f"# PESQ fine-tune comparison ({result['family']}, n={result['n_utterances']})", "",
             "| checkpoint | " + " | ".join(cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for name in names:
        means = result["means"][name]
        lines.append(f"| {name} | " + " | ".join(f"{means[c]:.3f}" for c in cols) + " |")
    if result["deltas"]:
        lines += ["", "Paired deltas, mean ± 95% CI over utterances (sign: first minus second; "
                  "`scoreq_ref` is a distance, lower is better):", "",
                  "| comparison | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for key, per_col in result["deltas"].items():
            cells = [f"{per_col[c]['mean']:+.3f} ± {per_col[c]['ci95']:.3f}" for c in cols]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cmd_compare(a) -> int:
    from datasets import load_dataset

    from common.dataset import HF_DATASET_NAME
    from common.quality import QualitySuite, backend_versions, resolve_metrics

    pairs = [spec.split("=", 1) for spec in a.checkpoints]
    if any(len(p) != 2 for p in pairs):
        raise SystemExit("checkpoints are given as name=path (or name=hf:<subfolder>)")
    device = torch.device("cuda:0" if torch.cuda.is_available() and not a.cpu else "cpu")

    # Only the test parquet (132 MB), not the 2.1 GB of train shards. The dataset card
    # also declares the train split, hence verification_mode="no_checks".
    test = load_dataset(HF_DATASET_NAME, data_files={"test": "data/test-*.parquet"}, split="test",
                        verification_mode="no_checks", cache_dir=a.hf_cache_dir)
    n = len(test) if a.max_utterances is None else min(a.max_utterances, len(test))
    rows = test.select(range(n))
    clean = [np.asarray(r["clean"]["array"], dtype=np.float32) for r in rows]
    noisy = [np.asarray(r["noisy"]["array"], dtype=np.float32) for r in rows]
    clean = [c[: min(len(c), len(x))] for c, x in zip(clean, noisy)]
    noisy = [x[: len(c)] for c, x in zip(clean, noisy)]

    metrics = resolve_metrics(a.metrics) or ("pesq",)
    suite = QualitySuite(metrics=metrics, n_jobs=a.jobs)
    per_utt, meta = {}, {}
    for name, spec in pairs:
        ckpt = resolve_checkpoint(spec, a.model, a.hf_cache_dir)
        model = build_model(a.model, load_config(ckpt)).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device)["generator"])
        t0 = time.time()
        ests = [enhance(a.model, model, x, device) for x in noisy]
        per_utt[name] = {k: v.tolist() for k, v in suite.score(ests, clean).items()}
        meta[name] = {"spec": spec, "path": str(ckpt)}
        print(f"{name}: enhanced + scored {n} utterances in {time.time() - t0:.0f}s")

    names = [name for name, _ in pairs]
    result = {
        "family": a.model, "n_utterances": n, "columns": suite.columns, "checkpoints": meta,
        "means": {k: QualitySuite.summarize(v) for k, v in per_utt.items()},
        "deltas": {},
        "provenance": {"dataset": HF_DATASET_NAME, "split": "test", "git_commit": _git_commit(),
                       "versions": backend_versions(), "device": str(device)},
        "per_utt": per_utt,
    }
    reference = "control" if "control" in names else names[0]
    for name in names:
        for other in dict.fromkeys([reference, names[0]]):
            if name == other:
                continue
            key = f"{name} − {other}"
            result["deltas"][key] = {}
            for col in suite.columns:
                mean, ci, k = paired_delta(per_utt[name][col], per_utt[other][col])
                result["deltas"][key][col] = {"mean": mean, "ci95": ci, "n": k}

    md = render_markdown(result)
    print(md)
    for path, text in ((a.json, json.dumps(result, indent=1) + "\n"), (a.md, md)):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            Path(path).write_text(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fine-tune one arm from a base checkpoint")
    t.add_argument("--model", choices=FAMILIES, required=True)
    t.add_argument("--base", required=True, help="path/to/g_best (config.json beside it) or hf:<subfolder>")
    t.add_argument("--out", required=True, help="checkpoint directory for this arm")
    t.add_argument("--pesq_weight", type=float, default=0.2, help="0 = the control arm")
    t.add_argument("--backend", choices=("auto", "triton", "torch"), default="auto")
    t.add_argument("--lr", type=float, default=1e-4, help="fine-tuning learning rate")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--validation_interval", type=int, default=None)
    t.add_argument("--hf_cache_dir", default=None)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="score checkpoints on the VBD test split, with paired deltas")
    c.add_argument("--model", choices=FAMILIES, required=True)
    c.add_argument("checkpoints", nargs="+", help="name=path/to/g_best or name=hf:<subfolder>")
    c.add_argument("--metrics", default="pesq", help="pesq (default) | all | comma list")
    c.add_argument("--max_utterances", type=int, default=None)
    c.add_argument("--jobs", type=int, default=16)
    c.add_argument("--cpu", action="store_true", help="enhance on CPU even if CUDA is available")
    c.add_argument("--json", default=None, help="write means, deltas, per-utterance scores, provenance")
    c.add_argument("--md", default=None, help="write the markdown table")
    c.add_argument("--hf_cache_dir", default=None)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
