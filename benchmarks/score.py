"""Stage 2: enhanced audio on disk -> metric JSON.

    python -m benchmarks.score                     # every condition not yet scored
    python -m benchmarks.score --metrics pesq,dnsmos
    python -m benchmarks.score --force             # re-score even if JSON exists

Writes one JSON per (model, condition) to ``<out>/scores/<slug>__<cond>.json``,
holding both the means and the per-utterance values (so a later question like
"where does int8 actually lose?" is answerable without re-running anything).

Resumable by design: an existing JSON is skipped unless ``--force``. A full sweep
is a few hours, and it should survive an interrupted session.

The clean reference is read from ``<out>/audio/clean``, i.e. the same float32
WAVs the enhanced conditions are compared against -- not re-decoded from the HF
dataset -- so every condition is scored against byte-identical references.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common.quality import QualitySuite, backend_versions, resolve_metrics


def _load_dir(d: Path, ids: list[str]) -> list[np.ndarray]:
    return [sf.read(str(d / f"{i}.wav"), dtype="float32")[0] for i in ids]


def _utt_ids(d: Path) -> list[str]:
    return sorted(p.stem for p in d.glob("*.wav"))


def discover(audio_root: Path) -> list[tuple[str, str, Path]]:
    """Find every (slug, condition, dir) under <out>/audio, baselines included."""
    found = []
    for d in sorted(audio_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in ("noisy", "clean"):
            # The floor and the ceiling. Clean is scored against itself, which is
            # not circular for the metrics that matter here: DNSMOS and NISQA are
            # no-reference, so clean's score is the corpus ceiling those two can
            # award -- and VoiceBank's "clean" is studio-clean, not perfect. Without
            # it an absolute DNSMOS of 3.2 is unreadable. (PESQ and scoreq_ref
            # trivially self-score at 4.64 and 0.0; they are there for orientation.)
            found.append((d.name, "-", d))
            continue
        for cond in sorted(c for c in d.iterdir() if c.is_dir()):
            found.append((d.name, cond.name, cond))
    return found


def main():
    p = argparse.ArgumentParser(description="Score enhanced audio with PESQ + DNSMOS + NISQA + SCOREQ.")
    p.add_argument("--out", default="benchmarks/out")
    p.add_argument("--metrics", default="all", help="all | none | pesq,dnsmos,nisqa,scoreq")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument("--slug", action="append", default=None)
    a = p.parse_args()

    out = Path(a.out)
    audio_root = out / "audio"
    clean_dir = audio_root / "clean"
    if not clean_dir.is_dir():
        raise SystemExit(f"no clean references at {clean_dir}; run `python -m benchmarks.enhance` first")

    scores_dir = out / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    metrics = resolve_metrics(a.metrics)
    if not metrics:
        raise SystemExit("--metrics resolved to nothing to do")

    ids = _utt_ids(clean_dir)
    refs = _load_dir(clean_dir, ids)
    print(f"{len(ids)} reference utterances; metrics={list(metrics)}")

    # One suite for the whole sweep: it holds the 378 MB SCOREQ session and the
    # clean-reference embedding cache, both of which are pure waste to rebuild
    # per condition.
    suite = QualitySuite(metrics=metrics, sr=16000, n_jobs=a.jobs)

    todo = discover(audio_root)
    if a.slug:
        todo = [t for t in todo if t[0] in a.slug]

    for i, (slug, cond, d) in enumerate(todo, 1):
        dst = scores_dir / f"{slug}__{cond}.json"
        if dst.exists() and not a.force:
            print(f"[{i}/{len(todo)}] skip {slug} {cond} (already scored)")
            continue

        have = _utt_ids(d)
        if have != ids:
            # A partial dump would silently produce a mean over a different
            # utterance set than every other row -- refuse rather than mislead.
            print(f"[{i}/{len(todo)}] SKIP {slug} {cond}: {len(have)} utts, expected {len(ids)}")
            continue

        t0 = time.perf_counter()
        per_utt = suite.score(_load_dir(d, ids), refs)
        means = QualitySuite.summarize(per_utt)
        fails = QualitySuite.failures(per_utt)
        el = time.perf_counter() - t0

        dst.write_text(json.dumps({
            "slug": slug,
            "condition": cond,
            "n_utts": len(ids),
            "metrics": list(metrics),
            "means": means,
            "failures": fails,
            "seconds": round(el, 1),
            "versions": backend_versions(),
            "per_utt": {k: [None if np.isnan(x) else round(float(x), 6) for x in v]
                        for k, v in per_utt.items()},
            "utt_ids": ids,
        }, indent=2))

        shown = " ".join(f"{k}={v:.3f}" for k, v in means.items()
                         if k in ("pesq", "dnsmos_ovrl", "nisqa_mos", "scoreq_nr", "scoreq_ref"))
        print(f"[{i}/{len(todo)}] {slug} {cond}: {shown}  ({el/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
