"""Stage 3: metric JSON -> markdown tables.

    python -m benchmarks.report                    # to stdout
    python -m benchmarks.report --md RESULTS_METRICS.md

Renders one table per family plus the FP32->int8 quantization deltas. Deltas are
oriented by ``common.quality.HIGHER_IS_BETTER``, because scoreq_ref is a distance
and would otherwise be reported backwards.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

from benchmarks.published import MODELS
from common.quality import HIGHER_IS_BETTER, backend_versions

# The columns worth a headline table. The remaining NISQA sub-dimensions
# (noisiness / discontinuity / coloration / loudness) and DNSMOS's P.808 stay in
# the JSON -- they diagnose *why* a model scores as it does, but they make a
# 19-row table unreadable.
HEADLINE = ["pesq", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "nisqa_mos",
            "scoreq_nr", "scoreq_ref"]

PRETTY = {
    "pesq": "PESQ",
    "dnsmos_ovrl": "DNSMOS OVRL",
    "dnsmos_sig": "DNSMOS SIG",
    "dnsmos_bak": "DNSMOS BAK",
    "dnsmos_p808": "DNSMOS P.808",
    "nisqa_mos": "NISQA",
    "nisqa_noi": "NISQA noi",
    "nisqa_dis": "NISQA dis",
    "nisqa_col": "NISQA col",
    "nisqa_loud": "NISQA loud",
    "scoreq_nr": "SCOREQ-nr",
    "scoreq_ref": "SCOREQ-ref",
}


def load(scores_dir: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for p in sorted(Path(scores_dir).glob("*.json")):
        d = json.loads(p.read_text())
        out[(d["slug"], d["condition"])] = d
    return out


def _fmt(col: str, v) -> str:
    if v is None:
        return "–"
    return f"{v:.3f}"


def _row(label: str, means: dict, cols: list[str]) -> str:
    return "| " + " | ".join([label] + [_fmt(c, means.get(c)) for c in cols]) + " |"


def _table(rows: list[str], cols: list[str], head: str) -> list[str]:
    header = "| " + " | ".join([head] + [PRETTY.get(c, c) for c in cols]) + " |"
    sep = "| " + " | ".join(["---"] + ["---:"] * len(cols)) + " |"
    return [header, sep, *rows]


def render(scores: dict, cols=None) -> str:
    cols = cols or HEADLINE
    L: list[str] = []

    n_utts = next((d["n_utts"] for d in scores.values()), 0)
    L += [
        "## Perceptual metrics on the published models",
        "",
        f"VoiceBank-DEMAND test split, n={n_utts} utterances. Every row is the same",
        "audio path the model actually deploys, scored through one metric harness",
        "(`common/quality.py`).",
        "",
        "**SCOREQ-ref is a distance, so lower is better.** Every other column is",
        "higher-is-better. DNSMOS and NISQA are no-reference: they never see the",
        "clean signal, which is why the `clean` row does not saturate them and is",
        "the ceiling those two columns can award on this corpus.",
        "",
    ]

    base = []
    for tag in ("noisy", "clean"):
        d = scores.get((tag, "-"))
        if d:
            base.append(_row(f"**{tag}**", d["means"], cols))
    if base:
        L += ["### Baselines", ""] + _table(base, cols, "signal") + [""]

    for family, title in (("nsnet2", "NSNet2"), ("convfsenet", "ConvFSENet"),
                          ("lisennet", "LiSenNet")):
        fam = [m for m in MODELS if m.family == family]
        rows = []
        for m in fam:
            for cond in m.conditions:
                d = scores.get((m.slug, cond))
                if d:
                    rows.append(_row(f"`{m.name}` {cond}", d["means"], cols))
        if rows:
            L += [f"### {title}", ""] + _table(rows, cols, "model / condition") + [""]

    # FP32 -> int8: the question the repo actually cares about. Signed so that a
    # positive delta always means "int8 is worse", regardless of metric polarity.
    dl = []
    for m in MODELS:
        a, b = scores.get((m.slug, "fp32")), scores.get((m.slug, "int8"))
        if not (a and b):
            continue
        vals = []
        for c in cols:
            x, y = a["means"].get(c), b["means"].get(c)
            if x is None or y is None:
                vals.append("–")
            else:
                loss = (x - y) if HIGHER_IS_BETTER[c] else (y - x)
                vals.append(f"{loss:+.3f}")
        dl.append("| " + " | ".join([f"`{m.name}`"] + vals) + " |")
    if dl:
        L += [
            "### Quantization cost (FP32 → int8)", "",
            "Positive = int8 is worse, on every column (sign-corrected for SCOREQ-ref).",
            "",
        ] + _table(dl, cols, "model") + [""]

    return "\n".join(L)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _provenance(scores: dict) -> dict:
    # Prefer the versions captured at scoring time; fall back to the current env
    # for score files written before provenance was recorded.
    recorded = next((d["versions"] for d in scores.values() if d.get("versions")), None)
    return {
        "dataset": "JacobLinCool/VoiceBank-DEMAND-16k",
        "split": "test",
        "sampling_rate": 16000,
        "git_commit": _git_commit(),
        "versions": recorded or backend_versions(),
        "versions_source": "scoring run" if recorded else "reporting env (not recorded at score time)",
    }


def summary(scores: dict) -> dict:
    """Means + failures per condition, with provenance. The compact artefact."""
    return {
        "provenance": _provenance(scores),
        "conditions": {
            f"{slug}/{cond}": {"n_utts": d["n_utts"], "means": d["means"],
                               "failures": d["failures"]}
            for (slug, cond), d in sorted(scores.items())
        },
    }


def audit_bundle(scores: dict) -> dict:
    """Every per-utterance value, for auditing: recompute a mean, hunt an outlier,
    run a paired test between two conditions.

    The utterance IDs are stored once at the top rather than repeated in all 44
    condition files, and every column is aligned to that order. `null` marks an
    utterance a metric failed on (there are none in the committed run).
    """
    ids = next(iter(scores.values()))["utt_ids"]
    for (slug, cond), d in scores.items():
        if d["utt_ids"] != ids:
            raise ValueError(
                f"{slug}/{cond} was scored over a different utterance set; "
                "the shared utt_ids index would silently misalign it"
            )
    return {
        "provenance": _provenance(scores),
        "n_utts": len(ids),
        "utt_ids": ids,
        "conditions": {
            f"{slug}/{cond}": {
                "means": d["means"],
                "failures": d["failures"],
                "per_utt": d["per_utt"],
            }
            for (slug, cond), d in sorted(scores.items())
        },
    }


def main():
    p = argparse.ArgumentParser(description="Render benchmark JSON as markdown.")
    p.add_argument("--out", default="benchmarks/out")
    p.add_argument("--md", default=None, help="Write markdown to this file instead of stdout")
    p.add_argument("--json", default=None, help="Write the means-only summary to this JSON path")
    p.add_argument("--audit", default=None,
                   help="Write every per-utterance value to this path; "
                        "gzipped automatically if it ends in .gz")
    p.add_argument("--all_columns", action="store_true",
                   help="Include the NISQA sub-dimensions and DNSMOS P.808")
    a = p.parse_args()

    scores = load(Path(a.out) / "scores")
    if not scores:
        raise SystemExit(f"no scores under {Path(a.out) / 'scores'}; run `python -m benchmarks.score`")

    cols = list(PRETTY) if a.all_columns else HEADLINE
    md = render(scores, cols)
    if a.md:
        Path(a.md).write_text(md + "\n")
        print(f"wrote {a.md} ({len(scores)} scored conditions)")
    else:
        print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(summary(scores), indent=2) + "\n")
        print(f"wrote {a.json}")

    if a.audit:
        blob = json.dumps(audit_bundle(scores), indent=1).encode()
        dst = Path(a.audit)
        if dst.suffix == ".gz":
            dst.write_bytes(gzip.compress(blob, 9))
        else:
            dst.write_bytes(blob)
        print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB, "
              f"{len(scores)} conditions x {next(iter(scores.values()))['n_utts']} utts)")


if __name__ == "__main__":
    main()
