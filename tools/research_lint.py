#!/usr/bin/env python3
"""Mechanical checks on the agent-control layer.

This is deliberately dumb. It does not judge science; it enforces the
invariants that are cheap to check and expensive to violate:

  * hypothesis cards have a falsifier before they are allowed to consume budget
  * nothing selects on the test split
  * every claim points at evidence a reader can actually open
  * experiment manifests exist for the runs that claims cite
  * the "current" tier has not silently gone stale
  * the frozen archive is still labelled as frozen

Deterministic scripts outrank agent agreement -- that is the whole reason this
file exists. Run it before saying a task is done; CI runs it on every push.

    uv run python tools/research_lint.py [--strict]

--strict promotes warnings to errors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("research_lint: needs pyyaml (`uv sync` provides it transitively)", file=sys.stderr)
    raise SystemExit(1)

REPO = Path(__file__).resolve().parent.parent
RESEARCH = REPO / "research"
HYPOTHESES = RESEARCH / "hypotheses"
EXPERIMENTS = RESEARCH / "experiments"
CLAIMS = RESEARCH / "CLAIMS.yaml"
NOW = RESEARCH / "NOW.md"

CARD_STATUSES = ["proposed", "critiqued", "accepted", "running", "complete",
                 "refuted", "abandoned"]
# Fields required once a card reaches this status or later.
REQUIRED_FROM = {
    "proposed": ["id", "question", "status", "created", "mechanism", "prediction"],
    "accepted": ["falsify_if", "accept_if", "baselines", "controls",
                 "primary_metric", "split", "seeds", "budget", "critique"],
    "running": ["experiments"],
    "complete": ["outcome"],
}
CLAIM_KINDS = ["measured", "derived", "cited", "hypothesis"]
CLAIM_STATUSES = ["measured", "compute-only", "pilot", "corrected", "unverified", "refuted"]

ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NOW_STALE_WARN_DAYS = 30
NOW_STALE_FAIL_DAYS = 90
AGENTS_MAX_LINES = 200

# Historical documents must say so, so no agent mistakes them for current state.
#
# NOTE: `.gitignore` lists `.planning/`, but the files below were committed
# before that rule existed, so they are still tracked and still travel to every
# clone -- including the ones an agent starts from. New files added there need
# `git add -f`. `--fix-archive` inserts a missing banner in place.
ARCHIVE_BANNER = "> **ARCHIVE"
ARCHIVE_FILES = [
    ".planning/PROJECT.md",
    ".planning/codebase/ARCHITECTURE.md",
    ".planning/codebase/CONCERNS.md",
    ".planning/codebase/CONVENTIONS.md",
    ".planning/codebase/INTEGRATIONS.md",
    ".planning/codebase/STACK.md",
    ".planning/codebase/STRUCTURE.md",
    ".planning/codebase/TESTING.md",
]
PROJECT_BANNER = """> **ARCHIVE — frozen 2026-04-27. Not current state.**
> This describes the NSNet2 int8-quantization milestone as it was scoped in
> April 2026, when the repo was still `sparse-nsnet2`. Since then ConvFSENet,
> LiSenNet, the perceptual-metric suite and the STM32N6 deployment have landed,
> and the "no existing test infrastructure" note below is false — `tests/` now
> holds a real pytest suite. Kept because the reasoning and the Key Decisions
> table are still useful history.
>
> **Current state lives in [`research/NOW.md`](../research/NOW.md); rules live in
> [`AGENTS.md`](../AGENTS.md).**
"""
CODEBASE_BANNER = """> **ARCHIVE — frozen 2026-04-27. Not current state.**
> A snapshot analysis of the codebase as it was in April 2026, under its former
> `sparse-nsnet2` layout (root-level `train.py`, `models/`, no `tests/`). The
> current layout is per-family packages (`nsnet2/`, `convfsenet/`, `lisennet/`,
> `common/`, `benchmarks/`) with a pytest suite. **Where this file and the code
> disagree, the code wins.** Kept for the rationale, not the facts.
>
> Current state: [`research/NOW.md`](../../research/NOW.md). Rules:
> [`AGENTS.md`](../../AGENTS.md).
"""

errors: list[str] = []
warnings: list[str] = []


def err(where: str, msg: str) -> None:
    errors.append(f"{where}: {msg}")


def warn(where: str, msg: str) -> None:
    warnings.append(f"{where}: {msg}")


def status_index(status: str) -> int:
    return CARD_STATUSES.index(status) if status in CARD_STATUSES else -1


def required_fields(status: str) -> list[str]:
    """Every requirement whose gate status is at or before `status`."""
    fields: list[str] = []
    for gate, names in REQUIRED_FROM.items():
        if status_index(status) >= status_index(gate) and status not in {"refuted", "abandoned"}:
            fields += names
        elif gate == "proposed":
            fields += names  # always required, even for abandoned cards
    return sorted(set(fields))


def check_cards() -> set[str]:
    ids: set[str] = set()
    if not HYPOTHESES.exists():
        err("research/hypotheses", "missing")
        return ids
    cards = {}
    for path in sorted(HYPOTHESES.glob("*.yaml")):
        rel = path.relative_to(REPO)
        if path.stem == "TEMPLATE":
            continue
        try:
            card = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            err(str(rel), f"unparseable YAML: {exc}")
            continue
        if not isinstance(card, dict):
            err(str(rel), "not a mapping")
            continue
        cards[path.stem] = (rel, card)
        ids.add(path.stem)

    for stem, (rel, card) in cards.items():
        cid = card.get("id")
        if cid != stem:
            err(str(rel), f"id {cid!r} does not match filename stem {stem!r}")
        if cid and not ID_RE.match(str(cid)):
            err(str(rel), f"id {cid!r} is not lowercase-kebab")

        status = card.get("status")
        if status not in CARD_STATUSES:
            err(str(rel), f"status {status!r} not one of {CARD_STATUSES}")
            continue

        for field in required_fields(status):
            if card.get(field) in (None, "", [], {}):
                err(str(rel), f"status={status} requires a non-empty {field!r}")

        created = card.get("created")
        if created is not None and not isinstance(created, (dt.date, dt.datetime)):
            err(str(rel), f"created {created!r} is not a YYYY-MM-DD date")

        # The rule that gets skipped first, so it is checked hardest.
        if status == "accepted" or status_index(status) > status_index("accepted"):
            if status not in {"refuted", "abandoned"}:
                fal = str(card.get("falsify_if") or "")
                acc = str(card.get("accept_if") or "")
                if fal and acc and _is_trivial_negation(fal, acc):
                    err(str(rel),
                        "falsify_if is just the negation of accept_if -- name the "
                        "rival explanation (SCHEMA.md)")
                baselines = card.get("baselines") or []
                if isinstance(baselines, list) and not any(
                    "match" in json.dumps(b).lower() for b in baselines
                ):
                    warn(str(rel),
                         "no compute-matched baseline listed (AGENTS.md rule 6)")

        split = card.get("split")
        if isinstance(split, str) and split.strip().lower() in {"test", "vbd_test", "test_split"}:
            err(str(rel), "split: test -- the test split selects nothing (AGENTS.md rule 7)")

        seeds = card.get("seeds")
        if isinstance(seeds, list) and 0 < len(seeds) < 3 and status_index(status) >= status_index("accepted"):
            warn(str(rel), f"{len(seeds)} seed(s); quality deltas in this repo's "
                           "saturation band need >= 3")

        for dep in card.get("depends_on") or []:
            if dep not in cards:
                err(str(rel), f"depends_on {dep!r} is not a card in research/hypotheses/")

        for exp in card.get("experiments") or []:
            if not (EXPERIMENTS / str(exp) / "manifest.json").exists():
                err(str(rel), f"experiments lists {exp!r} but "
                              f"research/experiments/{exp}/manifest.json is missing")
    return ids


def _is_trivial_negation(falsify: str, accept: str) -> bool:
    """Heuristic: a falsifier that only flips a comparator is not a falsifier."""
    def norm(s: str) -> str:
        s = s.lower()
        for a, b in [(">=", "<"), ("<=", ">"), ("better", "worse"), ("worse", "better"),
                     ("higher", "lower"), ("lower", "higher"), ("non-inferior", "inferior"),
                     ("not ", ""), ("no ", "")]:
            s = s.replace(a, b)
        return re.sub(r"[^a-z0-9]+", " ", s).strip()
    return norm(falsify) == norm(accept)


def check_claims(card_ids: set[str]) -> None:
    if not CLAIMS.exists():
        err("research/CLAIMS.yaml", "missing")
        return
    try:
        doc = yaml.safe_load(CLAIMS.read_text())
    except yaml.YAMLError as exc:
        err("research/CLAIMS.yaml", f"unparseable YAML: {exc}")
        return
    claims = (doc or {}).get("claims")
    if not isinstance(claims, list):
        err("research/CLAIMS.yaml", "top-level `claims:` list missing")
        return

    seen: set[str] = set()
    for i, claim in enumerate(claims):
        where = f"research/CLAIMS.yaml[{i}]"
        if not isinstance(claim, dict):
            err(where, "not a mapping")
            continue
        cid = claim.get("id")
        where = f"research/CLAIMS.yaml#{cid}"
        for field in ("id", "statement", "kind", "status", "evidence",
                      "counterevidence", "scope"):
            if field not in claim:
                err(where, f"missing required field {field!r}")
        if cid in seen:
            err(where, "duplicate claim id")
        seen.add(cid)
        if claim.get("kind") not in CLAIM_KINDS:
            err(where, f"kind {claim.get('kind')!r} not one of {CLAIM_KINDS}")
        status = claim.get("status")
        if status not in CLAIM_STATUSES:
            err(where, f"status {status!r} not one of {CLAIM_STATUSES}")

        evidence = claim.get("evidence")
        if not isinstance(evidence, list):
            err(where, "evidence must be a list")
            continue
        if not evidence and status != "unverified":
            err(where, f"status={status} with no evidence (only `unverified` may be empty)")
        for ref in evidence:
            _check_reference(where, str(ref), card_ids)

        if claim.get("kind") == "measured" and status == "compute-only":
            err(where, "kind=measured with status=compute-only -- a compute-only "
                       "result does not license a measured claim (AGENTS.md rule 8)")


def _check_reference(where: str, ref: str, card_ids: set[str]) -> None:
    """Evidence entries are repo-relative paths, optionally with a #anchor,
    or intra-file references of the form research/CLAIMS.yaml#claim-id."""
    path_part, _, anchor = ref.partition("#")
    path_part = path_part.strip()
    if not path_part:
        err(where, f"evidence {ref!r} has no path")
        return
    target = REPO / path_part
    if not target.exists():
        err(where, f"evidence path does not exist: {path_part}")
        return
    if path_part == "research/CLAIMS.yaml" and anchor:
        return  # cross-claim reference; ids are checked for duplicates already
    if path_part.startswith("research/hypotheses/"):
        stem = Path(path_part).stem
        if stem not in card_ids:
            err(where, f"evidence references unknown card {stem!r}")


def check_manifests() -> None:
    if not EXPERIMENTS.exists():
        return
    for manifest_path in sorted(EXPERIMENTS.glob("*/manifest.json")):
        rel = manifest_path.relative_to(REPO)
        try:
            m = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            err(str(rel), f"unparseable JSON: {exc}")
            continue
        if m.get("schema") != "eco8-experiment-manifest/1":
            err(str(rel), "wrong or missing `schema` -- generate it with tools/run_manifest.py")
        if m.get("id") != manifest_path.parent.name:
            err(str(rel), f"id {m.get('id')!r} does not match directory name")
        if m.get("git", {}).get("dirty") and not m.get("id", "").startswith("EXAMPLE"):
            warn(str(rel), "recorded from a dirty tree -- the diff sha is stored, "
                           "but prefer committing first")


def check_freshness() -> None:
    if not NOW.exists():
        err("research/NOW.md", "missing -- this is the file agents read first")
        return
    text = NOW.read_text()
    match = re.search(r"\*\*Last updated:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        err("research/NOW.md", "no `**Last updated:** YYYY-MM-DD` line")
        return
    age = (dt.date.today() - dt.date.fromisoformat(match.group(1))).days
    if age > NOW_STALE_FAIL_DAYS:
        err("research/NOW.md", f"{age} days stale (> {NOW_STALE_FAIL_DAYS}). "
                               "Stale current-state is worse than none -- this is "
                               "exactly how .planning/ drifted.")
    elif age > NOW_STALE_WARN_DAYS:
        warn("research/NOW.md", f"{age} days stale (> {NOW_STALE_WARN_DAYS})")


def check_archive_labelled(fix: bool = False) -> None:
    for rel in ARCHIVE_FILES:
        path = REPO / rel
        if not path.exists():
            continue
        lines = path.read_text().splitlines(keepends=True)
        if ARCHIVE_BANNER in "".join(lines[:12]):
            continue
        if not fix:
            err(rel, "historical document without an ARCHIVE banner in its first "
                     "12 lines -- an agent will read it as current state. "
                     "Run `python tools/research_lint.py --fix-archive`.")
            continue
        if not lines or not lines[0].startswith("# "):
            err(rel, "cannot insert banner: no `# ` heading on line 1")
            continue
        banner = PROJECT_BANNER if rel.endswith("PROJECT.md") else CODEBASE_BANNER
        path.write_text("".join([lines[0], "\n", banner] + lines[1:]))
        print(f"fixed {rel}: ARCHIVE banner inserted")


def check_agents_md() -> None:
    path = REPO / "AGENTS.md"
    if not path.exists():
        err("AGENTS.md", "missing")
        return
    n = len(path.read_text().splitlines())
    if n > AGENTS_MAX_LINES:
        warn("AGENTS.md", f"{n} lines (> {AGENTS_MAX_LINES}). Rules that keep growing "
                          "stop being read -- move procedure into .claude/skills/.")
    if not (REPO / "CLAUDE.md").exists():
        warn("CLAUDE.md", "missing pointer file")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--fix-archive", action="store_true",
                    help="insert the ARCHIVE banner into any unlabelled .planning/ "
                         "document (that directory is gitignored, so the banners "
                         "have to be reapplied per machine)")
    a = ap.parse_args(argv)

    card_ids = check_cards()
    check_claims(card_ids)
    check_manifests()
    check_freshness()
    check_archive_labelled(fix=a.fix_archive)
    check_agents_md()

    for w in warnings:
        print(f"warn  {w}")
    for e in errors:
        print(f"ERROR {e}")

    if errors or (a.strict and warnings):
        print(f"\nresearch_lint: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"research_lint: ok ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
