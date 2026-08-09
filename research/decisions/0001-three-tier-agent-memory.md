# 0001 — Split agent memory into rules / current / history

**Date:** 2026-08-09 **Status:** accepted

## Context

The repository was strong on evidence (streaming-vs-export parity tests,
per-utterance metric artifacts with provenance, hardware measurements, published
corrections of its own earlier claims) and weak on agent control. There was no
`AGENTS.md`, no CI, and the only persistent memory — `.planning/` — mixed stable
rules, current objectives and historical analysis in one place, with no owner and
no freshness check.

The result was drift with no alarm. `.planning/PROJECT.md` still presented the
April-2026 NSNet2 quantization milestone as the current goal, months after
ConvFSENet, LiSenNet, the metric suite and the STM32N6 deployment landed.
`.planning/codebase/TESTING.md` asserted there was no `tests/` directory while
`tests/` held 28 files. Nothing in either document was wrong when written;
nothing forced an update afterwards.

An agent cannot distinguish "true then" from "true now" in an undated document
that reads like a handover. Long context does not fix this — burying the current
truth in a large historical document makes it *less* likely to be used, not more.

## Decision

Three tiers, physically separated, with different mutability:

- **rules** — `AGENTS.md` (short, stable) plus `.claude/skills/` for procedures
  loaded only when relevant;
- **current** — `research/NOW.md`, `research/CLAIMS.yaml`,
  `research/hypotheses/`; rewritten freely, must be true today;
- **history** — `research/experiments/`, `research/decisions/`,
  `research/FAILURES.md`, `RESULTS_*.md`, `.planning/`; append-only, corrected
  by annotation rather than by edit.

`AGENTS.md` is capped (warning above 200 lines) so it keeps being read. Detail
goes into skills. `.planning/` is frozen and banner-labelled, not deleted.

Mechanical invariants are enforced by `tools/research_lint.py` in CI rather than
by intention: card schema, no selection on the test split, evidence paths that
resolve, and a staleness check on `NOW.md`.

## Consequences

- Every task now ends with a `NOW.md` update. That is friction, and it is the
  point — the previous cost of not doing it was invisible until it was months deep.
- Existing results keep their `evidence_gap` markers until manifests are
  backfilled. The gap is now visible instead of implied.
- `.planning/` remains tracked (the `.gitignore` entry postdates the files), so
  the banners travel to every clone. New files there need `git add -f`.

## Alternatives considered

- **One larger handover document.** Rejected: it is what already failed, and
  relevant content buried in a long context is retrieved less reliably.
- **Delete `.planning/`.** Rejected: the Key Decisions table and the pitfall
  write-ups are the most valuable part of the record, and deleting history to
  make the present tidy is the failure mode `AGENTS.md` rule 5 exists to prevent.
