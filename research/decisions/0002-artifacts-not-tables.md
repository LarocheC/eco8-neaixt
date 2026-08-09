# 0002 — Results tables are generated from committed artifacts

**Date:** 2026-08-09 (ratifying a practice established by `benchmarks/`)
**Status:** accepted

## Context

`benchmarks/` already does the right thing: enhancement and scoring are decoupled
through a float32 audio cache, `benchmarks/score.py` writes per-utterance JSON,
and `benchmarks/report.py` generates the markdown from `summary.json` +
`per_utterance.json.gz`, both carrying a provenance block. Adding a metric costs
a re-score, not a re-inference, and any mean in `RESULTS_METRICS.md` can be
recomputed, paired-tested or chased to an outlier.

Everywhere else in the repo, numbers reach `RESULTS_*.md` by hand. The failure
mode this invites is not fabrication — it is silent staleness: a number that was
correct when typed and was never invalidated when its underlying run was
re-measured. The Einsum int8 correction is the concrete instance: nine int8 PESQ
values had to be re-measured and re-typed, and nothing structural would have
caught it if one had been missed.

## Decision

A number in a results file or a paper must be generated from a committed
artifact. Concretely:

- per-utterance scores are retained, not just aggregates;
- the artifact carries provenance (dataset, git commit, versions of every package
  that can move the score);
- the table is emitted by a script that reads that artifact;
- hand-copying a number into a table is prohibited (`AGENTS.md` rule 14).

Claims whose evidence is currently prose carry `evidence_gap` in
`research/CLAIMS.yaml` until backfilled. That marker is the backlog.

## Consequences

- New measurement paths (board sessions, profiler runs, oracle sweeps) must emit
  a machine-readable artifact before their numbers can be published.
- `benchmarks/report.py` is the reference implementation to copy.
- Backfilling the NSNet2 and STM32N6 tables is real work and is listed in
  `research/NOW.md` rather than pretended away.
