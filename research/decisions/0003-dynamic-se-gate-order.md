# 0003 — Gate order for the dynamic speech-enhancement track

**Date:** 2026-08-09 **Status:** accepted

## Context

Dynamic / budget-conditioned capacity is the next research direction. It is also
the direction most exposed to this repo's two hardest-won lessons:

- **quality saturates** — ~10× parameters across three structure families spans
  only ~2.81–2.88 PESQ, non-monotonically (`CLAIMS.yaml#capacity-saturates`). If
  capacity barely buys quality on average, there may be no per-chunk headroom to
  route toward at all;
- **cost is not predicted by MACs** — the 2.78 M dense NSNet2 is memory-bound at
  RTF 1.43 on STM32N6 while a 1.10 M structured variant hits 0.13
  (`CLAIMS.yaml#cost-not-predicted-by-parameters`). A routed model that saves
  MACs may save no energy at all.

The tempting order is the fast one: train a router, see a MAC reduction, claim
efficiency. That order produces a result that cannot be defended against the two
questions a reviewer will actually ask — "versus a compute-matched static?" and
"measured where?".

## Decision

Four gates, in order. No stage starts before the previous one has produced a
recorded result.

1. **Oracle ceiling.** An oracle with reference access picks per chunk among
   already-trained statics. If it does not beat the best static at matched
   average cost, the direction is closed. `research/hypotheses/dynse-oracle-001.yaml`.
2. **Compute-matched statics.** Build the matched-cost static pool that any
   dynamic result will be compared against — *before* the dynamic model exists,
   so the comparison cannot be chosen after seeing the result.
3. **Router.** Train the causal router. Its ceiling is gate 1's oracle; a router
   that appears to exceed it means the evaluation is leaking.
4. **Hardware truth.** Compiled-target measurement on STM32N6, with router
   overhead and per-chunk reconfiguration included, reporting energy and P99
   latency against the 16 ms hop deadline. `research/hypotheses/dynse-router-002.yaml`.

## Consequences

- Most of the risk is retired in gate 1, at ~6 GPU-hours, before any router
  design work.
- The likely outcome of gate 1 is refutation. That is a publishable negative
  result about dynamic capacity in the saturated regime, and it goes in
  `research/FAILURES.md` either way.
- Agents may not skip a gate, including when a later gate looks more interesting.
  `depends_on` in the cards encodes this and `tools/research_lint.py` checks it.
