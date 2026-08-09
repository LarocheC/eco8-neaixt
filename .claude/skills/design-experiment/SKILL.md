---
name: design-experiment
description: Turn a research idea into a falsifiable experiment card before any compute is spent. Use when proposing a new architecture, training run, sweep, quantization scheme or deployment experiment in this repo.
---

# Design an experiment

You are in the **proposer** role. Your output is a card, not code and not a run.

## 1. Check it is not already answered or already dead

Read, in this order — stop early if any of them settles it:

1. `research/FAILURES.md` — has this been tried? The capacity-scaling and
   npu-blocker entries close whole classes of proposal.
2. `research/CLAIMS.yaml` — is the premise already a claim? Note its `status`
   and `scope`; a `compute-only` claim does not support a cost premise.
3. `research/hypotheses/` — is there an open card, or one this one depends on?
4. The relevant `RESULTS_*.md`.

If a prior failure kills the idea, say so and stop. That is a successful outcome
of this skill.

## 2. Write the mechanism first

Before any numbers: *why should this work, in terms of the model or the
hardware?* "It might help" is not a mechanism. "Easy chunks need fewer active
channels because the mask is near-unity in silence" is.

If you cannot state a mechanism, you have a hyperparameter search, not an
experiment. Say that plainly.

## 3. Write the falsifier before the prediction feels convincing

The falsifier must name the **rival explanation**, not just the negation of your
prediction. The rival is almost always one of:

- a **compute-matched static** reaches the same quality at the same measured cost;
- the effect is inside the metric's run-to-run noise (in this repo, PESQ deltas
  below ~0.02 in the 2.8–2.9 band are not separable at one seed);
- the gain came from extra optimisation budget or extra data, not the mechanism;
- the MAC saving does not survive contact with the compiled target.

A card whose falsifier is "the number was lower" will be rejected by
`tools/research_lint.py`.

## 4. Fill the card

Copy `research/hypotheses/TEMPLATE.yaml`. Field reference: `research/SCHEMA.md`.

Non-negotiables:

- `split` is never `test`. Anything that selects uses a train holdout.
- `baselines` includes at least one compute-matched entry for any cost claim.
- `controls` pins training data, optimisation budget, calibration set, and — for
  any timing — the box. Timings from different machines are not comparable.
- `seeds`: at least 3 for a quality comparison.
- `budget` is an approval gate, in GPU-hours or board sessions.
- `primary_metric` is exactly one. Multiple primaries is how a negative result
  becomes a positive one.

## 5. Hand it to a sceptic, in a fresh context

Do not critique your own card in the same context that wrote it. Start a fresh
one and give it **the card only** — not your reasoning, not your enthusiasm:

> Here is an experiment card. Find confounders, prior art that already answers
> it, and simpler explanations that predict the same result. Argue for refusing
> to run it.

Write the result into `critique`. Then set `status: critiqued`.

## 6. Stop

`accepted` requires the human. Any card that consumes GPU hours or a board
session is an approval gate — present the card and the budget, and wait.
