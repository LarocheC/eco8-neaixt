---
name: audit-evidence
description: Adversarially check a claim against its raw evidence. Use before publishing a result, updating CLAIMS.yaml, writing a model card, or accepting another agent's conclusion.
---

# Audit evidence

You are in the **auditor** role. You get the card, the diff and the raw
artifacts. You do **not** get, and must not ask for, the proposer's or
executor's narrative — a persuasive summary is exactly what you are auditing
against.

Your default posture is refusal. A claim survives only if you cannot break it.

## The checklist

Work through all of it. Record each item as pass / fail / not-applicable.

**Provenance**
1. Does every number trace to a committed artifact? A number that exists only in
   prose is not a result (`AGENTS.md` rule 14).
2. Does the manifest exist, and does its git SHA correspond to the code that
   produced the artifact? Was the tree dirty?
3. Were all runs in a comparison made with the same package versions, dataset
   revision and — for timings — the same box?

**Selection**
4. Did anything select on the test split? Architecture, checkpoint, threshold,
   calibration set, routing policy, early stopping, or "we tried a few and this
   one worked" (`AGENTS.md` rule 7).
5. Is the reported configuration the one the card pre-registered, or was it
   chosen after seeing results?

**Comparison**
6. Is there a compute-matched static baseline, matched on *measured* cost? A win
   over the largest reference model alone is not a win (`AGENTS.md` rule 6).
7. Did the baseline get the same training data, optimisation budget and tuning
   effort? Under-trained baselines are the most common silent confound.
8. Is the effect larger than run-to-run noise? In this repo, PESQ deltas below
   ~0.02 in the 2.8–2.9 band are not separable at one seed — see
   `CLAIMS.yaml#capacity-saturates`.

**Measurement validity**
9. Metric directions: PESQ / DNSMOS / NISQA / SCOREQ-MOS higher is better;
   SCOREQ full-reference **distance** lower is better. Check the sign of every
   delta.
10. Are per-utterance scores retained, and do the reported means recompute from
    them? Are failed evaluations counted, or silently dropped?
11. Is the evaluation subset large enough to support the claim, and is it the
    subset the card specified?

**Cost claims**
12. Is any latency or energy statement backed by a compiled-target measurement,
    or was it inferred from MACs or parameters? (`AGENTS.md` rule 8 — this repo
    has a counterexample on record.)
13. Is an "int8" model audited — operators, activations *and* weights? Check the
    graph, not the filename (`research/FAILURES.md#einsum-int8`).
14. Does a latency measurement include P99 where a deadline matters, or only a
    mean?

**Scope**
15. Does the statement claim more than the experiment covers — one dataset, one
    recipe, one target, one phase configuration?

## Output

For each surviving claim, the `CLAIMS.yaml` entry: `statement`, `kind`, `status`,
`evidence` paths, `counterevidence`, `scope`. Downgrade rather than delete:
`measured` → `compute-only` when only host numbers exist; → `pilot` when it is
one seed or a subset.

List every check that failed, with the specific file and line. If you found
nothing wrong, say which checks you actually ran — an audit that reports "looks
good" without naming its checks is not an audit.
