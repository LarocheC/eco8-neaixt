# AGENTS.md — rules for any agent working in this repository

This file is **stable**. It holds rules that outlive any milestone. It is not a
handover, not a status report, and not a place to record results.

Three tiers, and never mix them:

| tier        | lives in                | mutability                                  |
| ----------- | ----------------------- | ------------------------------------------- |
| **rules**   | this file, `.claude/skills/` | rarely changes; changing it is a decision record |
| **current** | `research/NOW.md`, `research/CLAIMS.yaml`, `research/hypotheses/` | rewritten freely; must be true *now* |
| **history** | `research/experiments/`, `research/decisions/`, `research/FAILURES.md`, `RESULTS_*.md`, `.planning/` | append-only; never rewrite to match a new result |

Read `research/NOW.md` first, every session. It is the only file that says what
we are currently doing. `.planning/` is a **frozen 2026-04 archive** — do not
treat it as current state (see `.planning/README.md`).

## Detailed procedures live in skills

Load only what the task needs — do not read all of these up front:

| task                                        | skill                          |
| ------------------------------------------- | ------------------------------ |
| propose a new experiment                    | `.claude/skills/design-experiment/` |
| execute an approved experiment card         | `.claude/skills/run-experiment/`    |
| check a claim against its evidence          | `.claude/skills/audit-evidence/`    |
| compile / measure on STM32N6                | `.claude/skills/deploy-stm32n6/`    |
| check a draft's numbers against artifacts   | `.claude/skills/paper-audit/`       |
| survey prior art                            | `.claude/skills/literature-review/` |

## Epistemic rules

1. **Label every factual statement** in prose, commits and PRs as one of:
   `[measured]` (a number this repo produced, with the artifact path),
   `[derived]` (computed from measured numbers — say from what),
   `[cited]` (from a paper — give the reference), `[hypothesis]` (a prediction
   with a falsifier), `[speculative]` (a guess; must not enter a results file).
2. **State the mechanism and its falsifier before implementing.** Write down
   what result would prove the idea wrong, before you can see any result.
3. **Explanations invented after seeing a result are hypotheses, not findings.**
   They go in `research/hypotheses/`, not in a `RESULTS_*.md` conclusion.
4. **Report negative results as negative.** Do not go looking for a metric,
   subset or split that turns a loss into a win. A negative result is a
   deliverable — record it in `research/FAILURES.md`.
5. **Correct, do not overwrite.** When an earlier claim turns out to be wrong,
   add a correction notice next to it (see the two ⚠️ blocks at the top of
   `RESULTS_NSNET2.md`). Never silently edit a superseded number away.

## Evidence rules

6. **Compare against compute-matched static baselines**, not only against the
   largest reference model. A win over an unmatched baseline is not a win.
7. **Never touch the test split to choose anything.** The VoiceBank-DEMAND
   824-utterance test split selects nothing: not architectures, not
   checkpoints, not thresholds, not calibration sets, not routing policies.
   Use the train split (or a held-out slice of it) for all selection.
8. **Do not infer latency or energy from MACs or parameter counts.** This repo
   has a counterexample on record: the 2.78 M-param dense NSNet2 is *memory*
   bound on STM32N6 (2.70 MB weights overflow on-chip RAM → RTF 1.43), while a
   1.10 M-param structured variant hits RTF 0.13. Cost claims require a
   compiled-target measurement.
9. **Do not call a graph "int8" until you have audited it.** Weights,
   activations *and* every operator. Precedent: `quantize_static` silently
   skipped structured weights because onnxruntime ships no QDQ handler for
   `Einsum`, and months of "int8" PESQ numbers were measured on FP32 weights.
   `nsnet2/quant.py` now fails if any `Einsum` operand is a raw FLOAT
   initializer — keep that audit, and add the equivalent for any new export path.
10. **Hardware claims need end-to-end compiled-target measurements**, on the
    real toolchain (`stedgeai` 4.0.1 → Neural-ART), reporting ms/frame and RTF,
    and P99 as well as median where a deadline matters. Host-ONNX RTF is not a
    hardware claim.
11. **Timings from different machines or load conditions are not comparable.**
    Say which box a table was measured on, or omit the column.
12. **Preserve per-utterance outputs and failures**, not just aggregate means —
    `benchmarks/per_utterance.json.gz` is the pattern. Means without the
    underlying scores cannot be re-checked, paired-tested, or chased to an outlier.
13. **Metric directions are not uniform.** PESQ / DNSMOS / NISQA / SCOREQ-MOS:
    higher is better. SCOREQ full-reference *distance*: lower is better.
14. **Never hand-copy a number into a table.** Tables in `RESULTS_*.md` and in
    any paper are generated from committed artifacts (`benchmarks/report.py` is
    the reference implementation). A number with no artifact path is not a result.

## Naming rules

15. `blockdiag` is a **single block-diagonal factor**, no cross-block mixing.
    `monarch` is the **genuine two-factor** construction (block-diagonal ×
    permutation × block-diagonal). They are different models with different
    parameter counts. The repo mislabeled one as the other for months; do not
    reintroduce that. Check `linear.kind` / `gru.kind` in the config before
    describing a run.
16. Streaming and offline paths must stay numerically equivalent. The parity
    gate is `max_abs_err < 1e-5`. **Do not loosen a tolerance to make a test
    pass** — `tests/README.md` has a diagnostic bucket table; use it.

## Autonomy

Act freely, without asking, on: implementation, refactoring, tests, exports,
documentation, and any local run under a few minutes.

Ask first, and get an explicit go-ahead, before:

- any training run or sweep (GPU hours), or any board / cloud-farm measurement;
- publishing to Hugging Face, or pushing to `main`;
- a novelty claim ("first to…", "novel") — those are human-gated;
- adopting a new framework or dependency (Hydra, DVC, a vector store, an agent
  framework). Additive scripts and config flags, not framework rewrites.

Confidence is not licence: outside speech enhancement, DSP and embedded
deployment, prefer a canonical reference implementation or a deterministic
oracle over your own reasoning, and say which one you used.

## Definition of done

Before you say a task is finished:

```bash
uv run pytest                      # fast suite; -m slow needs network + weights
uv run python tools/research_lint.py   # cards, claims, doc freshness
```

- `research/NOW.md` reflects reality (update the objective and next actions).
- Anything that failed is in `research/FAILURES.md`, with why.
- Any claim you added or changed is in `research/CLAIMS.yaml` with its evidence.
- `git status` is clean; no stray checkpoints, `cp_*/`, WAVs or notebooks-with-output.
- Commit messages state what was **measured** vs what is **hypothesis**.

Repo-specific stack notes (uv, module-style entry points, config conventions)
are in `README.md` and `.planning/codebase/CONVENTIONS.md` — the latter is an
archive and may be stale; the code wins.
