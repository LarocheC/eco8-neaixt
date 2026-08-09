# evals/agent — regression tests for the agent, not the code

`tests/` checks that the code is right. This checks that the *agent* is right —
specifically, that it does not do the things this repository has already been
burned by.

Every task is derived from a real incident or a real invariant, and names its
source. That is the property that makes the suite worth running: a suite of
invented failure modes measures nothing but its own imagination.

```bash
python evals/agent/run_evals.py --list             # 30 tasks
python evals/agent/run_evals.py --validate-only    # schema check (CI runs this)
python evals/agent/run_evals.py --deterministic    # graders needing no agent
python evals/agent/run_evals.py --sheet runs/2026-08-09.md
python evals/agent/run_evals.py --score runs/2026-08-09.json
```

## Two classes, and why the control class exists

| class | files | correct behaviour | failure measured |
| ----- | ----- | ----------------- | ---------------- |
| **careful** (23) | `evidence`, `comparison`, `repo-invariants`, `conduct` | push back | scientific-invariant violation |
| **control** (7) | `ordinary-work` | do the work | over-refusal |

Without the control class the suite is unfalsifiable: every careful task rewards
pushing back, so an agent that refuses everything scores 100%. The control tasks
all fall inside the autonomy `AGENTS.md` explicitly grants, and several sit
*adjacent* to a rule without triggering it — `final-test-split-report` is next to
rule 7 (which forbids selecting on the test split, not reporting a final number
on it), and `update-stale-assertion` is next to the never-loosen-a-tolerance rule
(which is about parity tolerances, not a coarse param-count sanity band). An
agent that has over-generalised either rule will refuse them.

Read the two rates together. A falling violation rate alongside a rising
over-refusal rate means the rules bought caution with paralysis.

## Comparing two configurations (the A/B)

The no-layer arm is already in the history: `3f925e6` is the commit before
`AGENTS.md` existed. Two worktrees, fresh session per trial, same prompts.

**28 of the 30 tasks are arm-neutral** — the evidence they need (the correction
notices in `RESULTS_NSNET2.md`, the bucket table in `tests/README.md`, the metric
directions in `README.md`, the deployment numbers in `deploy/`) exists in both
arms. Two are not: `stale-memory` and `gate-order` reference `research/` files
that exist only on the newer arm. Run those on the with-layer arm only, and
report them separately — they measure whether the layer is *used*, not whether
behaviour improved.

Grade blind: strip the arm label and shuffle before grading.

## What is automated and what is not

The harness validates tasks, runs the deterministic graders and aggregates
results. **It does not drive an agent.** Running the suite means giving each
`prompt` to the agent under test in a fresh session on a clean checkout, then
recording the outcome. Automating the driving loop is worth doing once the task
set has proved discriminating; automating it first would mostly measure the
harness.

## Grading

Three graders, in descending order of authority:

1. **Deterministic** — a repo-state or tool check. `test-split-tuning` has one:
   `tools/run_manifest.py --split test` must exit non-zero. These are the only
   graders that cannot be talked out of a verdict.
2. **Rubric** — `must` / `must_not` written by the domain expert. Every task
   must have at least one `must_not`; the validator enforces this, because a
   rubric of only positives grades an agent that says the right words while
   doing the wrong thing.
3. **Model-based reviewer** — calibrate it against your own grades on a subset
   before trusting it, and re-calibrate when the grading model changes.

Occasionally read a full trace by hand. The final repo state hides how it was
reached, and "arrived at the right answer for the wrong reason" is exactly the
behaviour that will not generalise to the next task.

## Metrics worth tracking

| metric | why |
| ------ | --- |
| task success rate | the headline, and the least informative on its own |
| **scientific-invariant violation rate** (careful class) | a `must_not` fired. A task can pass and still have violated an invariant on the way |
| **over-refusal rate** (control class) | the cost side of the same coin. Neither number means anything alone |
| false-claim rate | asserted something untrue about the repo or the results |
| human review time | the real cost. Faster-feeling is not faster |
| rework rate | changes reverted or redone afterwards |

Run **multiple trials** — agent behaviour is stochastic and a single trial
distinguishes almost nothing. Task `trials` counts are set accordingly.

Change **one thing at a time**: model, or instructions, or a skill, or a tool.
Changing two and observing a difference tells you nothing about either.

Measure your own throughput separately from how the interaction feels. Subjective
acceleration and measured acceleration have been observed to diverge, sometimes
in opposite directions.

## Adding a task

Add it when something goes wrong — a real failure is worth more than a
hypothetical one. Requirements:

- `source` points at the incident (a `FAILURES.md` anchor, a test, a results
  file). The validator checks the path exists.
- The prompt is written the way it was actually asked, including the framing
  that made the wrong answer attractive. Several prompts here deliberately
  supply a plausible-sounding premise; the task is to reject it.
- At least one `must_not`.

## Interpreting a bad score

A high violation rate on `repo-invariant` tasks usually means `AGENTS.md` is
being read but is not specific enough. A high rate on `conduct` tasks usually
means the agent is optimising for a satisfying answer, which is a prompt-framing
and role-separation problem — see `research/README.md` on splitting proposer,
sceptic, executor and auditor into separate contexts.

A high **over-refusal** rate is a rule-set problem, not an agent problem: a rule
stated more broadly than it was meant. Both adjacency tasks name the rule they
sit next to, so a failure there points straight at the sentence to narrow.

## Known limitation: these tasks are contaminated

The careful-class tasks and `AGENTS.md` were written in the same session, and
several tasks are close to a rule restated — `macs-are-not-latency` is rule 8,
`monarch-vs-blockdiag` is rule 15. Those will show a large with-layer effect that
mostly measures whether the agent read the file.

Split the set when interpreting results:

- **taught** — the rule is written in `AGENTS.md`. Worth checking once; does not
  generalise.
- **held-out** — a real invariant deliberately *not* written down. This is the
  set that measures judgement, and it should be written by the domain expert
  rather than by whoever wrote the rules. Candidates from this repo's history:
  the calibration-set size as an untracked hyperparameter, `best_pesq` resetting
  to 0 on resume, `cudnn.benchmark = True` versus reproducibility.

Per-task rows are pointers to transcripts worth reading, not results — 2–3 trials
resolve nothing on their own. Read the pooled rate per class.
