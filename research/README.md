# research/ — the current-state and history tier

`AGENTS.md` holds the **rules**. This directory holds **what we currently
believe** and **what actually happened**. Keeping those apart is the whole point:
an agent that cannot tell a 2026-04 handover from today's truth will confidently
act on an obsolete plan.

```
research/
  NOW.md            current objective + next actions.       Rewrite freely. Must be true today.
  CLAIMS.yaml       every claim we make, and its evidence.  Rewrite freely; status changes are the point.
  FAILURES.md       what did not work, and why.             Append-only.
  hypotheses/       one card per proposed experiment.       Mutable until status: accepted, then frozen.
  experiments/      run manifests + results.                IMMUTABLE once written.
  decisions/        short architecture/process decisions.   Append-only; supersede, never edit.
  SCHEMA.md         the field reference for cards and claims.
```

## The loop

```
hypothesis card (falsifier stated)
    → independent critique (fresh context, sceptic role)
    → cheap pilot
        → invalid / negative → FAILURES.md, card status: refuted
        → promising          → replicated full run (all seeds)
            → compiled-target measurement (if any cost claim)
                → CLAIMS.yaml entry with evidence paths
                    → generated tables in RESULTS_*.md / paper
```

Nothing skips a stage. In particular: no router training before an oracle
ceiling exists, and no publication claim before a compute-matched static
comparison exists.

## Roles are separate contexts, not separate agents

The same model in one long context will propose an idea, implement it, look at
the result, decide it worked, and write the claim — with no independent check
anywhere in the chain. Split it into sequential passes with fresh context:

| role         | gets                                        | must not get                       |
| ------------ | ------------------------------------------- | ---------------------------------- |
| **proposer** | the question, prior art, `FAILURES.md`      | —                                  |
| **sceptic**  | the card only                               | the proposer's narrative           |
| **executor** | the accepted card, the code                 | freedom to change the card         |
| **auditor**  | the card, the diff, the raw artifacts       | the proposer's or executor's prose |

Sequential is enough. Deterministic scripts outrank agent agreement every time.

## Validation

`uv run python tools/research_lint.py` checks the mechanical invariants: card
schema, claim→evidence paths resolve, no `accepted` card missing a falsifier, no
claim citing an experiment that does not exist, `NOW.md` not stale. CI runs it
on every push.
