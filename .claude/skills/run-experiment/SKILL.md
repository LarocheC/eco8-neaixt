---
name: run-experiment
description: Execute an accepted experiment card with full provenance capture. Use when running training, sweeps, quantization, export or evaluation whose results will be published or cited.
---

# Run an experiment

You are in the **executor** role. You implement the accepted card. You do not
redesign it, and you do not decide whether it worked.

## Before you start

1. Load the card from `research/hypotheses/<id>.yaml`. If `status` is not
   `accepted`, stop — designing is `design-experiment`'s job and accepting is
   the human's.
2. Check `depends_on`. A card whose dependency has not completed does not run,
   however interesting it looks.
3. Commit the tree first. A dirty run is recorded as dirty and is worth less.

**If the card is wrong, say so and stop.** Do not silently widen the split,
swap the baseline, add a seed, or change the metric. A card edited mid-run is
no longer a pre-registration, which is the only property that makes it useful.

## Run it through the manifest wrapper

Every run, without exception:

```bash
uv run python tools/run_manifest.py \
    --id <card-id>-<variant>-seed<N> \
    --hypothesis <card-id> \
    --config configs/<run>.json \
    --dataset JacobLinCool/VoiceBank-DEMAND-16k \
    --split train_holdout --seed <N> \
    -- python -m <family>.train --config configs/<run>.json --checkpoint_path cp_<run>
```

Export `ECO8_AGENT_MODEL` / `ECO8_AGENT_HARNESS` / `ECO8_AGENT_SESSION` first so
a later regression can be attributed to a model or instruction change rather
than to the code.

The wrapper refuses `--split test`. That is not an obstacle to work around.

## Quantization and export runs

- Re-run the export parity tests for the family before trusting any exported
  number: `uv run pytest tests/test_<family>_*parity*`.
- Never loosen a tolerance. `tests/README.md` has the diagnostic bucket table
  for `max_abs_err`; each bucket names a specific pitfall.
- After any int8 export, **audit the graph** — do not trust `quantize_static`.
  Confirm no weight initializer of a compute op is still FLOAT. `nsnet2/quant.py`
  does this for `Einsum`; a new export path needs its own equivalent before its
  numbers may be called int8. Precedent: `research/FAILURES.md#einsum-int8`.

## Reporting

Write what happened, in this order, and no further:

1. what the card predicted;
2. what was measured, with the manifest id and the artifact path;
3. whether the `accept_if` or the `falsify_if` condition was met, quoting both.

Then stop. Do not:

- explain *why* the result came out that way — a post-hoc mechanism is a new
  hypothesis, and belongs in a new card;
- try a different metric, split or subset because the primary one was negative;
- adjust the card's thresholds to match the result.

A failed or negative run keeps its manifest directory and gets an entry in
`research/FAILURES.md`.

## Definition of done

- manifest committed under `research/experiments/<id>/`, with raw per-utterance
  outputs where the run produced scores;
- card's `experiments:` list updated, `status: complete` with an `outcome`;
- `research/NOW.md` updated;
- `uv run pytest` and `uv run python tools/research_lint.py` both green;
- the interpretation handed to a fresh context — see `audit-evidence`.
