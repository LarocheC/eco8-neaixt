# research/experiments/ — immutable run records

One directory per run: `research/experiments/<id>/manifest.json`, plus whatever
raw artifacts that run produced (per-utterance scores, profiler output, board
logs).

**Immutable.** Once committed, a manifest is never edited. A re-run gets a new
id. A failed run keeps its directory — a failure that leaves no trace gets
repeated.

Manifests are generated, never hand-written:

```bash
# wrap a run
uv run python tools/run_manifest.py --id <id> --hypothesis <card-id> \
    --config configs/<run>.json --seed 13 --split train_holdout \
    -- python -m nsnet2.train --config configs/<run>.json --checkpoint_path cp_<run>

# record something measured off-host (a board session)
uv run python tools/run_manifest.py --id <id> --claim <claim-id> --no-run \
    --toolchain stedgeai=4.0.1 --toolchain board=STM32N6570-DK
```

Set `ECO8_AGENT_MODEL` / `ECO8_AGENT_HARNESS` / `ECO8_AGENT_SESSION` in the
environment and the manifest records which agent produced the run — needed to
attribute a regression to an instruction or model change rather than to the code.

`EXAMPLE-manifest/` holds one real manifest (from wrapping `research_lint.py`)
so the schema is visible without hunting. It is not evidence for anything.

## Backfill status

Most numbers in `RESULTS_*.md` predate this directory and exist only as
hand-maintained markdown tables. Those claims carry an `evidence_gap` in
`research/CLAIMS.yaml`. Closing that gap is a listed next action in
`research/NOW.md`; the priority order is the claims a paper would lean on
hardest — the NSNet2 int8 re-measurement and the STM32N6 on-board timings.
