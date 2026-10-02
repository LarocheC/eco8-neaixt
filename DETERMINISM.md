# Bitwise-reproducible NSNet2 / butterfly training

Same config, same seed, different answer. `s2_R_s1234` and stage-1 `ba_R` are
the same config and the same seed, yet differ by +0.022 / +0.038 / +0.046 PESQ
at epochs 60 / 80 / 90 (BUTTERFLY_ARCH.md, "Noise note") — larger than the
≤ 0.02 of the butterfly-a1 control rerun. This branch makes that run-to-run
drift go away, behind opt-in flags.

Everything here defaults to **off**: a config written before these flags
existed trains bit for bit as it did, so running and queued studies are
untouched.

## The flags

Add to a training config (all three independent, all default `false`):

```json
"deterministic":  true,
"data_generator": true,
"exact_resume":   true
```

| flag | what it fixes | cost |
|---|---|---|
| `deterministic` | Nondeterministic kernels: the MetricDiscriminator's conv backward (cuDNN algorithm choice) and torch_structured's Triton butterfly backward (`tl.atomic_add` into the `d_twiddle` scratch — float summation order varies per launch). Also seeds the main process's `random` / `numpy`, which train.py never seeded. | the butterfly backward routes through torch_structured's pure-PyTorch oracle |
| `data_generator` | The train DataLoader had no `generator=` / `worker_init_fn`, so worker seeds were drawn from the global RNG **after** model init — two architectures with the same seed saw **different crops**. | none |
| `exact_resume` | Resume-from-step-k now equals the uninterrupted run bitwise. | re-reads the already-trained batches of the interrupted epoch |

`torch.use_deterministic_algorithms(True)` is **not** usable here: the
discriminator's `adaptive_max_pool2d` backward has no deterministic CUDA
implementation.

## Why exact resume needs more than the optimizer

The LR schedule and the Adam state were already verified to restore exactly,
and a crash+resume still cost the butterfly control ~0.07 PESQ. Three things
were missing:

1. **Position within the epoch.** The old resume restarted the interrupted
   epoch at batch 0 while carrying `steps` forward, so the already-trained
   batches were trained on twice *and* the step counter desynchronised from the
   data (shifting the LR warmup and the checkpoint/validation grid).
   `next_batch` is now stored and those batches are skipped.
2. **The CUDA RNG.** The discriminator contains `nn.Dropout(0.3)` and trains in
   `.train()` mode, so the CUDA RNG advances on *every* step.
3. **Two snapshots, not one.** The DataLoader draws its worker base seed when
   the *iterator* is created — once per epoch, before any batch is seen. So a
   checkpoint stores `epoch_start_rng` (restored at the top of the epoch, which
   is what makes the workers replay the same crops) as well as `rng` (the state
   at the checkpointed batch, restored once the skipped batches are consumed).

4. **Restoring at the right instant.** The skipped batches are replayed by
   driving the iterator (`next()`), and the checkpointed RNG is restored
   *before* the next batch is drawn. Doing it from inside the loop body looks
   equivalent and is not: with `num_workers=0` there are no workers and the
   crop happens inside `next()`, so a restore after the draw rewinds the stream
   by one batch. That bug was caught by `test_resume_replays_the_same_batches[0]`,
   which is why that test is parametrised over `num_workers` 0 and 2 — today's
   configs all use `num_workers: 5`, where the bug is invisible.

Checkpoints written by this branch carry `next_batch`, `rng` and
`epoch_start_rng` unconditionally (a few KB), so a run started without
`exact_resume` can still be resumed exactly later. Checkpoints written before
this branch lack them and fall back to the old restart-the-epoch behaviour,
with a printed warning.

> **Trap:** the RNG snapshot must stay loadable under `torch.load`'s
> `weights_only=True` default (torch ≥ 2.6), which `common.utils.load_checkpoint`
> relies on. A raw `np.random.get_state()` tuple carries an ndarray whose
> unpickler global is not allowlisted — storing one makes every `do_` checkpoint
> fail to load, i.e. breaks resume outright. `common/determinism.py` packs the
> numpy state into primitives instead; `tests/test_determinism.py` guards it.

## Running the tests

Fast, no GPU or dataset needed:

```bash
pytest tests/test_determinism.py
```

End-to-end, driving the real `nsnet2.train` and comparing the checkpoints it
writes (`--flags-off` reruns the same comparison with the flags disabled, which
is how you show the flags are doing the work):

```bash
# two fresh runs must agree bitwise
python -m nsnet2.determinism_accept twin   --config configs/ba_R.json --steps 50 --ckpt-every 10

# stopping at step 20 and resuming must equal never having stopped
python -m nsnet2.determinism_accept resume --config configs/ba_R.json --steps 50 --cut 20 --ckpt-every 10

# same seed, different architecture -> same crops
python -m nsnet2.determinism_accept batches --config configs/ba_R.json --config-b configs/ba_D_grid.json

# step time and peak GPU memory, flags on vs off
python -m nsnet2.determinism_accept bench  --config configs/ba_R.json --batch 256
```

`--subset N` truncates the training set so a short run still crosses epoch
boundaries. That matters: at batch 256 an epoch is only ~45 batches, so
**epoch-boundary handling is on the normal resume path**, not an edge case.

`--max_steps` (new, `nsnet2/train.py`, default 0 = no limit) is what lets a run
stop on an exact step; it stops inside the epoch, before the schedulers step,
so a partial epoch never advances the per-epoch LR schedule. `train_subset`
(config, default 0 = off) is the matching epoch-length hook.

## Measured

Hardware caveat: these were run on a **GTX 1080 Ti (sm_61), torch 2.7.1+cu118,
torch-structured 1.2.4, Triton 3.3.1** — not the 4090 / torch 2.14 / ts 1.3.0
box the probe was written on, and CUDA 13 dropped Pascal so that environment
cannot be reproduced here. On this GPU the Triton butterfly **backward** does
not run at all (1.2.4 `CompilationError`, 1.3.0 `PTXASError`; the forward is
fine), so every `ba_R` flags-off cell is "not runnable", not "not run".

Acceptance, batch 256, 50 steps, resume cut at step 20
(`nsnet2/determinism_accept.py`). At batch 256 an epoch is 45 batches, so the
resume starts mid-epoch 0 and runs through into epoch 1 — the epoch boundary is
covered by construction.

| config | test | flags ON | flags OFF (control) |
|---|---|---|---|
| `ba_D_grid` | twin | **bitwise equal** | differ, max abs 2.787e-01 |
| `ba_D_grid` | resume | **bitwise equal** | differ, max abs 2.918e-01 |
| `ba_R` | twin | **bitwise equal** | not runnable on sm_61 |
| `ba_R` | resume | **bitwise equal** | not runnable on sm_61 |

The drift compounds: the same flags-off `ba_D_grid` twin differs by max abs
1.9e-04 after 6 steps and 2.8e-01 after 50. That is the mechanism behind the
+0.046 PESQ at epoch 90 — not a fixed noise floor.

Same seed, different architecture, first 4 batches hashed: `data_generator`
off → `ba_R` and `ba_D_grid` differ; on → identical (`e5b808ca5249` at batch
256, and likewise at 64 and 16).

Cost, `nsnet2.determinism_probe`, 5 steps, step time / peak GPU memory:

| config | batch | flags OFF | flags ON | delta |
|---|---|---|---|---|
| `ba_D_grid` | 256 | 7809 ms / 7638 MiB | 7834 ms / 7638 MiB | +0.3 % / +0 |
| `ba_D_grid` | 64 | 1860 ms / 2028 MiB | 1872 ms / 2092 MiB | +0.6 % / +64 MiB |
| `ba_D_grid` | 16 | 584 ms / 852 MiB | 604 ms / 852 MiB | +3.4 % / +0 |
| `ba_R` | 256 | not runnable | 3821 ms / 4534 MiB | — |
| `ba_R` | 64 | not runnable | 1256 ms / 1223 MiB | — |
| `ba_R` | 16 | not runnable | 850 ms / 329 MiB | — |

**Do not read `ba_D_grid`'s ~0 % as the cost of the flags in general.**
`bflygrid` is built from `StageButterfly`, a pure-PyTorch einsum, so it has no
Triton butterfly and no oracle fallback — its only determinism cost is
`cudnn.deterministic`, which is nearly free. The oracle backward is paid only
by butterfly layers, i.e. `ba_R`, which is exactly the column this box cannot
measure. The handoff's own figure from the 4090 (+0–19 % step time, up to
+0.5 GB at batch 64) remains the reference for that, and re-running

```bash
python -m nsnet2.determinism_accept bench --config configs/ba_R.json --batch 256
```

there is the one outstanding measurement.

## Known residuals

- **`g_best` can differ across a resume.** `checkpoint_interval` and
  `validation_interval` are both 5000 by default, so they fire on the same
  step, and the checkpoint is written *before* that step's validation. A resume
  skips the checkpointed batch, so that one validation never re-runs and its
  `best_pesq` update is lost. Weights are unaffected; only the `g_best`
  selection can shift.
- **Sparsity masks are re-derived on resume** by magnitude from the restored
  weights (`SparsityController.from_config`). That is stable for an already
  masked tensor, but it is re-derivation, not restoration — untested for
  bitwise equality under `h.sparsity`.
- **Multi-GPU (`num_gpus > 1`) is untested.** The RNG snapshot covers all
  visible devices, but `DistributedSampler` ordering and per-rank skipping have
  not been exercised.

## Not done: a deterministic *fast* butterfly backward

`deterministic` buys its determinism by routing the butterfly backward through
torch_structured's pure-PyTorch oracle. The faster fix is to make the Triton
backward itself deterministic. It was **not attempted**, because it cannot be
validated or benchmarked on the machine this branch was developed on: the
Triton butterfly backward does not run on a GTX 1080 Ti (sm_61) at all —
torch-structured 1.2.4 fails to compile it (`CompilationError`) and 1.3.0 gets
as far as `PTXASError: Internal Triton PTX codegen error`. The forward compiles
and runs fine in both. Writing a numerically delicate gradient kernel with no
way to run it would be worse than leaving it.

The shape of the fix, for whoever picks it up on the 4090 box:

- The nondeterminism is in `torch_structured/_triton/butterfly/op.py`: the
  backward launches `grid = (n_row_tiles, batch_size * nstacks)` and every
  program accumulates into a shared fp32 `d_twiddle_scratch` with
  `tl.atomic_add(..., sem='relaxed')` (4 per stage for real, 8 for complex).
  Float addition is not associative, so the summation order — and the result —
  varies per launch.
- A per-program partial buffer is **not** affordable: scratch would be
  `batch*nstacks x twiddle.numel()`, which for `Butterfly(512, 512)` at the
  training batch (256 x 126 frames = 32,256 rows) is ~1.2 GB.
- So fix the number of partials instead: give the reduced axis a fixed `G`
  (say 32–128), launch `grid = (G * nstacks, n_row_tiles)`, have each program
  loop over its `batch/G` slice accumulating in registers, write
  **non-atomically** into its own `(g, ...)` slice, and finish with
  `scratch.sum(0)`. Scratch becomes `G x twiddle.numel()` (~2.4 MB at G=64),
  the atomics disappear, and the result is reproducible as long as `G` is a
  constant and never derived from occupancy or autotuning.
- Keep the oracle as the fallback, and benchmark against atomics: contended
  fp32 atomics to the same address serialise badly, so removing them may well
  be faster, not merely more deterministic.
- Note the grid-axis bug worked around on `butterfly-arch` lives in the same
  launches (CUDA caps grid y at 65535; `bfly()` in `nsnet2/bfly_arch.py` splits
  the rows to stay under it). Swapping the axes as proposed above would make
  that workaround unnecessary — worth doing in the same change.
