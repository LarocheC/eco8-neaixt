# RT595 HiFi4 cycle benchmarks

int8 streaming graphs on the Cadence ISS, core `nxp_rt500_RI23_11_newlib` (Fusion-F1), built
through `iss/build_iss.sh` against `tflm/libtflm_rt500.a`. Per-frame output checksums are
identical across every column below, so each row is the same computation measured three ways.

## Results (2026-07-28)

| model | graph | flat, pre-fix | flat, Q16 driver | **`--mem_model`** | vs 3.17 M budget |
|---|---|---:|---:|---:|---:|
| NSNet2 plain GRU | 50 ops, 2 IO / 1 state | 1,575,579 | 1,522,055 | **27,976,329** | 8.8x over |
| ConvFSENet 192-384 | 190 ops, 10 IO / 9 states | 3,640,203 | 2,767,999 | **30,388,703** | 9.6x over |
| LiSenNet relu6-deep | 320 ops, 26 IO / 25 states | 19,186,677 | 8,902,092 | **76,428,487** | 24.1x over |

**No model is real-time on this chip today.** Arena high-water: NSNet2 9.8 KB, ConvFSENet 160 KB,
LiSenNet 277 KB. Weight blobs: 2.89 MB / 1.71 MB / 250 KB.

## Sparse NSNet2 — the variant that makes NSNet2 deployable

Dense NSNet2 is undeployable on this chip for one reason: 2.89 MB of int8 weights against a
1.30 MB `dram0_0_seg`. The structured-linear (sparse) variants fix that, and one of them is
faster *and* better-scoring than the dense model.

| variant | blocks | params | int8 PESQ | weights | arena | total | cyc/frame | vs budget | fits DRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense baseline | - | 2,783,657 | 2.8334 | 2886K | 17K | 2903K | 1,522,055 | 0.48x | **NO** |
| **blockdiag_full** | 4 | 701,657 | **2.8433** | 720K | 17K | **737K** | **1,464,617** | **0.46x** | **YES** |
| blockdiag_8 | 8 | 354,657 | 2.8256 | 376K | 17K | 393K | 1,736,358 | 0.55x | YES |
| monarch_full | 4 | 1,098,557 | 2.8463 | 1148K | **1169K** | 2317K | 3,310,045 | 1.04x | **NO** |
| monarch_8 | 8 | 553,369 | **2.8562** | 601K | 593K | 1194K | 4,537,793 | 1.43x | over budget |

PESQ is int8 over 824 VoiceBank-DEMAND test utterances, from `benchmarks/baselines.json`.

**`blockdiag_full` is the deploy pick:** 4x smaller than dense, fits with 565 KB to spare,
real-time with 2.2x headroom, and it *beats* the dense baseline on PESQ (2.8433 vs 2.8334).

**Monarch is a dead end on this chip in every configuration tested** — the two variants fail
differently but both fail, and the reason is structural rather than a tuning problem.

**The op-granularity law shows up again, and it dominates parameter count.** These variants all do
far less arithmetic than dense, yet only the blockdiag ones are competitive:

- torch_structured's linears do **not** lower to `FULLY_CONNECTED`. They become `BATCH_MATMUL`
  wrapped in `RESHAPE`/`TRANSPOSE`. Node counts: dense 50 (16 FULLY_CONNECTED), blockdiag_full 76
  (8 BATCH_MATMUL, 19 RESHAPE), **both monarch variants 128** (24 BATCH_MATMUL, 24 TRANSPOSE,
  35 RESHAPE).
- **`nblocks` does not change monarch's graph topology at all** — monarch_full (4 blocks) and
  monarch_8 (8 blocks) emit byte-for-byte the same op histogram. `nblocks` only sets the block
  *dimensions*. So the 2-factor permute's 24 TRANSPOSEs are a fixed tax you cannot tune away.
- **The arena is the real killer, not the cycles.** blockdiag needs **17 KB**; monarch needs
  593 KB (8 blk) to **1169 KB** (4 blk) — 35x to 69x more — because each factor's intermediate
  must be materialised, and bigger blocks mean bigger intermediates. monarch_full's arena alone
  (1.17 MB) nearly exhausts the 1.30 MB data segment before its 1.15 MB of weights are placed.
- Fewer, larger blocks *is* faster (monarch_full 3.31 M vs monarch_8 4.54 M; blockdiag_full
  1.46 M vs blockdiag_8 1.74 M) — but for monarch it trades cycles for arena and loses either way.
- **More blocks is worse for speed:** blockdiag_8 (8 blocks) is 19% slower than blockdiag_full
  (4 blocks) at half the size. Smaller blocks = smaller matmuls = less SIMD amortisation.

So on HiFi4: prefer **block-diagonal with few, large blocks**, and avoid any factorisation that
needs a permute — the transposes cost more than the MACs they save, and the materialised
intermediates cost more memory than the weights they save.

Reproduce (checkpoints' model code lives on `main`, so use a worktree overlay — never a branch
switch, since concurrent sessions share this checkout):

```bash
git worktree add --detach .lane/worktrees/main main
~/.venvs/rt595-export/bin/pip install "torch-structured>=1.3.0" "gru-qat>=0.4.0"   # PyPI, not the stale local clones
TORCH_STRUCTURED_BACKEND=torch ~/.venvs/rt595-export/bin/python deploy/rt595/host/export_tflite_multi.py \
    --model nsnet2 --checkpoint_file cp_blockdiag_full/g_best --repo $PWD/.lane/worktrees/main \
    --output deploy/rt595/host_out/nsnet2_blockdiagfull_streaming --int8 --calib vbd
```

Resolvers: `iss/resolvers/nsnet2_blockdiag_ops.cpp` (11 ops) and `nsnet2_monarch_ops.cpp` (12, adds
TRANSPOSE). The dense `nsnet2_ops.cpp` does **not** work for these — no `AddBatchMatMul`.

## Read the caveats before quoting any of these

**1. `--mem_model` is mandatory.** Plain `xt-run <elf>` models zero-latency single-cycle memory —
no caches, no stalls. That is a compute-only lower bound, not a latency estimate. It costs
8.6x–18.4x here, and it **inverts the ranking**: NSNet2 looks like the runaway winner on flat
memory and degrades the most under a cache model, because its 2.89 MB of int8 weights move more
bytes than the flat run claimed cycles. `--mem_model` is *still* zero-wait-state; the board adds
PSRAM latency on top and none of these weight blobs fit HiFi4 local SRAM.

**2. The budget is unverified.** `3.17 M cyc/frame` = 198 MHz x 16 ms hop, and the 198 MHz is a
TODO in `dsp_offload/dsp/dsp_main.cpp`. Read `CLOCK_GetDspClkFreq()` on the board before treating
any of the right-hand column as pass/fail.

**3. NSNet2 uses SYNTHETIC (random) weights** — `host/make_nsnet2_synth.py`. Valid for latency
(dense int8 op cost is weight-independent), never for quality. Replace with `nsnet2:baseline`
from HF.

**4. `--mem_model` runs ~10x slower to simulate.** These were taken with the frame loop capped to
1 (per-frame counts vary <0.1%).

## The Q16 driver fix

The middle column is not a model change — it is a fix to *our own* `app/model_se_stream.cpp`.
The state-feedback loop dequantised to float and re-quantised via `lrintf(x / scale)`, i.e. a
**scalar float divide per element**, measured at 59 cyc/elem. LiSenNet has 157,830 state elements
and 0/25 states memcpy-safe, so it ran on every one: **9.3 M of the 19.2 M — about 48% of what was
being attributed to the network was the driver.** Hoisting the loop-invariant scale ratio into
Q16 fixed point costs 5 cyc/elem. Bit-identical checksums.

The effect scales with state volume, which is why it is 2.16x for LiSenNet, 1.32x for ConvFSENet
and only 1.04x for NSNet2 (one 800-element state).

## What this says about architecture

Op granularity, not MAC count, sets HiFi4 throughput. LiSenNet (1.37 MMACs) and ConvFSENet
(1.45 MMACs) have near-identical arithmetic, but ConvFSENet is ~2.5x cheaper because its convs run
along the 257-wide frequency axis and amortise SIMD setup, while LiSenNet's per-frame streaming
convs do not. Under a memory model the picture shifts again toward *weight bytes moved*, which
punishes NSNet2's dense GRU/FC hardest.

**The untried lever is the windowed (stateless) view** — it trades recurrent state for larger
tensors and fewer, longer ops, which is exactly what both effects reward. It is already the
published STM32N6 deploy path and is not yet measured here.

## Reproducing

```bash
# headers
~/.venvs/rt595-export/bin/python host/gen_model_data.py      --model <m.tflite> --output <w>/model_data.h
~/.venvs/rt595-export/bin/python host/gen_io_layout_multi.py --model <arch> --checkpoint_file <cp>/g_best \
                                                             --tflite <m.tflite> --output <w>/model_io_layout.h
~/.venvs/rt595-export/bin/python host/gen_test_feats_multi.py --channels <1|3> --output <w>/se_test_feats.h
# build + run
cp app/model_se_stream.{cpp,h} iss/{iss_bench.cpp,fstubs.c,fsl_*.h} <w>/
cp iss/resolvers/<family>_ops.cpp <w>/model_ops_micro.cpp
SCR=<dir with tflm_rt500/> bash iss/build_iss.sh
xt-run --xtensa-core=nxp_rt500_RI23_11_newlib --mem_model iss_bench.elf
```
