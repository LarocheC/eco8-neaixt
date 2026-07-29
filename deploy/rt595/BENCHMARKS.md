# RT595 cycle benchmarks

## HiFi4 SILICON RESULT (2026-07-29) — real-time confirmed, ISS timing exact to 0.1%

`blockdiag_full` run on the **HiFi4 DSP of the EVK itself**, booted entirely over SWD
(`scripts/dsp_hw_run.py` — no M33 firmware change, no reflash; the debugger performs the
whole `BOARD_DSP_Init()` register sequence, places the three blobs, and reads
`g_bench_result` out of DSP DRAM).

| | cyc/frame | ms @198 MHz | vs 3,168,000 budget |
|---|---:|---:|---:|
| **HiFi4 (silicon)** | **1,466,196** | **7.41** | **0.46x — real-time, 2.16x headroom** |
| HiFi4 (ISS, flat) | 1,464,615 | 7.40 | 0.46x |
| M33 (silicon) | 5,319,161 | 26.86 | 1.68x over |

- **Silicon/ISS = 1.001x.** For an image that fits entirely in DSP local SRAM, the flat
  ISS number *is* the silicon number. This closes the `--mem_model` scare below: that
  8.6–18.4x penalty was measured through the generic sim LSP linking off-core, and the
  Fusion-F1 has no caches to model — local-RAM images run at the ISS's zero-wait-state
  figure on real hardware.
- **All 16 checksums match the ISS and the M33 exactly** — three implementations
  (xa_nnlib on silicon, xa_nnlib on ISS, CMSIS-NN on M33), one bit-exact answer.
- Frame spread on the DSP is 50 cycles (0.003%): single-frame measurements are exact.
- M33/HiFi4 silicon ratio: **3.63x**.
- Arena on the DSP build: 17,176 B vs 17,096 B on the ISS/M33 builds — TFLM's internal
  alignment padding shifts with the arena's base address; benign.
- The measured `XT_RSR_CCOUNT` read overhead is 1 cycle.
- The DSP ran at 198 MHz (SYSPLL0 PFD1 = 528×18/24 = 396 MHz, /2), NXP's own
  `dsp_support.c` configuration, so the 3,168,000 budget applies as-is. VDDCORE stayed at
  the PCA9420 power-on default 1.0 V, which is the required tier for 198/198 MHz.
- The run also empirically confirmed the M33↔DSP data alias: DSP D-side 0x00840000 is
  physical/M33 0x00040000 (−0x800000), while I-side addresses are identity-mapped.

**Bottom line: sparse (block-diagonal) NSNet2 speech enhancement runs in real time on the
i.MX RT595's HiFi4, measured on silicon, with 2.16x headroom and PESQ above the dense
baseline.**

## M33 SILICON RESULT (2026-07-29) — the ISS is numerically exact

`blockdiag_full` flashed to the EVK and run on the **Cortex-M33**. This was the first
hardware measurement in this project.

```
=== RT595 LiSenNet streaming SE (M33 / TFLite-Micro) ===
SE model: 2 IO, 1 states; arena used 17096 / 524288 B
core 198 MHz; 16 test frames of 257-bin features
frame, cycles, us, mask_checksum
0, 5328934, 26913, 202378        ...        15, 5316635, 26851, 250472
```

| | cyc/frame | ms @198 MHz | vs 3,168,000 budget |
|---|---:|---:|---:|
| **M33 (silicon)** | **5,319,161** | 26.86 | 1.68x over |
| HiFi4 (ISS) | 1,464,615 | 7.40 | 0.46x — real-time |

**All 16 per-frame mask checksums are byte-identical between silicon and the ISS.**
That is the load-bearing result. The M33 runs CMSIS-NN and the HiFi4 runs xa_nnlib —
completely different kernel implementations — yet they agree exactly on every frame. So
the ISS is faithfully executing the real model, int8 inference is deterministic across
both backends, and the recalibrated export behaves identically on hardware.

Other confirmations from the same run:
- **Core clock really is 198 MHz**, so the 3,168,000 cyc/frame budget is correct — this
  was previously an unverified assumption inherited from a TODO.
- **Arena is 17,096 B on silicon, exactly as the ISS reported.**
- Frame-to-frame spread is 0.24%, so single-frame ISS measurements are representative.
- The M33/HiFi4 ratio is **3.63x**, a sane SIMD-vs-scalar gap that gives an empirical
  anchor for the DSP figures.

What this does NOT settle: the HiFi4 number itself is still simulated, and the ISS assumes
zero-wait-state local RAM. Only a DSP-side run measures real memory behaviour — see
`ONBOARD_MEASUREMENT.md` Tier 2.

Worth noting for the M33 track: blockdiag NSNet2 at 5.32 M cyc/frame is **4x faster than
LiSenNet's 21.2 M** on the same core, i.e. 1.68x over real-time rather than 6.7x.

---

# HiFi4 cycle benchmarks (simulated)

int8 streaming graphs on the Cadence ISS, core `nxp_rt500_RI23_11_newlib` (Fusion-F1), built
through `iss/build_iss.sh` against `tflm/libtflm_rt500.a`. Per-frame output checksums are
identical across every column below, so each row is the same computation measured three ways.

## Results (2026-07-28)

| model | graph | flat, pre-fix | flat, Q16 driver | **`--mem_model`** | vs 3.17 M budget |
|---|---|---:|---:|---:|---:|
| NSNet2 plain GRU | 50 ops, 2 IO / 1 state | 1,575,579 | 1,522,055 | **27,976,329** | 8.8x over |
| ConvFSENet 192-384 | 190 ops, 10 IO / 9 states | 3,640,203 | 2,767,999 | **30,388,703** | 9.6x over |
| LiSenNet relu6-deep | 320 ops, 26 IO / 25 states | 19,186,677 | 8,902,092 | **76,428,487** | 24.1x over |

**None of these three is real-time on this chip** — but with the silicon validation above, the
reason differs per model and the `--mem_model` column should be ignored (sim-LSP artifact; see
caveat 1). LiSenNet *fits* local SRAM (250 KB weights + 277 KB arena), so its flat Q16 number
is the real one: 8.9 M = 2.8x over budget, a compute problem. Dense NSNet2 (2.89 MB) and
ConvFSENet (1.71 MB) exceed the 1.24 MB DSP data segment, so their flat numbers are
unrealizable as-linked — a memory problem first. Arena high-water: NSNet2 9.8 KB, ConvFSENet
160 KB, LiSenNet 277 KB.

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

**1. `--mem_model` vs flat memory — RESOLVED ON SILICON.** Plain `xt-run <elf>` models
zero-latency single-cycle memory. The 8.6x–18.4x `--mem_model` penalties in the table were
measured through the generic `-mlsp=sim` LSP, which links the image *off-core* — an artifact,
not a property of the chip: the Fusion-F1 has no caches, and the silicon run above shows a
local-SRAM image (real min-rt LSP) hitting the flat ISS figure to 0.1%. The distinction that
actually matters is **does the image fit DSP local SRAM** (data ≤1.24 MB at 0x840000, text
≤1 MB at 0x180400). blockdiag_full fits → flat ISS is exact. Dense NSNet2 (2.89 MB weights)
does not fit → would need PSRAM behind the outbound PIF bus, and no simulation here bounds
that cost.

**2. The budget is verified.** `3.17 M cyc/frame` = 198 MHz × 16 ms hop; both silicon runs
confirmed 198 MHz (M33 banner, and the DSP clocked at SYSPLL0 PFD1/2 = 198 MHz by
`dsp_hw_run.py` itself).

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
