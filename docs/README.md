# Results index

Everything this repository has measured, in two views: **how good a model is**, and
**what it costs on a chip**. Start with the matrix below, then follow a link down.

| | |
| --- | --- |
| [models/](models/) | One page per family — [NSNet2](models/nsnet2.md), [ConvFSENet](models/convfsenet.md), [LiSenNet](models/lisennet.md), [BASENet](models/basenet.md). Architecture, sweeps, FP32 vs int8, streaming. |
| [targets/](targets/) | One page per chip — [NXP RT595](targets/rt595.md), [ST STM32N6](targets/stm32n6.md). What ran on the hardware, at what cost, and how to point your own checkpoint at it. |
| [studies/](studies/) | [Structured factorisation](studies/structured-factorisation.md) (block-diagonal vs genuine two-factor Monarch) and [cross-family metrics](studies/cross-family-metrics.md) (every published model under one harness). |
| [publishing/](publishing/) | [HuggingFace model cards](publishing/hf-nsnet2.md). |

**Provenance is part of every hardware number here.** `SILICON` means read off a board with
a raw capture committed; `ISS` means the Cadence instruction-set simulator; `MODELLED` means
computed from datasheet constants with no instrument attached. A figure without one of those
labels has not been through this discipline yet. The authoritative record for the RT595 is
[deploy/rt595/results/PROVENANCE.md](../deploy/rt595/results/PROVENANCE.md).

---

Two tables. Table 1 is quality-and-size, measured on the host over the full 824-utterance VoiceBank-DEMAND test split. Table 2 is cost-on-a-chip, one row per (model, chip, engine) that has an actual measurement behind it. Every number below traces to a file; the source column names it.

Naming warning that must survive this reorganisation: **the `deploy/stm32n6/` documents still use the pre-rename labels.** What `deploy/stm32n6/ONBOARD_MEASUREMENT.md:103-104` calls `monarch_full` / `monarch_8` are the **block-diagonal** models `blockdiag_full` / `blockdiag_8` — same weights (0.72 MB / 0.37 MB), same on-target cosines (0.99979 / 0.99994), same latencies, as published under the corrected names in `docs/models/nsnet2.md:146-147`. Genuine two-factor Monarch has **never** been run on the STM32N6. The rows below use the corrected names.

## Table 1 — models

| model | params | FP32 PESQ | int8 PESQ | streaming? | source |
| --- | ---: | ---: | ---: | --- | --- |
| NSNet2 `baseline` (dense) | 2,783,657 | 2.845 | 2.833 | yes (streaming-shape ONNX) | `docs/models/nsnet2.md:51, 191`; `deploy/rt595/BENCHMARKS.md:109`; `benchmarks/baselines.json` (`nsnet2__baseline__int8` = 2.8334) |
| NSNet2 `blockdiag_full` (nblocks 4) | 701,657 | 2.827 | 2.843 | yes | `docs/models/nsnet2.md:53`; `deploy/rt595/BENCHMARKS.md:110`; `benchmarks/baselines.json` (`nsnet2__blockdiag_full__int8` = 2.8433) |
| NSNet2 `blockdiag_8` (nblocks 8) | 354,657 | 2.832 | 2.826 | yes | `docs/studies/cross-family-metrics.md:187`; `deploy/rt595/BENCHMARKS.md:111` (2.8256). `docs/models/nsnet2.md:52` prints the same figure rounded down, as 2.825 |
| NSNet2 `monarch_full` (2-factor, nblocks 4) | 1,098,557 | 2.838 | 2.846 | yes | `docs/models/nsnet2.md:87`; `deploy/rt595/BENCHMARKS.md:112` |
| NSNet2 `monarch_8` (2-factor, nblocks 8) | 553,369 | 2.861 | 2.856 | yes | `docs/models/nsnet2.md:85`; `deploy/rt595/BENCHMARKS.md:113` |
| NSNet2 `monarch_40` (2-factor, nblocks 40) ‡ | 0.117 M | 2.837 | 2.837 | yes (host only) | `docs/models/nsnet2.md` on the un-merged `monarch-nblocks-sweep` branch |
| LiSenNet `gru` (quality reference) | 36,783 | 3.006 | 2.930 (int8 + noisy phase) | yes (host streamer, parity 2e-7) | `docs/models/lisennet.md:40-44, 74-75`; `benchmarks/baselines.json` (`lisennet__gru__int8_rt` = 2.9300) |
| LiSenNet `conv-hardened` nc24 (N6 deploy model) | 36,288 | 3.013 | 2.982 harness · 2.998 model doc · 2.963 streaming export † | yes (both windowed and 17-state FIFO) | `docs/models/lisennet.md:208, 461`; `benchmarks/baselines.json` (`lisennet__conv-hardened__int8_rt` = 2.9816); `docs/studies/cross-family-metrics.md:57-62` |
| LiSenNet `conv-hardened-deep` relu6-deep (best PESQ) | 46,248 | 3.084 | 3.014 all-int8 · 3.052 decoder-FP32 hybrid | windowed only | `docs/models/lisennet.md:289-291`; `docs/targets/stm32n6-lisennet-npu.md:150-153`; `benchmarks/baselines.json` (`lisennet__conv-hardened-deep__int8_rt` = 3.0148) |
| ConvFSENet (streaming, deployed) | 1.45 M | 2.931 | 2.911 | yes (per-block FIFO) | `docs/models/convfsenet.md:29-33`; `benchmarks/baselines.json` (`convfsenet__convfsenet__int8` = 2.9108) |
| ConvFSENet windowed-256, `coldstart=replicate` | 1.45 M | 2.933 | 2.913 | stateless window (no FIFO) | `docs/models/convfsenet.md:136` |
| BASENet-3 (non-causal, re-derived + trainer fixes) | 0.830 M | 3.359 | — see note | PyTorch streamer only; no streaming ONNX | `docs/models/basenet.md:37, 60, 99-101, 263` |

‡ `monarch_40` is the one row not sourced from the `docs/` tree. It lives only on the un-merged `monarch-nblocks-sweep` branch and is absent from `benchmarks/baselines.json`, so it has not been through the cross-family metric harness. Treat it as provisional until that branch merges.

† The three LiSenNet `conv-hardened` int8 numbers are three different things, and none of them is a correction of another. **2.982** is the cross-family harness scoring the graph published to the Hub (`benchmarks/baselines.json`). **2.998** is `docs/models/lisennet.md:208` scoring the graph that produced that table row — a different calibration draw, one of the three cases `docs/studies/cross-family-metrics.md:57-62` names explicitly. **2.963** is the separate 17-state streaming int8 export, on the same 824-utterance split (`docs/models/lisennet.md:461`).

**BASENet int8.** Static full int8 (QDQ) **collapses to ~1.2 PESQ** regardless of calibration method — the cell is not empty for lack of trying (`docs/models/basenet.md:190-195`). Dynamic weight-only int8 is near-lossless there (3.24 → 3.19), but that pair is quoted against a 3.24 FP32 reference that matches none of the three checkpoints the file lists (3.116 / 3.330 / 3.359), so it is not reproducible from this file and is omitted. No BASENet checkpoint has been published to HuggingFace (`docs/models/basenet.md:265-267`).

## Table 2 — targets

Real-time budget on both chips is one 16 ms hop (hop 256 @ 16 kHz). On the RT595 that is 3,168,000 cycles at a silicon-verified 198 MHz (`deploy/rt595/BENCHMARKS.md:172-174`).

### i.MX RT595 (MIMXRT595-EVK)

| model | chip | engine | ms/frame | meets 16 ms? | provenance |
| --- | --- | --- | ---: | --- | --- |
| NSNet2 `blockdiag_full` | i.MX RT595 | Cortex-M33, TFLM + CMSIS-NN | **26.86** (5,319,161 cyc) | **NO** — 1.68× over | **SILICON**, raw capture committed (`results/silicon_m33_nsnet2_blockdiag_full.txt`) |
| NSNet2 `blockdiag_full` | i.MX RT595 | HiFi4 DSP, xa_nnlib | **7.41** (1,466,196 cyc) | **YES** — 0.46× of budget, i.e. 2.16× headroom | SILICON, **no capture retained** |
| NSNet2 `blockdiag_full` | i.MX RT595 | HiFi4, Cadence ISS (flat) | 7.40 (1,464,615–1,464,617 cyc) | YES — 0.46× | ISS (1.001× vs silicon) |
| NSNet2 `blockdiag_8` | i.MX RT595 | HiFi4, ISS | 8.77§ (1,736,358 cyc) | YES — 0.55× | ISS |
| NSNet2 `baseline` (dense) | i.MX RT595 | HiFi4, ISS | 7.69§ (1,522,055 cyc) | **NO** — 2.89 MB weights exceed the 1.24 MB DSP data segment; cycles unrealizable as linked | ISS, **synthetic weights** |
| NSNet2 `monarch_full` | i.MX RT595 | HiFi4, ISS | 16.72§ (3,310,045 cyc) | **NO** — 1.04× over, and its 1,169 KB arena alone nearly exhausts the data segment | ISS |
| NSNet2 `monarch_8` | i.MX RT595 | HiFi4, ISS | 22.92§ (4,537,793 cyc) | **NO** — 1.43× over | ISS |
| ConvFSENet | i.MX RT595 | HiFi4, ISS | 14.21§ (2,812,605 cyc, corrected) | **NO** — under the cycle budget (0.89×), but 1.71 MB exceeds the 1.24 MB data segment; cycles unrealizable as linked | ISS (the 2,767,999 in the older table was measured with a dead recurrent state) |
| LiSenNet relu6-deep (streaming TFLite export) | i.MX RT595 | HiFi4, ISS | 44.96§ (8,902,092 cyc) | **NO** — 2.8× over; fits SRAM (250 KB weights + 277 KB arena), so this is a compute limit, not memory | ISS |

### STM32N6570-DK (STM32N657, Cortex-M55 800 MHz + Neural-ART NPU 1 GHz, ST Edge AI Core 4.0.1)

| model | chip | engine | ms/frame | meets 16 ms? | provenance |
| --- | --- | --- | ---: | --- | --- |
| LiSenNet `conv-hardened` nc24, windowed `emit_T=64` | STM32N657 | Neural-ART NPU, `n6-allmems-O3` (0.49 MB octoFlash) | **1.15** per emitted frame (73.63 ms / 64-frame window) | **throughput yes** (RTF 0.072) — but block latency is 1.02 s, not 16 ms | on-board `validate`; no capture committed |
| NSNet2 `blockdiag_full` (re-exported) | STM32N657 | Neural-ART NPU, `n6-noextmem` (0.72 MB on-chip) | **2.13** | **YES** — RTF 0.13 | on-board `validate`; no capture committed |
| LiSenNet `conv-hardened` nc24, 17-state FIFO streaming | STM32N657 | Neural-ART NPU, `n6-noextmem` (47 KB weights + 147 KB activations, on-chip) | **2.79** | **YES** — RTF 0.174 | on-board `validate`; no capture committed |
| NSNet2 `blockdiag_8` (re-exported) | STM32N657 | Neural-ART NPU, `n6-noextmem` (0.37 MB on-chip) | 2.89 | YES — RTF 0.18 | on-board `validate`; no capture committed |
| ConvFSENet | STM32N657 | Neural-ART NPU, `n6-noextmem` (1.40 MB on-chip) | **4.40** | YES — RTF 0.275 | on-board `npu_profiler`; no capture committed |
| ConvFSENet | STM32N657 | Neural-ART NPU, `n6-allmems-O3` (weights in octoFlash) | 7.14¶ | YES — RTF 0.45, but memory-bound (27% core util) | on-board `npu_profiler`; no capture committed |
| NSNet2 `baseline` (dense) | STM32N657 | Neural-ART NPU, `n6-allmems-O3` (2.70 MB octoFlash) | **22.94** | **NO** — RTF 1.43; weights overflow on-chip npuRAM and restream every frame | on-board `npu_profiler`; no capture committed |

All STM32N6 figures are **single-run latencies** measured against a volatile RAM firmware image (`docs/models/nsnet2.md:184-185`, `deploy/stm32n6/ONBOARD_MEASUREMENT.md:143-148`). Note also that `validate`'s "duration by sample" runs ~1 ms higher than `npu_profiler`'s pure-inference figure, so the two method columns are not exactly like-for-like (`deploy/stm32n6/ONBOARD_MEASUREMENT.md:138-142`).

§ ms derived as cycles ÷ 198 MHz. The clock is silicon-verified on both the M33 and the DSP, so the division is safe, but the doc prints only the cycle count and a ×-budget ratio for these rows.

¶ `deploy/stm32n6/ONBOARD_MEASUREMENT.md:90` and `docs/models/convfsenet.md:103` both give 7.14 ms for this run; `docs/models/convfsenet.md:72` prints 7.21 ms for what is described as the same preliminary single run. The 7.14 figure is the one carried by the consolidated table and the noextmem comparison.

## How to read this

Table 1 answers "how good and how big", Table 2 answers "how fast, and does it fit". They are joined on the model name, and they do **not** multiply: a model with a good PESQ row and no Table 2 row has simply never been put on a chip.

Provenance is load-bearing, and `deploy/rt595/results/PROVENANCE.md:7-11` defines the three classes for that target: **SILICON** (read off a board, raw capture committed under `results/`), **ISS** (Cadence Xtensa instruction-set simulator), **MODELLED** (datasheet constants, no instrument). Two things follow.

First, exactly one row in this entire matrix — the M33 `blockdiag_full` run — has a committed raw capture; `deploy/rt595/results/` holds only that file and `PROVENANCE.md`. The HiFi4 7.41 ms figure was obtained on silicon over pure SWD but its stdout was not retained, and `deploy/rt595/results/PROVENANCE.md:35-40` flags that rerun as the single largest evidence gap in the target. The STM32N6 rows were likewise taken on the board (`validate` / `npu_profiler` against the STM32N6570-DK) with no capture committed anywhere under `deploy/stm32n6/` — and the RT595 provenance scheme does not formally extend to that target, so they are labelled by measurement method rather than borrowing the SILICON class.

Second, **ISS** is exact to 0.1% against silicon *for an image that fits DSP local SRAM*, which is why the fit column and the timing column have to be read together: two of the RT595 ISS rows are for configurations that cannot be linked at all.

**MODELLED** appears nowhere in Table 2. The only modelled figures in the tree are RT595 power and energy (19–23 mW, 0.14–0.17 mJ/frame), computed from NXP AN13657 and the RT500 datasheet with no ammeter attached (`deploy/rt595/results/PROVENANCE.md:52-57`, `deploy/rt595/POWER.md:97-99`), and they are deliberately not given a column here.

Two comparisons that look available but are not. The `--mem_model` column in `deploy/rt595/BENCHMARKS.md` is a sim-LSP artifact, resolved on silicon, and is excluded entirely (`deploy/rt595/BENCHMARKS.md:162-170`). Host-CPU RTF figures (onnxruntime, single thread) appear throughout the model docs but no host row names its CPU, and `docs/models/nsnet2.md:66-68` states outright that the block-diagonal and butterfly rows were timed on different machines and are not comparable — so no host-CPU rows appear in Table 2.

Two quality figures disagree with themselves, and the disagreements are worth carrying forward rather than papering over. `blockdiag_full` int8 PESQ is **2.843** in the headline table, in `benchmarks/baselines.json`, and in the RT595 table, but **2.848** in the STM32N6 section of the same file (`docs/models/nsnet2.md:146`). LiSenNet `conv-hardened` int8 is **2.982** in the harness and **2.998** in `docs/models/lisennet.md:208`. Both are instances of the same cause: `docs/studies/cross-family-metrics.md:55-63` reports that FP32 reproduces 13/13 but int8 differs for three models — `wide_blockdiag`, `blockdiag_full` and LiSenNet `conv-hardened` — because the graphs published to the Hub were quantized from a different calibration draw than the ones that produced the table rows. Table 1 carries the harness value first and names the alternate.

## Which cells are empty, and why

**Never run** (the experiment is possible; nobody has done it):

- **BASENet on any chip.** ONNX export of the *streaming* graph is explicitly not yet done (`docs/models/basenet.md:200`), so there is nothing to compile for either target. The deployment pipeline that does exist was built against the pre-re-derivation causal checkpoint (PESQ 3.116), not the current best.
- **`monarch_40` on any chip.** It exists only as host PESQ/RTF on an un-merged branch. No export, no target run, and it is not in the metric harness.
- **ConvFSENet windowed-256 on the STM32N6.** Host PESQ is done and clears the ≥2.85 gate at 2.913; on-board latency is the pending Gate-0/Phase-4 verdict that has to run on the deploy box (`docs/models/convfsenet.md:149-153`, `deploy/stm32n6/WINDOWED_DEPLOY_HANDOFF.md:120-138`).
- **LiSenNet relu6-deep on the STM32N6.** Both the windowed pure-int8 and the decoder-FP32 hybrid are queued for the deploy box; the hybrid's 118 unquantized QDQ decoder nodes are an open codegen question (`docs/targets/stm32n6-lisennet-npu.md:155-158`). The **N6 streaming (16 ms-hop) ONNX export** of relu6-deep has not been made either (`docs/targets/stm32n6-lisennet-npu.md:159-161`) — note this is a distinct pipeline from the RT595 TFLite streaming export (`host_out/relu6deep_streaming_int8.tflite`) that the RT595 ISS row above measures.
- **The windowed (stateless) graphs on the RT595 HiFi4.** Named as the untried lever in `deploy/rt595/BENCHMARKS.md:203-205` — it is the published STM32N6 deploy path and has never been measured on the DSP.

**Does not fit / does not map** (the answer is known and it is no):

- **NSNet2 dense on the RT595 HiFi4.** 2.89 MB of int8 weights against a 1.24 MB DSP data segment. The ISS cycle count is listed for shape only; it is not a realizable configuration.
- **ConvFSENet on the RT595 HiFi4.** 1.71 MB against the same 1.24 MB segment. Its cycle count is *inside* budget; the blocker is purely memory.
- **NSNet2 `monarch_full` on the RT595 HiFi4.** Not a size problem in the weights but in the arena: 1,169 KB of materialised intermediates, 69× block-diagonal's 17 KB. Both Monarch variants also emit a fixed 24-`TRANSPOSE` tax that `nblocks` cannot tune away — the two variants emit byte-for-byte the same op histogram (`deploy/rt595/BENCHMARKS.md:128-136`).
- **NSNet2 dense on the STM32N6** fits in the weaker sense — it compiles and runs, but 2.70 MB overflows on-chip npuRAM, so it restreams from octoFlash every frame and lands at RTF 1.43. Present in Table 2, marked NO.
- **LiSenNet `gru` on the Neural-ART.** Two hard stedgeai blockers — the 2-axis `nn.LayerNorm` and the `(b,t,f,d)→(b·t,f,d)` GRU reshape (`docs/models/lisennet.md:134-139`). This is why the conv/hardened variants exist at all.
- **NSNet2 butterfly variants on the Neural-ART.** NPU-hostile 6-D reshape/reduce; does not compile (`deploy/stm32n6/ONBOARD_MEASUREMENT.md:132-137`).
- **`wide_blockdiag` on the STM32N6.** Holds its int8 PESQ but at 2.36 M / 9.5 MB int8 it would not fit on-chip (`docs/models/nsnet2.md:185-187`).

**Withdrawn** (a number existed and was retracted, which is not the same as either of the above): a `dsp_offload` "13.12× speedup / 8080 µs" block, and a 9.8 KB dense arena figure. Both are listed in `deploy/rt595/results/PROVENANCE.md` as unsupported by any artifact.

**Measured but unretained** — a third category, and the one most easily confused with the above. A LiSenNet nc24 Cortex-M33 figure of ~21.2 M cycles/frame (~107 ms, 6.7× over budget) and the HiFi4 `blockdiag_full` 7.41 ms headline were both *observed on silicon*; neither console capture was saved into the repo. They are real measurements missing their evidence, not retracted claims, and both are reproducible from committed artifacts — see `deploy/rt595/results/PROVENANCE.md`.
