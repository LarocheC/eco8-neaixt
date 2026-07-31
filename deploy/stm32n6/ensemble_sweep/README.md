# Ensemble-fold sweep — K members in one streaming graph (2026-07-31)

> **Measured on-board same day — see "On-board measurements" below.** The
> compile-time estimates under-predicted: SW fallbacks, not epoch count,
> dominate the folds' added cost. Both fallback causes are identified and
> fixable; the HW/hybrid trunk scaled as predicted.

**Question.** The N6 streaming graphs are launch-bound (epoch count prices the
frame, not MACs), and the deployed models idle 70–83 % of the 16 ms budget.
Can K ensemble members be folded into the channel dimension — grouped convs,
one graph, one set of FIFO ops — for much less than K× latency? Members would
differ by quantization-rounding draws (chasing the int8→FP32 gap, e.g. the
undeployable FP32-decoder's +0.038 PESQ) or independently trained seeds; naive
weight jitter provably averages back to the single model.

**Method.** Offline `stedgeai generate` (v4.0.1, `n6-noextmem@user_neuralart.json`
— fully on-chip), no board needed for epoch mapping; latency estimated from a
regression on the six on-board-measured streaming variants
(`paper/data/compile_facts.json` + board ms; epochs-only R²=0.972,
epochs+acts R²=0.994, max residual 0.11–0.29 ms). Source graph: the local
nc20-shaped streaming QDQ export (`cp_lisennet_conv_hardened/
g_best_streaming_int8_signed.onnx`), measured on-board at **2.59 ms / 137
epochs**. Latency and epoch mapping are weight-independent (established by the
July-14 sweep), so identical member copies are a valid latency proxy for any
real ensemble (rounding draws / seeds change only weight values).

Pipeline: `sanitize_pads.py` (the stored export predates the Pad empty-input
export fix — atonn segfaults without it) → `fold_channels.py K` (grouped fold)
or `duplicate_branches.py K` (worst-case route) → generate → `parse_fit.py`.

## Results

**Grouped-conv probe** (`build_group_probe.py`): dense, group=2, group=4 and
depthwise QDQ convs all map to **pure HW epochs** (4 epochs, 0 SW/hybrid).
Grouped convs are natively supported; the fold does not hinge on a compiler gap.

| graph | epochs (HW/hyb/SW) | vs K=1 | MACC¹ | weights | acts | est. ms/frame² | RTF |
|---|---|---|---|---|---|---|---|
| K=1 baseline | 139 (82/51/6) | 1.00× | 1.84 M | 35.8 kB | 122 kB | 2.59 (measured 137-ep twin) | 0.16 |
| **K=2 fold** | **172 (88/60/24)** | **1.24×** | 3.68 M | 87 kB | 241 kB | **~3.5–4.3** | **0.22–0.27** |
| **K=3 fold** | **197 (91/73/33)** | **1.42×** | 5.52 M | 131 kB | 357 kB | **~4.2–5.1** | **0.26–0.32** |
| K=2 branches | 234 (118/104/12) | 1.68× | 3.69 M | 72 kB | 187 kB | ~4.7–6.3 | 0.29–0.39 |
| K=3 branches | 345 (172/155/18) | 2.48× | 5.53 M | 108 kB | 271 kB | ~7.3–9.8³ | 0.46–0.61 |

¹ report-line MACC (unit differs ~2× from compile_facts convention; ratios are
exact — the fold is ×K MACs by construction).
² fit band ∪ ratio-anchored to the measured 2.59 ms baseline.
³ extrapolation beyond the fitted epoch range (131–201); wider grain of salt.

**Findings.**
- **The fold is strongly sublinear: 2 members cost +24 % epochs, 3 members
  +42 %.** The conv trunk folds almost free (HW epochs 82→88 at K=2); the FIFO
  state plumbing is shared by construction.
- Of the fold's +33 epochs at K=2, most are the **decoder ConvTranspose
  expansion** (grouped ConvTranspose crashes atonn's expansion pass with
  garbage dequant scales, so each of the 3 ConvTransposes becomes per-member
  Slice → CT → Concat; SW epochs 6→24). A production export could restructure
  this (or accept it — it is per-member exact).
- Everything fits on-chip trivially at K=3 (488 kB total vs the noextmem
  pools); compile success on the noextmem profile *is* the fit proof.
- Even the zero-cleverness K-branch route streams in real time at K=3.

**Caveats / not yet done.**
- No board numbers — estimates come from the 6-point fit. `n6_loader` +
  `validate --mode target` on the DK is the real verdict.
- The fold is a **latency proxy**: identical members, and the GLU channel
  splits (shape-derived, dynamic) interleave members numerically. A quality
  ensemble needs per-member GLU (split fc1 into two grouped convs) and
  per-member decoder skips — structural cost ≈ what is measured here.
- Baseline compiles to 139 epochs vs the sweep's 137 (stored export differs
  cosmetically from the fresh July-14 exports).

## On-board measurements (2026-07-31, STM32N6570-DK, npu_profiler)

`stedgeai validate --mode target` works for the baseline but not the ensemble
graphs (folds: host-side `INTERNAL ERROR: 'NoneType' object is not
subscriptable` in the import pass; branches: `E801 Invalid firmware` at
runtime bind — the validation app rejects the 35-in/36-out IO table).
`npu_profiler` binds the folds fine (18/18 IO) and is the measurement path;
branch graphs hit the same E801 there and remain unmeasured. Fold profiling
hangs at `-b 16` (telemetry volume); `-b 1`/`-b 4` work, and baseline b=1 ==
b=16 (2.598 vs 2.598), so single-shot numbers are clean.

| graph | measured ms/frame | RTF | est. band | HW | HYBRID | SW |
|---|---:|---:|---|---:|---:|---:|
| K=1 baseline | **2.598** (σ .002, b=16; validate: 2.599) | 0.16 | 2.6–3.2 ✓ | 1.412 | 0.445 | 0.275 |
| K=2 fold | **7.387** (σ .003, b=4) | 0.46 | 3.5–4.3 ✗ | 1.980 | 0.712 | **4.057** |
| K=3 fold | **10.584** (b=1) | 0.66 | 4.2–5.1 ✗ | 2.592 | 0.958 | **6.025** |

(Per-epoch sums; wall total adds ~0.5–1.0 ms inter-epoch overhead. Raw
tables: `prof_*.log`; decomposition: `analyze_prof.py`.)

**Where the miss went — all of it is SW fallback, and it's identified:**
- **HW+HYBRID scaled as the epoch model predicted**: 1.86 → 2.69 → 3.55 ms
  for 1×/2×/3× members. The grouped trunk thesis holds on silicon.
- **Per-member ConvTranspose expansion fell to SW** (~2.79 ms @K=2, ~4.19 @K=3):
  epochs {158,159}, {175,176}, {193,194} are `ConvTranspose_*_expanded_conv_*`
  at 0.42–0.53 ms each, software convs on the M55. In the baseline the same
  expanded CTs map HW — the member-Slice→CT→Concat wrapper broke the mapping.
- **Three encoder grouped convs with 2-D kernels fell to SW** (~0.9 ms @K=2,
  growing with K): `Conv2D_45/61/…` = the (2,3)/(2,5)-kernel strided encoder
  convs. Grouped **1×1** maps HW (probe), grouped **2-D kernels do not**.
- Both have the same fix: represent exactly these ops **block-diagonal dense**
  (no `group` attr — shape-identical to a normal conv/CT, so HW mapping is
  preserved; their MACs are tiny, so the ×K² is irrelevant). Projected K=2
  ≈ trunk 2.69 + baseline-like SW ~0.3 + overhead ~0.7 ≈ **~3.7–4.3 ms** —
  back in the original band. Unverified until compiled + re-measured.

**Board ops note:** killing n6_loader mid-"Loading memories" wedges the
ST-LINK (persistent `Loading memories failed`, then `DEV_USB_COMM_ERR` at the
probe). Software recovery that worked, no power-cycle needed:
`usbipd.exe detach` → `unbind` → `bind` → `attach --wsl` → verify with
`STM32_Programmer_CLI -c port=SWD mode=UR`. Don't interrupt loads.

## Block-diagonal-dense rebuild (v2) — measured same day, fix confirmed

`fold_channels.py` v2 rule: 1×1 pointwise and depthwise convs stay **grouped**
(HW-proven); every other Conv and all ConvTranspose become **block-diagonal
dense** (members on the weight diagonal, no `group` attr — standard shapes,
so the HW mapping survives; zero blocks quantize exactly; the ×K² MACs are
negligible at these sizes). 28 grouped + 13 bd-dense convs.

| graph | epochs (HW/hyb/SW) | measured ms/frame | vs K=1 | RTF | HW / HYB / SW (ms) |
|---|---|---:|---:|---:|---|
| K=1 baseline | 139 (82/51/6) | 2.598 | 1.00× | 0.16 | 1.41 / 0.45 / 0.28 |
| **K=2 bd-fold** | **144 (88/50/6)** | **3.762** (σ .004) | **1.45×** | **0.24** | 2.04 / 0.66 / 0.44 |
| **K=3 bd-fold** | **149 (91/52/6)** | **5.312** (σ .006) | **2.04×** | **0.33** | 2.69 / 0.90 / 0.85 |
| (v1 K=2, grouped-everything) | 172 (88/60/24) | 7.387 | 2.84× | 0.46 | 1.98 / 0.71 / 4.06 |

- SW epochs back to the baseline's **6**; SW time 4.06 → **0.44 ms** at K=2;
  no epoch above 0.11 ms. HW scales ~+0.63 ms per member, linear.
- Epoch count is now nearly K-invariant (139 → 144 → 149): the fold thesis —
  members share the launch/state floor — holds end-to-end on silicon.
- K=2 lands **inside the original estimate band** (3.5–4.3); K=3 lands just
  above its band (4.2–5.1) on the replicated-prologue SW growth.
- Remaining shavable cost: `feat` is replicated **before** the FP32 prologue,
  so those SW ops run ×K (0.28 → 0.44 → 0.85 ms). Replicating after the
  prologue/quant boundary would return most of it.

**Bottom line measured:** a 2-member ensemble costs +45 % latency, a 3-member
ensemble +104 %, both fully on-chip and far under the 16 ms budget. Projected
relu6-deep K=2 (per-member add ≈ 1.2–2 ms on a 4.83 ms base): ~6–7 ms,
RTF ≤ 0.45.

## Quality: SR-draw ensembling recovers 37 % of the PTQ gap (2026-07-31)

Host-side, relu6-deep, Table-1 protocol (full-utterance static-int8 graph +
noisy phase; the `eval_conv_rt_int8.py` harness — its pc_signed anchor 3.0268
reproduced exactly). Members share the deployed quantization grid (all scales,
zero-points, activation Q/DQ fixed); only the weight-rounding realization
differs (stochastic-rounding draws); members' `est_mag` are averaged. Paired
per-utterance PESQ, 824 utts. Script: `sr_decoder_ensemble.py` + results JSON
at repo root.

| config | PESQ | Δ vs RTN (±1.96·se) |
|---|---:|---|
| RTN (deployed pc_signed) | 3.0268 | — |
| FP32 | 3.0682 | +0.0413 ± 0.0057 |
| decoder weights exact (acts int8) | 3.0215 | −0.0053 ± 0.0015 |
| all weights exact (acts int8) | 3.0140 | −0.0128 ± 0.0044 |
| SR all-weights, K=1 | 3.0282 | +0.0014 ± 0.0051 |
| SR all-weights, K=2 | 3.0365 | +0.0097 ± 0.0028 |
| **SR all-weights, K=4** | **3.0421** | **+0.0153 ± 0.0027** |
| SR all-weights, K=8 | 3.0423 | +0.0155 ± 0.0024 |
| SR decoder-only, K=4 | 3.0318 | +0.0050 ± 0.0013 |

Three findings, in the order they falsified each other:
1. **Weight rounding is not the gap.** Exact weights under the fixed
   activation grid *lose* (−0.005 decoder / −0.013 all): the MinMax activation
   scales were calibrated with RTN weights in place — grid and rounding are
   co-adapted. The int8 gap is entirely activation-side.
2. **Yet SR-draw ensembles win** — +0.010 at K=2, +0.015 at K=4, saturated by
   K=8 (37 % of the gap). The naive cap argument (ensemble → exact-weight net)
   fails because it holds activation rounding fixed: in reality the weight
   draws **dither the activation quantizers** — each member's
   activation-rounding errors decorrelate, and the est_mag average
   Monte-Carlos them away. The non-recoverable rest is shared across members
   (the input-feature quantization is identical) plus the co-adaptation
   deficit.
3. **Scope scales the effect**: dithering just the 7 decoder weight tensors
   recovers a third (+0.005) of the whole-net benefit.

**Deployment mapping** (latency already measured above, weight-independent):
whole-net K=2 bd-fold = 3.76 ms → projected streamed relu6-deep ≈ 3.013+0.010;
K=4 ≈ 7 ms (extrapolated) → ≈ +0.015. No retraining, no calibration change —
the members are free variations of the shipped weights.

## Uncertainty / OOD / adaptive-K from member disagreement (2026-07-31)

The fold computes all member masks side by side, so the variance across
members is nearly free on-device. Measured (824 ID utts; 200/condition OOD;
`sr_uncertainty.py`, `sr_adaptive_frames.py` + results JSONs at repo root):

**Per-frame quantization-noise meter — validated.** Frame-level Spearman of
member disagreement D vs squared est_mag error vs FP32: **0.787** pooled,
0.751 per-utt median, **0.530 within energy deciles** (not a loudness proxy).
Utterance-level inversion (informative): mean-D vs residual PESQ damage is
*negative* (−0.344) — D measures the *removable* noise; residual damage
concentrates where D is low (the shared, non-averageable part, chiefly the
common input-feature quantization).

**OOD (AUROC of utterance-mean D, ID vs corrupted):**
| condition | AUROC (D) | AUROC (input clip-rate) |
|---|---:|---:|
| gain +12 dB | 0.986 | 1.000 |
| unseen white noise @5 dB | **1.000** | 0.528 (blind) |
| gain −12 dB | 0.393 (variance collapse) | 0.435 (blind) |

D detects spectral novelty that envelope statistics cannot (white noise), and
"too hot" inputs; it cannot see "too quiet" (disagreement mildly *drops*) —
deploy it paired with a two-sided input-level check.

**Adaptive-K works at frame granularity only.** Utterance-level escalation by
D2 (the deployed pair's own disagreement) is useless (ρ = 0.057 vs escalation
gain; curve ≈ linear). Frame-level mask mixing (K=4 mask on the top-X% D2
frames, K=2 elsewhere), with a random-frame control:

| escalated frames | members avg | ΔPESQ vs K=2 | random control |
|---:|---:|---:|---:|
| 10 % | 2.2 | +0.0017 (30 % of K=4 gain) | +0.0006 |
| 25 % | 2.5 | +0.0035 (63 %) | +0.0006 |
| **50 %** | **3.0** | **+0.0054 (96 %)** | +0.0025 |
| 100 % | 4.0 | +0.0056 | — |

Caveat: offline full-utterance graphs = the *optimistic ceiling*; a streaming
deployment must keep escalated members' FIFO states warm (e.g. copy states
from the always-on pair — members differ only at rounding grain, so the
approximation should be mild; untested).

**Next steps.** (1) Member-correct fold for on-chip quality: per-member GLU
(split fc1 into two grouped convs) + per-member decoder skips — the current
bd-fold interleaves members through the GLU dynamic split, fine for latency,
wrong for non-identical members. Then board-validate a K=2/K=4 SR fold and
host-eval its streamed PESQ. (2) Streaming frame-escalation with state-copy
(tests the adaptive ceiling's deployability). (3) Optional: per-member
activation recalibration; relu6-deep bd-fold once trained checkpoints are
copied over; replicate feat after the prologue.

## Member-correct fold (v3) + streamed SR quality (2026-07-31, training box)

Continuation on the training box (the deep/relu6 checkpoints live here; the
DK + stedgeai stay on the laptop). Two deliverables: the member-correct fold
the previous section called for, and the *streamed* SR-ensemble quality it
unlocks — measured on relu6-deep's own streaming graph, not projected from the
offline study. Everything below is host-side; board latency of the new graphs
is the one piece still owed by the laptop.

### relu6-deep streaming graphs (new exports, tracked)

`lisennet/export_onnx.py --streaming` on
`cp_lisennet_conv_hardened_nc24_deep_relu6/g_best` → feat + **25** FIFO states
(889 nodes, 57 Conv / 3 ConvTranspose / 3 GLU chunk chains; the exporter's Pad
fix is built in, so no `sanitize_pads.py` step). Quantized with the deploy
recipe (`quant_onnx.py --mode static --streaming`: signed per-channel QDQ,
state-threaded percentile calibration). Tracked as
`paper/data/tmp_quant/relu6deep_streaming_{fp32,int8_signed}.onnx` (+ a
MinMax-calibrated sibling `..._int8_minmax.onnx`, below).

Sanity anchors: ORT streaming FP32 == offline torch frame-exactly (max diff
9.5e-7 over an utterance) and its full-824 PESQ **3.0682** equals the offline
FP32 number to 4 decimals. The regenerated percentile int8 grid scores
**3.0117** vs the 3.0132 recorded in `paper/data/stream_pesq.json` (r6) — the
grid is calibration-pipeline-identical but library-drift-sensitive at the
±0.0015 level; all deltas below are paired against *this* graph's grid.

### fold_members.py — K distinct members, bit-exact

`fold_channels.py` (v2) is a latency proxy: identical members, interleaved
through the GLU chunk and the skip concats. `fold_members.py` (v3) folds K
*different* member graphs (same structure + scales, different int8 weights —
the SR protocol) with per-member semantics:

- **GLU**: each `fc1` (1x1, emb→2h) becomes TWO grouped 1x1 convs — rows
  [0:h] / [h:2h] of every member's weight, group=K — and the traced
  `Shape→Gather→Add→Div→Slice` chunk chain is deleted. The two halves'
  existing quantizers are rewired onto the new convs; they reuse fc1's output
  scale, so requantization is idempotent and the member arithmetic unchanged.
- **Decoder skips**: `cat([x, enc], 1)` folds to segment-major member layout;
  the consuming convs are bd-dense (v2 rule), so member blocks are *placed* at
  segment-local offsets in the dense weight (permuted block layout, exact
  zeros elsewhere) — standard shapes, zero extra nodes. Segment widths come
  from ONNX shape inference on the member graph, not name heuristics.
- **Tail**: the `apply_mask` channel Gathers become stride-2 Slices (members'
  mask[0::2], mask[1::2]) and a keep-dims Slice for the noisy mag; `est_mag`
  widens to **(B, K, 1, F)** — one enhanced magnitude per member, so the mean
  *and* the disagreement D are one axis-op away for the caller.
- **Head**: `conv_1` is 1x1 on the 3 shared feat channels → members stack on
  its *out* axis only. No `feat` replication, no head Concat — the v2
  "replicate feat before the FP32 prologue" SW cost (0.28→0.44→0.85 ms with K)
  is deleted structurally, closing that open thread.

Op census is K-invariant (60 Conv + 3 CT + 68 Slice; no Gather/Shape/Div
left), IO stays 26-in/26-out at every K, no empty-Pad atonn blocker.

**Equivalence proof** (`--check`, plus a real-utterance run): fold vs the K
separate member sessions, streaming with propagated state —
est_mag and all 25 states **bit-exact (100.000 % of elements, worst diff
0.0)** over 40 random frames and over a full real utterance (391 frames × 4
members). The shared-scale Q/DQ chains snap identically, so the fold *is* the
ensemble, not an approximation of it.

### Streamed SR quality — measured, 824 utts (`sr_stream_ensemble.py`)

Frame-by-frame FIFO streaming (per-member state), members' est_mag averaged
per frame, noisy phase, paired vs the streamed RTN. Two activation grids on
the same weights: the deploy percentile grid, and a MinMax grid (only the
calibration method differs).

| config | PESQ | Δ vs same-grid RTN |
|---|---:|---|
| **percentile grid** (deployed recipe) | | |
| RTN | 3.0117 | — |
| FP32 (grid-free cap) | 3.0682 | +0.0565 ± 0.0054 |
| all weights exact | 3.0041 | −0.0077 ± 0.0071 (co-calibration, again) |
| SR K=1 | 3.0051 | −0.0067 ± 0.0065 |
| SR K=2 | 3.0132 | +0.0015 ± 0.0063 |
| SR K=4 | 3.0217 | +0.0100 ± 0.0051 |
| SR K=8 | 3.0240 | +0.0123 (22 % of the gap) |
| **MinMax grid** (same weights, same protocol) | | |
| RTN | **3.0354** | — (+0.0237 over percentile RTN!) |
| SR K=2 | 3.0418 | +0.0064 (20 % of its 0.0328 gap) |
| SR K=4 | **3.0523** | **+0.0169 (52 % of its gap)** |

Readings, in causal order:

1. **The offline recovery does not transfer as-is to the deployed streaming
   grid.** Offline: K=4 = +0.0153, saturated by K=8, 37 % of a 0.0413 gap.
   Streamed percentile: a *single* SR draw now hurts (−0.0067 vs +0.0014
   offline), K=2 barely breaks even, and the curve is still climbing at K=8
   (+0.0123, 22 % of a larger 0.0565 gap) — the same mechanism, lagged by
   roughly one octave of K.
2. **The lag is the percentile calibration, not state recirculation.** The
   streaming recipe clips activation tails (percentile 99.999); clipping error
   is deterministic and SR draws cannot dither it away. On a MinMax grid the
   same draws recover offline-like fractions at K=2. The recirculation
   hypothesis (member state trajectories decorrelating badly) is not needed.
3. **The calibration method itself is worth more than the ensemble.** MinMax
   RTN beats the deployed percentile RTN by **+0.0237** on this graph — a free
   recalibration, no weight change. MinMax + SR K=2 (3.0418) ≈ percentile
   K=8 + 0.018.
4. **On the MinMax grid the ensemble over-performs the offline study**: K=4
   recovers 52 % of its gap (offline 37 %). Combined, recalibration + K=4 SR
   = **+0.0405 over the deployed grid** — 72 % of the streamed PTQ gap closed,
   0.016 short of FP32. Fold-vs-separate PESQ identity checked: 3.1276 ==
   3.1276 on the 64-utt subset (bit-exact upstream, so equality is exact).

### Deployment picture (updated)

Ship candidate: **recalibrate the streaming graph MinMax, fold K SR members
member-correct**. Host-measured: MinMax K=2 = 3.0418, K=4 = 3.0523 (vs
deployed 3.0117 baseline). Latency is weight- and scale-independent (July-14 result),
so the v2 bd-fold numbers carry: K=2 ≈ +45 %, K=4 ≈ extrapolated ~7 ms on the
nc20-shaped graph — relu6-deep's own K-curve is exactly what the laptop still
needs to measure. The (B,K,1,F) est_mag also hands the app per-member masks,
so the UQ/adaptive results apply unchanged on-device.

### Artifacts + the laptop's TODO

Tracked in `paper/data/tmp_quant/ens_graphs/`:
`ens_memb_k{1,2,4}.onnx` (percentile grid) and `ens_memb_mm_k{2,4}.onnx`
(MinMax grid; the quality candidates). `ens_memb_k1` is the restructured K=1
twin — same arithmetic as the deployed RTN graph (bit-exact, checked), new op
structure — so **K=1-twin vs baseline is the A/B that isolates the
fc1-split/tail-slice cost**. SR members regenerate deterministically
(`sr_stream_ensemble.py` builds them on demand; seeds 0..K−1, CRC-per-tensor).

Laptop steps (per `profile_one.sh`, adapt `$SP`):
1. `stedgeai generate` (4.0.1, `n6-noextmem@user_neuralart.json`) on the five
   graphs + `relu6deep_streaming_int8_signed.onnx` as the true baseline.
2. `n6_loader` + `npu_profiler -b 1`/`-b 4` (never `-b 16` on folds; never
   kill mid-load). Risk: these are 26-in/26-out — npu_profiler bound 18/18
   fine, the E801 threshold is unknown; if it rejects, profile the K=1 twin
   first to size the IO problem before touching the folds.
3. Expected: epochs ≈ v2 bd-fold ± a few (+3 grouped fc1 convs, −3 chunk
   chains, −1 head Concat, tail Slices for Gathers); the K=1-twin delta
   prices the restructure.

Open threads after this session: board latency of the v3 folds (above);
streaming frame-escalation with state-copy (the K=4 fold + a state copy
between member slices is now buildable); seed-member folds — note
`fold_members.py` asserts shared activation scales, so genuinely different
seeds need a joint recalibration pass first; layer-resolved D unchanged.

## Acting on D: do-no-harm gating + downstream ASR gating (2026-07-31, cont.)

The "plausible, cheap, untested" pair from the follow-up list, both run on the
deployment artifact itself (`ens_memb_mm_k4.onnx` — its (B,K,1,F) est_mag
gives the member mean and the disagreement D per frame for free). Thresholds
in both experiments are FROZEN from the ID pool (deployment constants), and
both are evaluated on ID (824) plus the UQ study's OOD recipes (200 each):
`white5db` (D fires, AUROC 1.0), `gain-12` (D provably blind — negative
control), `gain+12`.

### E1 — do-no-harm gating (`sr_gating.py`, `sr_gating_results.json`)

Gate: `est' = (1−a)·est_avg + a·src_mag` with `a` from per-frame D_t
(= var_K est_mag, mean over F) or per-bin D_tf; linear and threshold forms;
θ ∈ {p75, p90}(ID); a_max ∈ {0.25, 0.5, 1.0}. Primary read-outs are tails
and harmed-utterance counts vs noisy — the study predicted (correctly) that
means would not reward it.

| pool | ungated vs noisy | best gate | what gating does |
|---|---|---|---|
| id | 3.0523 vs 1.9707; **harmed 5/824**, p1 +0.176 | bin-lin_p90_a0.25 | pays −0.071 mean, saves 2 of 5 victims; DNSMOS BAK −0.16 with NO SIG gain. **Not worth it in-distribution** — there is nobody to insure. Aggressive frame gates *create* harm (up to 66/824). |
| white5db | 1.6265 vs 1.0657; harmed **0/200**, p1 +0.327 | (none) | D is 6.8× ID — but the enhancer handles unseen *stationary* noise fine. **D flags novelty, not failure**; every gate only returns noise (−0.17 mean at the gentlest). Full bypass ≈ noisy (harmed 181/200). |
| gain-12 | 2.2475 vs 2.1620; **harmed 85/200**, p1 −1.299 | bin-thr_p90_a0.25 | the pool that actually needs insurance — and frame-D is blind (0.83× ID, variance collapse). **Per-bin D partially survives the collapse**: harmed 85→47, mean **+0.178**. A two-sided input-level check remains the robust fix. |
| gain+12 | 2.6625 vs 2.1620; harmed 5/200, p1 −0.135 | bin-thr_p90_a0.25 | the clean win: mean-neutral (+0.011), tail flips positive (p1 → **+0.027**), harmed 5→2. |

Verdict: D-gating is **not** a general do-no-harm mechanism. It is worth
having exactly where disagreement coincides with real failure — level-shifted
inputs, where the gentle per-bin threshold gate is mean-neutral-to-positive
and fixes the tail — and it is counterproductive where disagreement means
novelty-the-net-handles (white noise). The monitoring framing ("conditions
changed" flag → telemetry / recalibration trigger) survives intact; the
automatic per-frame insurance mostly does not. If one gate ships, it is
`bin-thr, θ=p90(ID), a_max=0.25` — per-bin, gentle, threshold — plus the
level check.

### E2 — D as a bypass gate for a downstream ASR (`sr_asr_gate.py`)

The product-shaped test: the enhancer feeds a recognizer (whisper-tiny.en,
greedy), and the gate hands the recognizer the RAW input when uncertain.
Differential WER — references are whisper's transcriptions of the *clean*
audio, a protocol that structurally *favors* the enhanced signal (it is
closer to clean by construction), so measured enhancement damage is a lower
bound. Policies frozen on ID: bypass if utt-mean D > p{90,95,99}(ID), and a
two-sided **feature-quantizer-domain** level check (waveform RMS is vacuous —
the data pipeline RMS-normalizes; the deployable signals are the input-mag
clip-rate `q>127` for hot and mean |q−zp| for quiet, both running means of
values the device already computes).

| pool | noisy | enh | D_p95 | level | D_p95+level | oracle |
|---|---:|---:|---:|---:|---:|---:|
| id | 0.084 | 0.099 | 0.096 @5 % | 0.099 @1 % | 0.095 @6 % | 0.056 |
| white5db | 0.218 | **0.381** | **0.218 @100 %** | 0.381 @0 % | **0.218 @100 %** | 0.180 |
| gain−12 | 0.041 | 0.119 | 0.093 @8 % | 0.090 @86 % | **0.072 @90 %** | 0.037 |
| gain+12 | 0.102 | 0.106 | 0.101 @88 % | 0.102 @100 % | 0.102 @100 % | 0.074 |

Findings:

1. **Enhancement hurts this recognizer in every pool** — ID +1.5 WER pts,
   white noise +16.3, quiet +7.8 — while helping PESQ/DNSMOS in almost all
   of them (white5db: +0.56 PESQ, +1.0 DNSMOS OVRL, and +16 WER!). Speech
   enhancement tuned for ears is a net-negative front-end for a noise-robust
   ASR; PESQ and the machine consumer *disagree violently* off-distribution.
2. **The D-gate converts the catastrophic case into a safe one.** On white
   noise D fires on 100 % of utterances (level: 0 % — it is spectral novelty,
   not level) and the bypass recovers the full +16.3 points. This is the
   confidence-channel story landing exactly as pitched.
3. **The two signals are exactly complementary**: D covers novelty
   (white5db 100 %/level 0 %), the quantizer-domain level check covers
   gain−12 (86 %/D 8 %) where D is variance-collapsed. Combined: quiet-pool
   damage 0.119 → 0.072 (60 % recovered; a stricter quiet threshold trades
   ID false-bypass for more).
4. **Per-utterance selection inside ID is still a null** (ρ(D, damage) =
   +0.04; oracle 0.056 vs achieved 0.095) — consistent with the study's
   utterance-level result. D routes *distribution shifts*, not individual ID
   utterances.
5. Deployment reading: if the output feeds only a robust ASR, bypass always.
   If one stream feeds ears *and* recognizer — the actual N6 pipeline shape —
   `D_p95 + level` keeps the recognizer within 1 pt of raw everywhere while
   the listener keeps the enhancement wins, for a few scalar ops per frame.
