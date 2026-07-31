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
