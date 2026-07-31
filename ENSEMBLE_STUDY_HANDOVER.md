# Idle-cycle ensemble study — handover (2026-07-31)

**Status: parked for a future study.** The first day's work was measured on
the WSL2 laptop + the STM32N6570-DK; a same-day continuation on the training
box added the **member-correct fold (bit-exact) and the measured streamed SR
quality** (see "Continuation" below and the lab log). This branch
(`ensemble-study`) carries the scripts, results, and this doc so the work can
be picked up on any machine. The detailed lab log with all tables lives in
[`deploy/stm32n6/ensemble_sweep/README.md`](deploy/stm32n6/ensemble_sweep/README.md);
this doc is the orientation layer.

## The idea

The N6 streaming enhancers idle 70–83 % of the 16 ms frame budget, and frame
cost is priced by epoch count (launch/state plumbing), not MACs. So K ensemble
members can be **folded into the channel dimension** of one graph — shared
FIFO ops, shared launches — for far less than K× latency. Members that differ
by **stochastic-rounding (SR) draws of the int8 weights** turn the idle
compute into (a) PESQ recovered from the PTQ gap, (b) a free per-frame
uncertainty / OOD signal. All three claims are now *measured*, not estimated.

## Measured results (all on-board latency; quality/UQ host-side, Table-1 protocol)

**Latency (STM32N6570-DK, npu_profiler, nc20-shaped streaming graph):**

| graph | epochs (HW/hyb/SW) | ms/frame | vs K=1 | RTF |
|---|---|---:|---:|---:|
| K=1 baseline | 139 (82/51/6) | 2.598 | 1.00× | 0.16 |
| K=2 bd-fold | 144 (88/50/6) | **3.762** | 1.45× | 0.24 |
| K=3 bd-fold | 149 (91/52/6) | **5.312** | 2.04× | 0.33 |

Winning fold rule: grouped convs **only** for 1×1 pointwise + depthwise
(HW-proven); every other Conv and all ConvTranspose as **block-diagonal
dense** (standard shapes keep the HW mapping; grouped 2-D-kernel convs and
any grouped/expanded ConvTranspose fall to SW on the M55 — measured the hard
way, v1 cost 7.39 ms).

**Quality (relu6-deep, full 824-utt VBD test, paired vs deployed RTN 3.0268):**

| config | PESQ | Δ vs RTN |
|---|---:|---|
| FP32 | 3.0682 | +0.0413 ± 0.0057 |
| all weights exact (acts int8) | 3.0140 | **−0.0128** (co-calibration!) |
| SR all-weights K=2 | 3.0365 | +0.0097 ± 0.0028 |
| SR all-weights K=4 | 3.0421 | **+0.0153 ± 0.0027** (saturates by K=8) |

Mechanism (the study's core insight): the int8 gap is ~100 % activation-side
— exact weights *lose* because activation scales are co-calibrated with RTN
weights — yet SR-draw ensembles win because the weight draws **dither the
activation quantizers**: members' activation-rounding errors decorrelate and
est_mag averaging Monte-Carlos them away. 37 % of the gap recovered, free
weights, no retraining.

**Uncertainty / OOD / adaptive (member disagreement D, nearly free on-device):**

- Per-frame D vs int8-vs-FP32 error: Spearman **0.79** (0.53 energy-controlled).
- OOD AUROC: unseen white noise **1.000** (input clip-rate blind, 0.53),
  gain+12 dB 0.986, gain−12 dB invisible (variance collapse) → pair with a
  two-sided level check.
- Adaptive-K: dead at utterance level (ρ=0.06); at **frame** level, escalating
  the top-50 % D2 frames K=2→K=4 captures **96 %** of the K=4 gain at 3.0
  members average (random-frame control: 45 %).

## Continuation (same day, training box): member-correct fold + streamed quality

- **`fold_members.py` (v3)** folds K *distinct* SR members into one streaming
  graph, **bit-exact vs K separate sessions** (est_mag + all 25 FIFO states,
  100 % of elements, over a full real utterance): per-member GLU (fc1 → two
  grouped 1x1s, chunk chain deleted), permuted-block dense weights under the
  skip concats, stride-2-Slice tail with `est_mag (B,K,1,F)` (per-member masks
  → mean *and* D on-device), and `conv_1` stacked on out-channels only — the
  feat-replication SW cost (old thread 6) is gone structurally.
- **Streamed SR quality, measured on relu6-deep's own streaming graph**
  (824 utts, `sr_stream_ensemble.py`): on the deployed percentile grid the
  offline recovery is lagged (K=1 −0.0067! K=4 +0.0100, K=8 +0.0123, still
  climbing). Attribution: **percentile tail-clipping**, which SR cannot
  dither. On a MinMax grid: RTN alone **+0.0237**, SR K=4 **+0.0405** over
  the deployed RTN (3.0117 → 3.0523; FP32 3.0682) — recalibration + fold
  closes 72 % of the streamed PTQ gap with the shipped weights.
- Board artifacts tracked for the laptop:
  `paper/data/tmp_quant/ens_graphs/ens_memb{_mm,}_k*.onnx` + streaming base
  graphs; the K=1 twin isolates the restructure cost. Laptop TODO + expected
  epochs: lab-log §"Artifacts + the laptop's TODO".
- **Acting-on-D experiments** (`sr_gating.py`, `sr_asr_gate.py` + results
  JSONs; lab-log §"Acting on D"): do-no-harm mask gating is a *negative* on
  ID (5/824 victims, every gate pays mean PESQ) and on spectrally-novel OOD
  (the enhancer handles white noise fine — D flags novelty, not failure);
  it is a real win only on level-shifted inputs (gentle per-bin gate:
  gain+12 tail flips positive; per-bin D partially survives the gain−12
  variance collapse, +0.18 mean). Downstream-ASR gating is the strong
  result: enhancement *hurts* whisper-tiny in every pool (up to **+16 WER
  points** on white noise while PESQ/DNSMOS improve!), D fires on 100 % of
  white-noise utterances (feature-domain level check: 0 % — exact
  complements) and the D+level bypass keeps the recognizer within ~1 pt of
  raw everywhere. Ship shape: enhanced stream for ears, D+level-gated bypass
  for the machine consumer.

## What's in this branch

| path | what |
|---|---|
| `deploy/stm32n6/ensemble_sweep/` | full lab log (README), fold/branch/probe/sanitize scripts, compile reports, npu_profiler logs, `sweep_results.json` |
| `deploy/stm32n6/ensemble_sweep/fold_members.py` | **v3 member-correct fold** — K *distinct* members in one graph, bit-exact vs separate sessions (per-member GLU, permuted-block skips, per-member tail, no feat replication) |
| `sr_decoder_ensemble.py` (+`_results.json`) | SR-member builder + PESQ harness; caps, K-sweep. **Run first** — it (re)builds member graphs into `paper/data/tmp_quant/sr_graphs/` deterministically |
| `sr_stream_ensemble.py` (+`_results.json`) | **streamed** SR-ensemble harness — frame-by-frame FIFO streaming graph, per-member state, `mm_` prefix for the MinMax-grid attribution configs, `memb_fold_k*` for the folded graphs |
| `sr_uncertainty.py` (+`_results.json`) | UQ correlations, OOD AUROCs, utterance-level adaptive sim |
| `sr_adaptive_frames.py` (+`_results.json`) | frame-level escalation curve + random control |
| `paper/data/tmp_quant/relu6deep_rt_int8_{fp32,pc_signed}.onnx` | the FP32/QDQ graphs the offline sr scripts consume (regenerable via `paper/data/eval_conv_rt_int8.py`, ~30 min) |
| `paper/data/tmp_quant/relu6deep_streaming_{fp32,int8_signed}.onnx` | the relu6-deep **streaming** graphs (feat + 25 FIFO states; signed percentile grid) — tracked copies; regenerate from the checkpoint with `lisennet/export_onnx.py --streaming` + `lisennet/quant_onnx.py --mode static --streaming` |
| `paper/data/tmp_quant/ens_graphs/ens_memb_k{1,2,4}.onnx` | **board artifacts**: member-correct SR folds (K=2/K=4) + the K=1 restructured twin (A/B for the fc1-split/tail cost) |
| `lisennet/eval_metrics_ext.py`, `paper/data/eval_conv_rt_int8.py` | harness dependencies (Table-1 protocol) |

## Reproducing on another machine

- Python: repo venv (`./.venv/bin/python3`); pip needs
  `PIP_INDEX_URL=https://pypi.org/simple` (the default JFrog index 401s).
- Dataset: `common.dataset.load_voicebank_demand()` pulls
  `JacobLinCool/VoiceBank-DEMAND-16k` via HF datasets (cached ~here; will
  download elsewhere). Run the harnesses with `HF_HUB_OFFLINE=1
  HF_DATASETS_OFFLINE=1` once cached — 24 joblib workers doing HF freshness
  HEAD checks get 429-rate-limited mid-eval.
- Quality/UQ scripts are host-only: `sr_decoder_ensemble.py --n 824 --configs
  rtn fp32 all_w_exact all_sr_k2 all_sr_k4` then `sr_uncertainty.py`,
  `sr_adaptive_frames.py`. ~2–4 min per single-member config, K× for ensembles.
- Board flow (N6 latency): ST Edge AI Core **4.0.1** (`~/stedgeai/install/4.0`
  on the laptop) + STM32CubeCLT 1.21 (`~/opt/st/stm32cubeclt_1.21.0`); WSL USB
  via `usbipd.exe attach --wsl --busid <ST-Link>`. Commands in
  `ensemble_sweep/README.md` + `deploy/stm32n6/ONBOARD_MEASUREMENT.md`. The
  fold graphs regenerate from the tracked
  `cp_lisennet_conv_hardened/g_best_streaming_int8_signed.onnx`
  (sanitize → fold → generate; shell helpers in `ensemble_sweep/` carry
  session-local paths — adapt `$SP`).

## Traps (each cost real time; all confirmed)

1. The stored streaming export predates the Pad empty-input fix — atonn
   segfaults. Strip trailing empty inputs first (`sanitize_pads.py`).
2. Grouped **ConvTranspose** crashes atonn (garbage dequant scales); grouped
   convs with **2-D kernels** compile but fall to SW. Block-diagonal dense
   fixes both. Grouped 1×1/depthwise are pure-HW (probe in `ensemble_sweep`).
3. `stedgeai validate --mode target` crashes host-side on the fold graphs
   (INTERNAL ERROR NoneType) and rejects >~20-IO graphs at bind (E801).
   `npu_profiler` is the measurement path; it hangs at `-b 16` on the bigger
   graphs — use `-b 1`/`-b 4`.
4. Killing `n6_loader` mid-"Loading memories" wedges the ST-LINK
   (`DEV_USB_COMM_ERR`). Recovery without power-cycle:
   `usbipd.exe detach → unbind → bind → attach --wsl`, verify with
   `STM32_Programmer_CLI -c port=SWD mode=UR`. Never interrupt loads.
5. `/tmp` is wiped on WSL reboot — anything session-local there is gone;
   everything needed is in this branch or regenerable.

## Open threads, in priority order

1. **Board latency of the v3 member-correct folds** (laptop):
   `stedgeai generate` + `npu_profiler` on
   `paper/data/tmp_quant/ens_graphs/ens_memb_k{1,2,4}.onnx` (+ the
   `relu6deep_streaming_int8_signed.onnx` baseline). The K=1 twin vs baseline
   A/B prices the fc1-split/tail restructure; expected epochs ≈ v2 bd-fold ±
   a few. Risk: 26-in/26-out vs npu_profiler's E801 threshold (18/18 worked).
2. **Ship-grade quality fold**: the MinMax recalibration is worth +0.0237
   before any ensembling — decide percentile→MinMax for the deploy recipe
   (check nc24/nc20 too), then board-validate `ens_memb_mm_k4` (host PESQ
   3.0523 vs deployed 3.0117).
3. **Streaming frame-escalation** — the 96 %-at-half-cost result is the
   offline ceiling; escalated members need FIFO-state warm-up (copy states
   from the always-on pair). The K=4 member-correct fold + per-member state
   slices make this buildable now.
4. **Seed members** — independently trained seeds through the same fold turn
   D into genuine epistemic uncertainty. Note: `fold_members.py` asserts
   members share activation scales; different seeds need a joint
   recalibration pass first.
5. **Layer-resolved D** — members differing in one layer's rounding rank the
   activation quantizers; targets the remaining ~0.016 (MinMax grid) gap.

## Anchor numbers to remember

baseline 2.598 ms / K=2 fold 3.762 ms / K=3 fold 5.312 ms · offline RTN
3.0268 → SR-K=4 3.0421 (cap FP32 3.0682) · frame-adaptive: 96 % of K=4 at
3.0 members · OOD white-noise AUROC 1.000 · **streamed** (relu6-deep own
grid): RTN 3.0117, SR-K=4 +0.0100; MinMax recal +0.0237, MinMax SR-K=4
**3.0523** (+0.0405) · member-correct fold bit-exact (fold == K sessions).
