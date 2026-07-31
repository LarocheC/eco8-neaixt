# Idle-cycle ensemble study — handover (2026-07-31)

**Status: parked for a future study.** Everything below was measured in one
day on the WSL2 laptop + the STM32N6570-DK; this branch (`ensemble-study`)
carries the scripts, results, and this doc so the work can be picked up on
any machine. The detailed lab log with all tables lives in
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

## What's in this branch

| path | what |
|---|---|
| `deploy/stm32n6/ensemble_sweep/` | full lab log (README), fold/branch/probe/sanitize scripts, compile reports, npu_profiler logs, `sweep_results.json` |
| `sr_decoder_ensemble.py` (+`_results.json`) | SR-member builder + PESQ harness; caps, K-sweep. **Run first** — it (re)builds member graphs into `paper/data/tmp_quant/sr_graphs/` deterministically |
| `sr_uncertainty.py` (+`_results.json`) | UQ correlations, OOD AUROCs, utterance-level adaptive sim |
| `sr_adaptive_frames.py` (+`_results.json`) | frame-level escalation curve + random control |
| `paper/data/tmp_quant/relu6deep_rt_int8_{fp32,pc_signed}.onnx` | the FP32/QDQ graphs the sr scripts consume (regenerable via `paper/data/eval_conv_rt_int8.py`, ~30 min) |
| `lisennet/eval_metrics_ext.py`, `paper/data/eval_conv_rt_int8.py` | harness dependencies (Table-1 protocol) |

## Reproducing on another machine

- Python: repo venv (`./.venv/bin/python3`); pip needs
  `PIP_INDEX_URL=https://pypi.org/simple` (the default JFrog index 401s).
- Dataset: `common.dataset.load_voicebank_demand()` pulls
  `JacobLinCool/VoiceBank-DEMAND-16k` via HF datasets (cached ~here; will
  download elsewhere).
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

1. **Member-correct fold** — the bd-fold is latency-exact but numerically
   interleaves members through the GLU's dynamic channel split. Needed for
   on-chip quality: split each `fc1` into two grouped convs (per-member GLU)
   + per-member decoder skips. Then board-validate a K=2/K=4 SR fold and
   host-eval its *streamed* PESQ.
2. **Streaming frame-escalation** — the 96 %-at-half-cost result is the
   offline ceiling; escalated members need FIFO-state warm-up (copy states
   from the always-on pair; members differ only at rounding grain).
3. **Seed members** — independently trained seeds through the same fold turn
   D into genuine epistemic uncertainty (and may beat SR draws on PESQ).
4. **Layer-resolved D** — members differing in one layer's rounding rank the
   activation quantizers; directly targets the remaining ~0.026 gap.
5. relu6-deep bd-fold latency (checkpoints for nc20/28/dil16/deep never left
   the training box; relu6-deep + nc24 are local and tracked).
6. Replicate `feat` after the FP32 prologue (SW grows 0.28→0.44→0.85 ms
   with K; avoidable).

## Anchor numbers to remember

baseline 2.598 ms / K=2 fold 3.762 ms / K=3 fold 5.312 ms · RTN 3.0268 →
SR-K=4 3.0421 (cap FP32 3.0682) · frame-adaptive: 96 % of K=4 at 3.0 members
· OOD white-noise AUROC 1.000.
