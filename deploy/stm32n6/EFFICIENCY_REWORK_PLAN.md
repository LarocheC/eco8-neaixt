# Plan — STM32N6 efficient speech enhancement: implement / test / deploy under ≤40 ms latency

> **Status (Track 1 — host work DONE, M1 reached).** Stateless-windowed ConvFSENet
> implemented + tested (34/34) + int8-PESQ'd on the full VBD test. The int8 deploy
> graph is stateless (zero `Gather`/`Pad`/state/BatchNorm). **Host int8 PESQ:
> windowed-256 = 2.913 (FP32 2.933), matching the streaming 2.911 — gate ≥2.85 met,
> no retrain.** Two findings refined the plan: (1) the 257→256 Nyquist drop is
> PESQ-neutral, so no 256-fine-tune; (2) the windowed ring-buffer cold start needs
> `replicate` (repeat first frame), not zeros, to avoid a ~0.045-PESQ frontend-bias
> transient (the `coldstart=zero` number is only 2.843). Gate-0 (`stedgeai generate`)
> and Phase-4 (on-board latency) run on the **deploy box** — this is the training box
> (no stedgeai/board). Handoff + commands: `WINDOWED_DEPLOY_HANDOFF.md`. **Track 2
> (small-STFT 256/128) trained to completion and was REJECTED on quality: int8 PESQ
> 2.725 (FP32 2.783), 0.13 below the ≥2.85 gate — the coarser 129-bin STFT costs ~0.19
> PESQ vs Track 1. Track 1 (512/256 windowed) is the winner.** Track 3 leg-C probe
> built (`gate0_artifacts/`), gated on the deploy-box Gate-0D.

## Context

This session deployed four speech-enhancement models on an STM32N6570-DK and characterized the
Neural-ART cost model down to the cycle/epoch level. Established facts (all measured, see
`deploy/stm32n6/NSNET2_DEPLOYMENT_NOTES.md`, `RESULTS_*.md`, `deploy/stm32n6/TODO.md`):

- **Best today:** `monarch_full` 2.13 ms / RTF 0.13 / int8 PESQ 2.848 (fastest); **ConvFSENet
  4.40 ms / 2.911 (quality leader, conv, HW-mappable)**; dense 22.94 ms (memory-bound).
- **Single-frame inference is overhead/memory-bound, NOT compute-bound** (MAC array 10–40 %
  utilized). Latency is driven by **epoch count**, **M55 state-plumbing** (per-block FIFO
  `Slice`/`Concat`, always Hybrid), and **weights-on-chip** — not by MACCs.
- **Lever 2 is closed:** ConvFSENet's ~3.1 ms M55 floor is the inherent FIFO state plumbing.
  Y1 (native dilation) was a wash; Y2 (int8 states) a no-op. The open path (TODO §"closed") is
  *a different streaming-state representation that removes the per-block FIFO entirely.*

The design panel's deep-rework thesis has three legs: **(A) stateless feed-forward + host context
window** (measured-true — kills the FIFO floor), **(B) block processing T>1 to amortize epoch
overhead** (measured principle), **(C) 2-D conv over an `[F,T]` grid to lift array utilization**
(hypothesized — every conv compiled so far is `h:1`; one adjacent Conv2d-int8 failure on degenerate
spatial). Goal of this plan: **maximize NPU efficiency without losing PESQ (≥~2.85, ideally 2.91),
while keeping algorithmic latency ≤ ~40 ms** — which is the new hard constraint that reshapes the
design.

## The latency budget (the constraint that drives everything)

Algorithmic latency ≈ **STFT framing + (T−1)·hop + compute**. With the current `n_fft=512`,
`hop=256`, `center=True` STFT (`common/dataset.py:11`), the framing alone is ~one window ≈ **16–32 ms**
(center adds `n_fft/2` = 16 ms lookahead; OLA synthesis adds ~window−hop = 16 ms). So:

| config | STFT | block T | algorithmic latency | block amortization |
|---|---|---:|---:|---|
| **A (primary)** | 512/256 (current) | **1** | **~32–36 ms** ✓ | none (but still banks legs A+C-at-T1) |
| B (stretch) | 512/256 | 2 | ~48 ms (✗ strict; "≈40" maybe) | 2× |
| C (retrain) | **256/128** (16 ms/8 ms, 129 bins) | 2–3 | ~24–32 ms ✓ | 2–3× |

**Implication:** the T=8 / 128 ms BTSE-Δ from the panel is OUT. Under ≤40 ms the bankable win is
**Config A — a *stateless windowed* ConvFSENet at T=1**, which keeps today's ~32 ms latency and
2.91 PESQ but removes the FIFO M55 floor and fills the convs with the context window (leg C even at
T=1, because the input is a `[256, ~43]` context grid, not `[256,1]`). Config C buys real block
amortization by shrinking the STFT (costs a retrain + coarser frequency resolution).

## Strategy: stateless windowed ConvFSENet (bank A + C@T1; gate the rest)

Key realization that simplifies implementation: **a stateless windowed model = the existing
*offline* causal `ConvFSENet_QuantFriendly` run on a short fixed window**, emitting the last T
frames. No FIFO, no per-block state I/O. The host keeps a ring buffer of the last `L = RF−1 = 42`
magnitude columns (receptive field RF = 1 + 2·Σdil = 1 + 2·3·(1+2+4) = **43 frames**), feeds
`[1, 256, L+T]`, the graph runs valid (padding-0) causal convs, and slices the last T outputs.
This is bit-exact to offline causal for the emitted frames (the same algebra
`convfsenet/streaming.py` already parity-tests, 22/22).

## Models to implement / test / deploy (tracks, in priority order)

1. **Track 1 — Stateless-Windowed ConvFSENet, T=1 (PRIMARY, lowest risk).** Config A. Same
   192/384 backbone, 256-bin (drop Nyquist), `mag_compressed` prologue kept FP32. Target:
   **< ~2.5 ms, RTF < 0.16, PESQ ≈ 2.90, ~32 ms latency.** Removes the 9-FIFO Hybrid class +
   fills convs. Highest-confidence quality (iso-architecture to 2.911).
2. **Track 2 — Stateless-Windowed + small-STFT block (STRETCH).** Config C: retrain at 256/128
   (129 bins) with T=2, for 2× epoch amortization under ~30 ms. Compare efficiency vs Track 1;
   accept only if PESQ holds ≥2.85.
3. **Track 3 — 2-D front-end graft (UPSIDE, GATED by Gate-0D).** Replace the `Conv1d(256→192,k=1)`
   frontend with a small causal 2-D conv stem over the `[256, L+T]` grid — only if a generate-only
   probe shows 2-D int8 maps to HW at high util. Do NOT build before Gate-0D passes.
4. **Track 4 — min-latency-floor reference (OPTIONAL).** Tiny wide-shallow stateless conv at T=2
   to establish the efficiency frontier / lower bound. Skip unless Tracks 1–2 disappoint.

Rejected by the panel (do not pursue): full causal 2-D U-Net CRN (~31× MACC, not real-time —
the "recompute the receptive-field pyramid every block" trap); deepfilter-lite as drafted (fatal
receptive-field/context arithmetic error); butterfly (NPU-hostile).

## Phase 0 — Gate-0 generate-only de-risk (DO FIRST; no board, no training)

Mirrors how Gate 0 settled Lever 2 before any retrain. All via `stedgeai 4.0.1 generate`, dummy
weights, reading `network_generate_report.txt`. **Stop at first hard fail.**

- **0A (go/no-go):** build a dummy `[1,256,L+1]→[1,256,1]` stateless-windowed ConvFSENet ONNX
  (frontend + 9 TCM + backend, 192/384, dil {1,2,4}, valid convs), int8 QDQ + `skip_optimization`.
  Generate (`n6-noextmem`). **Pass = 0 pure-SW epochs, no `Gather`/`state_*` nodes, weights ~1.4 MB
  on-chip, and convs show `h:43` not `h:1`** (the array-fill confirmation). This single run replaces
  the panel's un-reproduced epoch/util claims with a measurement.
- **0B:** 1-block stubs depthwise k=3 d∈{1,2,4} over `[384, L]` — measure the Hybrid cost of the
  dilation reshuffle (Gate-1 saw SpaceToDepth/Pad); fallback is a dilation≤2 deeper stack (but
  measure — may be worse).
- **0C:** diff the 0A epoch list vs today's deployed report — confirm the FIFO `Gather`/`Slice`/
  `Concat` class is gone (leg-A confirmation).
- **0D (only for Track 3):** compile a single `Conv2d(1→24, k=3, stride=(1,2))` over `[1,1,256,L]`
  int8. Pass = compiles int8 + maps HW + real `[h>1,w>1]` extent. If it fails like the NSNet2
  `lower_arith_set_in_batch` case, leg C is dead on this toolchain → Track 3 cancelled.

## Phase 1 — Implementation (PyTorch + export)

Reuse existing code; new code is the stateless-windowed export wrapper.

- **`convfsenet/streaming.py`** — add a `ConvFSENetWindowedONNX` wrapper: takes `noisy_mag_window`
  `[B, n_freq, L+T]` (no state inputs), runs the offline causal `ConvFSENet_QuantFriendly` forward
  (BN-folded, valid convs), returns the last `T` mask columns `[B, n_freq, T]`. Reuse
  `_fold_bn_into_conv` and the existing valid-conv math; do NOT reintroduce FIFO state. Drop the
  `native_dilation` flag idea (reverted — wash).
- **`convfsenet/export_onnx.py`** — `export_windowed_fp32(base, out, T, L)`; fixed-shape window I/O,
  `dynamic_axes` only on batch, `opset 17`, `dynamo=False`. Mirror the structure of
  `export_streaming_fp32` and `deploy/stm32n6/host/export_blockdiag_npu.py` (the stateless-export pattern).
- **`convfsenet/quant.py`** — reuse `quantize_fp32_onnx` verbatim (QDQ, per-channel MinMax,
  `skip_optimization`, prologue `Pow/Add/Unsqueeze` excluded). The windowed graph quantizes the same way.
- **256-bin alignment** — drop the Nyquist bin at the host STFT→feature boundary (`257→256`), pad the
  mask back to 257 on the host after the graph. Single edit at the feature boundary; `fc_in` 256, `fc_out`→256.
- **Config** — clone `configs/convfsenet.json` → `configs/convfsenet_win.json` (Track 1, same
  512/256). For Track 2, a second config with `n_fft 256 / hop 128 / win 256` (129 bins) — the only
  STFT edit point is `common/dataset.py` `mag_pha_stft/istft` calls + the config.
- **Host runner** — a windowed inference path (ring buffer of L context columns) extending
  `convfsenet/inference_onnx.py`; this is also what the on-board C glue mirrors (no in-graph state →
  simpler `ai_dpu` than the FIFO version).

## Phase 2 — Training & PTQ

- **Track 1** may not need a full retrain if 256-bin + windowing is bit-exact to the trained
  `cp_convfsenet` weights — verify parity first (Phase 3); if the 257→256 drop shifts results, do a
  short fine-tune. **Track 2/3 require training** at the new STFT/topology.
- Train via `convfsenet/train.py` with `gan.enabled=true` (PESQ MetricGAN is **mandatory** — no-GAN
  floor is ~2.77; GAN reaches >2.85). ~200 epochs; FP32 ~22 min + GAN 6–7 h on a 4090 (per
  `RESULTS_CONVFSENET.md`). `causal=True`.
- PTQ: `python -m convfsenet.quant` (un-normalized MinMax — the measured-best calibration; prologue
  excluded; `skip_optimization`).

## Phase 3 — Testing (gates that must hold before deploy)

- **Streaming/window parity:** extend `tests/test_convfsenet_streaming_parity.py` to the windowed
  form — assert the emitted T columns match the offline causal model to ~1e-6 (FP32) and the int8
  mask cosine ≥ 0.99 vs FP32. This inherits the 22/22 parity discipline already in place.
- **int8 PESQ:** `convfsenet/inference_onnx.py` on the VBD test split — require **≥ 2.85** (target
  2.90–2.91). Host int8-vs-stock cosine ≥ 0.999 as the fast proxy.
- Confirm no `Pad`/`Einsum`/`Gather`/state nodes in the int8 graph (op-histogram check).

## Phase 4 — Deployment & on-board measurement (Gate-2, the real verdict)

- **Recover the board first:** re-plug the ST-LINK USB (it's wedged — `DEV_USB_COMM_ERR`), then
  `usbipd attach --wsl --busid <id> --auto-attach`; confirm `/dev/ttyACM0`.
- Per `deploy/stm32n6/ONBOARD_MEASUREMENT.md`: `generate` (`n6-noextmem`) → `n6_loader.py` →
  `stedgeai validate` (latency + on-target cosine) → `npu_profiler` (per-epoch HW/SW split,
  utilization). `validate` should work (no GRU fusion). Record latency/RTF, epoch split, mask cos ≥0.99.
- **Decision gate:** does the measured per-frame latency beat ConvFSENet (4.40 ms) and ideally
  monarch_full (2.13 ms) at PESQ ≥2.85? If the convs stayed `h:1`-equivalent or dilation reshuffle
  dominated (the Gate-1 failure mode), fall back per Risks.

## Milestones / sequencing

1. **M0 — Gate-0A/B/C pass** (≈30 min, no board): stateless-windowed stub compiles, FIFO gone,
   convs `h:43`, fits on-chip. *Greenlights everything.*
2. **M1 — Track 1 parity + int8 PESQ ≥2.85** (host only).
3. **M2 — Track 1 on-board** (Gate-2): latency/epoch split measured. *The headline result.*
4. **M3 — (if M2 wins) Track 2 small-STFT block** for extra amortization; and/or Gate-0D → Track 3
   2-D graft if leg C proves out.
5. **M4 — pick the winner, document, deploy.**

## Risks & fallbacks

- **Biggest risk (leg C):** the `[256,43]` context convs may not amortize/fill on this toolchain
  (every conv to date is `h:1`; the array-fill lift is unmeasured). If so, Track 1 lands ~ConvFSENet
  speed (still removes the FIFO floor, so likely < 4.40 ms but maybe not < 2.13 ms). **Gate-0A
  measures this before any training.**
- **Dilation reshuffle** (SpaceToDepth/Pad Hybrid) could dominate → fallback dilation≤2 deeper stack
  (Gate-0B; measure, may not help).
- **Recompute cost:** stateless T=1 recomputes the 43-frame pyramid each frame — free only while the
  array is idle; if it pushes compute-bound, that's actually the *good* outcome (means leg C worked).
- **256-bin/STFT-change quality:** verify parity/PESQ before trusting "no retrain"; budget a fine-tune.
- **Board flakiness:** RAM-firmware loader wedges after `validate`; needs USB re-plug (not SWD reset).

## Measured vs hypothesized ledger

- **Measured:** the cost model (overhead-bound, epoch-count-driven), the FIFO = M55 floor, weights-
  on-chip 1.6×, conv math bit-exact over windows, ConvFSENet 1.42 MB/2.911 dims, one Conv2d-int8 failure.
- **Hypothesized (Gate-0/2 will settle):** that windowed convs fill the array / cut per-frame epochs;
  exact latency of Track 1/2/3; that 256-bin and small-STFT hold PESQ ≥2.85; that 2-D int8 maps (leg C).

## Key files

- `convfsenet/streaming.py` (windowed wrapper; reuse `_fold_bn_into_conv`, valid-conv math),
  `convfsenet/export_onnx.py` (`export_windowed_fp32`), `convfsenet/quant.py` (reuse `quantize_fp32_onnx`),
  `convfsenet/model.py` (offline causal forward), `convfsenet/inference_onnx.py` (host ring-buffer runner),
  `convfsenet/train.py` (GAN training).
- `configs/convfsenet.json` → `configs/convfsenet_win.json` (+ small-STFT variant); `common/dataset.py`
  (`mag_pha_stft` STFT params, Nyquist drop).
- `tests/test_convfsenet_streaming_parity.py` (extend for windowed parity).
- `deploy/stm32n6/host/export_blockdiag_npu.py` (stateless-export reference pattern),
  `deploy/stm32n6/ONBOARD_MEASUREMENT.md` (Gate-2 flow), `deploy/stm32n6/TODO.md` (lever-2-closed context).
