# STM32N6 deploy — TODO

## Lever 2 — move the Cortex-M55 software share onto the NPU/int8

**Status:** not started. Lever 1 (weight locality → internal npuRAM) is done and
gave 7.14 → 4.40 ms/frame. After it, the NPU core is only ~1.26 ms; the remaining
**~3.1 ms (≈70%) is Cortex-M55 software**, so this is now the dominant cost.

**Root cause (from the per-epoch profile).** The per-frame streaming FIFO state
handling — `Slice` / `Gather` / `Concat` on the per-block state buffers — and the
int8 quant boundary run as **float software epochs** on the M55. The QDQ PTQ wraps
the state path in `DequantizeLinear → Gather(float) → QuantizeLinear`, i.e. a float
round-trip per frame instead of staying int8.

**Goal.** Keep the FIFO state path **int8 end-to-end** so `Slice`/`Concat`/`Gather`
map to HW (or int8 SW) with no float dequant/requant. Targets the ~3.1 ms M55 share.

**Plan (model-export surgery — touches the pipeline that yields 2.911 PESQ, so verify):**
1. `convfsenet/quant.py` (and maybe `convfsenet/streaming.py`): quantize the state
   buffers once and keep `Slice`/`Concat`/`Gather` int8 — no float Q/DQ around them.
   Consider also whether `noisy_mag` can be presented int8 (currently float32 input →
   forces SW quantize at the boundary).
2. Re-export FP32 streaming ONNX → re-run int8 PTQ.
3. **Parity / quality gates** (must hold): `tests/test_convfsenet_*` streaming parity,
   and int8 PESQ ≈ 2.911 (no regression).
4. Re-deploy + re-measure via [ONBOARD_MEASUREMENT.md](ONBOARD_MEASUREMENT.md)
   (generate → n6_loader → validate → npu_profiler). Check the SW/HYBRID epoch count
   drops and on-target mask cosine stays ≥ 0.99.

**Expected payoff.** Cut into the ~3.1 ms M55 share; combined with lever 1, could push
well below 4.40 ms/frame. Best done with fresh context — it's multi-file with real
correctness risk, unlike lever 1's one-line profile swap.

---

## Gate 0 de-risk (2026-06-20) — both spikes compiled; mechanism refined

Two micro-experiments validated the load-bearing unknowns on `stedgeai` 4.0.1 (`generate` only,
no board). **Baseline reality (regenerated):** `cp_convfsenet/g_best.onnx` int8 → 70 epochs =
**30 HW / 27 Hybrid / 13 SW**; the 13 SW are **9× `Gather` (the dilation tap-select) + 4 misc**
(`Pow`/`Quantize`/frontend `Conv`/`Add` = the one-shot compression prologue). The state
`Slice`/`Concat` are already **Hybrid, not pure-SW** — so the float-round-trip framing above was
over-stated; the dominant SW cost is the 9 dilation Gathers. States are **f32 at the graph IO**
today (float `Gather`/`Slice` consume them).

- **0a — native dilated depthwise int8 conv (Y1, kill the Gather): PASS.**
  `Conv1d(384→384, K=3, groups=384, dilation∈{2,4})` on a contiguous window compiles, conv on
  **HW, zero pure-SW**, no crash. Dilation is realized as `SpaceToDepth → conv → DepthToSpace (+ Pad)`,
  all **Hybrid** — so the 9 SW Gathers become Hybrid, but **at ~3 extra Hybrid epochs per block**
  (raises epoch count). (`Pad` maps as Hybrid here, unlike the monarch crash — context-dependent.)
- **0b — int8 state, pinned shared scale (Y2): PASS.** Hand-authored `int8→DQ(s)→Slice/Concat→Q(s)→int8`:
  **matched s folds** (2 Hybrid epochs, no extra requant, no float round-trip); **mismatched s adds
  only one HW rescale epoch**, not a SW penalty. Shared-scale int8 states remove the float
  round-trip cleanly.

**Refined expectation (honest).** Both levers viable and crash-free, BUT native dilation *trades
9 SW Gathers for more Hybrid epochs* and every `Slice`/`Concat` is **Hybrid, never pure-HW DMA** —
so the design-panel "best case ~1.4–1.6 ms" is optimistic; realistic target ~**2–3 ms** (still a
clear win over 4.40 ms). On-board epoch-split + latency (Gate 2) is the real go/no-go. If native-
dilation Hybrid overhead dominates the epoch count, **fall back to a deeper stack of dilation≤2
contiguous-window plain-K=3 blocks** (equal receptive field, Gather-free, fewer reshuffle epochs).

---

## Gate 1 result (2026-06-20) — Y1 native dilation is a WASH/regression. Hypothesis overturned.

Implemented Y1 as an opt-in `native_dilation` flag on `ConvFSENetStreamingFast` /
`export_streaming_fp32` (`convfsenet/streaming.py`, `export_onnx.py`). **Parity perfect:**
native-dilation Fast is **bit-identical (Δ=0)** to the gather Fast and 4e-7 vs the naive reference;
all 22 `test_convfsenet_streaming_parity` pass. Re-quantized (200-utt MinMax, same recipe):
int8 vs FP32 mask **cos 0.9992**, vs deployed gather int8 **cos 0.9976** → carries ~2.911 PESQ.

**The graph change worked but the latency didn't.** `generate` (noextmem):

| | gather (baseline) | native-dilation (Y1) |
|---|---:|---:|
| total epochs | 70 | **103 (+47%)** |
| pure SW | 13 (9 are the Gather) | **4** |
| Hybrid | 27 | 51 |
| HW | 30 | 48 |
| on-board (validate, noextmem) | ~5.4 ms* | **5.42 ms** (cos 0.9997) |

*the documented 4.40 ms gather baseline was `npu_profiler`; `validate` runs ~1 ms higher, and a
clean gather `validate` couldn't be captured (board RAM-load wedged after the native-dil run —
needs a power-cycle). But the conclusion is method-independent:

**Y1 killed the 9 SW Gathers (SW 13→4) but it did NOT speed the model up — it slightly regressed.**
The compiler realizes native dilation as `Pad`+`SpaceToDepth`+`DepthToSpace` (Hybrid), adding ~33
epochs. **The 9 SW Gathers were cheap; the SpaceToDepth dilation realization is more expensive.**
This **overturns the design-panel premise** that the per-frame Gather was the ~3.1 ms M55 tax to
eliminate — it isn't. ConvFSENet's latency floor on the N6 is set by the **inherent per-block
Hybrid state plumbing (`Slice`/`Concat`, never pure-HW DMA) + the conv epochs themselves + epoch
overhead**, none of which Y1 removes (and Y1 makes the epoch count worse — see principle that
fewer/larger epochs win: monarch_full 88ep/2.13ms < monarch_8 134ep/2.89ms).

**Verdict (Y1).** Implemented as a `native_dilation` flag, parity-verified (bit-identical to the
gather form, 22/22 streaming-parity tests), measured on-board — then **reverted** (commit kept the
finding, not the code): it does not help on the N6. Re-derivable from this writeup if ever wanted
for a different target or the dilation≤2 redesign.

## Gate 1 result (Y2) — int8 states is a NO-OP. stedgeai already does it.

First, the scales were *already matched*: every `state_k_in` entry-quant scale already equals its
`state_k_out` exit-dequant scale (zp=−128, identical to 7+ digits; states 4/6 off by 1 ULP). So
the quantizer already pins the state scales (the "byte-exact memcpy" was real), and `validate`
already reports **int8 state IO on-device**. Y2's premise (pin scales → enable the fold) was
already satisfied.

Tested anyway: graph-surgery the deployed int8 ONNX to make all 9 state IO tensors **explicit
int8** (reuse the matched `state_out` scales; `int8 → DQ → Gather/Slice → Q` per Gate 0b).
Compiled (noextmem) to a **byte-identical epoch split**: 70 epochs, 30 HW / 27 Hybrid / 13 SW —
**the 9 `Gather`s are still SW**, nothing folded, nothing changed. stedgeai already optimizes the
f32 state boundary to int8; declaring it explicitly is a no-op. The float-state-round-trip the
design panel feared **does not exist as a separable cost.**

## Lever 2 — closed. The premise was wrong.

Both export-surgery sub-levers fail to help:
- **Y1 (kill the Gather):** the 9 SW Gathers are *cheap*; replacing them with native-dilation
  `SpaceToDepth`/`Pad` (Hybrid) costs *more* (+33 epochs) → wash/regression (5.42 vs ~4.4 ms).
- **Y2 (int8 states):** no-op; stedgeai already does it (byte-identical compile).

The ~3.1 ms M55 "software share" is **not** a removable Gather/quant-boundary tax — it is the
**inherent per-block Hybrid `Slice`/`Concat` state plumbing** (every state-movement op is Hybrid,
never pure-HW DMA on the Neural-ART) plus the conv epochs and per-epoch overhead. None of these
yield to export surgery. **ConvFSENet is at its N6 floor (~4.4 ms / RTF 0.275).**

**Where this leaves the N6 story:** the structured-recurrence path already won on latency —
**monarch_full 2.13 ms / RTF 0.13** is the fastest real-time model; ConvFSENet wins on PESQ
(2.91 vs 2.85). The only remaining ConvFSENet speed idea is a **dilation≤2 deeper-stack retrain**
(fewer reshuffle epochs) — a real retraining experiment with uncertain payoff, not export surgery
— or a different streaming-state representation that avoids the per-block FIFO Hybrid ops entirely.
