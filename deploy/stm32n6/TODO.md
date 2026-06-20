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
