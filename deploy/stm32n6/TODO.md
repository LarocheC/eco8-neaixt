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
