# N6Net through the Neural-ART compiler

Status 2026-09-16 — **compile-level only** (board not attached; no trained
checkpoint yet). stedgeai 4.0.1, `n6-noextmem` memory pools (all on-chip).
The weights are seeded random init, calibrated on VoiceBank-DEMAND frames with
propagated FIFO state. Op mapping, memory placement and the epoch split don't
depend on weight values. Quantization ranges do, so re-export once
`cp_n6net/g_best` exists.

```bash
python -m n6net.export_npu --out build/n6net_b3            # --layout time_split (default)
deploy/stm32n6/scripts/compile_n6net.sh build/n6net_b3/n6net_stream_int8.onnx
python deploy/stm32n6/host/npu_cycle_report.py build/n6net_b3/n6net_stream_int8_gen --epochs
```

## It compiles first time — but not the way the design intends

The design as drawn (`time_w`: FIFO on the W axis, 1×6 temporal conv, ORT
static QDQ) compiles unchanged. Result: 250 kB weights and 801 kB activations,
all in npuRAM3/4/5. It splits into **24 epochs: 12 HW, 6 hybrid, 6 SW**:

| cause | effect | fix |
|---|---|---|
| ORT leaves the PRelu slope as a float initializer; ST maps PRelu to HW only if the slope is quantized | all 6 PReLUs are pure-SW epochs on the M55 | `quantize_prelu_slopes` post-pass (int8 slope behind a DQ) → 0 SW epochs. The compiler expands PReLU into HW relu/mul/add |
| W-axis `Concat` (FIFO + current column) | runs on the M55 as a `PROCESSOR` memcpy, 148 kB per block | not fixable on this axis; see layouts |
| W-axis `Slice` (FIFO shift) | hybrid epoch | the channel-axis Slice maps to HW |
| FIFO in/out calibrated separately from the column it stores | concat inputs need requantization | `tie_fifo_qparams`: state in/out share the stored column's scale/zp, so the host threads raw int8 |

## Three layouts, same arithmetic (tests/test_n6net.py)

| layout | FIFO I/O | epochs (plain) | epochs with `--enable-epoch-controller` | acts |
|---|---|---|---|---|
| `time_w` | `(1,C,F,5)` ×3 | 26: 20 HW + 6 hybrid (Concat, Slice) | 7 EC blobs + 6 hybrid | 801 kB |
| `time_c` (1×6 → 1×1 over 576 ch) | `(1,5C,F,1)` ×3 | 21: 18 HW + 3 hybrid (Concat) | – | 807 kB |
| **`time_split`** (6 summed 1×1 convs) | `(1,C,F,1)` ×15 in, ×3 out | **42 HW, 0 hybrid** | **1 EC blob, whole net on the NPU** | **651 kB** |

For `time_split`, the host keeps the ring buffer. With compiler-allocated inputs,
that costs about 370 kB of int8 copies per frame; the `time_w` Concat already
copied that much on the M55. Compiled state outputs and inputs have identical
int8 scale/zp (0.0452/0, 0.0470/0, 0.0498/−1), so no requantization happens
between calls. The int8 mask matches the fp32 offline model with cosine 0.9999.

CONV_ACC tiling (from the `_ca_pipe_N` nodes in `network.c`): every 96→96 1×1
conv and the 1×6 conv use all **4** pipes. The 3×1 spectral conv always gets **3**.

## Static utilisation (compiler's `power_estimates`, not a measurement)

ops = multiply + add, so ops ≈ 2×MAC. The roof is 4 CA × 72 MAC = 288 MAC,
or 576 ops, per cycle. "NPU ms" is `max_cycles` at 1 GHz and excludes M55 time.

| model | MAC/frame | ops/compute-cycle | ops/max-cycle | U_MAC | est. NPU ms/frame |
|---|---|---|---|---|---|
| b3 c96 `time_w` | 64.3 M | 41 | 36 | 6.3 % | 3.55 |
| b3 c96 `time_c` | 64.3 M | 126 | 86 | 15.0 % | 1.49 (+3 M55 concats) |
| **b3 c96 `time_split`** | 64.7 M | 116 | 90 | 15.6 % | **1.44** |
| b1 c96 `time_split` | 21.6 M | 110 | 79 | 13.7 % | 0.55 |
| b3 c72 `time_split` | 36.5 M | 115 | 69 | 12.0 % | 1.06 |
| b3 c120 `time_split` | 100.8 M | 132 | 96 | 16.7 % | 2.10 |
| b3 c128 `time_split` | 114.7 M | 122 | 101 | 17.5 % | 2.28 |

What the estimates say:

* **The 1×6 kernel is the worst-mapped op in the model.** One `time_w` temporal
  conv takes 592k cycles, at 48 ops/cycle. The same arithmetic as a 576-channel 1×1
  conv needs 65.8k compute cycles (432 ops/cycle, 75 % of the roof), but it
  stalls on memory: 220k max cycles. The 3×1 spectral conv reaches 185–217
  ops/cycle.
* **A 96→96 1×1 conv costs one cycle per output element** (257×96 = 24,672),
  which is 96 MAC/cycle, a third of the roof. Utilisation grows with input
  width, so 96 is not a special point in the compiler's own model:
  72 → 96 → 120 → 128 gives 69 → 90 → 96 → 101 ops/cycle.
* 64 M MAC/frame in about 1.4 ms of estimated NPU time gives RTF ≈ 0.09 at a 16 ms hop,
  before M55 overhead. For scale, on-board numbers for earlier models:
  LiSenNet nc24 streaming 2.79 ms at 1.3 M MAC; ConvFSENet 4.40 ms.
* One estimator artefact: a 257-op input `Sub` epoch in `time_w`/`time_c`
  reports 21.6 W over 462k cycles. The report excludes it.

## Next

1. Train (`python -m n6net.train --config configs/n6net_b3.json`), then re-export with `--checkpoint`.
2. On-board: n6_loader + `stedgeai validate --mode target` and `npu_profiler.py`,
   to check these estimates against measured cycles/MAC and wall time,
   including the host ring-buffer cost.
