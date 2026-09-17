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

## N6Net-v2: reshaped around those estimates (`n6net/model_v2.py`, `configs/n6net_v2.json`)

Single-conv probes (one conv + ReLU, int8, `n6-noextmem`) gave these per-op costs:

| conv shape | MAC per compute cycle | MAC per real cycle (incl. memory waits) |
|---|---|---|
| 1×1, any width (96–576 in, 24–192 out) | 96 (cap) | 27–36 |
| 3×1 / 5×1 / 7×1 along frequency | 108–129 | 54–85 |
| 1×3 or 3×3 (kernel along time) | 48 | 34–42 |

Also, from v1: the elementwise-only stages were 40 % of the cycles and 0.5 % of
the ops. PReLU → ReLU alone takes v1 from 1.44 to 1.12 ms.

v2 applies those rules:
- 256 bins (Nyquist copied on the host), with a strided 4×1 stem to 128 rows.
- 4 blocks, each a dilated causal (5 × 3-tap) time-frequency conv, then a 5×1 spectral conv, then a skip. Dilations are 1/2/4/8, for a 31-frame receptive field.
- ReLU only.
- The head outputs 2 channels × 128 rows; the host interleaves them into the mask.

In the deploy graph, each temporal kernel becomes three 5×1 convs, one per
past column. The host keeps a ring of 2·d columns per block and passes only
the two columns each kernel reads: 8 columns in total, against v1's 15.
`tests/test_n6net_v2.py` checks causality, the 31-frame receptive field and
streaming parity, and that the deploy graph has no Concat, Slice or PRelu and
no conv kernel along time.

| model | MAC/frame | weights | acts | stages (EC blobs) | ops/max-cycle | U_MAC | est. NPU ms |
|---|---|---|---|---|---|---|---|
| v1 split, PReLU, C96 | 64.7 M | 243 kB | 651 kB | 42 (1) | 90 | 15.6 % | 1.44 |
| v1 split, ReLU, C96 | 64.6 M | 243 kB | 675 kB | 30 (1) | 115 | 20.0 % | 1.12 |
| **v2 C80 (MAC-matched)** | 65.8 M | 501 kB | 170 kB | 23 (1) | 184 | 31.9 % | **0.72** |
| **v2 C96** | 94.7 M | 721 kB | 192 kB | 23 (1) | **233** | **40.5 %** | 0.81 |
| v2 C128 | 168.2 M | 1.25 MB | 256 kB | 23 (1) | 196 | 34.1 % | 1.71 |
| v2 C192 | 377.7 M | 2.83 MB | – | does not fit on-chip (553 kB left unplaced) | | | |
| v2 C192 on octoFlash | 377.7 M | 2.83 MB | 384 kB | 23 (0) | 42 | 7.4 % | 17.8 (misses the 16 ms hop) |

All v2 points are 0 SW / 0 hybrid. The strided stem maps to HW. max_cycles ≈
compute_cycles, so v2 no longer stalls on memory. The int8 mask matches fp32
with cosine ≥ 0.9999.

What it shows:
* **At equal MACs, v2 needs half the NPU time of v1** (0.72 vs 1.44 ms), or
  36 % less than v1 with ReLU.
* **96 is a good width for tall kernels after all.** A 5×1 96→96 conv runs at
  256 ops/cycle (128 MAC, 44 % of the ceiling); at C80 it's 200. C128 reaches 256
  on most convs but drops to 160 on two of them, with npuRAM3–6 all ≥ 75 % full.
* **On-chip memory sets the size limit.** The four npuRAMs hold about 1.4 MB
  of weights (C ≈ 128). Beyond that, weights go to flash and the model is
  memory-bound again (C192: 7 % U_MAC, 17.8 ms).
* **Trade-off:** v2 has 2–3× more parameters than v1 (513 k at C80, against
  250 k), because each block is a 5 × 3-tap kernel. v1's "10× fewer parameters"
  claim is given up for NPU time.
* Each EC blob carries a small controller epoch (51–82 k cycles, reported at a
  multi-watt level). The report excludes it as an estimator artefact.

Not done yet (host/firmware side): zero-copy ring (`--no-inputs-allocation`
and pointer rotation), an 8 ms hop, a complex mask, and on-board validation of
all of the above.

## Frequency reach (v2): full band for < 1 % MAC

Reach here means how many input bins can change the centre output bin, measured
by perturbation (`model_v2.frequency_reach`):
- **v1: 9 of 257 bins (±125 Hz).**
- **v2: 72 of 256 (±1.1 kHz).**

Compiler probes for widening it:

| candidate | Neural-ART mapping |
|---|---|
| frequency-dilated conv (d = 2, 4, 8) | **SW Conv on the M55**, so it's ruled out |
| Resize / ConvTranspose / pixel shuffle along frequency | hybrid (DepthToSpace, Concat, Pad, Transpose) |
| mean over frequency → 1×1 → broadcast add/mul | HW, but only as mean over (H, W); mean over H alone is a hybrid Transpose |
| 4×1 stride 4 ×2 → full-height conv → broadcast add | HW |

So v2 got a `FullBand` branch (`gap` or position-aware `pool`), which can go on
any subset of blocks, plus an optional learned per-row embedding
(`freq_pos_emb`) so a broadcast context can still be told apart by band.
All results are at C96, n6-noextmem + epoch controller, and all are 0 SW / 0 hybrid:

| variant | reach | MAC/frame | weights | stages (EC blobs) | U_MAC | est. NPU ms |
|---|---|---|---|---|---|---|
| v2 (no branch) | 72 | 94.5 M | 721 kB | 23 (1) | 40.5 % | 0.81 |
| gap on block 1 | 256 | 94.5 M | 730 kB | 25 (1) | 39.2 % | 0.84 |
| gap on block 1 + pos emb | 256 | 94.5 M | 742 kB | 25 (1) | 37.3 % | 0.88 |
| **pool on block 1 + pos emb** (`configs/n6net_v2_fullband.json`) | 256 | 95.2 M | 796 kB | 31 (1) | 36.7 % | **0.90** |
| gap on all 4 blocks (+ pos, zero-init) | 256 | 94.5 M | 757 kB | 31 (1) | 35.8 % | 0.92 |
| pool on all 4 blocks (+ pos, zero-init) | 256 | 97.3 M | 973 kB | 55 (1) | 33.8 % | 1.00 |
| 64 rows (`stem_stride=4`), C128 | 144 | 84.0 M | 1.25 MB | 23 (1) | 24.6 % | 1.19 |
| 64 rows, C128, gap ×4 + pos | 256 | 84.1 M | 1.32 MB | 43 (5 + 4 hybrid Transpose, before the mean fix) | 22.1 % | 1.32 |
| 64 rows, C128, pool ×4 + pos | 256 | 86.5 M | 1.57 MB | 39 (1) | 19.6 % | 1.54 |

What it shows:
* **Full-band reach costs < 1 % MAC and about +4–11 % NPU time.** The added time
  is the two tensor-wide passes (pool and broadcast add) per branch, not the
  arithmetic. One branch gives full reach, and more branches only add passes.
* **Recommended: one position-aware `pool` branch on block 1, plus the
  embedding.** Unlike `gap`, the full-height conv gives each band its own
  weights, for 0.02 ms more. Which variant sounds better is for training to decide.
* **64 rows is not a win on the NPU.** It doubles the local reach, but C128
  runs at lower utilisation and sits at the on-chip weight ceiling.
* An all-zero positional embedding gets folded away by the toolchain, so it
  now starts from small random values; the "zero-init" rows above were
  compiled before that change.
