# NSNet2 on the STM32N6 Neural-ART — deployment notes

What it took to get the NSNet2 family onto the N6 NPU, why the dense baseline
*can* be deployed (and how), and why the structured/sparse variants still
can't. All findings are on **ST Edge AI Core 4.0.1** (the v3.0.0 verdicts in
older notes are superseded). Companion to
[ONBOARD_MEASUREMENT.md](ONBOARD_MEASUREMENT.md) (the ConvFSENet measurement
path) — the toolchain, board, and profiler setup are identical.

TL;DR:

| model | int8 PESQ | int8 weights | NPU result |
|---|---:|---:|---|
| ConvFSENet (conv) | 2.911 | 1.40 MB | ✅ **4.40 ms/frame, RTF 0.275** (real-time) |
| NSNet2 dense (`baseline`) | 2.833 | 2.70 MB | ✅ **22.94 ms/frame, RTF 1.43** (deployed, *not* real-time) |
| NSNet2 `monarch_8` (sparse) | 2.826 | 1.47 MB | ⛔ compiler can't map the monarch block layout (see §3) |
| other monarch / butterfly | varies | 0.48–9.5 MB | ⛔ same layout blocker; butterfly also loses int8 quality |

The one-line story: **the Neural-ART is a 4-D convolution engine.** Models
that are natively convolutional map cleanly; the FC/GRU/structured-matrix
models need their graph reshaped into that world first, and how hard that is
ranges from "a re-quantize flag" (dense) to "re-export the model" (monarch).

---

## 1. Why the dense baseline crashed — and the fix

### Symptom
`stedgeai generate` on `cp_baseline/g_best.onnx` (int8) dies about halfway
through the optimizer passes:

```
TOOL ERROR: list index out of range
```

### Root cause
PyTorch exports the unrolled GRU recurrence `h' = W_h·h + (W_i·x + b)`, and
**onnxruntime's graph optimizer fuses the `MatMul`+`Add` into a single `Gemm`
whose `C` (bias/beta) input is an *activation tensor*** — not a static bias.
You can see these nodes named `*/MatMulAddFusion` in the graph.

The Neural-ART int8 `Gemm` lowering expects `C` to be a constant per-channel
bias vector; it indexes into a quantized-bias list that is empty for an
activation-`C` Gemm → `list index out of range`.

Confirmed by bisection with `onnx.utils.extract_model`:

| subgraph cut at | result |
| --- | --- |
| a `Gemm` with **static bias** (`/Add_3_output_0`) | ✅ compiles |
| a `Gemm` with **activation `C`** (`/Add_4_output_0`) | ⛔ `list index out of range` |

(The FP32 graph *does* compile, but maps 60/69 epochs to **float software on
the M55** — the Neural-ART runs int8 only, so an FP32 deploy is meaningless.)

### The fix: re-quantize with `skip_optimization=True`
`nsnet2/quant.py` calls `quant_pre_process(...)` before `quantize_static`.
`quant_pre_process` runs ORT's optimizer, which is what introduces the
`MatMulAddFusion`. Passing **`skip_optimization=True`** keeps `MatMul` and
`Add` as separate ops, so the int8 graph maps to HW with no activation-`C`
`Gemm`:

```python
from onnxruntime.quantization.shape_inference import quant_pre_process
# ... same setup as nsnet2.quant.quantize_checkpoint ...
quant_pre_process(str(fp32_path), str(preprocessed_path),
                  skip_optimization=True)          # <-- the one change
quantize_static(str(preprocessed_path), str(out_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True, calibrate_method=CalibrationMethod.MinMax,
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
```

Effect on the int8 graph: the 4 recurrent activation-`C` Gemms become 12
`MatMul` + 20 `Add`. Same calibration (200 VBD utterances, MinMax), so it is
**numerically identical** to the committed int8:

| comparison (mask, 100-frame rollout) | cosine |
| --- | ---: |
| no-fuse int8 vs **committed fused int8** | **0.9996** |
| no-fuse int8 vs **FP32 ONNX** | 0.9946 |

→ no PESQ change (still 2.833). The re-quant is a graph reshaping, not a
re-quantization in any lossy sense.

### Deployed result (on-board, npu_profiler)
Compiled with `n6-allmems-O3` (44 epochs: 23 HW / 21 SW), measured on the
board exactly like ConvFSENet:

| metric | value |
| --- | ---: |
| latency / frame | **22.94 ms** |
| NPU core | 12.27 ms (53%) |
| octoFlash read / frame | 2.81 MB @ 122 MB/s avg |
| int8 weights | 2.70 MB (octoFlash) |
| activations | 1.53 MB |
| frame period (hop 256 @ 16 kHz) | 16 ms |
| **RTF** | **≈ 1.43 — not real-time** |
| mask cosine vs FP32 | 0.9946 |

**Why it's not real-time:** the 2.70 MB int8 weight set does **not** fit in
the N6's on-chip NPU RAM (~1.8 MB npuRAM + 1 MB cpuRAM), so the
weight-locality win that took ConvFSENet from 7.14 → 4.40 ms (the
`n6-noextmem` profile) is **unavailable** — `noextmem` is 4.15 MB short. The
weights must stream from external octoFlash every frame, and at 122 MB/s that
2.81 MB read *is* essentially the whole 22.9 ms. NSNet2 dense is
memory-bandwidth-bound on this part, full stop.

### ⚠️ Gotcha: `stedgeai validate` re-fuses and crashes
`stedgeai validate --mode target` re-runs ORT optimization on the reference
model internally, which **re-introduces the fusion** and crashes
(`INTERNAL ERROR: Overwriting Entry 0 of _Mul_5_output_0`). So you cannot use
`validate` to drive the un-fused model. Measure with **`npu_profiler.py`
against the already-loaded firmware** instead:

```bash
# 1. generate the un-fused int8 (the skip_optimization model)
stedgeai generate -m nsnet2_int8_nofuse.onnx --target stm32n6 \
  --st-neural-art n6-allmems-O3@user_neuralart.json \
  --fix-parametric-shapes "{'B':1}" -n network -o /tmp/n6val_nsnet2
# 2. load firmware + weights (see ONBOARD_MEASUREMENT.md §2)
python n6_loader.py --config config.json -nf /tmp/n6val_nsnet2/network.c -bc N6-DK
# 3. profile (NOT validate)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="$RUNNER" \
  /tmp/profenv/bin/python "$RUNNER/examples/npu_profiler.py" \
  -d serial:/dev/ttyACM0:921600 -c /tmp/n6val_nsnet2 -b 16
```
Accuracy is then verified off-board (host ORT int8-vs-FP32 cosine, above)
rather than via `validate`'s on-target cross-check.

---

## 2. Which sparse variant makes sense to deploy

The HF repo (`claroche1/sparse-nsnet2-checkpoints`) has 9 variants. Two gates
decide deployability:

1. **int8 PESQ must hold** — the deployed model is int8. This eliminates every
   **butterfly** variant: they collapse under int8 PTQ (2.1–2.6 PESQ, vs
   ~2.83 for the monarch/dense models). Butterfly needs QAT to be usable in
   int8, which we have but haven't re-exported for deploy.
2. **Weights must fit on-chip (~< 2 MB int8)** — otherwise it is
   memory-bound and not real-time, exactly like the dense baseline above.

| variant | params | int8 PESQ | int8 ONNX | fits on-chip? | quality OK? |
| --- | ---: | ---: | ---: | :---: | :---: |
| `baseline` (dense) | 2.78 M | 2.833 | 2.70 MB | ✗ | ✓ |
| `wide_monarch` | 2.36 M | 2.842 | 9.46 MB | ✗ | ✓ |
| `monarch_full` | 0.70 M | **2.848** | 2.85 MB | ✗ | ✓ |
| **`monarch_8`** | **0.36 M** | **2.826** | **1.47 MB** | **✓** | **✓** |
| `monarch_fc` | 2.14 M | 2.789 | 2.89 MB | ✗ | ~ |
| `butterfly_ortho` | 0.19 M | 2.577 | 0.48 MB | ✓ | ✗ |
| `butterfly_2blocks` | 0.36 M | 2.202 | 0.82 MB | ✓ | ✗ |

**`monarch_8` is the right target**: it is the only variant that clears *both*
gates — int8 PESQ 2.826 (matching the dense baseline) at 1.47 MB int8 (7.7×
compression), small enough to fit on-chip and therefore the only NSNet2 that
*could* be real-time on the N6. (`monarch_full` has the best int8 PESQ but at
2.85 MB hits the same memory wall as dense.)

---

## 3. Why `monarch_8` still won't compile (the monarch block layout)

`monarch_8`'s structured FC/GRU projections are exported as a per-block
batched matmul:

```
Einsum  'bkp,kqp->bkq'   A=[1, k=8, p]   W=[k=8, q, p]   ->  [1, 8, q]
```

i.e. 8 independent small matmuls (block-diagonal linear). `stedgeai generate`
fails early:

```
TOOL ERROR: Error in computation of shapes
```

Progress made (all reproducible):

1. **`Einsum` itself is lowerable.** `bkp,kqp->bkq` is exactly a **grouped
   1×1 convolution** (`groups=8`): reshape `A→[1,k·p,1,1]`, weight
   `[k·q, p, 1, 1]`, `Conv2d(groups=8, kernel=[1,1])`, reshape back. We do
   this mechanically for all 8 Einsums.
2. **The dynamic-shape scaffolding folds away.** The export emits a
   `Shape`/`Gather`/`ConstantOfShape`/`Slice`/`Pad`/`Reshape` subgraph to pad
   the input to a multiple of the block count. Fixing `B=1` and constant-folding
   with ORT (`ORT_ENABLE_EXTENDED`) collapses all of it to static tensors.
3. **But the residual layout still breaks the compiler.** After (1)+(2) the
   graph is clean and statically shaped, yet it crashes with
   `Error in computation of shapes` (full model) / `INTERNAL ERROR: Unknown
   dimensions: H` (isolated block). The cause is **rank/layout mixing**: the
   monarch graph carries rank-2 FC activations (`[1,257]`, `[1,400]`,
   `[1,1200]`) and rank-changing `Pad`/`Reshape` ops feeding the block matmul.
   The Neural-ART compiler models tensors as 4-D `NCHW` and cannot assign a
   consistent `H`/`W` to the rank-2 / padded / reshaped intermediates.

**Conclusion.** The blocker is not the `Einsum` (that maps to grouped conv) and
not the dynamic shapes (those fold). It is that the monarch model lives in a
**rank-2 FC + block-reshape world**, and the Neural-ART wants a **4-D
conv-native** world. Making `monarch_8` deployable means **re-exporting the
whole model conv-native** — every FC/projection as a 1×1 conv on a
`[1, C, 1, 1]` tensor, the block structure as grouped convs, the GRU gates as
elementwise conv-domain ops, and **no rank-2 intermediates or `Pad`/block
`Reshape`**. That is model-export surgery (in `nsnet2/` + `torch_structured`),
not an ONNX patch.

This is the same wall the dense baseline hit in a milder form — there the FC
reshapes landed as the 21 SW (M55) epochs rather than crashing. The conv model
(ConvFSENet) never hits it because it is 4-D throughout.

---

## 4. Recommendations / next steps

- **Dense NSNet2** is deployed and measured. Keep it as the "recurrent
  baseline on real silicon" data point: it runs, but at RTF 1.43 it misses
  real-time by ~1.4× because its weights overflow the NPU's fast RAM — a clean
  argument for the convolutional family on edge NPUs.
- **If a real-time NSNet2 is wanted**, `monarch_8` is the target (1.47 MB fits
  on-chip), but it needs a **conv-native re-export** so the monarch blocks
  become grouped convs end-to-end with no rank-2/`Pad`/`Reshape` plumbing.
  Scoped, but it's an export-pipeline change with a parity + PESQ gate, best
  done as its own effort.
- **Wiring the fix into the pipeline (optional):** expose
  `skip_optimization=True` as a deploy-only flag on
  `nsnet2.quant.quantize_checkpoint` (e.g. `--deploy-npu`) so the un-fused int8
  artifact is reproducible without the one-off script. Left out for now to
  avoid changing the default quantization path that the `RESULTS_NSNET2.md`
  numbers were produced with.
