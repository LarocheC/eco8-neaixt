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
| NSNet2 `monarch_full` (sparse) | 2.848 | 0.72 MB on-chip | ✅ **2.13 ms/frame, RTF 0.13** (real-time, fastest *and* best PESQ — §3) |
| NSNet2 `monarch_8` (sparse) | 2.826 | 0.37 MB on-chip | ✅ **2.89 ms/frame, RTF 0.18** (real-time — §3) |
| ConvFSENet (conv) | 2.911 | 1.40 MB | ✅ **4.40 ms/frame, RTF 0.275** (real-time) |
| NSNet2 dense (`baseline`) | 2.833 | 2.70 MB | ✅ **22.94 ms/frame, RTF 1.43** (deployed, *not* real-time) |
| other monarch / butterfly | varies | 0.19–9.5 MB | see §2 — `wide_monarch` too big for on-chip; butterfly loses int8 quality |

The one-line story: **the Neural-ART is a small-tensor matmul/conv engine that
wants its weights in on-chip RAM.** ConvFSENet and the *sparse* `monarch_8` fit
and run real-time; the *dense* GRU baseline runs but its 2.70 MB weights
overflow on-chip RAM so it stays memory-bound. Getting the FC/GRU/structured
models to map ranges from "a re-quantize flag" (dense, §1) to "re-export the
model into the right op vocabulary" (monarch, §4).

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

## 3. Why the stock `monarch_8` doesn't compile — and the re-export that fixes it

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

What the stock export does NOT support, and what does:

1. **`Einsum` is lowerable but not the real issue.** `bkp,kqp->bkq` is a
   grouped 1×1 conv; lowering it (and constant-folding the dynamic-shape
   `Pad`/`Reshape` scaffolding with `B=1` + ORT `ORT_ENABLE_EXTENDED`) gets a
   clean, statically-shaped graph — which *still* crashes
   (`Error in computation of shapes` / `Unknown dimensions: H`).
2. **A 4-D grouped-conv re-export compiles in FP32 but not int8.** Rebuilding
   the model with everything as `Conv2d` on `[1,C,1,1]` (grouped for the
   blocks, dense for the boundary layers, channel `Slice` for the trims, **no
   `Pad`**) compiles in FP32. But int8 hits Neural-ART HW-lowering assertions
   (`lower_arith_set_in_batch`, `conf_CONV_IN_ACTIVATION` — batch-dim asserts)
   on the convs that consume the GRU's elementwise outputs. An isolated grouped
   1×1 int8 conv maps to HW fine; chained through the gate math on degenerate
   `1×1` spatial, the conv-DMA path breaks.

### The fix that works: rank-2 `MatMul`, the dense-baseline op vocabulary (§1)

The dense baseline compiles because it is **rank-2 `MatMul` + `Add` on
`[1,N]`** — no `Pad`, no `Einsum`, no grouped conv. So re-express each monarch
block-diagonal projection the same way: **per-block `Slice` + `MatMul` +
`Concat`** (the block-diagonal structure made explicit), states as two flat
`[1,400]` tensors (no `[2,B,400]`+`Gather`), and the gate rewritten
`(1-z)*n + z*h == n + z*(h-n)` to drop the scalar-constant `Sub` that trips the
int8 elementwise HW lowering. This is exactly the op set the compiler already
maps to HW.

`deploy/stm32n6/host/export_monarch_npu.py` does this end-to-end: builds the
rank-2 model from the trained `cp_monarch_8` weights (parity ~5e-7 vs the
trained streaming model), exports FP32, then int8-quantizes with the same VBD
recipe + `skip_optimization=True`.

**Result (int8, `n6-noextmem`, weights on-chip) — two variants measured:**

| metric | `monarch_8` (nblocks 8) | `monarch_full` (nblocks 4) |
| --- | ---: | ---: |
| int8 ONNX | 0.578 MB (0.37 MB on-chip) | 0.859 MB (0.72 MB on-chip) |
| epochs | 134 (76 HW / 30 SW) | 88 (53 HW / 18 SW) |
| MACC / frame | 387,740 | 912,779 |
| **on-target latency** | 2.891 ms (2.885/2.910) | **2.128 ms (2.123/2.146)** |
| **RTF** (16 ms frame) | 0.18 | **0.13** |
| on-target cos (mask / h0 / h1) | 0.99994 / 0.99993 / 0.99995 | 0.99979 / 0.99997 / 0.99990 |
| host equiv. to stock int8 | cos 0.9990 → 2.826 PESQ | cos 0.9995 → 2.848 PESQ |

`monarch_full` is **fastest and best-quality** despite being larger: nblocks=4
means fewer, bigger blocks → fewer/larger matmuls that map more efficiently to
the NPU (88 epochs vs 134). Both fit on-chip and clear real-time comfortably.

Unlike the dense baseline, `stedgeai validate --mode target` **works** for the
monarch models (no GRU `Gemm` fusion to re-introduce), so latency + on-target
cross-accuracy come straight from `validate`; `npu_profiler` PER_LAYER also
works but is slow over serial for a 100+-node graph.

---

## 5. Status / next steps

- **Four models now run on the N6.** The sparse monarch variants dominate:
  `monarch_full` **2.13 ms / RTF 0.13** (best PESQ too, 2.848) and `monarch_8`
  2.89 ms / RTF 0.18, both weights-on-chip, vs ConvFSENet 4.40 ms and dense
  22.94 ms (RTF 1.43, memory-bound). *Structured sparsity is what lets a
  recurrent model hit real-time on this NPU* — it fits the weights in fast RAM.
- **Reproducible & general:** `host/export_monarch_npu.py` deploys any
  fully-monarch config (dims read from the checkpoint; `monarch_8` and
  `monarch_full` both verified end-to-end on the board). `wide_monarch` holds
  int8 PESQ but at 9.5 MB int8 it would not fit on-chip; `monarch_fc` has a
  dense GRU and is rejected by the exporter.
- **PESQ of the deployed artifacts:** each is numerically the stock int8 (host
  mask cos 0.999), so they carry the published PESQ. A from-scratch full-split
  PESQ run on the re-exported int8 would make that airtight (not yet done).
- **Wiring `skip_optimization` into `nsnet2.quant`** as a deploy flag remains a
  small optional cleanup (see §1).
