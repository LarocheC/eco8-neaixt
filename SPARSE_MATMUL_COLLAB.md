# Row Fusion / sparse-dense MatMul — collaboration notes

Working notes for the exchange with Shreya on sparse-dense MatMul packing and
code generation ("Row Fusion"). Branch: `sparse-masks-rowfusion`.

## What her reply establishes

* **It is an inference-side technique.** Packing plus code generation, the
  generator depending on the packing. Training workloads are untested. So the
  split is: we train, she deploys. Masked training here stays dense on the GPU.
* **The mask is a contract, not a hint.** She is working with `2:4`, `4:8`
  (both 50% sparse) and 80% `1x4`-block semi-structured patterns. Other
  structured patterns are possible; fully unstructured sparsity only buys
  memory, not the same speedup.
* **Narrow GEMM is not a blocker.** Her quick test showed similar gains for a
  very narrow GEMM as for a generic one — which was the main risk for us, since
  streaming inference is `(M,K)·(K,1)`. Worth confirming that "very narrow"
  means N=1, and on which ISA/dtype/thread count.
* **Target hardware is still open.** The poster benchmarks BLIS and MKL, i.e.
  x86 CPU. Whether the generator emits Arm/Helium or int8 decides whether this
  reaches the STM32N6 or stays a CPU-side result.

## Why this fits the existing work

The Monarch / block-diagonal NSNet2 sweep already produced the motivating
observation: fewer MACs ≠ lower latency. `monarch` is ~3× *slower* than dense
on the RT595 (transpose-heavy lowering, 35× arena), and on the STM32N6 NPU more
blocks made execution slower. Row Fusion attacks exactly that gap — shape the
sparsity around the kernel instead of around the MAC count.

Note the sparsity families are not interchangeable. The block-diagonal variants
sit at 75% / 87.5% / 95% / 97.5% nominal sparsity (4 / 8 / 20 / 40 blocks), but
block-*diagonal* structure is not the same object as 2:4 or 1×4-block
semi-structured sparsity. Equal sparsity percentage, different kernel
friendliness.

## What is implemented on this branch

`nsnet2/sparsity.py`
: Pattern parsing (`2:4`, `4:8`, `N:M`, `1xB:PCT`, `unstructured:PCT`),
  magnitude mask construction, `MaskedLinear`, and `SparsityController`.

  Two routes to a masked model:
  * `SparsityController` — masks any 2-D parameter of an existing model,
    including `nn.GRU`'s `weight_ih_l*` / `weight_hh_l*`. The architecture is
    untouched, so cuDNN's fused GRU, the streaming path, and the ONNX export all
    keep working. **This is the training route.**
  * `MaskedLinear` (`"kind": "masked"` in `make_linear`) — carries its mask as a
    buffer inside the module. Needed when the mask must survive a module-level
    export rather than living in the training loop.

  Groups run along the **input (K) axis**: contiguous within a row of the
  row-major `(M, K)` weight, matching the NVIDIA 2:4 convention. `axis="out"`
  exists so the assumption can be tested, not asserted.

  Ragged tails: `K=257` in `fc_in` is 64 groups of 4 plus one leftover column.
  Default `tail="keep"` leaves the remainder dense and reports how many elements
  that is, so achieved sparsity never silently disagrees with nominal.

`nsnet2/sparsity_probe.py`
: Retained weight energy `||W⊙M||_F / ||W||_F` per matrix per pattern. A cheap
  pre-screen, run before spending any fine-tuning time.

`nsnet2/export_sparse.py`
: `--dims-only` prints the GEMM shape table. Default mode writes
  `weights.npz` (dense weights with explicit zeros + uint8 masks + biases) and
  `manifest.json` (shape, pattern, group axis, achieved sparsity, ragged-tail
  count, dtype, N at inference vs training). Deliberately not ONNX — the
  compiler side needs matrices and metadata, not a graph.

`configs/sparse_{2to4,4to8,block1x4_80,unstructured_80}.json`
: `baseline.json` plus a `sparsity` block. The unstructured one is the control.

`nsnet2/train.py`
: Builds the controller after the checkpoint load and before DDP, calls
  `mask_grads()` before `optim_g.step()` and `apply()` after it. New
  `--init_from` warm-starts the generator from a dense `g_*` checkpoint.

## Recipe

```bash
# 0. shapes to hand over — no checkpoint needed
python -m nsnet2.export_sparse --config configs/sparse_2to4.json --dims-only

# 1. cheap pre-screen on the dense baseline
python -m nsnet2.sparsity_probe --config cp_nsnet2/config.json \
    --checkpoint cp_nsnet2/g_best

# 2. prune + masked fine-tune from the dense checkpoint
python -m nsnet2.train --config configs/sparse_2to4.json \
    --checkpoint_path cp_nsnet2_sparse24 --init_from cp_nsnet2/g_best \
    --training_epochs 40

# 3. quality, unchanged pipeline (architecture is still dense-shaped)
python -m nsnet2.eval_torch --checkpoint_path cp_nsnet2_sparse24

# 4. hand-off
python -m nsnet2.export_sparse --config cp_nsnet2_sparse24/config.json \
    --checkpoint cp_nsnet2_sparse24/g_best --out export_sparse_2to4
```

Fine-tuning from dense is far cheaper than training each pattern from scratch,
and it makes the comparison honest — every variant starts from the same model.

## Workload to hand over

NSNet2 baseline (`configs/baseline.json`, 2.78 M weights in these 8 matrices,
FP32 today, int8 after PTQ). `y = W·x`, `W` row-major `(M, K)`, `x` `(K, N)`,
**N = 1 for streaming inference**, `N = 256·T` during training.

| matrix              |    M |   K |  params | K mod 4 |
| ------------------- | ---: | --: | ------: | ------: |
| `fc_in`             |  400 | 257 | 102,800 |       1 |
| `gru.weight_ih_l0`  | 1200 | 400 | 480,000 |       0 |
| `gru.weight_hh_l0`  | 1200 | 400 | 480,000 |       0 |
| `gru.weight_ih_l1`  | 1200 | 400 | 480,000 |       0 |
| `gru.weight_hh_l1`  | 1200 | 400 | 480,000 |       0 |
| `fc1`               |  600 | 400 | 240,000 |       0 |
| `fc2`               |  600 | 600 | 360,000 |       0 |
| `fc_out`            |  257 | 600 | 154,200 |       0 |

The four GRU matrices are 69% of the weights and run **once per 16 ms frame**,
so they dominate. The two `weight_hh_l*` are the ones inside the recurrence.

## Retained weight energy, dense VBD baseline (PESQ 2.845)

`||W⊙M||_F / ||W||_F` from magnitude pruning, before any fine-tuning
(`claroche1/sparse-nsnet2-checkpoints`, run `baseline`):

| matrix             |   2:4 |   4:8 | 1x4:80 | unstr:50 | unstr:80 |
| ------------------ | ----: | ----: | -----: | -------: | -------: |
| `fc_in`            | 0.901 | 0.923 |  0.734 |    0.959 |    0.811 |
| `gru.weight_ih_l0` | 0.965 | 0.973 |  0.836 |    0.980 |    0.914 |
| `gru.weight_hh_l0` | 0.945 | 0.958 |  0.837 |    0.980 |    0.897 |
| `gru.weight_ih_l1` | 0.954 | 0.966 |  0.844 |    0.982 |    0.909 |
| `gru.weight_hh_l1` | 0.954 | 0.967 |  0.864 |    0.984 |    0.922 |
| `fc1`              | 0.952 | 0.964 |  0.834 |    0.981 |    0.903 |
| `fc2`              | 0.972 | 0.980 |  0.866 |    0.988 |    0.944 |
| `fc_out`           | 0.980 | 0.986 |  0.859 |    0.993 |    0.952 |
| **all**            | 0.959 | 0.970 |  0.848 |    0.984 |    0.919 |

Reading: at the same 50% sparsity, 4:8 keeps more energy than 2:4 (a looser
constraint, as expected) and unstructured keeps more than both — that gap
(0.984 vs 0.959) is the price of the pattern, and the thing fine-tuning has to
buy back. `fc_in` is consistently the weakest matrix and is also the one with
the ragged tail; it is the first candidate for `exclude` if the fine-tune
struggles. 80% `1x4` is a much bigger cut and should be expected to need real
fine-tuning, not just pruning.

This is a proxy for ordering candidates only. It says nothing about PESQ.

## PESQ after pruning, before any fine-tuning

Full VBD test split (824 utterances), offline forward, same metric the training
loop uses to select `g_best` (`python -m nsnet2.eval_masked`). The unmasked
number reproduces the published 2.845 exactly, so the harness is sound.

| pattern            |  PESQ | Δ vs dense | retained energy |
| ------------------ | ----: | ---------: | --------------: |
| dense (baseline)   | 2.845 |          — |           1.000 |
| unstructured 50%   | 2.799 |     −0.046 |           0.984 |
| 4:8                | 2.533 |     −0.312 |           0.970 |
| 2:4                | 2.467 |     −0.378 |           0.959 |
| 1x4, 80% sparse    | 2.189 |     −0.656 |           0.848 |

The PESQ ordering matches the retained-energy ordering exactly, which is what
makes the cheap screen usable for triage.

The gap between unstructured 50% (−0.046) and 2:4 (−0.378) is the entire cost of
the semi-structured *constraint* — not of the sparsity level. At the same 50%
of weights removed, being forced into 2 per group of 4 costs 8× more PESQ than
free choice does. That is the number worth putting in front of the compiler
side: it is what fine-tuning has to buy back, and it is the reason the pattern
choice is not a detail.

## PESQ after masked fine-tuning

Three arms from the same dense baseline on an identical schedule (lr 3e-4,
60 epochs, validation every 5), so the mask is the only variable. The dense
control is not decoration: it separates what the *mask* costs from what this
fine-tune schedule costs on its own. Same offline PESQ metric throughout;
`nsnet2.eval_masked` on each `g_best` reproduces the training-log value exactly.

| arm                          |  PESQ | vs dense control | vs 200-epoch baseline |
| ---------------------------- | ----: | ---------------: | --------------------: |
| dense baseline (200 epochs)  | 2.845 |                — |                     — |
| `ft_dense_control` (60 ep)   | 2.762 |                — |                −0.083 |
| `ft_sparse_4to8`             | 2.760 |           −0.002 |                −0.085 |
| `ft_sparse_2to4`             | 2.755 |           −0.007 |                −0.090 |

**The headline: 2:4 costs 0.007 PESQ.** Magnitude pruning alone cost 0.378;
60 epochs of masked fine-tuning recover all but 0.007 of it relative to a dense
model given the identical schedule. 4:8, the looser constraint, costs 0.002 —
the ordering the retained-energy screen predicted, but the gap has collapsed to
noise. Half the weights of every FC and GRU matrix are gone for essentially no
speech quality.

**Caveat, and it is not a small one.** The dense control lost 0.083 PESQ against
the published 200-epoch baseline, so this fine-tune schedule is itself harmful:
60 epochs at lr 3e-4 with a *freshly initialised* MetricDiscriminator does not
return to the 200-epoch optimum. The absolute 2.755 is therefore not the best
achievable 2:4 model — warm-starting the discriminator, or simply training the
masked model for the full recipe, should lift all three arms. What is solid is
the comparison, because all three arms ate the same penalty.

Both sparse checkpoints were verified against the declared pattern before export
(`verify_pattern`): every complete group of 4 holds at most 2 nonzeros, in the
saved file rather than in the live model. Feeding the dense control's checkpoint
to the exporter under a `2:4` manifest is correctly refused.

## Overnight sweep: six patterns, 120 epochs each

Same recipe extended to 120 epochs across the pattern space, run in two waves of
three (`run_sparsity_overnight.sh`). Every `g_best` was verified against its
declared pattern and its PESQ independently recomputed with `nsnet2.eval_masked`
— every arm reproduced its training-log value exactly.

| arm                | pattern         | sparsity |  best | last-5 mean | last-5 sd |
| ------------------ | --------------- | -------: | ----: | ----------: | --------: |
| `ov_dense_control` | dense           |       0% | 2.777 |       2.766 |     0.010 |
| `ov_4to8`          | 4:8             |    50.0% | 2.779 |       2.771 |     0.009 |
| `ov_2to4`          | 2:4             |    50.0% | 2.779 |       2.768 |     0.011 |
| `ov_1to4`          | 1:4             |    75.0% | 2.781 |       2.766 |     0.016 |
| `ov_unstruct_80`   | unstructured    |    80.0% | 2.776 |       2.770 |     0.006 |
| `ov_block1x4_80`   | 1x4 blocks      |    80.0% | 2.770 |       2.759 |     0.011 |

**Every pattern is free, and the experiment has hit its resolution limit.** The
spread across all six arms is 0.012 PESQ. The typical *within-arm* variation
across its own last five validations is 0.010 sd / 0.026 range. The differences
between masks are smaller than the noise of a single arm, so these six are
statistically indistinguishable — including 80% sparsity, and including 1:4,
which posted the single highest number (2.781) purely by luck of which
validation happened to land last.

Do not read an ordering into this table. The correct statement is that at 120
epochs of this recipe, mask choice does not move PESQ.

Three things follow:

1. **Pattern choice is entirely hers.** If Row Fusion prefers 4:8 over 2:4, or
   1×4 blocks over N:M, there is no quality argument on our side to weigh
   against it. That is a much stronger position than the 50%-only result, which
   still showed a measurable (if tiny) 2:4-vs-4:8 gap.
2. **The recipe, not the mask, is the binding constraint.** Every arm sits
   ~0.07 below the 200-epoch dense baseline (2.845), the dense control included.
   That gap is the shortened fine-tune with a freshly initialised discriminator,
   not the sparsity. All six curves were still rising at epoch 120 — the last
   validation is the maximum for four of the six — so none has converged.
3. **It is consistent with what this repo already knew.** `monarch_8` reaches
   2.832 FP32 at 0.36 M parameters. NSNet2 is heavily over-parameterised for
   VoiceBank-DEMAND, so 80% sparsity costing nothing is the expected result
   rather than a surprising one.

What this does *not* establish: that 80% is free at int8, or that any of it is
free at deeper sparsity than 80%. Both are open.

Fine-tuning recovery, end to end, for the deepest pattern: `1x4:80` scored 2.189
from magnitude pruning alone and 2.770 after fine-tuning — 0.58 PESQ recovered.
Pruning-only numbers are a triage tool for ordering candidates, never a verdict
on a pattern.

## int8: free for every pattern, and the mask survives exactly

Static int8 PTQ (QDQ, per-channel symmetric weights, MinMax calibration on 200
utterances) applied to all six arms, then PESQ on the full test split through
onnxruntime. Δ is int8 − FP32, so positive means int8 scored *higher*.

| arm                | sparsity |  FP32 |  int8 |      Δ | int8 RTF |
| ------------------ | -------: | ----: | ----: | -----: | -------: |
| `ov_dense_control` |       0% | 2.777 | 2.783 | +0.006 |    0.121 |
| `ov_2to4`          |      50% | 2.779 | 2.781 | +0.002 |    0.125 |
| `ov_4to8`          |      50% | 2.779 | 2.790 | +0.011 |    0.122 |
| `ov_1to4`          |      75% | 2.781 | 2.784 | +0.003 |    0.123 |
| `ov_block1x4_80`   |      80% | 2.770 | 2.779 | +0.009 |    0.124 |
| `ov_unstruct_80`   |      80% | 2.776 | 2.774 | −0.002 |    0.121 |

**Sparsity does not make quantization harder.** Every Δ is within the ±0.01
noise band, at every sparsity level, and five of six are positive — matching the
published dense baseline, which also gained (+0.012) under int8. The open
question from the FP32 sweep is closed: 80% sparsity is free at int8 too.

**The mask survives int8 bit-exactly.** Symmetric per-channel weight
quantization maps 0.0 to exactly 0. Verified in the int8 graphs themselves
(`nsnet2.verify_int8_sparsity`), not assumed:

* `2:4`, `4:8`, `1:4` — every matrix conforms; int8 sparsity lands slightly
  *above* the FP32 target (0.5016 / 0.5011 / 0.7510) because a few surviving
  small weights round to zero, which N:M permits.
* `1x4:80` — block support exactly 0.2000 live against a 0.2000 budget.
  Quantization does zero the occasional value *inside* a kept block, which a
  block-packed kernel absorbs: the block is still stored whole.

So her packer can target the int8 graph, not only FP32 — which matters, since
the deployment targets here are int8 on the M55 and RT595.

**And the punchline: the sparsity buys nothing today.** int8 RTF is 0.121-0.125
across *every* arm — dense and 80%-sparse alike, indistinguishable. We have
removed 80% of the multiplies mathematically and 0% of the latency in practice,
because onnxruntime stores the zeros explicitly and multiplies by them like any
other weight. The int8 file is 2.78 MiB whether or not four fifths of it is
zero. That gap — real zeros, no speedup — is exactly what Row Fusion exists to
close, and it is now measured rather than argued.

## Hand-off

Weights are published at
[`claroche1/nsnet2-sparse-rowfusion`](https://huggingface.co/claroche1/nsnet2-sparse-rowfusion):
one directory per pattern, each with the PyTorch checkpoint (`g_best`,
`config.json`) and a numpy export (`weights.npz` with explicit zeros, uint8
masks, biases and per-matrix golden vectors, plus `manifest.json`). A
numpy-only `verify.py` at the repo root checks shapes, mask/weight agreement,
structural pattern conformance and the golden vectors.

Rebuild the bundle with:

```bash
python -m nsnet2.package_handoff \
    --arm dense=cp_ov_dense_control --arm 2:4=cp_ov_2to4 --arm 4:8=cp_ov_4to8 \
    --arm 1:4=cp_ov_1to4 --arm 1x4:80=cp_ov_block1x4_80 \
    --arm unstructured:80=cp_ov_unstruct_80 --out nsnet2_sparse_handoff
```

The golden vectors are the part a kernel author actually needs: `ref_y = W @
ref_x + b` per matrix, so a generated kernel can be validated one GEMV at a time
without PyTorch or the surrounding model.

## Open questions for the call

1. Does "very narrow GEMM" in her test mean N=1 exactly? Which hardware, dtype,
   thread count?
2. Group orientation: does Row Fusion want the N:M groups along K (row-major
   contiguous, our default) or along M?
3. `1x4` block: contiguous along K, and is the 80% budget global per matrix or
   per row? Per-row keeps the work balanced, which usually suits a kernel better
   — `scope="row"` is implemented and untested against her flow.
4. Ragged `K=257`: leave the tail dense, or pad K to 260?
5. Must the mask be fixed from the start, or can it come from structured pruning
   afterwards? (We assume the latter — dense → prune → fine-tune.)
6. Codegen targets: Arm/Helium as well as x86? int8/int16 as well as FP32? That
   decides whether this reaches the STM32N6 / RT595 or stays CPU-side.
7. Row Fusion across *rows with complementary support* — a Monarch/block-diagonal
   layer has rows from different diagonal blocks with disjoint input support, so
   fusing across blocks looks like a natural fit. Is that the case Row Fusion is
   built for?

## Follow-up, September 2026: the kernel's constraints, and sparse vs a smaller dense model

Three new constraints came from the kernel side, and one question from ours. The
kernel accepts only four of the six 2:4 patterns (a *codebook*, 2 bits per group);
it wants square matrices; and every dimension should be a multiple of 32. The
question: is 50% sparsity better than simply training a smaller dense model with
the same number of nonzero weights?

**Answers.**

* **The codebook is free on NSNet2 and costs ConvFSENet ~0.025 PESQ.** Each of the
  three candidate codebooks is one way of pairing positions {0,1,2,3} and keeping
  one weight per pair (c1 = (0,1)(2,3) is plain 1:2). On NSNet2, c1/c2/c3 land
  within noise of free 2:4 and the codebook mask is free against the unmasked
  model of the same size. On ConvFSENet below the knee it trails plain 2:4 by
  0.028 / 0.023 (behind at every validation, both seeds).
* **Square, multiple-of-32 shapes cost at most ~0.02 on NSNet2** (`sq384`,
  hidden = fc = 384). A (3H, H) GRU weight splits into three square (H, H) gates
  bit-exactly, and the collaborator confirmed that counts as square; the 257-bin
  frequency axis is padded on her side at export.
* **Sparse vs a smaller dense model depends on the model.** Where the dense
  alternative is short of capacity (below the knee of the size curve):
  NSNet2's 2:4@c1 model beats a dense model with about as many nonzero weights by
  **+0.042** (both seeds, every validation); ConvFSENet's loses by **0.020**
  (0.024 after crediting its 7.8% extra weights), and plain 2:4 only ties. Above
  the knee (NSNet2 at ~1.2 M) the comparison is unresolved (+0.010 / -0.002).
* **The 120-epoch NSNet2 fine-tune itself is the largest NSNet2 loss**: 0.012 to
  0.061 below each parent's last-5 mean, masked or not. The model and recipe were
  kept unchanged across all runs for consistency.

### Designs and decision rules

Every rule below was committed before the runs it governs.

| Study | Runner | Commit | Design |
| --- | --- | --- | --- |
| Codebooks c1/c2/c3 on NSNet2 | `run_codebook_wave.sh` | 2c0150c | 120-ep fine-tunes from the 2.845 baseline, same recipe as the `ov_*` wave |
| Square/32 NSNet2 + compute-matched control | `run_square32_wave.sh`, `run_stageC_seed.sh` | b755391, e388686 | `sq384` and `dense_h256f384` parents 200 ep from scratch, then 120-ep fine-tunes, two seeds |
| NSNet2 below the knee | `run_knee_wave.sh` | 961541e | dense 128/128 vs 2:4@c1 pruned from 192/192 (+4.5% live), two seeds |
| ConvFSENet below the knee | `run_cf_knee_wave.sh` | 95cd1b1 | dense 64/128 vs 2:4@c1 pruned from 96/192 (+7.8% live), plain 2:4 as a secondary arm, two seeds |

For the below-the-knee studies: U = unmasked big model minus small dense model
(capacity left after the fine-tune), D = sparse model minus small dense model.
U < 0.02 means no power; with power, D (size-adjusted for ConvFSENet) >= U/2 and
positive in both seeds means sparsity is a real improvement, D <= 0.01 means it is
not. NSNet2: U = +0.027, D = +0.042 -> real improvement. ConvFSENet: U = +0.035,
D_adj = -0.024 -> not an improvement.

### Per-run results

Generated by `python -m nsnet2.sparsity_results --markdown`; the same numbers,
with every run's full PESQ trajectory, are in `SPARSE_MATMUL_RUNS.csv`. Best is
the maximum validation PESQ (optimistic: `g_best` is chosen on the same test
split); last-5 is the mean of the final five validations.

#### nsnet2-full-size

| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ov_dense_control` | 400/600 | dense | hf:baseline/g_best | 120 | 1234 | 2,783,657 | 2.777 | 2.766 |
| `ov_2to4` | 400/600 | 2:4 | hf:baseline/g_best | 120 | 1234 | 1,395,357 | 2.779 | 2.768 |
| `ov_4to8` | 400/600 | 4:8 | hf:baseline/g_best | 120 | 1234 | 1,395,357 | 2.779 | 2.771 |
| `ov_1to4` | 400/600 | 1:4 | hf:baseline/g_best | 120 | 1234 | 701,207 | 2.781 | 2.766 |
| `ov_unstruct_80` | 400/600 | unstructured:80 | hf:baseline/g_best | 120 | 1234 | 562,057 | 2.776 | 2.770 |
| `ov_block1x4_80` | 400/600 | 1x4:80 | hf:baseline/g_best | 120 | 1234 | 562,377 | 2.770 | 2.759 |
| `cb_c1` | 400/600 | 2:4@c1 | cp_baseline/g_best | 120 | 1234 | 1,395,357 | 2.772 | 2.763 |
| `cb_c2` | 400/600 | 2:4@c2 | cp_baseline/g_best | 120 | 1234 | 1,395,357 | 2.790 | 2.777 |
| `cb_c3` | 400/600 | 2:4@c3 | cp_baseline/g_best | 120 | 1234 | 1,395,357 | 2.774 | 2.763 |

#### nsnet2-square32

| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sq384_dense` | 384/384 | dense | scratch | 200 | 1234 | 2,267,777 | 2.835 | 2.801 |
| `sq384_cb_c1` | 384/384 | 2:4@c1 | cp_sq384_dense/g_best | 120 | 1234 | 1,137,089 | 2.761 | 2.754 |
| `sq384_cb_c1_s2345` | 384/384 | 2:4@c1 | cp_sq384_dense/g_best | 120 | 2345 | 1,137,089 | 2.759 | 2.751 |
| `sq384_ft` | 384/384 | dense | cp_sq384_dense/g_best | 120 | 1234 | 2,267,777 | 2.755 | 2.747 |
| `sq384_ft_s2345` | 384/384 | dense | cp_sq384_dense/g_best | 120 | 2345 | 2,267,777 | 2.749 | 2.740 |
| `dense_h256f384` | 256/384 | dense | scratch | 200 | 1234 | 1,201,025 | 2.833 | 2.765 |
| `dense_h256f384_ft` | 256/384 | dense | cp_dense_h256f384/g_best | 120 | 1234 | 1,201,025 | 2.749 | 2.744 |
| `dense_h256f384_ft_s2345` | 256/384 | dense | cp_dense_h256f384/g_best | 120 | 2345 | 1,201,025 | 2.761 | 2.753 |

#### nsnet2-knee

| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sq128_dense` | 128/128 | dense | scratch | 200 | 1234 | 297,345 | 2.782 | 2.768 |
| `sq192_dense` | 192/192 | dense | scratch | 200 | 1234 | 617,921 | 2.799 | 2.782 |
| `sq192_cb_c1` | 192/192 | 2:4@c1 | cp_sq192_dense/g_best | 120 | 1234 | 310,625 | 2.761 | 2.754 |
| `sq192_cb_c1_s2345` | 192/192 | 2:4@c1 | cp_sq192_dense/g_best | 120 | 2345 | 310,625 | 2.762 | 2.754 |
| `sq192_ft` | 192/192 | dense | cp_sq192_dense/g_best | 120 | 1234 | 617,921 | 2.749 | 2.743 |
| `sq192_ft_s2345` | 192/192 | dense | cp_sq192_dense/g_best | 120 | 2345 | 617,921 | 2.746 | 2.735 |
| `sq128_ft` | 128/128 | dense | cp_sq128_dense/g_best | 120 | 1234 | 297,345 | 2.722 | 2.710 |
| `sq128_ft_s2345` | 128/128 | dense | cp_sq128_dense/g_best | 120 | 2345 | 297,345 | 2.723 | 2.714 |

#### nsnet2-size-curve

| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dense_h52` | 52/78 | dense | scratch | 200 | 1234 | 77,087 | 2.749 | 2.733 |
| `dense_h68` | 68/102 | dense | scratch | 200 | 1234 | 117,863 | 2.751 | 2.719 |
| `dense_h68_s2345` | 68/102 | dense | scratch | 200 | 2345 | 117,863 | 2.754 | 2.724 |
| `dense_h68_s3456` | 68/102 | dense | scratch | 200 | 3456 | 117,863 | 2.755 | 2.736 |
| `dense_h100` | 100/150 | dense | scratch | 200 | 1234 | 223,607 | 2.784 | 2.760 |
| `dense_h148` | 148/222 | dense | scratch | 200 | 1234 | 442,703 | 2.815 | 2.775 |
| `dense_h168` | 168/252 | dense | scratch | 200 | 1234 | 555,413 | 2.840 | 2.803 |
| `dense_h168_s2345` | 168/252 | dense | scratch | 200 | 2345 | 555,413 | 2.824 | 2.776 |
| `dense_h168_s3456` | 168/252 | dense | scratch | 200 | 3456 | 555,413 | 2.834 | 2.806 |
| `dense_h216` | 216/324 | dense | scratch | 200 | 1234 | 877,325 | 2.783 | 2.744 |
| `dense_h216_s2345` | 216/324 | dense | scratch | 200 | 2345 | 877,325 | 2.814 | 2.771 |
| `dense_h216_s3456` | 216/324 | dense | scratch | 200 | 3456 | 877,325 | 2.821 | 2.773 |

#### convfsenet-full-size

| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cf_dense_control` | 192/384 | dense | cp_convfsenet_win/g_best | 20 | 1234 | 1,453,889 | 2.832 | 2.829 |
| `cf_2to4` | 192/384 | 2:4 | cp_convfsenet_win/g_best | 20 | 1234 | 741,089 | 2.820 | 2.817 |
| `cf_c1` | 192/384 | 2:4@c1 | cp_convfsenet_win/g_best | 20 | 1234 | 741,089 | 2.805 | 2.799 |

#### convfsenet-knee

| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cfs_c64_dense` | 64/128 | dense | scratch | 200 | 1234 | 189,889 | 2.834 | 2.828 |
| `cfs_c96_dense` | 96/192 | dense | scratch | 200 | 1234 | 395,297 | 2.855 | 2.847 |
| `cf96_cb_c1` | 96/192 | 2:4@c1 | cp_cfs_c96_dense/g_best | 20 | 1234 | 204,785 | 2.766 | 2.761 |
| `cf96_cb_c1_s2345` | 96/192 | 2:4@c1 | cp_cfs_c96_dense/g_best | 20 | 2345 | 204,785 | 2.789 | 2.787 |
| `cf96_2to4` | 96/192 | 2:4 | cp_cfs_c96_dense/g_best | 20 | 1234 | 204,785 | 2.794 | 2.788 |
| `cf96_2to4_s2345` | 96/192 | 2:4 | cp_cfs_c96_dense/g_best | 20 | 2345 | 204,785 | 2.816 | 2.810 |
| `cf96_ft` | 96/192 | dense | cp_cfs_c96_dense/g_best | 20 | 1234 | 395,297 | 2.833 | 2.824 |
| `cf96_ft_s2345` | 96/192 | dense | cp_cfs_c96_dense/g_best | 20 | 2345 | 395,297 | 2.847 | 2.834 |
| `cf64_ft` | 64/128 | dense | cp_cfs_c64_dense/g_best | 20 | 1234 | 189,889 | 2.795 | 2.793 |
| `cf64_ft_s2345` | 64/128 | dense | cp_cfs_c64_dense/g_best | 20 | 2345 | 189,889 | 2.805 | 2.795 |

#### convfsenet-size-curve

| Run | Width | Mask | Init | Epochs | Seed | Nonzero params | Best | Last-5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cfs_dense_r67` | 67/134 | dense | scratch | 200 | 1234 | 206,014 | 2.847 | 2.816 |
| `cfs_dense_r83` | 83/166 | dense | scratch | 200 | 1234 | 302,958 | 2.882 | 2.853 |
| `cfs_dense_r108` | 108/216 | dense | scratch | 200 | 1234 | 491,333 | 2.872 | 2.857 |
| `cfs_dense_r146` | 146/292 | dense | scratch | 200 | 1234 | 863,847 | 2.886 | 2.851 |
| `cfs_dense` | 192/384 | dense | scratch | 200 | 1234 | 1,453,889 | 2.877 | 2.860 |

#### Paired comparisons (Δ = first − second, mean over the last 5 validations)

| Study | Comparison | Δ | First run leads |
| --- | --- | ---: | ---: |
| nsnet2-full-size | codebook c1 vs free 2:4 | -0.005 | 4/11 |
| nsnet2-full-size | codebook c2 vs free 2:4 | +0.009 | 6/11 |
| nsnet2-full-size | codebook c3 vs free 2:4 | -0.005 | 4/11 |
| nsnet2-square32 | mask cost at 384 (C2), seed 1234 | +0.007 | 8/11 |
| nsnet2-square32 | mask cost at 384 (C2), seed 2345 | +0.011 | 10/11 |
| nsnet2-square32 | sparse vs matched dense at ~1.2M (C3), seed 1234 | +0.010 | 11/11 |
| nsnet2-square32 | sparse vs matched dense at ~1.2M (C3), seed 2345 | -0.002 | 7/11 |
| nsnet2-knee | D: sparse 192 vs dense 128, seed 1234 | +0.044 | 11/11 |
| nsnet2-knee | D: sparse 192 vs dense 128, seed 2345 | +0.040 | 11/11 |
| nsnet2-knee | U: dense 192 vs dense 128, seed 1234 | +0.033 | 11/11 |
| nsnet2-knee | U: dense 192 vs dense 128, seed 2345 | +0.021 | 11/11 |
| nsnet2-knee | mask cost at 192, seed 1234 | +0.011 | 11/11 |
| nsnet2-knee | mask cost at 192, seed 2345 | +0.019 | 11/11 |
| convfsenet-full-size | free 2:4 vs dense | -0.012 | 0/6 |
| convfsenet-full-size | codebook c1 vs free 2:4 | -0.018 | 0/6 |
| convfsenet-knee | D: sparse 96 (c1) vs dense 64, seed 1234 | -0.032 | 0/19 |
| convfsenet-knee | D: sparse 96 (c1) vs dense 64, seed 2345 | -0.008 | 0/19 |
| convfsenet-knee | U: dense 96 vs dense 64, seed 1234 | +0.031 | 19/19 |
| convfsenet-knee | U: dense 96 vs dense 64, seed 2345 | +0.039 | 19/19 |
| convfsenet-knee | plain 2:4 96 vs dense 64, seed 1234 | -0.004 | 2/19 |
| convfsenet-knee | plain 2:4 96 vs dense 64, seed 2345 | +0.015 | 17/19 |
| convfsenet-knee | codebook vs free 2:4 at 96, seed 1234 | -0.028 | 0/19 |
| convfsenet-knee | codebook vs free 2:4 at 96, seed 2345 | -0.023 | 0/19 |
| convfsenet-knee | mask cost at 96 (c1), seed 1234 | -0.063 | 0/19 |
| convfsenet-knee | mask cost at 96 (c1), seed 2345 | -0.047 | 0/19 |

### Caveats

* **Test-split selection.** `g_best` is picked on the reported test split, so best
  scores are best-of-N; every comparison above uses last-5 or paired means.
* **PyTorch upgrade.** PyTorch 2.14 was installed on 2026-09-17. The `ov_*` and
  full-size `cf_*` controls predate it; comparisons across that date (codebook vs
  free 2:4 on NSNet2 at full size, the full-size ConvFSENet codebook row, the
  square-shape comparison against `ov_dense_control`) carry that confound. The
  below-the-knee studies and the square/32 study are same-version throughout.
* **One parent per size.** In both below-the-knee studies the two seeds vary the
  fine-tune only. Dense parents of one size differ by up to 0.04 across seeds in
  this repo, more than the parent gaps here (NSNet2 0.014, ConvFSENet 0.019).
* **Short ConvFSENet fine-tune.** 20 epochs, the recipe of the earlier ConvFSENet
  mask tests; the pruned runs were still rising slowly at the end.
* **ConvFSENet knee.** It rested on a single-seed `r67` point that came in low: the
  new parents landed 0.019 apart against 0.048 predicted.
