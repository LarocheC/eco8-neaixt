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
