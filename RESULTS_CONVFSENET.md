# ConvFSENet — results

A fully-convolutional, causal speech enhancer. ConvFSENet is a
ConvTasNet-derived magnitude-mask predictor built from stacked Temporal
Conv Module (TCM) blocks (1×1 conv + depthwise dilated conv, BatchNorm,
ReLU, residual). The architecture here is the base enhancer from
Miccini, Laroche, Piechowiak & Pezzarossa, *Scalable Speech Enhancement
with Dynamic Channel Pruning*
([ICASSP 2025, arXiv:2412.17121](https://arxiv.org/abs/2412.17121)) —
this repo uses the static (un-pruned) variant of that model.
It runs **frame-by-frame**: each TCM block keeps a small FIFO state
buffer so the dilated convs stream with zero lookahead. The streaming
wrapper, FP32/int8 ONNX export and the ORT inference path are all
parity-checked against the offline model (`tests/test_convfsenet_*`).

See [RESULTS_NSNET2.md](RESULTS_NSNET2.md) for the recurrent model
family, and the [README](README.md) for setup and repo layout.

## Headline results

200-epoch training on VoiceBank-DEMAND-16k with an end-to-end
time-domain loss (`DynCompMSE`), a PESQ metric-GAN
(`MetricDiscriminator`), and a cosine LR schedule. PESQ on the full
824-utterance VBD test split; RTF is the int8 streaming session under
onnxruntime CPU (single thread), lower is faster.

| metric         | value     |
| -------------- | --------: |
| params         | 1.45 M    |
| FP32 PESQ      | **2.931** |
| int8 PESQ      | **2.911** |
| Δ (FP32→int8)  | +0.020    |
| int8 RTF       | 0.017     |
| int8 size      | 1.6 MiB   |

Static int8 PTQ is essentially loss-free (0.020 PESQ drop). The FP32
score of 2.931 beats NSNet2's dense baseline (2.845) and every
structured NSNet2 variant — at a fraction of the RTF (NSNet2 ranges
0.025–0.452 under the same onnxruntime-CPU conditions). The PESQ
metric-GAN is what pushes the FP32 score past 2.93; without it the same
architecture tops out around 2.77–2.79.

### Feature extractor — implementation note

ConvFSENet feeds `(|stft|+eps)^0.3` to the frontend Conv
(`extractor_type: mag_compressed` in the config). The compression
squashes the 60+ dB STFT magnitude range into an int8-friendly domain —
the same idea as NSNet2's log-magnitude input — and must stay in FP32 at
deployment.

`convfsenet/quant.py` automatically keeps the compression prologue
(`Add` / `Pow` / `Unsqueeze`) out of quantization: it walks the ONNX
graph from the magnitude input to the first Conv and adds those node
names to `quantize_static`'s `nodes_to_exclude`. Without this exclusion
`quantize_static` would treat the eps-`Add` as quantizable and drag the
raw `|stft|` onto a coarse int8 grid before the compression runs,
destroying exactly the low-energy detail the compression exists to
preserve.

## STM32N6 on-board deployment (preliminary)

First measurements of the int8 streaming model on real hardware: an
**STM32N6570-DK** (STM32N657 — Cortex-M55 @ 800 MHz + Neural-ART NPU @
1 GHz), compiled with ST Edge AI Core 4.0.1 and run via the bundled
`NPU_Validation` firmware. Single on-target `stedgeai validate` run
(10 random samples) of the per-frame graph, weights in external xSPI
flash, fully scripted — no STM32CubeIDE — through the `deploy/stm32n6/`
setup. **Preliminary** numbers, one model, one run.

| metric (on-target)                  |       value |
| ----------------------------------- | ----------: |
| inference latency / frame           | **7.21 ms** |
| frame period (hop 256 @ 16 kHz)     |       16 ms |
| hardware real-time factor           |  **≈ 0.45** |
| compute split (NPU / SW / SW-ctrl)  | 51.7% / 20.9% / 27.4% |
| mask cosine vs FP32 ONNX (on-target)|       0.990 |
| MACC / frame                        |      1.47 M |
| weights (external xSPI flash)       |    1.40 MiB |
| activations (on-chip SRAM)          |     ~34 KiB |

Real-time with ~2.2× headroom (7.2 ms inference against the 16 ms frame
budget, before the M55 STFT/iSTFT front-end). Note that **~48% of the
time runs on the Cortex-M55, not the NPU**: the convolutions map to the
Neural-ART accelerator, but the per-frame FIFO state handling
(Slice/Gather) and the int8 quant boundary fall back to software. That
software share — not the convs — is the lever for going faster. The
on-target int8 mask tracks the FP32 ONNX reference at cosine 0.990,
consistent with the loss-free PTQ above.

Caveats: only ConvFSENet currently compiles for the Neural-ART — NSNet2
(dense and structured) crashes the ST Edge AI compiler at this version;
the validation firmware is a volatile RAM image; and the headline
`int8 RTF 0.017` above is onnxruntime-CPU, not comparable to this 0.45
on-device factor. See `deploy/stm32n6/` for the generate → build → flash
→ gdb-load → validate procedure.

**Optimization — weight locality (npuRAM vs external flash).** The
`allmems` profile parks all 1.44 MB of weights in external octoFlash, and
the counters show the NPU is memory-bound: it re-reads the full weight set
from flash every frame (201 MB/s avg) at only ~27% core compute
utilization. The model is small enough to fit entirely on-chip, so
regenerating with the `n6-noextmem` profile packs weights into internal
npuRAM3/4/5/6. On-board latency then drops **7.14 → 4.40 ms/frame (1.62×;
RTF 0.45 → 0.275)**, NPU core time 3.73 → 1.26 ms, and core compute
utilization rises to **81%** — the NPU flips from memory-bound to
compute-bound. The remaining ~3.1 ms is the Cortex-M55 software share (the
per-frame FIFO state + int8 quant boundary). Caveat: in this layout the
weights load over the debugger (gdb) rather than being flashed; a
standalone power-on deploy needs an on-chip-resident boot layout.

## Stateless-windowed rework (Track 1 — removes the FIFO M55 floor)

That remaining ~3.1 ms M55 share is the per-block FIFO state plumbing
(`Slice`/`Concat`/`Gather`, always Hybrid). Track 1 of the efficiency rework
(`deploy/stm32n6/EFFICIENCY_REWORK_PLAN.md`) removes it entirely: instead of
threading per-block FIFO state across single frames, the host keeps a ring
buffer of the last `L = sum_blocks (K-1)*D = 42` magnitude columns and feeds a
fixed `[1, n_freq, L+T]` window through the BN-folded offline-causal model run as
**valid (padding-0) convs**. The dilated dconvs shrink the time axis by `(K-1)*D`
per block (residual cropped to match), so `L+T → T` with no state I/O and no Pad.
The exported int8 graph is **stateless** — `Conv`/`Add`/`Relu`/`Sigmoid` + 9
static residual-crop `Slice`s, **zero `Gather`/`Pad`/state/BatchNorm**
(`convfsenet/streaming.py:ConvFSENetWindowedONNX`, exported via
`export_onnx.py --windowed`, quantized via `quant_windowed.py`). FP32 is bit-exact
(<1e-6) to the offline causal model on full-context frames
(`tests/test_convfsenet_windowed_parity.py`, 34/34).

**Host PESQ (full 824-utt VBD test, no retrain — same v5 weights):**

| variant (int8) | FP32 PESQ | int8 PESQ | Δ |
| --- | ---: | ---: | ---: |
| streaming reference (deployed) | 2.931 | 2.911 | +0.020 |
| windowed-257, `coldstart=zero` | 2.858 | 2.836 | +0.022 |
| windowed-256, `coldstart=zero` | 2.865 | 2.843 | +0.022 |
| windowed-257, `coldstart=replicate` | 2.923 | 2.904 | +0.019 |
| **windowed-256, `coldstart=replicate` (deploy)** | **2.933** | **2.913** | +0.020 |

The 256-bin variant drops the Nyquist bin (frontend input + backend output) for
power-of-two HW alignment — it is **PESQ-neutral** (256 ≥ 257), so no 256-native
fine-tune was needed. The whole apparent gap was the **cold start**: the model
was trained with a zero-*activation* history before t=0, but a zero-*magnitude*
ring buffer feeds the frontend bias (`frontend(0) ≠ 0`) for the first `<L` frames
→ out-of-distribution, costing ~0.045 PESQ on short clips. Seeding the ring
buffer by **replicating the first frame** (`coldstart=replicate`, the default —
causal, no look-ahead, on-device deployable) recovers it: the windowed-256 int8
**2.913 matches/slightly beats the streaming 2.911**, at the documented int8 gate
≥2.85 (target 2.90–2.91, exceeded).

Net: same quality as the deployed streaming model, but the int8 deploy graph has
**no FIFO/state/Pad** class — the M55-Hybrid floor that capped ConvFSENet at 4.40
ms. On-board latency (does the per-frame epoch count drop, do the convs fill the
array at `h:43`?) is the Gate-0/Phase-4 verdict on the deploy box — see
[deploy/stm32n6/WINDOWED_DEPLOY_HANDOFF.md](deploy/stm32n6/WINDOWED_DEPLOY_HANDOFF.md).

### Track 2 (small-STFT block) — rejected on quality

Track 2 retrained the same 192/384 windowed backbone at a **smaller STFT**
(`n_fft 256 / hop 128`, 129 bins, 16 ms/8 ms framing, emit T=2) for 2× block
amortization under ~30 ms latency (`configs/convfsenet_win_smallstft.json`,
200-epoch GAN from scratch). The coarser frequency resolution costs too much
quality: windowed-128 int8 PESQ **2.725** (FP32 2.783) — **0.13 below the ≥2.85
gate**, ~0.19 below Track 1's 2.913. This is the plan's flagged Config-C risk
("accept only if PESQ holds ≥2.85") materializing. **Rejected** — Track 1
(512/256 windowed) stays the winner; the small-STFT latency win isn't worth the
PESQ loss.

## Structured pointwise convolutions — Monarch block-count sweep

ConvFSENet spends **92.4% of its per-frame MACs in eighteen 1x1 convolutions**
(`conv1x1` 192→384 and `conv1x1_out` 384→192 in each of the nine TCM blocks);
the frontend and backend add 6.9% and the nine depthwise convs 0.7%. That makes
it the same question `RESULTS_NSNET2.md` asks of NSNet2's FC and GRU
projections, on a model with no recurrence: **once you take MACs out of the
pointwise convolutions, is it better to spend what is left on structure at full
width, or on a plain dense model that is simply narrower?**

A 1x1 conv over `(B, C, T)` is a linear map applied per time step, so every
structured factorization that applies to `nn.Linear` applies here.
`convfsenet/layers.py` provides two, selected by a config block
(`"pointwise": {"kind": "monarch", "nblocks": 8, "scope": "tcm"}`; absent means
plain `nn.Conv1d`, bit-identical to the dense model):

- **`blockdiag`** — one block-diagonal factor, zero cross-block mixing.
- **`monarch`** — genuine two-factor Monarch (block-diagonal × permutation ×
  block-diagonal, [Dao et al. 2022](https://arxiv.org/abs/2204.00595)), whose
  permutation gives cross-block mixing that `blockdiag` has none of — but *full*
  mixing only while `nblocks ≤ √in_channels` (see
  [the reach column](#the-sweep)).

Both are numerically the layers `torch-structured` ships (`MonarchLinear`,
`BlockdiagLinear`) — same factor shapes, and the same weights bit-for-bit from a
common seed, asserted in `tests/test_convfsenet_monarch.py`. They are written
differently: as **two grouped 1x1 convolutions separated by a channel shuffle**
rather than reshape+einsum on a channels-last tensor. See
[Lowering](#lowering-grouped-convolutions-not-einsum) below for why that matters.

### Prior art on this model — block-diagonal masks (2026-08-20)

Four 200-epoch arms on the unmodified recipe, from-scratch, seed 1234. These
used *fixed masks* on the dense weights (`sparsity: {pattern: "blockdiag:N"}`,
all 20 pointwise matrices including frontend/backend), so the parameter count
never changes — they measure what the loss of cross-block connectivity costs,
not what the compression buys. Recorded here because they existed in no tracked
file. MACs/frame assume the mask is exploited.

| run                 | pattern        | params  | MACs/frame (if exploited) | FP32 PESQ |
| ------------------- | -------------- | ------: | ------------------------: | --------: |
| `cfs_dense`         | dense          | 1.454 M |                 1,436,160 | **2.877** |
| `cfs_blockdiag2`    | blockdiag:2    | 1.454 M |                   723,264 |     2.804 |
| `cfs_blockdiag4`    | blockdiag:4    | 1.454 M |                   366,816 |     2.783 |
| `cfs_blockdiag8`    | blockdiag:8    | 1.454 M |                   188,592 |     2.722 |

**Block-diagonal degrades monotonically** — −0.155 PESQ from dense to
`blockdiag:8` — the same direction NSNet2's block-count sweep found, and the
reason the Monarch arms are worth training. PESQ is the best of 19 validations
on the full 824-utterance VBD test split (`convfsenet/train.py` validates on the
test split, and `convfsenet/eval_ptq.py` re-scores `cp_cfs_dense/g_best` at
2.877, so the training-loop metric and the headline metric are one scale).

Note `cfs_dense` (2.877) sits 0.054 below the flagship 2.931 at the top of this
document despite a byte-identical architecture; the configs differ only in
`num_workers` (3 vs 4). Treat 2.877 — not 2.931 — as the anchor for everything
below, which is why the sweep reuses that run rather than the flagship.

### The sweep

Two arms, matched on **MACs per frame** (within 1.34%), one knob each. Arm A
holds the width at 192/384 and raises the Monarch block count; arm B holds the
structure at none and lowers the width, keeping the model's own 1:2
res:conv ratio.

| nblocks | Monarch MACs/frame |   params | reach (192→384 / 384→192) | dense control | dense MACs/frame |   params | MAC Δ |
| ------: | -----------------: | -------: | ------------------------: | ------------- | ---------------: | -------: | ----: |
|       4 |            855,552 |  873,281 |             100% / 100%   | 146 / 292     |          850,304 |  863,847 | −0.61% |
|       8 |            482,304 |  500,033 |             100% / 100%   | 108 / 216     |          481,248 |  491,333 | −0.22% |
|      16 |            295,680 |  313,409 |            **75%** / 100% |  83 / 166     |          295,148 |  302,958 | −0.18% |
|      32 |            202,368 |  220,097 |         **18.8% / 37.5%** |  67 / 134     |          199,660 |  206,014 | −1.34% |

**Reach** is the fraction of output channels one input channel influences —
measured, by perturbation. It is 100% only while `nblocks ≤ √in_channels`; past
that each output channel sees `min(in_channels/nblocks, nblocks)` of the
`nblocks` input blocks. This is Monarch's own property, not an artifact of the
implementation here (`torch_structured`'s `MonarchLinear` measures identically),
and it is why the column exists: **at `nblocks=32` the Monarch arm is not
holding connectivity fixed while trading MACs** — it is partway toward the
block-diagonal regime, and a result there is not evidence about "structure at
full width". Reach still beats block-diagonal's `1/nblocks` by 6× and 12×.

For calibration: NSNet2's published `monarch_40` — the arm that reaches dense
parity at 24× fewer parameters (2.837 vs the 2.845 dense baseline, against
`blockdiag_40`'s 2.608) — has reach **25%** on its GRU projection and 17.5% on
`fc_in`. So partial reach is not disqualifying; it is a variable to report.

Configs `configs/cfs_mon_nb{4,8,16,32}.json` and `configs/cfs_dense_r{146,108,83,67}.json`,
each derived from `cp_cfs_dense/config.json` so the only difference is the swept
knob. Driver: `run_convfsenet_monarch_sweep.sh` (`PREFLIGHT=1` checks the
configs without launching; `WAVE=1` runs the two decisive pairings).

Three things to keep in view when reading the results:

- **The grid starts at 4, not 2.** Monarch's first factor scales with `in²`, so
  the expanding 192→384 layer compresses `2·nblocks/3` but the contracting
  384→192 layer only `nblocks/3`. At `nblocks=2` the "compressed" model is
  **1.12× larger** than the dense one (1,602,048 MACs/frame vs 1,436,160).
- **Arm A has an un-structured floor.** The frontend and backend stay dense
  (257 is prime, so any `nblocks` would zero-pad, and the int8 prologue walk and
  the Nyquist slicing both assume a dense Conv there), so 109,056 MACs/frame are
  untouched: 12.7% / 22.6% / 36.9% / **53.9%** of the four arms. At `nblocks=32`
  half the model is not structured, and arm A cannot compress past ~9.2× at all.
- **MAC-matching hands arm A more parameters** — `params − MACs` is exactly
  17,729 for every Monarch arm (BN + biases, whose widths structure does not
  change) while the dense control shrinks that overhead too. The Monarch edge is
  +1.08 / +1.74 / +3.33 / +6.40%.

### Results — dense narrowing wins at every MAC budget

All nine arms, one recipe (200 epochs, from scratch, seed 1234), FP32 and int8
PESQ measured in one pass on the full 824-utterance VBD test split. **best** is
the max over the 19 validations (what `g_best` selects on, and what every
earlier table in this document reports); **last-5** is the mean of the final
five validations, which no selection touched — carried because the selection
margin is not uniform across arms (0.004 to 0.035), so `best` alone could be
reading the spread of the tail rather than the model. RTF is the int8 streaming
session under onnxruntime CPU; lower is faster.

| arm | MACs/frame | params | reach | FP32 (best) | FP32 (last-5) | int8 | Δ int8 | RTF |
| --- | ---------: | -----: | ----: | ----------: | ------------: | ---: | -----: | --: |
| `cfs_dense` *(anchor)* | 1,436,160 | 1.454 M | dense | 2.877 | 2.860 | 2.871 | 0.005 | 0.015 |
| `cfs_mon_nb4`   |   855,552 | 0.873 M | 100% / 100% | 2.857 | 2.838 | 2.789 | 0.065 | 0.015 |
| `cfs_dense_r146`|   850,304 | 0.864 M | dense       | **2.886** | **2.851** | **2.878** | 0.011 | 0.012 |
| `cfs_mon_nb8`   |   482,304 | 0.500 M | 100% / 100% | 2.854 | 2.830 | 2.790 | 0.063 | 0.014 |
| `cfs_dense_r108`|   481,248 | 0.491 M | dense       | **2.872** | **2.857** | **2.859** | 0.012 | 0.010 |
| `cfs_mon_nb16`  |   295,680 | 0.313 M | 75% / 100%  | 2.816 | 2.812 | 2.744 | 0.072 | 0.015 |
| `cfs_dense_r83` |   295,148 | 0.303 M | dense       | **2.882** | **2.853** | **2.862** | 0.019 | 0.008 |
| `cfs_mon_nb32`  |   202,368 | 0.220 M | 19% / 38%   | 2.811 | 2.802 | 2.761 | 0.048 | 0.016 |
| `cfs_dense_r67` |   199,660 | 0.206 M | dense       | **2.847** | **2.816** | **2.833** | 0.012 | 0.008 |

**The dense control wins every MAC-matched pairing, on every statistic.**

| MACs/frame | Monarch | dense control | Δ FP32 (best) | Δ FP32 (last-5) | Δ int8 |
| ---------: | ------- | ------------- | ------------: | --------------: | -----: |
|    855,552 | 2.857   | **2.886**     |        +0.029 |          +0.013 | **+0.089** |
|    482,304 | 2.854   | **2.872**     |        +0.018 |          +0.027 | **+0.069** |
|    295,680 | 2.816   | **2.882**     |        +0.066 |          +0.042 | **+0.118** |
|    202,368 | 2.811   | **2.847**     |        +0.036 |          +0.014 | **+0.072** |

Four pairings, three statistics, twelve comparisons, all the same direction.
**This is the opposite of the NSNet2 result**, where genuine Monarch beat a
param-matched dense NSNet2 at all five sizes by 0.021–0.086 and the gap widened
as the models shrank. Same structure, same library, same trainer, opposite
conclusion — so "structured matrices beat narrow dense" is not a property of
Monarch. See [Why it reverses](#why-it-reverses) below.

#### int8: the two-factor lowering costs 4–6× more than dense

| family  | Δ int8 (mean) | range         | `QuantizeLinear` nodes |
| ------- | ------------: | ------------- | ---------------------: |
| Monarch |     **0.062** | 0.048 – 0.072 |                    212 |
| dense   |     **0.014** | 0.011 – 0.019 |                     77 |

The node count is **constant within each family** — 212 for every Monarch arm
regardless of `nblocks`, 77 for every dense one — and so is the penalty. Each
Monarch layer materializes an intermediate activation between its two factors,
plus the shuffle's reshape/transpose, and every one of those is a rounding stage
a single dense conv never has. That is also why int8 RTF is flat at 0.014–0.016
across a 4.2× MAC range while the dense arms fall to 0.008: both the latency and
the accuracy cost are node-bound, not arithmetic-bound.

Note this **does not reproduce NSNet2's finding** that Monarch is int8-loss-free
(|Δ| ≤ 0.012 at nblocks 4–40). The penalty is not intrinsic to Monarch; it
depends on the model's activation ranges — ConvFSENet feeds compressed
magnitudes into a sigmoid mask head, a different regime from NSNet2's GRU stack.

#### Where the int8 penalty actually comes from

The 0.062 mean penalty is not a lowering artifact and not fixable in the
quantizer. Three candidate mechanisms, all measured on the trained
`cp_cfs_mon_nb8` checkpoint, all refuted:

| hypothesis | test | result |
| ---------- | ---- | ------ |
| The shuffle's `Reshape`/`Transpose` requantize with their own scales, adding rounding to pure data movement | re-quantize with `op_types_to_quantize=["Conv"]`: 212 → 94 `QuantizeLinear`, 705 → 479 nodes | mask error vs FP32 got **worse**, 0.0648 → 0.0727. onnxruntime already propagates one scale across those nodes; they are numerically free |
| The intermediate activation between the two factors has a wider dynamic range, so int8's `max/127` step is coarser there | measure `\|mid\|max` against the layer's own input and output over 6 utterances × 100 frames, all 18 layers | **narrower**, not wider: mean `mid/in` 0.72×, `mid/out` 1.01× |
| That intermediate is peaky — a high crest factor wastes the int8 grid on outliers | crest factor (max/std) of all three tensors | the intermediate is the **best-behaved** of the three: 10.1 mean, against 14.5 for the layer input and 10.5 for the output |

What is left is the simple thing: **a Monarch layer quantizes one more activation
than a dense one does.** A dense 1×1 conv rounds its input and its output; a
Monarch layer rounds its input, the intermediate between its factors, and its
output. Each stage is individually benign — that is what the crest and range
numbers say — but there are 18 to 20 of them in series, and the errors compound.

The consequence is worth stating because it is actionable: no `quantize_static`
setting recovers this, and neither does a different ONNX lowering. Only a
parameterization that never materializes the intermediate at int8 would — i.e.
a fused Monarch kernel that keeps it in higher precision internally. That is the
same fix the latency measurement points to (node-bound, flat in `nblocks`), so
one piece of work would address both costs. Nothing in this repo needs it today,
since the dense controls win outright.

#### Why it reverses

The mechanism is visible in the dense column: **ConvFSENet at 192/384 has about
5× of slack.** `dense_r83` matches the full-size anchor at 4.9× fewer MACs
(2.882 vs 2.877 best, 2.853 vs 2.860 last-5, 2.862 vs 2.871 int8 — a wash on all
three, in both directions) while running 1.9× faster in int8. NSNet2's dense
controls behaved nothing like this: they lost 0.094 PESQ shrinking to 0.12 M
(2.845 → 2.751), i.e. that baseline really was capacity-limited at the sizes
Monarch was beating it.

That is the reconciliation, and it is the transferable claim:

> A structured factorization can only win where the dense model it replaces is
> actually capacity-limited at the target size. Where plain narrowing is free —
> as it is here for ~5× — structure has nothing to buy back, and pays the
> quantization and node-count costs anyway.

Two further observations, both of which point at the same follow-up:

- **Monarch's quality tracks reach, not MACs.** The two full-reach arms score
  2.854–2.857 and the two reduced-reach arms 2.811–2.816, with the 0.04 step
  landing exactly where reach falls below 100% — across a 4.2× MAC range, the
  arm's quality is essentially bimodal by connectivity regime.
- **Monarch's curve is flat and low; dense's is flat and high.** From 855 k to
  202 k MACs Monarch moves 0.046 and dense 0.039, but the dense curve sits
  0.018–0.066 above it throughout.

So the arm worth trying next is not more blocks — it is **wider channels at
`nblocks` 4–8**, where reach stays 100% and Monarch buys width instead of depth
of factorization. That tests whether Monarch can beat dense *above* the anchor's
quality rather than below it. This sweep does not answer that, and nothing here
should be read as evidence about it.

#### Reproducing

```bash
WAVE=1 ./run_convfsenet_monarch_sweep.sh      # nb8/r108 + nb32/r67, ~12 h 4-up
WAVE=2 ./run_convfsenet_monarch_sweep.sh      # nb4/r146 + nb16/r83
./run_convfsenet_monarch_eval.sh              # FP32 + int8 PESQ, all arms
python -m convfsenet.sweep_report             # the tables above, from the artifacts
```

### Square matrices — the follow-up sweep, and the answer

The sweep above put Monarch on the wrong shape. A TCM block's pointwise convs
are a compression/expansion pair (192→384, 384→192), and Monarch's first factor
`w1` is square in the **input** block size, so an expanding layer compresses
`2·nblocks/3` while a contracting one compresses only `nblocks/3` — the
contraction spends its entire first factor on the wide side. At `nblocks=2` the
"compressed" model was 1.12× *larger* than dense.

Making every matrix square fixes that by construction: with
`n_features = n_channels_res = n_channels_conv`, both factors are
`(nblocks, C/nblocks, C/nblocks)` and compression is exactly **`nblocks/2`** on
every layer. `nblocks = √C` is then the canonical Dao construction (√C blocks of
√C×√C) and sits exactly at the full-reach boundary. Training natively on 256
bins (dropping Nyquist, ~−90 dB of VBD STFT power) makes the frontend and
backend square too, so **all 20 pointwise matrices are structured** and the
un-structured floor falls from 12.7–53.9% of an arm to 4.0% — the nine depthwise
convs, which are not matrices.

That also decouples the two variables the first sweep confounded. There, the
only way to reach 7.1× compression was `nblocks=32`, which had already dropped
to 18.8% reach. Here, shrinking `C` at `nblocks=8` holds **100% reach down to
33 k MACs/frame**, because full reach needs `nblocks ≤ √C` and `≤ √n_features`.

| MACs/frame | reach | Monarch | dense control | gap (best) | gap (last-5) |
| ---------: | ----: | ------- | ------------- | ---------: | -----------: |
|  1,317,632 |  100% | `sq_mon_nb2` 2.891 | `sq_dense_C256` 2.864 | +0.027 | −0.007 |
|    662,272 |  100% | `sq_mon_nb4` 2.855 | `sq_dense_C177` 2.855 |  +0.000 | −0.003 |
|    334,592 |  100% | `sq_mon_nb8` 2.850 | `sq_dense_C122` 2.833 | +0.017 | +0.026 |
|    170,752 |  100% | `sq_mon_nb16` 2.838 | `sq_dense_C84` 2.831 | +0.007 | +0.011 |
|    117,200 |  100% | `sq_mon_C144_nb8` 2.820 | `sq_dense_C67` 2.824 | −0.004 | −0.019 |
|     59,552 |  100% | `sq_mon_C96_nb8` 2.782 | `sq_dense_C44` 2.782 | +0.000 | +0.004 |
|     32,960 |  100% | `sq_mon_C64_nb8` 2.735 | `sq_dense_C30` 2.726 | +0.009 | +0.012 |
|     88,832 | **25%** | `sq_mon_nb32` 2.691 | `sq_dense_C57` 2.818 | **−0.127** | **−0.112** |

**At full reach, Monarch and dense are indistinguishable**: seven pairings across
a **40× MAC range**, mean +0.008, sd 0.010, never outside ±0.027. The square
geometry flips the *sign* of the first sweep's result (where dense won all four
by 0.018–0.066) without producing separation.

Two cautions that matter more than the mean. `sq_mon_nb2`'s +0.027 — the largest
gap, and the interesting one because that arm has **identical MACs and identical
parameters** to its control — **reverses to −0.007 on the selection-free last-5
mean.** It was a validation spike, not a win. And no arm here has repeat seeds,
so the whole ±0.027 band is of the order of the nuisance variation this model is
known to have (one `num_workers` change once moved it 0.054).

**Reduced reach is where the two models part company.** `sq_mon_nb32` is the
direct analogue of NSNet2's `monarch_40` — same 25% reach — and it loses by
0.127, the largest gap in either sweep. On NSNet2 that configuration was the
star: 2.837 at 110 k MACs, +0.086 over its dense control, near dense parity at
24× fewer parameters. On ConvFSENet the same construction collapses.

#### Why NSNet2 separated and ConvFSENet does not

Both models' MACs/frame on one axis (NSNet2's are computed here for the first
time; its published tables are in parameters):

| ~MACs/frame | NSNet2 dense | ConvFSENet square dense |
| ----------: | -----------: | ----------------------: |
|      117 k  | 2.751 (`dense_h68`) | **2.824** (`sq_dense_C67`) |
|       76 k  | 2.749 (`dense_h52`) | — |
|       33 k  | — | 2.726 (`sq_dense_C30`) |

**NSNet2's dense baseline was fragile under width reduction and ConvFSENet's is
not** — +0.073 in dense-vs-dense at matched MACs. That fragility is the hole
`monarch_40` filled. A convolutional mask predictor degrades gracefully instead:
from 1.32 M to 33 k MACs/frame (40×) its dense arms give up 0.138 PESQ, and its
Monarch arms give up 0.156 alongside them, never diverging.

The transferable claim, now measured on two architectures and two geometries:

> A structured factorization wins where the dense model it replaces is
> capacity-limited or fragile at the target size. Neither geometry nor the
> cleanliness of the decomposition changes that — the square, canonical,
> full-reach construction is indistinguishable from plain narrowing on a host
> whose dense baseline is already robust.

### Lowering: grouped convolutions, not Einsum

`MonarchPointwise` computes `blockdiag × permutation × blockdiag` as two grouped
1x1 convolutions with a channel shuffle between them, which is the same map as
`MonarchLinear`'s reshape+einsum form (bit-identical init, forward equal to
1e-16 in fp64) but a materially different ONNX graph:

| graph            | ops                                                    |
| ---------------- | ------------------------------------------------------ |
| dense, windowed  | 29 Conv, 19 Relu, 10 Add, 9 Slice                       |
| Monarch, windowed| 47 Conv, 72 Reshape, 36 Transpose, 19 Relu, 9 Slice     |

No `Einsum`, and — measured — **zero** `Shape` / `Gather` / `Pad` / `If` /
`BatchNormalization`, so the deploy asserts in `convfsenet/quant_windowed.py`
pass unchanged. This matters beyond tidiness: onnxruntime ships no QDQ handler
for `Einsum`, which is exactly how every structured NSNet2 int8 number published
before 2026-07-11 turned out to be a hybrid-precision artifact (see the
correction notice in `RESULTS_NSNET2.md`). Grouped convolutions are ordinary
`Conv` nodes that the static quantizer handles per-channel with no registry
patching. `common/quant_audit.py` now asserts it directly — every compute node's
weight must have gone through the quantizer — rather than trusting a
`QuantizeLinear` count, which is what let that bug through.

Two constraints the lowering imposes, both enforced in code: `nblocks` must
divide both channel counts (no zero-padding, so the MAC count the sweep matches
on is exact), and the channel shuffle needs static reshape targets while
tracing. The dynamic (`unflatten`) form traces to `Shape`/`Slice`/`Concat`
reshape targets, which are both the ops the windowed deploy path asserts against
and fatal to quantization: they abort onnxruntime's symbolic shape inference, so
`quant_pre_process` — which both quant paths call — dies on an `AssertionError`
and **no int8 model can be built from that graph at all** (measured at T = 1, 4
and 126; the dynamic graph is numerically correct at any batch size, it just
cannot be quantized). `convfsenet/layers.py:static_t_sizes` pins the targets for
export and restores them after.

### Measured: int8 latency does not follow the MACs

Per-frame streaming int8 (QDQ, per-channel, synthetic calibration) under
onnxruntime CPU, one thread, median of 400 runs after 50 warm-up, on the
training box. Full-size models, all measured together:

| arm             | MACs/frame | int8 size | ms/frame | vs dense 192/384 |
| --------------- | ---------: | --------: | -------: | ---------------: |
| dense 192/384   |  1,436,160 |  1611 KiB |    0.215 |            1.00× |
| `mon_nb4`       |    855,552 |  1140 KiB |    0.240 |            1.12× |
| `mon_nb8`       |    482,304 |   775 KiB |    0.237 |            1.10× |
| `mon_nb16`      |    295,680 |   593 KiB |    0.249 |            1.16× |
| `mon_nb32`      |    202,368 |   502 KiB |    0.240 |            1.12× |
| `dense_r146`    |    850,304 |  1009 KiB |    0.201 |            0.94× |
| `dense_r108`    |    481,248 |   624 KiB |    0.140 |            0.65× |
| `dense_r83`     |    295,148 |   425 KiB |    0.118 |            0.55× |
| `dense_r67`     |    199,660 |   321 KiB |    0.121 |            0.56× |

**Monarch latency is flat in `nblocks`** — 0.240 → 0.240 ms while MACs fall 4.2×
— because the node count is constant (47 Conv + 108 shape ops either way) and
this graph is overhead-bound, not arithmetic-bound. So a Monarch arm is
**1.10–1.16× slower than the full dense baseline it compresses up to 7×, and
1.7–2.0× slower than its own MAC-matched dense control.** The int8 *size* does
track the MACs (1611 → 502 KiB).

The trained arms reproduce this on the deployed path: end-to-end int8 RTF is
**0.014–0.016 for every Monarch arm** across a 4.2× MAC range, against
**0.008–0.012** for the dense controls — the `dense_r83` model is 1.9× faster
than any Monarch arm while also scoring higher (see the results table above).

State this before any PESQ number: on the lane this repo deploys, **the only
claim this sweep can make is quality per MAC and per byte, not latency.** A PESQ
win is a reason to look at a fused kernel, not a deployment recommendation. The
direction matches what is already recorded elsewhere — Monarch was ~3× slower
than block-diagonal on the RT595, and more blocks made the STM32N6 NPU slower.

Training cost is the mirror image and does not bite: a Monarch generator step is
2.2–2.7× a dense one (7.9 ms vs 3.5 ms at batch 16, RTX 4090), but the real
training step is ~150–300 ms — dominated by the metric-GAN's CPU PESQ — so the
sweep pays ~2–3% wall-clock for it.

## Low-bit weight PTQ study

`convfsenet/eval_ptq.py` sweeps `(w_bits, a_bits)` via the eager
fake-quant path in `common/quant_fake.py` (per-output-channel symmetric
weights, dynamic per-tensor symmetric activations) over the full
824-utterance test split. It is a study tool — the dynamic-symmetric
activation path is mildly optimistic versus the deployed
static-asymmetric int8 ONNX (so w8a8 reads slightly higher here than
the real ONNX number above).

| precision | PESQ  | note |
| --------- | ----: | --- |
| fp32      | 2.934 | reference |
| w8a8      | 2.928 | matches the deployed int8 ONNX within noise |
| **w4a8**  | 2.856 | 4-bit per-channel weights cost ~0.08 PESQ |

Even at 4-bit weights, ConvFSENet still beats every NSNet2 int8 variant.

## QAT (kept for the w4 study)

`convfsenet/qat_train.py` and the QAT machinery in `common/quant_fake.py`
(`StaticActFakeQuant`, `install_static_activation_fake_quant`) are kept
for the low-bit (w4) study and as a reference scaffold. They are **not
needed for int8 deployment** — static PTQ alone is loss-free thanks to
the compression-prologue exclusion above.

## Reproducing

```bash
source .venv/bin/activate
# train (configs/convfsenet.json — mag-compressed)
python -m convfsenet.train --config configs/convfsenet.json \
    --checkpoint_path cp_convfsenet --training_epochs 200
# streaming FP32 ONNX
python -m convfsenet.export_onnx --checkpoint_file cp_convfsenet/g_best
# static int8 ONNX (QDQ, per-channel, MinMax; compression prologue kept FP32)
python -m convfsenet.quant --checkpoint_dir cp_convfsenet --num_utterances 200
# dual FP32/int8 PESQ + RTF on the VBD test split
python -m convfsenet.inference_onnx --checkpoint_file cp_convfsenet/g_best.onnx
# low-bit weight PTQ study
python -m convfsenet.eval_ptq --checkpoint_file cp_convfsenet/g_best
```

FP32 training runs ~22 min on an RTX 4090; the metric-GAN runs take
6–7 h. The streaming wrappers live in `convfsenet/streaming.py`
(`ConvFSENetStreaming` naive per-frame, `ConvFSENetStreamingFast` with BN
folded into the convs, `ConvFSENetStreamingONNX` the real-valued export
wrapper); streaming requires `causal=True`.

## Trained checkpoint

The best-PESQ generator (`g_best`), the streaming FP32 ONNX
(`g_best_fp32.onnx`), the static int8 ONNX (`g_best.onnx`), and the
exact training config are mirrored on HuggingFace at
[`claroche1/convfsenet`](https://huggingface.co/claroche1/convfsenet).
The HF repo is a flat layout (one variant) — no per-run subdirs.

PyTorch:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from convfsenet.model import build_causal_model

REPO = "claroche1/convfsenet"
cfg  = json.load(open(hf_hub_download(REPO, "config.json")))
ckpt = torch.load(hf_hub_download(REPO, "g_best"),
                  map_location="cuda", weights_only=False)
model = build_causal_model(AttrDict(cfg)).cuda().eval()
model.load_state_dict(ckpt["generator"])
```

ONNX (FP32 or int8):

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "claroche1/convfsenet"
sess = ort.InferenceSession(
    hf_hub_download(REPO, "g_best.onnx"),       # or g_best_fp32.onnx
    providers=["CPUExecutionProvider"],
)
# Streaming shape: feed one frame of magnitude STFT (B, n_freq) + the
# per-block FIFO state buffers per call. End-to-end RMS-norm + STFT +
# frame loop + iSTFT pipeline is in convfsenet/inference_onnx.py.
```

To (re-)publish from a fresh training run:

```bash
python push_convfsenet_hf.py --source cp_convfsenet
```

(needs `huggingface-cli login` or `HF_TOKEN` in the environment; the
script is idempotent — re-running just makes another HF commit.)
