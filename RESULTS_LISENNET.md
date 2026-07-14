# LiSenNet — results

> **Loss change (branch `mpsenet-loss`).** Every number below was trained with
> this model's *previous* loss and corresponds to commit `0b43d14` or earlier.
> The repo now trains all four models on the MP-SENet objective
> (`common/losses.py`); for this model that is a **real change to the
> objective**, so these numbers are not directly comparable to anything trained
> after it. Retrain to compare.

A lightweight, causal speech enhancer. LiSenNet is a ~37 K-parameter
sub-band U-Net with a dual-path-recurrent (DPR) bottleneck that predicts a
**magnitude-only** mask and recovers phase with a 2-iteration Griffin-Lim
seeded from the noisy phase (no phase decoder — the key cost saving). The
architecture is a faithful port of

> H. Yan, J. Zhou, K. Chen, J. Lu, *LiSenNet: Lightweight Sub-band and
> Dual-Path Modeling for Real-Time Speech Enhancement*,
> [arXiv:2409.13285](https://arxiv.org/abs/2409.13285),

from the authors' MIT-licensed reference implementation,
[hyyan2k/LiSenNet](https://github.com/hyyan2k/LiSenNet). The port keeps the
architecture unchanged; it only drops two unused upstream bits (the
`mel_scale` helper, whose sole dependency `torchaudio` is not installed, and
the standalone `NoiseDetector`) and reuses this repo's CMGAN
`MetricDiscriminator` (identical to LiSenNet's) rather than copying it.

It runs **frame-by-frame**: the mask sub-network is causal in time, so each
causal time-conv keeps a `(kt-1)`-frame ring buffer and the DPR inter-time GRU
carries its hidden state. The streaming reference, the FP32 ONNX export, and the
int8 quantization are all parity-checked against the offline model
(`tests/test_lisennet_*`).

See [RESULTS_CONVFSENET.md](RESULTS_CONVFSENET.md) and
[RESULTS_NSNET2.md](RESULTS_NSNET2.md) for the other model families, and the
[README](README.md) for setup and repo layout.

## Headline results

100-epoch CMGAN training on VoiceBank-DEMAND-16k (complex + magnitude +
PESQ-metric-GAN losses, `n_fft=512`/`hop=256`, AdamW). PESQ is wideband, on the
full 824-utterance VBD test split. The model is trained with the offline
Griffin-Lim phase; for real-time deployment the non-causal Griffin-Lim is
dropped in favour of the noisy phase (its own seed).

| metric                                          |     value |
| ----------------------------------------------- | --------: |
| params                                          | 36,783    |
| FP32 PESQ (torch, Griffin-Lim)                  | **3.006** |
| FP32 PESQ (ONNX, Griffin-Lim)                   | **3.006** |
| static-int8 PESQ (Griffin-Lim)                  |     2.920 |
| **real-time PESQ (static int8 + noisy phase)**  | **2.930** |
| int8 RTF (1 thread CPU)                         |     0.13  |
| FP32 ONNX size                                  | 0.27 MiB  |

The reproduction lands at **PESQ 3.006**, within ~0.06 of the paper's reported
~3.07 — a faithful from-scratch reproduction in this framework's pipeline.

## What is exported, and what stays outside the graph

Only the **mask sub-network** is exported / streamed — the pure tensor-op core
that maps the 3-channel feature map
`feat = [compressed_mag, group_delay/π, instantaneous_freq_diff/π]` to the
enhanced magnitude `est_mag`. As with the other models here, the STFT, the
feature extraction, and the phase recovery live *outside* the ONNX graph
(`lisennet/export_onnx.py`, dynamic batch/time axes; the DPR GRUs survive
export; ORT matches PyTorch to ~1e-6, including at an untraced time length).

LiSenNet is causal in time **by construction**, which the deploy work made
explicit rather than having to re-engineer:

* every time convolution left-pads the time axis `(kt-1, 0)`, so frame *n*
  never reads a future frame;
* the DPR **inter-time** GRU is already unidirectional;
* the DPR **intra** GRU is bidirectional only over *frequency*, which is fully
  available at each frame, so it streams fine.

The single non-causal component is the 2-iteration Griffin-Lim phase
refinement (its iSTFT/STFT span the whole utterance and need future enhanced
magnitudes). It is kept offline; the real-time path reuses the **noisy phase**
— the very seed Griffin-Lim starts from — for a causal iSTFT. The
frame-by-frame streamer reproduces the offline mask to **2e-7**
(`tests/test_lisennet_streaming_parity.py`).

## Deployment eval — isolating each cost

`lisennet/eval_deploy.py` measures PESQ across every combination of *mask
backend* × *phase method* on the full 824-utterance test split, so each
deployment step's cost is isolated. (Hold phase = Griffin-Lim and vary the
backend to read the export/quant cost; hold backend = torch and vary the phase
to read the cost of dropping Griffin-Lim.)

Static int8 is the quantization reported here because it is the
embedded-deployable path — dynamic weight-only int8 is not supported by the
target edge runtimes (e.g. stedgeai), so it is not measured.

| variant                                            |  PESQ | Δ vs offline |
| -------------------------------------------------- | ----: | -----------: |
| torch + Griffin-Lim (offline recipe)               | 3.006 |            — |
| FP32 ONNX + Griffin-Lim                             | 3.006 |    **0.000** |
| torch + noisy phase (real-time phase)              | 2.989 |       −0.017 |
| static-int8 ONNX + Griffin-Lim                     | 2.920 |       −0.086 |
| **static-int8 ONNX + noisy phase (full real-time)** | 2.930 |   **−0.076** |

Takeaways:

* **The ONNX export is loss-free** (3.006 → 3.006): the exported mask
  sub-network is numerically the torch model.
* **Dropping the non-causal Griffin-Lim costs only ~0.017 PESQ** on FP32, so the
  causal real-time pipeline stays close to the offline model.
* **Static int8 PTQ costs ~0.08 PESQ** — a moderate drop, not a collapse (the
  mask is magnitude-only, with no sensitive `atan2` phase path); QAT could
  recover part of it, the same direction the BASENet and ConvFSENet int8 studies
  point. The static-int8 PESQ has ~±0.02 run-to-run variance from the stochastic
  calibration crops.
* For the quantized model, the **noisy phase is marginally better than
  Griffin-Lim** (2.930 vs 2.920): GL refinement seeded from the *quantized*
  magnitude no longer helps, so the causal real-time path costs nothing extra
  over GL here.
* The full real-time deployable config — **static int8 mask + noisy phase** —
  lands at **2.930**, 0.076 below the offline model, at RTF ≈ 0.13.

### Real-time factor

Frame-by-frame mask compute under a single onnxruntime / PyTorch CPU thread is
**~2.1 ms per 16 ms frame → RTF ≈ 0.13**, i.e. ~7.5× faster than real time,
with the bounded per-frame state described above.

### Note — int8 file size

Unlike the larger models here, the int8 ONNX **file is not smaller** than FP32
(static int8 ~0.44 MiB vs 0.27 MiB). At ~37 K parameters the per-weight-tensor
quant metadata (QuantizeLinear / DequantizeLinear nodes, scale / zero-point
initializers, int32 biases) outweighs the byte saving on such small weights —
the model is already tiny in FP32. The win from int8 here is integer-arithmetic
compute on edge hardware, not graph footprint. (Static int8 also needs
`per_channel=False` to dodge an onnxruntime int32-bias-scale-adjustment bug on
this graph.)

## NPU-deployable variant (dual-path conv)

The GRU model above is the **quality reference**, but it does **not** compile to
the STM32N6 Neural-ART NPU: the dual-path-RNN bottleneck hits two hard stedgeai
blockers — the 2-axis `nn.LayerNorm` (Neural-ART's LayerNorm primitive takes only
a 1-D per-channel affine) and the `(b,t,f,d)→(b·t,f,d)` GRU reshape ("Order of
dimensions of input cannot be interpreted"). GRU recurrence would not map to the
NPU hardware even if it compiled (it runs as Cortex-M55 software).

So there is a second, **NPU-mappable** variant (`bottleneck: "conv"`,
`configs/lisennet_conv_wide.json`) that keeps the sub-band U-Net and the
magnitude-only mask but replaces the GRU bottleneck with a **dual-path conv**
one (`DualPathConv`):

* **intra-frequency** mixing → a symmetric depthwise freq conv + 1×1 (frequency
  is fully available per frame, so the bidirectional-over-freq context needs no
  state), replacing the bidirectional GRU;
* **inter-time** mixing → a stack of causal dilated depthwise time convs
  (`_CausalTimeConv`, dilations 1/2/4/8 ≈ 496 ms receptive field) with FIFO
  state, replacing the unidirectional GRU;
* norms are the already-primitive `CustomLayerNorm` → **no `nn.LayerNorm`** at all.

The exported graph has **0 GRU and 0 `LayerNormalization`** nodes — both blockers
gone. A capacity bump (num_channels 16→20, intra_kernel 7→11, dilations
[1,2,4]→[1,2,4,8]; 41 K params, above the GRU's 36.8 K) recovers most of the
quality:

| model                        | params | NPU | FP32 PESQ | real-time int8 PESQ |
| ---------------------------- | -----: | :-: | --------: | ------------------: |
| GRU (dual-path-RNN)          | 36,783 |  ✗  | **3.006** |               2.930 |
| conv nc16                    | 27,759 |  ✓  |     2.925 |                   — |
| **conv wide (nc20)**         | 41,063 |  ✓  | **2.970** |           **2.855** |

The ~0.04 FP32 gap (and ~0.075 at real-time int8) is the cost of dropping
recurrence to fit the NPU. FP32 ONNX export is loss-free; static int8 costs
~0.12 PESQ (a bit more than the GRU's ~0.09; QAT could recover part — stedgeai
does the real on-board quantization).

### Streaming state-I/O export (16 ms hop — deploys to the NPU since the Pad fix)

For frame-by-frame inference on the board, the conv variant exports to a graph
with **explicit FIFO state I/O** (mirroring ConvFSENet's contract), so the board
carries state across frames instead of resetting it:

```
feat (B,3,1,F)  +  N × state_i_in   ->   est_mag (B,1,F)  +  N × state_i_out
```

Every causal time-conv's ring buffer (encoder `DSConv`s, the `ConvolutionalGLU`
depthwise conv, the `_CausalTimeConv` layers, the mask conv) is a positional
state tensor — 17 of them for the wide model, static shapes (only batch dynamic,
which Neural-ART wants). `lisennet.export_onnx --streaming` produces
`g_best_streaming_fp32.onnx`; an ONNX Runtime frame loop reproduces the offline
mask to ~1e-6. The FIFO-state graph initially segfaulted the Neural-ART codegen
(blocker #4 in `LISENNET_NPU_HANDOVER.md`); this was later root-caused **not** to
the state I/O but to the `Pad(data, pads, "")` node form `F.pad` exports (an
empty optional `constant_value` input), and is now fixed bit-exactly inside the
export. The hardened variant's streaming graph **compiles and deploys** — see
the frame-level on-board section below.

## NPU-hardened variant (the deploy model)

The conv variant above still crashes the Neural-ART compiler (`atonn`) at
codegen. Peeling the blockers (see `LISENNET_NPU_HANDOVER.md`) gave a *hardened*
recipe — `norm="batchnorm"` (per-channel, folds into the conv), `act="relu"`,
`upsample="convtranspose"` (4-D tensors only), and a stateless **windowed**
deploy graph instead of the 17-tensor FIFO state I/O. The windowed hardened int8
graph **compiles to the NPU** (verified end-to-end with random weights: bit-exact
parity vs offline, ~2.1 M MACC per emitted frame at `emit_T=64`).

Retraining the hardened recipe from scratch (`configs/lisennet_conv_hardened*.json`,
same CMGAN training) and sweeping `num_channels` over 20/24/28:

| model (hardened)   | params | FP32 PESQ | int8 + GL | **int8 + noisy phase (real-time)** |
| ------------------ | -----: | --------: | --------: | ---------------------------------: |
| nc20               | 25,682 |     2.895 |     2.864 |                              2.853 |
| **nc24 (deploy)**  | 36,288 | **3.013** |     3.001 |                          **2.998** |
| nc28               | 48,718 |     2.927 |     2.881 |                              2.867 |

(All full 824-utterance VBD test split; FP32 ONNX export loss-free for all
three. nc24 RTF: 3.24 ms/frame single-thread CPU → 0.20.)

Takeaways:

* **The hardened nc24 matches the GRU quality reference in FP32 (3.013 vs
  3.006) and beats every other real-time int8 number in this file (2.998 vs the
  GRU's 2.930)** — with 0 GRU / 0 LayerNorm / 0 PReLU and an NPU-compilable
  graph. The hardening cost nothing at the right capacity.
* **The hardened primitives quantize far more gracefully.** Int8 drop is
  −0.016 (nc24) / −0.042 (nc20) vs −0.115 for the un-hardened conv_wide — the
  BN-fold + ReLU + signed-QInt8 recipe, not capacity, is what fixed the
  quantization loss.
* **Capacity is non-monotonic:** nc28 (49 K) trains to a *worse* optimum than
  nc24 (36 K) under the same 100-epoch recipe. nc24 ≈ the GRU budget is the
  sweet spot; the earlier ~41 K "wide" sizing overshoots.
* **QAT is counterproductive here.** A 30-epoch w8/a8 QAT fine-tune
  (recon-loss-only, `lisennet/qat_train.py`) *lost* 0.085 PESQ on the nc20
  model: the trainer's reconstruction objective pulls the weights away from the
  MetricGAN(PESQ)-shaped optimum, and with only a −0.02…−0.04 PTQ gap there is
  nothing for QAT to recover. PTQ is the deploy path.

### Round 2 — temporal-context study (FP32 wins, int8 gives them back)

The residual gap vs the GRU is bounded temporal memory (RF 68 frames ≈ 1.1 s vs
unbounded recurrence), so round 2 extended the receptive field — deeper
(`n_blocks 3`) and/or an extra dilation stage (`[1,2,4,8,16]`) — plus a second
nc24 seed to calibrate run variance
(`configs/lisennet_conv_hardened_nc24_{deep,dil16,s2}.json`, 140-epoch runs):

| variant (hardened nc24)    | params |  RF | FP32 PESQ | int8 real-time | PTQ drop |
| -------------------------- | -----: | --: | --------: | -------------: | -------: |
| **b2, dil [1,2,4,8]** (deploy) | 36,288 |  68 |     3.013 |      **2.998** | **−0.016** |
| b2, dil [1,2,4,8,16]       | 37,680 | 132 |     3.034 |          2.954 |   −0.080 |
| b3, dil [1,2,4,8,16]       | 46,248 | 196 | **3.069** |          2.985 |   −0.084 |

* **Temporal context buys FP32 exactly as predicted** (+0.02 per RF doubling,
  +0.056 total at RF 196 — above the GRU reference 3.006) …
* **… and static int8 takes it all back.** Both long-RF variants lose ~0.08 to
  PTQ vs the short model's −0.016 — and it is *not* depth: dil16 has the same
  2-block quantized path as the deploy model and still loses −0.080. Features
  that integrate seconds of context appear to carry a wider activation dynamic
  range, which per-tensor int8 resolves poorly. Re-calibration (2× data, three
  draws: 2.979–2.990 on the deep model) does not close it — structural, not
  calibration noise.
* **The recipe is seed-stable**: a second nc24 seed lands at 3.009 vs 3.013
  (±0.004), so the nc28 width regression in the table above was real, not noise.
* Unlocking the FP32 headroom at int8 would need quantization-aware work — a
  QAT with the MetricGAN loss in the loop (the recon-only QAT above regresses),
  or per-channel activation handling on the deploy quantizer side. → Done in
  round 3 below, which localizes the loss and recovers most of it.

### Round 3 — the int8 loss lives in the decoder; ReLU6 + distillation

Three parallel attacks on the locked FP32 headroom:

**1. Quant-sensitivity scan** (selective quantization: keep one node group FP32
at a time, *identical seeded calibration crops* across scenarios — without the
seeding, calibration-draw noise alone spans ±0.04 and swamps the signal; 200
utts, deep model): keeping the **decoder** FP32 recovers **+0.052 of the +0.078**
all-int8 gap; every other group (the dilated time stack included!) is ≤ +0.010.
The round-2 "long-RF activations" hypothesis was mislocalized: **the PTQ pain is
the decoder** (USConv upsampling + mask head — a *linear* path: no ReLU between
the skip-concats and convs), whose input features just get richer with RF.

**2. `act="relu6"`** (NPU-native Clip; bounds every ReLU's dynamic range) on the
deep arch: FP32 **3.084** — the best FP32 in the study, clipping *helped*
training — and all-int8 real-time 3.014 (PTQ −0.069 vs deep's −0.084; partial,
as predicted, because the decoder has no ReLU to clip).

**3. Mask-level knowledge distillation** (`train.py --distill_from`, deep
teacher → nc24-b2 student, weight 0.45): student FP32 3.015 → int8 real-time
3.004. The nc24 arch's quant robustness carries the (small) gain through intact.

Full-split (824) results, int8-static + noisy phase:

| model | recipe | FP32 | **int8 real-time** |
| ----- | ------ | ---: | -----------------: |
| **relu6-deep** | **int8, decoder kept FP32 (hybrid)** | **3.084** | **3.052** |
| deep           | int8, decoder kept FP32 (hybrid)     | 3.069 | 3.022 |
| **relu6-deep** | **all-int8**                          | 3.084 | **3.014** |
| KD student (nc24 arch) | all-int8                     | 3.015 | 3.004 |
| nc24 b2 (round-1 pick) | all-int8                     | 3.013 | 2.998 |

**relu6-deep is the new deploy model** — best on every axis, one checkpoint, two
recipes:

* **pure int8** (`g_best_windowed_fp32.int8_static.onnx`, signed, window
  196+64=260): **3.014**, same all-int8 deploy story as before;
* **hybrid** (`g_best_windowed_int8_decoder_fp32.onnx`, decoder's 118 nodes
  left unquantized in the QDQ graph): **3.052**, at the cost of the decoder
  (~36 % of MACC) running as float epochs on the board (M55 or NPU FP16) —
  compile + latency to be verified on the deploy box; the pure-int8 artifact is
  the fallback.

Net effect of the whole study: real-time deployable PESQ **2.855 → 3.052**
(+0.197 over the un-hardened conv_wide; +0.122 over the GRU reference's
real-time 2.930), all NPU-safe ops.

The deploy artifact is the windowed signed-int8 graph
(`cp_lisennet_conv_hardened_nc24/g_best_windowed_fp32.int8_static.onnx`,
`feat_window (B,3,132,257) → est_mag (B,64,257)`, window = RF 68 + `emit_T` 64,
verified QInt8-only) — published as `conv-hardened/g_best_windowed_int8_static.onnx`
on the HF repo and deployed below.

### On-board deployment — STM32N6570-DK (measured 2026-07-03)

The published nc24 artifact compiles to the Neural-ART NPU (stedgeai 4.0.1,
`n6-allmems-O3` profile, `--fix-parametric-shapes "{'B':1}"`) and runs on the
board (STM32N657 @ MCU 800 MHz / NPU 1 GHz, `n6_loader` + `validate --mode
target` — the flow in `deploy/stm32n6/ONBOARD_MEASUREMENT.md`):

| metric (windowed int8, emit_T=64)   | value |
| ----------------------------------- | ----: |
| epochs (HW / hybrid / SW)           | 102 (60 / 36 / 6) |
| MACC per window (64 frames)         | 177,695,034 |
| weights                             | 491.6 KiB (octoFlash) |
| activations                         | 2.72 MiB (all on-chip: cpuRAM2 + npuRAM3–6) |
| **latency per window**              | **73.63 ms** (std 0.32, 10 runs) |
| **per emitted frame / RTF**         | **1.15 ms → RTF 0.072** (~14× headroom) |
| on-target cosine vs host int8       | 0.99829 (rmse 0.151) |

Notes:

* **Per emitted frame this is the fastest model measured on the N6 in this repo**
  (monarch_full 2.13 ms, ConvFSENet 4.40 ms) — and it carries the best real-time
  int8 PESQ (2.998). The trade is **block latency**: emit_T=64 buffers 1.02 s of
  audio per inference. The `emit_T` export knob trades that down (e.g. emit_T=16
  → 256 ms blocks at ~2.6× the per-frame recompute); every emit_T compiles.
* A fully on-chip (`n6-noextmem`) build is **impossible** for this graph —
  weights + activations = 3.35 MB > 2.8 MB of usable pools. It doesn't matter:
  the 492 KiB of weights stream from octoFlash at ~13 MB/s average, negligible
  at this size (the penalty that made dense NSNet2 non-real-time was 2.7 MB).
* Where the 73.6 ms goes (npu_profiler): NPU core 20.3 ms (27.7%); the **6 SW
  epochs cost 38.1 ms (52%)** — the three encoder down-sampling convs
  (k=(2,5), stride (1,3) over frequency — a geometry the conv engine won't map)
  at 23.2 ms, plus the input/output `Gather` layout ops at 14.9 ms; the rest is
  hybrid/runtime overhead. If the NPU share ever matters, the stride-3 encoder
  is the knob — at RTF 0.072 there is no pressure.

### Frame-level streaming deployment — 16 ms hop on the NPU (2026-07-03)

Blocker #4 root-caused and fixed (the `F.pad` export form, not the state I/O —
see `LISENNET_NPU_HANDOVER.md`), so the **17-state FIFO streaming graph** now
compiles and runs on the board too. This is LiSenNet operating **frame by
frame** — one 16 ms hop in, one enhanced frame out, bounded state carried
on-device — the same latency class as ConvFSENet/monarch:

| metric (streaming int8, per frame)  | value |
| ----------------------------------- | ----: |
| epochs (HW / hybrid / SW)           | 131 (74 / 51 / 6) |
| MACC per frame                      | 1,299,086 |
| memory                              | 47 KiB weights + 147 KiB activations, all on-chip (`n6-noextmem`) |
| **latency per frame**               | **2.791 ms** (std 0.006) |
| **RTF (16 ms hop)**                 | **0.174** (~5.7× headroom) |
| algorithmic latency                 | **one 16 ms hop** (vs 1.02 s windowed) |
| **PESQ, full 824-utt split (int8 streaming + noisy phase)** | **2.963** |
| host parity (int8 stream vs fp32 offline) | cos 0.9992 |
| on-device vs host int8 (threaded state, real speech) | cos 0.998 |

Notes:

* **This makes LiSenNet the best-quality streaming model on the N6** —
  PESQ 2.963 at 2.79 ms/frame, vs ConvFSENet 2.91 at 4.40 ms and monarch_full
  2.85 at 2.13 ms. The windowed graph keeps the throughput crown (1.15 ms/frame,
  int8 PESQ 2.998) for latency-tolerant (1 s block) use.
* The streaming int8 quant costs −0.037 vs the fp32 real-time reference
  (2.963 vs 3.000 on the same protocol) — slightly more than the windowed
  artifact's −0.016; the calibration regime differs (300 propagated-state
  frames vs windowed crops). Recalibration with more utterances is the obvious
  knob if that gap ever matters.
* Where the 2.78 ms goes (npu_profiler): NPU core 0.72 ms (26%); no hot epoch
  (top one 0.033 ms) — the floor is the distributed launch + state-plumbing
  overhead of 131 epochs, the same regime as ConvFSENet's streaming graph. At
  1.3 M MACC/frame the NPU is idle ~74% of the time; epoch count, not compute,
  sets the frame cost.
* Quantization: `quant_onnx --streaming --signed` (new) — signed QInt8,
  per-channel, percentile calibration that **threads the real FIFO state**
  through the trained model over VBD crops, so state tensors calibrate on
  realistic ranges.

## Reproducing

```bash
source .venv/bin/activate
# train (configs/lisennet.json — CMGAN, batch 16)
python -m lisennet.train --config configs/lisennet.json \
    --checkpoint_path cp_lisennet --training_epochs 100
# FP32 ONNX export of the mask sub-network (feat -> est_mag, dynamic B/T)
python -m lisennet.export_onnx --checkpoint_file cp_lisennet/g_best
# static full int8 (QDQ, per-tensor, percentile calibration) — embedded-deployable
python -m lisennet.quant_onnx --fp32 cp_lisennet/g_best_fp32.onnx --mode static \
    --config cp_lisennet/config.json
# full deploy eval: PESQ across backend × phase + streaming RTF
python -m lisennet.eval_deploy --checkpoint_file cp_lisennet/g_best --n_utts 824
```

The **NPU-mappable conv variant** trains and exports the same way, plus a
streaming graph with explicit state I/O for the board:

```bash
python -m lisennet.train --config configs/lisennet_conv_wide.json \
    --checkpoint_path cp_lisennet_conv_wide --training_epochs 100
python -m lisennet.export_onnx --checkpoint_file cp_lisennet_conv_wide/g_best              # whole-utterance
python -m lisennet.export_onnx --checkpoint_file cp_lisennet_conv_wide/g_best --streaming  # deploy target
python -m lisennet.eval_deploy --checkpoint_file cp_lisennet_conv_wide/g_best --n_utts 824
```

Training is fast — the model is tiny, so each step is dominated by the CPU
PESQ + Griffin-Lim cost rather than GPU compute (~0.8 GB at batch 16,
~94 s/epoch, ~2.7 h for 100 epochs on an RTX 4090). The streaming reference is
`lisennet/streaming.py` (`LiSenNetStreamer`); streaming reuses the trained
`g_best` with no retraining (the `state_dict` keys are unchanged when the
streaming flag is off, and the offline forward stays bit-identical).

## Trained checkpoint

Both variants live in one HuggingFace repo,
[`claroche1/LiSenNet`](https://huggingface.co/claroche1/LiSenNet), each under its
own subfolder:

* **`gru/`** — the GRU quality reference: `g_best`, `g_best_fp32.onnx`,
  `g_best_int8_static.onnx`, `config.json`.
* **`conv/`** — the NPU-deployable conv variant: the same, plus the frame-by-frame
  **streaming** graphs (`g_best_streaming_fp32.onnx`, the stedgeai target, and
  `g_best_streaming_int8_static.onnx`).

Publish either variant with `push_lisennet_hf.py` — it auto-selects the subfolder
and writes the combined root model card from the run's `config.json` `bottleneck`:

```bash
python push_lisennet_hf.py --checkpoint_dir cp_lisennet            # GRU  -> claroche1/LiSenNet (gru/)
python push_lisennet_hf.py --checkpoint_dir cp_lisennet_conv_wide  # conv -> claroche1/LiSenNet (conv/)
```

PyTorch:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from lisennet.model import build_lisennet

REPO = "claroche1/LiSenNet"
cfg  = json.load(open(hf_hub_download(REPO, "config.json")))
ckpt = torch.load(hf_hub_download(REPO, "g_best"),
                  map_location="cpu", weights_only=True)
model = build_lisennet(AttrDict(cfg)).eval()
model.load_state_dict(ckpt["generator"])
# end-to-end waveform -> waveform: model(noisy_wav)["est"]
```

ONNX (the mask sub-network — `feat (B,3,T,F)` -> `est_mag (B,T,F)`):

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "claroche1/LiSenNet"
sess = ort.InferenceSession(
    hf_hub_download(REPO, "g_best_int8_static.onnx"),   # or g_best_fp32.onnx
    providers=["CPUExecutionProvider"],
)
est_mag = sess.run(["est_mag"], {"feat": np.zeros((1, 3, 100, 257), np.float32)})[0]
```

To (re-)publish from a fresh run:

```bash
python push_lisennet_hf.py            # cp_lisennet/ -> claroche1/LiSenNet
```

(needs `huggingface-cli login` or `HF_TOKEN`; idempotent — re-running just makes
another HF commit.)
