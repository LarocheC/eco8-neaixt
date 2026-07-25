# LiSenNet — results

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

## Recurrence vs. conv + FIFO on the time axis — the hybrid bottleneck (2026-07-14)

Everything above replaced *both* of LiSenNet's GRUs with convolutions, because the
dual-path-RNN block as a whole was the Neural-ART blocker. But the two GRUs are not
the same bet, and lumping them together left a question unasked:

* the **intra** GRU runs over *frequency*, which is fully available at each frame.
  Bidirectional context there needs no state, so a symmetric depthwise conv is the
  strictly cheaper way to get it — replacing it is uncontroversial.
* the **inter** GRU runs over *time*, and there the conv variant pays for its
  finite receptive field: a stack of dilated causal convs whose FIFO buffers hold an
  RF-frame history of activations. Recurrence gets *unbounded* lookback out of one
  hidden state. Dropping it may have been a mistake.

`bottleneck: "hybrid"` (`DualPathHybrid`) tests exactly that — conv over frequency,
**GRU over time**. Against `bottleneck: "conv"` it is a single-variable swap: encoder,
decoder, norms, activations, the intra-frequency mixer and `n_blocks` are identical,
and `hidden_dim` sizes the GRU so the **parameter counts match** (36,384 vs 36,288;
45,600 vs 46,248). Any quality difference is recurrence-vs-FIFO, not capacity.

### Recurrence on an NPU — the GRU cell as 1×1 convolutions

A recurrent bottleneck looks undeployable: a traced `nn.GRU` exports an ONNX `GRU`
op (Neural-ART cannot map it) and needs the `(b,t,f,d) → (b·f,t,d)` reshape the
compiler rejects outright. Both objections are artifacts of *how the recurrence is
written*, not of recurrence itself. At `t == 1` — which is what streaming inference
does anyway — a GRU step is just

```
r = σ(W_ir x + W_hr h)    z = σ(W_iz x + W_hz h)
n = tanh(W_in x + r ⊙ W_hn h)          h' = z ⊙ (h − n) + n
```

and with the weights shared across frequency (exactly what `nn.GRU` does when it
treats `f` as batch) every one of those matmuls is a **1×1 convolution** over
`(b, d, 1, f)`, with `h` an explicit `(b, H, 1, f)` state tensor. So `TimeGRU` keeps
one set of weights and two evaluation paths: the fused `nn.GRU` sequence kernel for
training, and the unrolled conv cell for streaming/export. They agree to **1.2e-7**
(`tests/test_lisennet_hybrid_bottleneck.py` pins this — if they ever diverge, the
board runs a different model than the one that was trained).

The exported streaming graph therefore has **0 GRU ops**: recurrence reaches the NPU
as `Conv / Add / Sigmoid / Tanh / Mul` on 4-D tensors. It is the same `feat + state_i_in
→ est_mag + state_i_out` contract as the conv variant, so it deploys through the same
toolchain — the GRU hidden state is just another state tensor.

### On silicon — the GRU beats the FIFO by 1.5–2.2× at matched params (2026-07-14)

All four streaming int8 graphs compiled (`stedgeai` 4.0.1, `n6-noextmem`) and were
measured on the STM32N6570-DK in one session. **The two conv rows are re-measurements
of the trained deploy models and reproduce their published latencies** (2.787 vs
2.791; 4.823 vs 4.83), which is what makes the two new rows trustworthy. The GRU
models carry **re-initialized weights** — output is garbage, latency and the epoch
profile are not (the ‡ convention already used above).

| streaming int8 | params | RF | epochs (HW/hyb/SW) | MACC | activations | **ms/frame** | RTF |
| -------------- | -----: | --: | :-- | ---: | ---: | ---: | ---: |
| conv nc24            | 36,288 |  68 | 131 (74/51/6) | 1,299,086 | 146.6 KiB | 2.787 | 0.174 |
| **GRU nc24** ‡       | 36,384 |  ∞  | **95** (65/**24**/6) | 1,285,838 | **39.9 KiB** | **1.822** | **0.114** |
| conv relu6-deep      | 46,248 | 196 | 199 (116/77/6) | 1,677,197 | 307.1 KiB | 4.823 | 0.301 |
| **GRU relu6-deep** ‡ | 45,600 |  ∞  | **117** (86/**25**/6) | 1,588,814 | **52.3 KiB** | **2.181** | **0.136** |

* **Recurrence is 1.53× (nc24) and 2.21× (relu6-deep) faster per frame, at matched
  parameters and near-identical MACC** (within 1% / 5%). The GRU relu6-deep even beats
  the *conv nc24* (2.181 vs 2.787 ms) while being the deeper, longer-context model.
* **The win is overhead, not compute.** MACC barely moves, so none of it comes from
  arithmetic. What collapses is the epoch count (131→95, 199→117) and specifically the
  *hybrid* epochs — the state plumbing — 51→24 and 77→25. This is the direct
  confirmation of the earlier finding that the streaming graph is launch/state-bound
  (NPU idle ~74%): remove the state machinery and the frame cost falls with it.
* **The GRU cell maps to hardware.** SW epochs stay at 6 in every variant — the same
  three stride-3 encoder convs and the Gather layout ops that were always software.
  None of the cell's `Sigmoid`/`Tanh`/`Mul`/`Sub` fell to the M55. Recurrence on this
  NPU is not a compromise; it is native, once written as convolutions.
* **Recurrence streams cheaper than the FIFO it replaces — 3.7–5.9× less activation
  memory** (146.6→39.9 KiB, 307.1→52.3 KiB). This inverts the usual intuition. A
  conv+FIFO holds an *activation history* (RF × C × F, growing with every extension of
  the receptive field); a GRU holds a *compressed summary* (H × F, fixed) with
  unbounded lookback. Buying context with dilations means buying state; buying it with
  recurrence does not — which is why the gap *widens* with RF (1.53× at RF 68, 2.21×
  at RF 196).
* **The stateless windowed graph is not available to it, by construction.** A GRU has
  no finite receptive field, so no window reproduces the offline model — the windowed
  export raises rather than silently shipping a truncated model. That is the one axis
  where conv+FIFO strictly wins: only a finite RF can be recomputed statelessly. (Per
  the section below, that mode loses by 12–26× anyway, so it costs the GRU nothing.)

Caveat on the rig: `n6_loader` hit transient gdb "Loading memories failed" faults on
the conv relu6-deep model (3 attempts failed, a later attempt loaded first try and
gave 4.823 ms). It is a flaky ST-LINK link, not a memory limit — but a measurement
harness must never validate after a failed load, or it times whatever firmware is
still resident. The measure script gates on "Start operation achieved successfully".

**Quality — measured (2026-07-25, both trained to matched epoch budgets:
nc24 100 ep, deep 140 ep). The hypothesis holds: the GRU's disadvantage is
real at short RF and essentially vanishes at long RF.**

| model (matched params) | FP32 | RT-int8 | streaming int8 | PTQ drop |
| ---------------------- | ---: | ------: | -------------: | -------: |
| conv nc24 (RF 68)          | 3.013 | 2.998 | 2.963 | −0.015 |
| **GRU nc24** (RF ∞)        | 2.953 | 2.849 | 2.867 | −0.104 |
| conv relu6-deep (RF 196)   | 3.084 | 3.014 | 3.013 | −0.070 |
| **GRU relu6-deep** (RF ∞)  | 3.073 | 2.991 | 2.975 | −0.082 |

* **At RF 68 the GRU loses on quality** — FP32 −0.060, real-time int8 −0.149.
  Two causes: unbounded context does not help when 68 frames already suffice
  (the conv trains to a better FP32 optimum), and the GRU's recursive
  `tanh`/`sigmoid` state quantizes far worse (PTQ drop −0.104 vs the conv's
  −0.015, ~7×) — the round-3 "linear-path" worry, realized on the recurrence.
* **At RF 196 the gap essentially closes.** FP32 3.073 vs 3.084 is **−0.011**,
  within the ±0.004–0.02 seed noise this study already reports — a statistical
  tie. Real-time int8 trails by only −0.023, streaming by −0.038, and the PTQ
  drop (−0.082) is now comparable to the conv's (−0.070), not 7× worse. As
  predicted, unbounded recurrent context becomes competitive exactly where the
  conv+FIFO must spend the most state (196 frames) to fake it.
* **Net at depth: the GRU matches conv quality in FP32 and trails ~0.04 at
  int8, while running 2.21× faster (2.18 vs 4.82 ms) with 6× less state (52 vs
  307 KiB)** — and giving up only the FIFO's certifiable bounded-time recovery.
  The efficiency win now comes at a near-zero quality cost. At RF 68 the same
  swap is not worth it.

So the honest, complete trade: **conv+FIFO is the safe default (best quality
at any RF, certifiable recovery); recurrence-on-time is the better bet at long
receptive fields, where it matches quality at half the latency and a sixth of
the state.** Recompute-vs-FIFO (the window, earlier) loses on both axes;
GRU-vs-FIFO wins on efficiency and ties on quality once the context is long.

```bash
python -m lisennet.train --config configs/lisennet_hybrid_nc24.json \
    --checkpoint_path cp_lisennet_hybrid_nc24 --training_epochs 100
python -m lisennet.export_onnx --checkpoint_file cp_lisennet_hybrid_nc24/g_best --streaming
python -m lisennet.eval_deploy --checkpoint_file cp_lisennet_hybrid_nc24/g_best --n_utts 824
```

### What the TCM buys that the GRU cannot: bounded-time forgetting (2026-07-14)

The GRU wins on latency, state and (unbounded) context. The FIFO keeps exactly one
advantage, and it is not the obvious one.

**It is not that the GRU can diverge — it provably cannot.** With
`h' = (1-z)⊙n + z⊙h`, `z = σ(·) ∈ (0,1)`, `n = tanh(·) ∈ (-1,1)` and `h₀ = 0`, each `h'`
is a convex combination of `h` and something in `(-1,1)`, so **`|h| ≤ 1` for all time by
induction** — independent of weights, input or quantization. The state cannot blow up and
neither can the mask. An "out-of-domain sample makes it explode" claim is false and would
not survive review.

**What the FIFO has is a *dead-beat* recovery guarantee.** Its memory *is* the last `RF`
input frames, so two streams fed identical input from *any* two different states hold
identical FIFOs after `RF` frames — and from there compute **bit-identical** outputs.
Corruption is not damped, it is **evicted, in bounded time**, and the bound is a property
of the architecture, not of the trained weights. The GRU's forgetting is asymptotic and
data-dependent: an update gate driven toward 1 holds a stale state, and nothing bounds
for how long.

Measured (`python -m lisennet.state_recovery`, 6 VBD test utterances; the whole streaming
state is overwritten at frame 0 and the output compared per-frame against the clean-state
run; three corruptions — a state **carried over from another utterance** (in-distribution,
the realistic worst case), a mid-stream **reset to zeros**, and **random noise** at the
state's own scale):

| model | bit-exact recovery | to <1% error | to <0.1% |
| ----- | -----------------: | -----------: | -------: |
| conv+FIFO nc24 (RF 68)        | **68 frames** (1.1 s), all 3 corruptions | 52–56 | 59–60 |
| conv+FIFO relu6-deep (RF 196) | **196 frames** (3.1 s), all 3            | 104–119 | 144–159 |
| GRU (faithful LiSenNet, trained) | **never** | 297–504 (4.8–8.1 s) | **never** (>8.8 s) |

* **The conv models recover bit-exactly at exactly RF frames — 68 and 196, to the frame,
  for every corruption type.** The error is *identically zero*, not merely small, because
  past `RF` the two streams are the same computation. Theory and measurement agree with
  nothing left over.
* **The trained GRU never becomes exact** (it is IIR — it cannot), needs **4.8–8.1 s** to
  fall below 1%, and is **still above 0.1% after 8.8 s**. A wrong state is audible for
  seconds.
* Note the FIFO models cross the 1% line *before* their RF (52 and 104 frames): they decay
  first and then snap to zero. The guarantee is the RF bound; the typical case is faster.

So the honest trade for the paper is **efficiency vs. certifiability**, not efficiency vs.
safety:

| | conv + FIFO (TCM) | GRU |
| --- | --- | --- |
| latency / state | 2.79 ms, 147 KiB | **1.82 ms, 40 KiB** |
| context | bounded (RF) | **unbounded** |
| state bounded? | yes (it is an activation copy) | yes (`\|h\| ≤ 1`, provable) |
| **recovery from a bad state** | **exact, ≤ RF frames, guaranteed a priori** | asymptotic, seconds, weight-dependent |

If the deployment can tolerate seconds of degraded output after a glitch — a dropped
frame, a re-sync, a scene change, an int8 state that saturates — the GRU is strictly
better on every other axis. If it must *certify* recovery (a safety-adjacent product, a
watchdog that resets state, anything where a stuck enhancer is unacceptable), the FIFO's
`RF`-frame bound is a property the GRU cannot offer at any width. **That is the case for
the TCM, and it is the only one.**

Caveat: the GRU row is the *faithful* LiSenNet (`cp_lisennet`, trained, hidden 24) —
a trained time-GRU in this architecture, but not the hybrid. The hybrid's own recovery
curve needs the trained hybrid checkpoint; the mechanism (IIR, no dead-beat) is
architectural and will not change, but the time constants will.

Figure: `paper/figures/recovery.tex` (data `paper/data/state_recovery.{json,dat}`).

### FIFO state vs. stateless recompute — measured at equal latency (2026-07-14)

The windowed export's `emit_T` sets the algorithmic latency: **`emit_T=1` (window
= RF+1 frames in, last frame's mask out) is the stateless twin of the FIFO
streaming graph** — identical one-hop 16 ms latency, no state machinery at all,
paying RF-fold recompute instead. It is *bit-exact* to the offline model in
steady state (FP32 cos 1.0000001, max|diff| 7.2e-07). `emit_T=64` is a
throughput mode (1.02 s block latency), not a real-time deployment.

| variant | RF | FIFO streaming | stateless window (T=1) | recompute | net |
|---|---:|---|---|---:|---:|
| nc20 | 68 | **2.59 ms** (RTF 0.16) | **29.93 ms** (1.87) | 72× | 11.6× slower |
| nc24 | 68 | **2.79 ms** (0.17) | **32.81 ms** (2.05) | 72× | 11.8× |
| nc28 | 68 | 3.15 ms (0.20) | 40.01 ms (2.50) | 71× | 12.7× |
| dil16 | 132 | 3.63 ms (0.23) | 74.72 ms (4.67) | 134× | 20.6× |
| deep | 196 | 4.88 ms (0.30) | 127.16 ms (7.95) | 199× | 26.1× |
| **relu6-deep** | 196 | **4.83 ms** (0.30) | **119.86 ms** (7.49) | 202× | 24.8× |

* **The stateless window uses the NPU 6–8× better** (2.2–3.1 GMAC/s vs
  0.34–0.56): no Slice/Concat state ops, and a 69–197-frame conv extent fills
  the pipeline — so the streaming graph is confirmed **launch/state-bound, not
  compute-bound**.
* **…and it still loses by 11.6–26×**, because it recomputes the receptive field
  every frame (70× the MACs at RF 68, 200× at RF 196). A 6–8× efficiency gain
  cannot pay a 70–200× compute bill. **Not real-time for any variant.**
* Amortizing over an emit block restores real time only by re-adding latency
  (emit_T ≥ 3 → 48 ms for nc24; ≥ 9 → 144 ms for relu6-deep) and still costs
  11–15 ms/frame.
* **⇒ FIFO streaming dominates on both axes** — which is what makes the `Pad`
  export fix (blocker #4) load-bearing: without it the only NPU-mappable graph
  was the window, and LiSenNet would not be real-time on this chip.

### Variant sweep on silicon — every hardened variant, both graphs (2026-07-14)

All six hardened architectures were deployed on the STM32N6570-DK in both graph
forms (13 compiles, 13 on-board measurements, one toolchain pass each). Latency
is per **emitted** 16 ms frame; full table + method in
[`deploy/stm32n6/ONBOARD_MEASUREMENT.md`](deploy/stm32n6/ONBOARD_MEASUREMENT.md),
raw data in `paper/data/` (branch `paper`).

| variant | RF | streaming ms/frame (RTF) | windowed ms/frame (RTF) | int8 PESQ |
|---|---:|---:|---:|---:|
| nc20 ‡ | 68 | 2.59 (0.162) | 0.93 (0.058) | 2.853 |
| nc24 | 68 | 2.79 (0.174) | 1.15 (0.072) | 2.963 str / 2.998 win |
| nc28 ‡ | 68 | 3.15 (0.197) | 1.45 (0.091) | 2.867 |
| dil16 ‡ | 132 | 3.63 (0.227) | 1.82 (0.114) | 2.954 |
| deep ‡ | 196 | 4.88 (0.305) | 5.72 (0.358) | 2.985 |
| **relu6-deep** | 196 | **4.83 (0.302)** | 3.09 (0.193) | **3.013 / 3.014** |
| relu6-deep hybrid | 196 | — | 33.99 (2.125) ✗ | 3.052 |

‡ latency from the compiled architecture with re-initialized weights (no trained
checkpoint on the deploy box — see the note below); PESQ from the host study.

* **The whole design space is real-time streaming** (RTF 0.16–0.31, everything
  on-chip): the quality knobs of rounds 1–3 are affordable. Doubling the
  receptive field (68 → 196) costs only 1.7× frame time.
* **relu6-deep ships at 4.83 ms/frame, PESQ 3.013, 16 ms latency** — and its
  *streaming* graph gives up nothing vs windowed (3.013 vs 3.014), unlike nc24
  (2.963 vs 2.998), because ReLU6 also bounds the streamed state.
* **The hybrid decoder-fp32 recipe (PESQ 3.052) is NOT deployable**: the float
  decoder pulls 19 epochs onto the M55 and needs 3.5 MB weights + 12.3 MB
  activations → **34 ms/frame, RTF 2.1**. This closes the round-3 open question:
  ReLU6, not the hybrid, is the repair that fits the latency budget.
* **Windowed mode is only cheap on-chip.** The RF-196 windows (260 frames) need
  >5 MB of activations, spill ~2.5 MB into external hyperRAM and become
  memory-bound (`deep` 5.72 vs `relu6-deep` 3.09 ms/frame for the *same*
  topology — an allocator artifact, not an activation-function effect).
* **NB — the local nc20 checkpoint is not the trained one.**
  `cp_lisennet_conv_hardened/g_best` scores torch FP32 **2.198**, not the 2.895
  in the table above; it is a July-1 compile-proof checkpoint. nc28/dil16/deep
  were never copied off the training box either. Only **nc24** and
  **relu6-deep** (both on HF) reproduce their published quality here.

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
