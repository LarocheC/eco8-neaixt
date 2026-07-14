# BASENet — results

> **Loss refactor (branch `mpsenet-loss`).** The loss moved into the shared
> `common/losses.py`, but this model was already training on the MP-SENet
> objective at the same weights — the refactor is numerically a no-op, so the
> numbers below still stand.

A frequency-adaptive, causal-capable speech enhancer. BASENet partitions the
spectrum into Bark-scale bands and allocates each a scaled-capacity encoder
derived from critical-band density (perceptually dense low frequencies get
deeper branches, sparse high frequencies get lighter ones), restores
cross-band coherence with a compact cross-band attention module, and predicts
a magnitude mask + enhanced phase in the same paradigm as MP-SENet. The
architecture is a from-scratch reproduction of

> D. Martins Gomes, F. Capman, *BASENet: Band-Adapted Speech Enhancement
> Network with Cross-Band Attention*,
> [arXiv:2606.12662](https://arxiv.org/abs/2606.12662) (Thales SIX GTS, 2026),

built on inverted-residual blocks with dense connectivity and a convolutional
recurrent network (CRN), following the paper's stated reuse of MP-SENet's
(Lu et al. 2023) training procedure and loss functions. The causal variant
swaps the CRN's bidirectional GRU for a unidirectional one — the paper's
stated recipe — plus three more fixes this port found necessary for genuine
(not just nominal) causality: frequency-only normalization, frequency-only
squeeze-excitation pooling, and left-padded time convolutions (all detailed
below). It runs **frame-by-frame** in its causal form: every stateful op
(TFConv ring buffers, CRN GRU hidden state) carries state across frames, and
the streaming reference is parity-checked against the offline model
(`tests/test_basenet_streaming_parity.py`).

See [RESULTS_CONVFSENET.md](RESULTS_CONVFSENET.md),
[RESULTS_NSNET2.md](RESULTS_NSNET2.md), and
[RESULTS_LISENNET.md](RESULTS_LISENNET.md) for the other model families, and
the [README](README.md) for setup and repo layout.

## Status: reproduction in progress

Unlike the other three model families, **BASENet reproduction has not yet
matched the paper's headline numbers.** This file documents an active
gap-closing investigation, not a finished result. Current best: **PESQ-wb
3.359** (non-causal), paper's target **3.55** — gap **0.19**, down from
**0.43** at the start of the investigation documented below.

## Paper's reported numbers (Table 1, VoiceBank+DEMAND)

| variant                   | causal | params | MACs | PESQ | STOI % |
| -------------------------- | :----: | -----: | ---: | ---: | -----: |
| Noisy                       |   –    |     –  |    – | 1.97 |     91 |
| **BASENet-3**                | ✗     | 0.83 M | 7.3 G | **3.55** |     96 |
| **BASENet-3 (Causal)**       | ✓     | 0.81 M | 7.1 G | **3.44** |     96 |

Training recipe (paper Sec. 3.1): `n_fft=400`, `hop=100`, 16 kHz, 100 epochs,
batch size 8, Adam (`lr=1e-3`, β=(0.9, 0.999)), exponential LR decay 0.98.
Architecture: B=3 Bark-scale bands at [0, 1, 4, 8] kHz, depths [4, 3, 2]
(`L_max=4`), cross-band attention reduction ratio 4, power-law magnitude
compression `c=0.3`.

## Reproduction progress

| stage                                                        | PESQ-wb (non-causal) |
| -------------------------------------------------------------- | --------------------: |
| Original implementation (guessed widths, single-shot skip fix) | 3.116 *(causal only)* |
| Width re-derivation from the paper's params/MACs               |                  3.330 |
| + MP-SENet trainer fidelity fixes (this file's main result)    |              **3.359** |
| Paper's target                                                 |                   3.55 |

All PESQ numbers are wideband PESQ, mean over the full 824-utterance VBD test
split, using the validation loop's own reconstruction path (`basenet/train.py`
`_validate`).

## What was wrong the first time: guessed widths, not just under-tuning

The initial port matched the paper's architectural *scheme* faithfully (Bark
bands, density-derived depths, cross-band attention, dense/inverted-residual
blocks, CRN) but two hyperparameters the paper leaves unspecified were
guessed wrong, and the guesses were internally consistent enough to look
plausible:

* **The CRN GRU was ~15× too large and wired incorrectly.** The paper's
  causal↔non-causal delta is only 0.02 M params / 0.2 G MACs (0.83 M→0.81 M,
  7.3 G→7.1 G) — bidirectional vs. unidirectional GRUs at that scale can only
  differ that little if the GRU itself is tiny. A flattened-input GRU
  (`hidden=128` on a 512-dim input, 341 K params — 42 % of the model) has a
  0.31 M bi→uni delta, more than 15× the paper's. The wiring consistent with
  *both* deltas simultaneously is a **GRU shared across frequency bins**
  (BSRNN-style: one small GRU, `hidden=56` on a `C=44`-dim input, applied
  independently — same weights — at every frequency bin).
* **The fusion `AvgPool2d` crushed frequency 201→16 bins; the paper only
  halves it to ~100.** This single misread cascades: it explains the missing
  ~2× MACs (paper's compute runs at 100 bins, the original port's at 16), and
  it forced a **non-paper skip connection** from the fusion output straight
  into both decoder heads to keep phase learning from collapsing (verified
  this session: at 100 bins the skip is unnecessary — phase loss drops
  5.1→2.5 in 150 steps with no skip at all).

Re-solving the width allocation against the paper's four published numbers
(causal 0.81 M/7.1 G, non-causal 0.83 M/7.3 G, solved simultaneously) pins:
per-bin shared-weight GRU (`hidden=56`), fusion `AvgPool` at 201→100 bins,
the GRU's freed parameters moved into full-resolution convs
(`base_channels` 32→44), and Fig. 1-faithful per-band stems (each band gets
its own `Conv2d`+`InstanceNorm`+`Hardswish`, not one shared projection).
`basenet/profile_macs.py` is the hook-based per-module params/MACs
accountant this used; `configs/basenet3_noncausal.json` is the solved
non-causal config (0.830 M/7.55 G, causal twin 0.810 M/7.29 G — both within
~3 % of the paper, consistent with counting-convention noise). Commit
`69c8d84`.

Old checkpoints (`cp_basenet_causal/g_best`) still load `strict=True` against
the current model — every re-derivation knob defaults to the original
(guessed) behaviour unless a config sets it explicitly.

## Trainer fidelity: two bugs found by diffing the official MP-SENet trainer

The paper states it "follow[s] the same training procedure and loss
functions as MP-SENet." Diffing `basenet/train.py` line-by-line against the
official reference trainer (`yxlu-0102/MP-SENet`) found two real gaps:

1. **Missing STFT-consistency loss.** MP-SENet adds
   `2 * MSE(com_g, STFT(iSTFT(mag_g, pha_g)))` at weight 0.1, penalising
   `(magnitude, phase)` pairs that are not realisable spectrograms. Added as
   `loss.consistency` (default 0.1 in `configs/basenet3_noncausal.json`).
2. **Discriminator fed the wrong magnitude.** MP-SENet's MetricGAN
   discriminator scores `mag_g_hat` — the magnitude *after* the
   iSTFT→STFT round trip, i.e. what the model's output actually reconstructs
   to — not the network's raw output `mag_g`. Fixed in both the
   discriminator step and the generator's metric loss (`basenet/train.py`,
   `generator_loss`/`gan_step`).

The grad-accumulation exactness test's tolerance moved 1e-5→5e-5 (the extra
STFT round trip deepens the float32 graph; the float64 check stays ~2e-14 —
still exact, not a real regression). Commit `7c03e4b`. This closed
**3.330 → 3.359**, with the improvement concentrated in the LR-decay tail
(the two runs were statistically tied through step ~90 K of 144 K, then
diverged as the LR got small — consistent with a training-signal fix rather
than a capacity fix).

## Data pipeline verified, not the problem

A concrete, cheap check ruled out the dataset as a gap source: computing
noisy-vs-clean PESQ-wb over the full 824-utterance VBD test set gives
**1.9707**, matching the paper's reported "Noisy" baseline of **1.97** almost
exactly. The `JacobLinCool/VoiceBank-DEMAND-16k` HuggingFace mirror this repo
uses is not a source of resampling/provenance error. STFT windowing,
crop/pad logic, and the discriminator architecture (InstanceNorm-based, no
BatchNorm — the grad-accumulation trick is numerically exact for both
generator and discriminator) all match the official MP-SENet reference too.

## Open leads on the remaining ~0.19 gap

Two candidate causes were identified and one was investigated further, per
the project's judgment on which was more likely to matter:

* **Gradient clipping** (`max_norm=5.0` in `basenet/train.py`, both GAN and
  no-GAN paths) has no counterpart in the official MP-SENet trainer (zero
  clipping calls anywhere). A real, verified divergence — not yet tested
  empirically, since it was judged less likely to explain the gap than the
  architecture-level ambiguity below.
* **Expand-ratio ambiguity, searched and largely ruled out.** The paper's
  inverted-residual block spec ("Expand projects C channels to rC, r ∈
  {2, 4}") doesn't say which modules get which ratio; the reproduction uses
  `r=4` uniformly. `basenet/model.py` was extended with four independent
  expand-ratio knobs (`enc_dense_expand_ratio`, `enc_ir_expand_ratio`,
  `dec_dense_expand_ratio`, `dec_ir_expand_ratio`, each defaulting to the
  single `expand_ratio` value for full backward compatibility), and all 16
  combinations of `{2, 4}` were searched against the paper's params/MACs
  targets (re-solving `base_channels`/`growth`/`crn_hidden`/`dec_depth` per
  combination). Result: **any combination using r=2 in the encoder fits
  dramatically worse** (fit-score 48–116 vs. 3–10 for encoder-r=4
  combinations) — the search has to widen `base_channels` to compensate for
  lost encoder capacity, which blows MACs out to 8–10 G against the paper's
  7.1–7.3 G. The paper's own published numbers are essentially only
  self-consistent with **near-uniform r=4**; the best fit found
  (`dec_ir_expand_ratio=2`, everything else r=4, fit-score 2.76) is a small
  tweak, not a large hidden capacity split. This significantly weakens the
  "expand-ratio ambiguity explains the gap" hypothesis, though the marginal
  `dec_ir=2` variant has not been trained to confirm.

Neither lead has been trained and evaluated yet — this section will be
updated once one is.

## Deployment work (done, on the pre-re-derivation causal checkpoint)

The deployment pipeline was built and parity-verified against the original
causal checkpoint (`cp_basenet_causal/g_best`, PESQ 3.116) before the width
re-derivation; it is architecture-agnostic and applies unchanged to any
future causal checkpoint.

* **ONNX export** (`basenet/export_onnx.py`): FP32 export matches PyTorch to
  mag 2.6e-6 / complex 5.2e-5 (identical PESQ). The blocker was
  `AdaptiveAvgPool2d` + nearest interpolation — ONNX rejects non-constant
  output sizes — fixed with `FreqResample`, a parameter-free constant-matmul
  resampler that is numerically identical and stays causal. STFT/iSTFT stay
  outside the graph.
* **int8 quantization** (`basenet/quant_onnx.py`): dynamic weight-only int8
  is near-lossless (3.24→3.19 PESQ, 3.67→2.41 MiB, ~34 % smaller). Static
  full int8 (QDQ) collapses to ~1.2 PESQ regardless of calibration method
  (MinMax/percentile/entropy) — the wide-range compressed spectra and
  `atan2` phase path are too sensitive to fixed activation scales; full int8
  needs QAT (the same finding as ConvFSENet).
* **Frame-by-frame streaming** (`basenet/streaming.py`,
  `BASENetStreamer.reset()/step()`): only the TFConv ring buffers and the
  CRN's GRU hidden state carry cross-frame state; everything else is
  per-frame. Parity vs. the offline forward: mag 4.5e-6, identical PESQ.
  ONNX export of the *streaming* graph (explicit state I/O) is not yet done.

## Causality — what the paper glosses over

The paper states causal streaming needs only "replacing the bidirectional
GRU with a unidirectional variant." That's necessary but not sufficient — a
naive port leaks future context through three more components that pool or
normalise across time. The causal path (`basenet/model.py`) also uses:

* **left-padded time convolutions** (`TFConv`, causal padding `(kt-1, 0)`
  instead of centred) so frame *n* never reads frames > *n*;
* **frequency-only normalisation** (`CausalFreqNorm`) instead of
  `InstanceNorm2d`, which averages over (frequency, time) — a silent future
  leak;
* **frequency-only squeeze-excitation pooling**, giving a per-frame channel
  gate instead of a clip-global one.

Cross-band attention is causal as-is: every frame attends independently
(Eq. 11), which is exactly why the paper highlights its `O(N·F_b·B)` cost.
Genuine causality is verified by a bit-exact future-perturbation test
(`tests/test_basenet_model.py`): perturbing a future frame leaves every
earlier output bit-for-bit unchanged.

## Reproducing

```bash
source .venv/bin/activate
# non-causal BASENet-3 (current best PESQ 3.359 recipe)
python -m basenet.train --config configs/basenet3_noncausal.json \
    --checkpoint_path cp_basenet3_noncausal_v2 --training_epochs 100 --accum_steps 2
# or, SSH-resilient (tmux):
./run_basenet_train.sh cp_basenet3_noncausal_v2 configs/basenet3_noncausal.json 100 2

# params/MACs accounting for any config
python -m basenet.profile_macs

# ONNX export + int8 (on a causal checkpoint)
python -m basenet.export_onnx --checkpoint_file cp_basenet_causal/g_best
python -m basenet.quant_onnx --fp32 cp_basenet_causal/g_best_fp32.onnx --mode dynamic
```

`--accum_steps 2` on top of `batch_size: 4` in the config reproduces the
paper's effective batch 8 (numerically exact — every norm in the model is
per-sample). Real batch 8 OOMs a 24 GB GPU; the model is activation-heavy at
~3.1 GB/sample for 2 s segments, matching its ~7.3 GMACs. Training runs
~0.21 s/optimizer-step, ~10 min/epoch, ~17 h for the full 100-epoch recipe on
an RTX 4090.

There is not yet a `configs/basenet3_causal.json` — the causal twin of the
solved architecture (0.810 M/7.29 G) is reached by setting `"causal": true`
on the same architecture knobs as `configs/basenet3_noncausal.json`, but has
not been trained; the only trained causal checkpoint
(`cp_basenet_causal/g_best`, PESQ 3.116) predates the width re-derivation.
`configs/basenet_eff8_cosine.json` and `configs/basenet_wide.json` are
earlier, superseded gap-closing attempts against the original (wrong-width)
architecture and are kept for reference only.

## Checkpoints

| checkpoint                          | architecture               | causal | PESQ-wb   |
| ------------------------------------ | --------------------------- | :----: | --------: |
| `cp_basenet_causal/g_best`           | original (guessed widths)   |   ✓    |     3.116 |
| `cp_basenet3_noncausal/g_best`       | re-derived widths           |   ✗    |     3.330 |
| `cp_basenet3_noncausal_v2/g_best`    | re-derived + trainer fixes  |   ✗    | **3.359** |

No checkpoint has been published to HuggingFace yet (unlike ConvFSENet,
LiSenNet, and NSNet2) — publishing is deferred until the gap-closing
investigation concludes.
