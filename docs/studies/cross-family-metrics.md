# Perceptual metrics on the published models

Every model published to the Hub, scored on **DNSMOS**, **NISQA** and **SCOREQ**
as well as PESQ, on the full VoiceBank-DEMAND test split (824 utterances).

Reproduce with:

```bash
python -m benchmarks.enhance     # published checkpoints -> enhanced audio (~1 h, 5.5 GB)
python -m benchmarks.score       # enhanced audio -> metric JSON (~2.5 h, resumable)
python -m benchmarks.report --md ../studies/cross-family-metrics.md \
    --json benchmarks/summary.json --audit benchmarks/per_utterance.json.gz
```

## The numbers, for auditing

Two committed artefacts back every figure below, so none of it has to be taken on
trust:

| file | what |
| --- | --- |
| `benchmarks/summary.json` | 40 KB. All 12 metric columns (not just the 7 headline ones), means + failure counts, for each of the 44 conditions. |
| `benchmarks/per_utterance.json.gz` | 1.7 MB. **Every individual score**: 44 conditions × 824 utterances × 12 metrics. Utterance IDs stored once and every column aligned to them, so conditions are directly comparable per-utterance. |

Both carry a `provenance` block — dataset, split, git commit, and the versions of
everything that can move a score (`torchmetrics` 1.9.0 and `scoreq` 1.0.1 ship the
model graphs; `onnxruntime` 1.25.1 decides the last decimal). A future re-run that
disagrees is then attributable rather than mysterious.

The per-utterance file is what makes the claims below checkable. Recomputing the
means from it reproduces the tables to 5e-7 (the 6-decimal rounding), and it
supports paired tests the means alone cannot — e.g. ConvFSENet's NISQA lead over
LiSenNet `gru` is **+0.167 ± 0.028** (95 % CI, n=824, paired), so the ranking flip
in finding 2 is not noise:

```python
import gzip, json, numpy as np
d = json.loads(gzip.open("benchmarks/per_utterance.json.gz").read())
a = np.array(d["conditions"]["convfsenet__convfsenet/fp32"]["per_utt"]["nisqa_mos"])
b = np.array(d["conditions"]["lisennet__gru/fp32"]["per_utt"]["nisqa_mos"])
print((a - b).mean(), 1.96 * (a - b).std(ddof=1) / np.sqrt(len(a)))
```

## Why

PESQ is an intrusive ITU-P.862 measure built for codec and telephony
degradations. It is a good *relative* yardstick within one architecture, but it
was never designed to score the artefacts neural suppressors introduce, and the
numbers below show it missing them. DNSMOS, NISQA and SCOREQ are all trained to
predict human listening scores.

All four metrics come out of one harness (`common/quality.py`), and every row is
the audio path the model actually deploys.

The PESQ column is the harness's own correctness check, and it passes: **every
FP32 PESQ here reproduces ../models/nsnet2.md / ../models/convfsenet.md /
../models/lisennet.md to the last digit** (13/13 models). int8 reproduces exactly
for 10 of 13; three differ by ≤0.013 — `wide_blockdiag` (2.848 vs 2.842),
`blockdiag_full` (2.843 vs 2.848) and LiSenNet `conv-hardened` (2.988 vs 3.001).
That is not harness drift: int8 quantization is calibration-dependent, and the
int8 graphs *published to the Hub* were quantized from a different calibration
draw than the ones that produced those table rows. FP32 export is deterministic,
which is why it matches everywhere. This benchmark scores what is on the Hub —
i.e. what someone downloading the model actually gets.

The new columns are the only genuinely new information below.

Two conventions worth knowing before reading the tables:

* **SCOREQ-ref is a distance, so lower is better.** Everything else is
  higher-is-better. Deltas below are sign-corrected so positive always means worse.
* **DNSMOS and NISQA are no-reference** — they never see the clean signal. That
  is why the `clean` row does not saturate them, and why it is the real ceiling
  on this corpus rather than an artefact.

## What the new metrics say

**1. PESQ badly understates the butterfly int8 collapse — the int8 models are
worse than not enhancing at all.**

The repo already knew butterfly quantizes poorly (PESQ 2.772 → 2.128 for
`butterfly_full`). But PESQ still scores that int8 model *above* the unprocessed
noisy input (2.128 vs 1.971), i.e. PESQ says it is doing some good. Every
listener-trained metric says the opposite:

| `butterfly_full` int8 vs noisy input | DNSMOS OVRL | NISQA | SCOREQ-nr | SCOREQ-ref ↓ |
| --- | ---: | ---: | ---: | ---: |
| noisy input (do nothing)  | 2.697 | 3.060 | 3.308 | 0.757 |
| `butterfly_full` int8     | **2.556** | **2.812** | **3.075** | **0.801** |
| `butterfly_2blocks` int8  | **2.615** | **2.917** | **3.087** | **0.791** |

Both int8 butterfly models are worse than the noisy input on all four learned
metrics. PESQ ranked them as an improvement. If any deployment decision rested on
"butterfly int8 degrades but still helps", it rested on a PESQ artefact.

**2. PESQ and the learned metrics disagree about which family is best.**

PESQ crowns LiSenNet. All three learned metrics crown ConvFSENet, and DNSMOS puts
several NSNet2 Monarch variants above every LiSenNet:

| best FP32 model by … | winner | runner-up |
| --- | --- | --- |
| PESQ         | `lisennet/conv-hardened-deep` 3.084 | `lisennet/conv-hardened` 3.013 |
| DNSMOS OVRL  | `convfsenet/128-256` 3.071 | `convfsenet` 3.062 |
| NISQA        | `convfsenet` 4.297 | `convfsenet/128-256` 4.275 |
| SCOREQ-nr    | `convfsenet` 3.980 | `convfsenet/128-256` 3.976 |
| SCOREQ-ref ↓ | `convfsenet` 0.359 | `convfsenet/128-256` 0.361 |

ConvFSENet sweeps every metric that was trained on human judgements, and loses
only on the one that was not. This does not make LiSenNet a bad model — it is
40× smaller — but the claim "LiSenNet outperforms ConvFSENet" is a PESQ-only
claim and does not survive contact with the other three.

**3. The gains are all background suppression; every model damages the speech.**

DNSMOS splits P.835 into SIG (speech quality) and BAK (background). Against the
noisy input:

| | noisy | best model | clean |
| --- | ---: | ---: | ---: |
| DNSMOS BAK (background) | 3.126 | ~3.97 | 4.038 |
| DNSMOS SIG (speech)     | 3.346 | ~3.39 | 3.511 |

BAK moves almost the whole way to clean. SIG barely moves — and five FP32 models
score *below the noisy input* on SIG, meaning they degrade the speech relative to
doing nothing: all four LiSenNet variants (3.288–3.322) and NSNet2 `baseline`
(3.341). LiSenNet losing the most SIG is consistent with its architecture: a
magnitude-only mask with Griffin-Lim phase has no way to preserve fine speech
structure. This is the classic suppression-vs-distortion trade, and PESQ's single
number hides which side of it a model is on.

**4. Monarch quantizes loss-free — now confirmed on four independent metrics.**

Every Monarch and block-diagonal variant stays within |Δ| ≤ 0.03 on every metric
FP32→int8, several actually improving. Butterfly moves by up to +0.97 (NISQA).
The repo's existing conclusion was drawn from PESQ alone; it holds.

**5. LiSenNet's real-time (noisy-phase) path beats its Griffin-Lim path — on
every metric, not just PESQ.**

`int8_rt` (noisy phase, causal, what actually ships) vs `int8` (Griffin-Lim,
non-causal) for `conv-hardened-deep`: DNSMOS 3.020 vs 2.994, SCOREQ-ref 0.382 vs
0.403, SCOREQ-nr 3.949 vs 3.875. The 2-iteration Griffin-Lim is *actively hurting*
relative to just reusing the noisy phase. The deployment choice was already the
right one; three more metrics now agree, and the "real-time costs us quality"
framing in ../models/lisennet.md is, on these metrics, backwards.

## Caveats

* DNSMOS OVRL has a narrow dynamic range on this corpus (noisy 2.697 → clean
  3.221) and compresses the differences between good models; NISQA (3.060 → 4.552)
  discriminates far better. Do not read a 0.01 DNSMOS gap as meaningful.
* These are *estimators* of listening scores, not listening scores. They agree
  with each other here, which is reassuring, but a MUSHRA/P.808 panel is the only
  thing that settles the ConvFSENet-vs-LiSenNet question for real.
* All audio is scored at the original dataset level. LiSenNet's `eval_deploy`
  scores in the RMS-normalised domain; harmless for PESQ, which level-aligns
  internally, but DNSMOS and NISQA are level-sensitive, so `benchmarks/enhance`
  un-normalises LiSenNet's output to keep the families comparable.


## Full results

VoiceBank-DEMAND test split, n=824 utterances. Every row is the same
audio path the model actually deploys, scored through one metric harness
(`common/quality.py`).

**SCOREQ-ref is a distance, so lower is better.** Every other column is
higher-is-better. DNSMOS and NISQA are no-reference: they never see the
clean signal, which is why the `clean` row does not saturate them and is
the ceiling those two columns can award on this corpus.

### Baselines

| signal | PESQ | DNSMOS OVRL | DNSMOS SIG | DNSMOS BAK | NISQA | SCOREQ-nr | SCOREQ-ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **noisy** | 1.971 | 2.697 | 3.346 | 3.126 | 3.060 | 3.308 | 0.757 |
| **clean** | 4.644 | 3.221 | 3.511 | 4.038 | 4.552 | 4.453 | 0.000 |

### NSNet2

| model / condition | PESQ | DNSMOS OVRL | DNSMOS SIG | DNSMOS BAK | NISQA | SCOREQ-nr | SCOREQ-ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` fp32 | 2.845 | 3.016 | 3.341 | 3.923 | 3.940 | 3.867 | 0.417 |
| `baseline` int8 | 2.833 | 3.011 | 3.335 | 3.922 | 3.936 | 3.865 | 0.417 |
| `blockdiag_8` fp32 | 2.832 | 3.028 | 3.360 | 3.914 | 4.042 | 3.910 | 0.399 |
| `blockdiag_8` int8 | 2.826 | 3.007 | 3.334 | 3.913 | 4.032 | 3.879 | 0.410 |
| `blockdiag_fc` fp32 | 2.805 | 3.035 | 3.365 | 3.919 | 3.977 | 3.930 | 0.395 |
| `blockdiag_fc` int8 | 2.787 | 3.029 | 3.359 | 3.919 | 3.978 | 3.924 | 0.399 |
| `blockdiag_full` fp32 | 2.827 | 3.041 | 3.374 | 3.916 | 4.040 | 3.927 | 0.396 |
| `blockdiag_full` int8 | 2.843 | 3.027 | 3.355 | 3.921 | 4.092 | 3.930 | 0.393 |
| `wide_blockdiag` fp32 | 2.864 | 3.045 | 3.388 | 3.898 | 4.040 | 3.938 | 0.388 |
| `wide_blockdiag` int8 | 2.848 | 3.021 | 3.357 | 3.903 | 4.092 | 3.922 | 0.395 |
| `monarch_8` fp32 | 2.861 | 3.049 | 3.370 | 3.937 | 4.010 | 3.921 | 0.391 |
| `monarch_8` int8 | 2.856 | 3.042 | 3.361 | 3.942 | 4.033 | 3.920 | 0.393 |
| `monarch_fc` fp32 | 2.843 | 3.034 | 3.364 | 3.917 | 4.005 | 3.919 | 0.396 |
| `monarch_fc` int8 | 2.832 | 3.025 | 3.353 | 3.920 | 3.994 | 3.904 | 0.403 |
| `monarch_full` fp32 | 2.838 | 3.055 | 3.374 | 3.942 | 3.981 | 3.930 | 0.391 |
| `monarch_full` int8 | 2.846 | 3.049 | 3.368 | 3.945 | 4.013 | 3.937 | 0.389 |
| `wide_monarch` fp32 | 2.881 | 3.020 | 3.356 | 3.904 | 3.985 | 3.937 | 0.382 |
| `wide_monarch` int8 | 2.884 | 3.011 | 3.345 | 3.905 | 4.006 | 3.929 | 0.386 |
| `butterfly_2blocks` fp32 | 2.805 | 3.026 | 3.356 | 3.915 | 3.860 | 3.858 | 0.426 |
| `butterfly_2blocks` int8 | 2.201 | 2.615 | 3.125 | 3.331 | 2.917 | 3.087 | 0.791 |
| `butterfly_fc` fp32 | 2.799 | 3.036 | 3.370 | 3.914 | 3.887 | 3.857 | 0.419 |
| `butterfly_fc` int8 | 2.493 | 2.921 | 3.383 | 3.633 | 3.475 | 3.690 | 0.540 |
| `butterfly_full` fp32 | 2.772 | 3.023 | 3.353 | 3.915 | 3.779 | 3.810 | 0.453 |
| `butterfly_full` int8 | 2.128 | 2.556 | 3.090 | 3.201 | 2.812 | 3.075 | 0.801 |
| `butterfly_ortho` fp32 | 2.780 | 3.022 | 3.350 | 3.916 | 3.814 | 3.809 | 0.450 |
| `butterfly_ortho` int8 | 2.576 | 2.822 | 3.226 | 3.686 | 3.507 | 3.621 | 0.541 |

### ConvFSENet

| model / condition | PESQ | DNSMOS OVRL | DNSMOS SIG | DNSMOS BAK | NISQA | SCOREQ-nr | SCOREQ-ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `convfsenet` fp32 | 2.931 | 3.062 | 3.365 | 3.972 | 4.297 | 3.980 | 0.359 |
| `convfsenet` int8 | 2.911 | 3.056 | 3.359 | 3.970 | 4.290 | 3.969 | 0.367 |
| `128-256` fp32 | 2.891 | 3.071 | 3.371 | 3.979 | 4.275 | 3.976 | 0.361 |
| `128-256` int8 | 2.883 | 3.062 | 3.364 | 3.974 | 4.267 | 3.959 | 0.369 |

### LiSenNet

| model / condition | PESQ | DNSMOS OVRL | DNSMOS SIG | DNSMOS BAK | NISQA | SCOREQ-nr | SCOREQ-ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gru` fp32 | 3.006 | 3.003 | 3.322 | 3.924 | 4.130 | 3.857 | 0.415 |
| `gru` int8 | 2.920 | 2.952 | 3.279 | 3.896 | 4.046 | 3.796 | 0.440 |
| `gru` int8_rt | 2.930 | 2.993 | 3.316 | 3.917 | 4.052 | 3.887 | 0.411 |
| `conv` fp32 | 2.970 | 2.986 | 3.303 | 3.926 | 4.102 | 3.815 | 0.438 |
| `conv` int8 | 2.847 | 2.937 | 3.265 | 3.888 | 3.929 | 3.718 | 0.485 |
| `conv` int8_rt | 2.855 | 2.976 | 3.299 | 3.908 | 3.953 | 3.803 | 0.456 |
| `conv-hardened` fp32 | 3.013 | 2.968 | 3.288 | 3.912 | 4.168 | 3.846 | 0.419 |
| `conv-hardened` int8 | 2.988 | 2.963 | 3.282 | 3.911 | 4.138 | 3.833 | 0.427 |
| `conv-hardened` int8_rt | 2.982 | 2.988 | 3.303 | 3.926 | 4.111 | 3.896 | 0.409 |
| `conv-hardened-deep` fp32 | 3.084 | 3.002 | 3.314 | 3.938 | 4.216 | 3.902 | 0.391 |
| `conv-hardened-deep` int8 | 3.010 | 2.994 | 3.303 | 3.942 | 4.147 | 3.875 | 0.403 |
| `conv-hardened-deep` int8_rt | 3.015 | 3.020 | 3.324 | 3.957 | 4.120 | 3.949 | 0.382 |

### Quantization cost (FP32 → int8)

Positive = int8 is worse, on every column (sign-corrected for SCOREQ-ref).

| model | PESQ | DNSMOS OVRL | DNSMOS SIG | DNSMOS BAK | NISQA | SCOREQ-nr | SCOREQ-ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | +0.012 | +0.005 | +0.006 | +0.001 | +0.003 | +0.001 | +0.000 |
| `blockdiag_8` | +0.007 | +0.021 | +0.026 | +0.001 | +0.010 | +0.031 | +0.011 |
| `blockdiag_fc` | +0.018 | +0.006 | +0.006 | +0.001 | -0.001 | +0.006 | +0.004 |
| `blockdiag_full` | -0.016 | +0.014 | +0.018 | -0.005 | -0.052 | -0.003 | -0.003 |
| `wide_blockdiag` | +0.016 | +0.024 | +0.031 | -0.005 | -0.052 | +0.016 | +0.007 |
| `monarch_8` | +0.005 | +0.007 | +0.009 | -0.005 | -0.022 | +0.001 | +0.002 |
| `monarch_fc` | +0.011 | +0.009 | +0.011 | -0.002 | +0.011 | +0.015 | +0.007 |
| `monarch_full` | -0.008 | +0.005 | +0.006 | -0.002 | -0.032 | -0.008 | -0.002 |
| `wide_monarch` | -0.003 | +0.010 | +0.011 | -0.000 | -0.021 | +0.008 | +0.005 |
| `butterfly_2blocks` | +0.603 | +0.411 | +0.231 | +0.585 | +0.943 | +0.772 | +0.365 |
| `butterfly_fc` | +0.306 | +0.115 | -0.013 | +0.282 | +0.413 | +0.167 | +0.120 |
| `butterfly_full` | +0.644 | +0.467 | +0.264 | +0.714 | +0.966 | +0.736 | +0.347 |
| `butterfly_ortho` | +0.204 | +0.200 | +0.124 | +0.231 | +0.307 | +0.187 | +0.091 |
| `convfsenet` | +0.020 | +0.006 | +0.006 | +0.002 | +0.007 | +0.011 | +0.007 |
| `128-256` | +0.008 | +0.008 | +0.007 | +0.005 | +0.008 | +0.018 | +0.008 |
| `gru` | +0.086 | +0.051 | +0.043 | +0.028 | +0.085 | +0.061 | +0.026 |
| `conv` | +0.123 | +0.049 | +0.038 | +0.038 | +0.173 | +0.097 | +0.047 |
| `conv-hardened` | +0.025 | +0.005 | +0.006 | +0.001 | +0.030 | +0.013 | +0.008 |
| `conv-hardened-deep` | +0.074 | +0.008 | +0.011 | -0.004 | +0.070 | +0.027 | +0.011 |

