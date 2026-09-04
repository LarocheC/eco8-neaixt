# NSNet2 — results

> **⚠️ Naming correction.** Every run in this document previously named
> `monarch_*` (`blockdiag_8`, `blockdiag_full`, `blockdiag_fc`,
> `wide_blockdiag` — already renamed throughout below) is a **single
> block-diagonal factor** (zero cross-block mixing), **not** a genuine two-factor
> [Monarch](https://arxiv.org/abs/2204.00595). Genuine two-factor Monarch
> (`kind="monarch"`, torch-structured ≥ 1.3.0) has now been trained and measured
> — see [Genuine two-factor Monarch](#genuine-two-factor-monarch) below, and
> `../studies/structured-factorisation.md` for the plan.
>
> **⚠️ int8 correction (Einsum weight-quant bug).** Every int8 PESQ published
> before 2026-07-11 was measured on a model whose **structured weights were still
> FP32**. The structure-preserving export lowers each block-diagonal / Monarch
> matmul to an `Einsum`, and onnxruntime ships no QDQ handler for `Einsum` — so
> `quantize_static` skipped those nodes entirely, quantizing only activations and
> the residual dense MatMuls. Fixed in `nsnet2/qdq_einsum_quantizer.py` (registers
> `QDQRegistry["Einsum"]`, per-channel axis=1) plus an audit in `nsnet2/quant.py`
> that fails if any `Einsum` operand is still a raw FLOAT initializer. **All int8
> numbers below have been re-measured with the structured weights genuinely
> quantized.** Blast radius: the general-purpose streaming ONNX only — the
> STM32N6 NPU export (`export_blockdiag_npu.py`) re-expresses the blocks as
> per-block `MatMul`, which onnxruntime quantizes natively, so the deployment
> results further down are unaffected.

NSNet2 speech enhancement (Braun & Tashev, ICASSP 2021), with the FC and
GRU layers swappable between dense, [Butterfly](https://arxiv.org/abs/1903.05895),
block-diagonal, and genuine two-factor [Monarch](https://arxiv.org/abs/2204.00595)
structured factorizations
(via the [torch-structured](https://pypi.org/project/torch-structured/) PyPI package).
Built on top of the [MP-SENet](https://github.com/yxlu-0102/MP-SENet) training recipe.

See [../models/convfsenet.md](../models/convfsenet.md) for the convolutional
model family, and the [README](../../README.md) for setup and repo layout.

## Headline results — FP32 vs int8

9-config sweep on VoiceBank-DEMAND test set (200 epochs, batch 256,
n\_fft 512). PESQ measured on the full 824-utterance test split for both
the FP32 ONNX and the static int8 ONNX (QDQ format, per-channel weights,
MinMax calibration on 200 train utterances). RTF is for the int8 session
under onnxruntime CPU; lower is faster.

int8 PESQ for the block-diagonal rows is **re-measured post-Einsum-fix** (see the
int8 correction above); butterfly and dense are unaffected by that bug (their
exports emit no `Einsum`) and carry their original figures.

| run                 | params  | FP32 PESQ |  int8 PESQ |   Δ (FP32→int8) | compression |
| ------------------- | ------: | --------: | ---------: | --------------: | ----------: |
| `wide_blockdiag`    |  2.36 M | **2.864** |  **2.847** |         +0.016  |       1.2× |
| `baseline` (dense)  |  2.78 M |     2.845 |      2.833 |         +0.012  |       1.0× |
| `blockdiag_8`       |  0.36 M |     2.832 |      2.825 |         +0.007  |       7.7× |
| `blockdiag_full`    |  0.70 M |     2.827 |      2.843 |         −0.016  |       4.0× |
| `blockdiag_fc`      |  2.14 M |     2.805 |      2.787 |         +0.018  |       1.3× |
| `butterfly_2blocks` |  0.36 M |     2.805 |      2.202 |         +0.602  |       7.7× |
| `butterfly_fc`      |  1.99 M |     2.799 |      2.494 |         +0.306  |       1.4× |
| `butterfly_ortho`   |  0.19 M |     2.780 |      2.577 |         +0.203  |        15× |
| `butterfly_full`    |  0.19 M |     2.772 |      2.128 |         +0.644  |        15× |

`wide_blockdiag` leads on FP32 PESQ. Block-diagonal quantization is essentially
loss-free (|Δ| ≤ 0.018) **even with the structured weights genuinely quantized** —
the pre-fix "loss-free" claim happened to survive the correction, because these
block weights quantize cleanly rather than because they were being skipped.
Butterfly with randn init remains the outlier (Δ up to 0.64).

RTF is omitted from this table: the block-diagonal rows were re-timed on a
different machine/load than the original butterfly rows, so the columns are not
comparable. Internally-consistent timings (all measured together) are in the
Monarch table below.

## Genuine two-factor Monarch

The `monarch_*` runs below are the **real** [Monarch](https://arxiv.org/abs/2204.00595)
construction — block-diagonal × permutation × block-diagonal (`MonarchLinear`,
torch-structured ≥ 1.3.0), full cross-channel mixing — as opposed to the single
block-diagonal factor the repo previously mislabeled "monarch". Trained with the
same recipe as the block-diagonal sweep (200 epochs, batch 256, n\_fft 512), GRU
recurrence through gru-qat ≥ 0.5.0's fused (shared-`w1`) Monarch Triton kernel.
All rows here — PESQ **and** RTF — were measured together on one box, so they are
internally comparable.

| run             | params  | FP32 PESQ | int8 PESQ | Δ (FP32→int8) | int8 RTF |
| --------------- | ------: | --------: | --------: | ------------: | -------: |
| `wide_monarch`  | 3.635 M | **2.881** | **2.884** |        −0.003 |    0.051 |
| `monarch_8`     | 0.553 M |     2.861 |     2.856 |        +0.005 |    0.027 |
| `monarch_fc`    | 2.379 M |     2.843 |     2.831 |        +0.012 |    0.073 |
| `monarch_full`  | 1.099 M |     2.838 |     2.846 |        −0.009 |    0.039 |

Head-to-head vs the block-diagonal sibling at matched `nblocks` (FP32, the
quantization-free comparison):

| pair              | block-diagonal    | genuine Monarch   | Δ FP32 | Δ params |
| ----------------- | ----------------- | ----------------- | -----: | -------: |
| `*_8` (nblocks 8) | 2.832 (0.36 M)    | **2.861** (0.55 M) | +0.029 |  +0.19 M |
| `*_full` (nb 4)   | 2.827 (0.70 M)    | **2.838** (1.10 M) | +0.011 |  +0.40 M |
| `*_fc`            | 2.805 (2.14 M)    | **2.843** (2.38 M) | +0.038 |  +0.24 M |
| `wide_*`          | 2.864 (2.36 M)    | **2.881** (3.64 M) | +0.017 |  +1.28 M |

**Genuine Monarch is consistently but marginally better than block-diagonal**
(+0.011…+0.038 FP32), and it costs parameters to get there — Monarch's second
factor makes it larger at equal `nblocks`. It is also **genuinely int8-loss-free**
(|Δ| ≤ 0.012 with the weights actually quantized), matching block-diagonal's
quantization robustness.

### Quality is capacity-bound? No — it saturates

The most useful result here is a negative one. Across **three structure families
and ~10× parameters**, every configuration lands in a narrow band:

| family          | param range    | FP32 PESQ band |
| --------------- | -------------- | -------------- |
| dense           | 2.78 M         | 2.845          |
| block-diagonal  | 0.36 – 2.36 M  | 2.81 – 2.86    |
| genuine Monarch | 0.55 – 3.64 M  | 2.84 – 2.88    |

It is not even monotonic in capacity: the 0.553 M `monarch_8` (2.861) beats both
the 1.099 M `monarch_full` (2.838) and the 2.379 M `monarch_fc` (2.843). Going
7× from `monarch_8` to `wide_monarch` buys +0.020 PESQ — barely above the
run-to-run noise of the metric.

**This model class is architecture/data-bound, not capacity-bound.** NSNet2
predicts a magnitude gain mask and reuses the noisy phase, which caps achievable
PESQ regardless of how expressive the mask predictor is (ConvFSENet reaches 2.911
and LiSenNet ~3.0 in this same repo via different mechanisms); VoiceBank-DEMAND
is also small enough to impose its own ceiling. Consequences:

- **The dense model was already over-parameterized for this task**, which is
  precisely why aggressive structuring costs ~nothing — the compression story
  holds because capacity was never the binding constraint.
- **For deployment take the smallest/fastest** (`blockdiag_8` / `monarch_8`): no
  quality penalty, best RTF.
- **To move PESQ you must change the architecture** (phase-aware / complex mask,
  or lean harder on the GAN path) **or the data** — not scale the linear layers.
  Chasing structure or width within this architecture is not where the gains are.

## STM32N6 on-board deployment

All three speech-enhancement models in this repo now run on the
**STM32N6570-DK** (STM32N657 — Cortex-M55 @ 800 MHz + Neural-ART NPU @
1 GHz), compiled with ST Edge AI Core 4.0.1, fully scripted (no
STM32CubeIDE). On-target latency is per streaming frame (hop 256 @
16 kHz = 16 ms budget); RTF < 1 is real-time.

| model (int8)            | int8 PESQ | weights | on-chip? | latency/frame |     RTF | on-target cos |
| ----------------------- | --------: | ------: | :------: | ------------: | ------: | ------------: |
| **`blockdiag_full`** (sparse)| **2.848** | 0.72 MB |  ✅    |  **2.13 ms**  |**0.13** |      0.99979  |
| `blockdiag_8` (sparse)    |     2.826 | 0.37 MB |    ✅    |     2.89 ms   |   0.18  |      0.99994  |
| ConvFSENet (conv)       |     2.911 | 1.40 MB |    ✅    |     4.40 ms   |   0.275 |      0.990    |
| `baseline` (dense GRU)  |     2.833 | 2.70 MB |    ✗     |    22.94 ms   |   1.43  |      0.9946   |

**Structured sparsity is what lets the recurrent model hit real-time on
this NPU.** The Neural-ART runs fastest when weights live in on-chip
npuRAM. The dense GRU baseline's 2.70 MB int8 weights overflow it, so it
streams them from external octoFlash every frame and lands at RTF 1.43 —
*not* real-time. The sparse block-diagonal variants are 4–8× smaller, fit
entirely on-chip, and dominate: **`blockdiag_full` is the best of all four**
— fastest (2.13 ms, RTF 0.13 — **11× faster than dense, 2× faster than
ConvFSENet**) *and* highest int8 PESQ (2.848). `blockdiag_8` (more, smaller
blocks: nblocks 8 vs 4) is a touch slower at 2.89 ms; fewer larger blocks
map more efficiently to the NPU (88 epochs vs 134).

Two deployment subtleties, both detailed in
[`deploy/stm32n6/NSNET2_DEPLOYMENT_NOTES.md`](../../deploy/stm32n6/NSNET2_DEPLOYMENT_NOTES.md):

* **Dense** doesn't compile as-exported — onnxruntime fuses the GRU
  `MatMul`+`Add` into a `Gemm` with an *activation* `C`, which the
  Neural-ART int8 lowering can't index. Re-quantizing with
  `quant_pre_process(skip_optimization=True)` keeps them separate; the
  result is numerically identical to the published int8.
* **The block-diagonal variants** don't compile as-exported either — the
  block-diagonal block-matmul (`Einsum` + `Pad` + block reshapes) defeats the compiler's
  shape engine, and a 4-D grouped-conv re-export compiles in FP32 but
  hits int8 HW-lowering batch-dim asserts. The fix is to re-express the
  blocks in the *rank-2 `MatMul`* op vocabulary that the dense baseline
  already maps to HW: per-block `Slice` + `MatMul` + `Concat`, flat
  states, and the gate rewritten `(1-z)·n + z·h = n + z·(h-n)`.
  `deploy/stm32n6/host/export_blockdiag_npu.py` does this from the trained
  checkpoint (parity ~5e-7, any fully-blockdiag config — dims read from the
  checkpoint), then int8-quantizes with the same recipe. Each deployed
  artifact matches its stock int8 to mask cosine 0.999, so it carries the
  published PESQ.

Caveats: on-target cosine is vs the FP32 ONNX reference over a 10-sample
`stedgeai validate` run; the validation firmware is a volatile RAM image;
and these are single-run latencies. `wide_blockdiag` also holds int8 PESQ
but at 2.36 M / 9.5 MB int8 it would not fit on-chip; the export script
handles any fully-blockdiag config (`blockdiag_fc`'s dense GRU is rejected).

### Int8 quantization findings

All checkpoints export to streaming-shape FP32 ONNX and quantize to int8:

```bash
./run_quantize_sweep.sh                      # FP32 + int8 ONNX per cp dir
./run_eval_sweep.sh                          # PESQ on full test split
MAX_UTTERANCES=100 ./run_eval_sweep.sh       # quick directional read
```

Three findings worth flagging:

* **Block-diagonal and Monarch variants quantize loss-free** — |Δ| ≤ 0.018
  (block-diagonal) and ≤ 0.012 (Monarch), **re-measured with the structured
  weights genuinely quantized** (see the int8 correction at the top; the earlier
  figures were taken on models whose `Einsum` weights were still FP32). The block
  factors survive per-channel int8 on their own merits: the claim happens to hold
  after the fix, but it was not *tested* before it.
* **Butterfly with `init=ortho` is the right choice for int8 deployment**.
  The cumulative log\_n-stage transform's stage-by-stage activation
  magnitude stays bounded when twiddles are spectrally constrained; randn
  init lets it grow ~3× across 9 stages, compounding QDQ rounding error.
  Same architecture, same training data, same sweep — `butterfly_ortho`
  loses 0.20 PESQ to int8, `butterfly_full` (randn init) loses 0.64.
* **Longer training makes randn-init butterfly *worse* on int8.** The same
  `butterfly_full` config saw its int8 gap grow from 0.36 → 0.64 PESQ
  going from 50 → 200 epochs, as twiddles drifted further from
  orthogonality. Ortho-init butterfly does not show this regression.

For training-time mitigation when ortho init isn't available, a soft
orthogonality penalty (`butterfly_ortho_lambda`) is wired into `nsnet2/train.py`
— see `nsnet2.layers.butterfly_ortho_penalty`.

### Calibration: why un-normalized MinMax

A natural question is whether the static-int8 calibration set should be
RMS-normalized to match the deployment pipeline — training and
`inference_onnx.py` both scale each utterance by `sqrt(N / Σx²)` before the
STFT, whereas `nsnet2/calibration.py` calibrates on the raw audio. An ablation
on `blockdiag_8` (full 824-utt test PESQ; FP32 = 2.832) says **keep the
un-normalized MinMax calibration**:

| calibration                          | int8 PESQ |
| ------------------------------------ | --------: |
| un-normalized + MinMax (**default**) | **2.826** |
| RMS-normalized + Percentile          |     2.795 |
| RMS-normalized + Entropy             |     2.793 |
| RMS-normalized + MinMax              |     2.768 |

Matching calibration to the (narrower) RMS-normalized deployment range
*tightens* the quantization range and clips more activation outliers, costing
~0.03–0.06 PESQ on the wider block-diagonal variants; the wider un-normalized range
acts as a beneficial clipping margin. Outlier-robust calibration
(Entropy/Percentile) recovers part of the gap but does not beat un-normalized
MinMax. Calibration method and a per-utterance frame cap are exposed via
`python -m nsnet2.quant --calibration_method {MinMax,Percentile,Entropy}
--frames_per_utterance N`.

## Int4 weight + int8 activation (PTQ)

Pushing weights further to int4 (per-channel symmetric) with int8 activations,
post-training (no fine-tune), on a 100-utterance sample of the test split.
Same `g_best` checkpoints as the int8 table above; the eval applies
`common.quant_fake.apply_ptq` and runs through `nsnet2/eval_torch.py`'s streaming PESQ
pipeline.

| run                 | fp32 PESQ |  w4/a8 PTQ |   Δ      | verdict       |
| ------------------- | --------: | ---------: | -------: | :------------ |
| `blockdiag_fc`        |     2.841 |  **2.832** | −0.009   | OK (PTQ enough)|
| `baseline`          |     2.867 |      2.813 | −0.053   | borderline     |
| `wide_blockdiag`      |     2.905 |      2.742 | −0.163   | needs QAT      |
| `blockdiag_full`      |     2.853 |      2.558 | −0.295   | needs QAT      |
| `blockdiag_8`         |     2.864 |      2.502 | −0.362   | needs QAT      |
| `butterfly_2blocks` |     2.822 |      2.163 | −0.659   | needs QAT      |
| `butterfly_full`    |     2.811 |      2.034 | −0.776   | needs QAT      |
| `butterfly_ortho`   |     2.795 |      1.832 | −0.964   | needs QAT      |
| `butterfly_fc`      |     2.848 |      1.850 | −0.998   | needs QAT      |

Only `blockdiag_fc` (the dense-GRU + blockdiag-FC variant) and the dense
`baseline` survive int4 PTQ at the 200-epoch checkpoint quality;
everything with a structured GRU collapses. For those configurations
QAT closes the gap.

## Int4/a8 QAT recovery (reconstruction loss, LR 3e-4)

Same 7 needs-QAT 200-epoch checkpoints, fine-tuned with the
parametrize-based STE fake-quant scaffold in `common/quant_fake.py`. PESQ on
the full 824-utterance test split, with quant active during eval.
Two QAT durations shown — a quick 10-epoch first read and the
overnight 100-epoch sweep.

| run                 | fp32 PESQ | w4/a8 PTQ |  10-ep QAT | 100-ep QAT | PTQ gap  | 100-ep gap | recovered |
| ------------------- | --------: | --------: | ---------: | ---------: | -------: | ---------: | --------: |
| `blockdiag_full`      |     2.853 |     2.558 |      2.724 |  **2.783** | −0.295   |   −0.070   |   76%     |
| `wide_blockdiag`      |     2.905 |     2.742 |      2.784 |  **2.786** | −0.163   |   −0.119   |   27%     |
| `blockdiag_8`         |     2.864 |     2.502 |      2.682 |  **2.744** | −0.362   |   −0.120   |   67%     |
| `butterfly_fc`      |     2.848 |     1.850 |      2.596 |  **2.717** | −0.998   |   −0.131   |   87%     |
| `butterfly_ortho`   |     2.795 |     1.832 |      2.511 |  **2.663** | −0.964   |   −0.132   |   86%     |
| `butterfly_2blocks` |     2.822 |     2.163 |      2.542 |  **2.666** | −0.659   |   −0.156   |   76%     |
| `butterfly_full`    |     2.811 |     2.034 |      2.390 |  **2.643** | −0.776   |   −0.168   |   78%     |

**100 epochs lands every config within 0.07–0.17 PESQ of fp32**, even
the butterfly variants that lost ~1.0 PESQ to PTQ. Final gaps cluster
in a tight ~0.1-PESQ band despite PTQ baselines spanning an order of
magnitude. Highlights:

* `butterfly_fc` had the largest absolute recovery — **+0.867 PESQ**
  from PTQ 1.850 to QAT 2.717.
* `blockdiag_full` lands closest to fp32 of any QAT config — only
  −0.070 below the trained fp32 baseline.
* `wide_blockdiag` was already at its floor at 10 epochs; the other six
  all gained meaningfully from the 10→100 extension, with the
  worst-PTQ configs gaining the most.

The remaining ~0.1-PESQ gap is plausibly the floor of naïve dynamic-
scale STE QAT — further closure would likely need LSQ learnable
scales, LR cosine schedule, or training w4-aware from scratch.
Sweep drivers: `nsnet2/sweep_hf_ptq.py`, `run_qat_sweep.sh`
(`EPOCHS`-overridable). QAT driver: `nsnet2/qat_train.py`.

## Reproducing

Train a single config:

```bash
source .venv/bin/activate
python -m nsnet2.train --config configs/baseline.json --checkpoint_path cp_baseline
```

The full 9-run sweep:

```bash
EPOCHS=200 ./run_sweep.sh > sweep.log 2>&1 &
```

Each run writes to `cp_<name>/` (config copy, tensorboard logs under
`logs/`, stdout `train.log`, plus `g_best` once PESQ improves). Resumes
automatically from the latest checkpoint on re-invocation.

The pluggable backends are picked from each config:

```json
"linear": {"kind": "linear" | "butterfly" | "blockdiag" | "monarch", ...kwargs},
"gru":    {"kind": "gru"    | "butterfly" | "blockdiag" | "monarch"
                   | "triton" | "triton_blockdiag" | "triton_monarch"
                   | "triton_butterfly", ...kwargs}
```

Per-backend kwargs (all optional): butterfly takes `nblocks` (1+), `init`
(`randn` / `ortho`), `x_init`, `h_init`; blockdiag and monarch take `nblocks`
(≥ 2). `blockdiag` is a single block-diagonal factor; `monarch` is the genuine
two-factor construction (the runs below are all `blockdiag`).
See `nsnet2/layers.py` for the factory and `StructuredGRU`.

The `analyze_sweep.ipynb` notebook loads each run's `g_best`, plots PESQ
trajectories, visualizes the equivalent dense weight matrices for every
linear and GRU projection, and runs inference on a few test items with
side-by-side spectrograms and audio.

## Trained checkpoints

The 9 best-PESQ generators (`g_best`), the streaming FP32 ONNX
(`g_best_fp32.onnx`), the static int8 ONNX (`g_best.onnx`), and the
exact configs they were trained with are mirrored on HuggingFace at
[`claroche1/sparse-nsnet2-checkpoints`](https://huggingface.co/claroche1/sparse-nsnet2-checkpoints).

PyTorch:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from nsnet2.model import NSNet2

REPO = "claroche1/sparse-nsnet2-checkpoints"
RUN  = "wide_blockdiag"  # or any run name from the table above

cfg  = json.load(open(hf_hub_download(REPO, f"{RUN}/config.json")))
ckpt = torch.load(hf_hub_download(REPO, f"{RUN}/g_best"),
                  map_location="cuda", weights_only=False)
model = NSNet2(AttrDict(cfg)).cuda().eval()
model.load_state_dict(ckpt["generator"])
```

ONNX (FP32 or int8):

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

int8_path = hf_hub_download(REPO, f"{RUN}/g_best.onnx")          # deployment
sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
# Streaming shape: feed one frame (B, n_freq) + state (num_layers, B, hidden) per call.
# End-to-end RMS-norm + STFT + frame loop + iSTFT pipeline is in nsnet2/inference_onnx.py.
```
