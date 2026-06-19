# NSNet2 — results

NSNet2 speech enhancement (Braun & Tashev, ICASSP 2021), with the FC and
GRU layers swappable between dense, [Butterfly](https://arxiv.org/abs/1903.05895)
and [Monarch](https://arxiv.org/abs/2204.00595) structured factorizations
(via the [torch-structured](https://pypi.org/project/torch-structured/) PyPI package).
Built on top of the [MP-SENet](https://github.com/yxlu-0102/MP-SENet) training recipe.

See [RESULTS_CONVFSENET.md](RESULTS_CONVFSENET.md) for the convolutional
model family, and the [README](README.md) for setup and repo layout.

## Headline results — FP32 vs int8

9-config sweep on VoiceBank-DEMAND test set (200 epochs, batch 256,
n\_fft 512). PESQ measured on the full 824-utterance test split for both
the FP32 ONNX and the static int8 ONNX (QDQ format, per-channel weights,
MinMax calibration on 200 train utterances). RTF is for the int8 session
under onnxruntime CPU; lower is faster.

| run                 | params  | FP32 PESQ |  int8 PESQ |   Δ (FP32→int8) | int8 RTF | compression |
| ------------------- | ------: | --------: | ---------: | --------------: | -------: | ----------: |
| `wide_monarch`      |  2.36 M | **2.864** |      2.842 |         +0.021  |    0.166 |       1.2× |
| `baseline`          |  2.78 M |     2.845 |      2.833 |         +0.012  |    0.452 |       1.0× |
| `monarch_8`         |  0.36 M |     2.832 |      2.826 |         +0.006  | **0.025**|       7.7× |
| `monarch_full`      |  0.70 M |     2.827 |  **2.848** |         −0.021  |    0.039 |       4.0× |
| `monarch_fc`        |  2.14 M |     2.805 |      2.789 |         +0.016  |    0.448 |       1.3× |
| `butterfly_2blocks` |  0.36 M |     2.805 |      2.202 |         +0.602  |    0.441 |       7.7× |
| `butterfly_fc`      |  1.99 M |     2.799 |      2.494 |         +0.306  |    0.522 |       1.4× |
| `butterfly_ortho`   |  0.19 M |     2.780 |      2.577 |         +0.203  |    0.232 |        15× |
| `butterfly_full`    |  0.19 M |     2.772 |      2.128 |         +0.644  |    0.230 |        15× |

`wide_monarch` leads on FP32 PESQ; `monarch_full` leads on int8 PESQ
(slightly above its own FP32, within noise — quantization is essentially
loss-free). `monarch_8` runs at RTF 0.025 — over 18× faster than the
dense baseline at the same int8 PESQ.

## STM32N6 on-board deployment

All three speech-enhancement models in this repo now run on the
**STM32N6570-DK** (STM32N657 — Cortex-M55 @ 800 MHz + Neural-ART NPU @
1 GHz), compiled with ST Edge AI Core 4.0.1, fully scripted (no
STM32CubeIDE). On-target latency is per streaming frame (hop 256 @
16 kHz = 16 ms budget); RTF < 1 is real-time.

| model (int8)            | int8 PESQ | weights | on-chip? | latency/frame |     RTF | on-target cos |
| ----------------------- | --------: | ------: | :------: | ------------: | ------: | ------------: |
| **`monarch_full`** (sparse)| **2.848** | 0.72 MB |  ✅    |  **2.13 ms**  |**0.13** |      0.99979  |
| `monarch_8` (sparse)    |     2.826 | 0.37 MB |    ✅    |     2.89 ms   |   0.18  |      0.99994  |
| ConvFSENet (conv)       |     2.911 | 1.40 MB |    ✅    |     4.40 ms   |   0.275 |      0.990    |
| `baseline` (dense GRU)  |     2.833 | 2.70 MB |    ✗     |    22.94 ms   |   1.43  |      0.9946   |

**Structured sparsity is what lets the recurrent model hit real-time on
this NPU.** The Neural-ART runs fastest when weights live in on-chip
npuRAM. The dense GRU baseline's 2.70 MB int8 weights overflow it, so it
streams them from external octoFlash every frame and lands at RTF 1.43 —
*not* real-time. The sparse monarch variants are 4–8× smaller, fit
entirely on-chip, and dominate: **`monarch_full` is the best of all four**
— fastest (2.13 ms, RTF 0.13 — **11× faster than dense, 2× faster than
ConvFSENet**) *and* highest int8 PESQ (2.848). `monarch_8` (more, smaller
blocks: nblocks 8 vs 4) is a touch slower at 2.89 ms; fewer larger blocks
map more efficiently to the NPU (88 epochs vs 134).

Two deployment subtleties, both detailed in
[`deploy/stm32n6/NSNET2_DEPLOYMENT_NOTES.md`](deploy/stm32n6/NSNET2_DEPLOYMENT_NOTES.md):

* **Dense** doesn't compile as-exported — onnxruntime fuses the GRU
  `MatMul`+`Add` into a `Gemm` with an *activation* `C`, which the
  Neural-ART int8 lowering can't index. Re-quantizing with
  `quant_pre_process(skip_optimization=True)` keeps them separate; the
  result is numerically identical to the published int8.
* **The monarch variants** don't compile as-exported either — the monarch
  block-matmul (`Einsum` + `Pad` + block reshapes) defeats the compiler's
  shape engine, and a 4-D grouped-conv re-export compiles in FP32 but
  hits int8 HW-lowering batch-dim asserts. The fix is to re-express the
  blocks in the *rank-2 `MatMul`* op vocabulary that the dense baseline
  already maps to HW: per-block `Slice` + `MatMul` + `Concat`, flat
  states, and the gate rewritten `(1-z)·n + z·h = n + z·(h-n)`.
  `deploy/stm32n6/host/export_monarch_npu.py` does this from the trained
  checkpoint (parity ~5e-7, any fully-monarch config — dims read from the
  checkpoint), then int8-quantizes with the same recipe. Each deployed
  artifact matches its stock int8 to mask cosine 0.999, so it carries the
  published PESQ.

Caveats: on-target cosine is vs the FP32 ONNX reference over a 10-sample
`stedgeai validate` run; the validation firmware is a volatile RAM image;
and these are single-run latencies. `wide_monarch` also holds int8 PESQ
but at 2.36 M / 9.5 MB int8 it would not fit on-chip; the export script
handles any fully-monarch config (`monarch_fc`'s dense GRU is rejected).

### Int8 quantization findings

All checkpoints export to streaming-shape FP32 ONNX and quantize to int8:

```bash
./run_quantize_sweep.sh                      # FP32 + int8 ONNX per cp dir
./run_eval_sweep.sh                          # PESQ on full test split
MAX_UTTERANCES=100 ./run_eval_sweep.sh       # quick directional read
```

Three findings worth flagging:

* **Monarch variants quantize loss-free** (|Δ| ≤ 0.021 across all four).
  A single `Einsum` per FC layer plus per-channel int8 is genuinely
  friendly to QDQ calibration.
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
on `monarch_8` (full 824-utt test PESQ; FP32 = 2.832) says **keep the
un-normalized MinMax calibration**:

| calibration                          | int8 PESQ |
| ------------------------------------ | --------: |
| un-normalized + MinMax (**default**) | **2.826** |
| RMS-normalized + Percentile          |     2.795 |
| RMS-normalized + Entropy             |     2.793 |
| RMS-normalized + MinMax              |     2.768 |

Matching calibration to the (narrower) RMS-normalized deployment range
*tightens* the quantization range and clips more activation outliers, costing
~0.03–0.06 PESQ on the wider monarch variants; the wider un-normalized range
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
| `monarch_fc`        |     2.841 |  **2.832** | −0.009   | OK (PTQ enough)|
| `baseline`          |     2.867 |      2.813 | −0.053   | borderline     |
| `wide_monarch`      |     2.905 |      2.742 | −0.163   | needs QAT      |
| `monarch_full`      |     2.853 |      2.558 | −0.295   | needs QAT      |
| `monarch_8`         |     2.864 |      2.502 | −0.362   | needs QAT      |
| `butterfly_2blocks` |     2.822 |      2.163 | −0.659   | needs QAT      |
| `butterfly_full`    |     2.811 |      2.034 | −0.776   | needs QAT      |
| `butterfly_ortho`   |     2.795 |      1.832 | −0.964   | needs QAT      |
| `butterfly_fc`      |     2.848 |      1.850 | −0.998   | needs QAT      |

Only `monarch_fc` (the dense-GRU + monarch-FC variant) and the dense
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
| `monarch_full`      |     2.853 |     2.558 |      2.724 |  **2.783** | −0.295   |   −0.070   |   76%     |
| `wide_monarch`      |     2.905 |     2.742 |      2.784 |  **2.786** | −0.163   |   −0.119   |   27%     |
| `monarch_8`         |     2.864 |     2.502 |      2.682 |  **2.744** | −0.362   |   −0.120   |   67%     |
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
* `monarch_full` lands closest to fp32 of any QAT config — only
  −0.070 below the trained fp32 baseline.
* `wide_monarch` was already at its floor at 10 epochs; the other six
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
"linear": {"kind": "linear" | "butterfly" | "monarch", ...kwargs},
"gru":    {"kind": "gru"    | "butterfly" | "monarch", ...kwargs}
```

Per-backend kwargs (all optional): butterfly takes `nblocks` (1+), `init`
(`randn` / `ortho`), `x_init`, `h_init`; monarch takes `nblocks` (≥ 2).
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
RUN  = "wide_monarch"  # or any run name from the table above

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
