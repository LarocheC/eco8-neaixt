---
license: mit
language: en
library_name: pytorch
tags:
  - speech-enhancement
  - speech-denoising
  - nsnet2
  - butterfly
  - block-diagonal
  - monarch
  - structured-matrices
  - onnx
  - quantization
  - int8
datasets:
  - JacobLinCool/VoiceBank-DEMAND-16k
metrics:
  - pesq
---

# sparse-nsnet2 — checkpoints

Best-PESQ checkpoints from the
[eco8-neaixt](https://github.com/LarocheC/eco8-neaixt) compression sweep:
NSNet2 speech enhancement with the FC and GRU layers swappable between
dense, [Butterfly](https://arxiv.org/abs/1903.05895), and **block-diagonal**
factorizations.

> ### ⚠️ Naming correction (structure relabel)
> The runs previously published as `monarch_*` (`monarch_8`, `monarch_full`,
> `monarch_fc`, `wide_monarch`) are **single block-diagonal factor** models —
> one block-diagonal matrix per projection, with **zero cross-block mixing**.
> That is *not* a Monarch matrix. A genuine
> [Monarch](https://arxiv.org/abs/2204.00595) is a **two-factor** construction
> (block-diagonal × permutation × block-diagonal) with full cross-channel
> mixing. These checkpoints have therefore been **renamed `monarch_* →
> blockdiag_*`** to reflect what they actually are:
>
> | old name       | new name         |
> | -------------- | ---------------- |
> | `monarch_8`    | `blockdiag_8`    |
> | `monarch_full` | `blockdiag_full` |
> | `monarch_fc`   | `blockdiag_fc`   |
> | `wide_monarch` | `wide_blockdiag` |
>
> Each renamed run's `config.json` now uses `"kind": "blockdiag"` (and
> `"triton_blockdiag"` where applicable). Genuine two-factor Monarch runs
> (`kind="monarch"`, torch-structured ≥ 1.3.0) are a separate, upcoming set of
> `monarch_*` checkpoints — see the experiment plan in the source repo.

Trained on
[VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k)
for 200 epochs at batch 256, n\_fft 512, on a single GTX 1080 Ti.

Each variant ships in three formats:
- `g_best` — PyTorch checkpoint (training-time weights)
- `g_best_fp32.onnx` — streaming-shape FP32 ONNX (frame-by-frame, opset 17)
- `g_best.onnx` — static int8 ONNX (QDQ format, per-channel weight quant,
  MinMax calibration on 200 VBD-train utterances)

## Results

PESQ measured on the full VBD test split (824 utterances). RTF (real-time
factor) is for the int8 ONNX session under onnxruntime CPU; lower is faster.

| run                 | params  | FP32 PESQ |  int8 PESQ |  Δ (FP32→int8) | int8 RTF |
| ------------------- | ------: | --------: | ---------: | -------------: | -------: |
| `wide_blockdiag`    |  2.36 M | **2.864** |      2.842 |        +0.021  |    0.166 |
| `baseline`          |  2.78 M |     2.845 |      2.833 |        +0.012  |    0.452 |
| `blockdiag_8`       |  0.36 M |     2.832 |      2.826 |        +0.006  | **0.025** |
| `blockdiag_full`    |  0.70 M |     2.827 |  **2.848** |        −0.021  |    0.039 |
| `blockdiag_fc`      |  2.14 M |     2.805 |      2.789 |        +0.016  |    0.448 |
| `butterfly_2blocks` |  0.36 M |     2.805 |      2.202 |        +0.602  |    0.441 |
| `butterfly_fc`      |  1.99 M |     2.799 |      2.494 |        +0.306  |    0.522 |
| `butterfly_ortho`   |  0.19 M |     2.780 |      2.577 |        +0.203  |    0.232 |
| `butterfly_full`    |  0.19 M |     2.772 |      2.128 |        +0.644  |    0.230 |

## Key findings

- **Block-diagonal variants are essentially loss-free under int8** (|Δ| ≤ 0.021
  across the board). A single `Einsum` per FC plus per-channel weight
  quantization is genuinely friendly to int8 calibration.
- **`wide_blockdiag` is the best deployment target** for quality: highest FP32
  PESQ (2.864) with near-zero int8 loss (2.842). For speed-constrained
  deployment, **`blockdiag_full` and `blockdiag_8`** trade ~0.02 PESQ for an
  RTF of 0.04 / 0.025 — over 10× faster than the dense baseline.
- **Butterfly with random init degrades catastrophically under int8**
  (Δ up to 0.64 PESQ on `butterfly_full`). Int8 deployment with butterfly
  factorizations should use `init=ortho`: `butterfly_ortho` loses 0.20
  PESQ to int8 versus 0.64 for `butterfly_full` (same architecture, same
  training data, only the init differs).
- **Longer training makes randn-init butterfly *worse* on int8.** The same
  `butterfly_full` config saw its int8 gap grow from 0.36 PESQ at 50
  epochs → 0.64 PESQ at 200 epochs as twiddle factors drifted further
  from orthogonality. Ortho-init butterfly does not show this regression.

## Layout

Each subdirectory is one run, containing the saved generator (`g_best`),
the streaming FP32 ONNX, the static int8 ONNX, and the exact `config.json`
the run was trained with.

```
baseline/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
blockdiag_fc/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
butterfly_fc/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
blockdiag_full/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
butterfly_full/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
blockdiag_8/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
butterfly_ortho/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
butterfly_2blocks/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
wide_blockdiag/{g_best,g_best_fp32.onnx,g_best.onnx,config.json}
```

## Loading

### PyTorch checkpoint

Clone the repo first (model classes live there):

```bash
git clone https://github.com/LarocheC/eco8-neaixt
cd eco8-neaixt
uv sync
```

Then:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from nsnet2.model import NSNet2

REPO = "claroche1/sparse-nsnet2-checkpoints"
RUN  = "wide_blockdiag"  # or any name from the table

cfg  = json.load(open(hf_hub_download(REPO, f"{RUN}/config.json")))
ckpt = torch.load(hf_hub_download(REPO, f"{RUN}/g_best"),
                  map_location="cuda", weights_only=False)

model = NSNet2(AttrDict(cfg)).cuda().eval()
model.load_state_dict(ckpt["generator"])
```

### ONNX (FP32 or int8)

The ONNX models are streaming-shape: a single frame `(B, n_freq)` plus the
GRU state `(num_layers, B, hidden)` per session call, threaded across
frames. The end-to-end pipeline (RMS-norm → STFT → frame loop → iSTFT)
is in `inference_onnx.py` in the source repo.

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "claroche1/sparse-nsnet2-checkpoints"
RUN  = "wide_blockdiag"

# FP32:
fp32_path = hf_hub_download(REPO, f"{RUN}/g_best_fp32.onnx")
fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])

# int8 (deployment):
int8_path = hf_hub_download(REPO, f"{RUN}/g_best.onnx")
int8_sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
```

End-to-end inference example with PESQ measurement is in `inference_onnx.py`
in the source repo.

## Citations

```bibtex
@inproceedings{braun2021nsnet2,
    title={Towards efficient models for real-time deep noise suppression},
    author={Braun, Sebastian and Tashev, Ivan},
    booktitle={ICASSP},
    year={2021}
}

@inproceedings{dao2019butterfly,
    title={Learning fast algorithms for linear transforms using butterfly factorizations},
    author={Dao, Tri and Gu, Albert and Eichhorn, Matthew and Rudra, Atri and R{\'e}, Christopher},
    booktitle={ICML},
    year={2019}
}

@inproceedings{dao2022monarch,
    title={Monarch: Expressive structured matrices for efficient and accurate training},
    author={Dao, Tri and Chen, Beidi and Sohoni, Nimit S and Desai, Arjun and Poli, Michael and Grogan, Jessica and Liu, Alexander and Rao, Aniruddh and Rudra, Atri and R{\'e}, Christopher},
    booktitle={ICML},
    year={2022}
}
```
