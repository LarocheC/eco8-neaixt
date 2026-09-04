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
NSNet2 speech enhancement with the FC and GRU layers swappable between dense,
[Butterfly](https://arxiv.org/abs/1903.05895), **block-diagonal**, and genuine
two-factor [Monarch](https://arxiv.org/abs/2204.00595) factorizations.

Trained on
[VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k),
n\_fft 512, batch 256. PESQ on the full 824-utterance test split.

## ⚠️ Two corrections you should read before quoting these numbers

**1. Naming: the old `monarch_*` runs were NOT Monarch.** They are a *single
block-diagonal factor* (one block-diagonal matrix per projection, **zero
cross-block mixing**). A genuine [Monarch](https://arxiv.org/abs/2204.00595) is a
**two-factor** construction (block-diagonal × permutation × block-diagonal) with
full cross-channel mixing. The old runs have therefore been **renamed to
`blockdiag_*`**, and the `monarch_*` names now hold *genuinely* Monarch models:

| old name (was mislabeled) | now                | `monarch_*` today        |
| ------------------------- | ------------------ | ------------------------ |
| `monarch_8`               | → `blockdiag_8`    | genuine 2-factor Monarch |
| `monarch_full`            | → `blockdiag_full` | genuine 2-factor Monarch |
| `monarch_fc`              | → `blockdiag_fc`   | genuine 2-factor Monarch |
| `wide_monarch`            | → `wide_blockdiag` | genuine 2-factor Monarch |

If you previously pinned `monarch_8`, you now get a **different (genuinely
Monarch) model** — the block-diagonal one you had is at `blockdiag_8`.

**2. int8: the previously published int8 ONNX never quantized the structured
weights.** The structure-preserving export lowers each block-diagonal / Monarch
matmul to an `Einsum`, and onnxruntime ships no QDQ handler for `Einsum` — so
`quantize_static` skipped those nodes entirely. Only activations and the residual
dense MatMuls were int8; the **dominant weights stayed FP32**. Every int8 file
here has been **re-quantized with the structured weights genuinely int8** (fixed
in `nsnet2/qdq_einsum_quantizer.py` upstream). The "int8 is loss-free" property
does survive the fix — but it had never actually been tested before it.

## Results (int8 = genuinely quantized weights)

### Genuine two-factor Monarch

| run            | params | FP32 PESQ | int8 PESQ | Δ (FP32→int8) |
| -------------- | -----: | --------: | --------: | ------------: |
| `wide_monarch` | 3.64 M | **2.881** | **2.884** |        −0.003 |
| `monarch_8`    | 0.55 M |     2.861 |     2.856 |        +0.005 |
| `monarch_fc`   | 2.38 M |     2.843 |     2.831 |        +0.012 |
| `monarch_full` | 1.10 M |     2.838 |     2.846 |        −0.009 |

### Block-diagonal, dense, butterfly

| run                 | params | FP32 PESQ | int8 PESQ | Δ (FP32→int8) |
| ------------------- | -----: | --------: | --------: | ------------: |
| `wide_blockdiag`    | 2.36 M |     2.864 |     2.847 |        +0.016 |
| `baseline` (dense)  | 2.78 M |     2.845 |     2.833 |        +0.012 |
| `blockdiag_8`       | 0.36 M |     2.832 |     2.825 |        +0.007 |
| `blockdiag_full`    | 0.70 M |     2.827 |     2.843 |        −0.016 |
| `blockdiag_fc`      | 2.14 M |     2.805 |     2.787 |        +0.018 |
| `butterfly_2blocks` | 0.36 M |     2.805 |     2.202 |        +0.602 |
| `butterfly_fc`      | 1.99 M |     2.799 |     2.494 |        +0.306 |
| `butterfly_ortho`   | 0.19 M |     2.780 |     2.577 |        +0.203 |
| `butterfly_full`    | 0.19 M |     2.772 |     2.128 |        +0.644 |

## Key findings

- **Quality saturates — this model class is architecture-bound, not
  capacity-bound.** Across three structure families and ~10× parameters, every
  configuration lands in a **2.83–2.88** band, non-monotonically: the 0.55 M
  `monarch_8` (2.861) beats both the 1.10 M `monarch_full` and the 2.38 M
  `monarch_fc`. Going 7× from `monarch_8` to `wide_monarch` buys +0.020 PESQ.
  NSNet2 predicts a magnitude mask and reuses the noisy phase, which caps PESQ
  regardless of how expressive the mask predictor is. **The dense model was
  already over-parameterized** — which is exactly why aggressive structuring is
  nearly free. For deployment, take the smallest (`blockdiag_8` / `monarch_8`).
- **Genuine Monarch beats block-diagonal, but marginally** (+0.011…+0.038 FP32 at
  matched `nblocks`) and it costs parameters — its second factor makes it larger.
  Consistent with the saturation above.
- **Block-diagonal and Monarch quantize loss-free** (|Δ| ≤ 0.018 and ≤ 0.012),
  *with the weights genuinely quantized*.
- **Butterfly with randn init degrades catastrophically under int8** (Δ up to
  0.644). Use `init=ortho`: `butterfly_ortho` loses 0.203 to int8 vs 0.644 for
  `butterfly_full` — same architecture, same data, only the init differs.

## Layout

One subdirectory per run: the generator (`g_best`), the streaming FP32 ONNX, the
static int8 ONNX, and the exact `config.json` it was trained with.

```
baseline/          blockdiag_8/     monarch_8/       butterfly_fc/
blockdiag_fc/      blockdiag_full/  monarch_fc/      butterfly_full/
wide_blockdiag/    monarch_full/    wide_monarch/    butterfly_ortho/
                                                     butterfly_2blocks/

  each: {g_best, g_best_fp32.onnx, g_best.onnx, config.json}
```

## Loading

```bash
git clone https://github.com/LarocheC/eco8-neaixt && cd eco8-neaixt && uv sync
```

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from nsnet2.model import NSNet2

REPO = "claroche1/sparse-nsnet2-checkpoints"
RUN  = "monarch_8"   # or any run from the tables

cfg  = json.load(open(hf_hub_download(REPO, f"{RUN}/config.json")))
ckpt = torch.load(hf_hub_download(REPO, f"{RUN}/g_best"),
                  map_location="cuda", weights_only=False)

model = NSNet2(AttrDict(cfg)).cuda().eval()
model.load_state_dict(ckpt["generator"])
```

> **Note for the `monarch_*` runs:** their GRU was trained through gru-qat's fused
> Monarch Triton kernel (`"gru": {"kind": "triton_monarch"}`), so their
> `state_dict` uses gru-qat module names. Loading them requires
> **`gru-qat >= 0.5.0`** and **`torch-structured >= 1.3.0`** (both pulled in by
> `uv sync`). The `blockdiag_*` / `butterfly_*` / `baseline` runs use the native
> path and have no such requirement.

### ONNX (FP32 or int8)

Streaming-shape: one frame `(B, n_freq)` plus GRU state `(num_layers, B, hidden)`
per session call, threaded across frames. End-to-end pipeline in
`nsnet2/inference_onnx.py`.

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download
REPO, RUN = "claroche1/sparse-nsnet2-checkpoints", "monarch_8"
sess = ort.InferenceSession(hf_hub_download(REPO, f"{RUN}/g_best.onnx"),
                            providers=["CPUExecutionProvider"])   # int8
```

## Citations

```bibtex
@inproceedings{braun2021nsnet2,
    title={Towards efficient models for real-time deep noise suppression},
    author={Braun, Sebastian and Tashev, Ivan},
    booktitle={ICASSP}, year={2021}
}
@inproceedings{dao2019butterfly,
    title={Learning fast algorithms for linear transforms using butterfly factorizations},
    author={Dao, Tri and Gu, Albert and Eichhorn, Matthew and Rudra, Atri and R{\'e}, Christopher},
    booktitle={ICML}, year={2019}
}
@inproceedings{dao2022monarch,
    title={Monarch: Expressive structured matrices for efficient and accurate training},
    author={Dao, Tri and Chen, Beidi and Sohoni, Nimit S and Desai, Arjun and Poli, Michael and Grogan, Jessica and Liu, Alexander and Rao, Aniruddh and Rudra, Atri and R{\'e}, Christopher},
    booktitle={ICML}, year={2022}
}
```
