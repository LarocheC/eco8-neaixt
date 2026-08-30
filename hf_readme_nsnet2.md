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

#### Block-count sweep, both families (`nblocks` 5 → 40)

One knob varies. Every run below is the `*_8` architecture — hidden 400, fc 600,
both FCs and both GRU projections structured — with **only `nblocks` changed**,
so parameters move and nothing else does. Dense baseline for reference:
**2.845 FP32 / 2.834 int8 at 2.78 M**.

| nblocks | blockdiag params | FP32 | int8 | Δint8 | monarch params | FP32 | int8 | Δint8 |
| ------: | ---------------: | ---: | ---: | ----: | -------------: | ---: | ---: | ----: |
|       5 |          0.563 M | 2.826 | 2.793 | +0.033 |        0.880 M | 2.852 | 2.858 | −0.007 |
|       8 |          0.355 M | 2.832 | 2.825 | +0.007 |        0.553 M | **2.861** | 2.856 | +0.005 |
|      10 |          0.285 M | 2.772 | 2.744 | +0.028 |        0.443 M | 2.849 | 2.842 | +0.007 |
|      20 |          0.146 M | 2.719 | 2.627 | +0.092 |        0.225 M | 2.849 | 2.854 | −0.005 |
|      40 |          0.077 M | 2.608 | 2.455 | +0.153 |        0.117 M | 2.837 | 2.837 |  0.000 |

**Block-diagonal collapses as blocks narrow; Monarch does not.** Over nblocks
5→40 blockdiag loses 0.218 PESQ in FP32, and its int8 penalty grows from 0.033 to
0.153 (0.338 total in int8 terms). Monarch moves 0.015 in FP32 and stays
int8-loss-free throughout — exactly 0.000 at nblocks 40.

The separating variable is connectivity, not capacity. A block-diagonal factor
never mixes across blocks, so raising `nblocks` splits the network into narrower
non-communicating bands; Monarch's permutation restores full cross-channel reach
in one step. Two checks:

- **At matched parameters**: `blockdiag_5` (0.563 M) 2.826 vs `monarch_8`
  (0.553 M) 2.861 — +0.035 for Monarch at equal size.
- **Monarch wins while smaller**: `monarch_40` (0.117 M) beats `blockdiag_20`
  (0.146 M) by 0.130 FP32 and 0.227 int8.

**`monarch_40` reaches dense parity with 24× fewer parameters** (2.837 vs 2.845,
inside metric noise) and is loss-free in int8, where the dense baseline itself
gives up 0.012. It is the model to take unless you have a reason not to.

#### Param-matched dense controls

Structured-vs-structured comparisons cannot tell "Monarch's mixing is what
matters" apart from "any model of this width would do". The control is a plain
narrower **dense** NSNet2 at the same parameter count (`dense_h*`, hidden and fc
scaled together at the original 1.5 ratio, each within ~1% of its pair):

| params | block-diagonal | dense | Monarch | dense−monarch |
| -----: | -------------: | ----: | ------: | ------------: |
| 0.88 M |              — | 2.783 | **2.852** | −0.069 |
| 0.55 M |          2.826 | 2.840 | **2.861** | −0.021 |
| 0.44 M |              — | 2.815 | **2.849** | −0.034 |
| 0.23 M |          2.719 | 2.784 | **2.849** | −0.065 |
| 0.12 M |              — | 2.751 | **2.837** | −0.086 |
| 0.08 M |          2.608 | 2.749 |         — |      — |

**Monarch beats dense at all five matched sizes** (mean −0.055), and the gap
widens as models shrink. The small-end ordering is **Monarch > dense >
block-diagonal** — so `blockdiag_40` does not fail because 0.077 M is too small
for the task; a dense model of that exact size scores 2.749 against its 2.608.

Dense quantizes cleanly at every width (|Δ| ≤ 0.017), so int8 robustness does not
separate dense from Monarch; that failure is specific to narrow block-diagonal.

**Confirmed with repeat seeds** (three each: 1234, 2345, 3456) on the two
pairings that carry the argument:

| pairing | Monarch (n=3) | dense (n=3) | gap | overlap? |
| ------- | ------------- | ----------- | --: | -------- |
| 0.12 M | **2.832** ± 0.009 | **2.753** ± 0.002 | **+0.079** | none (+0.067) |
| 0.55 M | **2.858** ± 0.003 | **2.833** ± 0.008 | **+0.025** | none (+0.016) |

In both, the worst Monarch run beats the best dense run, and the advantage grows
as models shrink. Seed noise is sd 0.002–0.009, an order of magnitude below the
effect. Note that dense's scatter *across sizes* (0.089 band, the 0.55 M arm
beating the 0.88 M one) is therefore real rather than noise — dense NSNet2 trains
inconsistently at these widths in a way Monarch does not.

#### Latency: the trade-off runs the other way

All seven small-model arms below were re-timed **back-to-back in one session on
an idle box**, so these RTFs are directly comparable (the RTF figures elsewhere
in this file were collected across different days and are not).

| params | model | int8 PESQ | int8 RTF |
| -----: | ----- | --------: | -------: |
| 0.12 M | `monarch_40` | **2.837** |    0.013 |
| 0.12 M | `dense_h68`  |     2.754 | **0.007** |
| 0.23 M | `monarch_20` | **2.854** |    0.013 |
| 0.23 M | `dense_h100` |     2.794 | **0.011** |
| 0.08 M | `blockdiag_40` |   2.455 |    0.008 |
| 0.08 M | `dense_h52`  |     2.745 | **0.005** |

**At matched parameters the dense model is faster at every small size** — 1.9×
at 0.12 M, 1.2× at 0.23 M. Monarch's Einsum lowering carries a fixed overhead
that stops paying for itself as the models shrink; parameter count and latency
are not the same axis.

So the deployment choice at 0.12 M is a real trade, not a free win: **+0.083 PESQ
(Monarch) against ~1.9× lower CPU latency and no torch-structured / gru-qat
dependency (dense)**. On embedded targets the case is worse still — Monarch
measured ~3× slower than dense on the RT595, and raising the block count made the
STM32N6 NPU slower. Monarch wins the science; dense may well win the product.

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
  nearly free. For deployment, take the smallest (`blockdiag_8` / `monarch_20`).
  The block-count sweep extends this: 4× parameters across `nblocks` 5→20 moves
  PESQ by 0.012, within run-to-run noise.
- **Genuine Monarch beats block-diagonal, but marginally** (+0.011…+0.038 FP32 at
  matched `nblocks`) and it costs parameters — its second factor makes it larger.
  Consistent with the saturation above.
- **Monarch quantizes loss-free at every block count tested** (|Δ| ≤ 0.012 over
  nblocks 4–40). **Block-diagonal only up to `nblocks` 8** (|Δ| ≤ 0.018) — at 20
  and 40 the penalty is 0.092 and 0.153. The older unqualified "block-diagonal
  quantizes loss-free" claim was tested only on wide blocks,
  *with the weights genuinely quantized*.
- **Butterfly with randn init degrades catastrophically under int8** (Δ up to
  0.644). Use `init=ortho`: `butterfly_ortho` loses 0.203 to int8 vs 0.644 for
  `butterfly_full` — same architecture, same data, only the init differs.

## Layout

One subdirectory per run: the generator (`g_best`), the streaming FP32 ONNX, the
static int8 ONNX, and the exact `config.json` it was trained with.

```
baseline/          blockdiag_5/     monarch_5/       butterfly_fc/
blockdiag_fc/      blockdiag_8/     monarch_8/       butterfly_full/
blockdiag_full/    blockdiag_10/    monarch_10/      butterfly_ortho/
wide_blockdiag/    blockdiag_20/    monarch_20/      butterfly_2blocks/
wide_monarch/      blockdiag_40/    monarch_40/
monarch_fc/

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
