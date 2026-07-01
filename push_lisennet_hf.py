"""Upload a LiSenNet checkpoint + ONNX artifacts to a HuggingFace model repo.

Two variants, auto-detected from the run's config.json ``bottleneck``:

  * **rnn** (the faithful GRU LiSenNet, PESQ 3.006) -> default repo
    ``claroche1/LiSenNet``. Files: PyTorch g_best, whole-utterance FP32 + static
    int8 ONNX, config. This is the quality reference; it does NOT deploy to the
    STM32N6 Neural-ART NPU (GRU + 2-axis LayerNorm are compiler blockers).
  * **conv** (the NPU-deployable dual-path-conv variant, PESQ 2.970) -> default
    repo ``claroche1/LiSenNet-npu``. Adds the frame-by-frame **streaming** ONNX
    graphs with explicit state I/O — the actual stedgeai / Neural-ART target.

Idempotent: re-runs just create new commits. The checkpoint and the exported
ONNX graphs both come from the training run dir; ``lisennet.export_onnx`` /
``lisennet.quant_onnx`` / ``lisennet.eval_deploy`` write the ONNX next to g_best.

Usage:
    python push_lisennet_hf.py --checkpoint_dir cp_lisennet                 # GRU -> claroche1/LiSenNet
    python push_lisennet_hf.py --checkpoint_dir cp_lisennet_conv_wide       # conv -> claroche1/LiSenNet-npu
    python push_lisennet_hf.py --checkpoint_dir cp_lisennet_conv_wide --dry-run

Requires `huggingface_hub` (already in the venv) and a logged-in HF account
(`huggingface-cli login`, or set HF_TOKEN in the environment).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SRC_REPO = "https://github.com/LarocheC/eco8-neaixt"

# repo filename -> (source dir kind, local filename); "ckpt"=--checkpoint_dir, "onnx"=--onnx_dir
FILE_MAP_RNN = {
    "config.json": ("ckpt", "config.json"),
    "g_best": ("ckpt", "g_best"),
    "g_best_fp32.onnx": ("onnx", "g_best_fp32.onnx"),
    "g_best_int8_static.onnx": ("onnx", "g_best_int8_static.onnx"),
}
FILE_MAP_CONV = {
    **FILE_MAP_RNN,
    "g_best_streaming_fp32.onnx": ("onnx", "g_best_streaming_fp32.onnx"),
    "g_best_streaming_int8_static.onnx": ("onnx", "g_best_streaming_int8_static.onnx"),
}

DEFAULT_REPO = {"rnn": "claroche1/LiSenNet", "conv": "claroche1/LiSenNet-npu"}

CARD_RNN = """\
---
license: mit
library_name: pytorch
tags:
- speech-enhancement
- audio
- denoising
- onnx
- causal
- streaming
- real-time
datasets:
- JacobLinCool/VoiceBank-DEMAND-16k
---

# LiSenNet

An ultra-compact (~37 K-param), causal, real-time speech enhancer trained on
VoiceBank-DEMAND-16k. Sub-band U-Net with a dual-path-recurrent bottleneck and a
magnitude-only mask; phase is recovered with a 2-iteration Griffin-Lim (offline)
or reused from the noisy phase (real-time).

Faithful port of **Yan, Zhou, Chen & Lu, _LiSenNet: Lightweight Sub-band and
Dual-Path Modeling for Real-Time Speech Enhancement_,
[arXiv:2409.13285](https://arxiv.org/abs/2409.13285)** (authors' MIT reference:
[hyyan2k/LiSenNet](https://github.com/hyyan2k/LiSenNet)).

> **Deploying to an edge NPU?** This GRU model is the *quality reference*; its
> GRU + 2-axis LayerNorm do **not** map to the STM32N6 Neural-ART NPU. Use the
> NPU-friendly conv variant at
> [claroche1/LiSenNet-npu](https://huggingface.co/claroche1/LiSenNet-npu).

Code + full write-up: [{src}]({src}) — see
[RESULTS_LISENNET.md]({src}/blob/main/RESULTS_LISENNET.md).

## Headline numbers

PESQ is wideband, on the full 824-utterance VoiceBank-DEMAND test split.

| metric                                          |     value |
| ----------------------------------------------- | --------: |
| params                                          | 36,783    |
| FP32 PESQ (torch / ONNX, Griffin-Lim)           | **3.006** |
| static-int8 PESQ (Griffin-Lim)                  |     2.920 |
| **real-time PESQ (static int8 + noisy phase)**  | **2.930** |

## Files

| file                      | what it is |
| ------------------------- | --- |
| `g_best`                  | PyTorch checkpoint (`{"generator": state_dict}`) |
| `g_best_fp32.onnx`        | FP32 ONNX of the mask sub-network: `feat (B,3,T,F)` -> `est_mag (B,T,F)` |
| `g_best_int8_static.onnx` | Static full int8 (QDQ, per-tensor) |
| `config.json`             | Training + architecture config |

## Loading

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from lisennet.model import build_lisennet

REPO = "{repo_id}"
cfg  = json.load(open(hf_hub_download(REPO, "config.json")))
ckpt = torch.load(hf_hub_download(REPO, "g_best"), map_location="cpu", weights_only=True)
model = build_lisennet(AttrDict(cfg)).eval()
model.load_state_dict(ckpt["generator"])   # model(noisy_wav)["est"]
```

## License

MIT. See the [source repository]({src}) for training code and full attribution.
"""

CARD_CONV = """\
---
license: mit
library_name: pytorch
tags:
- speech-enhancement
- audio
- denoising
- onnx
- causal
- streaming
- real-time
- edge-ai
- stm32
datasets:
- JacobLinCool/VoiceBank-DEMAND-16k
---

# LiSenNet-npu — NPU-deployable conv variant

The **edge-NPU-deployable** variant of LiSenNet. It keeps LiSenNet's sub-band
U-Net + magnitude-only mask, but replaces the dual-path **GRU** bottleneck with a
dual-path **conv** bottleneck so the whole model compiles to the **STM32N6
Neural-ART** NPU via stedgeai. Both of the original blockers are gone:

- no recurrence (GRU state does not map to the NPU) — the inter-time path is a
  stack of causal dilated depthwise convs with FIFO state;
- no 2-axis `LayerNormalization` — norms are already-primitive ops.

Based on **Yan, Zhou, Chen & Lu, _LiSenNet_,
[arXiv:2409.13285](https://arxiv.org/abs/2409.13285)**
([hyyan2k/LiSenNet](https://github.com/hyyan2k/LiSenNet), MIT). The faithful GRU
model (quality reference, PESQ 3.006) is at
[claroche1/LiSenNet](https://huggingface.co/claroche1/LiSenNet).

Code + full write-up: [{src}]({src}) — see
[RESULTS_LISENNET.md]({src}/blob/main/RESULTS_LISENNET.md).

## Headline numbers (824-utt VoiceBank-DEMAND test, wideband PESQ)

| metric                                          |     value |
| ----------------------------------------------- | --------: |
| params                                          | 41,063    |
| FP32 PESQ (torch / ONNX, Griffin-Lim)           | **2.970** |
| static-int8 PESQ (Griffin-Lim)                  |     2.847 |
| **real-time PESQ (static int8 + noisy phase)**  | **2.855** |

~0.04 below the GRU model at FP32 — the cost of dropping recurrence to fit the
NPU. FP32 ONNX export is loss-free.

## Files

| file                                | what it is |
| ----------------------------------- | --- |
| **`g_best_streaming_fp32.onnx`**    | **the deploy target** — frame-by-frame graph with explicit state I/O: `feat (B,3,1,F)` + N `state_i_in` -> `est_mag (B,1,F)` + N `state_i_out`. Static state shapes, batch dynamic. No GRU / LayerNormalization. |
| `g_best_streaming_int8_static.onnx` | static-int8 of the streaming graph (characterization; stedgeai does the on-board int8) |
| `g_best_fp32.onnx`                  | whole-utterance FP32 graph (`feat (B,3,T,F)` -> `est_mag (B,T,F)`) for offline eval |
| `g_best_int8_static.onnx`           | whole-utterance static int8 |
| `g_best`                            | PyTorch checkpoint (`{"generator": state_dict}`) |
| `config.json`                       | architecture config (`bottleneck: conv`, dilations, kernels) |

The ONNX graphs are the mask sub-network only. STFT, the 3-channel feature build
(`[compressed_mag, group_delay/pi, ifd/pi]`), the magnitude combination and the
phase recovery (Griffin-Lim offline / noisy phase for real-time) stay host-side.

## Running the streaming graph (frame-by-frame)

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

sess = ort.InferenceSession(
    hf_hub_download("{repo_id}", "g_best_streaming_fp32.onnx"),
    providers=["CPUExecutionProvider"],
)
state_in = [i for i in sess.get_inputs() if i.name != "feat"]      # the FIFO states
out_names = [o.name for o in sess.get_outputs()]                   # est_mag + state_*_out
zeros = lambda s: np.zeros([d if isinstance(d, int) else 1 for d in s], np.float32)
states = {i.name: zeros(i.shape) for i in state_in}               # start-of-stream = zeros

# per frame: feat_t is (1, 3, 1, 257); feed states, get est_mag + new states back
def step(feat_t):
    res = sess.run(out_names, {"feat": feat_t, **states})
    for i, v in zip(state_in, res[1:]):
        states[i.name] = v
    return res[0]                                                  # est_mag (1, 1, 257)
```

## License

MIT. See the [source repository]({src}) for training code and full attribution.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--checkpoint_dir", default="cp_lisennet",
                   help="Run dir with config.json + g_best (default: cp_lisennet).")
    p.add_argument("--onnx_dir", default=None,
                   help="Dir with the exported ONNX graphs (default: same as --checkpoint_dir).")
    p.add_argument("--repo-id", default=None,
                   help="Target HF repo (default: by variant — claroche1/LiSenNet for rnn, "
                        "claroche1/LiSenNet-npu for conv).")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print what would happen; don't touch HF.")
    a = p.parse_args()

    ckpt_dir = Path(a.checkpoint_dir).resolve()
    onnx_dir = Path(a.onnx_dir).resolve() if a.onnx_dir else ckpt_dir
    dirs = {"ckpt": ckpt_dir, "onnx": onnx_dir}

    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    variant = "conv" if cfg.get("bottleneck") == "conv" else "rnn"
    file_map = FILE_MAP_CONV if variant == "conv" else FILE_MAP_RNN
    card_tmpl = CARD_CONV if variant == "conv" else CARD_RNN
    repo_id = a.repo_id or DEFAULT_REPO[variant]
    commit_msg = a.commit_message or f"upload LiSenNet {variant} checkpoint + ONNX"

    resolved, missing = {}, []
    for repo_name, (kind, local) in file_map.items():
        src = dirs[kind] / local
        (resolved.__setitem__(repo_name, src) if src.is_file() else missing.append(str(src)))
    if missing:
        sys.exit("missing required artifacts (run export_onnx/quant_onnx/eval_deploy first):\n  "
                 + "\n  ".join(missing))

    print(f"variant:        {variant}  (bottleneck={cfg.get('bottleneck', 'rnn')})")
    print(f"checkpoint dir: {ckpt_dir}")
    print(f"onnx dir:       {onnx_dir}")
    print(f"repo-id:        {repo_id}")
    print("files:")
    for repo_name, src in resolved.items():
        print(f"  {repo_name:<36} <- {src.name}  ({src.stat().st_size/1024:.1f} KiB)")
    print(f"  {'README.md':<36} <- generated {variant} model card")

    if a.dry_run:
        print("\n[dry-run] would create repo + upload the files above. exiting.")
        return

    api = HfApi()
    print(f"\nensuring repo exists: {repo_id}")
    create_repo(repo_id, repo_type="model", private=a.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for repo_name, src in resolved.items():
            shutil.copy2(src, staged / repo_name)
        # plain replace (not str.format): the card has literal { } braces in code blocks.
        card = card_tmpl.replace("{repo_id}", repo_id).replace("{src}", SRC_REPO)
        (staged / "README.md").write_text(card)
        print(f"uploading to {repo_id} ...")
        url = api.upload_folder(folder_path=str(staged), repo_id=repo_id,
                                repo_type="model", commit_message=commit_msg)
    print(f"\ndone. {url}")
    print(f"model page: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
