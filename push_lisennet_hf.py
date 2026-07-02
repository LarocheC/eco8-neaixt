"""Upload LiSenNet checkpoints + ONNX artifacts to one HuggingFace repo.

Both LiSenNet variants live in a single model repo (``claroche1/LiSenNet``), each
under its own subfolder, with a combined root model card:

  * **gru/**  — the faithful GRU LiSenNet (PESQ 3.006), the quality reference. Its
    GRU + 2-axis LayerNorm do NOT map to the STM32N6 Neural-ART NPU.
  * **conv/** — the dual-path-conv variant (PESQ 2.970). Its ops map to the NPU but
    the FIFO-state streaming graph crashes the Neural-ART codegen; kept as the
    CPU/onnxruntime streaming reference.
  * **conv-hardened/** — the NPU-DEPLOYABLE variant (FP32 3.013 / real-time int8
    2.998): batchnorm+relu+convtranspose primitives and a stateless **windowed**
    graph that compiles to Neural-ART — the stedgeai target.

The variant (and its subfolder) is auto-detected from the run's config.json
(``bottleneck``, and ``norm=batchnorm`` for the hardened recipe). Each push writes
its own subfolder plus the shared root README, so pushing one variant never
disturbs the other. Idempotent.

Usage:
    python push_lisennet_hf.py --checkpoint_dir cp_lisennet                     # -> gru/
    python push_lisennet_hf.py --checkpoint_dir cp_lisennet_conv_wide           # -> conv/
    python push_lisennet_hf.py --checkpoint_dir cp_lisennet_conv_hardened_nc24  # -> conv-hardened/
    python push_lisennet_hf.py --checkpoint_dir cp_lisennet_conv_hardened_nc24 --dry-run

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
DEFAULT_REPO = "claroche1/LiSenNet"
SUBDIR = {"rnn": "gru", "conv": "conv", "conv_hardened": "conv-hardened"}

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
# hardened: the deploy graph is the stateless WINDOWED one (the FIFO streaming
# graph segfaults the Neural-ART codegen — see LISENNET_NPU_HANDOVER.md).
FILE_MAP_CONV_HARDENED = {
    **FILE_MAP_RNN,
    "g_best_windowed_fp32.onnx": ("onnx", "g_best_windowed_fp32.onnx"),
    "g_best_windowed_int8_static.onnx": ("onnx", "g_best_windowed_fp32.int8_static.onnx"),
}
FILE_MAP = {"rnn": FILE_MAP_RNN, "conv": FILE_MAP_CONV,
            "conv_hardened": FILE_MAP_CONV_HARDENED}

# Shared root model card (describes both variants). Static — every push writes it.
CARD_INDEX = """\
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

# LiSenNet

Ultra-compact, causal, real-time speech enhancers trained on
VoiceBank-DEMAND-16k — a sub-band U-Net with a magnitude-only mask (phase from a
2-iteration Griffin-Lim offline, or the noisy phase for real-time). Port of
**Yan, Zhou, Chen & Lu, _LiSenNet_,
[arXiv:2409.13285](https://arxiv.org/abs/2409.13285)**
([hyyan2k/LiSenNet](https://github.com/hyyan2k/LiSenNet), MIT).

This repo holds **three variants**, each in its own subfolder:

| subfolder | recipe | params | NPU-compiles | FP32 PESQ | real-time int8 PESQ |
| --------- | ------ | -----: | :----------: | --------: | ------------------: |
| [`gru/`](./gru)   | dual-path **GRU** (faithful)  | 36,783 |  ✗  |     3.006 |               2.930 |
| [`conv/`](./conv) | dual-path **conv**            | 41,063 |  ✗  |     2.970 |               2.855 |
| [`conv-hardened/`](./conv-hardened) | conv + **NPU-hardened** | 36,288 |  ✓  | **3.013** |           **2.998** |

PESQ is wideband, on the full 824-utterance VoiceBank-DEMAND test split.

* **`gru/`** is the faithful reproduction and the original quality reference. Its
  GRU + 2-axis `LayerNorm` do **not** compile to the STM32N6 Neural-ART NPU.
* **`conv/`** replaces the GRU bottleneck with a dual-path conv one (0 GRU /
  0 LayerNormalization). Its ops map to the NPU, but the FIFO-state streaming
  graph (`conv/g_best_streaming_fp32.onnx`, `feat + N state_i_in -> est_mag +
  N state_i_out`) crashes the Neural-ART codegen — kept as the CPU/onnxruntime
  frame-by-frame reference.
* **`conv-hardened/`** is the **NPU-deployable** variant and the current best
  model overall: per-channel BatchNorm (folds into the convs), ReLU, plain
  ConvTranspose upsampling, and a stateless **windowed** deploy graph
  (`conv-hardened/g_best_windowed_int8_static.onnx`, signed QInt8,
  `feat_window (B,3,132,257) -> est_mag (B,64,257)`, window = receptive field
  68 + 64 emitted frames) that **compiles to Neural-ART** — the artifact handed
  to stedgeai. The hardened primitives also quantize far better (int8 drop
  −0.016 vs −0.115 for `conv/`).

Code + full write-up: [{src}]({src}) — see
[RESULTS_LISENNET.md]({src}/blob/main/RESULTS_LISENNET.md).

## Files (per subfolder)

`config.json`, `g_best` (PyTorch `{"generator": state_dict}`), `g_best_fp32.onnx`
and `g_best_int8_static.onnx` (whole-utterance mask sub-network,
`feat (B,3,T,F) -> est_mag (B,T,F)`). `conv/` additionally has
`g_best_streaming_fp32.onnx` and `g_best_streaming_int8_static.onnx` (single
frame + explicit state I/O); `conv-hardened/` has `g_best_windowed_fp32.onnx`
and `g_best_windowed_int8_static.onnx` (stateless windowed deploy graph, the
stedgeai / Neural-ART target). The ONNX graphs are the mask sub-network only —
STFT, feature build and phase recovery stay host-side.

## Loading (PyTorch)

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from lisennet.model import build_lisennet

REPO, SUB = "{repo_id}", "conv-hardened"      # or "gru" / "conv"
cfg  = json.load(open(hf_hub_download(REPO, f"{SUB}/config.json")))
ckpt = torch.load(hf_hub_download(REPO, f"{SUB}/g_best"), map_location="cpu", weights_only=True)
model = build_lisennet(AttrDict(cfg)).eval()
model.load_state_dict(ckpt["generator"])   # model(noisy_wav)["est"]
```

## Running the NPU windowed deploy graph (`conv-hardened/`)

Stateless: feed a sliding window of the last `68 + 64 = 132` feature frames and
read the 64 newest enhanced-magnitude frames (no state tensors to carry).

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

sess = ort.InferenceSession(
    hf_hub_download("{repo_id}", "conv-hardened/g_best_windowed_int8_static.onnx"),
    providers=["CPUExecutionProvider"],
)
feat_window = np.zeros((1, 3, 132, 257), np.float32)   # last 68+64 feature frames
est_mag = sess.run(["est_mag"], {"feat_window": feat_window})[0]  # (1, 64, 257)
```

## Running the CPU streaming graph frame-by-frame (`conv/`)

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

sess = ort.InferenceSession(
    hf_hub_download("{repo_id}", "conv/g_best_streaming_fp32.onnx"),
    providers=["CPUExecutionProvider"],
)
state_in = [i for i in sess.get_inputs() if i.name != "feat"]   # FIFO states
out_names = [o.name for o in sess.get_outputs()]                # est_mag + state_*_out
zeros = lambda s: np.zeros([d if isinstance(d, int) else 1 for d in s], np.float32)
states = {i.name: zeros(i.shape) for i in state_in}            # start-of-stream = zeros

def step(feat_t):                                              # feat_t: (1, 3, 1, 257)
    res = sess.run(out_names, {"feat": feat_t, **states})
    for i, v in zip(state_in, res[1:]):
        states[i.name] = v
    return res[0]                                              # est_mag (1, 1, 257)
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
    p.add_argument("--repo-id", default=DEFAULT_REPO,
                   help=f"Target HF repo (default: {DEFAULT_REPO}). Both variants share it.")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print what would happen; don't touch HF.")
    a = p.parse_args()

    ckpt_dir = Path(a.checkpoint_dir).resolve()
    onnx_dir = Path(a.onnx_dir).resolve() if a.onnx_dir else ckpt_dir
    dirs = {"ckpt": ckpt_dir, "onnx": onnx_dir}

    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    if cfg.get("bottleneck") == "conv":
        variant = "conv_hardened" if cfg.get("norm") == "batchnorm" else "conv"
    else:
        variant = "rnn"
    sub = SUBDIR[variant]
    file_map = FILE_MAP[variant]
    commit_msg = a.commit_message or f"upload LiSenNet {variant} variant -> {sub}/"

    resolved, missing = {}, []
    for repo_name, (kind, local) in file_map.items():
        src = dirs[kind] / local
        (resolved.__setitem__(repo_name, src) if src.is_file() else missing.append(str(src)))
    if missing:
        sys.exit("missing required artifacts (run export_onnx/quant_onnx/eval_deploy first):\n  "
                 + "\n  ".join(missing))

    print(f"variant:        {variant}  (bottleneck={cfg.get('bottleneck', 'rnn')})")
    print(f"repo-id:        {a.repo_id}   subfolder: {sub}/")
    print(f"checkpoint dir: {ckpt_dir}")
    print(f"onnx dir:       {onnx_dir}")
    print("files:")
    for repo_name, src in resolved.items():
        print(f"  {sub}/{repo_name:<34} <- {src.name}  ({src.stat().st_size/1024:.1f} KiB)")
    print(f"  {'README.md':<40} <- generated combined root card")

    if a.dry_run:
        print("\n[dry-run] would create repo + upload the files above. exiting.")
        return

    api = HfApi()
    print(f"\nensuring repo exists: {a.repo_id}")
    create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        (staged / sub).mkdir()
        for repo_name, src in resolved.items():
            shutil.copy2(src, staged / sub / repo_name)      # -> {sub}/{repo_name}
        # plain replace (not str.format): the card has literal { } braces in code blocks.
        card = CARD_INDEX.replace("{repo_id}", a.repo_id).replace("{src}", SRC_REPO)
        (staged / "README.md").write_text(card)
        print(f"uploading {sub}/ + README.md to {a.repo_id} ...")
        url = api.upload_folder(folder_path=str(staged), repo_id=a.repo_id,
                                repo_type="model", commit_message=commit_msg)
    print(f"\ndone. {url}")
    print(f"model page: https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
