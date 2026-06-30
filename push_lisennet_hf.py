"""Upload a LiSenNet checkpoint + ONNX artifacts to a HuggingFace model repo.

Pushes the trained PyTorch checkpoint, the FP32 ONNX of the mask sub-network,
the dynamic and static int8 ONNX graphs, and the training config to
`claroche1/LiSenNet` (or any repo via --repo-id), plus a generated model card.
Idempotent: re-runs just create new commits.

The checkpoint comes from a training run dir (default `cp_lisennet/`); the ONNX
graphs come from the deploy dir produced by `lisennet.eval_deploy` /
`lisennet.export_onnx` + `lisennet.quant_onnx` (default `deploy/lisennet/`).

Usage:
    # uploads cp_lisennet/ + deploy/lisennet/ -> claroche1/LiSenNet
    python push_lisennet_hf.py

    # preview without uploading
    python push_lisennet_hf.py --dry-run

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


# repo filename -> (source dir kind, local filename)
#   "ckpt" = --checkpoint_dir, "onnx" = --onnx_dir
FILE_MAP = {
    "config.json": ("ckpt", "config.json"),
    "g_best": ("ckpt", "g_best"),
    "g_best_fp32.onnx": ("onnx", "g_best_fp32.onnx"),
    "g_best_int8_dyn.onnx": ("onnx", "g_best_int8_dyn.onnx"),
    "g_best_int8_static.onnx": ("onnx", "g_best_int8_static.onnx"),
}

SRC_REPO = "https://github.com/LarocheC/eco8-neaixt"

MODEL_CARD = """\
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

Training / export / quantization code and the full write-up:
[{src}]({src}) — see
[RESULTS_LISENNET.md]({src}/blob/main/RESULTS_LISENNET.md).

## Headline numbers

PESQ is wideband, on the full 824-utterance VoiceBank-DEMAND test split.

| metric                                  |     value |
| --------------------------------------- | --------: |
| params                                  | 36,783    |
| FP32 PESQ (torch / ONNX, Griffin-Lim)   | **3.006** |
| dynamic-int8 PESQ (Griffin-Lim)         |     2.995 |
| **real-time PESQ (int8 + noisy phase)** | **2.982** |
| static-int8 PESQ (Griffin-Lim)          |     2.897 |
| RTF (1 thread CPU)                      |     0.15  |

The ONNX export is loss-free; dynamic weight-only int8 costs ~0.012 PESQ. The
reproduction lands within ~0.06 of the paper's reported ~3.07.

## Files

| file                      | what it is |
| ------------------------- | --- |
| `g_best`                  | PyTorch checkpoint (`{"generator": state_dict}`) |
| `g_best_fp32.onnx`        | FP32 ONNX of the mask sub-network: `feat (B,3,T,F)` -> `est_mag (B,T,F)`, dynamic batch/time |
| `g_best_int8_dyn.onnx`    | Dynamic weight-only int8 (robust, near loss-free) — **recommended** |
| `g_best_int8_static.onnx` | Static full int8 (QDQ, per-tensor, percentile calibration; QAT-grade) |
| `config.json`             | Training + architecture config (STFT, channels, blocks) |

The ONNX graphs cover only the mask sub-network (the pure tensor-op core). The
STFT, the 3-channel feature extraction
(`[compressed_mag, group_delay/pi, instantaneous_freq_diff/pi]`), the magnitude
combination, and the phase recovery (Griffin-Lim offline / noisy phase for
real-time) live outside the graph — see the source repo's
`lisennet/streaming.py` and `lisennet/eval_deploy.py`.

## Loading

PyTorch:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from lisennet.model import build_lisennet

REPO = "{repo_id}"
cfg  = json.load(open(hf_hub_download(REPO, "config.json")))
ckpt = torch.load(hf_hub_download(REPO, "g_best"),
                  map_location="cpu", weights_only=True)
model = build_lisennet(AttrDict(cfg)).eval()
model.load_state_dict(ckpt["generator"])
# end-to-end waveform -> waveform: model(noisy_wav)["est"]
```

ONNX (FP32 or int8) — the mask sub-network:

```python
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "{repo_id}"
sess = ort.InferenceSession(
    hf_hub_download(REPO, "g_best_int8_dyn.onnx"),   # or _fp32 / _int8_static
    providers=["CPUExecutionProvider"],
)
# input  feat: (B, 3, T, F=257) = [compressed_mag, group_delay/pi, ifd/pi]
# output est_mag: (B, T, F)  -> pair with the noisy phase + iSTFT for audio
est_mag = sess.run(["est_mag"], {"feat": np.zeros((1, 3, 100, 257), np.float32)})[0]
```

## License

MIT. See the [source repository]({src}) for training code and full attribution.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--checkpoint_dir", default="cp_lisennet",
                   help="Dir with config.json + g_best (default: cp_lisennet).")
    p.add_argument("--onnx_dir", default="deploy/lisennet",
                   help="Dir with the exported ONNX graphs (default: deploy/lisennet).")
    p.add_argument("--repo-id", default="claroche1/LiSenNet",
                   help="Target HF model repo (default: claroche1/LiSenNet).")
    p.add_argument("--commit-message", default="upload LiSenNet checkpoint + ONNX (fp32, int8 dyn/static)",
                   help="Commit message for the HF upload.")
    p.add_argument("--private", action="store_true",
                   help="Create the HF repo as private (default: public).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen, but don't touch HF.")
    a = p.parse_args()

    ckpt_dir = Path(a.checkpoint_dir).resolve()
    onnx_dir = Path(a.onnx_dir).resolve()
    dirs = {"ckpt": ckpt_dir, "onnx": onnx_dir}

    resolved = {}
    missing = []
    for repo_name, (kind, local) in FILE_MAP.items():
        src = dirs[kind] / local
        if src.is_file():
            resolved[repo_name] = src
        else:
            missing.append(str(src))
    if missing:
        sys.exit("missing required artifacts:\n  " + "\n  ".join(missing))

    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)
    print(f"checkpoint dir: {ckpt_dir}")
    print(f"onnx dir:       {onnx_dir}")
    print(f"repo-id:        {a.repo_id}")
    print(f"model:          num_channels={cfg.get('num_channels')}, n_blocks={cfg.get('n_blocks')}, "
          f"n_fft={cfg.get('n_fft')}, hop={cfg.get('hop_size')}")
    print("files:")
    for repo_name, src in resolved.items():
        print(f"  {repo_name:<24} <- {src}  ({src.stat().st_size/1024:.1f} KiB)")
    print(f"  {'README.md':<24} <- generated model card")
    print(f"private:        {a.private}")

    if a.dry_run:
        print("\n[dry-run] would create repo + upload the files above. exiting.")
        return

    api = HfApi()
    print(f"\nensuring repo exists: {a.repo_id}")
    create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for repo_name, src in resolved.items():
            shutil.copy2(src, staged / repo_name)        # copy (Windows-safe staging)
        # plain replace (not str.format): the card contains literal { } braces
        # in code blocks that would break format-field parsing.
        card = MODEL_CARD.replace("{repo_id}", a.repo_id).replace("{src}", SRC_REPO)
        (staged / "README.md").write_text(card)

        print(f"uploading to {a.repo_id} ...")
        url = api.upload_folder(
            folder_path=str(staged),
            repo_id=a.repo_id,
            repo_type="model",
            commit_message=a.commit_message,
        )
    print(f"\ndone. {url}")
    print(f"model page: https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
