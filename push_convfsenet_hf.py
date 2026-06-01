"""Upload a ConvFSENet checkpoint directory to a HuggingFace model repo.

Pushes the four deployment artifacts produced by a ConvFSENet training run
(PyTorch checkpoint, FP32 streaming ONNX, int8 streaming ONNX, training
config) plus a generated model card to `claroche1/convfsenet` (or any other
repo via --repo-id). Idempotent: re-runs just create new commits.

Usage:
    # uploads cp_convfsenet/ -> claroche1/convfsenet
    python push_convfsenet_hf.py

    # preview without uploading
    python push_convfsenet_hf.py --dry-run

    # custom source / destination
    python push_convfsenet_hf.py --source cp_convfsenet \\
        --repo-id myuser/my-convfsenet --commit-message "retrain v2"

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


REQUIRED_FILES = ("config.json", "g_best", "g_best.onnx", "g_best_fp32.onnx")

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
datasets:
- JacobLinCool/VoiceBank-DEMAND-16k
---

# ConvFSENet

A causal, fully-convolutional speech enhancer trained on VoiceBank-DEMAND-16k.
Source: [github.com/LarocheC/sparse-nsnet2](https://github.com/LarocheC/sparse-nsnet2).
See [RESULTS_CONVFSENET.md](https://github.com/LarocheC/sparse-nsnet2/blob/main/RESULTS_CONVFSENET.md)
for the full results, architecture description, and the magnitude-compression
trick that makes int8 deployment essentially loss-free.

## Headline numbers

| metric         | value     |
| -------------- | --------: |
| params         | 1.45 M    |
| FP32 PESQ      | **2.931** |
| int8 PESQ      | **2.911** |
| Δ (FP32→int8)  | +0.020    |
| int8 RTF (ORT CPU) | 0.017 |
| int8 size      | 1.6 MiB   |

PESQ is on the full 824-utterance VoiceBank-DEMAND test split.

## Files

| file                | what it is |
| ------------------- | --- |
| `g_best`            | PyTorch checkpoint (full state dict — `generator`, `optim`, etc.) |
| `g_best_fp32.onnx`  | Streaming FP32 ONNX (per-frame inputs + FIFO state buffers) |
| `g_best.onnx`       | Static int8 ONNX (QDQ, per-channel weights, MinMax calibration; compression prologue kept FP32) |
| `config.json`       | Training config (architecture + STFT params) |

## Loading

PyTorch:

```python
import json, torch
from huggingface_hub import hf_hub_download
from common.env import AttrDict
from convfsenet.model import build_causal_model

REPO = "{repo_id}"
cfg  = json.load(open(hf_hub_download(REPO, "config.json")))
ckpt = torch.load(hf_hub_download(REPO, "g_best"),
                  map_location="cuda", weights_only=False)
model = build_causal_model(AttrDict(cfg)).cuda().eval()
model.load_state_dict(ckpt["generator"])
```

ONNX (FP32 or int8):

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO = "{repo_id}"
sess = ort.InferenceSession(
    hf_hub_download(REPO, "g_best.onnx"),       # or g_best_fp32.onnx
    providers=["CPUExecutionProvider"],
)
# Streaming shape: feed one frame of magnitude STFT (B, n_freq) + the per-block
# FIFO state buffers per call. End-to-end RMS-norm + STFT + frame loop + iSTFT
# pipeline lives in convfsenet/inference_onnx.py in the source repo.
```

## License

MIT. See the [source repository](https://github.com/LarocheC/sparse-nsnet2) for
training code and full attribution.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--source", default="cp_convfsenet",
                   help="Local checkpoint directory to upload (default: cp_convfsenet).")
    p.add_argument("--repo-id", default="claroche1/convfsenet",
                   help="Target HF model repo (default: claroche1/convfsenet).")
    p.add_argument("--commit-message", default="upload ConvFSENet checkpoint",
                   help="Commit message for the HF upload.")
    p.add_argument("--private", action="store_true",
                   help="Create the HF repo as private (default: public).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen, but don't touch HF.")
    a = p.parse_args()

    source = Path(a.source).resolve()
    if not source.is_dir():
        sys.exit(f"source dir does not exist: {source}")

    missing = [f for f in REQUIRED_FILES if not (source / f).is_file()]
    if missing:
        sys.exit(f"source dir {source} is missing required files: {missing}")

    # Sanity-check the config to make sure we're uploading the right model.
    with open(source / "config.json") as f:
        cfg = json.load(f)
    extractor = cfg.get("extractor_type", "<unset>")
    print(f"source:        {source}")
    print(f"repo-id:       {a.repo_id}")
    print(f"extractor:     {extractor}")
    if extractor != "mag_compressed":
        print(f"  WARNING: extractor_type is {extractor!r}, not 'mag_compressed'. "
              "This is the deployment-broken path — confirm this is what you want.")
    print(f"files:         {', '.join(REQUIRED_FILES)} + README.md (model card)")
    print(f"private:       {a.private}")

    if a.dry_run:
        print("\n[dry-run] would create repo + upload the files above. exiting.")
        return

    api = HfApi()
    print(f"\nensuring repo exists: {a.repo_id}")
    create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)

    # Stage source files + generated model card in a temp dir, then upload as
    # one folder (one HF commit).
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for f in REQUIRED_FILES:
            # Copy (not symlink) so staging works on Windows, where symlinks
            # need elevated privileges.
            shutil.copy2(source / f, staged / f)
        (staged / "README.md").write_text(MODEL_CARD.format(repo_id=a.repo_id))

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
