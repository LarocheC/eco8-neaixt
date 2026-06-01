"""PTQ-at-w4/a8 sweep across the 9 HF-hosted 200-epoch checkpoints.

Downloads g_best + config.json per run, evaluates fp32 and w4/a8 PESQ on
the VBD test split, prints a comparison table identifying which models
need QAT (large fp32→w4/a8 drop).
"""
import argparse
import json
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from common.dataset import load_voicebank_demand
from common.env import AttrDict
from nsnet2.eval_torch import enhance_streaming_eager
from common.metrics import eval_pesq
from nsnet2.streaming import NSNet2Streaming
from common.quant_fake import QSpec, apply_ptq

REPO = "claroche1/sparse-nsnet2-checkpoints"
RUNS = [
    "wide_monarch", "baseline", "monarch_8", "monarch_full", "monarch_fc",
    "butterfly_2blocks", "butterfly_fc", "butterfly_ortho", "butterfly_full",
]


def _evaluate(streaming, h, test_split, max_utts):
    scores = []
    n_done = 0
    with torch.no_grad():
        for item in test_split:
            if max_utts is not None and n_done >= max_utts:
                break
            noisy = np.asarray(item["noisy"]["array"], dtype=np.float32)
            clean = np.asarray(item["clean"]["array"], dtype=np.float32)
            enhanced = enhance_streaming_eager(noisy, streaming, h)
            scores.append(eval_pesq(clean, enhanced, h.sampling_rate))
            n_done += 1
    valid = [s for s in scores if s != -1]
    return float(np.mean(valid)) if valid else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_utterances", type=int, default=100)
    p.add_argument("--hf_cache_dir", default=None)
    a = p.parse_args()

    hf_ds = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    test_split = hf_ds["test"]
    print(f"max_utterances={a.max_utterances or 'all'}  test_size={len(test_split)}")

    results = []
    for run in RUNS:
        print(f"\n--- {run} ---")
        cfg_path = hf_hub_download(REPO, f"{run}/config.json")
        ckpt_path = hf_hub_download(REPO, f"{run}/g_best")
        # The streaming wrapper reads config from the sibling dir of the
        # checkpoint, so copy/symlink config alongside the cached ckpt.
        ckpt_dir = os.path.dirname(ckpt_path)
        cfg_dst = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(cfg_dst):
            try:
                os.symlink(cfg_path, cfg_dst)
            except OSError:
                import shutil
                shutil.copy(cfg_path, cfg_dst)

        with open(cfg_path) as f:
            h = AttrDict(json.load(f))

        # fp32
        s = NSNet2Streaming.from_checkpoint(ckpt_path).eval()
        for sub in s.modules():
            if hasattr(sub, "use_triton") and isinstance(getattr(sub, "use_triton"), bool):
                sub.use_triton = False
        fp32_pesq = _evaluate(s, h, test_split, a.max_utterances)
        n_params = sum(p.numel() for p in s.parameters()) / 1e6
        print(f"  fp32: {fp32_pesq:.3f}  ({n_params:.2f}M params)")

        # w4/a8 PTQ
        s = NSNet2Streaming.from_checkpoint(ckpt_path).eval()
        for sub in s.modules():
            if hasattr(sub, "use_triton") and isinstance(getattr(sub, "use_triton"), bool):
                sub.use_triton = False
        nw, na = apply_ptq(s, weight=QSpec(bits=4), activation=QSpec(bits=8))
        w4_pesq = _evaluate(s, h, test_split, a.max_utterances)
        delta = w4_pesq - fp32_pesq
        print(f"  w4/a8: {w4_pesq:.3f}  (nw={nw}, na={na}, Δ={delta:+.3f})")
        results.append((run, n_params, fp32_pesq, w4_pesq, delta))

    print()
    print(f"{'run':22s} {'params':>7s} {'fp32':>6s} {'w4/a8':>6s} {'Δ':>7s} {'verdict':>14s}")
    print("-" * 70)
    for run, n, fp32, w4, d in results:
        verdict = "OK"
        if d < -0.10:
            verdict = "NEEDS QAT"
        elif d < -0.03:
            verdict = "borderline"
        print(f"{run:22s} {n:6.2f}M {fp32:6.3f} {w4:6.3f} {d:+7.3f} {verdict:>14s}")


if __name__ == "__main__":
    main()
