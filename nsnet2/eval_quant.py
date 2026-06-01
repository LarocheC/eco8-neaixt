"""Evaluate a trained NSNet2 checkpoint under fake-quantization.

Builds on eval_torch.py — same eager PESQ pipeline through the
streaming wrapper — but applies in-place weight fake-quant + activation
pre-hooks before evaluation. Lets us sweep (weight_bits, act_bits,
group_size) without touching the checkpoint on disk.

Usage:
    python -m nsnet2.eval_quant --checkpoint_file cp_tr_monarch_8/g_best \\
        --w_bits 4 --a_bits 8 --w_group_size 64 --max_utterances 50

Or run a small predefined sweep with --sweep:

    python -m nsnet2.eval_quant --checkpoint_file cp_tr_monarch_8/g_best \\
        --sweep --max_utterances 50
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from common.dataset import load_voicebank_demand
from common.env import AttrDict
from nsnet2.eval_torch import enhance_streaming_eager
from common.metrics import eval_pesq
from nsnet2.streaming import NSNet2Streaming
from common.quant_fake import QSpec, apply_ptq


# Small sweep of operating points to give a starting picture.
PRESET_SWEEP = [
    ("fp32",         QSpec(bits=32),                                     QSpec(bits=32)),
    ("w8_pc / a8",   QSpec(bits=8,  axis=None),                          QSpec(bits=8)),
    ("w4_pc / a8",   QSpec(bits=4,  axis=None),                          QSpec(bits=8)),
    ("w4_g64 / a8",  QSpec(bits=4,  axis=None, group_size=64),           QSpec(bits=8)),
    ("w4_g32 / a8",  QSpec(bits=4,  axis=None, group_size=32),           QSpec(bits=8)),
    ("w2_pc / a8",   QSpec(bits=2,  axis=None),                          QSpec(bits=8)),
]


def _spec(bits, axis, group_size):
    """Build a QSpec from CLI args. Default axis=None (per-channel via
    apply_weight_fake_quant's per-module default)."""
    return QSpec(bits=bits, axis=axis, group_size=group_size)


def _evaluate(streaming, h, test_split, max_utts):
    pesq_scores = []
    n_done = 0
    with torch.no_grad():
        for item in test_split:
            if max_utts is not None and n_done >= max_utts:
                break
            noisy = np.asarray(item["noisy"]["array"], dtype=np.float32)
            clean = np.asarray(item["clean"]["array"], dtype=np.float32)
            enhanced = enhance_streaming_eager(noisy, streaming, h)
            pesq_scores.append(eval_pesq(clean, enhanced, h.sampling_rate))
            n_done += 1
    clean = [s for s in pesq_scores if s != -1]
    n_fail = len(pesq_scores) - len(clean)
    mean = float(np.mean(clean)) if clean else float("nan")
    return mean, n_fail, len(pesq_scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_file", required=True)
    p.add_argument("--hf_cache_dir", default=None)
    p.add_argument("--max_utterances", type=int, default=None)
    p.add_argument("--w_bits", type=int, default=32)
    p.add_argument("--a_bits", type=int, default=32)
    p.add_argument("--w_axis", type=int, default=None,
                   help="Per-channel axis. Default: per-module sensible default.")
    p.add_argument("--w_group_size", type=int, default=None)
    p.add_argument("--sweep", action="store_true",
                   help="Ignore --w_bits/--a_bits and run PRESET_SWEEP.")
    a = p.parse_args()

    cp_dir = os.path.split(a.checkpoint_file)[0]
    with open(os.path.join(cp_dir, "config.json")) as f:
        h = AttrDict(json.loads(f.read()))
    torch.manual_seed(h.seed)

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    test_split = hf["test"]

    # Auto-detect a QAT recipe sidecar written by qat_train.py. Lets us run
    # `python -m nsnet2.eval_quant --checkpoint_file cp_qat_*/g_best` without re-
    # typing the bits — the eval automatically uses the same quant pattern
    # the model was trained against.
    recipe_path = os.path.join(cp_dir, "qat_recipe.json")
    qat_recipe = None
    if os.path.exists(recipe_path) and not a.sweep and a.w_bits == 32 and a.a_bits == 32:
        with open(recipe_path) as f:
            qat_recipe = json.load(f)
        print(f"detected qat_recipe.json: {qat_recipe}")

    if a.sweep:
        runs = PRESET_SWEEP
    elif qat_recipe is not None:
        runs = [
            (
                f"qat w{qat_recipe['w_bits']}/a{qat_recipe['a_bits']}",
                _spec(qat_recipe["w_bits"], qat_recipe.get("w_axis"),
                      qat_recipe.get("w_group_size")),
                _spec(qat_recipe["a_bits"], None, None),
            )
        ]
    else:
        runs = [
            (
                f"w{a.w_bits}/a{a.a_bits}",
                _spec(a.w_bits, a.w_axis, a.w_group_size),
                _spec(a.a_bits, None, None),
            )
        ]

    print(f"checkpoint: {a.checkpoint_file}")
    print(f"max_utterances: {a.max_utterances or 'all'} ({len(test_split)} available)")
    print(f"{'config':20s}  {'PESQ':>6s}  {'fail':>4s}  {'nw':>3s}  {'na':>3s}")
    for label, w_spec, a_spec in runs:
        # Reload checkpoint each run — weight quant is in-place mutation.
        streaming = NSNet2Streaming.from_checkpoint(a.checkpoint_file).eval()
        # If the checkpoint was trained with kind="triton_*", flip Triton off
        # so the per-step Python path runs (it's what our hooks intercept).
        for sub in streaming.modules():
            if hasattr(sub, "use_triton") and isinstance(getattr(sub, "use_triton"), bool):
                sub.use_triton = False
        nw, na = apply_ptq(streaming, weight=w_spec, activation=a_spec)
        mean, n_fail, n_total = _evaluate(streaming, h, test_split, a.max_utterances)
        print(f"{label:20s}  {mean:6.3f}  {n_fail:4d}  {nw:3d}  {na:3d}")


if __name__ == "__main__":
    main()
