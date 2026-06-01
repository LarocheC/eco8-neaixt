"""eval_ptq.py — eager-mode weight/activation PTQ study for ConvFSENet.

Measures PESQ degradation under post-training quantization at configurable
bit-widths, using the fake-quant tooling in quant_fake.py. The ConvFSENet
counterpart to the NSNet2 w4/w8 weight-quant study.

For each (w_bits, a_bits) config it builds a fresh offline ConvFSENet, loads
the checkpoint, applies apply_ptq (per-output-channel symmetric weight
fake-quant + dynamic per-tensor symmetric activation fake-quant on conv
inputs), runs the full VoiceBank-DEMAND test split through the offline model,
and reports mean PESQ.

Scope / caveats:
  - A STUDY tool, not a deployment path. The activation fake-quant is dynamic
    per-tensor symmetric (recomputed each forward) — more optimistic than the
    static asymmetric int8 the ONNX deployment uses. The weight fake-quant
    (per-output-channel symmetric) does faithfully model int-N weight quant.
    So absolute PESQ is mildly optimistic, but the comparison at fixed a_bits
    is clean: the activation path is identical across configs, so a
    w8a8 -> w4a8 delta isolates the cost of 4-bit weights.
  - The magnitude-compression extractor is a plain FP32 lambda, so int8/int4
    only ever sees the compressed, range-friendly features — matching the
    fixed ONNX quant path (raw |stft| stays FP32).
  - Per-channel int4 weights have range [-7, 7].

Usage:
    python -m convfsenet.eval_ptq --checkpoint_file cp_convfsenet/g_best
    python -m convfsenet.eval_ptq --checkpoint_file cp_convfsenet/g_best \\
        --configs fp32,w8a8,w4a8,w4a4
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from torch.utils.data import DataLoader

from convfsenet.model import build_causal_model
from common.dataset import Dataset, load_voicebank_demand
from common.env import AttrDict
from common.quant_fake import QSpec, apply_ptq
from convfsenet.train import _validate
from common.utils import load_checkpoint


def _parse_config(name: str) -> tuple[int, int]:
    """'fp32' -> (32, 32); 'w4a8' -> (4, 8); 'w8a8' -> (8, 8)."""
    name = name.strip().lower()
    if name == "fp32":
        return 32, 32
    m = re.fullmatch(r"w(\d+)a(\d+)", name)
    if not m:
        raise ValueError(f"bad config {name!r}; expected 'fp32' or 'wNaM'")
    return int(m.group(1)), int(m.group(2))


def _eval_config(h, ckpt_path, device, loader, w_bits: int, a_bits: int) -> tuple:
    """Fresh model + checkpoint + apply_ptq + full-testset PESQ. apply_ptq
    mutates weights in place, so a fresh model is rebuilt for every config."""
    model = build_causal_model(h).to(device)
    sd = load_checkpoint(ckpt_path, device)
    if "generator" not in sd:
        raise KeyError(f"{ckpt_path} is not a g_best checkpoint (no 'generator')")
    model.load_state_dict(sd["generator"], strict=True)
    nw, na = apply_ptq(
        model,
        weight=QSpec(bits=w_bits, axis=None),   # axis=None -> per-output-channel
        activation=QSpec(bits=a_bits),
    )
    pesq = _validate(model, loader, device, h)
    return pesq, nw, na


def main():
    p = argparse.ArgumentParser(description="ConvFSENet weight/activation PTQ study.")
    p.add_argument("--checkpoint_file", required=True,
                   help="g_best checkpoint to quantize. config.json must sit beside it.")
    p.add_argument("--configs", default="fp32,w8a8,w4a8",
                   help="Comma-separated bit configs (e.g. fp32,w8a8,w4a8).")
    p.add_argument("--hf_cache_dir", default=None)
    p.add_argument("--max_utterances", type=int, default=None,
                   help="Optional test-split slice for fast smoke iteration.")
    a = p.parse_args()

    config_path = os.path.join(os.path.dirname(a.checkpoint_file), "config.json")
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(h.seed))

    hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
    test = hf["test"]
    if a.max_utterances is not None:
        test = test.select(range(min(a.max_utterances, len(test))))
    validset = Dataset(test, h.segment_size, h.sampling_rate,
                       split=False, shuffle=False, seed=h.seed)
    loader = DataLoader(validset, num_workers=1, shuffle=False,
                        batch_size=1, pin_memory=True, drop_last=True)
    print(f"PTQ study: {a.checkpoint_file}  ({len(validset)} test utterances)")

    configs = [c for c in a.configs.split(",") if c.strip()]
    results = []
    for name in configs:
        w_bits, a_bits = _parse_config(name)
        pesq, nw, na = _eval_config(h, a.checkpoint_file, device, loader, w_bits, a_bits)
        results.append((name, w_bits, a_bits, pesq, nw, na))
        print(f"  {name:6s}  w{w_bits}/a{a_bits}  "
              f"weights_fq={nw} act_fq={na}  PESQ={pesq:.4f}")

    fp32_pesq = next((r[3] for r in results if r[1] >= 32 and r[2] >= 32), None)
    print("\n=== ConvFSENet PTQ study ===")
    print(f"{'config':8s} {'PESQ':>8s} {'Δ vs fp32':>11s}")
    for name, w_bits, a_bits, pesq, _, _ in results:
        delta = f"{pesq - fp32_pesq:+.4f}" if fp32_pesq is not None else "n/a"
        print(f"{name:8s} {pesq:8.4f} {delta:>11s}")


if __name__ == "__main__":
    main()
