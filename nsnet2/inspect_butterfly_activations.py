"""inspect_butterfly_activations.py - per-stage activation magnitude trace.

For two NSNet2Streaming checkpoints, runs streaming inference on a single
VBD test utterance and records max-abs / RMS of the activation tensor
*after each unrolled butterfly stage* of every Butterfly module. The trace
exposes whether activation magnitude grows / stays flat / shrinks across
the log_n stages — directly testing the "randn-init butterfly amplifies
across stages, compounding int8 quant error" hypothesis.

Implementation: monkey-patches torch_structured.butterfly.butterfly's
module-level `butterfly_multiply` with a torch-only re-implementation that
also records per-stage statistics into a global trace dict, keyed by the
Butterfly instance's named_modules() path. Forward pre/post hooks set the
"current instance" so the traced function knows which module's stages to
attribute its records to.

Run:
    uv run python -m nsnet2.inspect_butterfly_activations \\
        cp_butterfly_full/g_best cp_butterfly_ortho/g_best
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common.dataset import load_voicebank_demand, mag_pha_stft
from common.env import AttrDict
from nsnet2.streaming import NSNet2Streaming


_CURRENT_INSTANCE: list[str | None] = [None]
_TRACE: dict[str, list[dict]] = defaultdict(list)


def _butterfly_multiply_traced(twiddle, input, increasing_stride=True, output_size=None):
    """Drop-in replacement for butterfly_multiply that records per-stage stats.

    Mirrors torch_structured.butterfly.multiply.butterfly_multiply_torch
    verbatim, then appends one record per stage to _TRACE under the name set
    by the active forward pre-hook.
    """
    name = _CURRENT_INSTANCE[0]
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    if input_size < n:
        input = F.pad(input, (0, n - input_size))
    elif input_size > n:
        input = input[:, :, :n]
    if output_size is None:
        output_size = n
    output = input.contiguous()

    if name is not None:
        _TRACE[name].append({
            "stage": -1, "max_abs": float(output.abs().max()),
            "rms": float(output.pow(2).mean().sqrt()),
        })

    cur_inc = increasing_stride
    stage = 0
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_inc else log_n - 1 - idx
            stride = 1 << log_stride
            t = twiddle[:, block, idx].view(
                nstacks, n // (2 * stride), stride, 2, 2
            ).permute(0, 1, 3, 4, 2)
            out_r = output.view(
                batch_size, nstacks, n // (2 * stride), 1, 2, stride
            )
            output = (t * out_r).sum(dim=4)
            if name is not None:
                _TRACE[name].append({
                    "stage": stage,
                    "max_abs": float(output.abs().max()),
                    "rms": float(output.pow(2).mean().sqrt()),
                })
            stage += 1
        cur_inc = not cur_inc
    return output.view(batch_size, nstacks, n)[:, :, :output_size]


def _install_hooks(streaming):
    """Tag each Butterfly instance with its named_modules() path."""
    from torch_structured.butterfly.butterfly import Butterfly
    handles = []
    for name, mod in streaming.named_modules():
        if isinstance(mod, Butterfly):
            def _pre(_module, _inp, *, _name=name):
                _CURRENT_INSTANCE[0] = _name
            def _post(_module, _inp, _out):
                _CURRENT_INSTANCE[0] = None
            handles.append(mod.register_forward_pre_hook(_pre))
            handles.append(mod.register_forward_hook(_post))
    return handles


def _patch_butterfly_multiply():
    from torch_structured.butterfly import butterfly as _bf_mod
    original = _bf_mod.butterfly_multiply
    _bf_mod.butterfly_multiply = _butterfly_multiply_traced
    return _bf_mod, original


def _aggregate_trace():
    """Per (instance, stage), mean max_abs and mean RMS across all calls (= frames)."""
    out = {}
    for name, records in _TRACE.items():
        per_stage: dict[int, list[dict]] = defaultdict(list)
        for r in records:
            per_stage[r["stage"]].append(r)
        agg = []
        for stage in sorted(per_stage):
            rs = per_stage[stage]
            agg.append({
                "stage": stage,
                "max_abs_mean": float(np.mean([r["max_abs"] for r in rs])),
                "max_abs_max":  float(np.max([r["max_abs"] for r in rs])),
                "rms_mean":     float(np.mean([r["rms"] for r in rs])),
                "n_calls":      len(rs),
            })
        out[name] = agg
    return out


def trace_one_utterance(ckpt_path: str, noisy_wav: np.ndarray, h: AttrDict):
    streaming = NSNet2Streaming.from_checkpoint(ckpt_path).eval()
    _TRACE.clear()
    _CURRENT_INSTANCE[0] = None
    bf_mod, original = _patch_butterfly_multiply()
    handles = _install_hooks(streaming)
    try:
        with torch.no_grad():
            noisy_t = torch.from_numpy(np.asarray(noisy_wav, dtype=np.float32))
            norm_factor = torch.sqrt(len(noisy_t) / (torch.sum(noisy_t ** 2.0) + 1e-8))
            noisy_t = (noisy_t * norm_factor).unsqueeze(0)
            noisy_mag, _, _ = mag_pha_stft(
                noisy_t, h.n_fft, h.hop_size, h.win_size, h.compress_factor,
            )
            B, _, T = noisy_mag.shape
            states = torch.zeros(streaming.num_layers, B, streaming.hidden_size)
            for t in range(T):
                _, states = streaming.forward_step(noisy_mag[:, :, t], states)
        return _aggregate_trace()
    finally:
        for hndl in handles:
            hndl.remove()
        bf_mod.butterfly_multiply = original


def _format_table(agg, label):
    lines = [f"\n=== {label} ==="]
    for name in sorted(agg):
        stages = agg[name]
        n_stages = len([s for s in stages if s["stage"] >= 0])
        n_calls = stages[0]["n_calls"] if stages else 0
        lines.append(f"\n  [{name}]  (frames={n_calls}, stages={n_stages})")
        lines.append(f"    {'stage':>5} {'max_abs (mean)':>16} {'max_abs (max)':>16} {'rms (mean)':>14}")
        for s in stages:
            tag = "input" if s["stage"] == -1 else f"{s['stage']}"
            lines.append(
                f"    {tag:>5} "
                f"{s['max_abs_mean']:>16.4f} "
                f"{s['max_abs_max']:>16.4f} "
                f"{s['rms_mean']:>14.5f}"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="Two or more cp_*/g_best paths")
    ap.add_argument("--utt_index", type=int, default=0,
                    help="VBD test split index to use (default 0)")
    ap.add_argument("--hf_cache_dir", default=None)
    args = ap.parse_args()

    cp_dir = os.path.split(args.checkpoints[0])[0]
    with open(os.path.join(cp_dir, "config.json")) as f:
        h = AttrDict(json.loads(f.read()))
    torch.manual_seed(h.seed)

    hf = load_voicebank_demand(cache_dir=args.hf_cache_dir)
    item = hf["test"][args.utt_index]
    noisy_wav = np.asarray(item["noisy"]["array"], dtype=np.float32)
    print(f"Tracing on VBD test[{args.utt_index}] id={item['id']} "
          f"({len(noisy_wav)} samples = {len(noisy_wav) / h.sampling_rate:.1f}s)")

    for ckpt in args.checkpoints:
        agg = trace_one_utterance(ckpt, noisy_wav, h)
        print(_format_table(agg, ckpt))


if __name__ == "__main__":
    main()
