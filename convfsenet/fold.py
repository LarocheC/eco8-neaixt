"""Fold a block-design ConvFSENet into an equivalent plain (flags-off) model.

The block-design options (convfsenet/model.py) put two things in front of the
frontend ReLU that the export / streaming / quantisation code does not know:

    feats' = feats - mu                      (input_norm: frozen per-bin mean)
    z      = BN(W feats' + b)                (frontend_norm='batch', eval stats)

Both are affine, so for y = BN(W (x - mu) + b), s = sqrt(var + eps):

    W' = diag(g / s) W
    b' = (g / s) (b - W mu - m) + beta

and relu(W' x + b') is the same function. The folded state_dict has exactly
the keys and shapes of the plain model (no frontend BN, no input_mean), so it
loads strict into a model built with the flags off, and every existing path
(convfsenet/export_onnx.py, convfsenet/streaming.py, calibration / quant)
takes it unchanged. The TCM blocks' own BNs are left as they are -- those
paths already fold them.

Library:
    fold_state_dict(model)          -> plain state_dict (float64 math, cast back)
    plain_config(h)                 -> config with the block-design keys removed
    fold_model(model, h)            -> plain ConvFSENet_QuantFriendly_TD, eval()
    fold_gate(ref, folded, stft)    -> max relative differences of mask / stft_pred
    gate_report(model, h, stfts)    -> literal fp32 / exact fp64 / fp32-floor views of the fold

CLI (writes a checkpoint dir the existing exporters read):
    python -m convfsenet.fold --checkpoint_file cp_pilot_cf_B1/g_best \\
        --output_dir cp_pilot_cf_B1_folded [--gate_utts 20]
    python -m convfsenet.export_onnx --checkpoint_file cp_pilot_cf_B1_folded/g_best
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch
from torch import nn

from common.env import AttrDict
from convfsenet.model import ConvFSENet, build_causal_model

# Config keys that change the model's modules / forward. warmup_steps is a
# trainer knob and has no effect on the model, so it stays in the config.
BLOCK_MODEL_KEYS = ("frontend_norm", "input_norm")


def plain_config(h) -> AttrDict:
    """Copy of the config with the block-design model keys removed (flags off)."""
    return AttrDict({k: copy.deepcopy(v) for k, v in dict(h).items() if k not in BLOCK_MODEL_KEYS})


def is_plain(model: ConvFSENet) -> bool:
    return getattr(model, "frontend_norm", None) is None and not getattr(model, "input_centering", False)


@torch.no_grad()
def fold_state_dict(model: ConvFSENet) -> dict:
    """The plain-model state_dict equivalent to `model` (in eval mode).

    The frontend conv absorbs the input centring and the frontend BN's running
    statistics; every other tensor is copied unchanged. Math in float64, stored
    back in the frontend conv's dtype.
    """
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    conv = model.frontend[0]
    assert isinstance(conv, nn.Conv1d) and conv.kernel_size == (1,), "frontend.0 must be a 1x1 Conv1d"
    dtype = conv.weight.dtype
    W = conv.weight.detach().double()[:, :, 0]                              # (C, F)
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(W.shape[0], dtype=torch.float64,
                                                                               device=W.device)
    mu = model.input_mean.detach().double() if getattr(model, "input_centering", False) else None

    # centring: W (x - mu) + b = W x + (b - W mu)
    if mu is not None:
        b = b - W @ mu
        del sd["input_mean"]

    if getattr(model, "frontend_norm", None) == "batch":
        bn = model.frontend[1]
        assert isinstance(bn, nn.BatchNorm1d) and bn.track_running_stats, "frontend.1 must be a BatchNorm1d"
        g = bn.weight.detach().double() if bn.affine else torch.ones_like(b)
        beta = bn.bias.detach().double() if bn.affine else torch.zeros_like(b)
        s = torch.sqrt(bn.running_var.detach().double() + bn.eps)
        scale = g / s
        W = scale[:, None] * W
        b = scale * (b - bn.running_mean.detach().double()) + beta
        for k in [k for k in sd if k.startswith("frontend.1.")]:
            del sd[k]
    sd["frontend.0.weight"] = W.to(dtype).unsqueeze(-1).contiguous()
    sd["frontend.0.bias"] = b.to(dtype).contiguous()
    return sd


def fold_model(model: ConvFSENet, h) -> ConvFSENet:
    """Build the plain (flags-off) model from `h` and load the folded weights strict."""
    was_training = model.training
    model.eval()
    try:
        sd = fold_state_dict(model)
    finally:
        model.train(was_training)
    ref_param = next(model.parameters())
    # Cast BEFORE loading: load_state_dict copies into the existing tensors, so
    # a float32 plain model would round a float64 fold.
    plain = build_causal_model(plain_config(h)).to(device=ref_param.device, dtype=ref_param.dtype)
    plain.load_state_dict(sd, strict=True)
    return plain.eval()


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| / max |b| (complex-safe)."""
    return float((a - b).abs().max() / b.abs().max().clamp_min(torch.finfo(torch.float64).tiny))


def _mask(m: ConvFSENet, s: torch.Tensor) -> torch.Tensor:
    return m.backend(m.tcm(m.frontend(m.frontend_features(s))))


@torch.no_grad()
def fold_gate(ref: ConvFSENet, folded: ConvFSENet, stft: torch.Tensor) -> dict:
    """Compare the two models' masks and masked STFTs on `stft` ((B, [1,] F, T) complex)."""
    ref.eval(); folded.eval()
    s = stft.squeeze(1) if stft.dim() == 4 else stft
    m_ref, m_fold = _mask(ref, s), _mask(folded, s)
    y_ref, y_fold = ref(s.unsqueeze(1)), folded(s.unsqueeze(1))
    return {
        "mask_max_rel": _max_rel(m_fold, m_ref),
        "mask_max_abs": float((m_fold - m_ref).abs().max()),
        "stft_max_rel": _max_rel(y_fold, y_ref),
    }


@torch.no_grad()
def gate_report(model: ConvFSENet, h, stfts, tol: float = 1e-5, exact_tol: float = 1e-10) -> dict:
    """Three views of the fold on a list of complex STFTs, worst case over all of them.

    literal_fp32 : max relative mask / stft difference, folded vs unfolded, both fp32  (<= tol)
    exact_fp64   : the same in float64 -- the fold's own error, rounding removed       (<= exact_tol)
    floor_fp32   : distance of each fp32 model's mask to the fp64 truth. A trained
                   network amplifies fp32 rounding (the unfolded fp32 model is itself
                   ~1e-5..1e-4 off its fp64 truth), so the literal fp32 difference can
                   exceed tol with nothing wrong in the fold; this passes when the folded
                   fp32 model is no further from the truth than 2x the unfolded one.
    """
    model = model.eval()
    folded = fold_model(model, h)
    m64 = copy.deepcopy(model).double().eval()
    f64 = fold_model(m64, h)
    w = {"literal_fp32_mask_max_rel": 0.0, "literal_fp32_stft_max_rel": 0.0,
         "exact_fp64_mask_max_rel": 0.0, "exact_fp64_stft_max_rel": 0.0,
         "unfolded_fp32_vs_fp64_truth": 0.0, "folded_fp32_vs_fp64_truth": 0.0}
    for stft in stfts:
        s32 = (stft.squeeze(1) if stft.dim() == 4 else stft).to(torch.complex64)
        s64 = s32.to(torch.complex128)
        r32, r64 = fold_gate(model, folded, s32), fold_gate(m64, f64, s64)
        truth = _mask(m64, s64)
        vals = {"literal_fp32_mask_max_rel": r32["mask_max_rel"], "literal_fp32_stft_max_rel": r32["stft_max_rel"],
                "exact_fp64_mask_max_rel": r64["mask_max_rel"], "exact_fp64_stft_max_rel": r64["stft_max_rel"],
                "unfolded_fp32_vs_fp64_truth": _max_rel(_mask(model, s32).double(), truth),
                "folded_fp32_vs_fp64_truth": _max_rel(_mask(folded, s32).double(), truth)}
        w = {k: max(w[k], vals[k]) for k in w}
    w["pass_literal_fp32"] = max(w["literal_fp32_mask_max_rel"], w["literal_fp32_stft_max_rel"]) <= tol
    w["pass_exact_fp64"] = max(w["exact_fp64_mask_max_rel"], w["exact_fp64_stft_max_rel"]) <= exact_tol
    w["pass_floor_fp32"] = w["folded_fp32_vs_fp64_truth"] <= max(tol, 2.0 * w["unfolded_fp32_vs_fp64_truth"])
    return w


def _load_block_model(checkpoint_file, device):
    cfg = os.path.join(os.path.dirname(os.path.abspath(checkpoint_file)), "config.json")
    with open(cfg) as f:
        h = AttrDict(json.load(f))
    model = build_causal_model(h)
    state = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model.load_state_dict(state["generator"], strict=True)
    return model.to(device).eval(), h


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fold a block-design ConvFSENet checkpoint into a plain one.")
    ap.add_argument("--checkpoint_file", required=True, help="g_best-style checkpoint; sibling config.json is read")
    ap.add_argument("--output_dir", required=True, help="gets config.json (flags off) + g_best (folded)")
    ap.add_argument("--gate_utts", type=int, default=20, help="real VBD test utterances for the gate (0 = random only)")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--strict_fp32", action="store_true",
                    help="require the literal fp32 difference <= --tol (no fp32-floor allowance)")
    ap.add_argument("--hf_cache_dir", default=None)
    a = ap.parse_args(argv)

    device = torch.device("cpu")
    model, h = _load_block_model(a.checkpoint_file, device)
    folded = fold_model(model, h)

    torch.manual_seed(0)
    stfts = [model.preproc(torch.randn(4, 1, 32000))]
    if a.gate_utts > 0:
        from common.dataset import Dataset, load_voicebank_demand
        hf = load_voicebank_demand(cache_dir=a.hf_cache_dir)
        ds = Dataset(hf["test"], h.segment_size, h.sampling_rate, split=False, shuffle=False, seed=h.seed)
        stfts += [model.preproc(ds[i][1].view(1, 1, -1)) for i in range(min(a.gate_utts, len(ds)))]
    rep = gate_report(model, h, stfts, tol=a.tol)
    print(f"fold gate over random audio + {a.gate_utts} VBD test utterances:")
    for k, v in rep.items():
        print(f"  {k:<32} {v}")
    ok = rep["pass_exact_fp64"] and (rep["pass_literal_fp32"] or (rep["pass_floor_fp32"] and not a.strict_fp32))
    if not ok:
        print(f"FOLD GATE FAILED (tol {a.tol}{', --strict_fp32' if a.strict_fp32 else ''})", file=sys.stderr)
        return 1
    if not rep["pass_literal_fp32"]:
        print(f"note: literal fp32 difference exceeds {a.tol}, but the fold is exact in fp64 and the folded "
              f"fp32 model is within the unfolded model's own fp32 error of the fp64 truth")

    os.makedirs(a.output_dir, exist_ok=True)
    hp = plain_config(h)
    hp["folded_from"] = os.path.abspath(a.checkpoint_file)
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump(dict(hp), f, indent=4)
    torch.save({"generator": folded.state_dict()}, os.path.join(a.output_dir, "g_best"))
    print(f"wrote {a.output_dir}/config.json + g_best (plain model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
