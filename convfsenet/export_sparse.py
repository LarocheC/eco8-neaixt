"""Export ConvFSENet weight matrices + fixed sparsity masks for an external kernel.

The ConvFSENet counterpart of ``nsnet2/export_sparse.py`` (same hand-off format:
``weights.npz`` + ``manifest.json``, ``<layer>.pattern_index`` for codebook
patterns, ``--reference`` golden vectors).

Which layers are matrices. Every 1x1 ``Conv1d`` of the model -- ``frontend.0``,
``backend.0`` and each TCM block's ``conv1x1`` / ``conv1x1_out`` -- is a MatMul:
for weight ``(C_out, C_in, 1)`` the kernel sees ``W = weight[:, :, 0]`` of shape
``(M, K) = (C_out, C_in)``, row-major, applied per frame as ``y[:, t] = W @ x[:, t]
+ b`` with ``x`` of shape ``(K, N)`` (channels x frames; N = 1 when streaming).
The depthwise ``dconv`` (k=3, groups=C) and the BatchNorms are elementwise
per-channel ops, not MatMuls, and are not exported (they stay in the PyTorch
checkpoint).

Masks. The checkpoint is expected to already carry the sparsity (explicit
zeros); the mask is ``weight != 0`` and is verified against ``--pattern``.
``--prune`` instead builds a magnitude mask for ``--pattern`` here.

Usage
-----
    python -m convfsenet.export_sparse --config RUN/config.json \
        --checkpoint RUN/g_best --pattern 2:4@c1 --out export_cf --reference
"""

from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import torch
import torch.nn as nn

from common.env import AttrDict
from convfsenet.model import build_causal_model
from nsnet2.sparsity import build_mask, parse_pattern, pattern_indices, tail_elements, verify_pattern


def collect_matrices(model: nn.Module) -> list[tuple[str, torch.Tensor, torch.Tensor | None]]:
    """``(name, W (C_out, C_in), bias)`` for every 1x1 Conv1d, in module order."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv1d) and m.kernel_size == (1,) and m.groups == 1:
            b = None if m.bias is None else m.bias.detach()
            out.append((f"{name}.weight", m.weight.detach()[:, :, 0], b))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="plain (folded) ConvFSENet config.json")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="export_sparse_cf")
    ap.add_argument("--pattern", default="dense")
    ap.add_argument("--prune", action="store_true", help="build a magnitude mask for --pattern here")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--reference", action="store_true",
                    help="per-matrix golden vectors ref_x (K,), ref_y = W @ ref_x + b")
    a = ap.parse_args(argv)

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    model = build_causal_model(h)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["generator"], strict=True)
    model.eval()
    pattern = a.pattern

    manifest = {
        "model": "convfsenet",
        "config": os.path.abspath(a.config),
        "checkpoint": os.path.abspath(a.checkpoint),
        "pattern": pattern,
        "group_axis": "in",
        "group_axis_meaning": ("groups run along the input/K axis, i.e. contiguous "
                               "within a row of the row-major (M, K) weight"),
        "tail_policy": "keep",
        "dtype": "float32",
        "layout": ("row-major (M, K) = (C_out, C_in) = Conv1d weight[:, :, 0] of a 1x1 conv; "
                   "y[:, t] = W @ x[:, t] + b per frame, x of shape (K, N) = (channels, frames)"),
        "not_exported": ("tcm.*.dconv (depthwise k=3 causal conv) and tcm.*.norm1/norm2 (BatchNorm, "
                         "per-channel affine at inference) are elementwise, not MatMuls; they are in g_best"),
        "N_inference": 1,
        "N_training": h.get("batch_size", 16),
        "matrices": [],
    }
    desc = parse_pattern(pattern) if pattern != "dense" else None
    if desc and desc["family"] == "codebook":
        manifest["codebook"] = {
            "name": desc["codebook"], "group": desc["group"], "nonzeros_per_group": desc["n"],
            "patterns": ["".join(str(b) for b in pat) for pat in desc["allowed"]],
            "pairing": desc["pairing"],
            "bits_per_group": max(1, (len(desc["allowed"]) - 1).bit_length()),
            "index_semantics": ("<layer>.pattern_index[r, g] indexes codebook.patterns "
                                "for group g of row r; pattern bit k == 1 means column "
                                "g * group + k is kept"),
        }

    arrays: dict[str, np.ndarray] = {}
    violations = []
    for name, w, b in collect_matrices(model):
        if pattern == "dense":
            mask = torch.ones_like(w)
        elif a.prune:
            mask = build_mask(w, pattern)
        else:
            mask = (w != 0).to(w.dtype)
        wm = w * mask
        check = verify_pattern(wm, pattern) if pattern != "dense" else {"ok": True, "violations": 0}
        if not check["ok"]:
            violations.append((name, check["violations"]))
        arrays[f"{name}.weight"] = wm.numpy().astype(np.float32)
        arrays[f"{name}.mask"] = mask.numpy().astype(np.uint8)
        if desc and desc["family"] == "codebook" and check["ok"]:
            arrays[f"{name}.pattern_index"] = pattern_indices(wm, pattern).numpy().astype(np.uint8)
        if b is not None:
            arrays[f"{name}.bias"] = b.numpy().astype(np.float32)
        entry = {"name": name, "M": int(w.shape[0]), "K": int(w.shape[1]), "pattern": pattern,
                 "nonzero": int(mask.sum().item()), "sparsity": round(1.0 - mask.mean().item(), 6),
                 "tail_elements": tail_elements(w, pattern) if pattern != "dense" else 0,
                 "has_bias": b is not None, "pattern_verified": check["ok"]}
        if f"{name}.pattern_index" in arrays:
            entry["pattern_index_shape"] = list(arrays[f"{name}.pattern_index"].shape)
        manifest["matrices"].append(entry)

    if violations:
        for name, n in violations:
            print(f"PATTERN VIOLATION  {name}: {n} groups")
        raise SystemExit("refusing to export: the weights do not obey the declared pattern")
    manifest["pattern_verified"] = True

    if a.reference:
        for entry in manifest["matrices"]:
            name = entry["name"]
            w = arrays[f"{name}.weight"]
            x = np.random.default_rng(zlib.crc32(name.encode())).standard_normal(entry["K"]).astype(np.float32)
            y = w @ x
            if f"{name}.bias" in arrays:
                y = y + arrays[f"{name}.bias"]
            arrays[f"{name}.ref_x"] = x
            arrays[f"{name}.ref_y"] = y.astype(np.float32)
        manifest["reference_vectors"] = {
            "present": True, "definition": "ref_y = weight @ ref_x + bias, float32, N=1",
            "x_seed": "numpy default_rng(zlib.crc32(matrix_name)), standard_normal"}

    os.makedirs(a.out, exist_ok=True)
    npz_path = os.path.join(a.out, "weights.npz")
    (np.savez_compressed if a.compress else np.savez)(npz_path, **arrays)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(m["M"] * m["K"] for m in manifest["matrices"])
    nz = sum(m["nonzero"] for m in manifest["matrices"])
    print(f"wrote {npz_path} ({len(manifest['matrices'])} matrices, {total:,} weights, "
          f"{100 * (1 - nz / total):.1f}% zeros)")


if __name__ == "__main__":
    main()
