"""Export NSNet2 weight matrices + fixed sparsity masks for an external kernel.

This is the hand-off format for the Row-Fusion / sparse-dense MatMul
collaboration. It deliberately does *not* go through ONNX: the compiler side
only needs the matrices, the masks, and enough metadata to know how each one is
multiplied at inference time.

Two modes:

  --dims-only    print the GEMM shape table (no checkpoint needed). Every
                 matrix is listed as ``y = W @ x`` with W of shape (M, K) and
                 x of shape (K, N); N is 1 for streaming inference and
                 h.batch_size * T for training.

  (default)      write ``<out>/weights.npz`` + ``<out>/manifest.json``:
                   weights.npz : ``<layer>.weight`` (dense, explicit zeros),
                                 ``<layer>.mask`` (uint8), ``<layer>.bias``
                   manifest    : per-matrix shape, pattern, group axis,
                                 achieved sparsity, ragged-tail count, dtype,
                                 and the inference/training N.

Usage
-----
    python -m nsnet2.export_sparse --config configs/sparse_2to4.json --dims-only
    python -m nsnet2.export_sparse --config cp_nsnet2_sparse24/config.json \
        --checkpoint cp_nsnet2_sparse24/g_best --out export_sparse_2to4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from common.env import AttrDict
from common.utils import load_checkpoint
from nsnet2.model import NSNet2
from nsnet2.sparsity import (MaskedLinear, build_mask, parse_pattern,
                             tail_elements, verify_pattern)


# Which 2-D parameters are the MatMuls that matter, and a stable human name.
def collect_matrices(model: torch.nn.Module) -> list[tuple[str, torch.Tensor, torch.Tensor | None]]:
    """Return ``(name, weight, bias)`` for every 2-D weight in the model."""
    out = []
    params = dict(model.named_parameters())
    for name, p in params.items():
        if p.dim() != 2:
            continue
        bias_name = name.replace("weight", "bias")
        bias = params.get(bias_name)
        if bias is None and name.startswith("gru."):
            # nn.GRU biases are bias_ih_l0 / bias_hh_l0 next to weight_ih_l0.
            bias = params.get(name.replace("weight_", "bias_"))
        out.append((name, p.detach(), None if bias is None else bias.detach()))
    return out


def dims_table(h: AttrDict) -> list[dict]:
    """GEMM shapes for the configured NSNet2, without instantiating weights."""
    n_freq = h.n_fft // 2 + 1
    hidden = h.get("hidden_dim", 400)
    fc_hidden = h.get("fc_hidden_dim", 600)
    layers = h.get("num_gru_layers", 2)

    rows = [{"name": "fc_in", "M": hidden, "K": n_freq, "count": 1}]
    for li in range(layers):
        in_size = hidden if li == 0 else hidden
        rows.append({"name": f"gru.weight_ih_l{li}", "M": 3 * hidden, "K": in_size,
                     "count": 1})
        rows.append({"name": f"gru.weight_hh_l{li}", "M": 3 * hidden, "K": hidden,
                     "count": 1})
    rows += [
        {"name": "fc1", "M": fc_hidden, "K": hidden, "count": 1},
        {"name": "fc2", "M": fc_hidden, "K": fc_hidden, "count": 1},
        {"name": "fc_out", "M": n_freq, "K": fc_hidden, "count": 1},
    ]
    for r in rows:
        r["params"] = r["M"] * r["K"]
    return rows


def print_dims(h: AttrDict, pattern: str | None) -> None:
    rows = dims_table(h)
    total = sum(r["params"] for r in rows)
    desc = parse_pattern(pattern) if pattern else None

    print(f"NSNet2 GEMM shapes   y = W @ x,  W:(M,K)  x:(K,N)")
    print(f"  N = 1                       (streaming inference, per frame)")
    print(f"  N = {h.get('batch_size', 256)} * T                 (training)")
    if desc:
        print(f"  pattern = {desc['pattern']}  (nominal {100 * desc['sparsity']:.0f}% sparse, "
              f"groups along K)")
    print()
    hdr = f"{'matrix':<22}{'M':>8}{'K':>8}{'params':>12}"
    if desc:
        hdr += f"{'K % group':>12}"
    print(hdr)
    print("-" * len(hdr))
    group = (desc or {}).get("group") or (desc or {}).get("block")
    for r in rows:
        line = f"{r['name']:<22}{r['M']:>8}{r['K']:>8}{r['params']:>12,}"
        if desc:
            line += f"{(r['K'] % group if group else 0):>12}"
        print(line)
    print("-" * len(hdr))
    print(f"{'total':<22}{'':>8}{'':>8}{total:>12,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="",
                    help="g_* checkpoint. Omitted => random init (shapes/masks only).")
    ap.add_argument("--out", default="export_sparse")
    ap.add_argument("--pattern", default="",
                    help="Override the config's sparsity pattern (e.g. '2:4').")
    ap.add_argument("--dims-only", action="store_true",
                    help="Print the GEMM shape table and exit.")
    a = ap.parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))

    sp_cfg = dict(h.get("sparsity", None) or {})
    pattern = a.pattern or sp_cfg.get("pattern", "")

    if a.dims_only:
        print_dims(h, pattern or None)
        return

    model = NSNet2(h)
    if a.checkpoint:
        model.load_state_dict(load_checkpoint(a.checkpoint, torch.device("cpu"))["generator"])
    model.eval()

    axis = sp_cfg.get("axis", "in")
    tail = sp_cfg.get("tail", "keep")
    scope = sp_cfg.get("scope", "matrix")

    # Prefer the masks the model already carries. MaskedLinear stores its own;
    # for dense parameters trained under SparsityController the mask is exactly
    # "where the weight is nonzero" (apply() re-zeroes after every step), but if
    # the checkpoint was never trained sparse we build one from magnitude.
    masked_linears = {name: m for name, m in model.named_modules()
                      if isinstance(m, MaskedLinear)}
    trained_sparse = bool(sp_cfg.get("enabled", False)) and bool(a.checkpoint)

    os.makedirs(a.out, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    violations: list[tuple[str, int, int]] = []
    manifest = {
        "model": "nsnet2",
        "config": os.path.abspath(a.config),
        "checkpoint": os.path.abspath(a.checkpoint) if a.checkpoint else None,
        "pattern": pattern or "dense",
        "group_axis": axis,
        "group_axis_meaning": ("groups run along the input/K axis, i.e. contiguous "
                               "within a row of the row-major (M, K) weight"),
        "tail_policy": tail,
        "block_scope": scope,
        "dtype": "float32",
        "layout": "row-major (M, K); y = W @ x with x of shape (K, N)",
        "N_inference": 1,
        "N_training": h.get("batch_size", 256),
        "matrices": [],
    }

    for name, w, b in collect_matrices(model):
        owner = name.rsplit(".weight", 1)[0]
        if owner in masked_linears:
            mask = masked_linears[owner].mask.detach()
            pat = masked_linears[owner].pattern
        elif trained_sparse:
            mask = (w != 0).to(w.dtype)
            pat = pattern
        elif pattern:
            mask = build_mask(w, pattern, axis=axis, tail=tail, scope=scope)
            pat = pattern
        else:
            mask = torch.ones_like(w)
            pat = "dense"

        w_masked = w * mask
        # The mask is a contract with the generated kernel: verify the weights
        # we are about to ship actually obey it, rather than trusting the mask
        # that built them.
        check = verify_pattern(w_masked, pat, axis=axis, tail=tail)
        if not check["ok"]:
            violations.append((name, check["violations"], check["groups"]))
        arrays[f"{name}.weight"] = w_masked.numpy().astype(np.float32)
        arrays[f"{name}.mask"] = mask.numpy().astype(np.uint8)
        if b is not None:
            arrays[f"{name}.bias"] = b.numpy().astype(np.float32)

        manifest["matrices"].append({
            "name": name,
            "M": int(w.shape[0]),
            "K": int(w.shape[1]),
            "pattern": pat,
            "nonzero": int(mask.sum().item()),
            "sparsity": round(1.0 - mask.mean().item(), 6),
            "tail_elements": tail_elements(w, pat, axis=axis) if pat != "dense" else 0,
            "has_bias": b is not None,
            "pattern_verified": check["ok"],
        })

    if violations:
        for name, n, groups in violations:
            print(f"PATTERN VIOLATION  {name}: {n}/{groups} groups exceed {pattern}")
        raise SystemExit(
            "refusing to export: the weights do not obey the declared pattern. "
            "Re-check that the checkpoint was trained with the matching sparsity "
            "config (or pass --pattern to prune it here)."
        )
    manifest["pattern_verified"] = True

    npz_path = os.path.join(a.out, "weights.npz")
    np.savez(npz_path, **arrays)
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(m["M"] * m["K"] for m in manifest["matrices"])
    nz = sum(m["nonzero"] for m in manifest["matrices"])
    print(f"wrote {npz_path} ({len(manifest['matrices'])} matrices, "
          f"{total:,} weights, {100 * (1 - nz / total):.1f}% zeros)")
    print(f"wrote {os.path.join(a.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
