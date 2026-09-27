"""Fold the block-design options of NSNet2 back into a plain NSNet2.

A model trained with ``input_norm`` (frozen per-bin input mean ``mu``) and/or
``block_norm: "batch"`` (BatchNorm after fc_in/fc1/fc2, before the ReLU) is,
in eval mode, an affine re-parameterisation of the old-style network. For a
hidden layer

    y = BN(W (x - mu) + b),   BN(z) = g (z - m) / s + beta,   s = sqrt(var + eps)

is exactly ``y = W' x + b'`` with

    W' = diag(g / s) W
    b' = (g / s) (b - W mu - m) + beta

(``mu`` only on fc_in; ``g/s = 1, m = beta = 0`` where a layer has no BN).
``fold_for_export`` returns a flags-off ``NSNet2`` holding those weights, so
it goes through the existing export paths (``nsnet2.export_onnx`` etc.)
unchanged, and its state_dict loads strictly into NSNet2 built from the
config with ``input_norm`` / ``block_norm`` removed.

The fold is computed in float64 and cast back to the parameter dtype.

CLI::

    python -m nsnet2.fold --checkpoint_file cp_run/g_best --output_dir cp_run_folded

writes ``<output_dir>/g_best`` ({'generator': folded state_dict}) plus the
stripped ``config.json`` next to it, i.e. a directory every existing tool
(``export_onnx --checkpoint_file <output_dir>/g_best``, inference, eval)
already understands.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from common.env import AttrDict
from nsnet2.model import NSNet2
from nsnet2.sparsity import MaskedLinear

BLOCK_KEYS = ("input_norm", "block_norm")


def strip_block_config(h) -> AttrDict:
    """Copy of config ``h`` with the block-design keys removed."""
    h2 = AttrDict(copy.deepcopy(dict(h)))
    for k in BLOCK_KEYS:
        h2.pop(k, None)
    return h2


def _dense_weight(layer: nn.Module) -> torch.Tensor:
    if isinstance(layer, (nn.Linear, MaskedLinear)):
        return layer.weight
    raise NotImplementedError(
        f"fold supports dense/masked linears only, got {type(layer).__name__}")


@torch.no_grad()
def fold_affine(layer: nn.Module, bn: nn.BatchNorm1d | None,
                mu: torch.Tensor | None):
    """Return (W', b') for ``BN(layer(x - mu))`` as float64 tensors."""
    W = _dense_weight(layer).double()
    out_f = W.shape[0]
    b = (layer.bias.double() if layer.bias is not None
         else torch.zeros(out_f, dtype=torch.float64, device=W.device))
    if mu is not None:
        b = b - W @ mu.double()
    if bn is None:
        return W, b
    if bn.running_mean is None or not bn.affine:
        raise NotImplementedError("fold needs BatchNorm with running stats and affine params")
    s = torch.sqrt(bn.running_var.double() + bn.eps)
    g = bn.weight.double()
    scale = g / s
    W2 = scale[:, None] * W
    b2 = scale * (b - bn.running_mean.double()) + bn.bias.double()
    return W2, b2


@torch.no_grad()
def fold_for_export(model: NSNet2):
    """Fold ``model`` (block-design NSNet2) into an equivalent plain NSNet2.

    Returns ``(plain_model, state_dict)``. ``plain_model`` is in eval mode on
    the same device as ``model``; ``state_dict`` is its state_dict.
    A model with both flags off is returned as a copy (nothing to fold).
    """
    h_plain = strip_block_config(model.h)
    ref = next(model.parameters())
    # Match device AND dtype before loading: load_state_dict copies into the
    # existing tensors, so a float32 plain model would round a float64 fold.
    plain = NSNet2(h_plain).to(device=ref.device, dtype=ref.dtype)
    sd = {k: v.clone() for k, v in model.state_dict().items()
          if not (k == "in_mean" or k.split(".")[0] in ("bn_in", "bn1", "bn2"))}

    mu = model.in_mean if getattr(model, "input_norm", False) else None
    bnorm = getattr(model, "block_norm", False)
    plan = [("fc_in", model.bn_in if bnorm else None, mu),
            ("fc1", model.bn1 if bnorm else None, None),
            ("fc2", model.bn2 if bnorm else None, None)]
    for name, bn, m in plan:
        layer = getattr(model, name)
        W2, b2 = fold_affine(layer, bn, m)
        w_key, b_key = f"{name}.weight", f"{name}.bias"
        sd[w_key] = W2.to(sd[w_key].dtype)
        if b_key in sd:
            sd[b_key] = b2.to(sd[b_key].dtype)
        elif (m is not None or bn is not None) and b2.abs().max() > 0:
            raise NotImplementedError(f"{name} has no bias to absorb the fold")
        if isinstance(layer, MaskedLinear):
            # Row scaling keeps pruned entries at exactly zero; re-assert it.
            sd[w_key] = sd[w_key] * sd[f"{name}.mask"]

    plain.load_state_dict(sd, strict=True)
    plain.eval()
    return plain, plain.state_dict()


def fold_checkpoint(checkpoint_file, output_dir, config_file=None):
    """Fold a block-design checkpoint into ``output_dir/<name>`` + config.json."""
    ckpt = Path(checkpoint_file)
    cfg_path = Path(config_file) if config_file else ckpt.parent / "config.json"
    with open(cfg_path) as f:
        h = AttrDict(json.load(f))
    model = NSNet2(h)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state["generator"], strict=True)
    model.eval()
    _, sd = fold_for_export(model)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": sd}, out / ckpt.name)
    with open(out / "config.json", "w") as f:
        json.dump(strip_block_config(h), f, indent=4)
    print(f"Folded {ckpt} -> {out / ckpt.name} (+ config.json without {BLOCK_KEYS})")
    return out / ckpt.name


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None,
                        help="config.json (default: sibling of the checkpoint)")
    a = parser.parse_args()
    fold_checkpoint(a.checkpoint_file, a.output_dir, a.config)


if __name__ == "__main__":
    main()
