"""Fixed structured-sparsity masks for NSNet2 weight matrices.

Motivation
----------
This exists for the Row-Fusion / sparse-dense MatMul collaboration: an
external compilation flow packs *fixed* sparse weight matrices and generates a
specialised inference kernel. The patterns it consumes today are the
semi-structured families used by hardware sparse units:

    * ``"2:4"``   — 2 nonzeros per contiguous group of 4  (50% sparse)
    * ``"4:8"``   — 4 nonzeros per contiguous group of 8  (50% sparse)
    * ``"1x4:80"``— blocks of 4 contiguous weights kept or dropped whole,
                    80% of the blocks dropped (20% density)
    * ``"unstructured:80"`` — plain magnitude pruning to 80% sparsity; kept as
                    a control, since arbitrary sparsity buys memory but not
                    necessarily kernel speed.

Nothing here accelerates *training*. The mask is a fixed binary tensor applied
to dense weights, so the forward/backward still run dense GEMMs on the GPU at
whatever batch size training uses. The point is that the resulting checkpoint
obeys the pattern *exactly*, so the deployment-side kernel is free to exploit
it at batch-1.

Grouping direction
------------------
For a weight ``W`` of shape ``(out, in)`` and streaming inference
``y = W @ x`` with ``x`` of shape ``(in, 1)``, the groups run along the **input
(K) axis** — i.e. contiguous along a row, which is contiguous in memory for a
row-major ``(out, in)`` tensor. That matches how 2:4 is defined on NVIDIA
sparse tensor cores. ``axis="out"`` groups down a column instead; it is
supported so the assumption can be A/B-tested rather than asserted.

Ragged tails
------------
``K`` is not always divisible by the group size (NSNet2's ``fc_in`` has
``K=257``, i.e. 64 groups of 4 plus one leftover column). The leftover columns
are left **dense** by default (``tail="keep"``) and counted in the report, so
the achieved sparsity is slightly below nominal and the exporter can state
exactly how many elements fall outside the pattern.

Usage
-----
Fine-tune an existing dense checkpoint under a fixed mask::

    ctrl = SparsityController.from_config(model, h.sparsity)   # magnitude init
    ctrl.apply()                                               # project to mask
    ...
    loss.backward(); optim.step(); ctrl.apply()                # re-zero each step
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "parse_pattern",
    "build_mask",
    "verify_pattern",
    "MaskedLinear",
    "SparsityController",
]


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

_NM_RE = re.compile(r"^(\d+):(\d+)$")
_BLOCK_RE = re.compile(r"^1x(\d+):(\d+(?:\.\d+)?)$")
_UNSTRUCT_RE = re.compile(r"^unstructured:(\d+(?:\.\d+)?)$")


def parse_pattern(pattern: str) -> dict:
    """Parse a pattern string into a descriptor dict.

    Returns a dict with ``family`` in ``{"nm", "block", "unstructured",
    "dense"}`` plus the family's parameters. ``sparsity`` is the *nominal*
    fraction of zeros the pattern implies (before any ragged tail).
    """
    p = pattern.strip().lower()

    if p in ("dense", "none", ""):
        return {"family": "dense", "pattern": "dense", "sparsity": 0.0}

    m = _NM_RE.match(p)
    if m:
        n, group = int(m.group(1)), int(m.group(2))
        if not 0 < n < group:
            raise ValueError(f"N:M pattern needs 0 < N < M, got {pattern!r}")
        return {"family": "nm", "pattern": p, "n": n, "group": group,
                "sparsity": 1.0 - n / group}

    m = _BLOCK_RE.match(p)
    if m:
        block, pct = int(m.group(1)), float(m.group(2))
        if not 0.0 <= pct < 100.0:
            raise ValueError(f"block sparsity must be in [0, 100), got {pattern!r}")
        return {"family": "block", "pattern": p, "block": block,
                "sparsity": pct / 100.0}

    m = _UNSTRUCT_RE.match(p)
    if m:
        pct = float(m.group(1))
        if not 0.0 <= pct < 100.0:
            raise ValueError(f"sparsity must be in [0, 100), got {pattern!r}")
        return {"family": "unstructured", "pattern": p, "sparsity": pct / 100.0}

    raise ValueError(
        f"Unknown sparsity pattern {pattern!r}; expected 'N:M' (e.g. '2:4'), "
        f"'1xB:PCT' (e.g. '1x4:80'), 'unstructured:PCT', or 'dense'."
    )


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def _as_km(weight: torch.Tensor, axis: str) -> torch.Tensor:
    """View the weight as ``(rows, K)`` with the grouped axis last."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    if axis == "in":
        return weight
    if axis == "out":
        return weight.t()
    raise ValueError(f"axis must be 'in' or 'out', got {axis!r}")


@torch.no_grad()
def build_mask(weight: torch.Tensor, pattern: str, *, axis: str = "in",
               tail: str = "keep", scope: str = "matrix") -> torch.Tensor:
    """Build a fixed binary mask for ``weight`` from weight magnitude.

    weight  : (out, in) float tensor — typically a trained dense checkpoint.
    pattern : see :func:`parse_pattern`.
    axis    : "in" groups along K (row-contiguous, the deploy-relevant one),
              "out" groups along the output dim.
    tail    : "keep" leaves a ragged remainder dense, "drop" zeros it.
    scope   : for the block family, rank blocks per "matrix" (global budget,
              default) or per "row" (equal work per row — friendlier to a
              kernel that wants balanced rows).

    Returns a mask of the same shape/dtype/device as ``weight``.
    """
    desc = parse_pattern(pattern)
    fam = desc["family"]
    w = _as_km(weight, axis)
    rows, K = w.shape
    mag = w.abs()

    if fam == "dense":
        mask = torch.ones_like(w)

    elif fam == "unstructured":
        mask = torch.ones_like(w)
        n_zero = int(round(desc["sparsity"] * w.numel()))
        if n_zero > 0:
            thresh_idx = mag.flatten().argsort()[:n_zero]
            mask.view(-1)[thresh_idx] = 0.0

    elif fam == "nm":
        n, group = desc["n"], desc["group"]
        n_groups = K // group
        mask = torch.ones_like(w)
        if n_groups > 0:
            head = mag[:, : n_groups * group].reshape(rows, n_groups, group)
            keep = head.argsort(dim=-1, descending=True)[..., :n]
            gmask = torch.zeros_like(head)
            gmask.scatter_(-1, keep, 1.0)
            mask[:, : n_groups * group] = gmask.reshape(rows, n_groups * group)
        if K % group and tail == "drop":
            mask[:, n_groups * group:] = 0.0

    elif fam == "block":
        block = desc["block"]
        n_blocks = K // block
        mask = torch.ones_like(w)
        if n_blocks > 0:
            head = mag[:, : n_blocks * block].reshape(rows, n_blocks, block)
            score = head.sum(dim=-1)                         # (rows, n_blocks)
            if scope == "row":
                n_drop = int(round(desc["sparsity"] * n_blocks))
                drop = score.argsort(dim=-1)[:, :n_drop]     # smallest per row
                bmask = torch.ones_like(score)
                bmask.scatter_(-1, drop, 0.0)
            elif scope == "matrix":
                total = rows * n_blocks
                n_drop = int(round(desc["sparsity"] * total))
                bmask = torch.ones_like(score)
                if n_drop > 0:
                    drop = score.flatten().argsort()[:n_drop]
                    bmask.view(-1)[drop] = 0.0
            else:
                raise ValueError(f"scope must be 'matrix' or 'row', got {scope!r}")
            mask[:, : n_blocks * block] = (
                bmask.unsqueeze(-1).expand(rows, n_blocks, block).reshape(rows, -1)
            )
        if K % block and tail == "drop":
            mask[:, n_blocks * block:] = 0.0

    else:  # pragma: no cover — parse_pattern already validated
        raise ValueError(f"unhandled family {fam!r}")

    return mask if axis == "in" else mask.t().contiguous()


@torch.no_grad()
def verify_pattern(weight: torch.Tensor, pattern: str, *, axis: str = "in",
                   tail: str = "keep") -> dict:
    """Check that ``weight``'s zero pattern actually obeys ``pattern``.

    The mask is a contract with the deployment-side kernel: if a group carries
    one more nonzero than the pattern allows, the generated kernel is wrong, not
    merely slow. This checks the saved weights directly rather than trusting the
    mask that produced them, so it catches a checkpoint that was resumed without
    its sparsity config, saved mid-step, or fine-tuned dense by accident.

    Returns ``{"ok", "violations", "groups", "nonzero", "sparsity",
    "sparsity_shortfall"}``. ``violations`` counts groups holding more nonzeros
    than the pattern permits; fewer is fine (a weight may be exactly zero by
    chance).

    N:M pins the density through the group constraint alone. The block and
    unstructured families do not — an all-dense weight trivially satisfies
    "every 1x4 block is kept or dropped whole" — so for those the achieved
    sparsity is checked against the target as well.
    """
    desc = parse_pattern(pattern)
    w = _as_km(weight, axis)
    rows, K = w.shape
    nz = (w != 0)
    result = {"ok": True, "violations": 0, "groups": 0,
              "nonzero": int(nz.sum().item()),
              "sparsity": 1.0 - nz.float().mean().item(),
              "sparsity_shortfall": 0.0}

    if desc["family"] == "nm":
        group, n = desc["group"], desc["n"]
        n_groups = K // group
        if n_groups:
            head = nz[:, : n_groups * group].reshape(rows, n_groups, group)
            counts = head.sum(dim=-1)
            result["groups"] = rows * n_groups
            result["violations"] = int((counts > n).sum().item())
    elif desc["family"] == "block":
        block = desc["block"]
        n_blocks = K // block
        if n_blocks:
            head = nz[:, : n_blocks * block].reshape(rows, n_blocks, block)
            counts = head.sum(dim=-1)
            result["groups"] = rows * n_blocks
            # a block must be entirely kept or entirely dropped
            result["violations"] = int(((counts != 0) & (counts != block)).sum().item())

    if tail == "drop" and desc.get("group", desc.get("block")):
        g = desc.get("group") or desc["block"]
        if K % g:
            result["violations"] += int(nz[:, (K // g) * g:].sum().item())

    if desc["family"] in ("block", "unstructured"):
        # A ragged tail left dense caps how sparse the matrix can get, so
        # compare against the reachable target rather than the nominal one.
        reachable = desc["sparsity"]
        g = desc.get("block")
        if g and tail == "keep" and K % g:
            reachable *= (K - K % g) / K
        result["sparsity_shortfall"] = max(0.0, reachable - result["sparsity"])

    result["ok"] = result["violations"] == 0 and result["sparsity_shortfall"] <= 0.02
    return result


def tail_elements(weight: torch.Tensor, pattern: str, *, axis: str = "in") -> int:
    """Number of weight elements that fall outside a whole group (ragged tail)."""
    desc = parse_pattern(pattern)
    group = desc.get("group") or desc.get("block")
    if group is None:
        return 0
    w = _as_km(weight, axis)
    rows, K = w.shape
    return rows * (K % group)


# ---------------------------------------------------------------------------
# Masked linear layer
# ---------------------------------------------------------------------------

class MaskedLinear(nn.Module):
    """``nn.Linear`` with a fixed binary mask on the weight.

    The mask is a buffer (so it travels in the state_dict and survives
    checkpoint round-trips) and is applied in the forward pass, which keeps the
    masked weights at exactly zero regardless of what the optimizer does to
    ``self.weight``. Pruned entries also receive zero gradient, since
    ``d(w*m)/dw = m``.

    Mask initialisation is deferred: constructed with an all-ones mask, then
    ``rebuild_mask()`` selects the surviving entries by magnitude — call it
    *after* loading a dense checkpoint so the selection sees trained weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *,
                 pattern: str = "2:4", axis: str = "in", tail: str = "keep",
                 scope: str = "matrix"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pattern = pattern
        self.axis = axis
        self.tail = tail
        self.scope = scope
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("mask", torch.ones(out_features, in_features))
        self._reset_parameters()
        self.rebuild_mask()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / self.in_features ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def rebuild_mask(self) -> None:
        self.mask.copy_(build_mask(self.weight, self.pattern, axis=self.axis,
                                   tail=self.tail, scope=self.scope))

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        density = self.mask.mean().item()
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, pattern={self.pattern!r}, "
                f"axis={self.axis!r}, density={density:.3f}")


# ---------------------------------------------------------------------------
# Whole-model controller
# ---------------------------------------------------------------------------

# nn.GRU keeps its weights as plain parameters (weight_ih_l0, weight_hh_l0, ...)
# and cuDNN wants them untouched, so masking them by wrapping the module would
# cost the fused kernel. Instead the controller re-zeros them after every
# optimizer step: same fixed-mask semantics, cuDNN training speed preserved.
_DEFAULT_EXCLUDE = ()


class SparsityController:
    """Applies fixed masks to selected 2-D parameters of a model.

    Masks are built once (from weight magnitude) and then held fixed.
    ``apply()`` re-zeros the masked entries; call it right after every
    ``optimizer.step()``. With ``mask_grads()`` the pruned entries already stay
    pinned (zero grad + zero weight => AdamW's update and its decoupled decay
    are both zero); ``apply()`` is the belt-and-braces projection that keeps the
    guarantee under any optimizer, so the checkpoint on disk always obeys the
    pattern exactly.

    ``MaskedLinear`` modules mask themselves in forward and are skipped here.
    """

    def __init__(self, model: nn.Module, pattern: str, *, axis: str = "in",
                 tail: str = "keep", scope: str = "matrix",
                 include: str = r".*", exclude: Iterable[str] = _DEFAULT_EXCLUDE,
                 min_numel: int = 4096):
        self.model = model
        self.pattern = pattern
        self.axis = axis
        self.tail = tail
        self.scope = scope
        self.include = re.compile(include)
        self.exclude = [re.compile(e) for e in exclude]
        self.min_numel = min_numel

        masked_linear_prefixes = tuple(
            name for name, m in model.named_modules() if isinstance(m, MaskedLinear)
        )

        self.masks: dict[str, torch.Tensor] = {}
        self._params: dict[str, nn.Parameter] = {}
        for name, p in model.named_parameters():
            if p.dim() != 2 or p.numel() < min_numel:
                continue
            if any(name.startswith(pre + ".") for pre in masked_linear_prefixes):
                continue
            if not self.include.search(name) or any(e.search(name) for e in self.exclude):
                continue
            self.masks[name] = build_mask(p.data, pattern, axis=axis, tail=tail,
                                          scope=scope)
            self._params[name] = p

    @classmethod
    def from_config(cls, model: nn.Module, cfg: Optional[dict]) -> Optional["SparsityController"]:
        """Build from a config block; returns ``None`` when disabled/absent."""
        cfg = dict(cfg or {})
        if not cfg.get("enabled", False):
            return None
        return cls(
            model,
            cfg.get("pattern", "2:4"),
            axis=cfg.get("axis", "in"),
            tail=cfg.get("tail", "keep"),
            scope=cfg.get("scope", "matrix"),
            include=cfg.get("include", r".*"),
            exclude=cfg.get("exclude", ()),
            min_numel=int(cfg.get("min_numel", 4096)),
        )

    @torch.no_grad()
    def rebuild(self) -> None:
        """Re-select the masks from the current weights (magnitude pruning)."""
        for name, p in self._params.items():
            self.masks[name] = build_mask(p.data, self.pattern, axis=self.axis,
                                          tail=self.tail, scope=self.scope)

    @torch.no_grad()
    def apply(self) -> None:
        for name, p in self._params.items():
            mask = self.masks[name]
            if mask.device != p.device:
                mask = mask.to(p.device)
                self.masks[name] = mask
            p.data.mul_(mask)

    @torch.no_grad()
    def mask_grads(self) -> None:
        """Optional: zero the gradient of pruned weights before the step.

        Not required for correctness (``apply()`` re-projects afterwards) but
        it keeps the optimizer's second-moment estimates clean.
        """
        for name, p in self._params.items():
            if p.grad is not None:
                p.grad.mul_(self.masks[name].to(p.grad.device))

    def report(self) -> list[dict]:
        """Per-matrix sparsity summary, ready to print or serialise."""
        rows = []
        for name, p in self._params.items():
            mask = self.masks[name]
            rows.append({
                "name": name,
                "shape": list(p.shape),
                "numel": p.numel(),
                "nonzero": int(mask.sum().item()),
                "sparsity": 1.0 - mask.mean().item(),
                "pattern": self.pattern,
                "axis": self.axis,
                "tail_elements": tail_elements(p.data, self.pattern, axis=self.axis),
            })
        return rows

    def __len__(self) -> int:
        return len(self.masks)

    def __repr__(self) -> str:
        total = sum(m.numel() for m in self.masks.values())
        nz = sum(int(m.sum().item()) for m in self.masks.values())
        frac = 1.0 - nz / total if total else 0.0
        return (f"SparsityController(pattern={self.pattern!r}, axis={self.axis!r}, "
                f"matrices={len(self.masks)}, overall_sparsity={frac:.3f})")
