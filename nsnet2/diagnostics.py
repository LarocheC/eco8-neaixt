"""Conditioning and range diagnostics for structured (butterfly) layers.

Why this exists: the rank collapse in these models is present **at
initialisation**, not caused by training. Measured 2026-10-03 on the stage-1
checkpoints -- a 512-point butterfly with ``torch_structured``'s randn twiddles
composes to condition number ~2.6e10 and 99%-energy rank ~94 of 512, while an
orthogonally-initialised recurrent matrix starts at condition number 1.0 and
drifts only to 25-51 over a full run. Training *raises* rank from a bad start.
So the composed spectrum has to be measured at step 0 and tracked, not
inspected post hoc.

Everything here is implementation-agnostic: a layer's effective linear map is
recovered by probing it with basis vectors rather than by multiplying its
factors. That works identically for ``StageButterfly``, torch_structured's
``Butterfly``, ``nn.Linear``, Monarch and anything else, which matters because
the study compares them against each other.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def compose(module, in_features, device=None, probe_batch=256):
    """Recover a module's effective linear map as an (out, in) matrix.

    Probes with the identity and subtracts the zero-input response, so any bias
    or constant seed term is removed and only the linear part remains.
    """
    dev = device if device is not None else next(module.parameters()).device
    was_training = module.training
    module.eval()
    try:
        const = module(torch.zeros(1, in_features, device=dev))
        cols = []
        for i in range(0, in_features, probe_batch):
            e = torch.zeros(min(probe_batch, in_features - i), in_features, device=dev)
            e[torch.arange(e.shape[0]), torch.arange(i, i + e.shape[0])] = 1.0
            cols.append((module(e) - const).double().cpu())
    finally:
        module.train(was_training)
    return torch.cat(cols, dim=0).T.contiguous()


@torch.no_grad()
def spectrum(mat, energy=0.99):
    """Condition number and energy-rank of a composed matrix."""
    s = torch.linalg.svdvals(mat.double())
    smax = s[0].item()
    smin = s[-1].item()
    sq = s ** 2
    frac = torch.cumsum(sq, 0) / sq.sum()
    rank = int(torch.searchsorted(frac, torch.tensor(energy, dtype=frac.dtype)).item()) + 1
    return {
        "cond": (smax / smin) if smin > 0 else float("inf"),
        "rank{:g}".format(energy * 100): min(rank, s.numel()),
        "dim": int(s.numel()),
        "smax": smax,
        "smin": smin,
    }


def _probe_width(module):
    """The input width to probe a module with, or None if it is not a linear map."""
    for attr in ("in_size", "in_features"):
        w = getattr(module, attr, None)
        if isinstance(w, int):
            return w
    return None


@torch.no_grad()
def layer_spectra(model, include=(), max_dim=2048):
    """Composed spectrum of every linear-map submodule of ``model``.

    ``include`` optionally restricts to modules whose qualified name contains
    one of the given substrings. Modules wider than ``max_dim`` are skipped:
    probing costs one forward pass per input dimension.
    """
    out = {}
    for name, mod in model.named_modules():
        if mod is model or any(mod_ is mod for mod_ in ()):
            continue
        width = _probe_width(mod)
        if width is None or width > max_dim:
            continue
        if include and not any(tok in name for tok in include):
            continue
        if not any(p.requires_grad for p in mod.parameters(recurse=False)):
            continue
        try:
            out[name] = spectrum(compose(mod, width))
        except Exception as exc:                      # a non-linear or odd-signature module
            out[name] = {"error": "{}: {}".format(type(exc).__name__, exc)}
    return out


class ActivationRanges:
    """Collect per-module activation ranges with forward hooks.

    The quantisation phase needs these, and they are free to collect during the
    fp32 phase -- which is the point: record them now so the later phase does
    not have to re-run every arm. ``absmax`` is the number that decides the
    fixed-point range; ``clip_frac`` is reported against a provisional range so
    a layer that would saturate is visible before anyone quantises anything.
    """

    def __init__(self, model, include=(), provisional_range=None):
        self.stats = {}
        self._handles = []
        self.provisional = provisional_range
        for name, mod in model.named_modules():
            if include and not any(tok in name for tok in include):
                continue
            if _probe_width(mod) is None:
                continue
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_mod, _inp, out):
            if not isinstance(out, torch.Tensor):
                return
            with torch.no_grad():
                a = out.detach()
                s = self.stats.setdefault(name, {"absmax": 0.0, "n": 0, "clipped": 0})
                s["absmax"] = max(s["absmax"], a.abs().max().item())
                s["n"] += a.numel()
                if self.provisional is not None:
                    s["clipped"] += int((a.abs() > self.provisional).sum().item())
        return hook

    def report(self):
        out = {}
        for k, s in self.stats.items():
            r = {"absmax": s["absmax"]}
            if self.provisional is not None and s["n"]:
                r["clip_frac"] = s["clipped"] / s["n"]
            out[k] = r
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
