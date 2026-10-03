"""Separate optimiser treatment for butterfly twiddles.

Two independent reasons the twiddles should not share the dense layers'
optimiser settings, both opt-in and both default to the previous behaviour.

**Weight decay.** AdamW applies decay (default 0.01) to every parameter,
twiddles included. Penalising the *factors* of a product bounds the **nuclear**
norm of the composed matrix rather than its Frobenius norm (Khodak et al.,
ICLR 2021), and nuclear-norm regularisation is rank-shrinking by construction.
Butterfly Transform separately reports weight decay has a "significant negative
impact" on butterfly layers, because each input-output pair has a single
multiplicative path and pushing any edge toward zero severs it. Measurement
has since bounded how much of the observed rank collapse this can explain --
the collapse is present at initialisation, so decay is a secondary term -- but
the mechanism is real and switching it off costs one config key.

**Learning rate.** Under muP the Adam learning rate scales as 1/fan_in, and
**every radix-2 butterfly factor has fan-in 2 regardless of the layer's
width**, so the prescription for a twiddle differs from that of the dense layer
it replaced by roughly d_in/2. This is an INFERENCE from the per-factor
formulas in arXiv:2410.02117, not a published result for butterflies, and the
one attempt to read the paper that would settle the magnitude returned content
judged fabricated and was discarded. That is exactly why the multiplier is a
swept variable here and not a prescribed constant.

Config keys, both no-ops at their defaults:
    "twiddle_lr_mult":       float, default 1.0
    "twiddle_weight_decay":  float or null, default null (inherit)
"""

from __future__ import annotations


def is_twiddle(name):
    """Both butterfly implementations in this repo name the parameter `twiddle`:
    torch_structured's ``Butterfly`` and ``bfly_arch.StageButterfly``."""
    return name.rsplit(".", 1)[-1] == "twiddle"


def build_param_groups(model, lr, twiddle_lr_mult=1.0, twiddle_weight_decay=None,
                       weight_decay=None):
    """Split a model's parameters into a default group and a twiddle group.

    Returns a list usable directly as an optimiser's first argument. When the
    model has no twiddles, or the settings are at their defaults, the result is
    behaviourally identical to passing ``model.parameters()``.
    """
    twiddles, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (twiddles if is_twiddle(name) else rest).append(p)

    common = {} if weight_decay is None else {"weight_decay": float(weight_decay)}
    groups = []
    if rest:
        groups.append(dict(params=rest, lr=lr, group_name="default", **common))
    if twiddles:
        g = dict(params=twiddles, lr=lr * float(twiddle_lr_mult),
                 group_name="twiddle", **common)
        if twiddle_weight_decay is not None:
            g["weight_decay"] = float(twiddle_weight_decay)
        groups.append(g)
    return groups


def describe(groups):
    """One line per group, for the training log."""
    out = []
    for g in groups:
        n_p = sum(p.numel() for p in g["params"])
        wd = g.get("weight_decay", "(optimiser default)")
        out.append("{:<8} {:>10,} params  lr={:.3e}  weight_decay={}".format(
            g.get("group_name", "?"), n_p, g["lr"], wd))
    return out
