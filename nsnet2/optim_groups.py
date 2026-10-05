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

**Recurrent twiddles may want a different multiplier from the rest.** The
stage-1 sweep (2026-10-04, 6 runs, 50 epochs) found `twiddle_lr_mult` 10 beats
1 by +0.055 PESQ, but the conditioning diagnostics show it is not a clean win:
the multiplier helps the randn-initialised feed-forward butterflies
(99%-energy rank of `fcs.0` 109-112 -> 130-135) and *hurts* the
orthogonally-initialised recurrent matrix (`rnn.h_proj.0` 438-447 -> 230-282).
That is mechanically unsurprising -- `W_hh` starts at condition number 1.0 and
a larger step knocks it off that initialisation faster, while the `fc_*`
blocks start collapsed and need to move. Hence a separate multiplier for the
recurrent twiddles, defaulting to "same as the others" so nothing changes
unless it is set.

Config keys, all no-ops at their defaults:
    "twiddle_lr_mult":            float, default 1.0
    "twiddle_weight_decay":       float or null, default null (inherit)
    "recurrent_twiddle_lr_mult":  float or null, default null (use twiddle_lr_mult)
    "recurrent_twiddle_patterns": list[str], default ["h_proj"]
"""

from __future__ import annotations


def is_twiddle(name):
    """Both butterfly implementations in this repo name the parameter `twiddle`:
    torch_structured's ``Butterfly`` and ``bfly_arch.StageButterfly``."""
    return name.rsplit(".", 1)[-1] == "twiddle"


DEFAULT_RECURRENT_PATTERNS = ("h_proj",)


def is_recurrent(name, patterns=DEFAULT_RECURRENT_PATTERNS):
    """Does this parameter belong to a recurrent (hidden-to-hidden) projection?

    ``ButterflyGRU`` names them ``h_proj.*`` against ``x_proj.*`` for the input
    projection. Kept as a pattern list rather than hard-coded so other cells
    (a CIFG-LSTM's recurrent block, say) can opt in without touching this file.
    """
    return any(tok in name for tok in patterns)


def build_param_groups(model, lr, twiddle_lr_mult=1.0, twiddle_weight_decay=None,
                       weight_decay=None, recurrent_twiddle_lr_mult=None,
                       recurrent_patterns=DEFAULT_RECURRENT_PATTERNS):
    """Split a model's parameters into a default group and one or two twiddle groups.

    Returns a list usable directly as an optimiser's first argument. When the
    model has no twiddles, or the settings are at their defaults, the result is
    behaviourally identical to passing ``model.parameters()``.
    """
    split_recurrent = recurrent_twiddle_lr_mult is not None
    buckets = {"default": [], "twiddle": [], "twiddle_recurrent": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not is_twiddle(name):
            buckets["default"].append(p)
        elif split_recurrent and is_recurrent(name, recurrent_patterns):
            buckets["twiddle_recurrent"].append(p)
        else:
            buckets["twiddle"].append(p)

    common = {} if weight_decay is None else {"weight_decay": float(weight_decay)}
    mults = {"default": 1.0,
             "twiddle": float(twiddle_lr_mult),
             "twiddle_recurrent": float(recurrent_twiddle_lr_mult or twiddle_lr_mult)}
    groups = []
    for key in ("default", "twiddle", "twiddle_recurrent"):
        if not buckets[key]:
            continue
        g = dict(params=buckets[key], lr=lr * mults[key], group_name=key, **common)
        if key.startswith("twiddle") and twiddle_weight_decay is not None:
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
