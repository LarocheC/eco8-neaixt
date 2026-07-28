"""Quantization-health gates.

All four exist because of things that actually shipped in this repo, or because
a mutation experiment showed a malformed buffer sailing through:

* **G-SCALE-FINITE** -- `convfsenet_win_streaming.tflite` carries 182 tensors
  whose quantization scale is NaN/Inf.  The power-law magnitude-compression
  `Pow` produced non-finite activations during calibration, MinMax recorded
  them, and the converter wrote them straight into the buffer.  The fp32 sibling
  is fine, so nothing upstream of the conversion notices.  This is a hard fail:
  a NaN scale poisons every downstream requantization multiplier.

* **G-SCALE-POSITIVE** -- the TFLite spec requires ``scale > 0``.  A *negative*
  scale is finite and of ordinary magnitude, so neither G-SCALE-FINITE nor
  G-SCALE-FLOOR can see it; a *zero* scale is not a small range, it is a
  divide-by-zero one algebraic step later.  Both are hard fails, and TFLM's own
  `kernel_util.cc` shows why::

      const double effective_output_scale = input_scale * filter_scale / output_scale;
      QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);

  The per-*tensor* path guards this (``TF_LITE_ENSURE(context,
  input_product_scale >= 0)``) so AllocateTensors() fails at boot; the
  per-*channel* path has no sign guard at all, silently stores a negative
  multiplier, and `MultiplyByQuantizedMultiplier`'s
  ``TFLITE_DCHECK(double_multiplier >= 0)`` is commented out.  A zero output
  scale makes the quotient inf/nan; a zero input scale kills the channel.

* **G-QUANT-STRUCT** -- structural validity of the quantization block itself,
  independent of the values in it: the per-axis scale vector must be as long as
  the axis it quantizes, scales and zero_points must agree in length, and a
  zero_point must fit the tensor's own storage type.  These were previously
  caught only by the *host* XNNPACK delegate inside G-LOAD-ISOLATED -- a
  host-only artifact that does not exist on an M33 -- and `--no-load` removed
  the catch entirely.

* **G-SCALE-FLOOR** -- the calibrator pins an unexercised tensor's range to a
  floor (3.9215683651783184e-12 here, i.e. 1e-9/255).  When that happens to a
  *model input*, and that input is a streaming state-feedback port, it means the
  representative dataset never drove the recurrence: the state arrives quantized
  to a dead range and the model degrades silently on-device while looking
  perfectly healthy on the host.  Warn, and call out the inputs separately --
  a floored intermediate is ordinary, a floored input is the bug signature.
  Strictly ``0 < s <= threshold``: zero and negative are a different phenomenon
  with a different cause and a different consequence, and belong to
  G-SCALE-POSITIVE.
"""

from __future__ import annotations

import math
from typing import List

from . import Gate, GateResult, HARD_FAIL, WARN, skip
from ..runtime.flatbuffer import ZERO_POINT_RANGE

G_SCALE_FINITE = Gate(
    id="G-SCALE-FINITE",
    title="No non-finite quantization scales",
    severity=HARD_FAIL,
)

G_SCALE_POSITIVE = Gate(
    id="G-SCALE-POSITIVE",
    title="Every quantization scale is strictly positive",
    severity=HARD_FAIL,
)

G_QUANT_STRUCT = Gate(
    id="G-QUANT-STRUCT",
    title="Quantization block is structurally valid",
    severity=HARD_FAIL,
)

G_SCALE_FLOOR = Gate(
    id="G-SCALE-FLOOR",
    title="No collapsed (floored) quantization ranges",
    severity=WARN,
)

DEFAULT_FLOOR = 1e-11

# How many tensor names to spell out before truncating the human-readable detail.
_NAME_BUDGET = 6


def _fmt_names(items: List[str], budget: int = _NAME_BUDGET) -> str:
    if not items:
        return "none"
    shown = items[:budget]
    rest = len(items) - len(shown)
    text = ", ".join(shown)
    if rest > 0:
        text += " (+%d more)" % rest
    return text


def check_scale_finite(ctx: dict) -> GateResult:
    """Hard-fail if any tensor has a NaN or Inf quantization scale."""
    model = ctx.get("model")
    if model is None:
        return skip(G_SCALE_FINITE, "no model in context")

    bad: List[dict] = []
    for t in model.tensors:
        if not t.scales:
            continue
        offenders = [s for s in t.scales if not math.isfinite(s)]
        if offenders:
            bad.append(
                {
                    "index": t.index,
                    "name": t.name,
                    "dtype": t.dtype,
                    "n_bad": len(offenders),
                    "n_scales": len(t.scales),
                    "example": repr(offenders[0]),
                }
            )

    n_quant = sum(1 for t in model.tensors if t.is_quantized)
    evidence = {
        "scope": "subgraph 0",
        "n_tensors": len(model.tensors),
        "n_quantized_tensors": n_quant,
        "n_nonfinite_tensors": len(bad),
        "tensors": bad[:64],
    }

    if not bad:
        return GateResult(
            gate=G_SCALE_FINITE,
            passed=True,
            # Scoped on purpose: every gate here reads subgraph 0, so an
            # unqualified "all N quantized tensors" would be a whole-model claim
            # made from partial evidence.  G-SINGLE-SUBGRAPH covers the rest.
            detail="all %d quantized tensors in subgraph 0 have finite scales" % n_quant,
            evidence=evidence,
        )

    return GateResult(
        gate=G_SCALE_FINITE,
        passed=False,
        detail=(
            "%d of %d quantized tensors in subgraph 0 have a non-finite scale "
            "(e.g. %s on t%d %r). "
            "Every requantization multiplier derived from these is undefined; the "
            "model cannot be trusted on-device even if it loads. Usual cause: a Pow "
            "or log-domain op produced NaN/Inf during calibration."
            % (
                len(bad),
                n_quant,
                bad[0]["example"],
                bad[0]["index"],
                bad[0]["name"],
            )
        ),
        evidence=evidence,
    )


def check_scale_positive(ctx: dict) -> GateResult:
    """Hard-fail if any tensor has a zero or negative quantization scale."""
    model = ctx.get("model")
    if model is None:
        return skip(G_SCALE_POSITIVE, "no model in context")

    negative: List[dict] = []
    zero: List[dict] = []
    input_set = set(model.inputs)
    for t in model.tensors:
        if not t.scales:
            continue
        offenders = t.nonpositive_scales()
        if not offenders:
            continue
        rec = {
            "index": t.index,
            "name": t.name,
            "dtype": t.dtype,
            "per_axis": len(t.scales) > 1,
            "n_scales": len(t.scales),
            "n_bad": len(offenders),
            "bad_channels": [
                i for i, s in enumerate(t.scales) if math.isfinite(s) and s <= 0.0
            ][:16],
            "example": float(offenders[0]),
            "is_model_input": t.index in input_set,
        }
        if any(s < 0.0 for s in offenders):
            negative.append(rec)
        else:
            zero.append(rec)

    n_quant = sum(1 for t in model.tensors if t.is_quantized)
    evidence = {
        "scope": "subgraph 0",
        "n_quantized_tensors": n_quant,
        "n_negative_tensors": len(negative),
        "n_zero_tensors": len(zero),
        "negative": negative[:64],
        "zero": zero[:64],
    }

    if not negative and not zero:
        return GateResult(
            gate=G_SCALE_POSITIVE,
            passed=True,
            detail=(
                "all %d quantized tensors in subgraph 0 have strictly positive scales"
                % n_quant
            ),
            evidence=evidence,
        )

    problems: List[str] = []
    if negative:
        n = negative[0]
        problems.append(
            "%d tensor(s) carry a NEGATIVE scale (e.g. %r on t%d %r%s). The TFLite "
            "spec requires scale > 0. TFLM's per-tensor path returns kTfLiteError "
            "from Prepare(), so AllocateTensors() fails at boot; the per-channel "
            "path in kernel_util.cc has no sign guard at all, stores a negative "
            "requantization multiplier and silently produces garbage for that channel"
            % (
                len(negative),
                n["example"],
                n["index"],
                n["name"],
                " (per-axis, channel(s) %s)" % n["bad_channels"] if n["per_axis"] else "",
            )
        )
    if zero:
        z = zero[0]
        problems.append(
            "%d tensor(s) carry a scale of exactly ZERO (e.g. t%d %r%s). This is not "
            "a small range, it is a divide-by-zero: effective_output_scale = "
            "input*filter/output is inf/nan when the output scale is 0, and the "
            "channel is dead when an input scale is 0"
            % (
                len(zero),
                z["index"],
                z["name"],
                " (per-axis, channel(s) %s)" % z["bad_channels"] if z["per_axis"] else "",
            )
        )

    return GateResult(
        gate=G_SCALE_POSITIVE,
        passed=False,
        detail="; ".join(problems) + ".",
        evidence=evidence,
    )


def check_quant_struct(ctx: dict) -> GateResult:
    """Hard-fail on a structurally malformed quantization block."""
    model = ctx.get("model")
    if model is None:
        return skip(G_QUANT_STRUCT, "no model in context")

    bad_axis: List[dict] = []
    bad_pairing: List[dict] = []
    bad_zp: List[dict] = []

    for t in model.tensors:
        if not t.scales:
            continue

        if t.zero_points and len(t.zero_points) != len(t.scales):
            bad_pairing.append(
                {
                    "index": t.index,
                    "name": t.name,
                    "n_scales": len(t.scales),
                    "n_zero_points": len(t.zero_points),
                }
            )

        if len(t.scales) > 1:
            axis_len = t.quantized_dim_size()
            if axis_len is not None and axis_len != len(t.scales):
                bad_axis.append(
                    {
                        "index": t.index,
                        "name": t.name,
                        "shape": list(t.shape),
                        "quantized_dimension": t.quantized_dimension,
                        "axis_len": axis_len,
                        "n_scales": len(t.scales),
                    }
                )

        rng = ZERO_POINT_RANGE.get(t.dtype)
        if rng and t.zero_points:
            lo, hi = rng
            out = [z for z in t.zero_points if z < lo or z > hi]
            if out:
                bad_zp.append(
                    {
                        "index": t.index,
                        "name": t.name,
                        "dtype": t.dtype,
                        "allowed": [lo, hi],
                        "offenders": out[:8],
                        "n_bad": len(out),
                    }
                )

    n_quant = sum(1 for t in model.tensors if t.is_quantized)
    evidence = {
        "scope": "subgraph 0",
        "n_quantized_tensors": n_quant,
        "n_axis_length_mismatch": len(bad_axis),
        "n_scale_zeropoint_mismatch": len(bad_pairing),
        "n_zero_point_out_of_range": len(bad_zp),
        "axis_length_mismatch": bad_axis[:64],
        "scale_zeropoint_mismatch": bad_pairing[:64],
        "zero_point_out_of_range": bad_zp[:64],
    }

    if not (bad_axis or bad_pairing or bad_zp):
        return GateResult(
            gate=G_QUANT_STRUCT,
            passed=True,
            detail=(
                "all %d quantized tensors in subgraph 0 have a well-formed "
                "quantization block (per-axis vectors match their axis, "
                "scale/zero_point lengths agree, zero_points fit the dtype)"
                % n_quant
            ),
            evidence=evidence,
        )

    problems: List[str] = []
    if bad_axis:
        a = bad_axis[0]
        problems.append(
            "%d tensor(s) have a per-axis scale vector that does not match the "
            "quantized dimension (e.g. t%d %r shape %s, quantized_dimension %d has "
            "length %d but carries %d scales) -- every kernel that walks the "
            "per-channel multiplier array reads past the end of it"
            % (
                len(bad_axis),
                a["index"],
                a["name"],
                a["shape"],
                a["quantized_dimension"],
                a["axis_len"],
                a["n_scales"],
            )
        )
    if bad_pairing:
        p = bad_pairing[0]
        problems.append(
            "%d tensor(s) have mismatched scale/zero_point vector lengths (e.g. t%d "
            "%r: %d scales, %d zero_points)"
            % (len(bad_pairing), p["index"], p["name"], p["n_scales"], p["n_zero_points"])
        )
    if bad_zp:
        z = bad_zp[0]
        problems.append(
            "%d tensor(s) have a zero_point outside the dtype's range (e.g. t%d %r is "
            "%s, allowed %d..%d, found %s) -- the offset cannot be represented by the "
            "tensor's own storage type"
            % (
                len(bad_zp),
                z["index"],
                z["name"],
                z["dtype"],
                z["allowed"][0],
                z["allowed"][1],
                z["offenders"],
            )
        )

    return GateResult(
        gate=G_QUANT_STRUCT,
        passed=False,
        detail="; ".join(problems) + ".",
        evidence=evidence,
    )


def check_scale_floor(ctx: dict) -> GateResult:
    """Warn if any tensor's scale sits in ``0 < s <= threshold``."""
    model = ctx.get("model")
    if model is None:
        return skip(G_SCALE_FLOOR, "no model in context")

    threshold = float(ctx.get("floor_threshold", DEFAULT_FLOOR))
    input_set = set(model.inputs)

    floored: List[dict] = []
    for t in model.tensors:
        if not t.scales:
            continue
        # Strictly positive: a zero or negative scale is invalid rather than
        # collapsed, and saying "likely benign" about it -- as this gate used to
        # -- talks the reader out of a genuine deploy blocker.  G-SCALE-POSITIVE
        # owns those.
        smallest = t.min_positive_scale()
        if smallest is None or smallest > threshold:
            continue
        floored.append(
            {
                "index": t.index,
                "name": t.name,
                "dtype": t.dtype,
                "shape": list(t.shape),
                "min_scale": smallest,
                "is_model_input": t.index in input_set,
                "input_position": (
                    model.inputs.index(t.index) if t.index in input_set else None
                ),
            }
        )

    floored_inputs = [f for f in floored if f["is_model_input"]]
    distinct = sorted({f["min_scale"] for f in floored})

    evidence = {
        "scope": "subgraph 0",
        "threshold": threshold,
        "n_floored": len(floored),
        "n_floored_inputs": len(floored_inputs),
        "n_inputs": len(model.inputs),
        "distinct_floor_values": distinct[:8],
        "floored_inputs": floored_inputs[:64],
        "floored_tensors": floored[:64],
    }

    if not floored:
        return GateResult(
            gate=G_SCALE_FLOOR,
            passed=True,
            detail="no tensor scale in (0, %g]" % threshold,
            evidence=evidence,
        )

    parts = [
        "%d tensor(s) have a collapsed range (0 < scale <= %g; smallest observed %r)"
        % (len(floored), threshold, distinct[0] if distinct else None)
    ]
    if floored_inputs:
        names = _fmt_names(
            ["t%d %s" % (f["index"], f["name"]) for f in floored_inputs]
        )
        input_floors = sorted({f["min_scale"] for f in floored_inputs})
        evidence["input_floor_values"] = input_floors[:8]
        # NB the denominator is the model's input count, NOT len(floored) --
        # "9/10 of them" against a leading "110 tensor(s)" read as 9/10ths of
        # the 110.  Say which population is being divided.
        parts.append(
            "%d of the model's %d INPUTS are among them, pinned at %s: %s"
            % (
                len(floored_inputs),
                len(model.inputs),
                ", ".join(repr(v) for v in input_floors[:3]),
                names,
            )
        )
        parts.append(
            "floored *inputs* are the calibration-bug signature: the representative "
            "dataset never drove these ports (they are the streaming state-feedback "
            "inputs), so the recurrence is quantized to a dead range on-device"
        )
    else:
        parts.append("none of them are model inputs, so this is likely benign")

    return GateResult(
        gate=G_SCALE_FLOOR,
        passed=False,
        detail="; ".join(parts) + ".",
        evidence=evidence,
    )


check_scale_finite.gate = G_SCALE_FINITE  # type: ignore[attr-defined]
check_scale_positive.gate = G_SCALE_POSITIVE  # type: ignore[attr-defined]
check_quant_struct.gate = G_QUANT_STRUCT  # type: ignore[attr-defined]
check_scale_floor.gate = G_SCALE_FLOOR  # type: ignore[attr-defined]

__all__ = [
    "G_SCALE_FINITE",
    "G_SCALE_POSITIVE",
    "G_QUANT_STRUCT",
    "G_SCALE_FLOOR",
    "DEFAULT_FLOOR",
    "check_scale_finite",
    "check_scale_positive",
    "check_quant_struct",
    "check_scale_floor",
]
