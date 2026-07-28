"""Structural gates: model shape vs. what the firmware assumes about it.

* **G-IO-SYMMETRY** -- hard fail.  Every artifact in this lane is a *streaming*
  model: one feature port plus N state-feedback ports, and each state input must
  have a matching state output.  The firmware's feedback loop indexes them
  positionally -- `deploy/rt595/app/model_se_stream.cpp` does
  ``s_interp->output(m->out_pos)`` / ``s_interp->input(m->in_pos)`` for each
  entry of ``MODEL_STATE_MAP`` -- so an export that drops one state output is an
  out-of-range interpreter access on an M33 with no MMU.  Nothing else in the
  auditor relates the input count to the output count: G-LOAD-ISOLATED's own
  cross-check compares the interpreter's counts against the flatbuffer's, and
  both agree on the same asymmetric model.

* **G-SINGLE-SUBGRAPH** -- hard fail.  Every gate here reads subgraph 0.  A note
  saying so is not a substitute for a gate: an unregisterable op or a NaN scale
  parked in subgraph 1 produced a CLEAR verdict and exit 0 while G-OPSET printed
  "all 13 distinct ops are registerable".  A multi-subgraph model is not
  deployable on this lane anyway (the firmware invokes subgraph 0 only), so
  refusing to bless what was not read costs nothing and closes the hole.

* **G-IO-LAYOUT** -- warn.  `deploy/rt595/app/model_io_layout.h` is the
  generated model-vs-firmware contract: ``MODEL_N_IO``, ``MODEL_N_STATES``,
  ``MODEL_FEATURE_IN_POS``, ``MODEL_MASK_OUT_POS``, ``MODEL_FEATURE_IS_INT8``.
  It is exactly the same kind of contract G-RESOLVER-EXACT enforces for ops, and
  it is currently violated on the committed tree (the header was generated from
  an 18-IO fp32 LiSenNet; the artifact the sweep blesses has 26 IO and is int8).
  Warn rather than hard-fail, because the header describes *one* model and the
  sweep audits five -- the detail names the header's own source file so a stale
  header reads differently from "this is simply not that model".
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from . import Gate, GateResult, HARD_FAIL, WARN, skip

G_IO_SYMMETRY = Gate(
    id="G-IO-SYMMETRY",
    title="Streaming model has one output per input",
    severity=HARD_FAIL,
)

G_SINGLE_SUBGRAPH = Gate(
    id="G-SINGLE-SUBGRAPH",
    title="Model has exactly one subgraph (the only one audited)",
    severity=HARD_FAIL,
)

G_IO_LAYOUT = Gate(
    id="G-IO-LAYOUT",
    title="model_io_layout.h agrees with the model",
    severity=WARN,
)

_DEFINE_RE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+(MODEL_[A-Z0-9_]+)[ \t]+\(?\s*([-+]?[0-9.]+)f?\s*\)?",
    re.M,
)
_SOURCE_RE = re.compile(r"/\*\s*source:\s*(\S+)\s*\*/")


def parse_io_layout(path: str) -> dict:
    """Read the generated `model_io_layout.h` into a plain dict.

    Returns ``{"path", "source", "defines": {NAME: number}}``.  Raises OSError if
    the header cannot be read -- the caller turns that into a SKIP.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    defines: Dict[str, float] = {}
    for m in _DEFINE_RE.finditer(text):
        raw = m.group(2)
        try:
            defines[m.group(1)] = float(raw) if "." in raw else int(raw)
        except ValueError:  # pragma: no cover - regex constrains this
            continue
    src = _SOURCE_RE.search(text)
    return {
        "path": os.path.abspath(path),
        "source": src.group(1) if src else None,
        "defines": defines,
    }


def check_io_symmetry(ctx: dict) -> GateResult:
    """Hard-fail when the model does not have one output per input."""
    model = ctx.get("model")
    if model is None:
        return skip(G_IO_SYMMETRY, "no model in context")

    n_in, n_out = len(model.inputs), len(model.outputs)
    expected: Optional[int] = ctx.get("family_expected_io")
    evidence = {
        "scope": "subgraph 0",
        "n_inputs": n_in,
        "n_outputs": n_out,
        "family": ctx.get("family"),
        "family_expected_io": expected,
    }

    if n_in != n_out:
        return GateResult(
            gate=G_IO_SYMMETRY,
            passed=False,
            detail=(
                "%d input(s) but %d output(s). Every artifact on this lane is a "
                "streaming model whose state inputs must each have a matching state "
                "output; app/model_se_stream.cpp indexes them positionally via "
                "MODEL_STATE_MAP (interp->output(out_pos) / interp->input(in_pos)), "
                "so a dropped state output is an out-of-range interpreter access on "
                "an M33 with no MMU." % (n_in, n_out)
            ),
            evidence=evidence,
        )

    detail = "%d in / %d out -- one output per input, %d state port(s)" % (
        n_in,
        n_out,
        max(n_in - 1, 0),
    )
    if expected is not None and expected != n_in:
        detail += (
            "; NOTE the %s family is expected to have %d IO, not %d -- either this "
            "is a re-export or the family was mis-inferred"
            % (ctx.get("family"), expected, n_in)
        )
    return GateResult(gate=G_IO_SYMMETRY, passed=True, detail=detail, evidence=evidence)


def check_single_subgraph(ctx: dict) -> GateResult:
    """Hard-fail on a multi-subgraph model: the other subgraphs are never read."""
    model = ctx.get("model")
    if model is None:
        return skip(G_SINGLE_SUBGRAPH, "no model in context")

    n = int(model.n_subgraphs)
    evidence = {"n_subgraphs": n}
    if n == 1:
        return GateResult(
            gate=G_SINGLE_SUBGRAPH,
            passed=True,
            detail="1 subgraph, so the whole model was audited",
            evidence=evidence,
        )
    return GateResult(
        gate=G_SINGLE_SUBGRAPH,
        passed=False,
        detail=(
            "model has %d subgraphs. Every gate in this auditor reads SUBGRAPH 0 "
            "only -- opset, resolver, quantization, accel coverage and the IO "
            "summary alike -- so their results describe %d/%d of the model and "
            "cannot be read as a clean bill of health. The RT595 firmware invokes "
            "subgraph 0 only, so a multi-subgraph model is not deployable on this "
            "lane regardless." % (n, 1, n)
        ),
        evidence=evidence,
    )


def check_io_layout(ctx: dict) -> GateResult:
    """Warn when the generated firmware IO header disagrees with the model."""
    model = ctx.get("model")
    if model is None:
        return skip(G_IO_LAYOUT, "no model in context")

    layout = ctx.get("io_layout")
    if layout is None:
        reason = ctx.get("io_layout_error") or (
            "no app/model_io_layout.h found (pass --io-layout <header>)"
        )
        return skip(G_IO_LAYOUT, "cannot check the firmware IO layout: %s" % reason)

    d = layout.get("defines") or {}
    n_in, n_out = len(model.inputs), len(model.outputs)
    mismatches: List[str] = []

    n_io = d.get("MODEL_N_IO")
    n_states = d.get("MODEL_N_STATES")
    if n_io is not None and int(n_io) != n_in:
        mismatches.append("MODEL_N_IO is %d but the model has %d input(s)" % (n_io, n_in))
    if n_states is not None and n_io is not None and int(n_states) != int(n_io) - 1:
        mismatches.append(
            "MODEL_N_STATES (%d) is not MODEL_N_IO-1 (%d)" % (n_states, int(n_io) - 1)
        )

    for macro, count in (("MODEL_FEATURE_IN_POS", n_in), ("MODEL_MASK_OUT_POS", n_out)):
        pos = d.get(macro)
        if pos is not None and not (0 <= int(pos) < count):
            mismatches.append(
                "%s is %d, out of range for %d port(s)" % (macro, int(pos), count)
            )

    is_int8 = d.get("MODEL_FEATURE_IS_INT8")
    if is_int8 is not None and model.inputs:
        feat_pos = int(d.get("MODEL_FEATURE_IN_POS", 0))
        if 0 <= feat_pos < len(model.inputs):
            feat = model.tensor(model.inputs[feat_pos])
            actual_int8 = feat.dtype in ("INT8", "UINT8")
            if bool(int(is_int8)) != actual_int8:
                mismatches.append(
                    "MODEL_FEATURE_IS_INT8 is %d but input #%d (%s) is %s"
                    % (int(is_int8), feat_pos, feat.name, feat.dtype)
                )

    evidence = {
        "header": layout.get("path"),
        "header_source_model": layout.get("source"),
        "defines": d,
        "n_inputs": n_in,
        "n_outputs": n_out,
        "mismatches": mismatches,
    }

    src = layout.get("source") or "(unrecorded)"
    if not mismatches:
        return GateResult(
            gate=G_IO_LAYOUT,
            passed=True,
            detail="%s (generated from %s) agrees with this model"
            % (os.path.basename(layout.get("path") or "model_io_layout.h"), src),
            evidence=evidence,
        )

    return GateResult(
        gate=G_IO_LAYOUT,
        passed=False,
        detail=(
            "%s does not describe this model: %s. That header was generated from "
            "%s; the firmware indexes ports by these constants, so either the "
            "header is stale or this artifact is not the one the firmware carries."
            % (
                os.path.basename(layout.get("path") or "model_io_layout.h"),
                "; ".join(mismatches),
                src,
            )
        ),
        evidence=evidence,
    )


check_io_symmetry.gate = G_IO_SYMMETRY  # type: ignore[attr-defined]
check_single_subgraph.gate = G_SINGLE_SUBGRAPH  # type: ignore[attr-defined]
check_io_layout.gate = G_IO_LAYOUT  # type: ignore[attr-defined]

__all__ = [
    "G_IO_SYMMETRY",
    "G_SINGLE_SUBGRAPH",
    "G_IO_LAYOUT",
    "parse_io_layout",
    "check_io_symmetry",
    "check_single_subgraph",
    "check_io_layout",
]
