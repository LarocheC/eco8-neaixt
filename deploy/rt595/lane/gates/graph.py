"""Graph-level gates: can TFLM run these ops at all, and will the DSP help.

* **G-OPSET** -- hard fail.  An op with no `Add*` method in
  `micro_mutable_op_resolver.h` cannot be registered by *any* resolver, so the
  model is undeployable no matter what the firmware does.  This is how
  `convfsenet_nocompress` and `convfsenet_win` get caught: TFLM's `reduce.cc`
  registers MEAN, REDUCE_MAX and SUM -- but no REDUCE_ALL / REDUCE_ANY -- and
  there is no POW kernel anywhere in the tree.  The failure message names the
  tree that was consulted,
  because "unsupported" is only ever true *of a particular checkout*.

* **G-ACCEL-COVERAGE** -- warn.  Count-weighted fraction of the graph's ops that
  reach an xa_nnlib HiFi4 kernel in the built library.  Low coverage does not
  break anything; it just means the ops are running as portable C and the 6.7x
  real-time gap is not going to close.
"""

from __future__ import annotations

from typing import Dict

from . import Gate, GateResult, HARD_FAIL, WARN, skip

G_OPSET = Gate(
    id="G-OPSET",
    title="Every op has a TFLM Add* method",
    severity=HARD_FAIL,
)

G_ACCEL_COVERAGE = Gate(
    id="G-ACCEL-COVERAGE",
    title="Ops reaching an xa_nnlib HiFi4 kernel",
    severity=WARN,
)

DEFAULT_MIN_ACCEL_FRACTION = 0.60


def check_opset(ctx: dict) -> GateResult:
    """Hard-fail if the graph uses an op TFLM has no `Add*` method for."""
    model = ctx.get("model")
    if model is None:
        return skip(G_OPSET, "no model in context")

    supported = ctx.get("supported_ops")
    tree = ctx.get("tflm_tree")
    if not supported:
        reason = ctx.get("tflm_tree_error") or (
            "no TFLM op list available (pass --tflm-tree <tflite-micro checkout>)"
        )
        return skip(G_OPSET, "cannot check opset: %s" % reason, tflm_tree=tree)

    supported = set(supported)
    used = model.op_histogram
    missing = sorted(op for op in used if op not in supported)

    evidence = {
        "scope": "subgraph 0",
        "tflm_tree": tree,
        "n_supported_ops": len(supported),
        "ops_used": dict(sorted(used.items())),
        "missing": missing,
        "missing_counts": {op: used[op] for op in missing},
    }

    if not missing:
        return GateResult(
            gate=G_OPSET,
            passed=True,
            detail=(
                # Scoped: this gate reads subgraph 0, so an unqualified "all N
                # distinct ops" would be a whole-model claim from partial
                # evidence.  G-SINGLE-SUBGRAPH blocks anything with more.
                "all %d distinct ops in subgraph 0 (%d nodes) are registerable by "
                "the TFLM tree at %s" % (len(used), model.n_ops, tree)
            ),
            evidence=evidence,
        )

    listed = ", ".join("%s x%d" % (op, used[op]) for op in missing)
    return GateResult(
        gate=G_OPSET,
        passed=False,
        detail=(
            "%d op(s) in subgraph 0 have no Add* method in the TFLM tree at %s: %s. "
            "No MicroMutableOpResolver can register these, so the graph cannot "
            "run on the RT595 as exported -- it has to be re-exported without them."
            % (len(missing), tree, listed)
        ),
        evidence=evidence,
    )


def check_accel_coverage(ctx: dict) -> GateResult:
    """Warn if too few of the graph's nodes reach a HiFi4 kernel."""
    model = ctx.get("model")
    if model is None:
        return skip(G_ACCEL_COVERAGE, "no model in context")

    accel = ctx.get("accelerated")
    status = ctx.get("accel_status") or {}
    if not accel:
        reason = status.get("reason") or (
            "no accel map available (pass --lib <libtflm_rt500.a> and make xt-nm reachable)"
        )
        return skip(G_ACCEL_COVERAGE, "cannot check HiFi4 coverage: %s" % reason)

    accel = set(accel)
    threshold = float(ctx.get("accel_min_fraction", DEFAULT_MIN_ACCEL_FRACTION))

    total = model.n_ops
    covered = 0
    hot: Dict[str, int] = {}
    cold: Dict[str, int] = {}
    for op, count in model.op_histogram.items():
        if op in accel:
            covered += count
            hot[op] = count
        else:
            cold[op] = count

    fraction = (covered / total) if total else 0.0
    worst = sorted(cold.items(), key=lambda kv: -kv[1])

    evidence = {
        "scope": "subgraph 0",
        "n_ops": total,
        "n_accelerated_ops": covered,
        "fraction": fraction,
        "threshold": threshold,
        "accelerated": dict(sorted(hot.items())),
        "reference_c": dict(sorted(cold.items())),
        "biggest_gaps": worst[:8],
        "accel_source": status.get("lib_path"),
    }

    summary = "%d/%d nodes in subgraph 0 (%.1f%%) hit an xa_nnlib HiFi4 kernel" % (
        covered,
        total,
        100.0 * fraction,
    )
    if worst:
        # Truncating this list silently made the text stop reconciling with
        # n_accelerated_ops (convfsenet_win has 6 portable-C ops; showing 5 hid
        # REDUCE_ALL x1 and made 80 portable nodes read as 79).  Always account
        # for the remainder.
        shown = worst[:5]
        summary += "; portable C: " + ", ".join("%s x%d" % (op, n) for op, n in shown)
        rest = worst[len(shown) :]
        if rest:
            summary += " (+%d more op(s), %d node(s))" % (
                len(rest),
                sum(n for _, n in rest),
            )

    passed = fraction >= threshold
    if not passed:
        summary += " -- below the %.0f%% bar, so most of the graph is not using the DSP" % (
            100.0 * threshold
        )

    return GateResult(
        gate=G_ACCEL_COVERAGE,
        passed=passed,
        detail=summary,
        evidence=evidence,
    )


check_opset.gate = G_OPSET  # type: ignore[attr-defined]
check_accel_coverage.gate = G_ACCEL_COVERAGE  # type: ignore[attr-defined]

__all__ = [
    "G_OPSET",
    "G_ACCEL_COVERAGE",
    "DEFAULT_MIN_ACCEL_FRACTION",
    "check_opset",
    "check_accel_coverage",
]
