"""Gates: small pure predicates over a context dict.

A gate is just ``(ctx: dict) -> GateResult``.  It never opens a file, never
shells out and never converts a model -- everything it needs is already in
`ctx`.  That is what makes the whole suite unit-testable against a synthetic
dict with no .tflite on disk and no toolchain installed.

Context keys (all optional; a gate that needs a missing key reports SKIP):

===========================  ===============================================
``model``                    `lane.runtime.flatbuffer.ModelInfo`
``tflm_tree``                path to the tflite-micro checkout consulted
``tflm_tree_error``          why the tree could not be read, if it could not
``supported_ops``            set of op names TFLM has an ``Add*`` method for
``accelerated``              set of op names with an xa_nnlib HiFi4 kernel
``accel_status``             `lane.runtime.accel_map.LAST_STATUS` snapshot
``accel_min_fraction``       coverage below which G-ACCEL-COVERAGE warns (0.60)
``resolver_path``            path of the resolver .cpp under audit
``resolver_ops``             set of op names that .cpp registers
``resolver_methods_used``    the ``Add*`` calls found in that .cpp
``resolver_unmapped``        ``Add*`` calls that could not be mapped to an op
``resolver_arity``           the N in ``MicroMutableOpResolver<N>``
``load_result``              dict from `lane.runtime.litert_host.load_isolated`
``floor_threshold``          scale at or below which a range counts as collapsed
``io_layout``                parsed `app/model_io_layout.h`, or absent
``family`` / ``family_expected_io``  inferred model family and its IO hint
``firmware_resolver_*``      same shape as ``resolver_*`` for the shipping .cpp
===========================  ===============================================

A gate reports three outcomes, not two: pass, fail, and *skip*.  Skip is encoded
as ``passed=True`` with ``evidence["skipped"] is True`` -- a gate that could not
run must never be counted as evidence of health, but must also never block a
deploy on tooling that simply is not installed on this box.

That second half is a statement about *blocking*, not about *reporting*.  A skip
is not a pass, so it must not reach the reader as one: `lane.report.AuditReport`
returns the verdict ``INCOMPLETE`` whenever a hard-fail-severity gate skipped,
and `python -m lane` exits 3 for it.  ``is_hard_fail`` stays False for a skip --
"we could not check" is not "we checked and it is broken" -- and the two are
kept apart all the way to the exit code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

HARD_FAIL = "hard-fail"
WARN = "warn"


@dataclass(frozen=True)
class Gate:
    id: str
    title: str
    severity: str  # "hard-fail" | "warn"


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    passed: bool
    detail: str
    evidence: dict

    @property
    def skipped(self) -> bool:
        return bool(self.evidence.get("skipped"))

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.passed else ("FAIL" if self.is_hard else "WARN")

    @property
    def is_hard(self) -> bool:
        return self.gate.severity == HARD_FAIL

    @property
    def is_hard_fail(self) -> bool:
        """True only for a real, non-skipped failure of a hard-fail gate."""
        return self.is_hard and not self.passed and not self.skipped

    @property
    def is_hard_skip(self) -> bool:
        """True when a *hard-fail-severity* gate could not run.

        This is the state that used to be laundered into a green verdict: the
        gate layer encoded a skip honestly, and the reporting layer counted it
        with the passes.  Nothing downstream may treat it as evidence of health.
        """
        return self.is_hard and self.skipped

    def to_dict(self) -> dict:
        return {
            "id": self.gate.id,
            "title": self.gate.title,
            "severity": self.gate.severity,
            "status": self.status,
            "passed": bool(self.passed),
            "skipped": self.skipped,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def skip(gate: Gate, reason: str, **evidence) -> GateResult:
    """Build a SKIP result: not a pass, but not a blocker either."""
    ev = dict(evidence)
    ev["skipped"] = True
    return GateResult(gate=gate, passed=True, detail=reason, evidence=ev)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

# Populated by the submodule imports below.  Ordered: cheap structural checks
# first, the subprocess load last.
from .quant import (  # noqa: E402
    G_QUANT_STRUCT,
    G_SCALE_FINITE,
    G_SCALE_FLOOR,
    G_SCALE_POSITIVE,
    check_quant_struct,
    check_scale_finite,
    check_scale_floor,
    check_scale_positive,
)
from .structure import (  # noqa: E402
    G_IO_LAYOUT,
    G_IO_SYMMETRY,
    G_SINGLE_SUBGRAPH,
    check_io_layout,
    check_io_symmetry,
    check_single_subgraph,
    parse_io_layout,
)
from .graph import (  # noqa: E402
    G_ACCEL_COVERAGE,
    G_OPSET,
    check_accel_coverage,
    check_opset,
)
from .resolver import (  # noqa: E402
    G_RESOLVER_ARITY,
    G_RESOLVER_EXACT,
    G_RESOLVER_FIRMWARE,
    G_RESOLVER_LEAN,
    check_resolver_arity,
    check_resolver_exact,
    check_resolver_firmware,
    check_resolver_lean,
    parse_resolver_cpp,
)
from .runtime import G_LOAD_ISOLATED, check_load_isolated  # noqa: E402

GateFn = Callable[[dict], GateResult]

GATES: List[GateFn] = [
    check_single_subgraph,
    check_io_symmetry,
    check_io_layout,
    check_scale_finite,
    check_scale_positive,
    check_quant_struct,
    check_scale_floor,
    check_opset,
    check_accel_coverage,
    check_resolver_exact,
    check_resolver_lean,
    check_resolver_arity,
    check_resolver_firmware,
    check_load_isolated,
]

GATES_BY_ID: Dict[str, GateFn] = {
    getattr(fn, "gate").id: fn for fn in GATES  # noqa: B009
}

ALL_GATE_SPECS: List[Gate] = [getattr(fn, "gate") for fn in GATES]  # noqa: B009


def run_gates(ctx: dict, gates: Optional[Sequence[GateFn]] = None) -> List[GateResult]:
    """Run every gate against `ctx`, in order.

    A gate that raises is reported as a failure rather than propagating -- the
    auditor's job is to produce a verdict, not to crash on a weird artifact.
    """
    selected = list(GATES if gates is None else gates)
    results: List[GateResult] = []
    for fn in selected:
        spec = getattr(fn, "gate", None)
        try:
            results.append(fn(ctx))
        except Exception as exc:  # pragma: no cover - defensive
            gate = spec or Gate(id=getattr(fn, "__name__", "?"), title="?", severity=HARD_FAIL)
            results.append(
                GateResult(
                    gate=gate,
                    passed=False,
                    detail="gate raised %s: %s" % (type(exc).__name__, exc),
                    evidence={"exception": repr(exc)},
                )
            )
    return results


def any_hard_fail(results: Sequence[GateResult]) -> bool:
    return any(r.is_hard_fail for r in results)


def any_hard_skip(results: Sequence[GateResult]) -> bool:
    """True when some hard-fail-severity gate could not run at all."""
    return any(r.is_hard_skip for r in results)


__all__ = [
    "Gate",
    "GateResult",
    "HARD_FAIL",
    "WARN",
    "skip",
    "run_gates",
    "any_hard_fail",
    "any_hard_skip",
    "GATES",
    "GATES_BY_ID",
    "ALL_GATE_SPECS",
    "parse_resolver_cpp",
    "parse_io_layout",
    "G_SCALE_FINITE",
    "G_SCALE_POSITIVE",
    "G_QUANT_STRUCT",
    "G_SCALE_FLOOR",
    "G_IO_SYMMETRY",
    "G_SINGLE_SUBGRAPH",
    "G_IO_LAYOUT",
    "G_OPSET",
    "G_ACCEL_COVERAGE",
    "G_RESOLVER_EXACT",
    "G_RESOLVER_ARITY",
    "G_RESOLVER_FIRMWARE",
    "G_RESOLVER_LEAN",
    "G_LOAD_ISOLATED",
]
