"""Rendering: a terminal table for humans, JSON for CI.

Both renderers take the same `AuditReport`, so the JSON can never drift from
what was printed -- and both get their exit code from `exit_code()` here, so the
number CI reads can never drift from the word a human reads either.

There are four verdicts, not three.  ``CLEAR`` means every hard-fail gate ran
and passed.  ``INCOMPLETE`` means nothing hard-failed *but at least one
hard-fail gate could not run*, so the audit does not support a claim of health:
"we checked nothing" and "we checked everything and it is fine" used to print
the same word and the same exit 0, and two ordinary CLI typos were enough to get
there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .gates import GateResult

BLOCKED = "BLOCKED"
INCOMPLETE = "INCOMPLETE"
PASS_WITH_WARNINGS = "PASS-WITH-WARNINGS"
CLEAR = "CLEAR"

# Exit codes.  Distinct on purpose: a CI step must be able to tell "this
# artifact is broken" from "this box could not run the checks".
EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_INCOMPLETE = 3

_STATUS_ORDER = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}

# Plain ASCII: this output gets pasted into handover notes and CI logs.
_STATUS_TAG = {"PASS": "[ ok ]", "FAIL": "[FAIL]", "WARN": "[warn]", "SKIP": "[skip]"}


@dataclass
class AuditReport:
    """One model's verdict."""

    model_path: str
    results: List[GateResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    summary: Dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.model_path)

    @property
    def hard_fails(self) -> List[GateResult]:
        return [r for r in self.results if r.is_hard_fail]

    @property
    def warns(self) -> List[GateResult]:
        return [r for r in self.results if not r.passed and not r.is_hard_fail]

    @property
    def skips(self) -> List[GateResult]:
        return [r for r in self.results if r.skipped]

    @property
    def hard_skips(self) -> List[GateResult]:
        """Hard-fail-severity gates that could not run.

        These are the ones that must never be laundered into a green verdict:
        a skipped G-OPSET is "nobody looked at the op set", not "the op set is
        fine".  A skipped *warn* gate is not in here -- it cannot block a deploy
        either way, so it does not make the audit incomplete.
        """
        return [r for r in self.results if r.is_hard_skip]

    @property
    def verdict(self) -> str:
        if self.hard_fails:
            return BLOCKED
        if self.hard_skips:
            return INCOMPLETE
        if self.warns:
            return PASS_WITH_WARNINGS
        return CLEAR

    @property
    def complete(self) -> bool:
        """True when every hard-fail gate actually ran."""
        return not self.hard_skips

    def to_dict(self) -> dict:
        return {
            "model": self.model_path,
            "name": self.name,
            "verdict": self.verdict,
            "complete": self.complete,
            "n_hard_fails": len(self.hard_fails),
            "n_warns": len(self.warns),
            "n_skips": len(self.skips),
            "n_hard_skips": len(self.hard_skips),
            "skipped_hard_gates": [
                {"id": r.gate.id, "title": r.gate.title, "reason": r.detail}
                for r in self.hard_skips
            ],
            "summary": self.summary,
            "notes": self.notes,
            "gates": [r.to_dict() for r in self.results],
        }


def _wrap(text: str, width: int, indent: str) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = w if not cur else cur + " " + w
    if cur:
        lines.append(cur)
    return [indent + ln for ln in lines] or [indent]


def render_text(
    reports: Sequence[AuditReport], width: int = 96, verbose: bool = False
) -> str:
    """Human-readable report for one or many models."""
    out: List[str] = []
    for rep in reports:
        out.append("=" * width)
        out.append("%s   [%s]" % (rep.name, rep.verdict))
        out.append("-" * width)
        for key in (
            "path",
            "ops",
            "io",
            "resolver",
            "firmware",
            "io_layout",
            "tflm_tree",
            "lib",
        ):
            val = rep.summary.get(key)
            if val:
                out.extend(_wrap("%-9s %s" % (key + ":", val), width, ""))
        for note in rep.notes:
            out.extend(_wrap("note:     " + note, width, ""))
        out.append("")

        ordered = sorted(
            rep.results, key=lambda r: (_STATUS_ORDER.get(r.status, 9), r.gate.id)
        )
        for r in ordered:
            tag = _STATUS_TAG.get(r.status, "[ ?? ]")
            out.append("%s %-18s %s" % (tag, r.gate.id, r.gate.title))
            out.extend(_wrap(r.detail, width - 9, " " * 9))
            if verbose and r.evidence:
                blob = json.dumps(_jsonable(r.evidence), indent=2, sort_keys=True)
                for line in blob.splitlines():
                    out.append("         | " + line)
            out.append("")

        if rep.hard_skips:
            out.extend(
                _wrap(
                    "NOT CHECKED: %d deploy-blocking gate(s) could not run -- %s. "
                    "Nothing in this report is evidence that they would have "
                    "passed."
                    % (
                        len(rep.hard_skips),
                        ", ".join(r.gate.id for r in rep.hard_skips),
                    ),
                    width,
                    "",
                )
            )
        out.append(
            "verdict: %s   (%d hard-fail, %d warn, %d skip [%d deploy-blocking], "
            "%d pass)"
            % (
                rep.verdict,
                len(rep.hard_fails),
                len(rep.warns),
                len(rep.skips),
                len(rep.hard_skips),
                len([r for r in rep.results if r.status == "PASS"]),
            )
        )
        out.append("")

    if len(reports) > 1:
        out.append("=" * width)
        out.append("SUMMARY")
        out.append("-" * width)
        namew = max(len(r.name) for r in reports)
        for rep in reports:
            flags = " ".join(
                "%s=%s" % (r.gate.id.replace("G-", ""), r.status)
                for r in sorted(rep.results, key=lambda x: x.gate.id)
            )
            out.append(
                "  %-*s  %-19s %s" % (namew, rep.name, rep.verdict, flags)
            )
        blocked = [r.name for r in reports if r.verdict == BLOCKED]
        incomplete = [r.name for r in reports if r.verdict == INCOMPLETE]
        out.append("")
        out.append(
            "  %d/%d models blocked%s"
            % (
                len(blocked),
                len(reports),
                (": " + ", ".join(blocked)) if blocked else "",
            )
        )
        if incomplete:
            out.append(
                "  %d/%d models INCOMPLETE (a deploy-blocking gate could not run): %s"
                % (len(incomplete), len(reports), ", ".join(incomplete))
            )
        out.append("")
    return "\n".join(out)


def _jsonable(obj):
    """Make evidence JSON-safe (sets, tuples, numpy scalars)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else list(obj)
        return [_jsonable(v) for v in items]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # pragma: no cover
            pass
    return str(obj)


def exit_code(reports: Sequence[AuditReport], allow_incomplete: bool = False) -> int:
    """The single definition of the process exit status.

    ``1`` beats ``3``: if anything is genuinely broken, say so first.  ``3`` --
    a hard-fail gate could not run -- is *not* 0, because "the auditor could not
    check the thing that blocks deploys" is not a green build.  `allow_incomplete`
    exists for a box that knowingly lacks the toolchain; it changes the exit
    code only, never the verdict word or the JSON.
    """
    if any(r.hard_fails for r in reports):
        return EXIT_BLOCKED
    if any(r.hard_skips for r in reports) and not allow_incomplete:
        return EXIT_INCOMPLETE
    return EXIT_OK


def render_json(
    reports: Sequence[AuditReport], indent: int = 2, allow_incomplete: bool = False
) -> str:
    skipped_hard = sorted(
        {r.gate.id for rep in reports for r in rep.hard_skips}
    )
    payload = {
        "tool": "lane",
        "n_models": len(reports),
        "blocked": [r.name for r in reports if r.verdict == BLOCKED],
        "incomplete": [r.name for r in reports if r.verdict == INCOMPLETE],
        "complete": all(r.complete for r in reports),
        "n_skips": sum(len(r.skips) for r in reports),
        "n_hard_skips": sum(len(r.hard_skips) for r in reports),
        # The gates a CI job must not read `blocked: []` without looking at.
        "skipped_hard_gates": skipped_hard,
        "allow_incomplete": bool(allow_incomplete),
        "exit_code": exit_code(reports, allow_incomplete=allow_incomplete),
        "reports": [_jsonable(r.to_dict()) for r in reports],
    }
    return json.dumps(payload, indent=indent, sort_keys=False)


def write_json(
    reports: Sequence[AuditReport],
    path: str,
    indent: int = 2,
    allow_incomplete: bool = False,
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_json(reports, indent=indent, allow_incomplete=allow_incomplete))
        fh.write("\n")


__all__ = [
    "AuditReport",
    "render_text",
    "render_json",
    "write_json",
    "exit_code",
    "BLOCKED",
    "INCOMPLETE",
    "PASS_WITH_WARNINGS",
    "CLEAR",
    "EXIT_OK",
    "EXIT_BLOCKED",
    "EXIT_INCOMPLETE",
]
