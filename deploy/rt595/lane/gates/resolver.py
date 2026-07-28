"""Resolver gates: the firmware's op resolver must match the graph exactly.

**G-RESOLVER-EXACT** is set equality **in both directions**, because the two
directions fail differently and both have already happened in this tree:

* **missing** (graph op with no registration) -- `AllocateTensors()` fails at
  boot with "Didn't find op for builtin opcode", the model never runs.
  `deploy/rt595/app/model_ops_micro.cpp` has no `AddFullyConnected`, so NSNet2
  cannot run on the board with the committed firmware resolver.
* **unused** (registration with no graph op) -- silent.  It costs a resolver
  slot and links a kernel that is never called.  The same file registers
  `AddTanh` for LiSenNet, whose int8 graph contains no TANH at all.

Counting `Add*` calls textually is not enough, because the compiler does not
count them textually.  Two shapes defeat a naive count, and both are ordinary
C++:

* a call inside ``#if 0`` / ``#ifdef <macro the RT595 build never defines>``
  -- never compiled, so the op is not registered;
* a call in a helper function nothing calls -- never executed, same result.

Both hide the headline case ("a resolver missing exactly one op the graph
uses") behind a single line.  So `parse_resolver_cpp` (a) strips ``#if 0``
blocks, (b) reports `Add*` calls inside any other preprocessor conditional as
CONDITIONAL rather than counting them, and (c) only counts calls that occur
inside the body of the function that declares the resolver.  Anything else is
reported, never silently credited.

**G-RESOLVER-ARITY** is a separate gate because it can genuinely fail to run.
`N` in ``MicroMutableOpResolver<N>`` is the compile-time capacity of the
registration array: if `N` is *smaller* than the number of `Add*` calls, the
surplus registrations return `kTfLiteError` at runtime and are dropped -- and
since nothing checks that return value, the failure only shows up as a missing
op much later.  When `N` is a named constant the gate resolves it from the same
translation unit; when it cannot, it SKIPs with a reason instead of reporting
"arity <?> matches", which is a positive claim about a check that did not
happen.

**G-RESOLVER-FIRMWARE** runs the same comparison against
`deploy/rt595/app/model_ops_micro.cpp`, the resolver that actually ships.  The
per-family files under `iss/resolvers/` are hand-written references; the sweep
used to audit only those, so the deploy-blocking fact that the firmware cannot
boot NSNet2 was one flag away and never taken.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import Gate, GateResult, HARD_FAIL, WARN, skip
from ..runtime.tflm_introspect import _blank_comments

G_RESOLVER_EXACT = Gate(
    id="G-RESOLVER-EXACT",
    title="Resolver registers every op the graph uses",
    severity=HARD_FAIL,
)

G_RESOLVER_LEAN = Gate(
    id="G-RESOLVER-LEAN",
    title="Resolver has no dead registrations",
    severity=WARN,
)

G_RESOLVER_ARITY = Gate(
    id="G-RESOLVER-ARITY",
    title="MicroMutableOpResolver<N> capacity fits the registrations",
    severity=HARD_FAIL,
)

G_RESOLVER_FIRMWARE = Gate(
    id="G-RESOLVER-FIRMWARE",
    title="The resolver that actually ships matches the graph",
    severity=HARD_FAIL,
)

# A *declaration* of the resolver object: `<...> name;` / `= ...` / `(...)`.
# Deliberately not a bare `MicroMutableOpResolver<...>` search, which also
# matches `static_cast<MicroMutableOpResolver<13> &>(base)` inside a helper.
_DECL_RE = re.compile(
    r"MicroMutableOpResolver\s*<\s*([^<>]*?)\s*>\s+([A-Za-z_]\w*)\s*[;=({]"
)
# Fallback for files that only mention the template (e.g. a cast); used for the
# arity only, never as the reachability anchor.
_ARITY_RE = re.compile(r"MicroMutableOpResolver\s*<\s*([^<>]*?)\s*>")
_ADD_CALL_RE = re.compile(r"\.\s*(Add[A-Za-z0-9_]+)\s*\(")
_ADD_CUSTOM_LIT_RE = re.compile(r'\A\s*"([^"]*)"')
_INT_LITERAL_RE = re.compile(r"\A\s*(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*\Z")

# `constexpr int kNumOps = 13;`, `static const size_t N = 13;`, `#define N 13`
_CONST_DEF_RE = re.compile(
    r"\b(?:constexpr|const)\s+(?:unsigned\s+|signed\s+)?"
    r"(?:int|size_t|unsigned|long|short|uint\d+_t|int\d+_t)\s+"
    r"([A-Za-z_]\w*)\s*=\s*([^;]+);"
)
_DEFINE_RE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+([^\n]+)$", re.M)

_COND_OPEN = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef)\b[ \t]*(.*)$")
_COND_CLOSE = re.compile(r"^[ \t]*#[ \t]*endif\b")


def _parse_int(text: str) -> Optional[int]:
    text = text.strip()
    if not _INT_LITERAL_RE.match(text):
        return None
    text = text.strip().rstrip("uUlL")
    try:
        return int(text, 0)
    except ValueError:  # pragma: no cover - the regex already constrains this
        return None


def _resolve_symbol(src: str, name: str, depth: int = 0) -> Optional[int]:
    """Resolve a `constexpr` / `const` / `#define` integer constant in this TU."""
    if depth > 4 or not re.fullmatch(r"[A-Za-z_]\w*", name or ""):
        return None
    for regex in (_CONST_DEF_RE, _DEFINE_RE):
        for m in regex.finditer(src):
            if m.group(1) != name:
                continue
            val = _parse_int(m.group(2))
            if val is not None:
                return val
            inner = m.group(2).strip()
            if re.fullmatch(r"[A-Za-z_]\w*", inner):
                return _resolve_symbol(src, inner, depth + 1)
    return None


def _line_offsets(src: str) -> Tuple[List[str], List[int]]:
    lines = src.split("\n")
    offsets: List[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    return lines, offsets


def _strip_if_zero(src: str) -> Tuple[str, int]:
    """Blank out every ``#if 0 ... #endif`` region (nesting-aware).

    Returns (source, n_blocks_removed).  Line count is preserved, so every
    offset-to-line mapping downstream stays honest.
    """
    chars = list(src)
    lines, offsets = _line_offsets(src)

    removed = 0
    i = 0
    while i < len(lines):
        m = _COND_OPEN.match(lines[i])
        if m and m.group(1) == "if" and m.group(2).strip() in ("0", "false"):
            depth = 1
            j = i + 1
            while j < len(lines) and depth:
                if _COND_OPEN.match(lines[j]):
                    depth += 1
                elif _COND_CLOSE.match(lines[j]):
                    depth -= 1
                j += 1
            end = offsets[j - 1] + len(lines[j - 1]) if j - 1 < len(lines) else len(src)
            for k in range(offsets[i], min(end, len(chars))):
                if chars[k] != "\n":
                    chars[k] = " "
            removed += 1
            i = j
            continue
        i += 1
    return "".join(chars), removed


def _conditional_regions(src: str) -> List[Tuple[int, int, str]]:
    """(start, end, guard) for every surviving preprocessor conditional region.

    An `#include`-guard shape (``#ifndef FOO`` immediately followed by
    ``#define FOO``) is not reported: it wraps the whole translation unit and
    guards nothing about registration.
    """
    lines, offsets = _line_offsets(src)

    regions: List[Tuple[int, int, str]] = []
    stack: List[Tuple[int, str, bool]] = []
    for i, line in enumerate(lines):
        m = _COND_OPEN.match(line)
        if m:
            guard = ("#%s %s" % (m.group(1), m.group(2))).strip()
            include_guard = False
            if m.group(1) == "ifndef":
                parts = m.group(2).split()
                sym = parts[0] if parts else ""
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                dm = re.match(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", nxt)
                include_guard = bool(sym and dm and dm.group(1) == sym)
            stack.append((offsets[i], guard, include_guard))
            continue
        if _COND_CLOSE.match(line) and stack:
            start, guard, include_guard = stack.pop()
            if not include_guard:
                regions.append((start, offsets[i] + len(line), guard))
    for start, guard, include_guard in stack:  # unterminated -- assume to EOF
        if not include_guard:
            regions.append((start, len(src), guard))
    return regions


def _enclosing_block(src: str, offset: int) -> Optional[Tuple[int, int]]:
    """(start, end) of the outermost ``{...}`` block containing `offset`.

    `src` must already be comment-blanked.  String and char literals are
    skipped so a brace inside one cannot move the depth.
    """
    depth = 0
    start = -1
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c in ('"', "'"):
            quote = c
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    if start <= offset < i + 1:
                        return (start, i + 1)
                    start = -1
        i += 1
    return None


def _fallback_map() -> Dict[str, str]:
    """`Add*` method name -> builtin op name, derived from the schema enum.

    Used only when no TFLM tree is available.  Matching is exact-but-for-
    underscores, so ``AddDepthwiseConv2D`` -> ``DEPTHWISE_CONV_2D`` and
    ``AddSelectV2`` -> ``SELECT_V2`` without any hand-written table.
    """
    try:
        from ..runtime.flatbuffer import builtin_operator_names
    except Exception:  # pragma: no cover
        return {}
    try:
        names = builtin_operator_names().values()
    except Exception:  # pragma: no cover - schema module absent
        return {}
    out: Dict[str, str] = {}
    for op in names:
        out["Add" + op.replace("_", "").upper()] = op
    return out


def _line_of(src: str, offset: int) -> int:
    return src.count("\n", 0, offset) + 1


def parse_resolver_cpp(
    path: str, method_map: Optional[Dict[str, str]] = None
) -> dict:
    """Parse a hand-written `MicroMutableOpResolver` .cpp.

    Only `Add*` calls the compiler will actually emit are counted:

    * ``#if 0`` blocks are stripped outright;
    * calls inside any other preprocessor conditional are returned under
      ``"conditional"`` and are NOT counted;
    * calls outside the body of the function that declares the resolver are
      returned under ``"unreachable"`` and are NOT counted.

    `method_map` should be `tflm_introspect.resolver_methods(tree)`; without it a
    schema-derived fallback is used and ``"used_fallback_map"`` is set.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    src = _blank_comments(raw)
    src, n_if_zero = _strip_if_zero(src)

    regions = _conditional_regions(src)

    # ---- arity ---------------------------------------------------------
    decl = _DECL_RE.search(src)
    arity_expr: Optional[str] = None
    receiver: Optional[str] = None
    if decl:
        arity_expr = decl.group(1)
        receiver = decl.group(2)
    else:
        loose = _ARITY_RE.search(src)
        arity_expr = loose.group(1) if loose else None

    arity: Optional[int] = None
    arity_source = "absent"
    if arity_expr is not None:
        arity = _parse_int(arity_expr)
        if arity is not None:
            arity_source = "literal"
        else:
            arity = _resolve_symbol(src, arity_expr.strip())
            arity_source = "resolved-constant" if arity is not None else "unresolved"

    # ---- the region the compiler actually emits ------------------------
    body: Optional[Tuple[int, int]] = None
    if decl:
        body = _enclosing_block(src, decl.start())
    body_found = body is not None

    def _classify(offset: int) -> Tuple[bool, Optional[str]]:
        """(reachable, guard) for a call at `offset`."""
        guard = None
        for start, end, g in regions:
            if start <= offset < end:
                guard = g
                break
        # With no enclosing function (a file-scope declaration, or no
        # declaration at all) there is nothing to be outside of, so do not
        # invent an unreachability finding.
        reachable = True if body is None else (body[0] <= offset < body[1])
        return reachable, guard

    used_fallback = method_map is None
    lookup = dict(method_map or {})
    if used_fallback:
        lookup = _fallback_map()
    else:
        # Belt and braces: a tree-derived map may lag a locally-patched header.
        for k, v in _fallback_map().items():
            lookup.setdefault(k, v)

    def _op_for(method: str, after: str) -> Optional[str]:
        if method == "AddCustom":
            lit = _ADD_CUSTOM_LIT_RE.match(after)
            return "CUSTOM:%s" % lit.group(1) if lit else None
        return lookup.get(method) or lookup.get(
            "Add" + method[3:].replace("_", "").upper()
        )

    methods: List[str] = []
    ops: List[str] = []
    unmapped: List[str] = []
    conditional: List[dict] = []
    unreachable: List[dict] = []

    for m in _ADD_CALL_RE.finditer(src):
        method = m.group(1)
        op = _op_for(method, src[m.end() : m.end() + 200])
        reachable, guard = _classify(m.start())
        record = {
            "method": method,
            "op": op,
            "line": _line_of(src, m.start()),
            "guard": guard,
        }
        if guard is not None:
            conditional.append(record)
            continue
        if not reachable:
            unreachable.append(record)
            continue
        methods.append(method)
        if op is None:
            unmapped.append(method)
            continue
        ops.append(op)

    seen: Set[str] = set()
    duplicates: List[str] = []
    for op in ops:
        if op in seen:
            duplicates.append(op)
        seen.add(op)

    return {
        "path": os.path.abspath(path),
        "arity": arity,
        "arity_expr": arity_expr,
        "arity_source": arity_source,
        "receiver": receiver,
        "body_found": body_found,
        "methods": methods,
        "ops": sorted(seen),
        "n_registrations": len(methods),
        "unmapped": unmapped,
        "duplicates": duplicates,
        "conditional": conditional,
        "unreachable": unreachable,
        "n_if_zero_blocks_stripped": n_if_zero,
        "used_fallback_map": used_fallback,
    }


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def _fmt_calls(records: Sequence[dict]) -> str:
    out = []
    for r in records[:6]:
        label = "%s()" % r["method"]
        if r.get("op"):
            label += " [%s]" % r["op"]
        label += " at line %d" % r["line"]
        if r.get("guard"):
            label += " under %s" % r["guard"]
        out.append(label)
    if len(records) > 6:
        out.append("(+%d more)" % (len(records) - 6))
    return ", ".join(out)


def _compare(ctx: dict, gate: Gate, prefix: str, strict_unused: bool = True) -> GateResult:
    """The shared graph-vs-resolver comparison, keyed by `prefix` in `ctx`.

    `strict_unused` distinguishes the two files this runs against.  A
    *per-family* resolver under `iss/resolvers/` is supposed to be exact for its
    one model, so a registration the graph never emits is a finding.  The
    *firmware* resolver is a single file that ships with one model out of
    several, so a surplus registration only wastes a slot -- it is reported, but
    it does not block, or every model in the sweep would fail the same way and
    the column would stop distinguishing "cannot boot" from "wastes a slot".
    """
    model = ctx.get("model")
    if model is None:
        return skip(gate, "no model in context")

    resolver_path = ctx.get(prefix + "path")
    resolver_ops = ctx.get(prefix + "ops")
    if resolver_ops is None:
        reason = ctx.get(prefix + "error") or (
            "no resolver .cpp paired with this model (pass --resolver <ops.cpp>)"
        )
        return skip(gate, "cannot check resolver: %s" % reason)

    graph_ops: Set[str] = set(model.op_histogram)
    reg_ops: Set[str] = set(resolver_ops)
    unmapped = list(ctx.get(prefix + "unmapped") or [])
    conditional = list(ctx.get(prefix + "conditional") or [])
    unreachable = list(ctx.get(prefix + "unreachable") or [])
    n_registrations = ctx.get(prefix + "n_registrations")
    if n_registrations is None:
        n_registrations = len(reg_ops)

    missing = sorted(graph_ops - reg_ops)
    unused = sorted(reg_ops - graph_ops)

    from ..runtime.tflm_introspect import method_for_op

    method_names: Dict[str, str] = {}
    tree = ctx.get("tflm_tree")
    if tree:
        try:
            method_names = method_for_op(tree)
        except Exception:
            method_names = {}

    # Which dropped call would have supplied a missing op?  That is the
    # difference between "you deleted a line" and "you hid one behind #ifdef".
    excuse: Dict[str, dict] = {}
    for rec in list(conditional) + list(unreachable):
        if rec.get("op") in missing:
            excuse.setdefault(rec["op"], rec)

    evidence = {
        "resolver_path": resolver_path,
        "graph_ops": sorted(graph_ops),
        "resolver_ops": sorted(reg_ops),
        "missing": missing,
        "missing_counts": {op: model.op_histogram[op] for op in missing},
        "unused": unused,
        "n_registrations": n_registrations,
        "unmapped_methods": unmapped,
        "duplicate_registrations": list(ctx.get(prefix + "duplicates") or []),
        "conditional_calls": conditional,
        "unreachable_calls": unreachable,
        "body_found": ctx.get(prefix + "body_found"),
    }

    problems: List[str] = []
    if missing:
        suggest = ", ".join(
            "%s (needs %s, x%d)"
            % (op, method_names.get(op, "Add<%s>" % op), model.op_histogram[op])
            for op in missing
        )
        problems.append(
            "MISSING %d registration(s): %s -- AllocateTensors() will fail at boot "
            "with \"Didn't find op for builtin opcode\"" % (len(missing), suggest)
        )
        for op, rec in sorted(excuse.items()):
            if rec.get("guard"):
                problems.append(
                    "%s() IS written at line %d but sits under %s, which this build "
                    "does not define -- the compiler never emits that registration"
                    % (rec["method"], rec["line"], rec["guard"])
                )
            else:
                problems.append(
                    "%s() IS written at line %d but outside the body of the function "
                    "that declares the resolver, so the firmware never executes it"
                    % (rec["method"], rec["line"])
                )
    unused_msg = (
        "UNUSED %d registration(s): %s -- the graph never emits these; they burn "
        "a resolver slot and link a kernel that is never called"
        % (len(unused), ", ".join(unused))
    )
    if unused and strict_unused:
        problems.append(unused_msg)
    if unmapped:
        problems.append(
            "UNRECOGNISED Add* call(s): %s -- could not be mapped to an op"
            % ", ".join(unmapped)
        )
    if evidence["duplicate_registrations"]:
        problems.append(
            "DUPLICATE registration(s): %s -- the second AddX() returns kTfLiteError"
            % ", ".join(evidence["duplicate_registrations"])
        )

    # Advisories are reported whether or not the gate fails: a registration the
    # compiler may or may not emit is never something to be silent about.
    advisories: List[str] = []
    if unused and not strict_unused:
        advisories.append(unused_msg + " (not a boot blocker for this model)")
    if conditional:
        advisories.append(
            "%d Add* call(s) are behind a preprocessor conditional and are NOT "
            "counted (lane does not evaluate the preprocessor): %s"
            % (len(conditional), _fmt_calls(conditional))
        )
    if unreachable:
        advisories.append(
            "%d Add* call(s) sit outside the resolver function's body and are NOT "
            "counted: %s" % (len(unreachable), _fmt_calls(unreachable))
        )

    name = os.path.basename(resolver_path) if resolver_path else "<resolver>"
    if not problems:
        # The denominator is the GRAPH's op count, not the resolver's: with
        # `strict_unused=False` a passing resolver can be a strict superset, and
        # "exactly the 14 op(s) the graph uses" about a 13-op graph is a false
        # statement made by a green gate.
        if unused:
            detail = "%s covers all %d op(s) the graph uses" % (name, len(graph_ops))
        else:
            detail = "%s registers exactly the %d op(s) the graph uses" % (
                name,
                len(graph_ops),
            )
        if advisories:
            detail += "; " + "; ".join(advisories)
        return GateResult(gate=gate, passed=True, detail=detail, evidence=evidence)

    return GateResult(
        gate=gate,
        passed=False,
        detail="%s: " % name + "; ".join(problems + advisories),
        evidence=evidence,
    )


def check_resolver_exact(ctx: dict) -> GateResult:
    """Hard-fail on the *missing* direction only -- the one that stops the boot.

    A missing registration is fatal: ``AllocateTensors()`` returns kTfLiteError
    with "Didn't find op for builtin opcode", and the model never runs.  A
    *surplus* registration is not fatal -- it costs a resolver slot and links a
    kernel nobody calls, which is worth saying but is not a deploy blocker.
    G-RESOLVER-LEAN reports that separately, as a warning.

    Splitting the two directions is deliberate.  Folding them together made the
    one genuinely healthy artifact in the tree (LiSenNet relu6-deep) come back
    BLOCKED against its own matching resolver over a single dead ``AddTanh`` --
    a hard-fail nobody would act on, on the model most likely to be re-audited.
    A gate that cries wolf on the healthy case is a gate people learn to ignore.
    """
    return _compare(ctx, G_RESOLVER_EXACT, "resolver_", strict_unused=False)


def check_resolver_lean(ctx: dict) -> GateResult:
    """Warn when a resolver registers ops the graph never emits.

    Each dead entry burns a ``MicroMutableOpResolver<N>`` slot (~28 B of static
    TFLMRegistration storage) and pulls in a kernel that is never called.  Worth
    trimming before flash-size matters; never a reason to block a build.
    """
    res = _compare(ctx, G_RESOLVER_LEAN, "resolver_", strict_unused=True)
    if res.skipped:
        return res
    unused = res.evidence.get("unused") or []
    if not unused:
        return GateResult(
            gate=G_RESOLVER_LEAN,
            passed=True,
            detail="no dead registrations -- every registered op is used by the graph",
            evidence=res.evidence,
        )
    n = res.evidence.get("n_registrations")
    detail = (
        "%d dead registration(s): %s -- the graph never emits these, so each burns a "
        "resolver slot (~28 B) and links a kernel that is never called%s"
        % (len(unused), ", ".join(unused),
           "" if not n else " (%d of %d registrations)" % (len(unused), n))
    )
    return GateResult(gate=G_RESOLVER_LEAN, passed=False, detail=detail,
                      evidence=res.evidence)


def check_resolver_firmware(ctx: dict) -> GateResult:
    """The same comparison against the resolver the firmware actually links.

    Fails on the *missing* direction only: a graph op with no registration means
    this artifact cannot boot on the board as built, which is the deploy-blocking
    fact the sweep used to hide.  A surplus registration is reported as an
    advisory rather than a block -- the shipping resolver serves one model at a
    time, so extra entries are expected for the other four.
    """
    return _compare(ctx, G_RESOLVER_FIRMWARE, "firmware_resolver_", strict_unused=False)


def check_resolver_arity(ctx: dict) -> GateResult:
    """Hard-fail if ``MicroMutableOpResolver<N>`` cannot hold the registrations.

    SKIPs -- visibly, with a reason -- when `N` is not an integer literal and
    could not be resolved in the same translation unit.  It must never report
    "arity <?> matches": that is a positive claim about a check that did not run.
    """
    if ctx.get("resolver_ops") is None and ctx.get("resolver_n_registrations") is None:
        reason = ctx.get("resolver_error") or (
            "no resolver .cpp paired with this model (pass --resolver <ops.cpp>)"
        )
        return skip(G_RESOLVER_ARITY, "cannot check arity: %s" % reason)

    resolver_path = ctx.get("resolver_path")
    name = os.path.basename(resolver_path) if resolver_path else "<resolver>"
    arity = ctx.get("resolver_arity")
    arity_expr = ctx.get("resolver_arity_expr")
    arity_source = ctx.get("resolver_arity_source")
    n_registrations = int(ctx.get("resolver_n_registrations") or 0)
    n_conditional = len(ctx.get("resolver_conditional") or [])

    evidence = {
        "resolver_path": resolver_path,
        "arity": arity,
        "arity_expr": arity_expr,
        "arity_source": arity_source,
        "n_registrations": n_registrations,
        "n_conditional_calls": n_conditional,
    }

    if arity is None:
        if arity_expr is None:
            return skip(
                G_RESOLVER_ARITY,
                "cannot check arity: %s declares no MicroMutableOpResolver<N> -- "
                "NOT checked" % name,
                **evidence
            )
        return skip(
            G_RESOLVER_ARITY,
            "cannot check arity: template argument %r in %s is not an integer "
            "literal and could not be resolved in this translation unit -- NOT "
            "checked" % (arity_expr, name),
            **evidence
        )

    # Conditional calls make the true registration count a range, not a number.
    lo, hi = n_registrations, n_registrations + n_conditional
    evidence["expected_range"] = [lo, hi]
    ambiguity = (
        " (%d further Add* call(s) are behind preprocessor conditionals, so the "
        "compiled count is somewhere in %d..%d)" % (n_conditional, lo, hi)
        if n_conditional
        else ""
    )

    if lo <= arity <= hi:
        return GateResult(
            gate=G_RESOLVER_ARITY,
            passed=True,
            detail="%s: MicroMutableOpResolver<%d> (%s) fits its %d registration(s)%s"
            % (name, arity, arity_source, n_registrations, ambiguity),
            evidence=evidence,
        )

    if arity < lo:
        detail = (
            "%s: ARITY too small: MicroMutableOpResolver<%d> but %d Add* calls -- the "
            "last %d registration(s) silently return kTfLiteError and are dropped%s"
            % (name, arity, n_registrations, n_registrations - arity, ambiguity)
        )
    else:
        detail = (
            "%s: ARITY oversized: MicroMutableOpResolver<%d> for %d Add* calls -- "
            "%d unused slot(s) of static TFLMRegistration storage%s"
            % (name, arity, n_registrations, arity - hi, ambiguity)
        )
    return GateResult(
        gate=G_RESOLVER_ARITY, passed=False, detail=detail, evidence=evidence
    )


check_resolver_exact.gate = G_RESOLVER_EXACT  # type: ignore[attr-defined]
check_resolver_lean.gate = G_RESOLVER_LEAN  # type: ignore[attr-defined]
check_resolver_arity.gate = G_RESOLVER_ARITY  # type: ignore[attr-defined]
check_resolver_firmware.gate = G_RESOLVER_FIRMWARE  # type: ignore[attr-defined]

__all__ = [
    "G_RESOLVER_EXACT",
    "G_RESOLVER_ARITY",
    "G_RESOLVER_FIRMWARE",
    "parse_resolver_cpp",
    "check_resolver_exact",
    "check_resolver_arity",
    "check_resolver_firmware",
]
