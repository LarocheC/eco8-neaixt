"""What the TFLite-Micro tree is *able* to register, read from its own source.

The naive way to answer "does TFLM support FULLY_CONNECTED?" is to line-regex
`micro_mutable_op_resolver.h` for `AddXxx() { return AddBuiltin(BuiltinOperator_YYY`.
That silently loses every method whose body spans more than one line -- which is
exactly the methods that take a `const TFLMRegistration&` default argument, i.e.
FULLY_CONNECTED, TRANSPOSE_CONV, CONV_2D, DEPTHWISE_CONV_2D, ADD, AVERAGE_POOL_2D,
BATCH_MATMUL, SOFTMAX, SVDF...  A resolver audit that loses those is worse than
no audit, so this module brace-matches method bodies instead.

The scanner is comment- and string-literal-aware so a `//` inside a string or a
`{` inside a comment cannot desynchronise the brace depth.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set, Tuple

RESOLVER_HEADER = os.path.join(
    "tensorflow", "lite", "micro", "micro_mutable_op_resolver.h"
)
KERNELS_DIR = os.path.join("tensorflow", "lite", "micro", "kernels")
XTENSA_KERNELS_DIR = os.path.join(KERNELS_DIR, "xtensa")


class TflmTreeUnavailable(RuntimeError):
    """The TFLM source tree is missing or does not look like a TFLM tree."""


# --------------------------------------------------------------------------
# tiny C++ scanner
# --------------------------------------------------------------------------


def _blank_comments(src: str) -> str:
    """Return `src` with comments replaced by spaces (newlines preserved).

    String and char literals are left alone, so `"http://x"` does not turn into
    a comment and `'{'` does not move the brace depth.
    """
    out: List[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i = min(i + 2, n)
            continue
        if c in ('"', "'"):
            quote = c
            out.append(c)
            i += 1
            while i < n:
                if src[i] == "\\" and i + 1 < n:
                    out.append(src[i])
                    out.append(src[i + 1])
                    i += 2
                    continue
                out.append(src[i])
                if src[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _match_delims(src: str, start: int, opener: str, closer: str) -> int:
    """Index just past the delimiter matching the one at `src[start]`.

    `src` must already be comment-blanked.  String literals are skipped.
    """
    assert src[start] == opener, "expected %r at %d, got %r" % (opener, start, src[start])
    depth = 0
    i = start
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
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced %r starting at offset %d" % (opener, start))


# --------------------------------------------------------------------------
# resolver header
# --------------------------------------------------------------------------

_METHOD_RE = re.compile(r"\bTfLiteStatus\s+(Add[A-Za-z0-9_]*)\s*\(")
_ADD_BUILTIN_RE = re.compile(r"\bAddBuiltin\s*\(\s*BuiltinOperator_([A-Za-z0-9_]+)")
_ADD_CUSTOM_RE = re.compile(r'\bAddCustom\s*\(\s*"([^"]*)"')
# AddEthosU registers a custom op whose name comes from a helper, not a literal.
_ADD_CUSTOM_FN_RE = re.compile(r"\bAddCustom\s*\(\s*(?:\w+::)*GetString_([A-Za-z0-9_]+)\s*\(")
_GETSTRING_DEF_RE = re.compile(
    r'\bGetString_([A-Za-z0-9_]+)\s*\(\s*\)\s*\{\s*return\s+"([^"]*)"\s*;'
)

# Methods that are the registration machinery itself, not op registrations.
_MACHINERY = {"AddBuiltin", "AddCustom"}


def _resolve_get_string(tflm_tree: str, symbol: str) -> Optional[str]:
    """Resolve `GetString_<symbol>()` to its string literal by scanning the tree.

    TFLM ships both a stub (`return "";`) and a real definition (`return
    "ethos-u";`) for ETHOSU depending on which kernel dir is compiled in, so
    prefer a non-empty literal.
    """
    best: Optional[str] = None
    kdir = os.path.join(tflm_tree, KERNELS_DIR)
    for root, _dirs, files in os.walk(kdir):
        for fname in files:
            if not fname.endswith((".cc", ".h")):
                continue
            try:
                with open(os.path.join(root, fname), "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if "GetString_" not in text:
                continue
            for m in _GETSTRING_DEF_RE.finditer(_blank_comments(text)):
                if m.group(1) != symbol:
                    continue
                value = m.group(2)
                if value:
                    return value
                if best is None:
                    best = value
    return best


def _custom_op_from_body(tflm_tree: str, body: str) -> Optional[str]:
    """`CUSTOM:<name>` for an AddCustom call in `body`, or None."""
    lit = _ADD_CUSTOM_RE.search(body)
    if lit:
        return "CUSTOM:%s" % lit.group(1)
    fn = _ADD_CUSTOM_FN_RE.search(body)
    if fn:
        resolved = _resolve_get_string(tflm_tree, fn.group(1))
        if resolved:
            return "CUSTOM:%s" % resolved
        # Still a real custom registration -- name it after the helper so the
        # method is never silently dropped from the count.
        return "CUSTOM:<GetString_%s()>" % fn.group(1)
    return None


def _resolver_header_path(tflm_tree: str) -> str:
    path = os.path.join(tflm_tree, RESOLVER_HEADER)
    if not os.path.isfile(path):
        raise TflmTreeUnavailable(
            "no TFLM resolver header at %s (expected <tflm_tree>/%s). "
            "Point --tflm-tree at a tflite-micro checkout." % (path, RESOLVER_HEADER)
        )
    return path


_resolver_cache: Dict[str, Dict[str, str]] = {}


def resolver_methods(tflm_tree: str) -> dict:
    """{"AddConv2D": "CONV_2D", "AddRfft": "CUSTOM:SignalRfft", ...}.

    The value is the op the method registers, read out of the *brace-matched*
    method body -- not guessed from the method name.  Custom-op registrations are
    keyed as ``CUSTOM:<name>`` so they compose with `ModelInfo.op_histogram`.
    """
    tflm_tree = os.path.abspath(tflm_tree)
    cached = _resolver_cache.get(tflm_tree)
    if cached is not None:
        return dict(cached)

    path = _resolver_header_path(tflm_tree)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    src = _blank_comments(raw)

    methods: Dict[str, str] = {}
    for m in _METHOD_RE.finditer(src):
        name = m.group(1)
        if name in _MACHINERY:
            continue
        # m.end() - 1 is the '(' of the parameter list.
        try:
            after_params = _match_delims(src, m.end() - 1, "(", ")")
        except ValueError:
            continue
        brace = src.find("{", after_params)
        if brace < 0:
            continue
        # A declaration (`;` before `{`) is not a definition.
        semi = src.find(";", after_params)
        if 0 <= semi < brace:
            continue
        try:
            end = _match_delims(src, brace, "{", "}")
        except ValueError:
            continue
        body = src[brace:end]

        b = _ADD_BUILTIN_RE.search(body)
        if b:
            methods[name] = b.group(1)
            continue
        c = _custom_op_from_body(tflm_tree, body)
        if c:
            methods[name] = c

    _resolver_cache[tflm_tree] = dict(methods)
    return methods


def supported_builtins(tflm_tree: str) -> Set[str]:
    """Set of BuiltinOperator names TFLM has an `Add*` method for.

    Custom-op registrations are excluded (see `supported_custom`).
    """
    return {v for v in resolver_methods(tflm_tree).values() if not v.startswith("CUSTOM:")}


def supported_custom(tflm_tree: str) -> Set[str]:
    """Set of ``CUSTOM:<name>`` ops TFLM has an `Add*` method for."""
    return {v for v in resolver_methods(tflm_tree).values() if v.startswith("CUSTOM:")}


def supported_ops(tflm_tree: str) -> Set[str]:
    """Builtins + custom, i.e. everything registerable, in op_histogram keying."""
    return set(resolver_methods(tflm_tree).values())


def method_for_op(tflm_tree: str) -> Dict[str, str]:
    """Inverse of `resolver_methods`: op name -> `Add*` method name."""
    out: Dict[str, str] = {}
    for method, op in resolver_methods(tflm_tree).items():
        # Prefer the shortest method name when several register the same op.
        if op not in out or len(method) < len(out[op]):
            out[op] = method
    return out


# --------------------------------------------------------------------------
# kernel sources
# --------------------------------------------------------------------------

_REGISTER_DEF_RE = re.compile(
    r"\bTFLMRegistration\s*(?:\*\s*)?Register_([A-Za-z0-9_]+)\s*\(\s*\)\s*\{"
)
_REGISTER_CALL_RE = re.compile(r"\bRegister_([A-Za-z0-9_]+)\s*\(")

_sources_cache: Dict[str, Dict[str, str]] = {}


def registration_symbols(tflm_tree: str) -> Dict[str, str]:
    """op name -> the `Register_*` symbol its `Add*` method installs.

    e.g. ``{"CONV_2D": "CONV_2D", "RELU": "RELU"}`` -- the symbol suffix usually
    equals the op name but not always (SVDF/variants), so it is read from the
    method body rather than assumed.
    """
    path = _resolver_header_path(tflm_tree)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        src = _blank_comments(fh.read())

    out: Dict[str, str] = {}
    for m in _METHOD_RE.finditer(src):
        name = m.group(1)
        if name in _MACHINERY:
            continue
        try:
            after_params = _match_delims(src, m.end() - 1, "(", ")")
        except ValueError:
            continue
        brace = src.find("{", after_params)
        if brace < 0:
            continue
        semi = src.find(";", after_params)
        if 0 <= semi < brace:
            continue
        try:
            end = _match_delims(src, brace, "{", "}")
        except ValueError:
            continue
        # The default argument (between the parens) carries Register_X() too.
        region = src[m.end() - 1 : end]
        b = _ADD_BUILTIN_RE.search(src[brace:end])
        c = None if b else _custom_op_from_body(tflm_tree, src[brace:end])
        if not b and not c:
            continue
        op = b.group(1) if b else c
        reg = _REGISTER_CALL_RE.search(region)
        if reg:
            out.setdefault(op, reg.group(1))
        else:
            out.setdefault(op, op)
    return out


def kernel_sources(tflm_tree: str) -> Dict[str, str]:
    """op name -> path of the kernel .cc that defines its `Register_*` symbol.

    The xtensa kernel directory wins over the portable one, because that is what
    the RT500 build actually compiles.  Ops with no kernel source at all are
    simply absent from the mapping.
    """
    tflm_tree = os.path.abspath(tflm_tree)
    cached = _sources_cache.get(tflm_tree)
    if cached is not None:
        return dict(cached)

    kdir = os.path.join(tflm_tree, KERNELS_DIR)
    if not os.path.isdir(kdir):
        raise TflmTreeUnavailable("no TFLM kernels dir at %s" % kdir)
    xdir = os.path.join(tflm_tree, XTENSA_KERNELS_DIR)

    # symbol -> source path, xtensa preferred.
    sym_to_src: Dict[str, str] = {}
    for directory, preferred in ((kdir, False), (xdir, True)):
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".cc") or fname.endswith("_test.cc"):
                continue
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    body = _blank_comments(fh.read())
            except OSError:
                continue
            for m in _REGISTER_DEF_RE.finditer(body):
                sym = m.group(1)
                if preferred or sym not in sym_to_src:
                    sym_to_src[sym] = fpath

    out: Dict[str, str] = {}
    for op, sym in registration_symbols(tflm_tree).items():
        src = sym_to_src.get(sym)
        if src:
            out[op] = src

    _sources_cache[tflm_tree] = dict(out)
    return out


def register_defining_files(tflm_tree: str) -> Dict[str, Set[str]]:
    """source path -> set of `Register_*` symbol suffixes it defines.

    `accel_map` uses this to refuse to attribute e.g. `transpose_conv.o` to the
    TRANSPOSE kernel just because the filenames share a prefix.
    """
    kdir = os.path.join(tflm_tree, KERNELS_DIR)
    xdir = os.path.join(tflm_tree, XTENSA_KERNELS_DIR)
    out: Dict[str, Set[str]] = {}
    for directory in (kdir, xdir):
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".cc") or fname.endswith("_test.cc"):
                continue
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    body = _blank_comments(fh.read())
            except OSError:
                continue
            syms = {m.group(1) for m in _REGISTER_DEF_RE.finditer(body)}
            if syms:
                out[fpath] = syms
    return out


def counts(tflm_tree: str) -> Tuple[int, int]:
    """(n_builtin_methods, n_custom_methods) -- the golden-self-test numbers."""
    methods = resolver_methods(tflm_tree)
    builtin = sum(1 for v in methods.values() if not v.startswith("CUSTOM:"))
    custom = len(methods) - builtin
    return builtin, custom


__all__ = [
    "TflmTreeUnavailable",
    "resolver_methods",
    "supported_builtins",
    "supported_custom",
    "supported_ops",
    "method_for_op",
    "registration_symbols",
    "kernel_sources",
    "register_defining_files",
    "counts",
]
