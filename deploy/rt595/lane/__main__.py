"""`python -m lane` -- the RT595 artifact auditor CLI.

    python -m lane audit <model.tflite> [--resolver <ops.cpp>] [--tflm-tree <dir>]
                                        [--lib <libtflm_rt500.a>] [--json out.json]
    python -m lane audit --all [--host-out deploy/rt595/host_out] [--json out.json]

Exit codes, so a Makefile or CI step can tell the three outcomes apart:

===  ==========================================================================
0    every deploy-blocking gate ran, and none failed (warnings allowed)
1    BLOCKED -- a hard-fail gate failed
3    INCOMPLETE -- nothing failed, but a hard-fail gate *could not run*
2    usage error, including a `--resolver` / `--lib` / `--tflm-tree` path that
     does not exist: a typo is a usage error, never a silent skip
===  ==========================================================================

3 is not 0 on purpose.  A skip means the auditor could not check something, and
that must masquerade as neither a pass nor a block.  `--allow-incomplete`
downgrades 3 to 0 for a box that knowingly lacks the toolchain; it changes the
exit code only, never the verdict word or the JSON.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # allow `python lane/__main__.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "lane"

from . import __version__
from .gates import parse_io_layout, parse_resolver_cpp, run_gates
from .report import AuditReport, exit_code, render_text, write_json
from .runtime import accel_map
from .runtime.flatbuffer import ModelInfo, load_model
from .runtime.litert_host import load_isolated
from .runtime import tflm_introspect as ti

# deploy/rt595/
_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_HOST_OUT = os.path.join(_DEPLOY_ROOT, "host_out")
DEFAULT_RESOLVER_DIR = os.path.join(_DEPLOY_ROOT, "iss", "resolvers")
FIRMWARE_RESOLVER = os.path.join(_DEPLOY_ROOT, "app", "model_ops_micro.cpp")
FIRMWARE_IO_LAYOUT = os.path.join(_DEPLOY_ROOT, "app", "model_io_layout.h")

# Model family -> (hand-written resolver basename, filename hints, IO count hint)
_FAMILIES: List[Tuple[str, str, Tuple[str, ...], Optional[int]]] = [
    ("LiSenNet", "lisennet_ops.cpp", ("lisennet", "relu6deep", "lisen"), 26),
    ("ConvFSENet", "convfsenet_ops.cpp", ("convfsenet", "convfse"), 10),
    ("NSNet2", "nsnet2_ops.cpp", ("nsnet2", "nsnet"), 2),
]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def default_tflm_tree() -> Optional[str]:
    """Best guess at a tflite-micro checkout, or None."""
    env = os.environ.get("LANE_TFLM_TREE")
    candidates: List[str] = [env] if env else []
    candidates.append(os.path.join(_DEPLOY_ROOT, "tflm", "tflite-micro"))
    candidates.extend(
        sorted(glob.glob("/tmp/claude-*/*/*/scratchpad/tflm_rt500/tflite-micro"))
    )
    candidates.extend(sorted(glob.glob(os.path.expanduser("~/**/tflm_rt500/tflite-micro"))))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, ti.RESOLVER_HEADER)):
            return os.path.abspath(c)
    return None


def default_lib() -> Optional[str]:
    """Best guess at the built libtflm_rt500.a, or None."""
    env = os.environ.get("LANE_TFLM_LIB")
    candidates: List[str] = [env] if env else []
    candidates.append(os.path.join(_DEPLOY_ROOT, "tflm", "libtflm_rt500.a"))
    candidates.extend(sorted(glob.glob("/tmp/claude-*/*/*/scratchpad/tflm_rt500/*.a")))
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def default_cache() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "lane", "accel_map.json")


def family_expected_io(family: Optional[str]) -> Optional[int]:
    """The IO-port count `_FAMILIES` records for a family, or None."""
    for name, _resolver, _hints, io in _FAMILIES:
        if name == family:
            return io
    return None


def infer_family(model: ModelInfo, path: str) -> Tuple[Optional[str], Optional[str], str]:
    """(family, resolver_path, why) for a model, using filename then IO count."""
    base = os.path.basename(path).lower()
    for family, resolver, hints, _io in _FAMILIES:
        if any(h in base for h in hints):
            rp = os.path.join(DEFAULT_RESOLVER_DIR, resolver)
            if os.path.isfile(rp):
                return family, rp, "matched filename hint %r" % family
            return family, None, "filename says %s but %s is missing" % (family, rp)

    n_io = len(model.inputs)
    matches = [f for f in _FAMILIES if f[3] == n_io]
    if len(matches) == 1:
        family, resolver, _hints, _io = matches[0]
        rp = os.path.join(DEFAULT_RESOLVER_DIR, resolver)
        if os.path.isfile(rp):
            return family, rp, "matched %d model inputs -> %s" % (n_io, family)
        return family, None, "IO count says %s but %s is missing" % (family, rp)

    return (
        None,
        None,
        "cannot infer a resolver: filename %r matches no known family and %d model "
        "inputs matches %d family/families -- pass --resolver explicitly"
        % (os.path.basename(path), n_io, len(matches)),
    )


def int8_models(host_out: str) -> List[str]:
    """Every int8 .tflite in `host_out` (i.e. excluding the .fp32 controls)."""
    out = []
    for p in sorted(glob.glob(os.path.join(host_out, "*.tflite"))):
        if p.endswith(".fp32.tflite"):
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------
# context assembly
# --------------------------------------------------------------------------


def build_context(
    model_path: str,
    resolver_path: Optional[str],
    tflm_tree: Optional[str],
    accelerated: Optional[set],
    accel_status: Optional[dict],
    do_load: bool,
    load_timeout: float,
    floor_threshold: float,
    accel_min_fraction: float,
    firmware_resolver: Optional[str] = None,
    io_layout_path: Optional[str] = None,
    family: Optional[str] = None,
) -> Tuple[dict, List[str]]:
    """Assemble the dict the gates consume.  Never raises on tooling problems."""
    notes: List[str] = []
    model = load_model(model_path)

    # Every gate reads subgraph 0 only.  On a multi-subgraph model the ops,
    # tensors and IO of subgraphs 1..N are invisible to ALL of them -- opset,
    # resolver, quantization, accel coverage and the IO summary alike.  The note
    # is a courtesy; G-SINGLE-SUBGRAPH is what actually stops a partial read
    # being reported as a clean model.
    if model.n_subgraphs > 1:
        notes.append(
            "model has %d subgraphs; every gate inspects SUBGRAPH 0 ONLY, so the "
            "ops, tensors, quantization and IO of the other %d are NOT audited -- "
            "opset, resolver, scale and accel-coverage results all describe "
            "subgraph 0 alone (G-SINGLE-SUBGRAPH blocks on this)"
            % (model.n_subgraphs, model.n_subgraphs - 1)
        )

    ctx: dict = {
        "model": model,
        "tflm_tree": tflm_tree,
        "floor_threshold": floor_threshold,
        "accel_min_fraction": accel_min_fraction,
        "accelerated": accelerated or set(),
        "accel_status": accel_status or {},
        "family": family,
        "family_expected_io": family_expected_io(family),
    }

    method_map: Optional[Dict[str, str]] = None
    if tflm_tree:
        try:
            method_map = ti.resolver_methods(tflm_tree)
            ctx["supported_ops"] = ti.supported_ops(tflm_tree)
            n_b, n_c = ti.counts(tflm_tree)
            notes.append(
                "TFLM tree exposes %d builtin + %d custom Add* methods (brace-matched)"
                % (n_b, n_c)
            )
        except (ti.TflmTreeUnavailable, OSError) as exc:
            ctx["tflm_tree_error"] = str(exc)
            notes.append("TFLM tree unusable: %s" % exc)
    else:
        ctx["tflm_tree_error"] = (
            "no tflite-micro checkout found (set $LANE_TFLM_TREE or pass --tflm-tree)"
        )
        notes.append(ctx["tflm_tree_error"])

    def _load_resolver(path: str, prefix: str) -> None:
        try:
            parsed = parse_resolver_cpp(path, method_map)
        except OSError as exc:
            ctx[prefix + "error"] = "could not read %s: %s" % (path, exc)
            notes.append(ctx[prefix + "error"])
            return
        ctx[prefix + "path"] = parsed["path"]
        ctx[prefix + "ops"] = set(parsed["ops"])
        ctx[prefix + "arity"] = parsed["arity"]
        ctx[prefix + "arity_expr"] = parsed["arity_expr"]
        ctx[prefix + "arity_source"] = parsed["arity_source"]
        ctx[prefix + "unmapped"] = parsed["unmapped"]
        ctx[prefix + "duplicates"] = parsed["duplicates"]
        ctx[prefix + "n_registrations"] = parsed["n_registrations"]
        ctx[prefix + "methods_used"] = parsed["methods"]
        ctx[prefix + "conditional"] = parsed["conditional"]
        ctx[prefix + "unreachable"] = parsed["unreachable"]
        ctx[prefix + "body_found"] = parsed["body_found"]
        if parsed["used_fallback_map"]:
            notes.append(
                "resolver Add* names mapped via the schema enum fallback "
                "(no TFLM tree) -- less reliable than a tree-derived map"
            )
        if parsed["n_if_zero_blocks_stripped"]:
            notes.append(
                "%s: %d `#if 0` block(s) stripped before counting registrations"
                % (os.path.basename(path), parsed["n_if_zero_blocks_stripped"])
            )
        if not parsed["body_found"]:
            notes.append(
                "%s: no enclosing function found for the MicroMutableOpResolver "
                "declaration, so every Add* call in the file was counted -- "
                "reachability was NOT checked" % os.path.basename(path)
            )

    if resolver_path:
        _load_resolver(resolver_path, "resolver_")

    # The resolver that actually ships is audited unconditionally, not only when
    # no --resolver was given: `--all` used to bless the hand-written per-family
    # files while app/model_ops_micro.cpp -- the one on the board -- went
    # unread, which is how "the firmware cannot boot NSNet2" stayed invisible.
    if firmware_resolver:
        _load_resolver(firmware_resolver, "firmware_resolver_")

    if io_layout_path:
        try:
            ctx["io_layout"] = parse_io_layout(io_layout_path)
        except OSError as exc:
            ctx["io_layout_error"] = "could not read %s: %s" % (io_layout_path, exc)
            notes.append(ctx["io_layout_error"])

    if do_load:
        ctx["load_result"] = load_isolated(model_path, timeout=load_timeout)

    return ctx, notes


def audit_one(
    model_path: str,
    resolver_path: Optional[str],
    resolver_note: Optional[str],
    tflm_tree: Optional[str],
    lib_path: Optional[str],
    accelerated: Optional[set],
    accel_status: Optional[dict],
    do_load: bool,
    load_timeout: float,
    floor_threshold: float,
    accel_min_fraction: float,
    firmware_resolver: Optional[str] = None,
    io_layout_path: Optional[str] = None,
    family: Optional[str] = None,
) -> AuditReport:
    ctx, notes = build_context(
        model_path,
        resolver_path,
        tflm_tree,
        accelerated,
        accel_status,
        do_load,
        load_timeout,
        floor_threshold,
        accel_min_fraction,
        firmware_resolver=firmware_resolver,
        io_layout_path=io_layout_path,
        family=family,
    )
    if resolver_note:
        notes.insert(0, resolver_note)
    if accel_status and not accel_status.get("ok"):
        notes.append("accel map: %s" % accel_status.get("reason"))

    results = run_gates(ctx)
    model: ModelInfo = ctx["model"]

    quantized = sum(1 for t in model.tensors if t.is_quantized)
    report = AuditReport(
        model_path=os.path.abspath(model_path),
        results=results,
        notes=notes,
        summary={
            "path": os.path.abspath(model_path),
            "ops": "%d nodes / %d distinct  (%s)"
            % (
                model.n_ops,
                len(model.op_histogram),
                ", ".join(
                    "%s x%d" % (k, v)
                    for k, v in sorted(
                        model.op_histogram.items(), key=lambda kv: (-kv[1], kv[0])
                    )
                ),
            ),
            "io": "%d in / %d out, %d tensors (%d quantized), %d subgraph(s)"
            % (
                len(model.inputs),
                len(model.outputs),
                len(model.tensors),
                quantized,
                model.n_subgraphs,
            ),
            "resolver": resolver_path or "(none)",
            "firmware": firmware_resolver or "(not found)",
            "io_layout": io_layout_path or "(not found)",
            "tflm_tree": tflm_tree or "(not found)",
            "lib": lib_path or "(not found)",
        },
    )
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lane",
        description="Flatbuffer-only deploy auditor for the NXP RT595 lane.",
    )
    p.add_argument("--version", action="version", version="lane %s" % __version__)
    sub = p.add_subparsers(dest="command")

    a = sub.add_parser("audit", help="audit one or all .tflite artifacts")
    a.add_argument("model", nargs="?", help="path to a .tflite")
    a.add_argument("--all", action="store_true", help="audit every int8 model in --host-out")
    a.add_argument("--host-out", default=DEFAULT_HOST_OUT, help="dir of exported .tflite")
    a.add_argument("--resolver", help="MicroMutableOpResolver .cpp to check against")
    a.add_argument(
        "--firmware-resolver",
        default=FIRMWARE_RESOLVER,
        help="the resolver the firmware links (default app/model_ops_micro.cpp)",
    )
    a.add_argument(
        "--io-layout",
        default=FIRMWARE_IO_LAYOUT,
        help="generated app/model_io_layout.h to cross-check the model against",
    )
    a.add_argument("--tflm-tree", help="tflite-micro checkout to read the op list from")
    a.add_argument("--lib", help="built libtflm_rt500.a for the HiFi4 accel map")
    a.add_argument("--accel-cache", default=default_cache(), help="JSON cache for the accel map")
    a.add_argument("--no-accel-cache", action="store_true", help="ignore/skip the accel cache")
    a.add_argument("--json", dest="json_out", help="also write the full report as JSON")
    a.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit 0 instead of 3 when a deploy-blocking gate could not run "
        "(the verdict still reads INCOMPLETE)",
    )
    a.add_argument("--no-load", action="store_true", help="skip the subprocess interpreter load")
    a.add_argument("--load-timeout", type=float, default=120.0)
    a.add_argument("--floor-threshold", type=float, default=1e-11)
    a.add_argument("--accel-min-fraction", type=float, default=0.60)
    a.add_argument("-v", "--verbose", action="store_true", help="dump gate evidence")
    a.add_argument("--width", type=int, default=96)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "audit":
        build_parser().print_help()
        return 2

    if args.all and args.model:
        print("lane: --all takes no positional model", file=sys.stderr)
        return 2
    if not args.all and not args.model:
        print("lane: give a model path or --all", file=sys.stderr)
        return 2

    # A mistyped path must be a usage error, not a silent skip.  `--resolver
    # nsnet2_opps.cpp` used to SKIP G-RESOLVER-EXACT and print a green verdict;
    # the check that was asked for simply never happened.  Only *explicitly
    # passed* paths are validated -- an auto-discovered default that is missing
    # still degrades to a visible SKIP (and therefore an INCOMPLETE verdict).
    for flag, value, kind in (
        ("--resolver", args.resolver, "file"),
        ("--lib", args.lib, "file"),
        ("--tflm-tree", args.tflm_tree, "tree"),
        (
            "--firmware-resolver",
            args.firmware_resolver
            if args.firmware_resolver != FIRMWARE_RESOLVER
            else None,
            "file",
        ),
        (
            "--io-layout",
            args.io_layout if args.io_layout != FIRMWARE_IO_LAYOUT else None,
            "file",
        ),
    ):
        if not value:
            continue
        if kind == "file":
            if not os.path.isfile(value):
                print(
                    "lane: %s %s is not a readable file" % (flag, value),
                    file=sys.stderr,
                )
                return 2
            if not os.access(value, os.R_OK):
                print("lane: %s %s is not readable" % (flag, value), file=sys.stderr)
                return 2
        else:
            if not os.path.isdir(value):
                print("lane: %s %s is not a directory" % (flag, value), file=sys.stderr)
                return 2
            if not os.path.isfile(os.path.join(value, ti.RESOLVER_HEADER)):
                print(
                    "lane: --tflm-tree %s has no %s -- not a tflite-micro checkout"
                    % (value, ti.RESOLVER_HEADER),
                    file=sys.stderr,
                )
                return 2

    tflm_tree = args.tflm_tree or default_tflm_tree()
    lib_path = args.lib or default_lib()
    firmware_resolver = (
        args.firmware_resolver if os.path.isfile(args.firmware_resolver or "") else None
    )
    io_layout_path = args.io_layout if os.path.isfile(args.io_layout or "") else None

    cache = None if args.no_accel_cache else args.accel_cache
    accelerated: set = set()
    accel_status: dict = {"ok": False, "reason": "not attempted"}
    if lib_path and tflm_tree:
        accelerated = accel_map.accelerated_builtins(lib_path, tflm_tree, cache)
        accel_status = dict(accel_map.LAST_STATUS)
    else:
        accel_status = {
            "ok": False,
            "reason": "need both a built library (%s) and a TFLM tree (%s)"
            % (lib_path or "missing", tflm_tree or "missing"),
        }

    if args.all:
        models = int8_models(args.host_out)
        if not models:
            print("lane: no int8 .tflite under %s" % args.host_out, file=sys.stderr)
            return 2
    else:
        models = [args.model]

    reports: List[AuditReport] = []
    for path in models:
        if not os.path.isfile(path):
            print("lane: no such model %s" % path, file=sys.stderr)
            return 2
        try:
            probe = load_model(path)
        except (OSError, ValueError, ImportError) as exc:
            print("lane: cannot parse %s: %s" % (path, exc), file=sys.stderr)
            return 2

        resolver = args.resolver
        family, inferred, why = infer_family(probe, path)
        note: Optional[str]
        if resolver:
            note = "resolver given explicitly: %s" % resolver
        else:
            resolver = inferred
            note = why
        if firmware_resolver:
            note += (
                "; G-RESOLVER-FIRMWARE separately audits %s, the resolver that "
                "actually ships" % os.path.relpath(firmware_resolver, _DEPLOY_ROOT)
            )

        reports.append(
            audit_one(
                path,
                resolver,
                note,
                tflm_tree,
                lib_path,
                accelerated,
                accel_status,
                not args.no_load,
                args.load_timeout,
                args.floor_threshold,
                args.accel_min_fraction,
                firmware_resolver=firmware_resolver,
                io_layout_path=io_layout_path,
                family=family,
            )
        )

    sys.stdout.write(render_text(reports, width=args.width, verbose=args.verbose))
    sys.stdout.flush()

    if args.json_out:
        write_json(reports, args.json_out, allow_incomplete=args.allow_incomplete)
        print("wrote %s" % os.path.abspath(args.json_out))

    return exit_code(reports, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
