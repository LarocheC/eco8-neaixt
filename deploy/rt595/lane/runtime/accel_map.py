"""Which builtins actually get a HiFi4 (xa_nnlib) kernel in the shipped library.

This is derived from evidence, not from a hand-maintained table: an op is
"accelerated" iff the TFLM kernel translation unit that implements it (or one of
that kernel's private helper TUs) has an **undefined reference to an `xa_nn_*`
symbol** in the built `libtflm_rt500.a`.  If it never calls into xa_nnlib, it is
running the portable C reference kernel and the HiFi4 is idle.

Two wrinkles the naive "does `<op>.o` reference xa_nn_*" check gets wrong:

* CONV_2D's accel lives in `conv_hifi.o`, not `conv.o`.  Same for
  DEPTHWISE_CONV_2D (`depthwise_conv_hifi.o`), FULLY_CONNECTED
  (`fully_connected_int8.o`), the poolings (`pooling_int8.o`) and the reductions
  (`reduce_hifi.o`).  So helper TUs sharing the kernel's filename stem count.
* ...but `transpose_conv.o` must NOT be credited to TRANSPOSE just because
  `transpose_conv` starts with `transpose_`.  Helper TUs are therefore only
  credited if they are not themselves the registrar for some *other* op that the
  resolver can register.  (`fully_connected_int8.cc` defines only
  `Register_FULLY_CONNECTED_INT8`, a variant no `Add*` method installs, so it
  stays a helper; `transpose_conv.cc` defines `Register_TRANSPOSE_CONV`, which
  `AddTransposeConv` installs, so it does not.)

A kernel's accel can also live in a TU reached only through a local header --
`unidirectional_sequence_lstm.cc` includes `xtensa/lstm_eval.h` and all the
xa_nnlib calls are in `lstm_eval.o`.  Locally-included kernel headers are
therefore followed one level, under the same registrar guard.

Everything degrades to an empty set with an explanation if the toolchain, the
library or the TFLM tree is missing -- this module never raises.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Set, Tuple

from . import tflm_introspect as _ti

# Where the RI-2023.11 Xtensa tools live on the training box.  Only used as a
# fallback if xt-nm is not already on $PATH / in $XT_NM.
_DEFAULT_XT_BIN = "/home/clement/toolchains/tools/RI-2023.11-linux/XtensaTools/bin"

_ACCEL_PREFIX = "xa_nn"

# Populated by the last accelerated_builtins() call so the CLI can explain a
# degraded result instead of silently reporting 0% coverage.
LAST_STATUS: Dict[str, object] = {"ok": False, "reason": "not run"}


def find_nm() -> Optional[str]:
    """Locate an `nm` that can read Xtensa objects, or None."""
    env = os.environ.get("XT_NM")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    found = shutil.which("xt-nm")
    if found:
        return found
    candidate = os.path.join(_DEFAULT_XT_BIN, "xt-nm")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _nm_undefined(nm: str, lib_path: str) -> Tuple[Dict[str, Set[str]], Optional[str]]:
    """object basename -> set of undefined symbols, for every member of `lib_path`."""
    try:
        proc = subprocess.run(
            [nm, "--print-file-name", "--undefined-only", lib_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, "running %s failed: %s" % (nm, exc)

    out = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0 and not out.strip():
        return {}, "%s exited %d: %s" % (
            nm,
            proc.returncode,
            proc.stderr.decode("utf-8", "replace").strip()[:200],
        )

    table: Dict[str, Set[str]] = {}
    for line in out.splitlines():
        # "libtflm_rt500.a:conv_hifi.o:         U xa_nn_conv2d_std_sym8sxasym8s"
        if ".o:" not in line:
            continue
        _, _, rest = line.rpartition(":")
        obj = None
        for part in line.split(":"):
            if part.endswith(".o"):
                obj = os.path.basename(part)
        if obj is None:
            continue
        fields = rest.split()
        if not fields:
            continue
        sym = fields[-1]
        table.setdefault(obj, set()).add(sym)
    return table, None


def _accel_objects(table: Dict[str, Set[str]]) -> Set[str]:
    """Objects with at least one undefined `xa_nn_*` reference."""
    out = set()
    for obj, syms in table.items():
        if any(s.startswith(_ACCEL_PREFIX) for s in syms):
            out.add(obj)
    return out


_INCLUDE_RE = None  # set below (kept out of the import-time regex block for clarity)


def _local_include_stems(src_path: str) -> Set[str]:
    """Stems of `#include "tensorflow/lite/micro/kernels/..."` headers in `src_path`."""
    import re

    global _INCLUDE_RE
    if _INCLUDE_RE is None:
        _INCLUDE_RE = re.compile(r'#\s*include\s+"([^"]+)"')
    out: Set[str] = set()
    try:
        with open(src_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return out
    for m in _INCLUDE_RE.finditer(text):
        inc = m.group(1)
        if "/micro/kernels/" not in inc.replace(os.sep, "/"):
            continue
        stem = os.path.splitext(os.path.basename(inc))[0]
        out.add(stem)
    return out


def _candidate_objects(
    stem: str,
    accel_objs: Set[str],
    registrar_stems: Set[str],
    extra_stems: Optional[Set[str]] = None,
) -> List[str]:
    """Objects that count as "the kernel for `stem`".

    `stem.o` itself, plus `stem_*.o` helper TUs and any locally-included kernel
    header's TU -- but never one whose own stem registers a different op.
    """
    hits = []
    extra = extra_stems or set()
    for obj in sorted(accel_objs):
        base = obj[:-2] if obj.endswith(".o") else obj
        if base == stem:
            hits.append(obj)
        elif base in registrar_stems:
            continue
        elif base.startswith(stem + "_") or base in extra:
            hits.append(obj)
    return hits


def _cache_read(cache: Optional[str], lib_path: str, tflm_tree: str) -> Optional[dict]:
    """Load a cached map, or None if it is absent or stale.

    Staleness is keyed on both inputs the map is derived from: the archive
    (path + mtime + size) and the TFLM tree, since the tree is what maps an op to
    the kernel TU whose symbols we then look up.
    """
    if not cache or not os.path.isfile(cache):
        return None
    try:
        with open(cache, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        if blob.get("lib_path") != os.path.abspath(lib_path):
            return None
        if blob.get("tflm_tree") != os.path.abspath(tflm_tree):
            return None
        if abs(float(blob.get("lib_mtime", -1)) - os.path.getmtime(lib_path)) > 1e-6:
            return None
        if int(blob.get("lib_size", -1)) != os.path.getsize(lib_path):
            return None
    except OSError:
        return None
    return blob


def _cache_write(cache: Optional[str], blob: dict) -> None:
    if not cache:
        return
    try:
        parent = os.path.dirname(os.path.abspath(cache))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = cache + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
        os.replace(tmp, cache)
    except OSError:
        pass


def accelerated_builtins(
    lib_path: str, tflm_tree: str, cache: Optional[str] = None
) -> Set[str]:
    """Builtins with a HiFi4 xa_nnlib kernel in `lib_path`.

    Never raises.  On any missing prerequisite returns `set()` and records why in
    `LAST_STATUS` (`{"ok": bool, "reason": str, ...}`).
    """
    global LAST_STATUS

    if not lib_path or not os.path.isfile(lib_path):
        LAST_STATUS = {
            "ok": False,
            "reason": "built library not found at %r -- accel coverage unknown "
            "(pass --lib <libtflm_rt500.a>)" % lib_path,
        }
        return set()

    cached = _cache_read(cache, lib_path, tflm_tree)
    if cached is not None and cached.get("accelerated") is not None:
        LAST_STATUS = {
            "ok": True,
            "reason": "from cache %s" % cache,
            "lib_path": cached.get("lib_path"),
            "n_accel_objects": len(cached.get("accel_objects", [])),
            "cached": True,
            "detail": cached.get("detail", {}),
        }
        return set(cached["accelerated"])

    nm = find_nm()
    if nm is None:
        LAST_STATUS = {
            "ok": False,
            "reason": "no xt-nm on $PATH, in $XT_NM or at %s -- cannot read Xtensa "
            "objects, accel coverage unknown" % _DEFAULT_XT_BIN,
        }
        return set()

    try:
        sources = _ti.kernel_sources(tflm_tree)
        registrars = _ti.register_defining_files(tflm_tree)
    except _ti.TflmTreeUnavailable as exc:
        LAST_STATUS = {"ok": False, "reason": "%s -- accel coverage unknown" % exc}
        return set()
    except OSError as exc:  # pragma: no cover
        LAST_STATUS = {"ok": False, "reason": "reading TFLM tree failed: %s" % exc}
        return set()

    table, err = _nm_undefined(nm, lib_path)
    if err:
        LAST_STATUS = {"ok": False, "reason": "%s -- accel coverage unknown" % err}
        return set()

    accel_objs = _accel_objects(table)

    # A TU is off-limits as a helper only if it registers an op the resolver can
    # actually install -- variant registrars like Register_FULLY_CONNECTED_INT8
    # are private implementation details and stay creditable as helpers.
    resolver_symbols = set(_ti.registration_symbols(tflm_tree).values())
    registrar_stems = {
        os.path.splitext(os.path.basename(path))[0]
        for path, syms in registrars.items()
        if syms & resolver_symbols
    }

    accelerated: Set[str] = set()
    detail: Dict[str, List[str]] = {}
    for op, src in sources.items():
        stem = os.path.splitext(os.path.basename(src))[0]
        hits = _candidate_objects(
            stem,
            accel_objs,
            registrar_stems - {stem},
            _local_include_stems(src) - {stem},
        )
        if hits:
            accelerated.add(op)
            detail[op] = hits

    LAST_STATUS = {
        "ok": True,
        "reason": "read %d archive members via %s" % (len(table), nm),
        "lib_path": os.path.abspath(lib_path),
        "n_accel_objects": len(accel_objs),
        "cached": False,
        "detail": detail,
    }

    _cache_write(
        cache,
        {
            "lib_path": os.path.abspath(lib_path),
            "lib_mtime": os.path.getmtime(lib_path),
            "lib_size": os.path.getsize(lib_path),
            "tflm_tree": os.path.abspath(tflm_tree),
            "nm": nm,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "accel_objects": sorted(accel_objs),
            "accelerated": sorted(accelerated),
            "detail": {k: sorted(v) for k, v in detail.items()},
        },
    )
    return accelerated


__all__ = ["accelerated_builtins", "find_nm", "LAST_STATUS"]
