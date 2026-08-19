"""Check that a sparsity pattern survives int8 quantization, in the int8 graph.

Symmetric per-channel weight quantization maps 0.0 to exactly 0, so a fixed mask
*should* pass through PTQ untouched — but "should" is not "does", and the whole
value of the mask to a generated kernel rests on it holding exactly. This reads
the quantized weight initializers out of the int8 ONNX and checks them directly.

Two wrinkles the check has to handle:

* **Orientation.** ONNX stores Gemm weights as (M, K) and MatMul weights as
  (K, M), so the axis the groups run along differs per initializer. Rather than
  track which is which, each matrix is tested both ways and passes if it
  conforms on either — an N:M-conforming matrix cannot conform on the wrong axis
  by accident at these sizes.
* **Extra zeros.** Quantization rounds some surviving small weights to zero, so
  int8 sparsity comes out slightly *above* the FP32 mask's. That is harmless:
  N:M requires *at most* N nonzeros per group, so extra zeros can never violate
  it, and a block pattern's blocks stay whole.

    python -m nsnet2.verify_int8_sparsity --onnx cp_ov_2to4/g_best.onnx --pattern 2:4
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import numpy_helper

from nsnet2.sparsity import parse_pattern


def _violations(a: np.ndarray, desc: dict, axis: int) -> int | None:
    """Violations assuming the groups run along ``axis``. None if not applicable."""
    x = a if axis == 1 else a.T                      # groups along the last axis
    rows, K = x.shape
    nz = x != 0

    if desc["family"] == "nm":
        g, n = desc["group"], desc["n"]
        ng = K // g
        if ng == 0:
            return None
        counts = nz[:, : ng * g].reshape(rows, ng, g).sum(-1)
        return int((counts > n).sum())

    if desc["family"] == "block":
        b = desc["block"]
        nb = K // b
        if nb == 0:
            return None
        counts = nz[:, : nb * b].reshape(rows, nb, b).sum(-1)
        return int(((counts != 0) & (counts != b)).sum())

    return None                                       # unstructured: no structure


def _density_shortfall(a: np.ndarray, desc: dict) -> float:
    """How far the achieved zero fraction falls short of the pattern's target.

    N:M pins density through the group constraint, but block and unstructured
    patterns do not — an all-dense matrix trivially satisfies "every block is
    whole". Quantization only ever *adds* zeros, so this is one-sided.
    """
    if desc["family"] == "nm":
        return 0.0
    zeros = float((a == 0).mean())
    return max(0.0, desc["sparsity"] - zeros)


def check_matrix(a: np.ndarray, desc: dict, *, tol: float = 0.02) -> tuple[str, bool]:
    """Verdict string and pass/fail for one int8 weight matrix."""
    v0, v1 = _violations(a, desc, 0), _violations(a, desc, 1)
    short = _density_shortfall(a, desc)

    if short > tol:
        return f"{100 * short:.0f}% short", False
    if v0 is None and v1 is None:
        return "OK (density)", True
    if (v0 == 0) or (v1 == 0):
        return "OK", True
    return f"{min(v for v in (v0, v1) if v is not None)} BAD", False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="int8 QDQ ONNX (e.g. cp_x/g_best.onnx)")
    ap.add_argument("--pattern", required=True, help="declared pattern, e.g. '2:4'")
    ap.add_argument("--min-size", type=int, default=4096,
                    help="skip initializers smaller than this (default 4096)")
    a = ap.parse_args()

    desc = parse_pattern(a.pattern)
    model = onnx.load(a.onnx)

    print(f"{a.onnx}   declared pattern: {a.pattern}")
    print(f"{'int8 weight initializer':<40}{'shape':>13}{'zeros':>9}{'verdict':>10}")
    print("-" * 72)

    total = nonzero = 0
    failures = 0
    checked = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.int8 or arr.ndim != 2 or arr.size < a.min_size:
            continue
        checked += 1
        zeros = int((arr == 0).sum())
        total += arr.size
        nonzero += arr.size - zeros

        verdict, ok = check_matrix(arr, desc)
        if not ok:
            failures += 1

        print(f"{init.name[:38]:<40}{str(arr.shape):>13}"
              f"{zeros / arr.size:>9.4f}{verdict:>10}")

    print("-" * 72)
    if not checked:
        raise SystemExit("no int8 2-D weight initializers found — is this an int8 QDQ graph?")
    sparsity = 1 - nonzero / total
    print(f"{checked} matrices, int8 sparsity {sparsity:.4f} "
          f"(FP32 mask target {desc['sparsity']:.4f})")
    if failures:
        print(f"FAILED: {failures} matrix/matrices break the pattern in int8")
        raise SystemExit(1)
    print("pattern preserved through int8 quantization")


if __name__ == "__main__":
    main()
