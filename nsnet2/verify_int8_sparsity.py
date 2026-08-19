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
  int8 sparsity comes out slightly *above* the FP32 mask's. For N:M that is
  trivially harmless — the constraint is *at most* N nonzeros per group. For a
  block pattern it does split blocks open (a kept 1x4 with one value rounded
  away holds 3 nonzeros), so what is checked there is the block *support*: the
  fraction of blocks holding at least one nonzero must stay within the pattern's
  budget. A block-packed kernel still stores that block whole; it just wastes a
  multiply on the new zero.
* **Granularity.** Density is checked on the graph as a whole, not per
  initializer. The export splits each packed (3H, K) GRU matrix into three
  per-gate (H, K) slices, and a matrix-global sparsity budget (block or
  unstructured, ``scope="matrix"``) distributes unevenly across those gates —
  slices ranging 62%-92% around an exact 80% are normal, not a defect. The
  structural checks are per-matrix, which is safe because slicing by rows cannot
  disturb groups that run along K.

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

    return None            # block support and unstructured are graph-level checks


def block_support(a: np.ndarray, desc: dict) -> tuple[int, int]:
    """(live blocks, total blocks) for the orientation that yields fewer live.

    A block is live if it holds at least one nonzero. Quantization can zero a
    value *inside* a kept block, which a block-packed kernel absorbs without
    trouble — it still stores the block whole. What would break the contract is
    more live blocks than the budget allows, and that is a graph-level property:
    a matrix-global block budget spreads unevenly across the per-gate slices the
    export produces, so a single slice's live fraction proves nothing.
    """
    b = desc["block"]
    best = None
    for x in (a, a.T):
        rows, K = x.shape
        nb = K // b
        if nb == 0:
            continue
        counts = (x[:, : nb * b] != 0).reshape(rows, nb, b).sum(-1)
        cand = (int((counts != 0).sum()), int(counts.size))
        if best is None or cand[0] < best[0]:
            best = cand
    return best or (0, 0)


def density_shortfall(zeros_fraction: float, desc: dict) -> float:
    """How far an achieved zero fraction falls short of the pattern's target.

    N:M pins density through the group constraint, so it needs no separate
    check. Block and unstructured patterns do — an all-dense matrix trivially
    satisfies "every block is whole". Quantization only ever *adds* zeros, so
    this is one-sided. Apply it to the graph as a whole (see module docstring on
    granularity), never to a single initializer.
    """
    if desc["family"] == "nm":
        return 0.0
    return max(0.0, desc["sparsity"] - zeros_fraction)


def check_matrix(a: np.ndarray, desc: dict) -> tuple[str, bool]:
    """Structural verdict for one int8 weight matrix. Density is checked on the
    graph as a whole by the caller, not here."""
    v0, v1 = _violations(a, desc, 0), _violations(a, desc, 1)
    if v0 is None and v1 is None:
        return "n/a", True                             # unstructured
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
    live_blocks = total_blocks = 0
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
        if desc["family"] == "block":
            lb, tb = block_support(arr, desc)
            live_blocks += lb
            total_blocks += tb
            verdict = f"{lb / tb:.3f} live" if tb else verdict

        print(f"{init.name[:38]:<40}{str(arr.shape):>13}"
              f"{zeros / arr.size:>9.4f}{verdict:>10}")

    print("-" * 72)
    if not checked:
        raise SystemExit("no int8 2-D weight initializers found — is this an int8 QDQ graph?")
    sparsity = 1 - nonzero / total
    short = density_shortfall(sparsity, desc)
    print(f"{checked} matrices, int8 sparsity {sparsity:.4f} "
          f"(FP32 mask target {desc['sparsity']:.4f})")
    if failures:
        print(f"FAILED: {failures} matrix/matrices break the pattern in int8")
        raise SystemExit(1)
    if short > 0.02:
        print(f"FAILED: graph is {100 * short:.1f}% short of the declared sparsity")
        raise SystemExit(1)
    if total_blocks:
        live_frac = live_blocks / total_blocks
        budget = 1.0 - desc["sparsity"]
        print(f"block support: {live_frac:.4f} live (budget {budget:.4f})")
        if live_frac > budget + 0.02:
            print(f"FAILED: {100 * (live_frac - budget):.1f}% more live blocks "
                  f"than the budget allows")
            raise SystemExit(1)
    print("pattern preserved through int8 quantization")


if __name__ == "__main__":
    main()
