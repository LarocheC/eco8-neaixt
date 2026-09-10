"""Build the ConvFSENet Monarch sweep result tables from the run artifacts.

Reads each arm's training log (validation PESQ trajectory), its config (params
and MACs/frame, recomputed from the built model rather than trusted) and its
eval log (FP32/int8 PESQ + RTF), and emits the markdown tables
RESULTS_CONVFSENET.md carries.

Two quality statistics per arm, deliberately:

* **best** — the max over the 19 validations, which is the statistic ``g_best``
  selects on and the one every earlier table in this repo reports. It is
  mildly optimistic by construction.
* **last-5 mean** — the mean of the final five validations, which no selection
  touched. Reported alongside because the selection margin is not uniform
  across arms here (0.004 to 0.035), so a comparison on ``best`` alone could
  be reading the spread of the tail rather than the model.

Usage:  python -m convfsenet.sweep_report [--arms ...] [--md]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from torch import nn

PESQ_RE = re.compile(r"PESQ Score: ([\d.]+)")
EVAL_DUAL_RE = re.compile(r"PESQ FP32=([\d.]+),\s*primary=([\d.]+),\s*delta=(-?[\d.]+)")
EVAL_RTF_RE = re.compile(r"RTF(?:\s+primary)?=([\d.]+)")

# arm -> (label, matched partner). The sweep is a set of MAC-matched pairs.
PAIRS = [
    ("cfs_mon_nb4", "cfs_dense_r146"),
    ("cfs_mon_nb8", "cfs_dense_r108"),
    ("cfs_mon_nb16", "cfs_dense_r83"),
    ("cfs_mon_nb32", "cfs_dense_r67"),
]
ANCHOR = "cfs_dense"


def macs_and_params(cp_dir: Path):
    """Recompute from the built model — never trust a number in a comment."""
    from convfsenet.layers import BlockdiagPointwise, MonarchPointwise
    from convfsenet.model import build_causal_model

    h = json.loads((cp_dir / "config.json").read_text())
    model = build_causal_model(h)
    macs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            macs += (m.in_channels // m.groups) * m.out_channels * m.kernel_size[0]
        elif isinstance(m, MonarchPointwise):
            macs += m.w1.numel() + m.w2.numel()
        elif isinstance(m, BlockdiagPointwise):
            macs += m.weight.numel()
    return macs, sum(p.numel() for p in model.parameters()), h


def reach(h) -> str:
    """Fraction of output channels one input channel influences, per layer shape.

    Monarch mixes fully only while nblocks <= sqrt(in_channels); past that each
    output sees min(in/nblocks, nblocks) of the nblocks input blocks.
    """
    pw = h.get("pointwise") or {}
    if pw.get("kind") != "monarch":
        return "dense"
    b = int(pw["nblocks"])
    res, conv = int(h["n_channels_res"]), int(h["n_channels_conv"])
    out = []
    for ci, co in ((res, conv), (conv, res)):
        out.append(f"{100.0 * min(ci // b, b) * (co // b) / co:.0f}%")
    return " / ".join(out)


def quality(arm: str):
    log = Path(f"cp_{arm}.log")
    if not log.exists():
        return None
    vals = [float(v) for v in PESQ_RE.findall(log.read_text())]
    if not vals:
        return None
    return {
        "best": max(vals),
        "argmax": vals.index(max(vals)) + 1,
        "n_val": len(vals),
        "last5": statistics.mean(vals[-5:]),
    }


def int8(arm: str):
    log = Path(f"cp_{arm}_eval.log")
    if not log.exists():
        return None
    text = log.read_text()
    m = None
    for line in text.splitlines():
        if line.startswith("Mean: PESQ"):
            m = line
    if not m:
        return None
    d = EVAL_DUAL_RE.search(m)
    r = EVAL_RTF_RE.search(m)
    if not d:
        return None
    return {"fp32": float(d.group(1)), "int8": float(d.group(2)),
            "delta": float(d.group(3)), "rtf": float(r.group(1)) if r else None}


def fmt(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    arms = a.arms or [x for pair in PAIRS for x in pair] + [ANCHOR]
    rows = {}
    for arm in arms:
        cp = Path(f"cp_{arm}")
        if not (cp / "config.json").exists():
            continue
        macs, params, h = macs_and_params(cp)
        rows[arm] = {"macs": macs, "params": params, "h": h, "reach": reach(h),
                     "q": quality(arm), "i8": int8(arm)}

    print("## Per-arm\n")
    print("| arm | MACs/frame | params | reach | FP32 PESQ (best) | last-5 mean | "
          "int8 PESQ | Δ | int8 RTF |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, r in rows.items():
        q, i8 = r["q"] or {}, r["i8"] or {}
        print(f"| `{arm}` | {r['macs']:,} | {r['params']:,} | {r['reach']} | "
              f"{fmt(q.get('best'))} | {fmt(q.get('last5'))} | "
              f"{fmt(i8.get('int8'))} | {fmt(i8.get('delta'))} | "
              f"{fmt(i8.get('rtf'), 4)} |")

    print("\n## MAC-matched pairs\n")
    print("| MACs/frame | Monarch | dense control | gap (best) | gap (last-5) |")
    print("|---:|---|---|---:|---:|")
    for mon, den in PAIRS:
        if mon not in rows or den not in rows:
            continue
        m, d = rows[mon], rows[den]
        gb = d["q"]["best"] - m["q"]["best"]
        gl = d["q"]["last5"] - m["q"]["last5"]
        print(f"| {m['macs']:,} | `{mon}` {fmt(m['q']['best'])} | "
              f"`{den}` {fmt(d['q']['best'])} | {gb:+.3f} | {gl:+.3f} |")

    if ANCHOR in rows:
        anc = rows[ANCHOR]
        print(f"\nAnchor `{ANCHOR}`: {anc['macs']:,} MACs/frame, "
              f"{fmt(anc['q']['best'])} best / {fmt(anc['q']['last5'])} last-5.")


if __name__ == "__main__":
    main()
