"""Static NPU-utilisation report from a stedgeai ``network_c_info.json``.

The Neural-ART compiler estimates, per epoch, the arithmetic ops it schedules
(multiply and add counted separately, so ops ~ 2 x MAC), the pure compute
cycles, and ``max_cycles`` — the cycle count including memory stalls. Summed,
that gives the question the N6Net design asks before any board is attached:

    ops / max_cycle   against the roof of 4 CONV_ACC x 72 MAC x 2 = 576
    U_MAC             = MAC / (288 * max_cycles)
    est. NPU time     = max_cycles at the 1 GHz NPU clock

These are the compiler's model, not a measurement; the on-board profiler
(npu_profiler.py) is the ground truth. Epochs reporting multi-watt average
power are an estimator artefact (seen on a 257-op input Sub) and are excluded.

    python deploy/stm32n6/host/npu_cycle_report.py <gen_dir> [...] [--epochs]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOF_OPS_PER_CYCLE = 4 * 72 * 2
OUTLIER_MW = 1000


def epoch_rows(c_info: dict):
    nodes = {}

    def walk(ns):
        for n in ns:
            nodes[n["id"]] = n
            walk(n["subgraph_nodes"])

    walk(c_info["graphs"][0]["nodes"])
    for pe in c_info["power_estimates"]:
        n = nodes[pe["node_id"]]
        kinds = collections.Counter(s["type"] for s in n["subgraph_nodes"]
                                    if s["type"] not in ("Param", "Identity"))
        yield {"name": n["name"], "ops": pe["ops"], "cc": pe["compute_cycles"],
               "mc": pe["max_cycles"], "acc_mw": pe["average_acc_power_mw"],
               "mem_mw": pe["average_memory_power_mw"], "kinds": dict(kinds)}


def summarise(gen_dir: Path) -> dict:
    # zero-op epochs are pure data movement; their cycles are real time, keep them
    rows = list(epoch_rows(json.load(open(gen_dir / "network_c_info.json"))))
    kept = [r for r in rows if r["acc_mw"] <= OUTLIER_MW]
    ops = sum(r["ops"] for r in kept)
    mc = sum(r["mc"] for r in kept)
    return {
        "rows": kept,
        "excluded": [r for r in rows if r["acc_mw"] > OUTLIER_MW],
        "ops": ops,
        "compute_cycles": sum(r["cc"] for r in kept),
        "max_cycles": mc,
        "u_mac": ops / mc / ROOF_OPS_PER_CYCLE,
        "npu_ms": mc / 1e6,
        "energy_mj": sum(r["mc"] * (r["acc_mw"] + r["mem_mw"]) for r in kept) / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dirs", nargs="+", type=Path)
    ap.add_argument("--epochs", action="store_true", help="per-epoch breakdown")
    a = ap.parse_args()
    for g in a.gen_dirs:
        s = summarise(g)
        print(f"== {g.name}: ops={s['ops']:,} (~{s['ops'] / 2e6:.1f} M MAC)  "
              f"compute_cycles={s['compute_cycles']:,}  max_cycles={s['max_cycles']:,}")
        print(f"   ops/compute_cycle={s['ops'] / s['compute_cycles']:.0f}  "
              f"ops/max_cycle={s['ops'] / s['max_cycles']:.0f} (roof {ROOF_OPS_PER_CYCLE})  "
              f"U_MAC={s['u_mac']:.1%}  est. NPU {s['npu_ms']:.2f} ms/frame  "
              f"est. energy {s['energy_mj']:.3f} mJ/frame")
        for r in s["excluded"]:
            print(f"   excluded outlier {r['name']}: {r['ops']} ops, {r['mc']} cycles, {r['acc_mw']} mW")
        if a.epochs:
            for r in s["rows"]:
                if not r["mc"]:
                    continue
                print(f"   {r['name']:9s} ops={r['ops']:>11,} cc={r['cc']:>8,} mc={r['mc']:>8,} "
                      f"ops/mc={r['ops'] / max(r['mc'], 1):4.0f}  {r['kinds']}")


if __name__ == "__main__":
    main()
