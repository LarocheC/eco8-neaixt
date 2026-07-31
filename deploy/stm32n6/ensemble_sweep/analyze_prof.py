"""Decompose measured per-epoch time: CT-expansion cost vs folded trunk.

Joins npu_profiler per-node tables (prof_<name>.log) with the compiler's
epoch -> original-ONNX-layer mapping (network_c_info.json).
"""
import json
import re
import sys

SP = "/tmp/claude-1000/-home-claroche-eco8-neaixt/cead2b27-c88f-41b4-b58e-07c77fd6ee40/scratchpad"

ROW = re.compile(
    r"^\s+(\d+)\s+\S+\s+epoch(?:\s+\((\w+)\))?\s+(\d+\.\d+)\s+.*(EpochBlock_(\d+))\s*$")


def prof_rows(name):
    rows = {}
    for line in open(f"{SP}/prof_{name}.log", errors="replace"):
        m = ROW.match(line.rstrip())
        if m:
            rows[int(m.group(5))] = (float(m.group(3)), m.group(2) or "HW")
    return rows


def epoch_map(name):
    d = json.load(open(f"{SP}/out_{name}/network_c_info.json"))
    out = {}
    for n in d["graphs"][0]["nodes"]:
        m = re.match(r"epoch_(\d+)$", n["name"])
        if m:
            out[int(m.group(1))] = n
    return out


def is_ct(node):
    names = " ".join(str(x) for x in node.get("original_nodes", []))
    return ("ctm" in names or "_m0" in names or "_m1" in names or "_m2" in names
            or "ConvTranspose" in names)


def analyze(name):
    rows, emap = prof_rows(name), epoch_map(name)
    tot = sum(d for d, _ in rows.values())
    ct = sum(d for e, (d, _) in rows.items() if e in emap and is_ct(emap[e]))
    by_map = {}
    for e, (d, typ) in rows.items():
        by_map[typ] = by_map.get(typ, 0.0) + d
    print(f"{name}: parsed {len(rows)} epochs, total {tot:.3f} ms")
    print(f"  by mapping: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(by_map.items())))
    print(f"  CT/decoder-transpose epochs: {ct:.3f} ms; everything else: {tot - ct:.3f} ms")
    top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:10]
    for e, (d, typ) in top:
        node = emap.get(e, {})
        orig = ",".join(str(x) for x in node.get("original_nodes", []))[:90]
        print(f"    epoch_{e:<4} {d:.3f} ms  [{typ:>6}]  {orig}")
    return tot, ct


for name in sys.argv[1:]:
    analyze(name)
    print()
