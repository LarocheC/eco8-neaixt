"""Parse the sweep's generate reports and fit latency from the measured board data.

Fit basis: the six on-board streaming variants (compile_facts.json + board ms).
Models tried: ms ~ epochs; ms ~ epochs+MACC; ms ~ epochs+acts. Predictions for
the new graphs are reported per-model with the fit's max |residual| as the band.
"""
import json
import os
import re
import sys

import numpy as np

SP = os.environ.get("SWEEP_DIR", os.path.dirname(os.path.abspath(__file__)))

# measured streaming points: epochs sw hybrid hw, macc, acts_B, on-board ms
MEASURED = {
    "nc20str": (137, 6, 53, 78, 922018, 125248, 2.59),
    "nc24str": (131, 6, 51, 74, 1299086, 150065, 2.79),
    "nc28str": (145, 6, 50, 89, 1750393, 176689, 3.15),
    "dil16str": (154, 6, 60, 88, 1380878, 266033, 3.63),
    "deepstr": (201, 6, 77, 118, 1690382, 311729, 4.88),
    "r6str": (199, 6, 77, 116, 1663373, 311729, 4.83),
}


def parse_report(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    pat = {
        "epochs": r"Total number of epochs\s+(\d+)",
        "sw": r"pure software \(SW\) epochs\s+(\d+)",
        "hybrid": r"hybrid epochs[^\d]+(\d+)",
        "hw": r"pure hardware \(HW or EC\) epochs\s+(\d+)",
    }
    for k, p in pat.items():
        m = re.search(p, txt)
        out[k] = int(m.group(1)) if m else None
    m = re.search(r"model: macc=([\d,]+)", txt)
    out["macc"] = int(m.group(1).replace(",", "")) if m else None

    def to_bytes(val, unit):
        return float(val) * {"B": 1, "kB": 1e3, "MB": 1e6}[unit]

    m = re.search(r"Total:.*?-- weights:\s+([\d.]+) (B|kB|MB)\s+"
                  r"activations:\s+([\d.]+) (B|kB|MB)", txt)
    if m:
        out["weights_B"] = int(to_bytes(m.group(1), m.group(2)))
        out["acts_B"] = int(to_bytes(m.group(3), m.group(4)))
    else:
        out["weights_B"] = out["acts_B"] = None
    return out


def fits():
    rows = list(MEASURED.values())
    ep = np.array([r[0] for r in rows], float)
    macc = np.array([r[4] for r in rows], float) / 1e6
    acts = np.array([r[5] for r in rows], float) / 1e3
    ms = np.array([r[6] for r in rows], float)
    ones = np.ones_like(ep)
    del macc  # report-macc units differ from compile_facts; not used for fits
    designs = {
        "epochs": np.stack([ones, ep], 1),
        "epochs+acts": np.stack([ones, ep, acts], 1),
    }
    out = {}
    for name, X in designs.items():
        coef, *_ = np.linalg.lstsq(X, ms, rcond=None)
        resid = ms - X @ coef
        out[name] = (coef, float(np.abs(resid).max()),
                     1 - float((resid ** 2).sum() / ((ms - ms.mean()) ** 2).sum()))
    return out


def predict(f, ep, macc, acts):
    del macc
    preds = {}
    for name, (coef, band, r2) in f.items():
        x = [1.0, ep]
        if name == "epochs+acts":
            x.append(acts / 1e3)
        preds[name] = (float(np.array(x) @ coef), band, r2)
    return preds


def main():
    f = fits()
    print("fit quality on the 6 measured streaming variants:")
    for name, (coef, band, r2) in f.items():
        print(f"  {name}: R2={r2:.3f} max|resid|={band:.2f} ms coef={np.round(coef, 4)}")
    print()
    results = {}
    for tag in sys.argv[1:]:
        rep = f"{SP}/out_{tag}/network_generate_report.txt"
        try:
            r = parse_report(rep)
        except FileNotFoundError:
            print(f"{tag}: no report (compile failed?)")
            continue
        results[tag] = r
        line = (f"{tag}: epochs={r['epochs']} (HW {r['hw']} / hyb {r['hybrid']} / "
                f"SW {r['sw']})  MACC={r['macc']:,}  weights={r['weights_B']:,} B  "
                f"acts={r['acts_B']:,} B")
        print(line)
        if all(r.get(k) is not None for k in ("epochs", "macc", "acts_B")):
            p = predict(f, r["epochs"], r["macc"], r["acts_B"])
            lo = min(v[0] - v[1] for v in p.values())
            hi = max(v[0] + v[1] for v in p.values())
            mid = ", ".join(f"{k}:{v[0]:.2f}" for k, v in p.items())
            print(f"    est ms/frame: {mid}  -> band [{lo:.2f}, {hi:.2f}] ms")
    with open(f"{SP}/sweep_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {SP}/sweep_results.json")


if __name__ == "__main__":
    main()
