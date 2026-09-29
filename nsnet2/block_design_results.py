"""Rebuild the block-design study's tables from logs, checkpoints and the Phase-3 records.

Sibling of ``nsnet2.sparsity_results`` (same log parsing, same definitions of
last-5 and nonzeros), for the study in BLOCK_DESIGN.md. Its runs live in two
checkouts: the pilot and Phase-2 parents were trained in the ``block-design``
worktree, the old-block parents they are compared with in the main checkout.
Each run is looked up in every ``--roots`` entry in order.

Outputs:

* ``BLOCK_DESIGN_RUNS.csv``: one row per training run (pilot, Phase-2 parents,
  and the old-block parents), with design flags and the full PESQ trajectory.
* ``BLOCK_DESIGN_PHASE3.csv``: one row per Phase-3 parent, built from
  ``results/block_design/phase3/<parent>.json`` (dead units, parameter and
  nonzero counts, dense / c1+perm+refit / free 2:4+refit / int8 PESQ).
* ``BLOCK_DESIGN_PARETO.csv``: one row per point of the dense vs 2:4 Pareto
  study (dense, 2:4@c1+perm+refit, free 2:4+refit, 2:4@c1 no perm), with
  nonzeros, MACs per frame and the parent's validation summary. New widths
  come from ``results/block_design/pareto/``, reused parents from ``phase3/``.
* ``--update-doc`` / ``--check-doc``: rewrite, or verify against a fresh
  regeneration, every generated table in BLOCK_DESIGN.md (between
  ``<!-- BEGIN generated:NAME -->`` and ``<!-- END generated:NAME -->``).

    python -m nsnet2.block_design_results --runs-csv BLOCK_DESIGN_RUNS.csv \\
        --phase3-csv BLOCK_DESIGN_PHASE3.csv --pareto-csv BLOCK_DESIGN_PARETO.csv \\
        --update-doc BLOCK_DESIGN.md
    python -m nsnet2.block_design_results --check-doc BLOCK_DESIGN.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

from nsnet2.sparsity_results import _EPOCH, _INIT, _PESQ, _log_text, _params

DATA = "results/block_design"

# (study, model, design, run). Designs: "old" = the block every earlier run used;
# A0-A3 / B0-B1 = pilot arms; A1 / B1 = the designs Phase 2 carried forward.
RUNS = [
    *[("pilot", "nsnet2", d, f"pilot_ns_{d}") for d in ("A0", "A1", "A2", "A3")],
    *[("pilot", "convfsenet", d, f"pilot_cf_{d}") for d in ("B0", "B1")],
    ("phase2", "nsnet2", "old", "sq192_dense"),
    ("phase2", "nsnet2", "old", "p2_ns_sq192_old_s2345"),
    ("phase2", "nsnet2", "A1", "p2_ns_sq192_a1"),
    ("phase2", "nsnet2", "A1", "p2_ns_sq192_a1_s2345"),
    ("phase2", "nsnet2", "old", "dense_h68"),
    ("phase2", "nsnet2", "old", "dense_h68_s2345"),
    ("phase2", "nsnet2", "old", "dense_h68_s3456"),
    ("phase2", "nsnet2", "A1", "p2_ns_h68_a1"),
    ("phase2", "nsnet2", "A1", "p2_ns_h68_a1_s2345"),
    ("phase2", "convfsenet", "old", "cfs_c96_dense"),
    ("phase2", "convfsenet", "old", "p2_cf_c96_old_s2345"),
    ("phase2", "convfsenet", "B1", "p2_cf_c96_b1"),
    ("phase2", "convfsenet", "B1", "p2_cf_c96_b1_s2345"),
    # Old-block NSNet2 128/128: a Phase-3 point on the old sparse curve only.
    ("old-curve", "nsnet2", "old", "sq128_dense"),
    # Pareto widths of the new designs (Phase-2 recipes, only the widths changed).
    *[("pareto", "nsnet2", "A1", f"p3_ns_sq{h}_a1") for h in (64, 96, 128, 256, 384)],
    *[("pareto", "convfsenet", "B1", f"p3_cf_c{c}_b1") for c in (64, 128, 160, 192)],
    # Old-block ConvFSENet dense widths from the Row-Fusion study: dense-only context.
    *[("old-dense", "convfsenet", "old", r) for r in ("cfs_c64_dense", "cfs_dense_r67", "cfs_dense_r83",
                                                     "cfs_dense_r108", "cfs_dense_r146", "cfs_dense")],
]

# Phase-2 comparisons: (label, old runs, new runs).
PHASE2 = [
    ("NSNet2 192/192", ["sq192_dense", "p2_ns_sq192_old_s2345"], ["p2_ns_sq192_a1", "p2_ns_sq192_a1_s2345"]),
    ("NSNet2 68/102", ["dense_h68", "dense_h68_s2345", "dense_h68_s3456"], ["p2_ns_h68_a1", "p2_ns_h68_a1_s2345"]),
    ("ConvFSENet 96/192", ["cfs_c96_dense", "p2_cf_c96_old_s2345"], ["p2_cf_c96_b1", "p2_cf_c96_b1_s2345"]),
]

# Phase-3 parents: (model, size, design, run). Sizes group the old sparse curve.
PARENTS = [
    ("nsnet2", "h68", "old", "dense_h68"), ("nsnet2", "h68", "old", "dense_h68_s2345"),
    ("nsnet2", "h68", "old", "dense_h68_s3456"), ("nsnet2", "sq128", "old", "sq128_dense"),
    ("nsnet2", "sq192", "old", "sq192_dense"), ("nsnet2", "sq192", "old", "p2_ns_sq192_old_s2345"),
    ("nsnet2", "sq192", "A1", "p2_ns_sq192_a1"), ("nsnet2", "sq192", "A1", "p2_ns_sq192_a1_s2345"),
    ("nsnet2", "h68", "A1", "p2_ns_h68_a1"), ("nsnet2", "h68", "A1", "p2_ns_h68_a1_s2345"),
    ("convfsenet", "c96", "old", "cfs_c96_dense"), ("convfsenet", "c96", "old", "p2_cf_c96_old_s2345"),
    ("convfsenet", "c96", "B1", "p2_cf_c96_b1"), ("convfsenet", "c96", "B1", "p2_cf_c96_b1_s2345"),
]

# Fine-tuned masked runs (SPARSE_MATMUL_RUNS.csv) whose parent is a Phase-3 old parent.
FINETUNES = [("sq192_dense", "c1", ["sq192_cb_c1", "sq192_cb_c1_s2345"]),
             ("cfs_c96_dense", "c1", ["cf96_cb_c1", "cf96_cb_c1_s2345"]),
             ("cfs_c96_dense", "free", ["cf96_2to4", "cf96_2to4_s2345"])]

# Pre-registered Phase-3 rules. ADOPT needs all three; margin < 0 is REJECT; else NO EVIDENCE.
MARGIN = 0.03        # (a) new sparse PESQ minus the old sparse curve at the new nonzero count
INT8_SLACK = 0.01    # (c) new int8 loss may exceed the old one by at most this


def _root(run: str, roots: list[str]) -> str:
    for r in roots:
        if os.path.exists(os.path.join(r, f"cp_{run}", "config.json")):
            return r
    raise FileNotFoundError(f"cp_{run} not found under any of {roots}")


def collect_runs(roots: list[str]) -> list[dict]:
    rows = []
    for study, model, design, run in RUNS:
        root = _root(run, roots)
        text = _log_text(run, root)
        traj = [float(x) for x in _PESQ.findall(text)]
        cfg = json.load(open(os.path.join(root, f"cp_{run}", "config.json")))
        width = (f"{cfg.get('hidden_dim')}/{cfg.get('fc_hidden_dim')}" if model == "nsnet2"
                 else f"{cfg.get('n_channels_res')}/{cfg.get('n_channels_conv')}")
        init = _INIT.search(text)
        total, nonzero = _params(run, root)
        rows.append({
            "study": study, "model": model, "design": design, "run": run,
            "checkout": os.path.basename(os.path.abspath(root)), "width": width,
            "block_norm": cfg.get("block_norm") or cfg.get("frontend_norm") or "none",
            "input_norm": (cfg.get("input_norm") or {}).get("kind", "none"),
            "warmup_steps": cfg.get("warmup_steps") or 0,
            "init": init.group(1) if init else "scratch",
            "epochs": len(_EPOCH.findall(text)), "seed": cfg.get("seed"),
            "lr": cfg.get("learning_rate"), "total_params": total, "nonzero_params": nonzero,
            "n_val": len(traj), "best": round(max(traj), 4),
            "last5": round(st.mean(traj[-5:]), 4) if len(traj) >= 5 else "",
            "pesq_trajectory": ";".join(f"{v:.3f}" for v in traj),
        })
    return rows


def collect_phase3(data: str = DATA) -> list[dict]:
    rows = []
    for model, size, design, run in PARENTS:
        x = json.load(open(os.path.join(data, "phase3", f"cp_{run}.json")))
        if model == "nsnet2":
            dead, units = f"{x['dead_fc_in']['train']}/{x['dead_fc2']['train']}", \
                f"{x['dead_fc_in']['units']}/{x['dead_fc2']['units']}"
            params, compacted, dense = x["params_dense"], x["params_compacted"], x["pesq_dense"]
            free, i8, i8d = x["2:4"], x["int8_sparse_c1perm"], x["int8_dense"]
            method, i8_ref, i8_pesq = "onnx-qdq-static-minmax", i8["pesq_fp32_onnx"], i8["pesq_int8"]
            i8_drop, i8d_drop = i8["drop_int8_minus_fp32onnx"], i8d["drop_int8_minus_fp32onnx"]
        else:
            dead, units = str(x["frontend_dead"]["train"]), str(x["frontend_dead"]["units"])
            params = compacted = x["params"]    # no ReLU unit is removed: nothing to compact
            dense = x.get("pesq_dense_folded", x["pesq_dense_asis"])
            free, i8, i8d = x["free_2:4"], x["int8_sparse"], x["int8_dense"]
            method, i8_ref, i8_pesq = "torch-fakequant-w8a8-static-minmax", x["c1_perm"]["pesq"], i8["w8a8_static_minmax"]
            i8_drop, i8d_drop = i8["drop_static"], i8d["drop_static"]
        rows.append({
            "model": model, "size": size, "design": design, "parent": run, "seed": x["seed"],
            "dead_units": dead, "units": units, "params_dense": params, "params_compacted": compacted,
            "nonzeros": x["c1_perm"]["nonzeros"], "pesq_dense": dense,
            "pesq_c1_perm_refit": x["c1_perm"]["pesq"],
            "pesq_c1_noperm_refit": x["c1_noperm"]["pesq"],
            "pesq_free24_refit": free["pesq"], "int8_method": method,
            "int8_fp32_ref": i8_ref, "pesq_int8": i8_pesq,
            "int8_drop": i8_drop, "int8_dense_drop": i8d_drop,
        })
    return rows


def verdicts(p3: list[dict]) -> list[dict]:
    """Apply the Phase-3 rules. The old sparse curve is piecewise linear in ln(nonzeros)
    through the per-size means of the old parents (extended linearly past its ends);
    a straight least-squares line through every old parent is reported alongside."""
    out = []
    for model in dict.fromkeys(r["model"] for r in p3):
        old = [r for r in p3 if r["model"] == model and r["design"] == "old"]
        pts = sorted((st.mean(math.log(r["nonzeros"]) for r in g), st.mean(r["pesq_c1_perm_refit"] for r in g))
                     for g in ([r for r in old if r["size"] == s] for s in dict.fromkeys(r["size"] for r in old)))
        if len(old) > 1 and len({r["nonzeros"] for r in old}) > 1:
            b1, b0 = np.polyfit([math.log(r["nonzeros"]) for r in old], [r["pesq_c1_perm_refit"] for r in old], 1)
        else:
            b1 = b0 = None

        def curve(lx):
            if len(pts) == 1:
                return pts[0][1], "same nonzeros"
            i = max(0, min(len(pts) - 2, sum(p[0] <= lx for p in pts) - 1))
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            return y0 + (y1 - y0) * (lx - x0) / (x1 - x0), "extrapolated" if not pts[0][0] <= lx <= pts[-1][0] else "interpolated"

        for size in dict.fromkeys(r["size"] for r in p3 if r["model"] == model and r["design"] != "old"):
            new = [r for r in p3 if r["model"] == model and r["size"] == size and r["design"] != "old"]
            same = [r for r in old if r["size"] == size]
            lx = st.mean(math.log(r["nonzeros"]) for r in new)
            ref, how = curve(lx)
            mean = st.mean(r["pesq_c1_perm_refit"] for r in new)
            loss_new = -st.mean(r["int8_drop"] for r in new)
            loss_old = -st.mean(r["int8_drop"] for r in same)
            a = mean - ref >= MARGIN - 1e-12
            b = min(r["pesq_c1_perm_refit"] for r in new) >= st.mean(r["pesq_c1_perm_refit"] for r in same)
            c = loss_new <= loss_old + INT8_SLACK
            out.append({
                "model": model, "size": size, "design": new[0]["design"],
                "old_nonzeros": round(math.exp(st.mean(math.log(r["nonzeros"]) for r in same))),
                "new_nonzeros": round(math.exp(lx)),
                "old_sparse": [r["pesq_c1_perm_refit"] for r in same],
                "new_sparse": [r["pesq_c1_perm_refit"] for r in new], "new_mean": mean,
                "curve_at_new": ref, "curve": how, "margin": mean - ref,
                "margin_ls": None if b1 is None else mean - (b0 + b1 * lx),
                "int8_loss_old": loss_old, "int8_loss_new": loss_new, "a": a, "b": b, "c": c,
                "verdict": "ADOPT" if a and b and c else "REJECT" if mean - ref < 0
                           else "NO EVIDENCE" if mean - ref < MARGIN else "FAILS (b)/(c)",
            })
    return out


# ------------------------------------------------------------------ pareto
# (model, family, design, run, record dir). "square" = NSNet2 hidden = fc = H; ConvFSENet res = C, conv = 2C.
PARETO = [
    *[("nsnet2", "square", "A1", f"p3_ns_sq{h}_a1", "pareto") for h in (64, 96, 128)],
    ("nsnet2", "square", "A1", "p2_ns_sq192_a1", "phase3"), ("nsnet2", "square", "A1", "p2_ns_sq192_a1_s2345", "phase3"),
    *[("nsnet2", "square", "A1", f"p3_ns_sq{h}_a1", "pareto") for h in (256, 384)],
    ("nsnet2", "h68", "A1", "p2_ns_h68_a1", "phase3"), ("nsnet2", "h68", "A1", "p2_ns_h68_a1_s2345", "phase3"),
    *[("nsnet2", "square" if s != "h68" else "h68", "old", r, "phase3")
      for m, s, d, r in PARENTS if m == "nsnet2" and d == "old"],
    ("convfsenet", "square", "B1", "p3_cf_c64_b1", "pareto"),
    ("convfsenet", "square", "B1", "p2_cf_c96_b1", "phase3"), ("convfsenet", "square", "B1", "p2_cf_c96_b1_s2345", "phase3"),
    *[("convfsenet", "square", "B1", f"p3_cf_c{c}_b1", "pareto") for c in (128, 160, 192)],
    ("convfsenet", "square", "old", "cfs_c96_dense", "phase3"), ("convfsenet", "square", "old", "p2_cf_c96_old_s2345", "phase3"),
    *[("convfsenet", "square", "old", r, "pareto/dense_only") for study, _, _, r in RUNS if study == "old-dense"],
]
SPARSE_POINTS = (("c1_perm_refit", "c1_perm"), ("free24_refit", None), ("c1_noperm_refit", "c1_noperm"))


def _ns_macs(x: dict) -> int:
    """Weight multiply-accumulates per frame in NSNet2's 8 GEMMs after dead-unit compaction
    (fc_in, 2 GRUs x input/recurrent, fc1, fc2, fc_out); biases and gate elementwise ops excluded."""
    h, f = x["hidden"], x["fc_hidden"]
    ni, nf = x["compacted_dims"]["fc_in_out/gru_in"], x["compacted_dims"]["fc2_out/fc_out_in"]
    return 257 * ni + 3 * h * (ni + h) + 6 * h * h + h * f + f * nf + nf * 257


def collect_pareto(runs: list[dict], data: str = DATA) -> list[dict]:
    """One row per (parent, point). x = nonzero weights as Phase 3 defines them (dense: compacted
    parameters incl. biases; sparse: that minus the pruned weights). MACs/frame count weight
    multiplies only; a pruned weight is a multiply saved."""
    by = {r["run"]: r for r in runs}
    cf_macs = json.load(open(os.path.join(data, "pareto", "convfsenet_macs.json")))
    rows = []
    for model, family, design, run, where in PARETO:
        x = json.load(open(os.path.join(data, where, f"cp_{run}.json")))
        r = by[run]
        if model == "nsnet2":
            nnz, macs, dense = x["params_compacted"], _ns_macs(x), x["pesq_dense"]
            assert x.get("macs_dense_compacted", macs) == macs, run
            size = f"{x['hidden']}/{x['fc_hidden']}"
        else:
            m = cf_macs[f"cp_{run}"]
            nnz, macs, dense = x["params"], round(m["macs_frame_dense"]), x.get("pesq_dense_folded", x["pesq_dense_asis"])
            assert m["params_plain"] == nnz, run
            size = f"{m['res']}/{m['conv']}"
        base = {"model": model, "family": family, "design": design, "width": size, "run": f"cp_{run}",
                "seed": x["seed"], "record": f"{where}/cp_{run}.json"}
        par = {"dense_pesq": dense, "dense_val_best": r["best"], "dense_last5": r["last5"], "n_val": r["n_val"]}
        rows.append({**base, "point": "dense", "nonzeros": nnz, "macs_per_frame": macs, "pesq": dense, **par})
        if "c1_perm" not in x:
            continue
        for point, key in SPARSE_POINTS:
            rec = x[key] if key else x.get("2:4", x.get("free_2:4"))
            if model == "convfsenet" and key:     # 2:4@c1: the masked-MAC count measured on the folded model
                smacs = round(cf_macs[f"cp_{run}"]["macs_frame_c1"])
                assert rec["nonzeros"] == cf_macs[f"cp_{run}"]["nnz_c1"], run
            else:
                smacs = macs - rec["pruned"]
            assert rec["nonzeros"] == nnz - rec["pruned"], run
            rows.append({**base, "point": point, "nonzeros": rec["nonzeros"], "macs_per_frame": smacs,
                         "pesq": rec["pesq"], **par})
    return rows


def _pw(pts: list[tuple[float, float]], lx: float) -> tuple[float, str]:
    """Piecewise linear through sorted (ln x, y), extended linearly past its ends."""
    i = max(0, min(len(pts) - 2, sum(p[0] <= lx for p in pts) - 1))
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    kind = "below range" if lx < pts[0][0] - 1e-12 else "above range" if lx > pts[-1][0] + 1e-12 else "interpolated"
    return y0 + (y1 - y0) * (lx - x0) / (x1 - x0), kind


def _front(rows: list[dict], x: str = "nonzeros") -> list[tuple[float, float]]:
    """Dense curve of the given parents: per-width means of (ln x, PESQ)."""
    g = {}
    for r in rows:
        g.setdefault(r["width"], []).append(r)
    return sorted((st.mean(math.log(r[x]) for r in v), st.mean(r["pesq"] for r in v)) for v in g.values())


def pareto_analysis(pts: list[dict]) -> dict:
    """Sparse minus the new-design dense front at equal nonzeros.

    PW = piecewise linear in ln(nonzeros) through the per-width means of the new dense parents.
    NSNet2 also: LS = straight line through every new dense parent, and paired = the LS cost of
    the parent's own nonzero ratio minus the sparse model's loss against that parent."""
    out = {}
    for model in ("nsnet2", "convfsenet"):
        sel = lambda d, p: [r for r in pts if r["model"] == model and r["design"] == d and r["point"] == p  # noqa: E731
                            and r["family"] == "square"]
        new_design = "A1" if model == "nsnet2" else "B1"
        dense = sel(new_design, "dense")
        front, front_m = _front(dense), _front(dense, "macs_per_frame")
        lx = np.array([math.log(r["nonzeros"]) for r in dense])
        ly = np.array([r["pesq"] for r in dense])
        b1, b0 = np.polyfit(lx, ly, 1)
        res = ly - (b0 + b1 * lx)
        sd = math.sqrt(float((res ** 2).sum()) / (len(lx) - 2))
        se = sd / math.sqrt(float(((lx - lx.mean()) ** 2).sum()))
        b1_l5 = np.polyfit(lx, [r["dense_last5"] for r in dense], 1)[0]
        parent = {r["run"]: r for r in pts if r["model"] == model and r["point"] == "dense"}
        rows = []
        for design in (new_design, "old"):
            for point in ("c1_perm_refit", "free24_refit"):
                for r in sel(design, point):
                    ref, kind = _pw(front, math.log(r["nonzeros"]))
                    ref_m, kind_m = _pw(front_m, math.log(r["macs_per_frame"]))
                    p = parent[r["run"]]
                    ls = b0 + b1 * math.log(r["nonzeros"])
                    rows.append({"run": r["run"], "design": design, "width": r["width"], "seed": r["seed"],
                                 "point": point, "nonzeros": r["nonzeros"], "pesq": r["pesq"],
                                 "own_dense": p["pesq"], "loss": p["pesq"] - r["pesq"],
                                 "pw": ref, "kind": kind, "d_pw": r["pesq"] - ref,
                                 "d_pw_macs": r["pesq"] - ref_m, "kind_macs": kind_m, "d_ls": r["pesq"] - ls,
                                 "paired": b1 * math.log(p["nonzeros"] / r["nonzeros"]) - (p["pesq"] - r["pesq"])})
        out[model] = {"front": front, "b0": b0, "b1": b1, "resid_sd": sd, "se_b1": se, "b1_last5": b1_l5,
                      "n_dense": len(dense), "rows": rows}
    return out


def _mean_se(v: list[float]) -> str:
    return f"{st.mean(v):+.3f} ± {st.stdev(v) / math.sqrt(len(v)):.3f} SE"


def md_pareto_runs(pts: list[dict], model: str) -> str:
    by = {}
    for r in pts:
        if r["model"] == model:
            by.setdefault(r["run"], {})[r["point"]] = r
    f4 = lambda g, k: f"{g[k]['pesq']:.4f}" if k in g else "-"  # noqa: E731
    rows = []
    for run, g in by.items():
        d = g["dense"]
        sp = g.get("c1_perm_refit")
        rows.append([f"`{run}`", d["design"], d["width"], d["seed"],
                     f"{d['nonzeros']:,} → {sp['nonzeros']:,}" if sp else f"{d['nonzeros']:,}",
                     f"{d['macs_per_frame']:,} → {sp['macs_per_frame']:,}" if sp else f"{d['macs_per_frame']:,}",
                     f"{d['pesq']:.4f}", f"{d['dense_val_best']:.3f}", f"{d['dense_last5']:.4f}",
                     f4(g, "c1_perm_refit"), f4(g, "free24_refit"), f4(g, "c1_noperm_refit"),
                     f"{sp['pesq'] - d['pesq']:+.4f}" if sp else "-"])
    return _t(["Parent", "Design", "Width", "Seed", "Nonzeros dense → 2:4", "MACs/frame dense → 2:4",
               "Dense (g_best)", "Val max", "Dense last-5", "2:4@c1+perm+refit", "Free 2:4+refit", "2:4@c1 no perm",
               "c1 − own dense"], "lllrrrrrrrrrr", rows)


def md_pareto_front(an: dict, model: str) -> str:
    a = an[model]
    rows, c1, fr = [], {}, {}
    for r in a["rows"]:
        (c1 if r["point"] == "c1_perm_refit" else fr)[r["run"]] = r
    fmt = lambda r, k: f"{r[k]:+.4f}"  # noqa: E731
    for run, c in c1.items():
        f = fr[run]
        below = c["kind"] == "below range"
        pw = lambda r: f"{r['d_pw']:+.4f}" + (" (extrapolated)" if r["kind"] != "interpolated" else "")  # noqa: E731
        if model == "nsnet2":
            rows.append([f"`{run}`", c["design"], f"{c['nonzeros']:,}", pw(c), fmt(c, "d_ls"),
                         fmt(c, "paired") if c["design"] != "old" else "-", pw(f), fmt(f, "d_ls"),
                         fmt(f, "paired") if f["design"] != "old" else "-"])
        else:
            rows.append([f"`{run}`", c["design"], f"{c['nonzeros']:,}",
                         "below the dense range" if below else f"{c['pw']:.4f}",
                         "-" if below else pw(c), "-" if below else pw(f),
                         "-" if c["kind_macs"] == "below range" else f"{c['d_pw_macs']:+.4f}"])
    new = [r for r in a["rows"] if r["design"] != "old"]
    mean = lambda p, k, interp=False: [r[k] for r in new if r["point"] == p  # noqa: E731
                                       and (not interp or r["kind"] == "interpolated")]
    if model == "nsnet2":
        rows.append(["mean, new design", "", "", f"{st.mean(mean('c1_perm_refit', 'd_pw', True)):+.4f} (interpolated only)",
                     _mean_se(mean("c1_perm_refit", "d_ls")), f"{st.mean(mean('c1_perm_refit', 'paired')):+.4f}",
                     f"{st.mean(mean('free24_refit', 'd_pw', True)):+.4f} (interpolated only)",
                     _mean_se(mean("free24_refit", "d_ls")), f"{st.mean(mean('free24_refit', 'paired')):+.4f}"])
        return _t(["Sparse model (parent)", "Design", "2:4 nonzeros", "c1: Δ PW", "c1: Δ LS", "c1: paired",
                   "free: Δ PW", "free: Δ LS", "free: paired"], "llrrrrrrr", rows)
    rows.append(["mean, new design (interpolated)", "", "", "", f"{st.mean(mean('c1_perm_refit', 'd_pw', True)):+.4f}",
                 f"{st.mean(mean('free24_refit', 'd_pw', True)):+.4f}", ""])
    return _t(["Sparse model (parent)", "Design", "2:4 nonzeros", "B1 dense curve at those nonzeros",
               "c1+perm+refit − curve", "free 2:4+refit − curve", "c1 − curve, equal MACs/frame"], "llrrrrr", rows)


# ------------------------------------------------------------------ markdown
def _t(header: list[str], align: str, rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---:" if c == "r" else "---" for c in align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def md_pilot(runs: list[dict], data: str = DATA) -> str:
    dead = json.load(open(os.path.join(data, "pilot_gates", "dead.json")))
    full = json.load(open(os.path.join(data, "pilot_gates", "fulltrain.json")))
    fold = json.load(open(os.path.join(data, "pilot_gates", "fold.json")))
    by = {r["run"]: r for r in runs}
    rows = []
    for d in ("A0", "A1", "A2", "A3", "B0", "B1"):
        m = "ns" if d[0] == "A" else "cf"
        r = by[f"pilot_{m}_{d}"]
        g = dead[m][d]["g_best"]["train"]
        flags = ", ".join(f for f in (
            f"{'block' if m == 'ns' else 'frontend'}_norm=batch" if r["block_norm"] != "none" else "",
            "input_norm=mean" if r["input_norm"] != "none" else "",
            f"warmup_steps={r['warmup_steps']}" if r["warmup_steps"] else "") if f) or "none (control)"
        if m == "ns":
            first = f"{g['fc_in']['dead']}/{g['fc_in']['units']}"
            later = f"{g['fc1']['dead']} / {g['fc2']['dead']}"
            ft = (_last(full, d) or {}).get("fc_in")
            ft = f"{ft['dead']}/{ft['units']}" if ft else "-"
        else:
            first = f"{g['frontend']['dead']}/{g['frontend']['units']}"
            later = f"TCM {g['tcm']['dead']}/{g['tcm']['units']}"
            ft = _last(full, d)
            ft = f"{ft['dead']}/{ft['units']}" if ft else "-"
        fk = _last(fold, d)
        fr = f"{float(fk.get('fp32_mask', fk.get('literal_fp32_mask_max_rel'))):.1e}" if fk else "not folded (no BN)"
        rows.append([f"{d}", r["model"], flags, r["epochs"], first, ft, later, fr, f"{r['best']:.3f}"])
    return _t(["Arm", "Model", "Flags", "Epochs", "First ReLU dead (400-utt train sample)",
               "First ReLU dead (full train split, last ckpt)", "fc1 / fc2 dead, or TCM", "Fold max rel. err (fp32 mask)",
               "Best val PESQ"], "lllrrrrrr", rows)


def _last(d: dict, arm: str) -> dict:
    """g_best if it was measured, else the arm's last checkpoint (None if the arm is absent)."""
    keys = [k for k in d if k.split("/")[0] == arm and "/" in k]
    return d[f"{arm}/g_best"] if f"{arm}/g_best" in d else d[max(keys)] if keys else None


def md_phase2(runs: list[dict]) -> str:
    by = {r["run"]: r for r in runs}
    rows = []
    for label, old, new in PHASE2:
        o, n = [by[r]["last5"] for r in old], [by[r]["last5"] for r in new]
        rows.append([label, by[new[0]]["design"], " / ".join(f"{v:.3f}" for v in o), f"{st.mean(o):.4f}",
                     " / ".join(f"{v:.3f}" for v in n), f"{st.mean(n):.4f}", f"{st.mean(n) - st.mean(o):+.4f}",
                     f"{by[new[0]]['total_params']:,}"])
    return _t(["Model", "New design", "Old block, last-5 per seed", "Old mean", "New block, last-5 per seed",
               "New mean", "Δ new − old", "Params (training)"], "lllrlrrr", rows)


def md_runs(runs: list[dict]) -> str:
    return _t(["Run", "Study", "Design", "Checkout", "Width", "Epochs", "Seed", "Params", "Best", "Last-5"], "llllrrrrrr",
              [[f"`cp_{r['run']}`", r["study"], r["design"], r["checkout"], r["width"], r["epochs"], r["seed"],
                f"{r['total_params']:,}", f"{r['best']:.3f}", f"{r['last5']:.3f}" if r["last5"] != "" else "-"]
               for r in runs])


def md_phase3(p3: list[dict]) -> str:
    return _t(["Parent", "Design", "Seed", "Dead", "Params → compacted", "Nonzeros", "Dense", "c1+perm+refit",
               "c1 no perm", "Free 2:4+refit", "int8", "int8 drop", "int8 drop, dense"], "llrrrrrrrrrrr",
              [[f"`cp_{r['parent']}`", r["design"], r["seed"], f"{r['dead_units']} of {r['units']}",
                f"{r['params_dense']:,} → {r['params_compacted']:,}", f"{r['nonzeros']:,}", f"{r['pesq_dense']:.4f}",
                f"{r['pesq_c1_perm_refit']:.4f}", f"{r['pesq_c1_noperm_refit']:.4f}", f"{r['pesq_free24_refit']:.4f}",
                f"{r['pesq_int8']:.4f}", f"{r['int8_drop']:+.4f}", f"{r['int8_dense_drop']:+.4f}"] for r in p3])


def md_verdicts(p3: list[dict]) -> str:
    names = {("nsnet2", "h68"): "NSNet2 68/102", ("nsnet2", "sq192"): "NSNet2 192/192", ("convfsenet", "c96"): "ConvFSENet 96/192"}
    return _t(["Model", "Design", "Nonzeros old → new", "Old sparse, same size", "New sparse", "Old curve at new nonzeros",
               "Margin (a)", "Straight-line margin", "(b)", "int8 loss old → new (c)", "Verdict"], "llrllrrrlll",
              [[names[(v["model"], v["size"])], v["design"], f"{v['old_nonzeros']:,} → {v['new_nonzeros']:,}",
                " / ".join(f"{x:.4f}" for x in v["old_sparse"]), " / ".join(f"{x:.4f}" for x in v["new_sparse"]),
                f"{v['curve_at_new']:.4f} ({v['curve']})", f"{v['margin']:+.4f}",
                "-" if v["margin_ls"] is None else f"{v['margin_ls']:+.4f}", "pass" if v["b"] else "fail",
                f"{v['int8_loss_old']:+.4f} → {v['int8_loss_new']:+.4f} {'pass' if v['c'] else 'fail'}",
                f"**{v['verdict']}**"] for v in verdicts(p3)])


def md_front(p3: list[dict]) -> str:
    """Nonzeros vs PESQ per (model, size, design), seed means: the dense and 2:4 ends of each front."""
    rows = []
    for key in dict.fromkeys((r["model"], r["size"], r["design"]) for r in p3):
        g = [r for r in p3 if (r["model"], r["size"], r["design"]) == key]
        m = lambda k: st.mean(float(r[k]) for r in g)  # noqa: E731
        rows.append([key[0], key[1], key[2], len(g), f"{m('params_compacted'):,.0f}", f"{m('pesq_dense'):.4f}",
                     f"{m('nonzeros'):,.0f}", f"{m('pesq_c1_perm_refit'):.4f}", f"{m('pesq_free24_refit'):.4f}",
                     f"{m('pesq_int8'):.4f}"])
    return _t(["Model", "Size", "Design", "Seeds", "Dense nonzeros (compacted)", "Dense PESQ", "2:4 nonzeros",
               "2:4@c1+perm+refit", "Free 2:4+refit", "2:4@c1 int8"], "lllrrrrrrr", rows)


def md_refit(p3: list[dict], sparse_csv: str) -> str:
    ft = {r["run"]: r for r in csv.DictReader(open(sparse_csv))}
    par = {r["parent"]: r for r in p3}
    rows = []
    for parent, kind, runs in FINETUNES:
        p = par[parent]
        rows.append([f"`cp_{parent}`", "2:4@c1" if kind == "c1" else "free 2:4",
                     f"{p['nonzeros']:,}", f"{p['pesq_c1_perm_refit' if kind == 'c1' else 'pesq_free24_refit']:.4f}",
                     ", ".join(f"`{r}`" for r in runs), f"{ft[runs[0]]['epochs']}",
                     f"{int(ft[runs[0]]['nonzero_params']):,}",
                     " / ".join(f"{float(ft[r]['last5']):.4f}" for r in runs)])
    return _t(["Parent", "Mask", "Refit nonzeros", "Refit PESQ (seconds, no training)", "Fine-tuned runs",
               "Fine-tune epochs", "Fine-tune nonzeros", "Fine-tune last-5"], "llrrlrrr", rows)


def blocks(runs: list[dict], p3: list[dict], sparse_csv: str, pts: list[dict]) -> dict[str, str]:
    an = pareto_analysis(pts)
    return {"pilot": md_pilot(runs), "phase2": md_phase2(runs), "phase3-verdicts": md_verdicts(p3),
            "phase3": md_phase3(p3), "front": md_front(p3), "refit": md_refit(p3, sparse_csv), "runs": md_runs(runs),
            "pareto-ns": md_pareto_runs(pts, "nsnet2"), "pareto-ns-front": md_pareto_front(an, "nsnet2"),
            "pareto-cf": md_pareto_runs(pts, "convfsenet"), "pareto-cf-front": md_pareto_front(an, "convfsenet")}


_BLOCK = re.compile(r"(<!-- BEGIN generated:(\S+) -->\n)(.*?)(<!-- END generated:\2 -->)", re.S)


def render_doc(text: str, gen: dict[str, str]) -> str:
    return _BLOCK.sub(lambda m: m.group(1) + gen[m.group(2)] + "\n" + m.group(4), text)


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[".", "../eco8-neaixt"],
                    help="checkouts searched, in order, for cp_<run>/ and its log")
    ap.add_argument("--sparse-csv", default="SPARSE_MATMUL_RUNS.csv", help="fine-tuned masked runs")
    ap.add_argument("--runs-csv", default="")
    ap.add_argument("--phase3-csv", default="")
    ap.add_argument("--pareto-csv", default="")
    ap.add_argument("--update-doc", default="", help="rewrite the generated tables in this file")
    ap.add_argument("--check-doc", default="", help="exit 1 if this file's tables differ from a fresh regeneration")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    runs, p3 = collect_runs(a.roots), collect_phase3()
    if a.runs_csv:
        _write_csv(a.runs_csv, runs)
    if a.phase3_csv:
        _write_csv(a.phase3_csv, p3)
    pts = collect_pareto(runs)
    if a.pareto_csv:
        _write_csv(a.pareto_csv, pts)
    gen = blocks(runs, p3, a.sparse_csv, pts)
    if a.markdown:
        print("\n\n".join(f"### {k}\n\n{v}" for k, v in gen.items()))
    if a.update_doc:
        text = open(a.update_doc).read()
        open(a.update_doc, "w").write(render_doc(text, gen))
        print(f"updated {len(_BLOCK.findall(text))} tables in {a.update_doc}", file=sys.stderr)
    if a.check_doc:
        text = open(a.check_doc).read()
        found = {m.group(2): m.group(3).rstrip("\n") for m in _BLOCK.finditer(text)}
        bad = [k for k, v in found.items() if v != gen[k]]
        print(f"{len(found)} tables checked, {len(bad)} differ: {bad}", file=sys.stderr)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
