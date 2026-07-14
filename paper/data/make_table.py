"""Generate paper/table2_body.tex from the measured sweep.

Sources: compile_facts.json (stedgeai reports) + board_results.csv (on-target
validate) + the host PESQ numbers. Re-run after any re-measurement; the paper
picks it up with no hand-editing.
"""
import csv
import os
import json

S = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(S, "..", "table2_body.tex")
HOP_MS, EMIT_T = 16.0, 64

# key -> (label, params, RF, trained?, PESQ_windowed(all-int8), PESQ_streaming)
VAR = {
    "nc20":  ("hardened, 20\\,ch",       25_682,  68, False, 2.853, None),
    "nc24":  ("hardened, 24\\,ch",       36_288,  68, True,  2.998, 2.963),
    "nc28":  ("hardened, 28\\,ch",       48_718,  68, False, 2.867, None),
    "dil16": ("\\quad+ dilation 16",     37_680, 132, False, 2.954, None),
    "deep":  ("\\quad+ 3 blocks (deep)", 46_248, 196, False, 2.985, None),
    "r6":    ("\\textbf{relu6-deep}",    46_248, 196, True,  3.014, None),
    "r6hyb": ("relu6-deep, hybrid dec.$^\\dagger$", 46_248, 196, True,  3.052, None),
}
# streaming PESQ measured this session (filled in by the eval jobs)
STREAM_PESQ = json.load(open(f"{S}/stream_pesq.json")) if __import__("os").path.exists(f"{S}/stream_pesq.json") else {}

facts = json.load(open(f"{S}/compile_facts.json"))
board = {}
with open(f"{S}/board_results.csv") as f:
    for r in csv.DictReader(f):
        board[r["name"]] = r

# other model families measured on the same board with the same flow (streaming)
BASELINES = [
    ("ConvFSENet~\\cite{luo2019convtasnet}", "1.45\\,M", 1.47, "1.40\\,MB", 4.40, 0.275, "2.911"),
    ("NSNet2 monarch\\_full~\\cite{dao2022monarch}", "0.36\\,M", None, "0.72\\,MB", 2.13, 0.133, "2.848"),
    ("NSNet2 monarch\\_8~\\cite{dao2022monarch}", "0.36\\,M", 0.39, "0.37\\,MB", 2.89, 0.181, "2.826"),
    ("NSNet2 dense~\\cite{braun2020nsnet2}$^\\dagger$", "2.78\\,M", None, "2.70\\,MB", 22.94, 1.434, "2.833"),
]


def mem(kb):
    return f"{kb/1024:.2f}\\,MB" if kb >= 1000 else f"{kb:.0f}\\,KB"


def row(key, graph):
    name = "r6hyb" if key == "r6hyb" else f"{key}{graph}"
    f, b = facts.get(name), board.get(name)
    label, params, rf, trained, pesq_w, pesq_s_known = VAR[key]
    mark = "" if trained else "$^\\ddagger$"
    if not f or not f.get("compiled"):
        return f"    {label}{mark} & {params/1000:.1f}\\,k & {rf} & \\multicolumn{{5}}{{c}}{{\\emph{{does not compile}}}} \\\\"
    if not b or b.get("status") != "OK":
        note = {"LOAD_FAILED": "does not run on-board",
                "VALIDATE_FAILED": "does not run on-board",
                "USB_DEAD": "not measured"}.get(b["status"] if b else "", "not measured")
        return (f"    {label}{mark} & {params/1000:.1f}\\,k & {rf} & "
                f"{(f['macc']/(1 if graph=='str' else EMIT_T))/1e6:.2f}\\,M & {mem(f['weights_B']/1024)} & "
                f"\\multicolumn{{3}}{{c}}{{\\emph{{{note}}}}} \\\\")
    ms_inf = float(b["ms_per_inference"])
    if graph == "str":
        ms_f, macc_f = ms_inf, f["macc"]
        pesq = STREAM_PESQ.get(key, pesq_s_known)   # new eval wins, else the known value
    else:
        ms_f, macc_f = ms_inf / EMIT_T, f["macc"] / EMIT_T
        pesq = pesq_w
    p = f"{pesq:.3f}" if isinstance(pesq, (int, float)) else "---"
    bold = "\\textbf{" if key == "r6" else ""
    end = "}" if bold else ""
    return (f"    {label}{mark} & {params/1000:.1f}\\,k & {rf} & {macc_f/1e6:.2f}\\,M & "
            f"{mem(f['weights_B']/1024)} & {bold}{ms_f:.2f}{end} & {bold}{ms_f/HOP_MS:.3f}{end} & {p} \\\\")


L = []
L.append("  \\begin{tabular}{@{}lrrrrrrr@{}}")
L.append("    \\toprule")
L.append("    deployment & params & RF & MAC/frame & weights & ms/frame & RTF & PESQ \\\\")
L.append("    \\midrule")
L.append("    \\multicolumn{8}{@{}l}{\\emph{streaming} --- one frame in, one out; \\textbf{16\\,ms} algorithmic latency} \\\\")
for k in ("nc20", "nc24", "nc28", "dil16", "deep", "r6"):
    L.append(row(k, "str"))
L.append("    \\midrule")
L.append("    \\multicolumn{8}{@{}l}{\\emph{windowed} --- 64-frame emit block; \\textbf{1.02\\,s} algorithmic latency} \\\\")
for k in ("nc20", "nc24", "nc28", "dil16", "deep", "r6"):
    L.append(row(k, "win"))
L.append(row("r6hyb", "hyb"))
L.append("    \\midrule")
L.append("    \\multicolumn{8}{@{}l}{\\emph{other model families --- same board, same flow, streaming}} \\\\")
for lab, par, macc, w, ms, rtf, p in BASELINES:
    mc = f"{macc:.2f}\\,M" if macc else "---"
    L.append(f"    {lab} & {par} & --- & {mc} & {w} & {ms:.2f} & {rtf:.3f} & {p} \\\\")
L.append("    \\bottomrule")
L.append("  \\end{tabular}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\n-> wrote {OUT}")
