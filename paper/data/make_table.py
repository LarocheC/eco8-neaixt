"""Generate paper/table2_body.tex from the measured sweeps.

Inputs (all in this directory):
  compile_facts.json  - stedgeai reports (streaming + windowed T=64 graphs)
  board_results.csv   - on-target validate: streaming + windowed T=64
  win1_results.csv    - on-target validate: stateless RF-window at emit_T=1
  win1_macc.json      - MACC of the emit_T=1 graphs (from their compile reports)
  stream_pesq.json    - streaming-graph PESQ on the full 824-utt test split

Re-run after any re-measurement; the paper picks it up with no hand-editing.
"""
import csv
import json
import os

S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "..", "table2_body.tex")
HOP_MS, EMIT_T = 16.0, 64

# key -> (label, params, RF, trained?, PESQ all-int8 [Table 1], PESQ streaming [known])
VAR = {
    "nc20":  ("hardened, 20\\,ch",       25_682,  68, False, 2.853, None),
    "nc24":  ("hardened, 24\\,ch",       36_288,  68, True,  2.998, 2.963),
    "nc28":  ("hardened, 28\\,ch",       48_718,  68, False, 2.867, None),
    "dil16": ("\\quad+ dilation 16",     37_680, 132, False, 2.954, None),
    "deep":  ("\\quad+ 3 blocks (deep)", 46_248, 196, False, 2.985, None),
    "r6":    ("\\textbf{relu6-deep}",    46_248, 196, True,  3.014, None),
    "r6hyb": ("relu6-deep, hybrid dec.", 46_248, 196, True,  3.052, None),
}
KEYS = ("nc20", "nc24", "nc28", "dil16", "deep", "r6")

facts = json.load(open(f"{S}/compile_facts.json"))
STREAM_PESQ = json.load(open(f"{S}/stream_pesq.json"))
W1_MACC = json.load(open(f"{S}/win1_macc.json"))
board = {r["name"]: r for r in csv.DictReader(open(f"{S}/board_results.csv"))}
win1 = {r["name"]: r for r in csv.DictReader(open(f"{S}/win1_results.csv"))}

BASELINES = [
    ("ConvFSENet~\\cite{luo2019convtasnet}", "1.45\\,M", 1.47, "1.40\\,MB", 4.40, "2.911"),
    ("NSNet2 monarch\\_full~\\cite{dao2022monarch}", "0.36\\,M", None, "0.72\\,MB", 2.13, "2.848"),
    ("NSNet2 monarch\\_8~\\cite{dao2022monarch}", "0.36\\,M", 0.39, "0.37\\,MB", 2.89, "2.826"),
    ("NSNet2 dense~\\cite{braun2020nsnet2}$^\\dagger$", "2.78\\,M", None, "2.70\\,MB", 22.94, "2.833"),
]


def mem(kb):
    return f"{kb/1024:.2f}\\,MB" if kb >= 1000 else f"{kb:.0f}\\,KB"


def line(label, params, rf, macc, weights_kb, ms, pesq, bold=False, mark=""):
    b, e = ("\\textbf{", "}") if bold else ("", "")
    p = f"{pesq:.3f}" if isinstance(pesq, (int, float)) else "---"
    return (f"    {label}{mark} & {params/1000:.1f}\\,k & {rf} & {macc/1e6:.2f}\\,M & "
            f"{mem(weights_kb)} & {b}{ms:.2f}{e} & {b}{ms/HOP_MS:.2f}{e} & {p} \\\\")


L = ["  \\begin{tabular}{@{}lrrrrrrr@{}}", "    \\toprule",
     "    deployment & params & RF & MAC/frame & weights & ms/frame & RTF & PESQ \\\\",
     "    \\midrule",
     "    \\multicolumn{8}{@{}l}{\\textbf{(a) FIFO streaming} --- one frame in, one out; "
     "\\textbf{16\\,ms} algorithmic latency} \\\\"]
for k in KEYS:
    lab, par, rf, tr, pw, ps = VAR[k]
    f = facts[f"{k}str"]
    L.append(line(lab, par, rf, f["macc"], f["weights_B"] / 1024,
                  float(board[f"{k}str"]["ms_per_inference"]),
                  STREAM_PESQ.get(k, ps), bold=(k == "r6"),
                  mark="" if tr else "$^\\ddagger$"))

L += ["    \\midrule",
      "    \\multicolumn{8}{@{}l}{\\textbf{(b) stateless RF-window}, $T{=}1$ --- no state, "
      "recomputes the receptive field every frame; same \\textbf{16\\,ms} latency} \\\\"]
for k in KEYS:
    lab, par, rf, tr, pw, _ = VAR[k]
    L.append(line(lab, par, rf, W1_MACC[k], facts[f"{k}win"]["weights_B"] / 1024,
                  float(win1[k]["ms_per_frame"]), pw,
                  mark="$^\\dagger$" + ("$^\\ddagger$" if not tr else "")))

L += ["    \\midrule",
      "    \\multicolumn{8}{@{}l}{\\textbf{(c) RF-window amortized}, $T{=}64$ --- throughput mode, "
      "\\textbf{1.02\\,s} block latency (not real-time \\emph{latency})} \\\\"]
for k in ("nc24", "r6", "r6hyb"):
    lab, par, rf, tr, pw, _ = VAR[k]
    nm = "r6hyb" if k == "r6hyb" else f"{k}win"
    f = facts[nm]
    L.append(line(lab, par, rf, f["macc"] / EMIT_T, f["weights_B"] / 1024,
                  float(board[nm]["ms_per_inference"]) / EMIT_T, pw,
                  mark="$^\\dagger$" if k == "r6hyb" else ""))

L += ["    \\midrule",
      "    \\multicolumn{8}{@{}l}{\\emph{other model families --- same board, same flow, FIFO streaming}} \\\\"]
for lab, par, macc, w, ms, p in BASELINES:
    mc = f"{macc:.2f}\\,M" if macc else "---"
    L.append(f"    {lab} & {par} & --- & {mc} & {w} & {ms:.2f} & {ms/HOP_MS:.2f} & {p} \\\\")
L += ["    \\bottomrule", "  \\end{tabular}"]

open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
