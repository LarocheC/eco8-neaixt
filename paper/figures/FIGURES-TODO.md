# Planned figures

1. **Fig. 1 — architecture + hardening** (`architecture.pdf`): ✅ DONE
   (2026-07-14, v2). Standalone TikZ (`architecture.tex`; `make` from
   `paper/`). Drawn as a **diff against the original paper's Fig. 1**
   (arXiv:2409.13285 — same two-panel layout, same A/P/c notation): (a)
   overall structure with green=unchanged, blue=NPU replacement (DPC, Sigmoid),
   orange was-tags, ghosted Griffin-Lim (deploy uses noisy phase), violet FIFO
   chips; (b) the DPC module mirroring the original's DPR module panel
   (was: LN·Bi-GRU·Linear / LN·GRU·Linear, Mish→ReLU6, no batch-merging
   reshapes). CVD-validated palette. Wired into `main.tex` as `figure*`.

2. **Fig. 2 — quality vs. on-device latency** (`pesq_vs_latency.pdf`): scatter
   of every deployment measured on the N6 (Table 2 rows): x = ms/frame
   (log?), y = RT-int8 PESQ, marker size = weight memory, annotate 16 ms
   real-time budget line. Data: `deploy/stm32n6/ONBOARD_MEASUREMENT.md`.
   Waiting on the relu6-deep rows before drawing.

3. (Optional) **PTQ sensitivity bar chart**: per-group PESQ recovery from the
   round-3 seeded selective-quantization scan (decoder +0.052, everything else
   ≤ +0.010) — currently in prose; a small bar figure may read better than
   text if space allows.

Build note: generate as vector PDF, single-column width (~3.4 in), fonts ≥ 8 pt.
