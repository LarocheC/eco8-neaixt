# Planned figures

1. **Fig. 1 — architecture + hardening** (`architecture.pdf`): ✅ DONE
   (2026-07-14). Standalone TikZ (`architecture.tex`, `pdflatex` — or `make`
   from `paper/`); two-row U-Net snake with the dual-path conv block expanded
   in a detail panel, four hardening tags (orange), FIFO state chips (violet),
   CVD-validated palette. Wired into `main.tex` as a full-width `figure*`.

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
