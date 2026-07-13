# Planned figures

1. **Fig. 1 — architecture + hardening** (`architecture.pdf`): block diagram of
   the NPU-hardened LiSenNet — sub-band encoder (257→128→64→32, low/high split
   per stage), dual-path conv bottleneck (intra-frequency depthwise k=11 ‖
   causal dilated time stack d=1/2/4/8(/16) + ConvGLU), mask decoder with
   skip-cats. Annotate the four hardening swaps (LN→BN fold, PReLU/Mish→ReLU6,
   subpixel→ConvTranspose, GRU→conv) and the FIFO state tensors exposed by the
   streaming export. Source material: `lisennet/model.py`, the dossier in
   LISENNET_NPU_HANDOVER.md; the original paper's fig for visual reference
   (arXiv:2409.13285).

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
