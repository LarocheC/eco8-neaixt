# N6Net on the board: latency and power

Status 2026-09-17. **Measured**, on the STM32N6570-DK (Neural-ART NPU at 1 GHz,
M55 at 800 MHz), with the FNB58 board-power rig. Compare
[N6NET_COMPILE.md](N6NET_COMPILE.md), which has the compiler's estimates for the
same graphs.

- **Weights.** The N6Net graphs use seeded-random weights, calibrated on
  VoiceBank-DEMAND frames (`n6net/export_npu.py` without `--checkpoint`). The
  trained checkpoints are on the training box. Latency and power depend on the
  compiled graph, not on the weight values, so these numbers carry over to the
  trained models. Board-side quality does not, so this page has no PESQ from
  the board.
- **Anchors.** Four trained models from earlier campaigns run through the same
  harness in the same session. Two of them (`blockdiag_full`, `monarch_20`)
  reproduce this morning's `nsnet_dse` campaign, so the numbers here compare
  directly with that table.

## Results

Latency comes from `stedgeai validate --mode target`. Energy is per inference:
(saturated − bracketed idle) / inference rate, the mean of 2 counterbalanced
passes, with the spread between passes. "mW rt" is one inference every 16 ms,
minus idle: the added power the model costs a product. The board idles at
565 mW. MACs are the `macc` that `validate` reports.

| model | MACs/frame | ms/frame | µJ/inference | nJ/MAC | mW sat | mW rt |
|---|--:|--:|--:|--:|--:|--:|
| `n6net_b1` (EC) | 21.7 M | 0.619 | 148.0 ± 0.4 | 6.8 | +239 | +23.4 |
| `n6net_b3` (EC) | 65.0 M | 1.611 | 444.9 ± 0.4 | 6.8 | +276 | +56.7 |
| `n6net_b3` (no EC) | 65.0 M | 2.447 | 470.7 ± 1.4 | 7.2 | +192 | +56.0 |
| `n6net_v2_c96` (EC) | 94.7 M | 1.154 | 471.5 ± 0.1 | 5.0 | +409 | +59.9 |
| `n6net_v2` (C128, EC) | 168.2 M | 2.597 | 935.3 ± 1.8 | 5.6 | +360 | +69.6 |
| `n6net_v2_fullband` (rewritten pool, EC) | 95.5 M | 1.314 | 504.7 ± 0.0 | 5.3 | +384 | +64.2 |
| `n6net_v2_fullband_native` (EC) | 95.4 M | 1.257 | 489.7 ± 0.6 | 5.1 | +390 | +60.9 |
| `n6net_v2_fullband` as exported before 5588e0f | 95.4 M | **hangs on the board** | – | – | – | – |
| *anchor* `monarch_20` (NSNet2) | 0.28 M | 0.789 | 9.8 ± 0.6 | 35 | +12.6 | +4.1 |
| *anchor* `blockdiag_full` (NSNet2) | 0.73 M | 0.674 | 17.6 ± 0.2 | 24 | +26.5 | +5.5 |
| *anchor* ConvFSENet | 0.72 M | 3.108 | 22.1 ± 1.0 | 31 | +7.2 | +4.7 |
| *anchor* LiSenNet streaming | 1.44 M | 2.788 | 29.7 ± 0.3 | 21 | +11.0 | +2.8 |

All 22 runs passed the graph-identity gate: the firmware's own timing is within
3.2 % of `validate`. On target, board and host int8 agree at cos ≥ 0.99998 on
every N6Net graph. Two anchors score lower on validate's random-input check:
ConvFSENet 0.73 and LiSenNet 0.97. That check feeds uniform noise to every
state tensor, so it is not a fidelity measure, but ConvFSENet's 0.73 is worth a
look before its row is reused.

## What it shows

**1. N6Net is fast per MAC, but the energy is not free.** The design bet was
that the NPU's unused throughput was free, so adding MACs to fill it would cost
almost nothing. For time that held. `v2_c96` runs 130× the MACs of
`blockdiag_full` in 1.7× the time: 12 ps per MAC against 910 ps.

Energy did not follow. Per MAC it fell only 4–7× (24–35 nJ → 5–7 nJ), because
an idle NPU draws almost nothing. Unused throughput was never being paid for,
so filling it adds its full switching energy. With 15–600× the anchors' MACs, N6Net
costs **15–95× the anchors' energy per inference**, and adds 23–70 mW at the
16 ms hop against their 3–6 mW.

**2. What N6Net's layout does buy is a lower cost per MAC.** Small graphs
spend most of each inference on fixed overhead (epoch launches, memory
traffic); N6Net spreads that overhead over far more MACs, at 5–7 nJ/MAC. At an
anchor-sized budget of 1–5 M MACs, that rate would give about 7–35 µJ per
inference, in the anchors' range. Whether the quality survives at that size is
the open question.

**3. Faster is not cheaper.** The epoch controller makes `b3` 34 % faster
(2.447 → 1.611 ms) but saves only 5.5 % energy (471 → 445 µJ). The board's
power while inferring rises almost in proportion (+192 → +276 mW). The same
work costs about the same energy however quickly it is done. At a fixed 16 ms
hop the two versions cost the same (+56.0 vs +56.7 mW).

**4. Higher NPU utilisation, higher power.** `v2_c96` runs its convs at the
highest utilisation measured (233 ops/cycle, compiler figure) and draws the
most power while inferring (+409 mW), for the same energy per MAC as the
rest. A layout that makes the NPU faster raises its power; it does not lower
the energy.

**5. The compiler's energy estimate is 3.3–5× low** (b3 129 µJ estimated vs
445 measured; `v2_c96` 94 vs 472). Its time estimate is 13–66 % low (b3 1.39
vs 1.61 ms; `v2` 1.56 vs 2.60 ms). This is board-level power at the 5 V input,
so regulator losses inflate the measured side. Rankings and ratios are the
reliable part.

**6. The real-time figure is about 2× the per-inference prediction** for
N6Net (b3: 445 µJ × 62.5 fps = 27.8 mW predicted, 56.7 mW measured). This is
the same pedestal from duty-cycling the NPU that `nsnet_dse` measured and
traced to the board, not to the network. It is proportionally larger for the
anchors (4–7×), where the network itself costs less.

**7. Quality per joule, as of today.**
- `n6net_b3`: 2.657 PESQ FP32, 445 µJ. Loses on both axes to `monarch_20`
  (2.848 FP32 / 2.848 int8, 9.8 µJ).
- `v2_fullband`: 2.906 at epoch 31 of 200, the only N6Net point with a chance
  of beating the anchors on quality. It now runs on the board (below), at
  1.314 ms and 505 µJ — 52× `monarch_20`'s energy, and +64 mW at the 16 ms hop.

## The full-band hang: a compiler kernel rewrite, now fixed

As exported before 5588e0f, `v2_fullband`, `_nopos` and `_lsig` compiled
cleanly (one EC blob, 0 SW / 0 hybrid epochs) and loaded, then inference never
returned: `validate` timed out after 50 s. After about three such hangs the
board wedges and needs a power cycle. `_phain` and `_phase` carry the same
branch and were never run.

The cause is **the compiler's rewrite of the branch's full-height 8×1 conv**,
which it splits into eight masked 1×1 sub-convs plus a 7-Add tree
(`_subm_0..7` in `network_c_info.json`). Not the broadcast Add, and not the
epoch controller. Probes at the real shapes
(`deploy/stm32n6/host/fullband_probes.py`, plus `power/probes/probe_broadcast*.py`;
`power/probes/run_probes.sh` drives them and stops at the first hang):

| probe | split 8×1 kernel | broadcast Add | result |
|---|---|---|---|
| `control` | no | no | runs, 0.100 ms |
| `gap_only` (GAP + 1×1, height-1 out) | no | no | runs, 0.105 ms |
| `gap` (GAP + 1×1 + broadcast) | no | **yes** | runs, 0.109 ms |
| `pool_native` (native strided convs + broadcast) | no | **yes** | runs, 0.193 ms |
| `fullh_only` (the split kernel alone, no broadcast) | **yes** | no | **hangs** |
| `pool` / my 5-layer probe (with and without EC) | **yes** | yes | **hangs** |

Broadcast Add is therefore safe on this NPU; a conv the compiler splits into
sub-convs is not. Two exports avoid the split and **both run on the board**:

- **`v2_fullband` re-exported at 5588e0f** (an 8×1 valid conv over 8 rows
  rewritten as a 4×1 stride-4 conv plus a fixed 2×1 conv, same arithmetic):
  1.314 ms, 505 µJ. The trained full-band checkpoint deploys with no retraining.
- **`v2_fullband_native`** (`configs/n6net_v2_fullband_native.json`, the branch
  built from native strided convs): 1.257 ms, 490 µJ, but needs retraining.

The lesson for `N6NET_COMPILE.md`'s "Frequency reach" section: the compiler
reporting an op as HW-mapped says nothing about whether the kernel it emits
executes. `_subm_` nodes in `network_c_info.json` are the warning sign, and
only the board settles it.

## Reproduce

Needs `~/nsnet_dse` (its harness: firmware, schedule, analyser, gates) and the
rig from `nsnet_dse/deploy/stm32n6/harness/power_harness/README-LINUX.md`
(FNB58 inline on CN8, JP2 = 5V_USB_SNK, debug on CN6).

```bash
python -m n6net.export_npu --config configs/<cfg>.json --out build/power/<cfg>
deploy/stm32n6/scripts/compile_n6net.sh build/power/<cfg>/n6net_stream_int8.onnx
# campaign tree: build/power/campaign/out/{n6/<cell>/, <run>_int8.onnx, cells.txt}
deploy/stm32n6/power/run_campaign.sh validate   # stock firmware, stedgeai validate
deploy/stm32n6/power/run_campaign.sh power      # 2 passes, installs/restores the power firmware
deploy/stm32n6/power/run_campaign.sh summary
```

Data: `results/n6net_power/`. It holds the latency CSV, `power/campaign_{runs,cells}.csv`,
and per run the raw FNB58 trace, UART marks and analysis. Anchor graphs are
`nsnet_dse`'s (`deploy/sparse_nsnet2/out`, `deploy/export/out`), plus LiSenNet
`conv-hardened/g_best_streaming_int8_static.onnx` from HF.

Board traps met in this session, each needing a physical replug:
- A hanging graph wedges the board after a few attempts (`DEV_USB_COMM_ERR`, or
  AP1 not answering).
- Probing the board with `STM32_Programmer_CLI` straight after a `validate` run
  wedged the ST-LINK itself (`DEV_USB_COMM_ERR`, CN6 replug, a software USB
  reset does not clear it). Validate cells back to back instead;
  `measure_streaming.sh` already refuses to measure if the load failed.
- One unprovoked AP1 wedge, mid-campaign.
- The FNB58 stops accepting commands (and drops off the bus when the kernel
  resets it) if a client leaves a gap of a few seconds between its init and its
  first read. Use `fnb58_hidraw.py`, which never leaves a gap.
