---
name: deploy-stm32n6
description: Compile a model to the STM32N6 Neural-ART NPU and measure it on silicon. Use for any stedgeai/atonn compile, on-board latency or energy measurement, or claim about embedded cost.
---

# STM32N6 deployment and measurement

Procedure lives in the repo and is authoritative — this skill is the map plus
the rules that keep a compile from becoming an unfounded cost claim.

| you need                     | read |
| ---------------------------- | ---- |
| the compile/flash pipeline   | `deploy/stm32n6/README.md` (every step is a `make` target) |
| the measurement procedure    | `deploy/stm32n6/ONBOARD_MEASUREMENT.md` |
| NSNet2-specific findings     | `deploy/stm32n6/NSNET2_DEPLOYMENT_NOTES.md` |
| current efficiency backlog   | `deploy/stm32n6/TODO.md` |
| what blocks the compiler     | `LISENNET_NPU_HANDOVER.md` |

## Known compiler blockers — check the graph before burning a compile

`atonn` (stedgeai 4.0.1) fails on these, and three of the four were found the
expensive way:

| construct | symptom | fix |
| --------- | ------- | --- |
| 2-axis LayerNorm | not NPU-mappable, lowers to ReduceMean/Sqrt/Div | `norm="batchnorm"` (folds into conv) |
| PReLU / Mish (per-channel float slope) | blocks full int8, forces M55 hybrid | `act="relu"` |
| rank-5 tensors (`SPConvTranspose2d`'s view→permute→view) | **segfault (signo=11)** | `upsample="convtranspose"` |
| `Pad` with an **empty** optional `constant_value` input | **segfault (signo=11)**, only in the dual-branch sub-band context | `_strip_empty_pad_value_inputs` in `lisennet/export_onnx.py` |
| GRU | not compilable | conv bottleneck variant |

The fourth one cost five bisection rounds because the first diagnosis blamed the
newest, most complex thing in the graph (the 17-tensor FIFO state I/O), which was
innocent. **Bisect to a minimal repro before forming a narrative** — and note
that this blocker needs the dual-branch context to reproduce, so a whole-graph
proof does not clear it.

## Rules for anything you measure here

1. **A compile is not a measurement.** `stedgeai generate` gives you epoch
   counts and MACC. It does not give you latency or energy. Do not report
   generate-only numbers as on-board results.
2. **MACs do not predict cost on this target.** The dense NSNet2 is memory-bound
   — 2.70 MB weights overflow on-chip RAM, 22.94 ms/frame, RTF 1.43 — while
   `monarch_full` at 1.10 M params reaches 2.13 ms/frame, RTF 0.13, weights
   resident. Weight locality is usually the dominant variable; check whether the
   build is `noextmem` (fully on-chip) or `allmems`.
3. **Report the deadline, not just the mean.** The streaming hop is 16 ms. State
   P99 per-frame latency against it, not only the median.
4. **Record the toolchain in the manifest.** stedgeai version, CubeProgrammer
   version, ST-LINK firmware, board, memory profile, build flags:
   `uv run python tools/run_manifest.py --id <id> --no-run --toolchain stedgeai=4.0.1 ...`
   A board number without its toolchain version is not comparable to the next one.
5. **Check numerical fidelity on target**, not just that it ran. On-target
   cosine against the host int8 output; the deployed LiSenNet graphs report
   0.9983 (windowed) and 0.998 (streaming threaded-state). A fast model that
   computes the wrong mask is not a result.
6. **Energy is unmeasured in this repo.** Every existing embedded number is
   latency. Do not let a latency measurement become an energy claim
   (`CLAIMS.yaml#lisennet-n6-streaming-deployed` records this scope explicitly).

## Board sessions are an approval gate

Ask before using board time or the ST cloud farm. Bring the card, the specific
measurements to be taken, and what will be concluded from each.
