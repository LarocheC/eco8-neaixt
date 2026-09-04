# HiFi4 cycle bench on the Xtensa ISS

Runs a converted int8 streaming model on the Cadence instruction-set simulator and reports
cycles/frame — the HiFi4 side of the M33-vs-DSP comparison, no board required. It compiles the
**same** `app/model_se_stream.cpp` driver the firmware uses, so a cycle number here includes the
real state-feedback cost.

## Read this before trusting a number

**1. Always pass `--mem_model`.** Plain `xt-run <elf>` models *zero-latency single-cycle memory* —
no caches, no stalls. That is a compute-only lower bound, not a latency estimate. Measured on
NSNet2 (identical output checksum, so identical computation):

| | flat (default) | `--mem_model` |
|---|---|---|
| NSNet2 int8 | 1,575,680 cyc/frame | **28,227,363 cyc/frame** (17.9x) |

The gap is worst for models with large weight blobs — NSNet2's int8 weights are 2.89 MB, more
bytes than the flat run claimed cycles. `--mem_model` is still zero-wait-state; the real board
adds PSRAM latency on top and the weights do not fit HiFi4 local SRAM.

**2. `--mem_model` runs ~10x slower to simulate.** Cap the bench to 1-2 frames (per-frame counts
are stable to <0.1%) or it will time out. Edit the loop bound in `iss_bench.cpp`.

**3. The real-time budget is not yet verified.** `3.17 M cyc/frame` assumes a 198 MHz DSP clock
that `../dsp_offload/dsp/dsp_main.cpp` still flags as a TODO. Read `CLOCK_GetDspClkFreq()` on the
board before treating any cycles figure as pass/fail.

## Build + run

```bash
export XTENSA_LICENSE_FILE=/path/to/RT500SDK.lic
cd <a work dir containing the generated headers>          # model_data.h, model_io_layout.h,
                                                          # se_test_feats.h, model_ops_micro.cpp
cp <repo>/deploy/rt595/iss/{iss_bench.cpp,fstubs.c,fsl_*.h} .
cp <repo>/deploy/rt595/app/model_se_stream.{cpp,h} .
cp <repo>/deploy/rt595/iss/resolvers/<family>_ops.cpp model_ops_micro.cpp
SCR=<dir holding tflm_rt500/> bash <repo>/deploy/rt595/iss/build_iss.sh
xt-run --xtensa-core=nxp_rt500_RI23_11_newlib --mem_model iss_bench.elf
```

Generate the headers with `../host/gen_model_data.py`, `../host/gen_io_layout_multi.py` and
`../host/gen_test_feats_multi.py`.

## Op resolvers

`resolvers/*_ops.cpp` are the per-family `MicroMutableOpResolver` registrations, derived from each
model's actual op histogram.

> **Superseded.** This block used to warn that `app/model_ops_micro.cpp` lacks
> `AddFullyConnected` and that NSNet2 therefore "cannot run on the board". The silicon capture in
> `../results/silicon_m33_nsnet2_blockdiag_full.txt` refutes that: NSNet2 `blockdiag_full` ran 16
> frames to completion on the M33 with stable mask checksums. The structured layers lower to
> `BATCH_MATMUL`, which the resolver does register, so no `FULLY_CONNECTED` appears in that graph.
> A *dense* NSNet2 graph would need it — check the op histogram before assuming the resolver
> covers a new model. The LiSenNet resolver does still register an unused `AddTanh` (harmless).
> Generating the resolver from the flatbuffer is still the right fix; until then, re-derive by
> hand whenever a graph changes, and let `python -m lane audit` check it for you.

## Link notes

- `-mlsp=sim` for the simulator LSP; `-stdlib=libc++` is mandatory for C++11-and-later with newlib.
- `-Wl,--orphan-handling=discard` absorbs a bogus `/DISCARD/` section in the TFLM archive.
- `fstubs.c` stubs two float kernels (`xa_nn_elm_mul_f32xf32_f32`,
  `xa_nn_vec_activation_min_max_f32_f32`) that the linker demands but an int8-only graph never calls.
