# LiSenNet SE — HiFi4 DSP offload (staged for the Xtensa toolchain box)

The M33 runs the LiSenNet streaming SE correctly but at **~21.2 M cycles / ~107 ms per
frame** — **~6.7× over** the 16 ms real-time budget (<3.168 M cycles @ 198 MHz). This
directory stages moving the inference onto the RT595's **HiFi4 (Fusion F1) DSP**, whose
xa_nnlib int8 kernels cover **every op in our model** (verified in the SDK's prebuilt
`libtflm.a`, incl. `xa_nn_transpose_conv_sym8sxasym8s` for the decoder).

**This box can't build it** — no Cadence Xtensa toolchain / RT500 HiFi4 core config /
license here (same deploy-box split as the STM32N6 flow). Everything below is written to be
finished and built on the box that has `xt-clang` + the registered RT500 HiFi4 core.

## Architecture

```
   M33 (primary)                         HiFi4 (secondary)
   ── audio I/O, STFT, features ──        ── owns model + arena + 17 states ──
   main.cpp  (unchanged)                  dsp_main.cpp: SE_Init once, then
     BOARD_Init(); DSP_Boot();              loop: wait cmd -> SE_ProcessFrame -> done
     SE_Init(); ... SE_ProcessFrame() ──┐
   se_remote.cpp  (SE_* as RPC)         │  shared se_ipc_shared_t in SRAM:
     write feat[3*257] ─────────────────┼──▶ feat  ──▶ SE_ProcessFrame(feat,mask)
     spin/IRQ on status==DONE  ◀─────────┼─── mask  ◀── (int8 quant/dequant + state
     read mask[257]                      │             feedback all inside the DSP)
                                         └── kick via MAILBOX (bring-up: polling)
```

The RPC boundary is exactly `SE_ProcessFrame(feat floats -> mask floats)`, so `main.cpp`
and the model driver are reused **unchanged**; only the linked kernel library differs.

## What's in here

| file | side | role |
|------|------|------|
| `se_ipc.h`                     | both | shared-memory mailbox struct + command/status protocol |
| `dsp/dsp_main.cpp`             | DSP  | init + per-frame service loop (calls the real `SE_*`)   |
| `dsp/se_ipc_place.c`           | DSP  | `g_se_ipc` at the HiFi4 SRAM alias                      |
| `dsp/CMakeLists.txt`,`prj.conf`| DSP  | links the HiFi4 `libtflm.a`; reuses `../app` SE sources |
| `cm33/se_remote.cpp`           | M33  | `SE_*` API implemented as DSP RPCs (drop-in)            |
| `cm33/dsp_boot.{h,c}`          | M33  | `DSP_Boot()` + `g_se_ipc` at the M33 SRAM alias         |
| `cm33/CMakeLists.snippet.txt`  | M33  | exact source swap + the one-line `main.cpp` edit        |

Reused verbatim from `../app/`: `model_se_stream.{cpp,h}`, `model_ops_micro.cpp`.

## Steps to finish on the toolchain box

1. **Int8 model headers.** The offload targets the **int8** model (that's what the HiFi4
   kernels accelerate). Regenerate the two headers from the int8 export:
   ```
   PY=~/.venvs/rt595-export/bin/python
   $PY host/gen_model_data.py --model host_out/relu6deep_streaming_int8.tflite --output app/model_data.h
   $PY host/gen_io_layout.py  --model host_out/relu6deep_streaming_int8.tflite \
        --checkpoint_file cp_lisennet_conv_hardened_nc24_deep_relu6/g_best --output app/model_io_layout.h
   ```
   After this `MODEL_FEATURE_IS_INT8` becomes 1 and `MODEL_STATE_MAP` carries the int8
   requant params — the driver already branches on these. (The int8 tflite is committed at
   `host_out/relu6deep_streaming_int8.tflite`, VBD-calibrated; see the rt595-export notes.)

2. **Xtensa toolchain + core** — the one true blocker; all three parts are Cadence/NXP
   license-gated and cannot be obtained without an authenticated account. `sdk/cmake/
   toolchain/xtensa.cmake` needs exactly these env vars:
   - `XCC_DIR`      — XtensaTools install (`xt-clang`), version **≥ 10.0.1** (per
     `default_tool_version.cmake`; **RI-2023.11** satisfies this).
   - `XTENSA_CORE`  — **`nxp_rt600_RI2021_8_newlib`** (the RT500/RT600 HiFi4 core config;
     RT595 == RT500 family. This is the name the SDK's DSP examples export — grep confirmed).
   - `XTENSA_SYSTEM`— the config registry dir that has that core registered.

   A HiFi5 build host has the *tools* (RI-2023.11) but its registered core is HiFi5, not
   this RT500 HiFi4 core, and its license may be core-locked. To reuse it:
   ```
   xt-run --show-config=configs          # list registered cores; is nxp_rt600_RI2021_8_newlib there?
   xt-run --show-config=xttools          # tools path/version
   # if the core is missing: obtain the RT500/RT600 HiFi4 core config from NXP
   #   (MCUXpresso "Xtensa configuration for RT500/RT600 DSP" / the SDK DSP toolchain pack)
   #   and register it:  xt-regfile -r <core-package>     (or via Xtensa Xplorer core installer)
   # then verify the FlexLM license lists a feature covering that core (not just AIR/HiFi5).
   ```
   Note the core is **RI-2021.8**-built while the tools are RI-2023.11 — newer tools normally
   consume older core configs, but confirm on first `xt-clang --show-config=core`.
   Nothing here can be done on the training box (no tools, no license); it's a build-host /
   procurement step. If that box is SSH-reachable, the DSP target can be built there once the
   core+license gap is closed.

   **Toolchain env is pre-adapted: `env_rt595_hifi4.sh`.** It is a standard Cadence
   RI-2023.11 environment with the core set to the RT500 HiFi4
   (`nxp_rt600_RI2021_8_newlib`). Supply your own `XTENSA_LICENSE_FILE`; the script
   requires it and has no default.
   Source it on the machine that has the Cadence tools + a running FlexLM daemon
   then run the west build above. The three prerequisites it can't
   create (tools install, license daemon, RT500 core registered+entitled) are listed in the
   file, plus the RI-2021.8-core-vs-RI-2023.11-tools version check.

3. **Build the DSP image** (from `dsp/`), producing the reset/text/data blobs:
   ```
   west build -b evkmimxrt595 --sysbuild dsp -- -Dcore_id=hifi4
   ```
   Model it on `sdk/examples/dsp_examples/audio_demo_bm` (dual-core sysbuild) — that example
   is the closest template: M33 + DSP, audio streaming between them.

4. **Shared memory (the main board-specific seam).** Reserve a small non-cacheable SRAM
   block (`sizeof(se_ipc_shared_t)` ≈ 4 KB) that BOTH cores map:
   - M33 linker + `SE_IPC_SHARED_ADDR_M33` in `cm33/dsp_boot.c`
   - DSP linker + `SE_IPC_SHARED_ADDR_DSP` in `dsp/se_ipc_place.c`
   These are the M33 and HiFi4 **bus aliases of the same physical SRAM** — cross-check
   against the SDK rpmsg_lite/multicore shared-mem addresses for evkmimxrt595. If you must
   make it cacheable, fill in the `ipc_*` cache ops (marked `TODO`) instead.

5. **Inter-core kick.** Bring-up works with **polling** (the `kick_*` helpers are no-ops and
   both sides spin on `status`/`cmd`) — verify correctness first this way. Then wire the
   RT500 **MAILBOX** peripheral for interrupt-driven wake (replace the spins with WAITI/WFI)
   to cut per-frame latency; `sdk/examples/driver_examples/mailbox` is the reference.

6. **M33 changes.** Apply `cm33/CMakeLists.snippet.txt`: swap `model_se_stream.cpp`+
   `model_ops_micro.cpp`+`model_data.h` out of the M33 target for `se_remote.cpp`+
   `dsp_boot.c`, and add the one `DSP_Boot();` line to `main.cpp`.

7. **Pack + flash.** Combine M33 + DSP into a Multicore Packed Image (elf2sb / `mpi_loader`
   flow — see `sdk/.../mpi_loader/dsp_hello_world/example_board_readme.md`) and flash to
   `0x08000000`. The existing M33 test harness (16 baked frames, UART telemetry) then prints
   the **DSP** cycles/frame — same `frame, cycles, us, mask_checksum` format.

## Benchmark mode — measure cycles + latency, with and without HiFi4

`bench/` is a drop-in replacement for `main.cpp` that runs the same baked frames through
BOTH backends in one firmware, one run, and prints them side by side:

- **M33 int8 (CMSIS-NN)** — the "without HiFi4" baseline (local `SE_*`).
- **HiFi4 int8 (xa_nnlib)** — the "with HiFi4" path (DSP RPC).

Per frame it reports the **M33-observed latency** (DWT cycles + µs — for HiFi4 this is the
full RPC round-trip, the number that decides real-time), and for HiFi4 also the **DSP's own
compute cycles** (CCOUNT) so raw kernel speed is separated from RPC overhead. A per-frame
output checksum is cross-checked M33-vs-HiFi4 to confirm the offload is numerically faithful.
Example shape of the output. **Every number in the block below is invented to show the
format — none of it was measured.** This bench has never been run; see
`../results/PROVENANCE.md`.

```
=== ILLUSTRATIVE FORMAT ONLY — NOT MEASUREMENTS ===
=== M33 int8 (CMSIS-NN) ===
mean: 21000000 M33-cyc/frame (106060 us observed latency)
=== HiFi4 int8 (xa_nnlib) ===
mean:  1600000 M33-cyc/frame (8080 us observed latency)
       1200000 DSP-cyc/frame compute (6060 us); RPC overhead ~2020 us/frame
=== summary ===
effective speedup (incl. RPC): 13.12x
checksum cross-check: close (max |diff| = 4 over 16 frames)
```

Build per `bench/CMakeLists.snippet.txt`. It **degrades gracefully**: if the DSP image
isn't present yet, the HiFi4 backend prints "unavailable" and you still get the **M33 int8
baseline on its own** — so it's useful before the DSP half is built (and gives the int8 M33
number to compare against the current fp32 firmware's 21.2 M cyc/frame). The host-mockable
logic in `bench/bench_main.cpp` compiles clean (verified off-target).

## What "done" looks like

`main.cpp` is unchanged output-wise; the per-frame `cycles` column now reflects the HiFi4.
Target: **< 3.168 M cycles/frame** (real-time at the 16 ms hop). Expect a large drop from
the M33's 21.2 M — the model is 1.368 MMACs/frame and HiFi4 int8 SIMD runs ~2–4 MACs/cycle,
so the NN compute is well under budget; the remaining cost is the per-frame RPC + any
non-accelerated glue. Confirm the `mask_checksum` column still matches the M33 run
bit-for-similar (int8 rounding aside) so the offload is numerically faithful.

## Correctness cross-check (no DSP needed)

Before trusting on-target numbers, the int8 model itself is already verified on the i.MX8MP
path and on the M33; the HiFi4 kernels are NXP-validated int8. The one new risk is the RPC
data marshalling — the `static_assert`s in `dsp_main.cpp` catch header/layout drift at
compile time, and `se_ipc.h`'s `magic`/`abi_version`/`feature_len` guards catch it at boot.
