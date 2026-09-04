# deploy/rt595 — NXP i.MX RT595 (EVK-MIMXRT595) deploy target

Speech-enhancement deploy target for the **i.MX RT595**, mirroring `deploy/stm32n6/`.
The RT595 has two compute engines; this target does the **M33 first**, then the **HiFi 4**:

| Engine | Runtime | Status |
|---|---|---|
| **Cortex-M33** @ 200 MHz | TFLite-Micro / CMSIS-NN (eIQ) | **this scaffold** — M33 track |
| **Cadence HiFi 4 DSP** (Fusion F1) | eIQ NN kernels + Xtensa toolchain | later — see "HiFi4 track" |

Model family = the same as the N6 target (**LiSenNet** / ConvFSENet / NSNet2). The
first candidate is **LiSenNet hybrid-streaming nc24**: 11 states, GRU-on-time
unrolled as 1×1 convs, so every op is CMSIS-NN-native and the same int8 graph
carries forward to the HiFi4 path.

## Current status

Decided: **M33 runtime = TFLite-Micro (eIQ)**; **PyTorch→TFLite = ai-edge-torch** (direct,
no ONNX round-trip). Done and validated:
- ✅ **Host export pipeline** (`host/export_tflite.py`) — LiSenNet hybrid-nc24 streaming →
  FP32 tflite (216 KiB) and int8 w8/a8 (145 KiB) via ai-edge-torch + ai-edge-quantizer.
  No RNN ops; 12-in/12-out (feat + 11 states). Run in `~/.venvs/rt595-export`.
- ✅ **I/O contract** (`host/gen_io_layout.py` → `app/model_io_layout.h`) — maps feat/est_mag/
  state positions + quant params + `MODEL_STATE_MAP` feedback table (needs `--checkpoint_file`;
  see the ai-edge-torch note below).
- ✅ **Firmware core** (`app/`) — compiles against the real TFLM headers (arm-none-eabi-g++ 13.3):
  `model_se_stream.cpp` (MicroInterpreter driver, per-frame feat→Invoke→dequant→state requant),
  `model_ops_micro.cpp` (14-op resolver), `model_data.h` (embedded tflite), `main.cpp` (bring-up
  loop over `se_test_feats.h`, DWT-cycle telemetry over VCP).
- ✅ **SDK** in `~/toolchains`, symlinked `sdk/` (gitignored). ✅ Arm GCC 13.3, pyocd + pack, serial.

**Not yet:** the mcuxsdk **build integration** (west/mcux env + example.yml/CMakeLists + board
overlay) — see "Next steps". `make build`/`flash` are still stubs.

### Two findings baked into the firmware
- **State feedback requantises, not memcpy.** ai-edge-quantizer gives `state_in`/`state_out`
  independent scales (0/11 memcpy-safe), unlike the N6 stedgeai path. The driver dequant→requant
  each state each frame per `MODEL_STATE_MAP`.
- **ai-edge-torch shuffles tflite output positions** and its signature runner is buggy — only
  value-matching against the PyTorch model reliably recovers which output is est_mag / which state
  (hence `gen_io_layout.py --checkpoint_file`).
- **Bring up on the FP32 tflite first** — the only checkpoint today is the untrained `_dummy`, so
  its int8 scales are degenerate (feat scale ~3.9e-12). FP32 validates the loop; real int8 needs the
  trained ckpt + VoiceBank-DEMAND calibration (install `onnxruntime` in the export venv).

## Flashing (important)

The on-board **LPC-Link2 (CMSIS-DAP, HID)** does **not** work through WSL2 usbip — the
interrupt transfers reset (`ECONNRESET -104`); pyocd opens the probe but never gets a
DAP response. **Flash from the Windows side** (NXP **LinkServer** or pyocd-on-Windows)
against the real USB probe: WSL builds the `.axf`/`.bin`, Windows flashes it (the WSL
filesystem is reachable from Windows). The **serial VCP (`/dev/ttyACM0`, bulk CDC) works
fine over usbip** for the console / telemetry.

To (re)attach the board to WSL after plugging in:
```
# Windows admin (once):   usbipd bind --busid 1-7
usbipd attach --wsl --busid 1-7
# hidraw nodes come up root-owned each attach; for pyocd inspection only:
sudo chmod a+rw /dev/hidraw*
```

## HiFi4 track (later)

`xt-clang` (RI-2023.11) is installed, but a build host set up for a different Xtensa
target will have that target's core registered, not the RT595's HiFi4. To target the DSP
you need the NXP SDK's **RT500 HiFi4 core config** registered into XtensaTools, and the
Xtensa **license must cover the RT500 core** (licenses can be core-locked — verify). The
M33 int8 graph is designed to carry over unchanged.

## Layout

```
deploy/rt595/
├── config.mk      machine paths + RT595 geometry (edit, then `make doctor`)
├── Makefile       doctor | export | convert | build | flash | console
├── scripts/
│   └── doctor.sh  toolchain validator (works now)
├── host/          (pending) export + quant reusing ../../lisennet/{export_onnx,quant_onnx}.py
├── app/           (pending) M33 TFLite-Micro inference glue (mirrors ../stm32n6/app)
└── sdk/           (you) unzip the NXP MCUXpresso SDK for EVK-MIMXRT595 here
```

## Next steps

1. **You:** build+download the NXP SDK (see `config.mk` header) → unzip to `sdk/`.
2. **You:** decide the two open questions above (recommend: onnx2tf + TFLite-Micro).
3. **Then:** wire `host/` (reuse `lisennet/quant_onnx.py` signed-int8 recipe → onnx2tf →
   `.tflite`), build the M33 app in `app/`, flash from Windows, read PESQ/latency over serial.

The `host/` export half is **SDK-independent** and can be built as soon as decision #1 is
made — it doesn't need the board or the SDK.
