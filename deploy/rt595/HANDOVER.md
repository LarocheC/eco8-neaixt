# RT595 deploy — takeover checklist

Everything needed to **build and flash** the LiSenNet streaming speech-enhancement
firmware for the **NXP i.MX RT595 (EVK-MIMXRT595), Cortex-M33 / TFLite-Micro** on a new
machine. Branch `nxp-rt595-deploy` (off `main`; no paper WIP).

**Current state:** the real trained **`conv-hardened`** LiSenNet model is exported to a
FP32 `.tflite`, embedded (`app/model_data.h`, committed), and the firmware **builds to a
flashable `.bin` in WSL**. The only thing that didn't work in WSL is **flashing** — do
that on this new machine.

> ### ⚠️ Flash from NATIVE Linux, not WSL2
> Every USB path to this board fails under WSL2's **usbip** (`vhci_hcd` throws `-104`
> ECONNRESET on both HID *and* CDC): SWD probe, USB-HID ISP, and even the UART-ISP VCP
> (raw read hangs). It's a WSL2 transport limitation. On **bare-metal Linux** the board
> is on the real USB stack — no usbip — and `pyocd`/`blhost`/LinkServer just work. Build
> anywhere; **flash on native Linux** (or Windows).

---

## 0. Prereqs (install once)

- **Arm GNU toolchain 13.3.Rel1** (`arm-none-eabi-gcc`). Install to `~/toolchains/` or
  set `ARMGCC_DIR`. (14.2+ is "recommended" by the SDK but 13.3 builds fine.)
- **CMake ≥ 3.22**, **Ninja**, **Python 3.10** (for the venvs; 3.13 is too new for the
  export stack).
- Note: if your `pip` points at a private index that 401s, prefix installs with
  `PIP_INDEX_URL=https://pypi.org/simple/ PIP_EXTRA_INDEX_URL=`.

## 1. NXP MCUXpresso SDK (reinstall — not in the repo, 2.4 GB)

Get the SDK for **EVK-MIMXRT595** from **mcuxpresso.nxp.com** (SDK Builder or the
`mcuxsdk` west manifest). It's the **new CMake/Kconfig `mcuxsdk`** flavour (v26.06 used
here). Required components: **eIQ / TensorFlow-Lite-Micro** (the M33 runtime) and, for
the later HiFi4 track, the **DSP/Cadence** middleware.

```bash
# unzip/checkout so that <SDK>/mcuxsdk/... exists, then point deploy/rt595/sdk at it:
ln -sfn /path/to/SDK_.../mcuxsdk deploy/rt595/sdk
```
Sanity-check the key pieces exist:
`sdk/devices/RT/RT500/MIMXRT595S/MIMXRT595S_cm33.h`,
`sdk/middleware/eiq/tensorflow-lite/lib/cm33/armgcc/libtflm.a`,
`sdk/examples/eiq_examples/tflm_kws` (the template our example was cloned from).

## 2. Build environment (west + mcux build reqs)

```bash
python3.10 -m venv ~/.venvs/rt595-build
~/.venvs/rt595-build/bin/pip install west
~/.venvs/rt595-build/bin/pip install -r deploy/rt595/sdk/scripts/requirements-base.txt
```

## 3. Build the firmware

```bash
make -C deploy/rt595 build          # or: deploy/rt595/scripts/build.sh
```
`build.sh` runs `install_example.sh` (materialises the example in the SDK tree via
symlinks) then `west build`. Output: `deploy/rt595/build/lisennet_se_cm33.{elf,bin}`
(~465 KB `.bin`). Overridable env: `SDK_DIR`, `ARMGCC_DIR`, `BUILD_VENV`, `CONFIG`
(`debug`|`release`).

**Build gotchas (already handled — FYI):** the dual-core chip **requires
`-Dcore_id=cm33`** (build.sh passes it); the example was **cloned from `tflm_kws`**
because hand-authored CMakeLists hit an opaque board-port resolution error; `main.cpp`
declares `extern "C" void BOARD_Init(void)` (the overlay's `app.h` proto lacks it →
C++ mangling link error). It links against the **prebuilt `libtflm.a`**.

## 4. Flash (native Linux)

```bash
# one-time: probe access + target pack
pipx install pyocd && pyocd pack install mimxrt595sffoc
sudo cp deploy/rt595/scripts/50-cmsis-dap.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger      # replug the probe

deploy/rt595/scripts/flash.sh                              # pyocd flash over LPC-Link2 SWD
```
Alternatives: **blhost** via the RT595 ROM ISP over `/dev/ttyACM0` (set SW7 to serial/UART
ISP, reset), or **LinkServer** (`LinkServer flash EVK-MIMXRT595 load ....bin --addr 0x08000000`).

## 5. Serial console (telemetry)

`main.cpp` runs the baked `se_test_feats.h` frames and prints per-frame DWT cycles/µs
over the VCP:
```bash
screen /dev/ttyACM0 115200        # Ctrl-A k to quit
```
On first boot check the `AllocateTensors` "used" print and, if needed, tune
`SE_TENSOR_ARENA_SIZE` in `app/model_se_stream.cpp` (currently 512 KB for the fp32 model).

---

## 6. (Optional) Regenerate the model — different checkpoint / int8

Only needed to change the embedded model. Needs the research repo's `lisennet/`,
`common/` packages (present on `main`, so on this branch too) + a heavier venv.

```bash
python3.10 -m venv ~/.venvs/rt595-export
~/.venvs/rt595-export/bin/pip install ai-edge-torch ai-edge-litert ai-edge-quantizer
deploy/rt595/scripts/fetch_checkpoint.sh conv-hardened      # -> cp_lisennet_conv_hardened/

PY=~/.venvs/rt595-export/bin/python
# export -> fp32 tflite (int8 needs VoiceBank-DEMAND for calibration; see notes)
$PY deploy/rt595/host/export_tflite.py --checkpoint_file cp_lisennet_conv_hardened/g_best \
     --output deploy/rt595/host_out/model_fp32.tflite
# regenerate the firmware headers from the tflite (+ torch ground-truth):
$PY deploy/rt595/host/gen_model_data.py --model deploy/rt595/host_out/model_fp32.tflite \
     --output deploy/rt595/app/model_data.h
$PY deploy/rt595/host/gen_io_layout.py  --model deploy/rt595/host_out/model_fp32.tflite \
     --checkpoint_file cp_lisennet_conv_hardened/g_best --output deploy/rt595/app/model_io_layout.h
make -C deploy/rt595 build
```

**Export gotchas (already solved in the scripts):**
- ai-edge-torch was **renamed to `litert_torch`** — import that (the shim lacks `.convert`).
- Its **PT2E int8 path is broken** with the pinned torch (needs ≥2.11) — int8 is done via
  **ai-edge-quantizer** (`static_wi8_ai8`), calibrated on VoiceBank-DEMAND. **VBD is not in
  the repo** — real int8 needs the dataset + `onnxruntime` in the export venv. FP32 is the
  right bring-up model regardless.
- `gen_io_layout.py` needs `--checkpoint_file`: ai-edge-torch shuffles tflite output
  positions and its signature runner is buggy, so it maps output→role by **value-matching
  against the PyTorch model** (the only reliable method). It also emits `MODEL_STATE_MAP`
  (per-state feedback + requant params) and `MODEL_FEATURE_IS_INT8`; the driver
  (`model_se_stream.cpp`) branches on these (fp32 copies floats, int8 quant/dequant +
  requant state feedback — ai-edge-quantizer gives state_in/state_out independent scales).

---

## 7. Layout

```
deploy/rt595/
├── HANDOVER.md          this file
├── README.md            design notes, findings, HiFi4-later plan
├── config.mk            paths + RT595 geometry (n_fft 512, hop 256, 257 bins)
├── Makefile             doctor | build | flash | console
├── scripts/             doctor.sh · build.sh · install_example.sh · flash.sh ·
│                        fetch_checkpoint.sh · 50-cmsis-dap.rules
├── app/                 M33 firmware: main.cpp, model_se_stream.{cpp,h}, model_ops_micro.cpp,
│                        + generated model_data.h / model_io_layout.h / se_test_feats.h
├── host/                PyTorch→tflite export + header generators (litert_torch / ai-edge-quantizer)
├── mcux_example/        mcuxsdk example scaffolding (installed into the SDK tree by install_example.sh)
└── sdk/                 -> symlink you create to the reinstalled mcuxsdk (gitignored)
```

## 8. HiFi4 DSP track (future)

The M33 int8 graph is designed to carry over to the RT595's **HiFi4 (Fusion F1) DSP**
(eIQ HiFi4 kernels + `middleware/cadence/nnlib`; seeds: `examples/eiq_examples/tflm_kws_hifi4`,
`tflm_cifar10_fusionf1`). Needs the Cadence Xtensa toolchain with the **RT500 HiFi4 core
config** registered and a license that covers it. See README §"HiFi4 track".
