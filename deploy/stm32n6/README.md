# STM32N6570-DK deployment (fully scripted, no STM32CubeIDE)

Deploy this repo's streaming speech-enhancement ONNX models to the STM32N6570-DK
(Neural-ART NPU + Cortex-M55) from the command line only — generate → build → sign →
flash, plus board-free benchmarking on ST's cloud farm. Designed for agent-driven
iteration: every step is a non-interactive `make` target.

> **Deploy ConvFSENet first.** On `stedgeai` 4.0.1, measured on-board:
>
> | model | result |
> |---|---|
> | `monarch_8` sparse NSNet2 (re-exported) | ✅ **2.89 ms/frame, RTF 0.18** — real-time, fastest, weights on-chip |
> | `cp_convfsenet/g_best.onnx` (pure Conv1d) | ✅ **4.40 ms/frame, RTF 0.275** (real-time) |
> | `cp_baseline/g_best.onnx` (NSNet2 dense) | ✅ **22.94 ms/frame, RTF 1.43** — deployable after a re-quant, but *not* real-time |
>
> All three speech-enhancement models now run on the N6. The sparse `monarch_8` is the
> fastest (its 0.37 MB weights fit on-chip), then ConvFSENet; the dense GRU baseline runs
> but is memory-bound (2.70 MB weights overflow on-chip RAM → RTF 1.43). NSNet2 dense needs
> a `skip_optimization` re-quant to compile; `monarch_8` needs a conv-native rank-2
> re-export (`host/export_monarch8_npu.py`). Full analysis — root causes, the fixes, the
> deployed numbers, variant selection: **[NSNET2_DEPLOYMENT_NOTES.md](NSNET2_DEPLOYMENT_NOTES.md)**.

## Layout

```
deploy/stm32n6/
  Makefile            orchestrates the whole pipeline (run `make help`)
  config.mk           all machine-local paths + the N6 memory map (edit this)
  scripts/
    doctor.sh         validate toolchain paths/versions  → make doctor
    generate.sh       stedgeai ONNX → Neural-ART C        → make generate
    flash.sh          sign + 3× STM32_Programmer_CLI write → make flash
  host/
    gen_io_layout.py  emits app/model_io_layout.h from the compiled model
  app/
    model_io_layout.h AUTO-GENERATED I/O map (counts, dtypes, mask dequant, feedback)
    ai_dpu_se_stream.c/.h  multi-I/O + recurrent-state DPU glue (the real C work)
  cloud/
    dev_cloud_bench.py     benchmark on ST's N6 board farm (no local board)
    .env.example           MyST credential template (copy to .env, gitignored)
```

## 1. Prerequisites

| tool | needed for | status on this machine |
|---|---|---|
| **ST Edge AI Core** (`stedgeai`) | ONNX → NPU C | ✅ `v4.0.1` at `install/4.0` (matches the app's ll_aton) |
| **Arm GNU toolchain** | cross-compile | ✅ `13.3.Rel1` at `~/toolchains/arm-gnu-toolchain-13.3.rel1-...` |
| **STM32CubeProgrammer 2.21+** | sign + flash | ❌ to install in WSL (see "WSL flashing" below) |
| **STM32N6570-DK board** | flash + run | needed for `flash` / `validate-target` |

`make doctor` is green for stedgeai + gcc; install CubeProgrammer, then:

```bash
cd deploy/stm32n6
make doctor      # confirms paths/versions
make bootstrap   # clones STM32N6-GettingStarted-Audio (the ready-made app)
```

### ⚠️ Version caveat (largely resolved)
`STM32N6-GettingStarted-Audio`'s bundled `ll_aton` NPU middleware is version-locked to
the `stedgeai` that generated it (**4.0.0**). We installed **4.0.1** (a patch off) so the
generator and app agree on the 4.0 major/minor. If `make build` still warns *"Possible
mismatch in ll_aton library used"*, refresh the app's `ll_aton` per ST's "update project
with a new version of ST Edge AI Core" procedure (or `git checkout` an app tag built with
4.0.1). The old 3.0 install remains at `install/3.0` if you ever need to compare.

### WSL flashing (usbipd path — chosen)
WSL2 can't see USB devices natively, so the ST-LINK on the STM32N6570-DK is passed in
from Windows with **usbipd-win**:

```powershell
# On the Windows host (PowerShell as Administrator), once:
winget install usbipd
usbipd list                                   # find the ST-Link bus id
usbipd bind   --busid <id>                    # one-time
# each session — --auto-attach survives the re-enumeration the flasher triggers on reset:
usbipd attach --wsl --busid <id> --auto-attach
```
```bash
# In WSL, once: usbip client + let the attached ST-LINK be seen
sudo apt install -y linux-tools-generic usbutils hwdata
lsusb | grep -i st-link                        # confirm it's visible
# install CubeProgrammer's udev rules for non-root access (ships in the package):
sudo cp ~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/Drivers/rules/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Caveat: the STM32N6 ST-LINK re-enumerates across the reset each flash triggers, which can
drop the usbip attachment mid-`flash`; `--auto-attach` mitigates it but expect the
occasional re-attach. config.mk's `PROG_CLI`/`SIGN_CLI`/`EXT_LOADER` already point at the
default WSL install path (`~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/...`).

## 2. The pipeline

```bash
make generate        # ONNX → network.c/.h, stai_network.c/.h, int8 weights (.raw)
make io-layout       # regenerate app/model_io_layout.h from the compiled model
make model-install   # copy generated files into the app + objcopy weights → .hex
make build           # arm-none-eabi-gcc, pure Makefile (no IDE)
make sign            # add the STM32N6 boot header (dev mode, no key)
make flash           # FSBL + signed app + weights over SWD
# …or all at once:
make deploy
```

- **`generate`** runs from the Neural-ART profile dir so `default@neural_art.json`
  resolves its `stm32n6.mpool`, pins batch with `--fix-parametric-shapes "{'B':1}"`,
  and does **not** force int8 I/O (ConvFSENet's `noisy_mag` can't be folded to int8 —
  it stays float32; the states/mask are int8 automatically).
- **`io-layout`** re-derives `app/model_io_layout.h` and **hard-fails** if any
  `state_out[k]`/`state_in[k]` pair stops sharing an int8 scale (which would break the
  memcpy feedback — see §6).
- **`model-install`** sets the weights base to `0x70180000`; this **must** match the
  `objcopy --change-addresses` and the `flash` address.

## 3. Boot & flash reality (STM32N6 has no internal flash)

The ROM bootloader loads an FSBL from **external xSPI flash** at `0x70000000`. So:
- The board must be in **development boot mode** (BOOT1 DIP switch) before `make flash`,
  or the external loader can't write — this is a **manual, non-scriptable** step.
- Every image the ROM jumps to needs an **authentication header** even in dev mode
  (`-nk` = header, no key). Production secure-boot needs real ECDSA keys + OTP fuses.
- Flashing is 3 independent writes (FSBL `.hex`, signed app `.bin` @ `0x70100000`,
  weights `.hex`). Weights flash **separately**, so you can push a re-quantized model
  without rebuilding firmware.
- After flashing, set BOOT1 back to flash-boot and power-cycle.

## 4. Board-free benchmarking (recommended for iteration)

Get real NPU latency/cycles/RAM/ROM from ST's cloud board farm before you even wire up
hardware:

```bash
cp cloud/.env.example cloud/.env && $EDITOR cloud/.env   # your MyST login (gitignored)
git clone --depth 1 https://github.com/STMicroelectronics/stm32ai-modelzoo-services  # has the client
make bench-cloud      # uploads MODEL, benchmarks STM32N6570-DK, prints JSON
```

Credentials are the **same MyST account** you used in the web GUI; the password is read
from `cloud/.env` (never the command line) and the client caches a JWT in `~/.stmai_token`.
Note: the local `stedgeai validate` has **no** cloud backend — remote on-target runs go
only through this client.

## 5. On-target numerical validation (board wired over UART)

```bash
make validate-target   # stedgeai validate --mode target -d serial:921600
```
`--mode host` is **not** supported for Neural-ART, so accuracy validation needs the board.

## 6. The one piece of real integration: `ai_dpu_se_stream.c`

The stock audio app's `ai_dpu.c` asserts a single input and single output. Our models
have 10 of each with recurrent state, so this module replaces that glue. Verified facts
it relies on (all from the compiled `g_best.onnx`, see `model_io_layout.h`):

- **I/O order** = ONNX order: index 0 = `noisy_mag`(float32, 257) / `mask`(int8, 257);
  indices 1..9 = state tensors. Feedback is simply `out[k] → in[k]`.
- **State feedback is a raw `memcpy`**: each `state_out[k]` shares an *identical* int8
  scale + zero-point (zp = −128) with `state_in[k]` — verified, and re-checked on every
  `make io-layout`. No per-frame requantization.
- **Zero-state = `memset(0x80)`**, not `0x00` (int8 code for float 0.0 when zp = −128).
- **Mask dequant** = `(q − (−128)) · 0.003922` → a `[0,1]` mask.
- **Cache discipline**: clean D-cache after writing NPU inputs, invalidate before reading
  outputs (top cause of garbage NPU output). Uses the package's `mcu_cache_*` helpers.

Wire it into the app's audio loop:
```c
se_stream_init();                              // once at boot
/* per frame, after the M55 STFT produces a 257-bin magnitude: */
se_stream_process_frame(mag257, mask257);      // float in → float mask out
/* then apply mask to the saved complex spectrum and iSTFT/overlap-add on the M55 */
```
STFT/iSTFT stay on the M55 (CMSIS-DSP `arm_rfft_fast_f32`, wrapped by the package's
`STM32_AI_AudioPreprocessing_Library`); the NPU only predicts the mask.

**Zero-copy option:** to drop the state memcpy, regenerate with
`--no-inputs-allocation --no-outputs-allocation` and alias `out[k]`/`in[k]` to the same
32-byte-aligned RAM via `stai_network_set_inputs/outputs` (not compatible with `--reloc`).

## 7. Performance note

ConvFSENet compiles to **88 epochs: 30 on the NPU, 27 hybrid, 31 in software on the M55**
— the SW epochs are the streaming FIFO plumbing (`Slice/Gather/Dequantize/Quantize`) plus
the float boundary, so per-frame latency is M55-bound, not NPU-bound. Activations are
~42 KiB (on-chip SRAM); weights 1.42 MiB (external xSPI). Measure real latency with
`make bench-cloud` or `make validate-target` before optimizing; the lever is rethinking
how the FIFO state is expressed in the ONNX export so those ops stay int8 / map to HW.
