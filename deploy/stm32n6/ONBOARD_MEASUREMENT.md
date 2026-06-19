# STM32N6 on-board measurement — reproducible checklist

The validated, end-to-end procedure used to compile **ConvFSENet int8** to the
STM32N6570-DK Neural-ART NPU and measure **on-board latency / per-epoch cycles**,
fully scripted (no STM32CubeIDE). This is the *measurement* path — a RAM-resident
validation firmware driven over gdb + serial. For a standalone power-on
application see [README.md](README.md).

Paths below are the ones used during bring-up (WSL2 Ubuntu host); adjust to taste.

## 0. One-time setup

**Tools**
- ST Edge AI Core **4.0.1** — `~/stedgeai/install/4.0/Utilities/linux/stedgeai`
- STM32CubeProgrammer **2.22** — `~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/`
- STM32CubeCLT **1.21** (provides `ST-LINK_gdbserver`) — extracted **without root**:
  `dpkg-deb -x st-stm32cubeclt-*.deb <dst>` from the bundle, kept at
  `~/opt/st/stm32cubeclt_1.21.0/`
- Arm GNU **13.3** — `~/toolchains/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/`
  (CubeCLT's bundled gcc 14.3 is what `n6_loader` actually builds with)

**Hardware / WSL (the gotchas that cost the most time)**
- **ST-LINK firmware ≥ V3J17M11** is required by `ST-LINK_gdbserver`. Upgrade it on the
  **Windows host natively** (STSW-LINK007 / CubeProgrammer GUI). Do **not** upgrade over
  usbipd — the USB re-enumeration drops the WSL attachment mid-flash (`JNI 0x1002`).
- WSL2 networking = **mirrored** (`%UserProfile%\.wslconfig` → `[wsl2]\nnetworking=mirrored`,
  then `wsl --shutdown`). NAT mode hits a Windows-firewall TCP-3240 block on the usbip link.
- Board **BOOT switch in development-boot** — the RAM-resident firmware only persists in dev-boot.
- Attach the ST-LINK into WSL (Windows PowerShell):
  `usbipd attach --wsl --busid <id> --auto-attach`
  Verify in WSL: `lsusb | grep 0483` and `ls /dev/ttyACM0`.
- `n6_loader` tool config — `<N6_scripts>/config.json` (committed copy of the values used):
  `compiler_type=gcc`, `gdb_server_path`=CubeCLT `STLink-gdb-server/bin`,
  `gcc_binary_path`=CubeCLT `GNU-tools-for-STM32/bin`, `make_binary_path=/usr/bin/make`,
  `objcopy_binary_path`=CubeCLT objcopy, `cubeProgrammerCLI_binary_path`=CubeProgrammer 2.22.
- Profiler Python deps (the corporate pip mirror 401s → force public PyPI):
  ```
  python3 -m venv /tmp/profenv
  /tmp/profenv/bin/pip install --index-url https://pypi.org/simple/ \
        pyserial numpy tqdm colorama protobuf tabulate
  ```

`N6DIR=~/stedgeai/install/4.0/scripts/N6_scripts` ; `STEDGEAI=~/stedgeai/install/4.0/Utilities/linux/stedgeai`

## 1. Generate the NPU model
Run from `$N6DIR` so the profile's relative `./my_mpools/*.mpool` resolves.
```bash
cd "$N6DIR"
$STEDGEAI generate -m ~/eco8-neaixt/cp_convfsenet/g_best.onnx --target stm32n6 \
  --st-neural-art n6-noextmem@user_neuralart.json \
  --fix-parametric-shapes "{'B':1}" -n network -o /tmp/n6val_int
```
Profile choice (lever 1): `n6-allmems-O3` → weights in **external octoFlash** (7.14 ms/frame);
`n6-noextmem` → weights in **internal npuRAM**, fits on-chip, **4.40 ms/frame (1.62×)**.
(A `PRIx64` compile warning during generate is a benign debug-printf probe; the C model still emits.)

## 2. Build firmware + load weights + run it
```bash
cd "$N6DIR"
pkill -x ST-LINK_gdbserver 2>/dev/null   # clear any stale server
python n6_loader.py --config config.json -nf /tmp/n6val_int/network.c -bc N6-DK
```
This copies `network.c` into the bundled `NPU_Validation` app, builds it (gcc), flashes any
external-flash blobs via CubeProgrammer, then uses `ST-LINK_gdbserver` + gdb to load the RAM
firmware (and any internal-RAM weight blobs) and run it — leaving it alive at the serial
validation loop. Expect "Start operation achieved successfully".

## 3. Validate — accuracy + total latency
```bash
cd "$N6DIR"
$STEDGEAI validate -m ~/eco8-neaixt/cp_convfsenet/g_best.onnx --target stm32n6 \
  --st-neural-art n6-noextmem@user_neuralart.json \
  --mode target -d serial:/dev/ttyACM0:921600
```
Reports `duration ... ms by sample` and per-output cross-accuracy (cosine vs the FP32 ONNX).

## 4. Profile — per-epoch NPU/MCU cycles + memory bandwidth
```bash
RUNNER=~/stedgeai/install/4.0/scripts/ai_runner
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="$RUNNER" \
  /tmp/profenv/bin/python "$RUNNER/examples/npu_profiler.py" \
  -d serial:/dev/ttyACM0:921600 -c /tmp/n6val_int -b 16
```
Gives the per-epoch table (HW/HYBRID/SW), NPU vs MCU cycles, compute-utilization, and the
per-region memory bandwidth that exposed the external-flash bottleneck.

## Results (ConvFSENet int8, STM32N657 @ MCU 800 MHz / NPU 1 GHz)
| layout | latency/frame | RTF (16 ms) | NPU core | NPU core util | mask cos vs FP32 |
|---|---:|---:|---:|---:|---:|
| weights in octoFlash (`n6-allmems-O3`) | 7.14 ms | 0.45 | 3.73 ms | 27% (memory-bound) | 0.990 |
| weights in npuRAM (`n6-noextmem`)      | **4.40 ms** | **0.275** | 1.26 ms | **81% (compute-bound)** | 0.990 |

## Caveats
- Only **ConvFSENet** compiles for the Neural-ART today; NSNet2 (dense + structured) crashes
  the ST Edge AI compiler at this version.
- The validation firmware is a **volatile RAM image** — re-run step 2 after any power-cycle.
- With `n6-noextmem`, weights load over **gdb** (not flashed). That's fine for measurement; a
  standalone power-on deploy needs an on-chip-resident boot layout (separate effort).
- Lever 2 (move the M55 software share — per-frame FIFO state + int8 quant boundary — onto the
  NPU) requires re-quantizing the streaming export and is **not yet done**.
