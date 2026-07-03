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

## Results — all five models (same generate → load → validate/profile flow)
On-board, int8, STM32N657 @ MCU 800 MHz / NPU 1 GHz; frame period = hop 256 @ 16 kHz = 16 ms.
The NSNet2 family needed graph surgery first (see
[NSNET2_DEPLOYMENT_NOTES.md](NSNET2_DEPLOYMENT_NOTES.md)).

| model (int8) | profile | latency/frame | RTF | weights | on-target mask cos | method |
|---|---|---:|---:|---:|---:|---|
| **LiSenNet conv-hardened nc24** (windowed, emit_T=64) | `n6-allmems-O3` | **1.15 ms**¹ | **0.072** | 0.49 MB octoFlash | 0.99829 | `validate` |
| NSNet2 `monarch_full` (sparse, re-exported)     | `n6-noextmem` | 2.13 ms | 0.13 | 0.72 MB on-chip | 0.99979 | `validate` |
| NSNet2 `monarch_8` (sparse, re-exported)        | `n6-noextmem` | 2.89 ms | 0.18 | 0.37 MB on-chip | 0.99994 | `validate` |
| ConvFSENet (conv)                                | `n6-noextmem` | 4.40 ms | 0.275 | 1.40 MB on-chip | 0.990 | `npu_profiler` |
| NSNet2 dense (`baseline`)                        | `n6-allmems-O3` | 22.94 ms | 1.43 | 2.70 MB octoFlash | 0.9946 | `npu_profiler` |

¹ LiSenNet is **windowed, not streaming**: 73.63 ms per 64-frame window = 1.15 ms per
emitted frame, but each inference buffers 1.02 s of audio (the `emit_T` export knob
trades block latency vs recompute). The streaming rows above emit every 16 ms hop.
LiSenNet compile: 102 epochs (60 HW / 36 hybrid / 6 SW), MACC 177.7 M/window,
activations 2.72 MB all on-chip; `n6-noextmem` cannot fit it (weights+activations
3.35 MB > 2.8 MB pools) and isn't needed — 0.49 MB of octoFlash weights stream at
~13 MB/s avg, negligible. Needs `--fix-parametric-shapes "{'B':1}"`. SW share = the
3 encoder stride-(1,3) k=(2,5) convs (23.2 ms) + I/O `Gather` layout ops (14.9 ms).

Fastest per emitted frame: **LiSenNet windowed (1.15 ms / RTF 0.072)**, which also
carries the best real-time int8 PESQ (2.998). Fastest *streaming* (16 ms hop) model:
**`monarch_full` (2.13 ms / RTF 0.13)**; ConvFSENet wins streaming int8 PESQ (2.91 vs
2.85). Dense overflows on-chip RAM → memory-bound → not real-time.

## Caveats
- **All four models above now deploy** on the Neural-ART. NSNet2 dense needs a
  `skip_optimization` re-quant to compile; the sparse monarch variants need the conv-native
  rank-2 re-export (`host/export_monarch_npu.py`). **Butterfly is NPU-hostile** (6-D
  reshape/reduce — doesn't compile). Full root-cause analysis:
  [NSNET2_DEPLOYMENT_NOTES.md](NSNET2_DEPLOYMENT_NOTES.md).
- `stedgeai validate --mode target` works for ConvFSENet and the monarch models, but **crashes
  for the un-fused dense NSNet2** (it re-runs ORT optimization and re-introduces the GRU fusion)
  → measure dense via `npu_profiler` instead. `npu_profiler` PER_LAYER is slow over serial for
  100+-node graphs (it can time out); `validate`'s `duration ... by sample` runs ~1 ms higher
  than the profiler's pure-inference figure, so compare like-for-like.
- The validation firmware is a **volatile RAM image** — re-run step 2 after any power-cycle.
  After a `validate` run the ST-LINK/loader state can wedge (`Loading memories failed` /
  `DEV_USB_COMM_ERR`); recover by re-plugging the USB (then `usbipd attach --wsl`) — an SWD
  software reset does **not** clear it.
- With `n6-noextmem`, weights load over **gdb** (not flashed). That's fine for measurement; a
  standalone power-on deploy needs an on-chip-resident boot layout (separate effort).
- **Lever 2** (move ConvFSENet's M55 software share onto the NPU) was attempted and **closed**:
  Y1 (native dilation, kill the FIFO `Gather`) is a wash — the `Gather` is cheap SW and the
  compiler's `SpaceToDepth` dilation realization costs more; Y2 (int8 states) is a no-op —
  stedgeai already optimizes the state boundary. ConvFSENet is at its N6 floor (~4.40 ms).
  See [TODO.md](TODO.md).
