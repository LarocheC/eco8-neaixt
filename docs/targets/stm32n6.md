# STMicroelectronics STM32N6 — deploy target

Sources: `deploy/stm32n6/`, plus `docs/targets/stm32n6-lisennet-npu.md` and the deployment
sections of `docs/models/lisennet.md` and `docs/models/convfsenet.md`. Unlike the RT595
target, **no raw board capture is committed for any number on this page** (§7).

> **The board and the compiler are not on the machine that trains.** The **training box**
> runs PyTorch, export, PTQ and host PESQ — it has *no* `stedgeai`, no board, no ST cloud
> credentials (`WINDOWED_DEPLOY_HANDOFF.md:12`, `EFFICIENCY_REWORK_PLAN.md:11`). The
> **deploy box** holds `stedgeai` 4.0.1, the Arm toolchain, STM32CubeProgrammer and the
> wired STM32N6570-DK, and compiles only — training happens elsewhere
> (`stm32n6-lisennet-npu.md:180`). The unit of work between them is an ONNX file plus a
> command list, not a shell session — `WINDOWED_DEPLOY_HANDOFF.md` is the template. Nothing
> in §4 steps 4–6 will run on the training box.

## 1. What this target is

STM32N6570-DK carrying an STM32N657: a **Cortex-M55 at 800 MHz** alongside the
**Neural-ART NPU at 1 GHz** (`ONBOARD_MEASUREMENT.md:87`). **ST Edge AI Core 4.0.1** lowers
an int8 ONNX to Neural-ART C (`network.c`) plus a weight blob
(`network_atonbuf.xSPI2.raw`, `Makefile:60-61`). The standalone app is cross-compiled with
**Arm GNU 13.3.Rel1** (`README.md:50`); note the *measurement* path is not — `n6_loader`
actually builds with CubeCLT's bundled gcc 14.3 (`ONBOARD_MEASUREMENT.md:19-20`).
STFT/iSTFT stay on the M55; the graph consumes `noisy_mag` (float32, 257) and predicts an
int8 magnitude `mask` (`README.md:159`). Hop 256 at 16 kHz = a **16 ms frame budget**
(`ONBOARD_MEASUREMENT.md:94`).

Two hardware facts shape everything below. The NPU is **int8-only and wants its weights on
chip** — ~1.8 MB npuRAM plus 1 MB cpuRAM (`NSNET2_DEPLOYMENT_NOTES.md:106`); profile
`n6-noextmem` packs them there, `n6-allmems-O3` parks them in octoFlash. And the part has
**no internal flash**: the ROM loads an FSBL from xSPI at `0x70000000`, and development
boot is a manual DIP-switch step (`README.md:118-122`).

## 2. Results

int8, STM32N657 at MCU 800 MHz / NPU 1 GHz, against the 16 ms frame period. All six rows
from `ONBOARD_MEASUREMENT.md:101-106`.

| Model (int8) | Profile | Latency/frame | RTF | Weights | PESQ† | PROVENANCE |
|---|---|---:|---:|---:|---:|---|
| LiSenNet nc24, **windowed** (emit_T=64) | `allmems-O3` | **1.15 ms**‡ | **0.072** | 0.49 MB octoFlash | 2.998 | **SILICON** — `validate`; the measured quantity is 73.63 ms/window, the per-frame figure is derived. No capture retained |
| NSNet2 `monarch_full` (sparse, re-exported) | `noextmem` | 2.13 ms | 0.13 | 0.72 MB on-chip | 2.848 | **SILICON** — `validate`, no capture retained |
| LiSenNet nc24, **streaming** (17-state FIFO) | `noextmem` | 2.79 ms | 0.174 | 47 KB on-chip | 2.963 | **SILICON** — `validate`, no capture retained |
| NSNet2 `monarch_8` (sparse, re-exported) | `noextmem` | 2.89 ms | 0.18 | 0.37 MB on-chip | 2.826 | **SILICON** — `validate`, no capture retained |
| ConvFSENet 192-384 (streaming) | `noextmem` | 4.40 ms | 0.275 | 1.40 MB on-chip | 2.911 | **SILICON** — `npu_profiler`, no capture retained |
| NSNet2 dense (`baseline`) | `allmems-O3` | 22.94 ms | 1.43 | 2.70 MB octoFlash | 2.833 | **SILICON** — `npu_profiler`, **not real-time** |

† PESQ is **host**-measured (ONNX Runtime, full 824-utterance VoiceBank-DEMAND test split),
never a board number — LiSenNet from `docs/models/lisennet.md:208,461`, the rest from
`NSNET2_DEPLOYMENT_NOTES.md:13-16`. What licenses carrying it onto the device is a mask
cosine of 0.990–0.99994 across the table, but the reference differs by row and one row is
not on-target at all: the LiSenNet rows are **on-target vs the host int8 ONNX** (0.99829 /
0.9941, `docs/models/lisennet.md:425`, `ONBOARD_MEASUREMENT.md:121-123`), the monarch and
ConvFSENet rows are on-target vs FP32 (`ONBOARD_MEASUREMENT.md:88-91,103-104`), and dense
NSNet2's 0.9946 is a **host** int8-vs-FP32 cosine, verified off-board because `validate`
crashes on that graph (`NSNET2_DEPLOYMENT_NOTES.md:85,132-133`).
‡ Derived: the windowed graph measures **73.63 ms per 64-frame window** (std 0.32, 10 runs,
`docs/models/lisennet.md:423`) and buffers 1.02 s per inference — throughput, not latency.
The streaming rows emit every 16 ms hop.

**SILICON** means read off the board over serial by `validate --mode target` or
`npu_profiler.py`; `validate` runs ~1 ms higher than the profiler, so the method is named
per row and the two must not be mixed (`ONBOARD_MEASUREMENT.md:140-142`). The provenance
vocabulary is the one defined in `deploy/rt595/results/PROVENANCE.md` — SILICON / ISS /
MODELLED. Nothing on this page is ISS (no instruction-set simulator is involved on this
part), and a `generate` static report (epoch split, MACC, placement) would be **MODELLED**,
not a measurement — none is quoted as a latency here.

Weight locality is the lever on this part: moving ConvFSENet's weights from octoFlash to
npuRAM took it **7.14 → 4.40 ms/frame**, NPU utilisation 27% → 81%
(`ONBOARD_MEASUREMENT.md:90-91`). Dense NSNet2 cannot make that move — 2.70 MB does not
fit — and that is essentially the whole of its 22.94 ms
(`NSNET2_DEPLOYMENT_NOTES.md:105-111`).

## 3. Power — no figures exist

This repo carries **no power or energy number for the STM32N6**, modelled or measured; a
grep over `deploy/stm32n6/*.md` for mW/mJ/current returns only power-cycle and PowerShell
prose. Nothing is quoted because nothing traces to a file.

## 4. Running your own model on it

Steps 1–3 run on the training box; 4 needs the deploy box but no board; 5–6 need the board.

1. **Export** a streaming FIFO-state or stateless windowed FP32 ONNX
   (`lisennet.export_onnx --streaming | --windowed --emit_T N`,
   `convfsenet.export_onnx --windowed`). Structured NSNet2 needs the rank-2 re-export in
   `host/export_blockdiag_npu.py` — per-block `Slice`+`MatMul`+`Concat`, the vocabulary the
   compiler already maps (`NSNET2_DEPLOYMENT_NOTES.md:208-222`).
2. **Quantize** to signed int8 QDQ. `QInt8` is **mandatory**; the NPU rejects `QUInt8`
   (`stm32n6-lisennet-npu.md:87-88`). GRU-bearing graphs also need `skip_optimization=True`
   in `quant_pre_process`, or ORT's `MatMulAddFusion` emits an activation-`C` `Gemm` and
   `generate` dies with `list index out of range` (`NSNET2_DEPLOYMENT_NOTES.md:35,58-63`).
3. **Verify on host** — parity vs the offline model (<1e-6 for the windowed ConvFSENet,
   `docs/models/convfsenet.md:125`; ~5e-7 for the monarch re-export,
   `NSNET2_DEPLOYMENT_NOTES.md:220`), int8 PESQ on the full split, an op histogram clear of
   `Pad`/`Einsum`/rank-5 — then hand the `.onnx` over.
4. **Generate** — `stedgeai generate ... --st-neural-art n6-noextmem@user_neuralart.json
   --fix-parametric-shapes "{'B':1}"`, run from stedgeai's `N6_scripts` dir so the profile's
   relative `.mpool` resolves. Read `network_generate_report.txt`; grep for `signo=11`/`E103`.
5. **Load** — `n6_loader.py -nf network.c -bc N6-DK` builds the `NPU_Validation` app and
   gdb-loads a RAM-resident firmware over ST-LINK.
6. **Validate / profile** — `stedgeai validate --mode target -d serial:/dev/ttyACM0:921600`
   for latency and per-output cosine; `npu_profiler.py -b 16` for the per-epoch
   HW/hybrid/SW split and memory bandwidth.

For a standalone power-on application instead, `make deploy` runs generate → io-layout →
model-install → build → sign → flash (`Makefile:79`). Recurrent-state glue is
`app/ai_dpu_se_stream.c` — the stock app's `ai_dpu.c` assumes one input and one output and
cannot carry these models (`README.md:155-156`).

## 5. Without hardware

**`stedgeai generate` alone is a real gate** — it settles compile-or-crash, the epoch split,
and whether weights fit on-chip; that is how all four Neural-ART blockers were found
(`stm32n6-lisennet-npu.md:36-48`) and how `EFFICIENCY_REWORK_PLAN.md`'s Gate-0 checks are
framed. **ST's Edge AI Developer Cloud board farm** returns MACC, cycles, duration, RAM and
ROM for an uploaded model (`make bench-cloud`, credentials in `cloud/.env`;
`README.md:130-138`, `cloud/dev_cloud_bench.py:3,98-99`). Accuracy has no host path:
`validate --mode host` is **unsupported** on Neural-ART (`README.md:151`).

## 6. Prerequisites

**ST Edge AI Core 4.0.1**, installed to line up with `STM32N6-GettingStarted-Audio`'s
`ll_aton` middleware — but not exactly matched: the bundled app was generated with 4.0.0, a
patch behind, and `make build` can still warn "Possible mismatch in ll_aton library used"
(`config.mk:7-10`) · **STM32CubeProgrammer 2.21+** to sign and flash, where `-align` is now
required (`Makefile:69-71`); 2.22 is the version actually used on the deploy box
(`ONBOARD_MEASUREMENT.md:15`), and README still lists it as not installed on the training
box (`README.md:51`) · **STM32CubeCLT 1.21** for the `ST-LINK_gdbserver` `n6_loader` drives
· **Arm GNU 13.3.Rel1** for `make build` · **ST-LINK firmware ≥ V3J17M11**, upgraded
natively on Windows, never over usbipd · board in **development boot mode**. Under WSL2 the
ST-LINK arrives via usbipd-win with `mirrored` networking. `make doctor` validates
`config.mk`.

## 7. Known gaps

- **No committed raw capture for any number here** — `git ls-files deploy/stm32n6` returns
  22 files: scripts, app C sources/headers and prose, no measurement output. Re-running the
  six measurements with stdout saved under `results/` would close the single largest
  evidence gap.
- **Windowed ConvFSENet never reached the board** — host int8 PESQ 2.913 (FP32 2.933) on a
  graph with zero FIFO/state/Pad nodes (`docs/models/convfsenet.md:136`,
  `WINDOWED_DEPLOY_HANDOFF.md:22-25`); Gate-0 and Phase-4 are still queued. **LiSenNet
  relu6-deep never reached it either** — host int8 3.014, or 3.052 with the decoder left
  FP32, whose codegen is an open question (`stm32n6-lisennet-npu.md:152-158`,
  `docs/models/lisennet.md:289,291`).
- **ConvFSENet is at its floor (~4.4 ms).** Both export-surgery levers are closed: native
  dilation was a wash (5.42 ms, +33 epochs), int8 states a no-op (`TODO.md:116-126`). The
  5.42 ms is a `validate` figure and the 4.40 ms baseline a profiler figure, ~1 ms apart by
  method — the source flags the mismatch and calls the conclusion method-independent
  (`TODO.md:81-85`). The cost is inherent per-block Hybrid `Slice`/`Concat` plumbing.
- **An `n6-noextmem` measurement is not a deployment** — weights load over gdb rather than
  being flashed; a power-on build needs an on-chip-resident boot layout
  (`ONBOARD_MEASUREMENT.md:147-148`).
- **Toolchain edges.** `validate` crashes on the un-fused dense NSNet2 (it re-introduces the
  GRU `Gemm` fusion) → use `npu_profiler` there. The validation firmware is a volatile RAM
  image and the loader can wedge; recovery is a USB re-plug, not an SWD reset
  (`ONBOARD_MEASUREMENT.md:138-146`).
- **Butterfly NSNet2 variants stay undeployable** — NPU-hostile ops (6-D reshape/reduce,
  `ONBOARD_MEASUREMENT.md:135-136`), and they collapse under int8 PTQ (2.577 and 2.202, vs
  ~2.83 for monarch/dense; `NSNET2_DEPLOYMENT_NOTES.md:143,162-163`). Re-exported monarch
  PESQ is **inherited, not re-measured** (host cosine 0.9990 / 0.9995 to the stock int8,
  `NSNET2_DEPLOYMENT_NOTES.md:234`).
