# deploy/rt595/config.mk — machine-local paths and the RT595 target config.
# Edit the paths for your machine, then run `make doctor` to validate them.
# Every value is overridable on the command line, e.g. `make build MODEL=/path/x.tflite`.
#
# Target: NXP i.MX RT595 (MIMXRT595SFFOC) on the EVK-MIMXRT595.
#   - Cortex-M33  @ 200 MHz  -> M33 track (this scaffold): TFLite-Micro / CMSIS-NN
#   - Cadence HiFi 4 DSP     -> HiFi4 track (later): eIQ NN kernels + Xtensa toolchain
#
# STATUS: scaffold only. Two decisions still open (see README "Open decisions"):
#   (1) ONNX->TFLite bridge tool, (2) M33 runtime (TFLM vs CMSIS-NN).

# ---------------------------------------------------------------------------
# NXP MCUXpresso SDK for EVK-MIMXRT595 — device headers, startup, BSP, linker,
# clock config, and (if selected at SDK-build time) eIQ / TensorFlow Lite Micro.
# NOT on disk yet. Get it from the MCUXpresso SDK Builder (mcuxpresso.nxp.com):
#   Board = EVK-MIMXRT595, Host OS = Linux, Toolchain = GCC ARM Embedded,
#   middleware = eIQ (TFLite-Micro + CMSIS-NN) [+ DSP/HiFi4 for the later track].
# Unzip it to $(SDK_DIR), then `make doctor`.
# ---------------------------------------------------------------------------
# This is the new west/CMake `mcuxsdk` layout (v26.06). Point SDK_DIR at the
# `mcuxsdk` subdir after copying it into WSL (see README "Flashing"/build notes).
SDK_DIR         ?= $(CURDIR)/sdk
SDK_DEVICE_HDR  ?= $(SDK_DIR)/devices/RT/RT500/MIMXRT595S/MIMXRT595S_cm33.h
# eIQ / TFLite-Micro middleware inside the SDK (verified present in 26.06).
TFLM_DIR        ?= $(SDK_DIR)/middleware/eiq/tensorflow-lite
CMSIS_NN_DIR    ?= $(SDK_DIR)/arch/arm/CMSIS/NN
# Cadence HiFi4 NN kernels (DSP track) + optimized DSP primitives (FFT for STFT).
CADENCE_NNLIB   ?= $(SDK_DIR)/middleware/cadence/nnlib
NATUREDSP_DIR   ?= $(SDK_DIR)/middleware/cadence/naturedsp
# Example to seed app/ from: audio TFLM pipeline (audio->DSP feats->Invoke->postproc).
SEED_EXAMPLE    ?= $(SDK_DIR)/examples/eiq_examples/tflm_kws
BOARD_DIR       ?= $(SDK_DIR)/examples/_boards/evkmimxrt595

# ---------------------------------------------------------------------------
# Arm GNU toolchain for the M33. 13.3.rel1 already present in ~/toolchains
# (same one the N6 target uses). M33: -mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard.
# ---------------------------------------------------------------------------
GCC_PATH        ?= $(HOME)/toolchains/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin

# ---------------------------------------------------------------------------
# Cadence Xtensa toolchain (HiFi4 track — LATER, not used by the M33 build).
# RI-2023.11 is installed, BUT its registered cores are AIR/HiFi5 (a different
# chip) — there is NO RT500 HiFi4 core config here. The RT595 HiFi4 core config
# ships with the NXP SDK's DSP component; it must be registered into XtensaTools
# and the Xtensa LICENSE must cover the RT500 core (the AIR license may not).
# ---------------------------------------------------------------------------
XTENSA_TOOLS    ?= $(HOME)/toolchains/tools/RI-2023.11-linux/XtensaTools
XTENSA_BUILDS   ?= $(HOME)/toolchains/builds/RI-2023.11-linux
XTENSA_CORE     ?=                      # RT500 HiFi4 core name — UNSET until SDK core is registered

# ---------------------------------------------------------------------------
# Flash / debug. NOTE: the on-board LPC-Link2 (CMSIS-DAP, HID) does NOT work
# reliably through WSL2 usbip — the interrupt transfers reset (ECONNRESET -104).
# Flash from the WINDOWS side (NXP LinkServer or pyocd-on-Windows) against the
# real USB probe. WSL builds the .axf/.bin; Windows flashes it.
#   pyocd (WSL, for pack/inspection only) : $(PYOCD)
#   Windows LinkServer (actual flash)     : set LINKSERVER_WIN for scripts/flash_windows.ps1
# ---------------------------------------------------------------------------
PYOCD           ?= $(HOME)/.local/bin/pyocd
PYOCD_TARGET    ?= mimxrt595sffoc
LINKSERVER_WIN  ?= C:/nxp/LinkServer_*/LinkServer.exe   # edit to your Windows install

# ---------------------------------------------------------------------------
# Serial console — the ST-LINK/LPC VCP. This DOES work over usbip (bulk CDC).
# ---------------------------------------------------------------------------
SERIAL_PORT     ?= /dev/ttyACM0
SERIAL_BAUD     ?= 115200

# ---------------------------------------------------------------------------
# Model under test. LiSenNet hybrid-streaming nc24 is the M33 candidate
# (11 states, GRU-on-time as 1x1 convs — all CMSIS-NN-native ops).
# MODEL is the .tflite once the ONNX->TFLite bridge is chosen (see README).
# ---------------------------------------------------------------------------
MODEL_CONFIG    ?= $(abspath $(CURDIR)/../../configs/lisennet_hybrid_nc24.json)
MODEL_ONNX      ?= $(CURDIR)/host_out/lisennet_hybrid_nc24_streaming_int8.onnx
MODEL           ?= $(CURDIR)/host_out/lisennet_hybrid_nc24_streaming_int8.tflite
GEN_OUT         ?= $(CURDIR)/host_out

# ---------------------------------------------------------------------------
# RT595 runtime geometry (from lisennet/model.py — deploy configs).
# n_fft=512, hop=256 (16 ms @ 16 kHz), 257 freq bins. STFT/iSTFT stay on the M33
# (CMSIS-DSP arm_rfft_fast_f32); the accelerator only predicts the magnitude mask.
# ---------------------------------------------------------------------------
N_FFT           ?= 512
HOP             ?= 256
N_FREQS         ?= 257
