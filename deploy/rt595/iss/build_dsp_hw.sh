#!/usr/bin/env bash
# Build the HiFi4 bench for REAL SILICON: min-rt LSP + memory-reported results,
# then split it into the three blobs the DSP loader copies into place.
#
# Differs from build_iss.sh in two ways that both matter on hardware:
#   * -mlsp=<...>/min-rt      the deployable LSP. gdbio needs a debugger to service
#                             its I/O; min-rt does not, but silently DROPS stdio --
#                             hence dsp_bench_mem.cpp reporting through memory.
#   * emits dsp_{reset,text,data}.bin via the xt-objcopy section lists taken from
#     NXP's own evkmimxrt595_fusionf1 TFLM example Makefile.include.
#
# Inputs expected in $B (the build dir): model_data.h, model_io_layout.h,
# se_test_feats.h, model_ops_micro.cpp.
set -euo pipefail

XT="${XT:-$HOME/toolchains/tools/RI-2023.11-linux/XtensaTools}"
export XTENSA_SYSTEM="$XT/config"
export XTENSA_CORE=nxp_rt500_RI23_11_newlib
export PATH="$XT/bin:$PATH"
: "${XTENSA_LICENSE_FILE:?set XTENSA_LICENSE_FILE to port@host or a path to your Cadence .lic}"
export XTENSA_LICENSE_FILE
export LM_LICENSE_FILE="$XTENSA_LICENSE_FILE"

R="$(cd "$(dirname "$0")/.." && pwd)"                 # deploy/rt595
B="${B:-$(pwd)}"                                      # build dir
SCR="${SCR:?set SCR to the dir holding tflm_rt500/}"
TF="$SCR/tflm_rt500/tflite-micro"
LIB="$SCR/tflm_rt500/libtflm_rt500.a"
LSP="${LSP:-$R/sdk/examples/eiq_examples/common/tflm/dsp/evkmimxrt595_fusionf1/min-rt}"
CORE=nxp_rt500_RI23_11_newlib

for p in "$TF" "$LIB" "$LSP"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 1; }
done

cd "$B"
# Always refresh the sources this script owns (NOT -n: a stale copy in the build dir
# silently rebuilds the previous revision).
cp "$R/app/model_se_stream.cpp" "$R/app/model_se_stream.h" .
cp "$R/iss/dsp_bench_mem.cpp" "$R/iss/fstubs.c" "$R/iss/fsl_common.h" \
   "$R/iss/fsl_debug_console.h" .

INC=(-I. -I"$TF"
     -I"$TF/tensorflow/lite/micro/tools/make/downloads/gemmlowp"
     -I"$TF/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include"
     -I"$TF/tensorflow/lite/micro/tools/make/downloads/ruy")
ARENA="${ARENA:-32768}"        # measured use is 17,096 B; the driver default of 512 KB
                               # overflows the DSP's 1.30 MB dram0_0_seg
FOREVER="${FOREVER:-0}"        # 1 = sustained-inference build for power measurement:
                               # re-runs the 16 frames endlessly (see dsp_bench_mem.cpp)
CXXFLAGS=(--xtensa-core=$CORE -mlsp="$LSP" -stdlib=libc++ -O2 -std=c++17
          -DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG
          -DSE_TENSOR_ARENA_SIZE="$ARENA" -DSE_BENCH_FOREVER="$FOREVER" "${INC[@]}")
CFLAGS=(--xtensa-core=$CORE -mlsp="$LSP" -O2 -DTF_LITE_STATIC_MEMORY "${INC[@]}")

echo "[1/5] dsp_bench_mem.cpp"; xt-clang++ "${CXXFLAGS[@]}" -c dsp_bench_mem.cpp  -o dsp_bench_mem.o
echo "[2/5] model_se_stream.cpp"; xt-clang++ "${CXXFLAGS[@]}" -c model_se_stream.cpp -o model_se_stream.o
echo "[3/5] model_ops_micro.cpp"; xt-clang++ "${CXXFLAGS[@]}" -c model_ops_micro.cpp -o model_ops_micro.o
echo "[4/5] fstubs.c";           xt-clang   "${CFLAGS[@]}"   -c fstubs.c          -o fstubs.o
echo "[5/5] link (min-rt)"
xt-clang++ --xtensa-core=$CORE -mlsp="$LSP" -stdlib=libc++ -O2 \
  -Wl,--orphan-handling=discard \
  dsp_bench_mem.o model_se_stream.o model_ops_micro.o fstubs.o \
  -Wl,--start-group "$LIB" -Wl,--end-group -lm -o dsp_bench.elf

echo "=== dsp_bench.elf: $(stat -c%s dsp_bench.elf) bytes ==="
xt-size dsp_bench.elf || true
echo "=== g_bench_result (read this address over SWD) ==="
xt-nm dsp_bench.elf | grep -i g_bench_result || true

# --- split into the loader's three blobs (section lists from NXP's Makefile.include) ---
xt-objcopy -O binary dsp_bench.elf dsp_reset.bin --only-section=.ResetVector.text
xt-objcopy -O binary dsp_bench.elf dsp_text.bin \
  --only-section=.WindowVectors.text --only-section=.Level2InterruptVector.text \
  --only-section=.Level3InterruptVector.literal --only-section=.Level3InterruptVector.text \
  --only-section=.DebugExceptionVector.literal --only-section=.DebugExceptionVector.text \
  --only-section=.NMIExceptionVector.literal --only-section=.NMIExceptionVector.text \
  --only-section=.KernelExceptionVector.text --only-section=.UserExceptionVector.literal \
  --only-section=.UserExceptionVector.text --only-section=.DoubleExceptionVector.text \
  --only-section=.text
xt-objcopy -O binary dsp_bench.elf dsp_data.bin \
  --only-section=.rtos.rodata --only-section=.rodata --only-section=.clib.data \
  --only-section=.rtos.percpu.data --only-section=.data

echo "=== blobs (dest: reset 0x00180000, text 0x00180400, data 0x00040000) ==="
ls -l dsp_reset.bin dsp_text.bin dsp_data.bin | awk '{printf "  %-16s %s bytes\n", $9, $5}'
