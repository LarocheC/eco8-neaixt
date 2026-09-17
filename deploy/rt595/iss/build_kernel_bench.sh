#!/bin/bash
# Build the per-op HiFi4 kernel cost bench (iss/kernel_bench.cpp).
# Unlike build_iss.sh this links no SE driver -- each model is one op, driven
# straight through MicroInterpreter.
#
#   TFLM=<dir with libtflm_rt500.a + tflite-micro/>  B=<work dir holding
#   kernel_bench_models.h>  bash build_kernel_bench.sh
set -e
XTENSA_BASE="${XTENSA_BASE:-$HOME/toolchains}"
XTENSA_VER="${XTENSA_VER:-RI-2023.11-linux}"
XT="$XTENSA_BASE/tools/$XTENSA_VER/XtensaTools"
export XTENSA_SYSTEM=$XT/config
export XTENSA_CORE=nxp_rt500_RI23_11_newlib
export PATH="$XT/bin:$PATH"
: "${XTENSA_LICENSE_FILE:?set XTENSA_LICENSE_FILE to port@host or a path to your Cadence .lic}"
export XTENSA_LICENSE_FILE
export LM_LICENSE_FILE=$XTENSA_LICENSE_FILE

HERE="$(cd "$(dirname "$0")" && pwd)"
TFLM="${TFLM:-$HERE/../tflm}"
TF=$TFLM/tflite-micro
LIB=$TFLM/libtflm_rt500.a
B="${B:-$(pwd)}"
[ -f "$LIB" ] || { echo "missing $LIB -- run tflm/build_rt500.sh first" >&2; exit 1; }
cd "$B"

CORE=nxp_rt500_RI23_11_newlib
INC="-I$B -I$TF \
 -I$TF/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
 -I$TF/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
 -I$TF/tensorflow/lite/micro/tools/make/downloads/ruy"
DEF="-DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG"
CXXFLAGS="--xtensa-core=$CORE -mlsp=sim -stdlib=libc++ -O2 -std=c++17 $DEF $INC"

echo "[1/2] kernel_bench.cpp"
xt-clang++ $CXXFLAGS -c "$HERE/kernel_bench.cpp" -o kernel_bench.o
echo "[2/2] link"
xt-clang++ --xtensa-core=$CORE -mlsp=sim -stdlib=libc++ -O2 \
  -Wl,--orphan-handling=discard kernel_bench.o \
  -Wl,--start-group "$LIB" -Wl,--end-group -lm \
  -o kernel_bench.elf
echo "=== linked: $(ls -l kernel_bench.elf | awk '{print $5}') bytes ==="
