#!/bin/bash
# Build the NSNet2 int8 HiFi4 ISS bench. Same recipe as the LiSenNet iss_build,
# driven off MODEL_FEATURE_TOTAL for the single-channel feature.
set -e
XT=/home/clement/toolchains/tools/RI-2023.11-linux/XtensaTools
export XTENSA_SYSTEM=$XT/config
export XTENSA_CORE=nxp_rt500_RI23_11_newlib
export PATH="$XT/bin:$PATH"
export XTENSA_LICENSE_FILE=/home/clement/eco8-neaixt/RT500SDK.lic
export LM_LICENSE_FILE=$XTENSA_LICENSE_FILE

SCR="${SCR:-$(cd "$(dirname "$0")" && pwd)/work}"   # override: SCR=/path/to/tflm-tree
TF=$SCR/tflm_rt500/tflite-micro
LIB=$SCR/tflm_rt500/libtflm_rt500.a
B="${B:-$(pwd)}"
cd "$B"

CORE=nxp_rt500_RI23_11_newlib
INC="-I$B -I$TF \
 -I$TF/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
 -I$TF/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
 -I$TF/tensorflow/lite/micro/tools/make/downloads/ruy"
DEF="-DTF_LITE_STATIC_MEMORY -DTF_LITE_MCU_DEBUG_LOG"
CXXFLAGS="--xtensa-core=$CORE -mlsp=sim -stdlib=libc++ -O2 -std=c++17 $DEF $INC"
CFLAGS="--xtensa-core=$CORE -mlsp=sim -O2 $DEF $INC"

echo "[1/4] model_ops_micro.cpp"; xt-clang++ $CXXFLAGS -c model_ops_micro.cpp -o model_ops_micro.o
echo "[2/4] model_se_stream.cpp";  xt-clang++ $CXXFLAGS -c model_se_stream.cpp -o model_se_stream.o
echo "[3/4] iss_bench.cpp";        xt-clang++ $CXXFLAGS -c iss_bench.cpp -o iss_bench.o
echo "[3b] fstubs.c";             xt-clang $CFLAGS -c fstubs.c -o fstubs.o
echo "[4/4] link"
xt-clang++ --xtensa-core=$CORE -mlsp=sim -stdlib=libc++ -O2 \
  -Wl,--orphan-handling=discard \
  iss_bench.o model_se_stream.o model_ops_micro.o fstubs.o \
  -Wl,--start-group "$LIB" -Wl,--end-group -lm \
  -o iss_bench.elf
echo "=== linked: $(ls -l iss_bench.elf | awk '{print $5}') bytes ==="