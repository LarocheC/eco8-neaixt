#!/bin/bash
set -e
XT=/home/clement/toolchains/tools/RI-2023.11-linux/XtensaTools
export XTENSA_BASE=/home/clement/toolchains
export XTENSA_SYSTEM=/home/clement/toolchains/builds/RI-2023.11-linux/nxp_rt500_RI23_11_newlib/config
export XTENSA_CORE=nxp_rt500_RI23_11_newlib XTENSA_TOOLS_VERSION=RI-2023.11-linux
# numpy-having python FIRST on PATH (for TFLM's generate_cc_arrays.py), then xtensa tools
export PATH="/home/clement/.venvs/rt595-export/bin:$XT/bin:$PATH"
export XTENSA_LICENSE_FILE=/home/clement/eco8-neaixt/RT500SDK.lic LM_LICENSE_FILE=/home/clement/eco8-neaixt/RT500SDK.lic
cd "$(dirname "$0")/tflite-micro"
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release_with_logs TARGET=xtensa TARGET_ARCH=hifi4 XTENSA_USE_LIBC=true OPTIMIZED_KERNEL_DIR=xtensa microlite -j4
cp gen/xtensa_hifi4_release_with_logs_xtensa_gcc/lib/libtensorflow-microlite.a ../libtflm_rt500.a
echo "=== DONE: $(ls -l ../libtflm_rt500.a | awk '{print $5}') bytes ==="
