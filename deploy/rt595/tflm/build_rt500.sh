#!/bin/bash
set -e
XT=/home/clement/toolchains/tools/RI-2023.11-linux/XtensaTools
export XTENSA_BASE=/home/clement/toolchains
export XTENSA_SYSTEM=/home/clement/toolchains/builds/RI-2023.11-linux/nxp_rt500_RI23_11_newlib/config
export XTENSA_CORE=nxp_rt500_RI23_11_newlib
export XTENSA_TOOLS_VERSION=RI-2023.11-linux
export PATH="$XT/bin:$PATH"
export XTENSA_LICENSE_FILE=/home/clement/eco8-neaixt/RT500SDK.lic
export LM_LICENSE_FILE=/home/clement/eco8-neaixt/RT500SDK.lic
cd "$(dirname "$0")"
if [ ! -d tflite-micro ]; then
  git clone --quiet https://github.com/cad-audio/tflite-micro.git tflite-micro
fi
cd tflite-micro
git checkout --quiet 54331e9fe42af1dcc8221d693ae85890fb399944
git apply ../tflm.patch 2>/dev/null || echo "(patch already applied or partial)"
echo "=== building microlite for rt500 hifi4 ==="
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release_with_logs TARGET=xtensa TARGET_ARCH=hifi4 XTENSA_USE_LIBC=true OPTIMIZED_KERNEL_DIR=xtensa microlite -j4
cp gen/xtensa_hifi4_release_with_logs_xtensa_gcc/lib/libtensorflow-microlite.a ../libtflm_rt500.a
echo "=== DONE: $(ls -l ../libtflm_rt500.a | awk '{print $5}') bytes ==="
