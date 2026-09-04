#!/bin/bash
set -e
XTENSA_BASE="${XTENSA_BASE:-$HOME/toolchains}"
XTENSA_VER="${XTENSA_VER:-RI-2023.11-linux}"
XT="$XTENSA_BASE/tools/$XTENSA_VER/XtensaTools"
export XTENSA_BASE
export XTENSA_CORE="${XTENSA_CORE:-nxp_rt500_RI23_11_newlib}"
export XTENSA_SYSTEM="${XTENSA_SYSTEM:-$XTENSA_BASE/builds/$XTENSA_VER/$XTENSA_CORE/config}"
export XTENSA_CORE=nxp_rt500_RI23_11_newlib XTENSA_TOOLS_VERSION=RI-2023.11-linux
# numpy-having python FIRST on PATH (for TFLM's generate_cc_arrays.py), then xtensa tools
export PATH="${RT595_EXPORT_VENV:-$HOME/.venvs/rt595-export}/bin:$XT/bin:$PATH"
: "${XTENSA_LICENSE_FILE:?set XTENSA_LICENSE_FILE to port@host or a path to your Cadence .lic}"
export XTENSA_LICENSE_FILE LM_LICENSE_FILE="${LM_LICENSE_FILE:-$XTENSA_LICENSE_FILE}"
cd "$(dirname "$0")/tflite-micro"
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release_with_logs TARGET=xtensa TARGET_ARCH=hifi4 XTENSA_USE_LIBC=true OPTIMIZED_KERNEL_DIR=xtensa microlite -j4
cp gen/xtensa_hifi4_release_with_logs_xtensa_gcc/lib/libtensorflow-microlite.a ../libtflm_rt500.a
echo "=== DONE: $(ls -l ../libtflm_rt500.a | awk '{print $5}') bytes ==="
