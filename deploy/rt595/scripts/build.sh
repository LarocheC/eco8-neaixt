#!/usr/bin/env bash
# Build the RT595 M33 SE firmware in WSL. Installs the example into the SDK tree
# (idempotent), then runs the mcuxsdk west build for evkmimxrt595@cm33 / armgcc.
# Output: <build_dir>/lisennet_se_cm33.{elf,bin}
set -euo pipefail
cd "$(dirname "$0")/.."                        # deploy/rt595
RT595=$(pwd)

SDK="${SDK_DIR:-$RT595/sdk}"
WS=$(dirname "$(readlink -f "$SDK")")          # the west workspace (parent of mcuxsdk)
BUILD_VENV="${BUILD_VENV:-$HOME/.venvs/rt595-build}"
ARMGCC="${ARMGCC_DIR:-$HOME/toolchains/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi}"
BUILD_DIR="${BUILD_DIR:-$RT595/build}"
CONFIG="${CONFIG:-debug}"                      # debug | release

"$RT595/scripts/install_example.sh"

export PATH="$BUILD_VENV/bin:$PATH"
export ARMGCC_DIR="$ARMGCC"
cd "$WS"
west build -b evkmimxrt595 --toolchain armgcc --config "$CONFIG" \
  --cmake-opt=-Dcore_id=cm33 \
  mcuxsdk/examples/eiq_examples/lisennet_se -d "$BUILD_DIR" "$@"

echo
echo "Artifacts:"
ls -lh "$BUILD_DIR"/lisennet_se_cm33.elf "$BUILD_DIR"/lisennet_se_cm33.bin 2>/dev/null
