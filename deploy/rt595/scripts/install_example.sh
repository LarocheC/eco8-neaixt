#!/usr/bin/env bash
# Materialise the mcuxsdk example for our SE app inside the (gitignored) SDK tree.
# The repo owns the sources (deploy/rt595/app) and the scaffolding
# (deploy/rt595/mcux_example); this script wires them into the SDK's example layout
# via symlinks, so `west build` can find them. Re-run after adding/removing app files.
set -euo pipefail
cd "$(dirname "$0")/.."                       # deploy/rt595
RT595=$(pwd)
APP="$RT595/app"
SCAF="$RT595/mcux_example"

SDK="${SDK_DIR:-$RT595/sdk}"                   # symlink to the mcuxsdk root
[ -d "$SDK/examples/eiq_examples" ] || { echo "SDK not found at $SDK (set SDK_DIR)"; exit 1; }

EX="$SDK/examples/eiq_examples/lisennet_se"
OV="$SDK/examples/_boards/evkmimxrt595/eiq_examples/lisennet_se"
mkdir -p "$EX" "$OV/cm33" "$OV/cpu"

# CMake glue + board boilerplate (copied — small, board-specific)
cp "$SCAF/example/CMakeLists.txt" "$SCAF/example/example.yml" "$EX/"
cp "$SCAF/overlay/reconfig.cmake" "$SCAF/overlay/prj.conf" "$OV/"
cp "$SCAF/overlay/cm33/"* "$OV/cm33/"

# App sources (symlinked — repo is the source of truth)
for f in main.cpp model_se_stream.cpp model_se_stream.h model_io_layout.h se_test_feats.h; do
  ln -sfn "$APP/$f" "$EX/$f"
done
ln -sfn "$APP/model_data.h"       "$OV/cpu/model_data.h"
ln -sfn "$APP/model_ops_micro.cpp" "$OV/cpu/model_ops_micro.cpp"

echo "Installed lisennet_se example into $SDK"
echo "  example: $EX"
echo "  overlay: $OV"
