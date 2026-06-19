#!/usr/bin/env bash
# Compile an ONNX (QDQ int8) model to Neural-ART C code + int8 weights for the STM32N6.
#
# We `cd` into the Neural-ART profile dir so `default@neural_art.json` resolves its
# companion stm32n6.mpool by relative path (passing a bare json name from elsewhere
# fails with "neural_art.json does not exist"). Output still lands in $GEN_OUT (abs path).
#
# --fix-parametric-shapes "{'B':1}" pins the dynamic batch dim to 1 (per-frame streaming).
# We intentionally do NOT pass --input-data-type int8: ConvFSENet's noisy_mag input reaches
# Slice/Gather before any QuantizeLinear, so the compiler cannot fold it to int8 and errors
# ("input of this model is not quantized"). The state inputs/outputs become int8 anyway.
set -euo pipefail

: "${STEDGEAI:?set STEDGEAI}" "${MODEL:?set MODEL}" "${NPU_PROFILE_DIR:?}" "${NPU_PROFILE:?}" "${GEN_OUT:?}"

MODEL_ABS="$(readlink -f "$MODEL")"
GEN_OUT_ABS="$(mkdir -p "$GEN_OUT" && readlink -f "$GEN_OUT")"

echo "[generate] model:   $MODEL_ABS"
echo "[generate] profile: $NPU_PROFILE  (cwd=$NPU_PROFILE_DIR)"
echo "[generate] output:  $GEN_OUT_ABS"

cd "$NPU_PROFILE_DIR"
"$STEDGEAI" generate \
  -m "$MODEL_ABS" \
  --target stm32n6 \
  --st-neural-art "$NPU_PROFILE" \
  --fix-parametric-shapes "{'B':1}" \
  -o "$GEN_OUT_ABS"

echo "[generate] done. Key artifacts in $GEN_OUT_ABS:"
echo "           network.c/.h, stai_network.c/.h, network_atonbuf.xSPI2.raw (weights),"
echo "           *_OE_*.onnx + *_Q.json (compiled graph), network_generate_report.txt"
