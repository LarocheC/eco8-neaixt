#!/usr/bin/env bash
# ConvFSENet pointwise-conv sparsity sweep: six masked fine-tunes from the same
# dense baseline (cp_convfsenet_win/g_best, FP32 PESQ 2.931 / int8 2.911).
#
# Masks cover the 20 pointwise (1x1) convs = 98.07% of parameters. The nine
# depthwise k=3 dconvs stay dense: they lower through im2col rather than as a
# GEMM, so an N:M group along C_in would not correspond to anything a sparse
# kernel sees.
#
# Identical schedule in every arm (lr 8e-5, 20 epochs, validation every 3) so
# the mask is the only variable. The dense control is required, not optional:
# on NSNet2 the equivalent shortened fine-tune cost 0.083 PESQ on its own.
#
# 723 steps/epoch at batch 16. Two waves of three.
#
# Usage: ./run_convfsenet_sparsity.sh [path-to-dense-g_best]
set -u

INIT="${1:-cp_convfsenet_win/g_best}"
PY="${PY:-.venv/bin/python}"
EPOCHS="${EPOCHS:-20}"

COMMON=(--training_epochs "$EPOCHS" --stdout_interval 200
        --validation_interval 2169 --checkpoint_interval 7230
        --best_checkpoint_start_epoch 0)

WAVE1=(cf_dense_control cf_block1x4_80 cf_unstruct_80)
WAVE2=(cf_2to4 cf_4to8 cf_1to4)

run_wave() {
  local pids=() arm
  for arm in "$@"; do
    echo "$(date +%H:%M:%S)  start  $arm"
    $PY -m convfsenet.train \
      --config "configs/${arm}.json" \
      --checkpoint_path "cp_${arm}" \
      --init_from "$INIT" \
      "${COMMON[@]}" > "cp_${arm}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then rc=1; echo "$(date +%H:%M:%S)  FAILED (pid ${pids[$i]}) — see cp_*.log"; fi
  done
  return $rc
}

echo "=== wave 1: ${WAVE1[*]} ==="; run_wave "${WAVE1[@]}"
echo "=== wave 2: ${WAVE2[*]} ==="; run_wave "${WAVE2[@]}"

echo
echo "=== best validation PESQ per arm ==="
for arm in "${WAVE1[@]}" "${WAVE2[@]}"; do
  best=$(grep 'PESQ Score' "cp_${arm}.log" 2>/dev/null | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)
  printf '%-22s %s\n' "$arm" "${best:-no validation completed}"
done
echo "$(date +%H:%M:%S)  done"
