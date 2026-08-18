#!/usr/bin/env bash
# Overnight sparsity sweep: six masked fine-tunes from the same dense baseline.
#
# Every arm gets an identical schedule (lr 3e-4, 120 epochs, validation every
# 10) so the mask is the only variable. The dense control is not optional — a
# 60-epoch run of this recipe cost 0.083 PESQ on its own (freshly initialised
# discriminator; no do_* checkpoint exists to warm-start from), so absolute
# numbers are only interpretable against a control that ate the same penalty.
#
# Two waves of three: three concurrent arms use ~15 GB of the 24 GB card, and
# the box is already throughput-saturated at three (110 s/epoch each vs 37 s
# solo), so more concurrency would buy nothing but OOM risk.
#
# Usage: ./run_sparsity_overnight.sh <path-to-dense-g_best>
set -u

INIT="${1:?usage: $0 <path-to-dense-g_best>}"
PY="${PY:-.venv/bin/python}"
EPOCHS="${EPOCHS:-120}"

# 45 steps/epoch at batch 256 => validate every 10 epochs, checkpoint every 40.
COMMON=(--training_epochs "$EPOCHS" --stdout_interval 45
        --validation_interval 450 --checkpoint_interval 1800
        --best_checkpoint_start_epoch 0)

WAVE1=(ov_dense_control ov_block1x4_80 ov_unstruct_80)
WAVE2=(ov_1to4 ov_2to4 ov_4to8)

run_wave() {
  local pids=() arm
  for arm in "$@"; do
    echo "$(date +%H:%M:%S)  start  $arm"
    $PY -m nsnet2.train \
      --config "configs/${arm}.json" \
      --checkpoint_path "cp_${arm}" \
      --init_from "$INIT" \
      "${COMMON[@]}" > "cp_${arm}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      rc=1
      echo "$(date +%H:%M:%S)  FAILED $* (pid ${pids[$i]}) — see cp_*.log"
    fi
  done
  return $rc
}

echo "=== wave 1: ${WAVE1[*]} ==="
run_wave "${WAVE1[@]}"
echo "=== wave 2: ${WAVE2[*]} ==="
run_wave "${WAVE2[@]}"

echo
echo "=== best validation PESQ per arm ==="
for arm in "${WAVE1[@]}" "${WAVE2[@]}"; do
  best=$(grep 'PESQ Score' "cp_${arm}.log" 2>/dev/null \
         | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)
  printf '%-22s %s\n' "$arm" "${best:-no validation completed}"
done
echo "$(date +%H:%M:%S)  done"
