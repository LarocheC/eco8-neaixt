#!/usr/bin/env bash
# Codebook-constrained 2:4: one masked fine-tune per allowed pattern table.
#
# The Row-Fusion kernel indexes a fixed table of four 2:4 patterns rather than
# choosing freely among all six, so each group's choice is 2 bits. The three
# named codebooks are the three ways of pairing {0,1,2,3}:
#
#   c1  (0,1)(2,3)  1010 0101 1001 0110   -- identical to plain "1:2"
#   c2  (0,2)(1,3)  1100 0011 1001 0110
#   c3  (0,3)(1,2)  1010 0101 1100 0011
#
# Schedule is bit-identical to run_sparsity_overnight.sh, so these arms compare
# directly against the published cp_ov_dense_control (2.777) and cp_ov_2to4
# (2.779) without retraining either control.
#
# Three concurrent arms use ~15 GB of the 24 GB card and the box saturates at
# three, so all three fit in one wave (~4 h at 120 epochs).
#
# Usage: ./run_codebook_wave.sh <path-to-dense-g_best>
set -u

INIT="${1:?usage: $0 <path-to-dense-g_best>}"
PY="${PY:-.venv/bin/python}"
EPOCHS="${EPOCHS:-120}"
ARMS=(${ARMS:-cb_c1 cb_c2 cb_c3})

# 45 steps/epoch at batch 256 => validate every 10 epochs, checkpoint every 40.
COMMON=(--training_epochs "$EPOCHS" --stdout_interval 45
        --validation_interval 450 --checkpoint_interval 1800
        --best_checkpoint_start_epoch 0)

pids=()
for arm in "${ARMS[@]}"; do
  echo "$(date +%H:%M:%S)  start  $arm"
  $PY -m nsnet2.train \
    --config "configs/${arm}.json" \
    --checkpoint_path "cp_${arm}" \
    --init_from "$INIT" \
    "${COMMON[@]}" > "cp_${arm}.log" 2>&1 &
  pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    rc=1
    echo "$(date +%H:%M:%S)  FAILED ${ARMS[$i]} (pid ${pids[$i]}) — see cp_${ARMS[$i]}.log"
  fi
done

echo
echo "=== best validation PESQ per arm ==="
for arm in "${ARMS[@]}"; do
  best=$(grep 'PESQ Score' "cp_${arm}.log" 2>/dev/null \
         | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)
  printf '%-22s %s\n' "$arm" "${best:-no validation completed}"
done
echo "$(date +%H:%M:%S)  done (rc=$rc)"
exit $rc
