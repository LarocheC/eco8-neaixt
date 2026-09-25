#!/usr/bin/env bash
# Second seed of stage C: the three 120-epoch fine-tunes from the SAME parents as
# run_square32_wave.sh, with seed 2345 instead of 1234.
#
# One seed cannot settle what stage C asks. The paired noise between arms of one
# wave is sigma ~0.0074-0.011 PESQ, so a one-sided 1.645-sigma test needs a gap of
# 0.018-0.038 before it means anything, and C3's "dense within 0.015" verdict is
# unreachable at n=1 for any outcome. Holding the parents fixed and changing only
# the seed (discriminator init, crops, data order) gives n=2 for C2 (mask cost)
# and a second C3 draw, without paying for new 200-epoch parents.
#
# Reading rule, fixed 2026-09-26 00:35 before either seed of stage C finished:
#   statistic = mean paired difference over the last 5 validations (epochs 70-110);
#   C2: >= ~0 => free; <= -0.03 => real cost.
#   C3: inside +/-1.645 sigma, or 0.015 < |d| < 0.03 => unresolved at this n.
#
# Usage: ./run_stageC_seed.sh [pid-to-wait-for]
set -u

WAIT_PID="${1:-}"
PY="${PY:-.venv/bin/python}"
NS_COMMON=(--training_epochs 120 --stdout_interval 45 --validation_interval 450
           --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)

if [ -n "$WAIT_PID" ]; then
  echo "$(date +%H:%M:%S)  waiting for pid $WAIT_PID (stage C, seed 1234)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "$(date +%H:%M:%S)  card free"
fi

pids=(); arms=()
for spec in "sq384_cb_c1_s2345:cp_sq384_dense/g_best" "sq384_ft_s2345:cp_sq384_dense/g_best" \
            "dense_h256f384_ft_s2345:cp_dense_h256f384/g_best"; do
  arm="${spec%%:*}"; init="${spec##*:}"
  echo "$(date +%H:%M:%S)  start  $arm  (init $init)"
  $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
    --init_from "$init" "${NS_COMMON[@]}" > "cp_${arm}.log" 2>&1 &
  pids+=($!); arms+=("$arm")
done
rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { rc=1; echo "$(date +%H:%M:%S)  FAILED ${arms[$i]}"; }
done

echo
echo "=== best validation PESQ per arm ==="
for arm in "${arms[@]}"; do
  best=$(grep 'PESQ Score' "cp_${arm}.log" 2>/dev/null | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)
  printf '%-26s %s\n' "$arm" "${best:-no validation completed}"
done
echo "$(date +%H:%M:%S)  done (rc=$rc)"
exit $rc
