#!/usr/bin/env bash
# Is 50% sparsity better than a smaller dense model? Asked where the answer can show.
#
# Stage C compared sq384_cb_c1 (1.14M live) with a 1.20M dense model and could not
# tell them apart (+0.010 / -0.002 over two seeds). That comparison had no power:
# the dense-size curve (claroche1/sparse-nsnet2-checkpoints, late-training means)
# is flat from ~0.55M upward, so neither side was short of capacity.
#
#   params   0.08-0.12M  0.22M  0.44M  0.56M  0.88M(n=3)  2.27M
#   late     2.72-2.73   2.760  2.775  2.803  2.763       2.812
#
# So put the dense model BELOW the knee and the sparse model's parent AT it. With
# every dim a multiple of 32 and live weights matched, one pair fits:
#   sq128  dense               297,345 params
#   sq192  2:4@c1              310,625 live of 617,921   (+4.5% vs sq128)
# 192/288 also matches (+6.7%) but both sit on the plateau -- stage C again.
#
# Same family (hidden = fc_hidden = H, n_freq 257) and the same recipe as the
# sq384 study: 200 ep dense from scratch at lr 3e-3, then 120 ep fine-tunes at
# lr 3e-4 with a fresh discriminator. Nothing but the width changes.
#
# Reading rule, fixed 2026-09-26 before any of these runs started. Statistic: each
# fine-tune's mean over its last 5 validations (epochs 70-110), averaged over seeds.
#   U = sq192_ft    - sq128_ft   the capacity gap left after the recipe: the most
#                                 sparsity could possibly buy back.
#   D = sq192_cb_c1 - sq128_ft   the question.
#   U < 0.02                       no power at this size -> inconclusive, not a null.
#   U >= 0.02, D >= U/2, D > 0 in both seeds   -> sparsity is a real improvement.
#   U >= 0.02, D <= 0.01           -> it is not; a smaller dense model does as well.
#   anything else                  -> unresolved.
#
# GATE: if the two dense parents' late means (last 5 validations, epochs 150-190)
# differ by < 0.01, the knee is not where the 1.5-ratio sweep put it for this
# family; stop rather than spend 7 h fine-tuning two equivalent models.
#
# Usage: ./run_knee_wave.sh [pid-to-wait-for]
#   touch .knee_skip_seed2  -> skip the seed-2345 fine-tunes
set -u

WAIT_PID="${1:-}"
PY="${PY:-.venv/bin/python}"
NS_COMMON=(--stdout_interval 45 --validation_interval 450
           --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)
GATE_MIN=0.01

late() {  # mean of the last 5 validation PESQ values in an arm's log
  $PY - "$1" <<'PY'
import re, sys, statistics as st
t = [float(x) for x in re.findall(r"PESQ Score: ([\d.]+)", open(f"cp_{sys.argv[1]}.log").read())]
print(f"{st.mean(t[-5:]):.4f}" if len(t) >= 5 else "")
PY
}

run_group() {  # run_group EPOCHS "arm:init" ...   (init '-' = from scratch)
  local epochs="$1"; shift
  local pids=() arms=() spec arm init
  for spec in "$@"; do
    arm="${spec%%:*}"; init="${spec##*:}"
    echo "$(date +%H:%M:%S)  start  $arm  (${init/-/scratch}, $epochs ep)"
    if [ "$init" = "-" ]; then
      $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
        --training_epochs "$epochs" "${NS_COMMON[@]}" > "cp_${arm}.log" 2>&1 &
    else
      $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
        --init_from "$init" --training_epochs "$epochs" "${NS_COMMON[@]}" > "cp_${arm}.log" 2>&1 &
    fi
    pids+=($!); arms+=("$arm")
  done
  local i
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || echo "$(date +%H:%M:%S)  FAILED ${arms[$i]}"
  done
  for arm in "${arms[@]}"; do
    printf '  %-22s best %s  late %s\n' "$arm" \
      "$(grep 'PESQ Score' "cp_${arm}.log" | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)" \
      "$(late "$arm")"
  done
}

if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

echo "=== parents: sq128_dense + sq192_dense, 200 ep from scratch ==="
run_group 200 "sq128_dense:-" "sq192_dense:-"

L128=$(late sq128_dense); L192=$(late sq192_dense)
if [ -z "$L128" ] || [ -z "$L192" ] || [ ! -f cp_sq128_dense/g_best ] || [ ! -f cp_sq192_dense/g_best ]; then
  echo "$(date +%H:%M:%S)  STOP: a parent produced no g_best / too few validations"; exit 1
fi
if ! awk -v a="$L192" -v b="$L128" -v g="$GATE_MIN" 'BEGIN{exit !(a - b >= g)}'; then
  echo "$(date +%H:%M:%S)  GATE FAILED: sq192_dense late $L192 vs sq128_dense late $L128"
  echo "                     (gap < $GATE_MIN). The two parents are equivalent, so the"
  echo "                     sparse-vs-dense test would have no power here. Not fine-tuning."
  exit 2
fi
echo "$(date +%H:%M:%S)  gate passed: sq192 $L192 vs sq128 $L128"

echo "=== fine-tunes, seed 1234, 120 ep ==="
run_group 120 "sq192_cb_c1:cp_sq192_dense/g_best" "sq192_ft:cp_sq192_dense/g_best" \
              "sq128_ft:cp_sq128_dense/g_best"

if [ -f .knee_skip_seed2 ]; then
  echo "$(date +%H:%M:%S)  .knee_skip_seed2 present: skipping seed 2345"
else
  echo "=== fine-tunes, seed 2345, 120 ep ==="
  run_group 120 "sq192_cb_c1_s2345:cp_sq192_dense/g_best" "sq192_ft_s2345:cp_sq192_dense/g_best" \
                "sq128_ft_s2345:cp_sq128_dense/g_best"
fi
echo "$(date +%H:%M:%S)  done"
