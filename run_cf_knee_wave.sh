#!/usr/bin/env bash
# ConvFSENet: is 50% sparsity better than a smaller dense model, asked below the knee.
#
# Same design as run_knee_wave.sh (NSNet2), on the ConvFSENet family of cf_c1 and
# the deployed model: n_features 257, res = C, conv = 2C, 9 TCM blocks.
#
# The existing from-scratch width sweep (cp_cfs_dense_r*, 200 ep, lr 8e-4 cosine)
# puts the knee between r67 and r83 (last-5 of 19 validations):
#   r67 206k 2.816 | r83 303k 2.853 | r108 491k 2.857 | r146 864k 2.851 | r192 1.45M 2.860
# None of those widths is a multiple of 32. With every dim a multiple of 32 and
# live weights matched, one pair straddles the knee:
#   cfs64  dense   64/128   189,889 params          (below r67: on the slope)
#   cfs96  2:4@c1  96/192   204,785 live of 395,297 (between r83 and r108: plateau)
# The sparse side carries +7.8% more nonzero weights -- flag it when reading D.
# 96/128 (-12%) and 128/192 (+10%) both sit on the plateau: no power.
#
# Recipes are the existing ones with only the width changed:
#   parents    = configs/cfs_dense.json         200 ep from scratch, lr 8e-4 cosine
#   fine-tunes = configs/cf_dense_control.json  20 ep, lr 8e-5 cosine, from parent g_best
#                configs/cf_c1.json             the same + 2:4@c1 on the 20 pointwise convs
#
# Reading rule, fixed before any of these runs started. Statistic: each fine-tune's
# mean over its last 5 validations (validated every epoch, so epochs 15-19, past the
# prune-recovery transient), averaged over seeds.
#   U     = cf96_ft    - cf64_ft   capacity gap left after the fine-tune
#   D     = cf96_cb_c1 - cf64_ft   the question
#   D_adj = D - 0.103 * U          removes the sparse side's +7.8% extra live weights,
#                                  calibrated by the parents: ln(204785/189889) /
#                                  ln(395297/189889) = 0.103 of the capacity gap
#   U < 0.02                                      -> inconclusive (no power), not a null
#   U >= 0.02, D_adj >= U/2, D > 0 in both seeds  -> sparsity is a real improvement
#   U >= 0.02, D_adj <= 0.01                      -> it is not
#   anything else                                 -> unresolved
# Secondary, not part of the verdict: cf96_2to4 (plain 2:4 from the same parent)
# tells a codebook-specific loss apart from a sparsity loss. At 1.45 M the codebook
# cost ConvFSENet 0.030 vs dense, against 0.010 for plain 2:4.
#
# Validation cadence and checkpoint intervals differ from the earlier waves; both
# are training-neutral (validation runs under no_grad in eval mode with its own
# DataLoader generator and restores model.train(), train.py:264-274). Validating
# every epoch moves last-5 past the transient; the checkpoint intervals make the
# final weights land on disk (the old ones never reached the last step).
# GATE: the parents' last-5 means must differ by >= 0.01, else stop before fine-tuning.
# Expect ~14 h: parents ~12 h (215 s/epoch two at a time), then ~75 min per seed.
#
# Usage: ./run_cf_knee_wave.sh          (touch .cf_knee_skip_seed2 to skip seed 2345)
set -u

PY="${PY:-.venv/bin/python}"
PARENT_FLAGS=(--training_epochs 200 --stdout_interval 723 --validation_interval 7230
              --checkpoint_interval 20657 --best_checkpoint_start_epoch 10)
FT_FLAGS=(--training_epochs 20 --stdout_interval 200 --validation_interval 723
          --checkpoint_interval 14459 --best_checkpoint_start_epoch 0)
GATE_MIN=0.01

late() {
  $PY - "$1" <<'PY'
import re, sys, statistics as st
t = [float(x) for x in re.findall(r"PESQ Score: ([\d.]+)", open(f"cp_{sys.argv[1]}.log").read())]
print(f"{st.mean(t[-5:]):.4f}" if len(t) >= 5 else "")
PY
}

report() {
  local arm
  for arm in "$@"; do
    printf '  %-22s best %s  last-5 %s\n' "$arm" \
      "$(grep 'PESQ Score' "cp_${arm}.log" | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)" \
      "$(late "$arm")"
  done
}

run_ft_seed() {  # run_ft_seed SUFFIX
  local sfx="$1" pids=() arms=() spec arm init i
  for spec in "cf96_cb_c1${sfx}:cp_cfs_c96_dense/g_best" "cf96_ft${sfx}:cp_cfs_c96_dense/g_best" \
              "cf64_ft${sfx}:cp_cfs_c64_dense/g_best" "cf96_2to4${sfx}:cp_cfs_c96_dense/g_best"; do
    arm="${spec%%:*}"; init="${spec##*:}"
    echo "$(date +%H:%M:%S)  start  $arm  (init $init)"
    $PY -m convfsenet.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
      --init_from "$init" "${FT_FLAGS[@]}" > "cp_${arm}.log" 2>&1 &
    pids+=($!); arms+=("$arm")
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || echo "$(date +%H:%M:%S)  FAILED ${arms[$i]}"; done
  report "${arms[@]}"
}

while pgrep -f "python.* -m [c]onvfsenet\.train|python.* -m [n]snet2\.train" > /dev/null 2>&1; do
  echo "$(date +%H:%M:%S)  waiting for in-flight training to finish..."; sleep 120
done

echo "=== parents: cfs_c64_dense + cfs_c96_dense, 200 ep from scratch ==="
pids=()
for arm in cfs_c64_dense cfs_c96_dense; do
  echo "$(date +%H:%M:%S)  start  $arm"
  $PY -m convfsenet.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
    "${PARENT_FLAGS[@]}" > "cp_${arm}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "$(date +%H:%M:%S)  FAILED (pid $p)"; done
report cfs_c64_dense cfs_c96_dense

L64=$(late cfs_c64_dense); L96=$(late cfs_c96_dense)
if [ -z "$L64" ] || [ -z "$L96" ] || [ ! -f cp_cfs_c64_dense/g_best ] || [ ! -f cp_cfs_c96_dense/g_best ]; then
  echo "$(date +%H:%M:%S)  STOP: a parent produced no g_best / too few validations"; exit 1
fi
if ! awk -v a="$L96" -v b="$L64" -v g="$GATE_MIN" 'BEGIN{exit !(a - b >= g)}'; then
  echo "$(date +%H:%M:%S)  GATE FAILED: cfs96 last-5 $L96 vs cfs64 last-5 $L64 (gap < $GATE_MIN). Not fine-tuning."
  exit 2
fi
echo "$(date +%H:%M:%S)  gate passed: cfs96 $L96 vs cfs64 $L64"

echo "=== fine-tunes, seed 1234, 20 ep ==="; run_ft_seed ""
if [ -f .cf_knee_skip_seed2 ]; then
  echo "$(date +%H:%M:%S)  .cf_knee_skip_seed2 present: skipping seed 2345"
else
  echo "=== fine-tunes, seed 2345, 20 ep ==="; run_ft_seed "_s2345"
fi
echo "$(date +%H:%M:%S)  done"
