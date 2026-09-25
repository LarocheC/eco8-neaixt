#!/usr/bin/env bash
# Square + multiple-of-32 hand-off: the ConvFSENet codebook arm, the square trunk,
# and the compute-matched dense control that can falsify the whole sparsity story.
#
# The collaborator's two new constraints are (1) square matrices and (2) every
# dimension a multiple of 32. Neither is met by the baseline (257/400/600/1200).
# The design here keeps n_freq = 257 in the graph and pads to 384 IN THE EXPORTER,
# so no trainable weight is spent on a structural zero and the STFT/streaming/ONNX
# contracts do not move. hidden = fc_hidden = 384 makes fc1/fc2 square outright and
# the four GRU matrices (1152, 384) row-slice into three square (384, 384) gate
# blocks -- measured bit-exact, mask-preserving, index array splits identically.
#
# Stages, in dependency order:
#   A  cf_c1             ConvFSENet 2:4@c1, 20 ep, solo. Controls already measured
#                        (cf_dense_control 2.832, cf_2to4 2.820), so this is the
#                        cheapest real test of the codebook: unlike NSNet2, the
#                        ConvFSENet mask sweep still has resolution.
#   B  sq384_dense       200 ep from scratch. Answers C1 (does the geometry cost
#      dense_h256f384    quality?) and is the pruning parent for stage C.
#                        dense_h256f384 is the compute-matched control: 1,196,672
#                        live weights against sq384_cb_c1's 1,131,072 (+5.6%).
#   GATE               if sq384_dense lands >0.03 below the 2.845 baseline, the
#                        1.0 FC ratio is the suspect -- stop rather than prune a
#                        broken parent. Re-run at hidden 384 / fc_hidden 768.
#   C  sq384_cb_c1       the deliverable, 120 ep warm-started from sq384_dense
#      sq384_ft          same recipe, no mask: isolates mask from schedule (C2)
#      dense_h256f384_ft same recipe on the dense control, so C3 compares
#                        draw for draw rather than across recipes
#
# Usage: ./run_square32_wave.sh [pid-to-wait-for]
set -u

WAIT_PID="${1:-}"
PY="${PY:-.venv/bin/python}"
BASELINE_PESQ=2.845
GATE_DROP=0.03

NS_COMMON=(--stdout_interval 45 --validation_interval 450
           --checkpoint_interval 1800 --best_checkpoint_start_epoch 0)
CF_COMMON=(--training_epochs 20 --stdout_interval 200 --validation_interval 2169
           --checkpoint_interval 7230 --best_checkpoint_start_epoch 0)

best_pesq() {  # best validation PESQ from an arm's log, empty if none yet
  grep 'PESQ Score' "cp_${1}.log" 2>/dev/null \
    | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1
}

report() {
  echo
  echo "=== best validation PESQ per arm ==="
  for arm in "$@"; do printf '%-22s %s\n' "$arm" "$(best_pesq "$arm" || true)"; done
}

if [ -n "$WAIT_PID" ]; then
  echo "$(date +%H:%M:%S)  waiting for pid $WAIT_PID (codebook wave) to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "$(date +%H:%M:%S)  card free"
fi

# --- A: ConvFSENet, solo (20 epochs, ~25 min) --------------------------------
echo "$(date +%H:%M:%S)  stage A: cf_c1"
$PY -m convfsenet.train --config configs/cf_c1.json --checkpoint_path cp_cf_c1 \
  --init_from cp_convfsenet_win/g_best "${CF_COMMON[@]}" > cp_cf_c1.log 2>&1 \
  || echo "$(date +%H:%M:%S)  FAILED cf_c1 — see cp_cf_c1.log"

# --- B: the square trunk and its compute-matched control, 2-concurrent -------
echo "$(date +%H:%M:%S)  stage B: sq384_dense + dense_h256f384 (200 ep each)"
pids=()
for arm in sq384_dense dense_h256f384; do
  $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
    --training_epochs 200 "${NS_COMMON[@]}" > "cp_${arm}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "$(date +%H:%M:%S)  FAILED (pid $p)"; done
report sq384_dense dense_h256f384

# --- GATE: do not prune a broken parent -------------------------------------
PARENT=$(best_pesq sq384_dense)
if [ -z "$PARENT" ] || [ ! -f cp_sq384_dense/g_best ]; then
  echo "$(date +%H:%M:%S)  STOP: sq384_dense produced no g_best / no validation"
  exit 1
fi
if ! awk -v p="$PARENT" -v b="$BASELINE_PESQ" -v d="$GATE_DROP" \
     'BEGIN{exit !(p >= b - d)}'; then
  echo "$(date +%H:%M:%S)  GATE FAILED: sq384_dense $PARENT is more than $GATE_DROP"
  echo "                     below the $BASELINE_PESQ baseline. The fc_hidden/hidden"
  echo "                     ratio of 1.0 is the suspect (the FC stack lost 47.8% of"
  echo "                     its weights, the GRU only 7.8%). Re-run at hidden 384 /"
  echo "                     fc_hidden 768 before pruning. Stage C not started."
  exit 2
fi
echo "$(date +%H:%M:%S)  gate passed: sq384_dense $PARENT"

# --- C: the deliverable, its recipe control, and the matched dense ft --------
echo "$(date +%H:%M:%S)  stage C: sq384_cb_c1 + sq384_ft + dense_h256f384_ft (120 ep)"
pids=()
for spec in "sq384_cb_c1:cp_sq384_dense/g_best" "sq384_ft:cp_sq384_dense/g_best" \
            "dense_h256f384_ft:cp_dense_h256f384/g_best"; do
  arm="${spec%%:*}"; init="${spec##*:}"
  $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
    --init_from "$init" --training_epochs 120 "${NS_COMMON[@]}" > "cp_${arm}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "$(date +%H:%M:%S)  FAILED (pid $p)"; done

report cf_c1 sq384_dense dense_h256f384 sq384_cb_c1 sq384_ft dense_h256f384_ft
echo "$(date +%H:%M:%S)  done"
