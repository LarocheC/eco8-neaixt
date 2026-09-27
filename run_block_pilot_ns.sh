#!/usr/bin/env bash
# Block-design pilot, NSNet2 sq192 (hidden = fc_hidden = 192), 15 epochs from scratch.
#
#   A0  sq192_dense unchanged (control; byte-identical copy of configs/sq192_dense.json)
#   A1  + input_norm (frozen per-bin training mean, configs/stats/...) + 200-step LR warmup
#   A2  + block_norm "batch" (fc -> BatchNorm -> ReLU on fc_in / fc1 / fc2)
#   A3  input_norm + block_norm + 200-step LR warmup
#
# All four arms run concurrently on the one GPU. Checkpoints: cp_pilot_ns_A*/,
# logs: cp_pilot_ns_A*.log (both in the directory this script lives in).
# A block-design checkpoint must be folded (python -m nsnet2.fold) before any
# streaming / ONNX / quant tool touches it.
#
# Usage: ./run_block_pilot_ns.sh
set -u
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
if [ ! -x "$PY" ]; then   # git worktree: use the main checkout's venv
  PY="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/.venv/bin/python"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

NS_COMMON=(--training_epochs 15 --stdout_interval 45 --validation_interval 450
           --checkpoint_interval 100 --best_checkpoint_start_epoch 0)
ARMS=(pilot_ns_A0 pilot_ns_A1 pilot_ns_A2 pilot_ns_A3)

# nsnet2.train RESUMES from any g_*/do_* already in the checkpoint dir; the
# pilot must start from scratch, so refuse to run over a previous attempt.
for arm in "${ARMS[@]}"; do
  if compgen -G "cp_${arm}/g_*" > /dev/null || compgen -G "cp_${arm}/do_*" > /dev/null; then
    echo "cp_${arm} already holds checkpoints; the trainer would RESUME, not start from scratch. Move it away first."
    exit 1
  fi
done

pids=()
for arm in "${ARMS[@]}"; do
  echo "$(date +%H:%M:%S)  start  $arm"
  $PY -m nsnet2.train --config "configs/${arm}.json" --checkpoint_path "cp_${arm}" \
    "${NS_COMMON[@]}" > "cp_${arm}.log" 2>&1 &
  pids+=($!)
done

for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || echo "$(date +%H:%M:%S)  FAILED ${ARMS[$i]}"
done

for arm in "${ARMS[@]}"; do
  printf '  %-14s best %s  last %s\n' "$arm" \
    "$(grep 'PESQ Score' "cp_${arm}.log" | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)" \
    "$(grep 'PESQ Score' "cp_${arm}.log" | sed 's/.*PESQ Score: //;s/,.*//' | tail -1)"
done
echo "$(date +%H:%M:%S)  done"
