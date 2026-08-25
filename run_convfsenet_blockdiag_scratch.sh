#!/usr/bin/env bash
# Block-diagonal ConvFSENet: FROM SCRATCH, full recipe (lr 8e-4 cosine, 200
# epochs) — no --init_from anywhere.
#
# Why from scratch and not a fine-tune like the N:M arms: blockdiag is
# value-agnostic. Which entries survive is fixed by position, not magnitude, so
# there is nothing for pruning-from-trained to preserve — it discards sqrt(1/N)
# of the weight energy no matter what the checkpoint holds. A factorization is
# not trying to approximate the dense matrix; it is a different parameterisation
# that has to be trained. This is also the protocol the published monarch NSNet2
# results used.
#
# cfs_dense is a from-scratch dense control on the identical recipe. It is NOT
# redundant with the published 2.931: it absorbs any environmental drift (torch
# / cudnn / worker-count changes since that run) that would otherwise be
# misattributed to the mask. On NSNet2 the analogous confound was 0.083 PESQ.
#
# 723 steps/epoch at batch 16 => 144,600 steps. Validation every 10 epochs.
set -u

PY="${PY:-.venv/bin/python}"
EPOCHS="${EPOCHS:-200}"
ARMS=(cfs_dense cfs_blockdiag2 cfs_blockdiag4 cfs_blockdiag8)

# Wait for any in-flight convfsenet training to finish before adding load.
# The bracket trick keeps this from matching unrelated shells whose command
# line merely mentions the module name -- including a status check typed while
# the runner is polling, which would otherwise stall it indefinitely.
while pgrep -f "python.* -m [c]onvfsenet\.train" > /dev/null 2>&1; do
    echo "$(date +%H:%M:%S)  waiting for in-flight convfsenet training to finish..."
    sleep 120
done
echo "$(date +%H:%M:%S)  box is clear, starting from-scratch sweep"

pids=()
for arm in "${ARMS[@]}"; do
    echo "$(date +%H:%M:%S)  start  $arm"
    $PY -m convfsenet.train \
        --config "configs/${arm}.json" \
        --checkpoint_path "cp_${arm}" \
        --training_epochs "$EPOCHS" --stdout_interval 723 \
        --validation_interval 7230 --checkpoint_interval 28920 \
        --best_checkpoint_start_epoch 10 > "cp_${arm}.log" 2>&1 &
    pids+=($!)
done

for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || echo "$(date +%H:%M:%S)  FAILED ${ARMS[$i]} — see cp_${ARMS[$i]}.log"
done

echo
echo "=== best validation PESQ per arm (published dense baseline: 2.931) ==="
for arm in "${ARMS[@]}"; do
    best=$(grep 'PESQ Score' "cp_${arm}.log" 2>/dev/null | sed 's/.*PESQ Score: //;s/,.*//' | sort -g | tail -1)
    printf '%-20s %s\n' "$arm" "${best:-none}"
done
echo "$(date +%H:%M:%S)  done"
