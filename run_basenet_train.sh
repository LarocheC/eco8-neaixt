#!/usr/bin/env bash
# Robust launcher for BASENet training that survives SSH disconnects.
#
# Runs the trainer inside a DETACHED tmux session: immune to SIGHUP (so an SSH
# drop won't kill it) and reattachable so you can watch it live. The trainer
# auto-resumes from the latest rolling checkpoint in CKPT_DIR, so re-running this
# after a crash or reboot continues where it left off rather than restarting.
#
# Usage:
#   ./run_basenet_train.sh [CKPT_DIR] [CONFIG] [EPOCHS]
#     CKPT_DIR  checkpoint + log dir            (default: cp_basenet_causal)
#     CONFIG    training config json            (default: configs/basenet_causal.json)
#     EPOCHS    number of epochs                (default: 100)
#
# Monitor:
#   tmux attach -t <session>            # printed below; detach with Ctrl-b then d
#   tail -f  <CKPT_DIR>/train.log
#   tensorboard --logdir <CKPT_DIR>/logs
#
# Resume after a crash/reboot: just run this script again with the same CKPT_DIR.
set -euo pipefail

CKPT_DIR="${1:-cp_basenet_causal}"
CONFIG="${2:-configs/basenet_causal.json}"
EPOCHS="${3:-100}"
ACCUM="${4:-1}"          # gradient-accumulation micro-batches (effective batch = batch_size*ACCUM)
SESSION="bn_$(basename "$CKPT_DIR")"

cd "$(dirname "$0")"
mkdir -p "$CKPT_DIR"

# Guard against double-launch: two trainers on the same dir corrupt checkpoints
# and OOM the GPU.
if pgrep -fa "basenet.train" | grep -q -- "--checkpoint_path[ =]*$CKPT_DIR"; then
  echo "ERROR: a basenet.train process for '$CKPT_DIR' is already running. Refusing to double-launch." >&2
  echo "       Inspect it with: pgrep -fa basenet.train" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   .venv/bin/python -u -m basenet.train \
     --config '$CONFIG' \
     --checkpoint_path '$CKPT_DIR' \
     --training_epochs '$EPOCHS' \
     --accum_steps '$ACCUM' \
     --stdout_interval 20 \
     --validation_interval 1000 \
     --checkpoint_interval 2000 \
     --best_checkpoint_start_epoch 1 \
   2>&1 | tee '$CKPT_DIR/train.log'"

echo "Launched BASENet training in tmux session '$SESSION'  ->  $CKPT_DIR"
echo "  attach:      tmux attach -t $SESSION        (detach: Ctrl-b then d)"
echo "  log:         tail -f $CKPT_DIR/train.log"
echo "  tensorboard: tensorboard --logdir $CKPT_DIR/logs"
echo "  best model:  $CKPT_DIR/g_best"
echo "  resume:      re-run this script (auto-resumes from the latest checkpoint)"
