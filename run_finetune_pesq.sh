#!/usr/bin/env bash
# PESQ-loss fine-tuning study (docs/studies/pesq-finetune.md).
#
# From one base checkpoint, trains a control arm (no PESQ term) and a PESQ arm with
# the identical recipe, then scores base / control / pesq on the full VBD test split
# with paired deltas. Logs go to <arm>/train.log, results to results/pesq_finetune/.
#
#   ./run_finetune_pesq.sh lisennet hf:conv-hardened          # the STM32N6 deploy model
#   ./run_finetune_pesq.sh lisennet hf:conv-hardened-deep 0.5 # another weight
#   ./run_finetune_pesq.sh convfsenet hf:                     # ConvFSENet (repo root on the Hub)
#   ./run_finetune_pesq.sh lisennet cp_lisennet_conv/g_best   # a local run (config.json beside it)
#
# Override the recipe with EPOCHS (default 10), LR (1e-4), METRICS (all), ROOT (cp_ft).
# Needs: uv sync --group pesq-loss   (on CUDA the loss runs on triton-pesq's Triton backend)
set -euo pipefail

FAMILY=${1:?usage: $0 <lisennet|convfsenet> <base: path/to/g_best | hf:subfolder> [pesq_weight]}
BASE=${2:?base checkpoint}
WEIGHT=${3:-0.2}
EPOCHS=${EPOCHS:-10}
LR=${LR:-1e-4}
METRICS=${METRICS:-all}
ROOT=${ROOT:-cp_ft}

if [[ $BASE == hf:* ]]; then
    TAG="${FAMILY}__${BASE#hf:}"
else
    TAG="${FAMILY}__$(basename "$(dirname "$BASE")")"
fi
TAG=${TAG%__}
CONTROL="$ROOT/${TAG}__control_lr${LR}_e${EPOCHS}"
PESQ="$ROOT/${TAG}__pesq${WEIGHT}_lr${LR}_e${EPOCHS}"
mkdir -p "$CONTROL" "$PESQ" results/pesq_finetune

for arm in "$CONTROL:0" "$PESQ:$WEIGHT"; do
    dir=${arm%%:*}
    weight=${arm##*:}
    echo "=== $dir (pesq weight $weight)"
    uv run python finetune_pesq.py train --model "$FAMILY" --base "$BASE" --out "$dir" \
        --pesq_weight "$weight" --lr "$LR" --epochs "$EPOCHS" 2>&1 | tee -a "$dir/train.log"
done

OUT="results/pesq_finetune/${TAG}__pesq${WEIGHT}_lr${LR}_e${EPOCHS}"
uv run python finetune_pesq.py compare --model "$FAMILY" --metrics "$METRICS" \
    base="$BASE" control="$CONTROL/g_best" pesq="$PESQ/g_best" \
    --json "$OUT.json" --md "$OUT.md"
echo "Results: $OUT.md"
