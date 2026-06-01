#!/usr/bin/env bash
# w4/a8 QAT sweep over the 7 needs-QAT NSNet2 checkpoints hosted on HuggingFace.
#
# For each run: resolves the g_best checkpoint from the HF repo (downloading it
# if not already cached), then QAT-fine-tunes it via nsnet2.qat_train. Output
# goes to cp_qat<EPOCHS>_<name>/ so different epoch budgets never collide.
#
# Override via env:
#   ./run_qat_sweep.sh                     # 100-epoch sweep -> cp_qat100_*
#   EPOCHS=10 ./run_qat_sweep.sh           # 10-epoch sweep  -> cp_qat10_*
#   RUNS="monarch_8 butterfly_fc" ./run_qat_sweep.sh   # subset of the 7
#   LR=1e-4 ./run_qat_sweep.sh             # override the QAT learning rate

set -euo pipefail

cd "$(dirname "$0")"

ALL_RUNS="wide_monarch monarch_full monarch_8 butterfly_2blocks \
          butterfly_full butterfly_ortho butterfly_fc"

RUNS="${RUNS:-$ALL_RUNS}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
HF_REPO="${HF_REPO:-claroche1/sparse-nsnet2-checkpoints}"

if [[ ! -d .venv ]]; then
    echo "ERROR: no .venv found. Run 'uv sync' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Resolve a checkpoint path from the HF repo. hf_hub_download caches per file
# under ~/.cache/huggingface/hub/...; this returns the local path, downloading
# only on a cache miss. `python` here is the activated-venv interpreter.
resolve () {
    python - "$1" "$HF_REPO" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[2], f"{sys.argv[1]}/g_best"))
PY
}

run () {
    local name="$1"
    local dst="cp_qat${EPOCHS}_${name}"
    local src
    if ! src=$(resolve "$name"); then
        echo "SKIP $name (could not resolve $name/g_best from $HF_REPO)"
        return
    fi
    mkdir -p "$dst"
    echo "=== [$(date +%H:%M:%S)] QAT: $name (${EPOCHS}ep, lr=$LR, src=$src) -> $dst ==="
    PYTHONUNBUFFERED=1 python -u -m nsnet2.qat_train \
        --init_from "$src" \
        --checkpoint_path "$dst" \
        --w_bits 4 --a_bits 8 \
        --epochs "$EPOCHS" --lr "$LR" 2>&1 | tee "$dst/train.log"
    echo "=== [$(date +%H:%M:%S)] Done: $dst ==="
    echo
}

for name in $RUNS; do
    run "$name"
done

echo "All QAT runs complete."
