#!/usr/bin/env bash
# Train NSNet2 across an initialisation x layout sweep of the two GRU input
# projections (configs/ig_*.json), several seeds per arm.
#
# The ig_r* arms have the architecture and hyperparameters of
# configs/butterfly_ortho.json and differ only in the "gru" block: x_radices,
# the block sizes of the input-projection butterfly (nsnet2/mixed_radix.py),
# and x_init (randn, ortho or rownorm). ig_full is a copy of
# configs/butterfly_full.json. The input-projection layers draw their
# coefficients from a private generator, so at one seed the ig_r* arms share
# every other initial weight and the data order.
#
# Each run lives in cp_<arm>_s<seed>/ (cp_<arm>_<TAG>_s<seed>/ with TAG set):
# config.json, run.json, val.jsonl, best.json, train.log, logs/, g_best, weight
# snapshots (snapshots/g_eNNN) and full training states (states/s_eNNN).
#
# Environment, with defaults:
#   ARMS                 arms, in order (the eight ig_* configs)
#   SEEDS                seeds (1235 1236 1237); runs go seed by seed
#   EPOCHS=200 VAL_INTERVAL=200 CHECKPOINT_INTERVAL=500 BEST_START=5
#   STDOUT_INTERVAL=50   the training flags of run_sweep.sh
#   SNAPSHOT_EPOCHS="0-20,25-200:5"   STATE_EPOCHS="50,100,150,200"
#   GPU=0                the single GPU visible to every run
#   TAG                  extra tag in the run directory name
#   INIT_FROM, INIT_KEYS start from these tensors of another snapshot
#                        (nsnet2/train.py --init_from / --init_keys)
#   REF_DIR              each run must start from the weights recorded in
#                        $REF_DIR/<arm>/s<seed>/run.json (e.g. an --init_only pass)
#   EXPECT_FINGERPRINT   fingerprint that g_e000 must have (takes precedence
#                        over REF_DIR; meant for a single run)
# Examples:
#   ./run_init_sweep.sh                                   # every arm, seeds 1235-1237
#   ARMS="ig_r24_randn ig_r42_randn" SEEDS=1235 ./run_init_sweep.sh
#
# No resuming (train.py would resume from rolling checkpoints): a run whose
# run.json has an end time is skipped, and any other existing run directory is
# a partial run, moved to <dir>.aborted<k> (never deleted) before the run is
# started again in a fresh directory. A new attempt must start from the weights
# of the first one. Only the rolling g_/do_ checkpoints of a finished run are
# pruned. A failed run stops the sweep.

set -euo pipefail

cd "$(dirname "$0")"

ARMS="${ARMS:-ig_r2_randn ig_r2_ortho ig_r24_randn ig_r42_randn ig_full ig_r2_rownorm ig_r24_ortho ig_r42_ortho}"
SEEDS="${SEEDS:-1235 1236 1237}"
EPOCHS="${EPOCHS:-200}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
BEST_START="${BEST_START:-5}"
STDOUT_INTERVAL="${STDOUT_INTERVAL:-50}"
SNAPSHOT_EPOCHS="${SNAPSHOT_EPOCHS:-0-20,25-200:5}"
STATE_EPOCHS="${STATE_EPOCHS:-50,100,150,200}"
GPU="${GPU:-0}"
TAG="${TAG:-}"
INIT_FROM="${INIT_FROM:-}"
INIT_KEYS="${INIT_KEYS:-}"
REF_DIR="${REF_DIR:-}"
EXPECT_FINGERPRINT="${EXPECT_FINGERPRINT:-}"

if [[ ! -d .venv ]]; then
    echo "ERROR: no .venv found. Run 'uv sync' first." >&2
    exit 1
fi
if [[ -n "$INIT_FROM" && -z "$INIT_KEYS" ]] || [[ -z "$INIT_FROM" && -n "$INIT_KEYS" ]]; then
    echo "ERROR: INIT_FROM and INIT_KEYS go together." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# json_get FILE KEY: the value of a top-level key of a JSON file, empty if the
# file, the key or its value is missing.
json_get() {
    python - "$1" "$2" <<'PY'
import json, sys
try:
    v = json.load(open(sys.argv[1])).get(sys.argv[2])
except (OSError, ValueError):
    v = None
print("" if v is None else v)
PY
}

for seed in $SEEDS; do
    for arm in $ARMS; do
        cfg="configs/${arm}.json"
        name="cp_${arm}${TAG:+_${TAG}}_s${seed}"
        if [[ ! -f $cfg ]]; then
            echo "ERROR: no $cfg" >&2
            exit 1
        fi
        if [[ -n "$(json_get "$name/run.json" end_time)" ]]; then
            echo "SKIP $name (finished)"
            continue
        fi
        if [[ -e $name ]]; then
            k=1
            while [[ -e "$name.aborted$k" ]]; do k=$((k + 1)); done
            mv "$name" "$name.aborted$k"
            echo "MOVED partial run $name -> $name.aborted$k"
        fi

        expect="$EXPECT_FINGERPRINT"
        if [[ -z $expect && -n $REF_DIR ]]; then
            expect="$(json_get "$REF_DIR/$arm/s$seed/run.json" g_e000_fingerprint)"
            if [[ -z $expect ]]; then
                echo "ERROR: no g_e000 fingerprint in $REF_DIR/$arm/s$seed/run.json" >&2
                exit 1
            fi
        fi
        # The first attempt that wrote g_e000 fixes the weights of the next ones.
        k=1
        while [[ -e "$name.aborted$k" ]]; do
            first="$(json_get "$name.aborted$k/run.json" g_e000_fingerprint)"
            if [[ -n $first ]]; then
                if [[ -n $expect && $expect != "$first" ]]; then
                    echo "ERROR: $name.aborted$k started from $first, expected $expect" >&2
                    exit 1
                fi
                expect="$first"
                break
            fi
            k=$((k + 1))
        done

        args=(--config "$cfg" --checkpoint_path "$name" --seed "$seed"
              --training_epochs "$EPOCHS" --validation_interval "$VAL_INTERVAL"
              --checkpoint_interval "$CHECKPOINT_INTERVAL" --best_checkpoint_start_epoch "$BEST_START"
              --stdout_interval "$STDOUT_INTERVAL"
              --snapshot_epochs "$SNAPSHOT_EPOCHS" --state_epochs "$STATE_EPOCHS")
        if [[ -n $INIT_FROM ]]; then
            args+=(--init_from "$INIT_FROM" --init_keys "$INIT_KEYS")
        fi
        if [[ -n $expect ]]; then
            args+=(--expect_fingerprint "$expect")
        fi

        mkdir -p "$name"
        echo "=== [$(date +%H:%M:%S)] Run: $name (seed=$seed, epochs=$EPOCHS, val=$VAL_INTERVAL) ==="
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 python -u -m nsnet2.train "${args[@]}" \
            2>&1 | tee "$name/train.log"

        if [[ -z "$(json_get "$name/run.json" end_time)" ]]; then
            echo "ERROR: $name has no end time in run.json" >&2
            exit 1
        fi
        # Finished: prune the rolling checkpoints only (states/ holds the full
        # training state at the end of the run).
        find "$name" -maxdepth 1 -regex ".*/\(g\|do\)_[0-9]+" -delete
        echo "=== [$(date +%H:%M:%S)] Done: $name (cp size: $(du -sh "$name" | cut -f1)) ==="
        echo
    done
done
