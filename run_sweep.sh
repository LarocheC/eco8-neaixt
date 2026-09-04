#!/usr/bin/env bash
# Train NSNet2 across a compression sweep.
#
# Each run lives in its own checkpoint directory (cp_<name>) with a copy of
# its config, tensorboard logs under cp_<name>/logs, and a stdout log file
# at cp_<name>/train.log. Runs execute sequentially.
#
# These nine runs reproduce the published checkpoints at
#   https://huggingface.co/claroche1/sparse-nsnet2-checkpoints
# Their configs (configs/<name>.json) are byte-identical to the config.json
# uploaded next to each HF checkpoint.
#
# Override per-run by setting RUNS, EPOCHS, VAL_INTERVAL, BEST_START in env.
# Skip / pick a subset by setting RUNS, e.g.:
#   RUNS="baseline blockdiag_fc"    ./run_sweep.sh   # original first batch
#   RUNS="$ORIGINAL_RUNS $NEW_RUNS" ./run_sweep.sh   # everything (block-diagonal)
#   RUNS="$MONARCH_RUNS"            ./run_sweep.sh   # genuine 2-factor Monarch
#
# TRITON=1 reproduces the SAME nine architectures using the Triton GRU
# kernel from gru-qat (>=0.4.0) + the structured-matrix primitives from
# torch-structured, via the configs/<name>_triton.json variants:
#   TRITON=1 ./run_sweep.sh
# These map each run's GRU onto gru_qat.GRULayer:
#   gru      -> "triton"             dense GRU through gru_qat's per-step path
#                                    (no persistent kernel exists for a dense
#                                    hidden; same architecture as cuDNN nn.GRU).
#   blockdiag-> "triton_blockdiag"   single block-diagonal hidden via the
#   monarch  -> "triton_monarch"     persistent Triton scan kernel (gru-qat
#   butterfly-> "triton_butterfly"   >=0.4.0 ships scan_blockdiag + a genuine
#                                    scan_monarch); all add struct_input=true so
#                                   the GRU *input* projection also stays
#                                   structured (matching the HF runs that
#                                   structured BOTH GRU projections). The input
#                                   GEMM is hoisted out of the recurrence and
#                                   streamed into the hidden kernel.
# The structured FC layers are unchanged and already run on torch-structured's
# Triton/native backend. Same parameter count and structured-matrix family as
# the HF runs (verified +0.0%); NOT bit-identical (gru_qat packs the gates as
# three per-gate matrices vs the original fused input->3H matrix). For genuine
# Monarch the +0.0% parity requires gru-qat >= 0.5.0's FUSED (shared-w1) layout
# (StructureConfig.monarch_fused=True, the default) — the older per-gate layout
# triplicated the Monarch w1 factor (+~29% params). blockdiag/butterfly are
# +0.0% regardless of layout. The Triton
# hidden kernel needs a CUDA device; on CPU gru_qat falls back to an equivalent
# per-step loop. Triton runs land in cp_<name>_triton/ so they never clobber
# the canonical HF-reproduction runs.
#
# Resume a run by re-invoking with the same name -- train.py picks up the
# latest checkpoint from cp_<name>/ automatically.

set -euo pipefail

cd "$(dirname "$0")"

ORIGINAL_RUNS="baseline blockdiag_fc butterfly_fc blockdiag_full butterfly_full"
NEW_RUNS="blockdiag_8 butterfly_ortho butterfly_2blocks wide_blockdiag"

# Genuine two-factor Monarch runs (torch-structured >= 1.3.0). These are the
# real-Monarch analogs of the blockdiag_* runs above, for a like-for-like
# comparison of block-diagonal vs true-Monarch structure. Opt in explicitly:
#   RUNS="$MONARCH_RUNS" ./run_sweep.sh
MONARCH_RUNS="monarch_8 monarch_full monarch_fc wide_monarch"

# Default: full 9-run sweep (5 originals + 4 follow-ups), all at n_fft=512.
RUNS="${RUNS:-$ORIGINAL_RUNS $NEW_RUNS}"
EPOCHS="${EPOCHS:-30}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
BEST_START="${BEST_START:-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
STDOUT_INTERVAL="${STDOUT_INTERVAL:-50}"

# TRITON=1 selects the configs/<name>_triton.json variants and writes to
# cp_<name>_triton/. Default 0 = canonical HF-reproduction sweep.
TRITON="${TRITON:-0}"
if [[ "$TRITON" == "1" ]]; then
    cfg_suffix="_triton"
else
    cfg_suffix=""
fi

if [[ ! -d .venv ]]; then
    echo "ERROR: no .venv found. Run 'uv sync' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

for name in $RUNS; do
    cfg="configs/${name}${cfg_suffix}.json"
    cp_dir="cp_${name}${cfg_suffix}"
    log="${cp_dir}/train.log"

    if [[ ! -f $cfg ]]; then
        echo "SKIP $name (no $cfg)"
        continue
    fi

    mkdir -p "$cp_dir"
    echo "=== [$(date +%H:%M:%S)] Run: $name -> $cp_dir (epochs=$EPOCHS, val=$VAL_INTERVAL) ==="
    PYTHONUNBUFFERED=1 python -u -m nsnet2.train \
        --config "$cfg" \
        --checkpoint_path "$cp_dir" \
        --training_epochs "$EPOCHS" \
        --validation_interval "$VAL_INTERVAL" \
        --checkpoint_interval "$CHECKPOINT_INTERVAL" \
        --best_checkpoint_start_epoch "$BEST_START" \
        --stdout_interval "$STDOUT_INTERVAL" \
        2>&1 | tee "$log"

    # Auto-prune: keep only the latest rolling g_/do_ checkpoint and g_best.
    # We don't need the historical rolling ones for this sweep.
    if compgen -G "${cp_dir}/g_????????" > /dev/null; then
        latest_g=$(ls -1 "${cp_dir}"/g_???????? | sort | tail -1)
        latest_do=$(ls -1 "${cp_dir}"/do_???????? | sort | tail -1)
        find "$cp_dir" -maxdepth 1 -regex ".*/\(g\|do\)_[0-9]+" \
             ! -path "$latest_g" ! -path "$latest_do" -delete
    fi

    echo "=== [$(date +%H:%M:%S)] Done: $name (cp size: $(du -sh "$cp_dir" | cut -f1)) ==="
    echo
done

echo "All runs complete. Compare with:"
echo "  tensorboard --logdir_spec=$(echo "$RUNS" | sed "s/\([^ ]*\)/\1:cp_\1${cfg_suffix}\/logs/g; s/ /,/g")"
