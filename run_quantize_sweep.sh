#!/usr/bin/env bash
# Export + int8-quantize the 9 trained NSNet2 variants.
#
# For each cp_<name>/ that contains a g_best PyTorch checkpoint:
#   1. export_onnx.py  -> cp_<name>/g_best_fp32.onnx   (FP32 ONNX)
#   2. quant.py        -> cp_<name>/g_best.onnx        (static int8 ONNX)
#
# Idempotent: re-exports/re-quantizes on every invocation (silent overwrite).
# Override the run set or calibration count via env:
#   RUNS="baseline blockdiag_fc"  ./run_quantize_sweep.sh
#   N_CALIB_UTTS=400            ./run_quantize_sweep.sh

set -euo pipefail

cd "$(dirname "$0")"

ALL_RUNS="baseline blockdiag_fc butterfly_fc blockdiag_full butterfly_full \
          blockdiag_8 butterfly_ortho butterfly_2blocks wide_blockdiag"

RUNS="${RUNS:-$ALL_RUNS}"
N_CALIB_UTTS="${N_CALIB_UTTS:-200}"

if [[ ! -d .venv ]]; then
    echo "ERROR: no .venv found. Run 'uv sync' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

for name in $RUNS; do
    cp_dir="cp_${name}"
    ckpt="${cp_dir}/g_best"
    log="${cp_dir}/quantize.log"

    if [[ ! -f $ckpt ]]; then
        echo "SKIP $name (no $ckpt)"
        continue
    fi

    echo "=== [$(date +%H:%M:%S)] Quantize: $name (n_calib=$N_CALIB_UTTS) ==="
    {
        echo "--- export_onnx ---"
        PYTHONUNBUFFERED=1 python -u -m nsnet2.export_onnx \
            --checkpoint_file "$ckpt"
        echo
        echo "--- quantize_static ---"
        PYTHONUNBUFFERED=1 python -u -m nsnet2.quant \
            --checkpoint_dir "$cp_dir" \
            --num_utterances "$N_CALIB_UTTS"
    } 2>&1 | tee "$log"

    fp32_size=$(du -h "${cp_dir}/g_best_fp32.onnx" | cut -f1)
    int8_size=$(du -h "${cp_dir}/g_best.onnx"      | cut -f1)
    echo "=== [$(date +%H:%M:%S)] Done: $name (fp32=$fp32_size, int8=$int8_size) ==="
    echo
done

echo "All exports complete."
