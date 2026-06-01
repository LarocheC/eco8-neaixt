#!/usr/bin/env bash
# Evaluate FP32 vs int8 PESQ on the VBD test split for the 9 trained variants.
#
# For each cp_<name>/ that contains both g_best_fp32.onnx and g_best.onnx,
# runs inference_onnx.py end-to-end and tees the output to cp_<name>/eval.log.
# Prints a final summary table extracted from each log's "Mean:" line.
#
# Override: RUNS, MAX_UTTERANCES (default: full split = unset).
#   MAX_UTTERANCES=100 ./run_eval_sweep.sh
#   RUNS="baseline monarch_fc" ./run_eval_sweep.sh

set -euo pipefail

cd "$(dirname "$0")"

ALL_RUNS="baseline monarch_fc butterfly_fc monarch_full butterfly_full \
          monarch_8 butterfly_ortho butterfly_2blocks wide_monarch"

RUNS="${RUNS:-$ALL_RUNS}"
MAX_UTTERANCES="${MAX_UTTERANCES:-}"

if [[ ! -d .venv ]]; then
    echo "ERROR: no .venv found. Run 'uv sync' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

extra_args=()
if [[ -n $MAX_UTTERANCES ]]; then
    extra_args+=(--max_utterances "$MAX_UTTERANCES")
fi

for name in $RUNS; do
    cp_dir="cp_${name}"
    int8_path="${cp_dir}/g_best.onnx"
    fp32_path="${cp_dir}/g_best_fp32.onnx"
    log="${cp_dir}/eval.log"
    out_dir="${cp_dir}/enhanced"

    if [[ ! -f $int8_path || ! -f $fp32_path ]]; then
        echo "SKIP $name (missing $int8_path or $fp32_path)"
        continue
    fi

    echo "=== [$(date +%H:%M:%S)] Eval: $name (max_utts=${MAX_UTTERANCES:-all}) ==="
    PYTHONUNBUFFERED=1 python -u -m nsnet2.inference_onnx \
        --checkpoint_file "$int8_path" \
        --output_dir "$out_dir" \
        "${extra_args[@]}" 2>&1 | tee "$log" | grep -E "^Mean:|Error|Traceback" || true

    echo "=== [$(date +%H:%M:%S)] Done: $name ==="
    echo
done

echo
echo "=== Summary ==="
printf "%-22s %8s %8s %8s %6s\n" "config" "fp32" "int8" "delta" "rtf"
for name in $RUNS; do
    log="cp_${name}/eval.log"
    if [[ ! -f $log ]]; then
        printf "%-22s %s\n" "$name" "(no eval.log)"
        continue
    fi
    line=$(grep "^Mean:" "$log" | tail -1)
    fp32=$(echo "$line"  | sed -nE 's/.*FP32=([0-9.]+).*/\1/p')
    int8=$(echo "$line"  | sed -nE 's/.* int8=([0-9.]+),.*/\1/p')      # comma anchors; "RTF int8=" lacks one
    delta=$(echo "$line" | sed -nE 's/.*delta=(-?[0-9.]+).*/\1/p')
    rtf=$(echo "$line"   | sed -nE 's/.*RTF int8=([0-9.]+).*/\1/p')
    printf "%-22s %8s %8s %8s %6s\n" "$name" "${fp32:--}" "${int8:--}" "${delta:--}" "${rtf:--}"
done
