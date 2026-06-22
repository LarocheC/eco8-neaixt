#!/usr/bin/env bash
# Track 1 (stateless-windowed ConvFSENet) host evaluation driver.
#
# Quantizes the windowed FP32 ONNX (256-bin deploy target + 257-bin reference)
# from the v5 / 2.911 weights with the full 200-utterance VBD calibration, then
# measures int8 PESQ vs FP32 on the full 824-utterance VBD test split for both.
#
# Run under tmux so it survives an SSH drop:
#   tmux new -d -s win_eval 'bash deploy/stm32n6/scripts/run_windowed_eval.sh'
#   tmux attach -t win_eval        # to watch
#
# All output is tee'd into deploy/stm32n6/windowed_eval/*.log.
set -euo pipefail

cd "$(dirname "$0")/../../.."        # repo root
PY=.venv/bin/python
CP=cp_convfsenet_win
LOGDIR=deploy/stm32n6/windowed_eval
mkdir -p "$LOGDIR"

CALIB_UTTS="${CALIB_UTTS:-200}"
CALIB_FRAMES="${CALIB_FRAMES:-80}"

COLDSTART="${COLDSTART:-replicate}"
echo "=== [$(date -Is)] Track 1 windowed eval start (calib=${CALIB_UTTS} utts, ${CALIB_FRAMES} frames, coldstart=${COLDSTART}) ===" | tee "$LOGDIR/run.log"

run_step () {  # name, logfile, command...
  local name="$1"; local log="$2"; shift 2
  echo "--- [$(date -Is)] $name ---" | tee -a "$LOGDIR/run.log"
  "$@" 2>&1 | tee "$log" | tail -3 | tee -a "$LOGDIR/run.log"
  echo "--- [$(date -Is)] $name DONE ---" | tee -a "$LOGDIR/run.log"
}

# 0. (Re)export FP32 windowed ONNX (stamps window geometry metadata).
run_step "export 257" "$LOGDIR/export257.log" \
  $PY -m convfsenet.export_onnx --windowed --checkpoint_file "$CP/g_best" --emit_T 1 \
    --output "$CP/g_best_win257_fp32.onnx"
run_step "export 256" "$LOGDIR/export256.log" \
  $PY -m convfsenet.export_onnx --windowed --checkpoint_file "$CP/g_best" --emit_T 1 --drop_nyquist \
    --output "$CP/g_best_win256_fp32.onnx"

# 1. Quantize int8 (full calibration).
run_step "quant 257" "$LOGDIR/quant257.log" \
  $PY -m convfsenet.quant_windowed --checkpoint_dir "$CP" \
    --num_utterances "$CALIB_UTTS" --frames_per_utterance "$CALIB_FRAMES" --emit_T 1
run_step "quant 256" "$LOGDIR/quant256.log" \
  $PY -m convfsenet.quant_windowed --checkpoint_dir "$CP" \
    --num_utterances "$CALIB_UTTS" --frames_per_utterance "$CALIB_FRAMES" --emit_T 1 --drop_nyquist

# 2. Full-test PESQ (int8 primary + FP32 sidecar => delta).
run_step "pesq 257" "$LOGDIR/pesq257.log" \
  $PY -m convfsenet.inference_windowed_onnx \
    --checkpoint_file "$CP/g_best_win257.onnx" \
    --fp32_checkpoint_file "$CP/g_best_win257_fp32.onnx" \
    --coldstart "$COLDSTART" --output_dir "$LOGDIR/enhanced_257"
run_step "pesq 256" "$LOGDIR/pesq256.log" \
  $PY -m convfsenet.inference_windowed_onnx \
    --checkpoint_file "$CP/g_best_win256.onnx" \
    --fp32_checkpoint_file "$CP/g_best_win256_fp32.onnx" \
    --coldstart "$COLDSTART" --output_dir "$LOGDIR/enhanced_256"

echo "=== [$(date -Is)] Track 1 windowed eval COMPLETE ===" | tee -a "$LOGDIR/run.log"
echo "Summary lines:" | tee -a "$LOGDIR/run.log"
grep -h "^Mean:" "$LOGDIR"/pesq257.log "$LOGDIR"/pesq256.log | tee -a "$LOGDIR/run.log"
