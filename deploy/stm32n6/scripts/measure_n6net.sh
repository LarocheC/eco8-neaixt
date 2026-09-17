#!/usr/bin/env bash
# On-board latency sweep for the N6Net deploy graphs (STM32N6570-DK).
#
#   deploy/stm32n6/scripts/measure_n6net.sh OUT_DIR [model ...]
#   SKIP_BOARD=1 deploy/stm32n6/scripts/measure_n6net.sh OUT_DIR     # export + compile only
#
# Per model: export (seeded init: the graph is all convolutions, so latency does
# not depend on weight values), compile with the same Neural-ART profile the
# firmware is built from, load the NPU_Validation firmware, time it with
# `stedgeai validate --mode target`, then run npu_profiler for the HW/SW split.
#
# HARD GATE (from measure_streaming.sh): validate only runs if the firmware for
# THIS model loaded. gdb's "Loading memories failed" is transient; validating
# anyway would time whatever firmware was still resident from the previous model.
#
# Prerequisites: board in dev-boot, ST-LINK attached to WSL
# (`usbipd attach --wsl --busid <id> --auto-attach`), /dev/ttyACM0 present.
# Output: OUT_DIR/latency.tsv (+ one directory of logs per model).
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # deploy/stm32n6
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$(mkdir -p "${1:?usage: $0 OUT_DIR [model ...]}" && readlink -f "$1")"; shift
PROFILE="${PROFILE:-n6-noextmem-ec}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python3}"
STEDGEAI="${STEDGEAI:-/home/claroche/stedgeai/install/4.0/Utilities/linux/stedgeai}"
N6DIR="${N6DIR:-/home/claroche/stedgeai/install/4.0/scripts/N6_scripts}"
RUNNER="${RUNNER:-/home/claroche/stedgeai/install/4.0/scripts/ai_runner}"
PROFPY="${PROFPY:-$HOME/.venvs/n6prof/bin/python}"
PORT="${PORT:-/dev/ttyACM0}"
SAMPLES="${SAMPLES:-20}"

# name -> "config|extra export args". fullband_mpsenet is not listed: the
# objective only changes training, its deploy graph is n6net_v2_fullband's.
declare -A MODELS=(
  [v1_timew]="configs/n6net_b3.json|--layout time_w"
  [v1_split]="configs/n6net_b3.json|--layout time_split"
  [v2_c96]="configs/n6net_v2_c96.json|"
  [v2_fullband]="configs/n6net_v2_fullband.json|"
  [fb_nopos]="configs/n6net_v2_fullband_nopos.json|"
  [fb_lsig]="configs/n6net_v2_fullband_lsig.json|"
  [fb_phain]="configs/n6net_v2_fullband_phain.json|"
  [fb_phase]="configs/n6net_v2_fullband_phase.json|"
)
ORDER=(v1_timew v1_split v2_c96 v2_fullband fb_nopos fb_lsig fb_phain fb_phase)
[ $# -gt 0 ] && ORDER=("$@")

if [ -z "${SKIP_BOARD:-}" ] && [ ! -e "$PORT" ]; then
  echo "no $PORT: attach the ST-LINK to WSL first (usbipd attach --wsl --busid <id> --auto-attach)"
  exit 2
fi

TSV="$OUT/latency.tsv"
[ -f "$TSV" ] || printf "model\tprofile\tmean_ms\tmin_ms\tmax_ms\tstd_ms\tloaded\n" > "$TSV"

for name in "${ORDER[@]}"; do
  spec="${MODELS[$name]:?unknown model $name}"
  cfg="${spec%%|*}"; extra="${spec#*|}"
  d="$OUT/$name"; mkdir -p "$d"

  echo "[$name] export ($cfg $extra)"
  ( cd "$REPO" && "$PYTHON" -m n6net.export_npu --config "$cfg" $extra --out "$d" ) \
      > "$d/export.log" 2>&1 || { echo "[$name] EXPORT FAILED"; tail -5 "$d/export.log"; continue; }
  onnx="$d/n6net_stream_int8.onnx"

  echo "[$name] compile ($PROFILE)"
  "$HERE/scripts/compile_n6net.sh" "$onnx" "$d/gen" "$PROFILE" > "$d/compile.log" 2>&1 \
      || { echo "[$name] COMPILE FAILED"; tail -5 "$d/compile.log"; continue; }
  grep -E 'implemented in|ops/max' "$d/compile.log" | tr -s ' ' | sed "s/^/[$name]   /"

  [ -n "${SKIP_BOARD:-}" ] && continue

  loaded=0
  for attempt in 1 2 3; do
    pkill -x ST-LINK_gdbserver 2>/dev/null
    sleep 5                                     # let the ST-LINK settle before re-grabbing it
    echo "[$name] load firmware (attempt $attempt)"
    ( cd "$N6DIR" && timeout 900 python3 n6_loader.py --config config.json \
          -nf "$d/gen/network.c" -bc N6-DK ) > "$d/load.log" 2>&1
    if grep -q "Start operation achieved successfully" "$d/load.log"; then loaded=1; break; fi
    echo "[$name]   load failed ($(grep -c 'Loading memories failed' "$d/load.log") memory-load faults)"
  done
  if [ "$loaded" -ne 1 ]; then
    echo "[$name] LOAD_FAILED -- refusing to validate (would time stale firmware)"
    printf "%s\t%s\t\t\t\t\t0\n" "$name" "$PROFILE" >> "$TSV"
    continue
  fi

  sleep 2
  echo "[$name] validate on target ($SAMPLES samples)"
  ( cd "$HERE" && timeout 1800 "$STEDGEAI" validate -m "$onnx" --target stm32n6 \
        --st-neural-art "$PROFILE@n6net_neuralart.json" -b "$SAMPLES" \
        --mode target -d "serial:$PORT:921600" -o "$d/val" -w "$d/val_ws" ) \
      > "$d/validate.log" 2>&1
  # "duration : 0.912ms by sample (0.905/0.921/0.004)"  -> mean, min, max, std
  line="$(tr '\r' '\n' < "$d/validate.log" | grep -a -m1 'by sample')"
  echo "[$name]   ${line##*duration}"
  nums="$(echo "$line" | sed -E 's/.*duration *: *([0-9.]+) *ms by sample \(([0-9.]+)\/([0-9.]+)\/([0-9.]+)\).*/\1\t\2\t\3\t\4/')"
  [ "$nums" = "$line" ] && nums="$(printf '\t\t\t')"
  printf "%s\t%s\t%s\t1\n" "$name" "$PROFILE" "$nums" >> "$TSV"

  echo "[$name] profile"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="$RUNNER" timeout 900 \
      "$PROFPY" "$RUNNER/examples/npu_profiler.py" -d "serial:$PORT:921600" \
      -c "$d/gen" -b 16 --no-color > "$d/profile.log" 2>&1 \
      || echo "[$name]   profiler failed (see $d/profile.log)"
done

echo "== $TSV"
column -t -s $'\t' "$TSV"
