#!/usr/bin/env bash
# On-board latency for one LiSenNet streaming int8 graph on the STM32N6570-DK.
#   $1 = short name (must match the n6_<name> generate dir)
#   $2 = path to the .onnx that was compiled
#
# HARD GATE: `validate` only runs if the firmware for THIS model actually loaded.
# The gdb "Loading memories failed" fault is transient, and if we validated anyway we
# would be timing whatever firmware was still resident from the previous model — a
# plausible-looking number for the wrong graph. Retry the load; refuse to measure if
# it never comes up.
set -u
NAME=$1; ONNX=$2
N6DIR=~/stedgeai/install/4.0/scripts/N6_scripts
STEDGEAI=~/stedgeai/install/4.0/Utilities/linux/stedgeai
OUT=/tmp/claude-1000/-home-claroche-eco8-neaixt/fcbba1cd-2ff9-4f41-b78e-09cdb1fa31a2/scratchpad

cd "$N6DIR" || exit 1

loaded=0
for attempt in 1 2 3; do
  pkill -x ST-LINK_gdbserver 2>/dev/null
  sleep 5                                   # let the ST-LINK/USB settle before re-grabbing it
  echo "[$NAME] loading firmware (attempt $attempt)..."
  timeout 900 python n6_loader.py --config config.json -nf "$OUT/n6_$NAME/network.c" -bc N6-DK \
      > "$OUT/load_$NAME.log" 2>&1
  if grep -q "Start operation achieved successfully" "$OUT/load_$NAME.log"; then
    loaded=1; echo "[$NAME] firmware up (attempt $attempt)"; break
  fi
  echo "[$NAME] load failed: $(grep -c 'Loading memories failed' "$OUT/load_$NAME.log") memory-load faults"
done

if [ "$loaded" -ne 1 ]; then
  echo "[$NAME] LOAD_FAILED — refusing to validate (would time stale firmware)"
  echo "MEASURE_DONE_$NAME"
  exit 1
fi

sleep 2
echo "[$NAME] validating on target..."
timeout 900 "$STEDGEAI" validate -m "$ONNX" --target stm32n6 \
    --st-neural-art n6-noextmem@user_neuralart.json \
    --fix-parametric-shapes "{'B':1}" \
    --mode target -d serial:/dev/ttyACM0:921600 \
    > "$OUT/val_$NAME.log" 2>&1
echo "[$NAME] validate exit=$?"
grep -a "by sample" "$OUT/val_$NAME.log" | sed 's/.*duration/duration/' | head -2
echo "MEASURE_DONE_$NAME"
