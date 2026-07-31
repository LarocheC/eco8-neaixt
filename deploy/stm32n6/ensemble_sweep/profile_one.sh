#!/usr/bin/env bash
# Load one graph's firmware (hard gate, 2 attempts) and npu_profile it.
set -u
NAME=$1
BATCH=${2:-16}
N6DIR=~/stedgeai/install/4.0/scripts/N6_scripts
RUNNER=~/stedgeai/install/4.0/scripts/ai_runner
SP=/tmp/claude-1000/-home-claroche-eco8-neaixt/cead2b27-c88f-41b4-b58e-07c77fd6ee40/scratchpad

cd "$N6DIR" || exit 1
loaded=0
for attempt in 1 2; do
  pkill -x ST-LINK_gdbserver 2>/dev/null
  sleep 5
  timeout 240 python n6_loader.py --config config.json \
      -nf "$SP/out_$NAME/network.c" -bc N6-DK > "$SP/pload_$NAME.log" 2>&1
  if grep -q "Start operation achieved successfully" "$SP/pload_$NAME.log"; then
    loaded=1; echo "[$NAME] firmware up (attempt $attempt)"; break
  fi
  echo "[$NAME] load attempt $attempt failed"
done
[ "$loaded" -ne 1 ] && { echo "[$NAME] LOAD_FAILED"; exit 1; }

sleep 2
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH="$RUNNER" \
  timeout 300 "$SP/profenv/bin/python" "$RUNNER/examples/npu_profiler.py" \
    -d serial:/dev/ttyACM0:921600 -c "$SP/out_$NAME" -b "$BATCH" --no-color \
    > "$SP/prof_$NAME.log" 2>&1
rc=$?
echo "[$NAME] profile exit=$rc"
grep -a -i -E "inference time|per inference|total|duration" "$SP/prof_$NAME.log" | head -6
exit $rc
