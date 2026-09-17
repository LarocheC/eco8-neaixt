#!/usr/bin/env bash
# Validate probe (or model) cells one after another, stopping at the first hang
# so one wedged board costs one cell. Cells live in the campaign tree that
# run_campaign.sh uses; CAMP overrides its location.
#
#   run_probes.sh <cell> [cell ...]
#
# Do NOT probe the board with STM32_Programmer_CLI between cells: connecting
# straight after a `validate` run wedges the ST-LINK itself (DEV_USB_COMM_ERR,
# CN6 replug). measure_streaming.sh already refuses to measure a failed load.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
NSNET_DSE="${NSNET_DSE:-$HOME/nsnet_dse}"
CAMP="${CAMP:-$REPO/build/power/campaign/out}"
# shellcheck disable=SC1091
source "$NSNET_DSE/deploy/stm32n6/harness/env.sh" > /dev/null
export OUT="$CAMP/measure"

for cell in "$@"; do
  sleep 5
  log="$OUT/val_$cell.log"
  NPU_PROFILE="${cell##*__}@$(readlink -f "$CAMP/n6/$cell/neuralart.json")" timeout 1200 \
    "$NSNET_DSE/deploy/stm32n6/harness/scripts/measure_streaming.sh" \
    "$cell" "$CAMP/${cell%%__*}_int8.onnx" "$CAMP/n6/$cell" > /dev/null 2>&1
  if grep -qa "by sample" "$log" 2>/dev/null; then
    echo "RUNS  $cell $(grep -a 'by sample' "$log" | sed 's/.*: *//' | head -1)" \
         "min cos=$(grep -ao 'cos=[0-9.]*' "$log" | sort | head -1 | cut -d= -f2)"
  elif grep -qa "read timeout" "$log" 2>/dev/null; then
    echo "HANGS $cell"; exit 1
  else
    echo "FAIL  $cell (load or other; see $OUT/load_$cell.log)"; exit 3
  fi
done
echo PROBES_DONE
