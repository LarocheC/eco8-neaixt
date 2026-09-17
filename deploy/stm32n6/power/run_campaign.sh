#!/usr/bin/env bash
# run_campaign.sh -- on-target latency + FNB58 board power for the N6Net graphs,
# with nsnet_dse's validated harness so the numbers compare directly with
# nsnet_dse/deploy/sparse_nsnet2 (same firmware, schedule, analyser, gates).
#
#   deploy/stm32n6/power/run_campaign.sh [validate|power|summary|all]   (default all)
#
# Campaign tree (CAMP, default build/power/campaign/out):
#   n6/<run>__<profile>/network.c   compiled cells (N6Net: compile_n6net.sh)
#   <run>_int8.onnx                 the graph each cell was compiled from
#   cells.txt                       one cell per line (# comments out a cell)
# Results (RESULTS, default deploy/stm32n6/results/n6net_power): latency CSV,
# power/pass{1,2}/<cell>/ raw FNB58 trace + UART marks + analysis, and the
# summarised campaign_{runs,cells}.csv.
# Profile suffix -> Neural-ART profile file used by `validate`:
#   n6-noextmem-ec, and N6Net's own n6-noextmem: <cell>/neuralart.json
#   anything else: ST's N6_scripts/user_neuralart.json
#
# Rig: FNB58 inline on CN8 (JP2 = 5V_USB_SNK), debug on CN6 around the meter.
# See nsnet_dse/deploy/stm32n6/harness/power_harness/README-LINUX.md.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
NSNET_DSE="${NSNET_DSE:-$HOME/nsnet_dse}"
H="$NSNET_DSE/deploy/stm32n6/harness"
CAMP="${CAMP:-$REPO/build/power/campaign/out}"
RESULTS="${RESULTS:-$REPO/deploy/stm32n6/results/n6net_power}"
WHAT="${1:-all}"

# shellcheck disable=SC1091
source "$H/env.sh" > /dev/null
export PYTHON="$NSNET_DSE/deploy/.venv/bin/python"
PROJ="$STEDGEAI_DIR/Projects/STM32N6570-DK/Applications/NPU_Validation"
MAIN="$PROJ/Core/Src/main.c"
mkdir -p "$RESULTS"

cells() { grep -v -e '^#' -e '^$' "$CAMP/cells.txt"; }

profile_of() {                       # cell -> --st-neural-art argument
  local cell=$1 prof=${1##*__}
  if [ -f "$CAMP/n6/$cell/neuralart.json" ]; then
    echo "$prof@$(readlink -f "$CAMP/n6/$cell/neuralart.json")"
  else
    echo "$prof@user_neuralart.json"
  fi
}

do_validate() {
  if grep -q PWR_BEGIN "$MAIN"; then
    echo "FATAL: $MAIN is the power-loop firmware; restore main.c.pwr_orig first"; return 1
  fi
  export OUT="$CAMP/measure"; mkdir -p "$OUT"
  for cell in $(cells); do
    if [ -f "$OUT/val_$cell.log" ] && grep -qa "by sample" "$OUT/val_$cell.log"; then
      echo "[$cell] cached"; continue
    fi
    echo "######## validate $cell ########"
    NPU_PROFILE="$(profile_of "$cell")" timeout 2400 \
      "$H/scripts/measure_streaming.sh" "$cell" "$CAMP/${cell%%__*}_int8.onnx" \
      "$CAMP/n6/$cell" 2>&1 | tail -3
  done
  # collect_latency.py does not skip commented cells: hand it a filtered view
  local view; view=$(mktemp -d)
  cells > "$view/cells.txt"; ln -s "$OUT" "$view/measure"
  "$PYTHON" "$NSNET_DSE/deploy/sparse_nsnet2/collect_latency.py" --out "$view" \
    > "$RESULTS/n6_latency_matrix.csv"
  rm -rf "$view"
  column -s, -t "$RESULTS/n6_latency_matrix.csv"
}

do_power() {
  GENROOT="$CAMP/n6" OUTROOT="$RESULTS/power" OUTDIR="$CAMP" \
    PROJ="$PROJ" "$NSNET_DSE/deploy/sparse_nsnet2/run_power.sh"
}

do_summary() {
  local P="$RESULTS/power"
  "$PYTHON" "$H/power_harness/pwr_campaign_summarize.py" "$P" \
    --latency-csv "$RESULTS/n6_latency_matrix.csv" \
    --out-runs "$P/campaign_runs.csv" --out-cells "$P/campaign_cells.csv"
}

case "$WHAT" in
  validate) do_validate ;;
  power)    do_power ;;
  summary)  do_summary ;;
  all)      do_validate && do_power && do_summary ;;
  *)        echo "usage: $0 [validate|power|summary|all]"; exit 2 ;;
esac
