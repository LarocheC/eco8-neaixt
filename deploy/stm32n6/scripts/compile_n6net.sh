#!/usr/bin/env bash
# Compile an N6Net deploy ONNX (n6net/export_npu.py) with stedgeai and summarise
# what the compiler did with it: memory placement, epoch split, and the static
# ops / NPU-cycle estimates from network_c_info.json.
#
#   compile_n6net.sh <model.onnx> [out_dir] [profile]
#   profile: n6-noextmem-ec (default) | n6-noextmem
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
for c in "${STEDGEAI:-}" "$HOME/stedgeai/install/4.0/Utilities/linux/stedgeai" \
         "$HOME/stedgeai/4.0/Utilities/linux/stedgeai"; do
  [ -x "$c" ] && { STEDGEAI="$c"; break; }
done
# <root>/Utilities/linux/stedgeai; the profile file names its mpool under <root>
ST_ROOT="$(dirname "$(dirname "$(dirname "$STEDGEAI")")")"
MODEL="$(readlink -f "$1")"
OUT="$(mkdir -p "${2:-${MODEL%.onnx}_gen}" && readlink -f "${2:-${MODEL%.onnx}_gen}")"
PROFILE="${3:-n6-noextmem-ec}"

sed "s|/home/claroche/stedgeai/install/4.0|$ST_ROOT|" "$HERE/n6net_neuralart.json" \
  > "$OUT/neuralart.json"
cd "$HERE"
"$STEDGEAI" generate -m "$MODEL" --target stm32n6 \
  --st-neural-art "$PROFILE@$OUT/neuralart.json" \
  -o "$OUT" -w "$OUT/ws" > "$OUT/generate.log" 2>&1 \
  || { tail -30 "$OUT/generate.log"; exit 1; }

R="$OUT/network_generate_report.txt"
sed 's/\r/\n/g' "$R" | grep -E 'npuRAM|^Total:|Total number of epochs|epochs +[0-9]+'
sed 's/\r/\n/g' "$R" | grep -E '^\| epoch' | grep -v -E '\|\s+HW\s+\|' || true
python3 "$HERE/host/npu_cycle_report.py" "$OUT"
