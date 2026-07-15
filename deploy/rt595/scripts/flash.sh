#!/usr/bin/env bash
# Flash the built firmware to the EVK-MIMXRT595 over the on-board LPC-Link2 (SWD).
# Intended for NATIVE Linux (real USB). Does NOT work under WSL2 — usbip cannot carry
# the CMSIS-DAP probe (interrupt-transfer resets); see HANDOVER.md.
#
# Prereqs: pyocd + the RT595 pack, and the udev rule installed (see below).
#   pipx install pyocd   # or: pip install pyocd
#   pyocd pack install mimxrt595sffoc
#   sudo cp "$(dirname "$0")/50-cmsis-dap.rules" /etc/udev/rules.d/ && \
#     sudo udevadm control --reload && sudo udevadm trigger   # then replug the probe
set -euo pipefail
cd "$(dirname "$0")/.."                            # deploy/rt595
BIN="${1:-build/lisennet_se_cm33.bin}"
TARGET="${PYOCD_TARGET:-mimxrt595sffoc}"
ADDR="${FLASH_ADDR:-0x08000000}"                  # FlexSPI0 base on RT500

[ -f "$BIN" ] || { echo "binary not found: $BIN (run scripts/build.sh first)"; exit 1; }

echo "Probes:"; pyocd list || true
echo "Flashing $BIN -> $TARGET @ $ADDR ..."
pyocd flash -t "$TARGET" --base-address "$ADDR" "$BIN"
echo "Done. Read telemetry: screen /dev/ttyACM0 115200   (Ctrl-A k to quit)"
