#!/usr/bin/env bash
# Flash the RT595 M33 firmware from Linux and capture the serial report.
#
# Supersedes the "flash from Windows" note in the Makefile. That advice existed
# because both pyocd and LinkServer appeared unable to talk to the probe -- but
# the real cause was neither tool: the LPC-Link2's CMSIS-DAP HID interface
# (usually 1-4:1.0) had NO KERNEL DRIVER BOUND, so every tool saw an enumerated
# USB device that returned zero bytes to DAP_INFO. One root cause, two misleading
# symptoms ("index out of range" from pyocd, "No probes detected" from LinkServer).
#
# If the probe is unbound this script says so and prints the one-line fix; it
# needs root but NOT physical access to the board.
set -euo pipefail
cd "$(dirname "$0")/.."
RT595=$(pwd)

ELF="${ELF:-$RT595/build/lisennet_se_cm33.elf}"
PYOCD="${PYOCD:-$HOME/.venvs/rt595-flash/bin/pyocd}"
TARGET="${PYOCD_TARGET:-mimxrt595sffoc}"
PORT="${SERIAL_PORT:-/dev/ttyACM0}"
BAUD="${SERIAL_BAUD:-115200}"
CAPTURE="${CAPTURE:-$RT595/build/serial_capture.txt}"
SECONDS_CAPTURE="${SECONDS_CAPTURE:-25}"

# ---- 1. is the CMSIS-DAP HID interface actually bound? --------------------
bad=""
for i in /sys/bus/usb/devices/*:1.0; do
  [ -f "${i%:*.*}/idVendor" ] || continue
  [ "$(cat "${i%:*.*}/idVendor" 2>/dev/null)" = "1fc9" ] || continue
  [ "$(cat "$i/bInterfaceClass" 2>/dev/null)" = "03" ] || continue
  if [ ! -e "$i/driver" ]; then bad="$(basename "$i")"; fi
done
if [ -n "$bad" ]; then
  cat >&2 <<EOF
ERROR: the probe's CMSIS-DAP HID interface ($bad) has no driver bound.
The device enumerates on USB but returns nothing to DAP_INFO, so pyocd fails with
"index out of range" and LinkServer reports "No probes detected". Bind it:

    echo -n '$bad' | sudo tee /sys/bus/usb/drivers/usbhid/bind

(needs root; does NOT need physical access to the board). A physical debug-USB
re-plug also works.
EOF
  exit 2
fi

[ -f "$ELF" ] || { echo "no ELF at $ELF -- run scripts/build.sh first" >&2; exit 1; }

# ---- 2. capture serial across the reset, so we catch the boot banner ------
echo "== capturing $PORT @ $BAUD -> $CAPTURE =="
stty -F "$PORT" "$BAUD" raw -echo -echoe -echok -crtscts 2>/dev/null || true
( timeout "$SECONDS_CAPTURE" cat "$PORT" > "$CAPTURE" ) &
CAP=$!
sleep 1

# ---- 3. flash ------------------------------------------------------------
# NOTE: pyocd/SWD flashing is KNOWN BROKEN on this part (investigated 2026-07-23).
# pyocd types the SRAM alias 0x10000000-0x10020000 as ROM, so the DFP's FlexSPI algo
# (RAMstart 0x1001c000) is rejected by flm_region_builder._select_flash_ram; forcing
# it past that makes the algo's pc_init hardfault (IPSR=3). It is attempted here
# because it costs seconds and would be the nicest path if a newer DFP ever fixes it,
# but the SUPPORTED route on this board is blhost over the ROM ISP -- see below.
echo "== flashing $(basename "$ELF") (pyocd; known-fragile on this part) =="
if ! "$PYOCD" flash -t "$TARGET" --format elf "$ELF"; then
  cat >&2 <<EOF

pyocd flash failed -- expected on this part. Use the ROM ISP route instead, which is
NXP's supported path and sidesteps the whole SWD/FLM problem. It DOES need physical
access to SW7:

  1. SW7 -> Serial ISP:  1-ON, 2-OFF, 3-OFF   then power-cycle the board
  2. ~/.venvs/rt595-flash/bin/blhost -p $PORT -- flash-image \\
         $RT595/build/$(basename "${ELF%.elf}").bin erase
  3. SW7 -> flash boot:  1-OFF, 2-OFF, 3-ON   then power-cycle to run

SWD attach/read/reset still work without changing SW7, so once the probe is bound you
can verify and remotely reset an already-flashed board from here:
  pyocd commander -t $TARGET   (then: reg, read32 0xE000ED08, ...)
EOF
  kill $CAP 2>/dev/null || true; exit 1
fi

echo "== resetting =="
"$PYOCD" reset -t "$TARGET" || true

wait $CAP 2>/dev/null || true
echo
echo "==== serial capture ($CAPTURE) ===="
cat "$CAPTURE" || true
