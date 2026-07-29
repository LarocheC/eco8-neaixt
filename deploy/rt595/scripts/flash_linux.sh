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

# ---- 1. is the probe's DAP engine actually alive? -------------------------
# Test the CHANNEL, not the driver binding. pyocd talks to this probe over libusb
# (its default backend is `pyusb`), so interface :1.0 having no kernel driver is
# NORMAL and correct -- binding usbhid to it would stop libusb claiming it. The
# only thing that matters is whether the probe answers a DAP command.
if ! "${PYTHON:-$HOME/.venvs/rt595-flash/bin/python}" - <<'PROBE'
import sys
try:
    from pyocd.probe.pydapaccess.interface import INTERFACE
    devs = INTERFACE['pyusb'].get_all_connected_interfaces()
    if not devs:
        print("no CMSIS-DAP probe found on USB", file=sys.stderr); sys.exit(1)
    d = devs[0]; d.open()
    try:
        pkt = bytearray(64); pkt[1] = 0xF0          # DAP_Info(capabilities)
        d.write(list(pkt))
        sys.exit(0 if d.read() else 2)
    finally:
        d.close()
except SystemExit:
    raise
except Exception as e:
    print(f"probe check failed: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
PROBE
then
  cat >&2 <<'EOF'
ERROR: the probe enumerates but its CMSIS-DAP engine does not answer.

Confirmed at the lowest reachable level: libusb claims interface 0 cleanly, writes
of both 64 and 1024 bytes are accepted, and every read returns ZERO bytes at any
timeout. That surfaces as pyocd "index out of range" (an IndexError on resp[0] of
an empty DAP_INFO reply) and LinkServer "No probes detected" -- one dead channel,
two confusing symptoms. The probe's VCOM (/dev/ttyACM0) keeps working throughout,
because that is a separate CDC interface on the same LPC4322.

This is NOT fixable in software. Do NOT bind usbhid to the interface -- pyocd uses
libusb and needs it unbound. USBDEVFS_RESET does not help either: it resets the USB
port, not the LPC4322 running the CMSIS-DAP firmware.

THE FIX IS PHYSICAL: unplug and replug the DEBUG USB cable (de-powers the LPC4322 so
it reboots its CMSIS-DAP firmware). Then re-run. Confirm with the probe serial:
    pyocd list      # expect ORA2CQKQ
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
