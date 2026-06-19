#!/usr/bin/env bash
# Sign the app and flash FSBL + app + AI weights to the STM32N6570-DK external xSPI flash.
#
# PRECONDITION (manual, not scriptable): the board must be in DEVELOPMENT BOOT mode
# (BOOT1 DIP switch) so the ROM lets the external loader write the xSPI NOR. If it is
# in flash/XIP boot mode you will get "flash loader cannot be loaded" / erase failures.
#
# The N6 has no internal user flash: the ROM bootloader loads the FSBL from external
# flash at 0x70000000. Every image the ROM jumps to needs an authentication header even
# in dev mode (`-nk` = header, no key). FSBL .hex carries its own address; the app .bin
# needs an explicit address; the weights .hex carries its own (set at objcopy time).
set -euo pipefail

: "${PROG_CLI:?}" "${SIGN_CLI:=}" "${EXT_LOADER:?}" "${FSBL_HEX:?}" "${WEIGHTS_HEX:?}"
: "${APP_BIN:=}" "${APP_SIGNED:?}" "${ADDR_APP:?}"

CONN=(-c port=SWD mode=HOTPLUG -el "$EXT_LOADER" -hardRst)

# 1. Sign the app binary (skip if APP_SIGNED already provided pre-signed).
if [[ -n "${APP_BIN}" && -f "${APP_BIN}" ]]; then
  echo "[flash] signing $APP_BIN -> $APP_SIGNED"
  # -align is mandatory on STM32CubeProgrammer 2.21+ (incl. 2.22.0): no more auto-pad to 0x400.
  "${SIGN_CLI:?set SIGN_CLI to sign}" -bin "$APP_BIN" -nk -t ssbl -hv 2.3 -align -o "$APP_SIGNED"
fi
test -f "$APP_SIGNED" || { echo "ERROR: signed app $APP_SIGNED not found"; exit 1; }

# 2. Three independent writes. Weights are flashed separately from the app, so you can
#    re-flash a newly-quantized model without rebuilding/re-signing the firmware.
echo "[flash] (1/3) FSBL    : $FSBL_HEX"
"$PROG_CLI" "${CONN[@]}" -w "$FSBL_HEX"
echo "[flash] (2/3) app     : $APP_SIGNED @ $ADDR_APP"
"$PROG_CLI" "${CONN[@]}" -w "$APP_SIGNED" "$ADDR_APP"
echo "[flash] (3/3) weights : $WEIGHTS_HEX"
"$PROG_CLI" "${CONN[@]}" -w "$WEIGHTS_HEX"

echo "[flash] done. Set BOOT1 back to flash-boot and power-cycle to run."
