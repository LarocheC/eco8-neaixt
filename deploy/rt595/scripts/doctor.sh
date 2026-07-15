#!/usr/bin/env bash
# Validate the RT595 toolchain before deploying. Non-fatal: prints status + warnings.
# Pulls fully-resolved paths from the Makefile (so $(VAR) references expand correctly).
set -uo pipefail
cd "$(dirname "$0")/.."

cfg() { make -s "print-$1"; }
SDK_DIR=$(cfg SDK_DIR);           SDK_DEVICE_HDR=$(cfg SDK_DEVICE_HDR)
TFLM_DIR=$(cfg TFLM_DIR);         CMSIS_NN_DIR=$(cfg CMSIS_NN_DIR)
GCC_PATH=$(cfg GCC_PATH)
XTENSA_TOOLS=$(cfg XTENSA_TOOLS); XTENSA_BUILDS=$(cfg XTENSA_BUILDS); XTENSA_CORE=$(cfg XTENSA_CORE)
PYOCD=$(cfg PYOCD);               PYOCD_TARGET=$(cfg PYOCD_TARGET)
SERIAL_PORT=$(cfg SERIAL_PORT)

ok(){   printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
warn(){ printf '  \033[33m[warn]\033[0m %s\n' "$1"; }
bad(){  printf '  \033[31m[fail]\033[0m %s\n' "$1"; }

echo "== NXP MCUXpresso SDK (EVK-MIMXRT595) =="
if [[ -f "$SDK_DEVICE_HDR" ]]; then
  ok "device header present ($SDK_DEVICE_HDR)"
  [[ -d "$TFLM_DIR"     ]] && ok "eIQ TFLite-Micro middleware present" \
    || warn "TFLite-Micro not at $TFLM_DIR — re-build the SDK with the eIQ component."
  [[ -d "$CMSIS_NN_DIR" ]] && ok "CMSIS-NN present" || warn "CMSIS-NN not at $CMSIS_NN_DIR"
else
  bad "SDK not found at $SDK_DIR"
  warn "Get it: mcuxpresso.nxp.com -> SDK Builder -> EVK-MIMXRT595, Linux, GCC ARM Embedded,"
  warn "middleware = eIQ (+ DSP/HiFi4). Unzip to $SDK_DIR, then re-run 'make doctor'."
fi

echo "== Arm GNU toolchain (Cortex-M33) =="
GCC="$GCC_PATH/arm-none-eabi-gcc"
if [[ -x "$GCC" ]]; then
  GV=$("$GCC" -dumpversion 2>/dev/null); ok "$GCC ($GV)"
  [[ "${GV%%.*}" -ge 10 ]] && ok "gcc >= 10 (Cortex-M33 OK)" || warn "gcc $GV is old for -mcpu=cortex-m33"
else bad "arm-none-eabi-gcc not found at $GCC"; fi

echo "== Cadence Xtensa (HiFi4 track — later) =="
if [[ -x "$XTENSA_TOOLS/bin/xt-clang" ]]; then
  ok "xt-clang present ($XTENSA_TOOLS)"
  CORES=$(ls "$XTENSA_BUILDS" 2>/dev/null | grep -v xtensa_licserver | tr '\n' ' ')
  echo "        registered cores: $CORES"
  if ls "$XTENSA_BUILDS" 2>/dev/null | grep -qiE 'rt5|hifi4|fusion'; then
    ok "an RT500/HiFi4 core appears registered"
  else
    warn "no RT500/HiFi4 core registered — only AIR/HiFi5 (a different chip)."
    warn "Register the SDK's RT595 HiFi4 core into XtensaTools and confirm the LICENSE covers it."
  fi
  [[ -n "$XTENSA_CORE" ]] && ok "XTENSA_CORE=$XTENSA_CORE" || warn "XTENSA_CORE unset (set once the RT500 core is registered)"
else warn "xt-clang not at $XTENSA_TOOLS/bin — HiFi4 track unavailable"; fi

echo "== Flash / debug probe =="
if [[ -x "$PYOCD" ]]; then
  ok "pyocd present ($($PYOCD --version 2>/dev/null))"
  "$PYOCD" pack find "$PYOCD_TARGET" 2>/dev/null | grep -qi "True" \
    && ok "target pack for $PYOCD_TARGET installed" \
    || warn "run: pyocd pack install $PYOCD_TARGET"
else warn "pyocd not found at $PYOCD"; fi
warn "usbip CANNOT carry the CMSIS-DAP probe (HID interrupt resets). Flash from Windows"
warn "(LinkServer / pyocd-on-Windows). The serial VCP below works fine over usbip."

echo "== Serial console =="
if [[ -e "$SERIAL_PORT" ]]; then
  ok "$SERIAL_PORT present"
  [[ -r "$SERIAL_PORT" && -w "$SERIAL_PORT" ]] && ok "readable/writable (in dialout)" \
    || warn "no rw on $SERIAL_PORT — add yourself to the 'dialout' group"
else warn "$SERIAL_PORT not present — attach the board via usbipd (see README)"; fi
command -v screen >/dev/null && ok "screen present (make console)" || warn "no serial terminal (apt install screen)"

echo "Done."
