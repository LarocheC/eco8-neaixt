#!/usr/bin/env bash
# Validate the toolchain before deploying. Non-fatal: prints status + warnings.
# Pulls fully-resolved paths from the Makefile (so $(VAR) references expand correctly).
set -uo pipefail
cd "$(dirname "$0")/.."

cfg() { make -s "print-$1"; }
STEDGEAI=$(cfg STEDGEAI);         NPU_PROFILE_DIR=$(cfg NPU_PROFILE_DIR)
GCC_PATH=$(cfg GCC_PATH);         PROG_CLI=$(cfg PROG_CLI)
SIGN_CLI=$(cfg SIGN_CLI);         EXT_LOADER=$(cfg EXT_LOADER)
APP_DIR=$(cfg APP_DIR)

ok(){   printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
warn(){ printf '  \033[33m[warn]\033[0m %s\n' "$1"; }
bad(){  printf '  \033[31m[fail]\033[0m %s\n' "$1"; }

echo "== ST Edge AI Core (model compiler) =="
if [[ -x "$STEDGEAI" ]]; then
  V=$("$STEDGEAI" --version 2>/dev/null | head -1)
  ok "$STEDGEAI"; echo "        $V"
  case "$V" in
    *v4.*) ok "v4.x matches STM32N6-GettingStarted-Audio's ll_aton middleware";;
    *)     warn "NOT v4.x — the audio app was generated with STEdgeAI 4.0.0. Mixing majors"
           warn "raises 'Possible mismatch in ll_aton library used' at build. Update Core to 4.x.";;
  esac
else bad "stedgeai not found at $STEDGEAI"; fi
[[ -f "$NPU_PROFILE_DIR/neural_art.json" ]] && ok "neural_art.json profile present" \
  || warn "neural_art.json not at $NPU_PROFILE_DIR"

echo "== Arm GNU toolchain =="
GCC="$GCC_PATH/arm-none-eabi-gcc"
if [[ -x "$GCC" ]]; then
  GV=$("$GCC" -dumpversion 2>/dev/null)
  ok "$GCC ($GV)"
  if [[ "${GV%%.*}" -ge 13 ]]; then ok "gcc >= 13 (Cortex-M55 validated)";
  else warn "gcc $GV < 13 — ST validates 13.3.rel1 for -mcpu=cortex-m55 -mcmse. Install Arm GNU 13.3.Rel1."; fi
else bad "arm-none-eabi-gcc not found at $GCC"; fi

echo "== STM32CubeProgrammer (flash + sign) =="
[[ -x "$PROG_CLI" ]] && ok "$PROG_CLI" || bad "STM32_Programmer_CLI missing — install STM32CubeProgrammer 2.21+"
[[ -x "$SIGN_CLI" ]] && ok "$SIGN_CLI" || bad "STM32_SigningTool_CLI missing (ships with CubeProgrammer 2.21+)"
[[ -f "$EXT_LOADER" ]] && ok "external loader present" \
  || bad "MX66UW1G45G_STM32N6570-DK.stldr missing at $EXT_LOADER"

echo "== ST ready-made app =="
[[ -d "$APP_DIR/.git" ]] && ok "$APP_DIR cloned" || warn "not cloned — run 'make bootstrap'"

echo "== Dev Cloud credentials (optional, for 'make bench-cloud') =="
[[ -f cloud/.env ]] && ok "cloud/.env present" || warn "no cloud/.env — copy cloud/.env.example and fill in MyST creds"
echo "Done."
