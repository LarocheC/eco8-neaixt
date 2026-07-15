# deploy/rt595/Makefile — NXP i.MX RT595 deploy target (M33 track first).
# Mirrors deploy/stm32n6/ but swaps ST Edge AI -> TFLite-Micro and STAI -> TFLM.
#
# SCAFFOLD: `doctor` works today. `export`/`convert`/`build`/`flash` are stubs
# pending (1) the NXP SDK on disk and (2) the ONNX->TFLite bridge decision.
# See README.md "Open decisions".

include config.mk

.PHONY: doctor export convert build flash console print-%

## doctor : validate the toolchain (arm-gcc, xtensa, pyocd, SDK, serial)
doctor:
	@bash scripts/doctor.sh

## export : PyTorch LiSenNet -> int8 QDQ ONNX (reuses ../../lisennet/quant_onnx.py)
export:
	@echo "TODO: reuse lisennet/export_onnx.py (export_streaming_fp32, hybrid nc24)"
	@echo "      then lisennet/quant_onnx.py --signed -> $(MODEL_ONNX)"
	@echo "      (this half is SDK-independent and can be built now — see README)"

## convert : int8 QDQ ONNX -> .tflite (int8)  [BRIDGE DECISION PENDING]
convert:
	@echo "TODO: ONNX->TFLite bridge not chosen yet (onnx2tf vs ai-edge-torch)."
	@echo "      See README 'Open decisions #1'. Target: $(MODEL)"

## build : compile the M33 TFLite-Micro inference app (evkmimxrt595@cm33, armgcc)
build:
	@if [ ! -f "$(SDK_DEVICE_HDR)" ]; then \
	  echo "SDK not found at $(SDK_DIR) — get it from mcuxpresso.nxp.com (see config.mk)."; exit 1; fi
	@bash scripts/build.sh


## flash : flash from the WINDOWS side (usbip can't carry the CMSIS-DAP probe)
flash:
	@echo "Flash from Windows: LinkServer flash <axf>  (or pyocd-on-Windows)."
	@echo "See scripts/flash_windows.ps1 (to be added) and README 'Flashing'."

## console : open the serial console (works over usbip)
console:
	@echo "screen $(SERIAL_PORT) $(SERIAL_BAUD)   (Ctrl-A k to quit)"
	@screen $(SERIAL_PORT) $(SERIAL_BAUD)

# Introspection hook used by scripts/doctor.sh (matches the N6 pattern).
print-%:
	@echo '$($*)'
