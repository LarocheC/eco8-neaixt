# deploy/stm32n6/config.mk — machine-local paths and the N6 memory map.
# Edit the paths for your machine, then run `make doctor` to validate them.
# Every value is overridable on the command line, e.g. `make generate MODEL=/path/x.onnx`.

# ---------------------------------------------------------------------------
# ST Edge AI Core — the ONNX -> Neural-ART model compiler.
# v4.0.1 installed to match STM32N6-GettingStarted-Audio's
# ll_aton middleware. The app was generated with 4.0.0 and we have 4.0.1 (a patch off):
# if `make build` still warns "Possible mismatch in ll_aton library used", refresh the
# app's bundled ll_aton per ST's "update project with a new ST Edge AI Core" procedure.
# ---------------------------------------------------------------------------
STEDGEAI_DIR    ?= /home/claroche/stedgeai/install/4.0
STEDGEAI        ?= $(STEDGEAI_DIR)/Utilities/linux/stedgeai
# Dir holding neural_art.json + stm32n6.mpool (the bundled Neural-ART profile).
NPU_PROFILE_DIR ?= $(STEDGEAI_DIR)/Utilities/linux/targets/stm32/resources/NPU/STM32N6xx
NPU_PROFILE     ?= default@neural_art.json

# ---------------------------------------------------------------------------
# STM32CubeProgrammer 2.21+ — provides STM32_Programmer_CLI and STM32_SigningTool_CLI.
# NOT installed yet on this machine (see README "Prerequisites"). Install, then set this.
# ---------------------------------------------------------------------------
CUBEPROG_DIR    ?= $(HOME)/STMicroelectronics/STM32Cube/STM32CubeProgrammer
PROG_CLI        ?= $(CUBEPROG_DIR)/bin/STM32_Programmer_CLI
SIGN_CLI        ?= $(CUBEPROG_DIR)/bin/STM32_SigningTool_CLI
EXT_LOADER      ?= $(CUBEPROG_DIR)/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr

# ---------------------------------------------------------------------------
# Arm GNU toolchain. ST validates 13.3.rel1 for Cortex-M55 (-mcpu=cortex-m55
# -mcmse -mfpu=fpv5-d16). This machine has 10.3.1, which predates good M55
# support — install Arm GNU Toolchain 13.3.Rel1 and point GCC_PATH at its bin/.
# ---------------------------------------------------------------------------
GCC_PATH        ?= $(HOME)/toolchains/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin

# ---------------------------------------------------------------------------
# ST ready-made application (cloned by `make bootstrap`). Ships the Speech
# Enhancement use case + a real GCC Makefile (no STM32CubeIDE needed).
# ---------------------------------------------------------------------------
APP_REPO        ?= https://github.com/STMicroelectronics/STM32N6-GettingStarted-Audio
APP_DIR         ?= $(CURDIR)/STM32N6-GettingStarted-Audio
APP_CONFIG      ?= bm                 # bm | bm_lp | freertos | freertos_lp
APP_GS_DIR      ?= $(APP_DIR)/Projects/GS
APP_MODEL_DIR   ?= $(APP_DIR)/Projects/X-CUBE-AI/models
# Built firmware binary (verify after `make bootstrap` — path differs per config).
APP_BIN         ?= $(APP_GS_DIR)/BuildGCC/BM/GS_Audio_N6.bin
APP_SIGNED      ?= $(CURDIR)/GS_Audio_N6_sign.bin
FSBL_HEX        ?= $(APP_DIR)/FSBL/ai_fsbl.hex

# ---------------------------------------------------------------------------
# Model under test. ConvFSENet int8 is the only family that compiles on the
# Neural-ART today (NSNet2 dense + structured variants crash the compiler).
# ---------------------------------------------------------------------------
MODEL           ?= $(abspath $(CURDIR)/../../cp_convfsenet/g_best.onnx)
GEN_OUT         ?= $(CURDIR)/st_ai_output

# ---------------------------------------------------------------------------
# STM32N6570-DK external xSPI-flash memory map.
# FSBL .hex carries its own address (0x70000000); app + weights need explicit
# addresses. Weights base must match the objcopy --change-addresses in model-install.
# ---------------------------------------------------------------------------
ADDR_APP        ?= 0x70100000
ADDR_WEIGHTS    ?= 0x70180000
WEIGHTS_HEX     ?= $(APP_MODEL_DIR)/network_data.hex
CROSS           ?= $(GCC_PATH)/arm-none-eabi-
