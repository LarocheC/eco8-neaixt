# deploy/stm32n6/Makefile — fully-scripted (no-IDE) STM32N6570-DK deployment.
#
#   make doctor       # check toolchain paths/versions before anything else
#   make bootstrap    # clone the ST ready-made audio app
#   make generate     # ONNX -> Neural-ART C + int8 weights (stedgeai)
#   make io-layout    # regenerate app/model_io_layout.h from the compiled model
#   make model-install# drop generated files into the app + objcopy weights -> hex
#   make build        # cross-compile the app with arm-none-eabi-gcc (no IDE)
#   make sign         # add the N6 boot header to the app binary
#   make flash        # FSBL + signed app + weights, over SWD (board in dev-boot)
#   make deploy       # generate -> io-layout -> model-install -> build -> sign -> flash
#   make bench-cloud  # benchmark the model on ST's N6 board farm (no local board)
#   make clean
#
# All paths live in config.mk. Override anything: `make generate MODEL=/x.onnx`.

include config.mk
SHELL := /bin/bash
.DEFAULT_GOAL := help

OE_ONNX = $(firstword $(wildcard $(GEN_OUT)/*_OE_*.onnx))

.PHONY: help doctor bootstrap generate io-layout model-install build sign flash \
        deploy bench-cloud validate-target clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

print-%: ## Print a fully-resolved config variable (used by doctor.sh): make print-STEDGEAI
	@echo '$($*)'

doctor: ## Validate tool paths and versions (start here)
	@bash scripts/doctor.sh

bootstrap: ## Clone the ST STM32N6-GettingStarted-Audio app (idempotent)
	@if [ -d "$(APP_DIR)/.git" ]; then \
	  echo "[bootstrap] $(APP_DIR) already cloned"; \
	else \
	  echo "[bootstrap] cloning $(APP_REPO)"; \
	  git clone --depth 1 "$(APP_REPO)" "$(APP_DIR)"; \
	fi
	@echo "[bootstrap] NOTE: verify APP_BIN / APP_MODEL_DIR in config.mk against the cloned tree."

generate: ## Compile MODEL to Neural-ART C code + int8 weights into GEN_OUT
	@MODEL="$(MODEL)" STEDGEAI="$(STEDGEAI)" NPU_PROFILE_DIR="$(NPU_PROFILE_DIR)" \
	  NPU_PROFILE="$(NPU_PROFILE)" GEN_OUT="$(GEN_OUT)" bash scripts/generate.sh

io-layout: ## (Re)generate app/model_io_layout.h from the compiled model in GEN_OUT
	@test -n "$(OE_ONNX)" || { echo "ERROR: no *_OE_*.onnx in $(GEN_OUT); run 'make generate' first"; exit 1; }
	@PY=$$( [ -x ../../.venv/bin/python ] && echo ../../.venv/bin/python || echo python3 ); \
	 echo "[io-layout] using $$PY on $(OE_ONNX)"; \
	 $$PY host/gen_io_layout.py "$(OE_ONNX)" app/model_io_layout.h

model-install: ## Copy generated C model into the app and build the flashable weights .hex
	@test -d "$(APP_MODEL_DIR)" || { echo "ERROR: $(APP_MODEL_DIR) missing; run 'make bootstrap'"; exit 1; }
	cp $(GEN_OUT)/network.c $(GEN_OUT)/network.h \
	   $(GEN_OUT)/stai_network.c $(GEN_OUT)/stai_network.h $(APP_MODEL_DIR)/
	cp app/ai_dpu_se_stream.c app/ai_dpu_se_stream.h app/model_io_layout.h $(APP_MODEL_DIR)/
	$(CROSS)objcopy -I binary $(GEN_OUT)/network_atonbuf.xSPI2.raw \
	   --change-addresses $(ADDR_WEIGHTS) -O ihex $(WEIGHTS_HEX)
	@echo "[model-install] weights -> $(WEIGHTS_HEX) @ $(ADDR_WEIGHTS)"
	@echo "[model-install] NOTE: wire se_stream_init/process_frame into the app's audio loop (see README §6)."

build: ## Cross-compile the app (pure Makefile, no STM32CubeIDE)
	$(MAKE) -C "$(APP_GS_DIR)" $(APP_CONFIG) -j8 GCC_PATH="$(GCC_PATH)"

sign: ## Add the STM32N6 boot header to the built app binary (dev mode, no key)
	"$(SIGN_CLI)" -bin "$(APP_BIN)" -nk -t ssbl -hv 2.3 -o "$(APP_SIGNED)"
	@echo "[sign] signed -> $(APP_SIGNED)"

flash: ## Flash FSBL + signed app + weights over SWD (board must be in dev-boot mode)
	@APP_SIGNED="$(APP_SIGNED)" FSBL_HEX="$(FSBL_HEX)" WEIGHTS_HEX="$(WEIGHTS_HEX)" \
	  PROG_CLI="$(PROG_CLI)" EXT_LOADER="$(EXT_LOADER)" \
	  ADDR_APP="$(ADDR_APP)" bash scripts/flash.sh

deploy: generate io-layout model-install build sign flash ## Full pipeline end-to-end

bench-cloud: ## Benchmark MODEL on ST's N6 board farm (needs cloud/.env credentials)
	@PY=$$( [ -x ../../.venv/bin/python ] && echo ../../.venv/bin/python || echo python3 ); \
	 $$PY cloud/dev_cloud_bench.py --model "$(MODEL)" --board STM32N6570-DK

validate-target: ## On-target numerical validation over UART (board wired to this PC)
	"$(STEDGEAI)" validate -m "$(MODEL)" --target stm32n6 \
	  --st-neural-art "$(NPU_PROFILE)" --mode target -d serial:921600

clean: ## Remove generated model output and signed binary
	rm -rf "$(GEN_OUT)" "$(APP_SIGNED)"
