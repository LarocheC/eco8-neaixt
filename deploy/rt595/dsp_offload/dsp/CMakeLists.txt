# DSP (HiFi4) target for the LiSenNet SE offload.
#
# Modeled on the SDK's eiq_examples/tflm_kws_hifi4/hifi4 + dsp_examples/audio_demo_bm/dsp.
# The decisive line is `include(.../eiq_examples/hifi4.cmake)` with core_id=hifi4: it links
# middleware/eiq/tensorflow-lite/lib/hifi4/xcc/rt685s/libtflm.a — the xa_nnlib HiFi4 int8
# kernels. Reuses the M33 app's model driver + op resolver VERBATIM (single source of truth);
# only the linked kernel library differs.
#
# Build on the TOOLCHAIN BOX (needs xt-clang + the registered RT500 HiFi4 core config):
#   west build -b evkmimxrt595 --sysbuild <this dir> -- -Dcore_id=hifi4
# (exact invocation per the SDK's dsp_examples flow; see README_DSP_OFFLOAD.md).

cmake_minimum_required(VERSION 3.22.0)
include(${SdkRootDirPath}/cmake/extension/mcux.cmake)
project(lisennet_se_hifi4 LANGUAGES C CXX ASM
        PROJECT_BOARD_PORT_PATH ${board_root}/${board}/dsp_examples/lisennet_se)
include(${SdkRootDirPath}/CMakeLists.txt)

# --- our sources ---------------------------------------------------------------------
set(SE_APP ${CMAKE_CURRENT_LIST_DIR}/../../app)     # deploy/rt595/app  (portable SE driver)
mcux_add_include(
  INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}                        # dsp_main.cpp, se_ipc_place.c
    ${CMAKE_CURRENT_LIST_DIR}/..                     # se_ipc.h
    ${SE_APP}                                        # model_se_stream.h, model_io_layout.h
)
mcux_add_source(
  SOURCES
    dsp_main.cpp
    se_ipc_place.c
    # --- reused UNCHANGED from the M33 app: the driver + op resolver are identical; the
    #     HiFi4 speedup is purely the linked libtflm.a (see README) ---
    ${SE_APP}/model_se_stream.cpp
    ${SE_APP}/model_ops_micro.cpp
    # --- int8 model (regenerate these two from host_out/relu6deep_streaming_int8.tflite
    #     with host/gen_model_data.py + host/gen_io_layout.py; see README) ---
    ${SE_APP}/model_data.h
    ${SE_APP}/model_io_layout.h
)

# HiFi4 TFLM binary kernel library + fft2d (same components the kws_hifi4 example uses).
# These come from prj.conf; the core-specific cmake wires the prebuilt lib.
include(${SdkRootDirPath}/${board_root}/${board}/dsp_examples/lisennet_se/${core_id}/reconfig.cmake OPTIONAL)
include(${SdkRootDirPath}/examples/dsp_examples/${core_id}.cmake OPTIONAL)
include(${SdkRootDirPath}/examples/eiq_examples/${core_id}.cmake)
