cmake_minimum_required(VERSION 3.22.0)

include(${SdkRootDirPath}/cmake/extension/mcux.cmake)

project(lisennet_se LANGUAGES C CXX ASM PROJECT_BOARD_PORT_PATH ${board_root}/${board}/eiq_examples/lisennet_se)

include(${SdkRootDirPath}/CMakeLists.txt)
include(${SdkRootDirPath}/${board_root}/${board}/eiq_examples/lisennet_se/reconfig.cmake OPTIONAL)

# App sources are symlinks into the eco8-neaixt repo (deploy/rt595/app), the source of
# truth. The model (model_data.h + model_ops_micro.cpp) is added via cpu/ in the board
# overlay's reconfig.cmake; board init (BOARD_Init/pins) via the cm33/ port path.
mcux_add_include(INCLUDES .)

mcux_add_source(
  BASE_PATH ${SdkRootDirPath}
  SOURCES examples/eiq_examples/lisennet_se/main.cpp
          examples/eiq_examples/lisennet_se/model_se_stream.cpp
          examples/eiq_examples/lisennet_se/model_se_stream.h
          examples/eiq_examples/lisennet_se/model_io_layout.h
          examples/eiq_examples/lisennet_se/se_test_feats.h
  )

mcux_add_mdk_configuration(LD "--library_type=standardlib")
mcux_convert_binary(BINARY ${APPLICATION_BINARY_DIR}/${MCUX_SDK_PROJECT_NAME}.bin)
