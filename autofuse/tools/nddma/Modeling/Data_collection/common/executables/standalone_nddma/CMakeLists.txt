# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)

project(nddma_data_collection)

set(ASC_SRC_FILE nddma_data_collection.asc)
set(NDDMA_STAGE "2" CACHE STRING "NDDMA sensitivity binary id: 1 dtype/dim, 2 bytes/shape, 3 input stride, 4 output stride, 5 alignment, 6 broadcast")
set(NPU_ARCH "$ENV{NPU_ARCH}" CACHE STRING "Ascend NPU architecture passed to ASC compiler")
if(NOT DEFINED CMAKE_ASC_STANDARD OR "${CMAKE_ASC_STANDARD}" STREQUAL "")
    set(CMAKE_ASC_STANDARD 17 CACHE STRING "ASC language standard" FORCE)
endif()
set(CMAKE_ASC_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(NDDMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../" ABSOLUTE)
get_filename_component(AUTOFUSE_ROOT "${NDDMA_ROOT}/../.." ABSOLUTE)
set(REGBASE_INCLUDE_DIR
    "${AUTOFUSE_ROOT}/v35/ascendc/api_regbase"
    CACHE PATH "Directory containing datacopy_nddma.h"
)

if(NOT EXISTS "${REGBASE_INCLUDE_DIR}/datacopy_nddma.h")
    message(FATAL_ERROR
        "datacopy_nddma.h not found in ${REGBASE_INCLUDE_DIR}. "
        "Set -DREGBASE_INCLUDE_DIR to the directory containing this header.")
endif()
message(STATUS "NDDMA API include directory: ${REGBASE_INCLUDE_DIR}")

if("${RUN_MODE}" STREQUAL "cpu")
    find_package(tikicpulib REQUIRED)
    set(NDDMA_CPU_TARGET "" CACHE STRING "Imported tikicpulib target for the selected v35 CPU simulator")
    if(NOT TARGET "${NDDMA_CPU_TARGET}")
        message(FATAL_ERROR "Set -DNDDMA_CPU_TARGET to an available tikicpulib target for your v35 platform")
    endif()
    set_source_files_properties(${ASC_SRC_FILE} PROPERTIES LANGUAGE CXX)
else()
    if("${NPU_ARCH}" STREQUAL "")
        message(FATAL_ERROR "NPU_ARCH is not configured. For a matching v35 device, run export NPU_ARCH=dav-3510 or configure with -DNPU_ARCH=dav-3510. Replace dav-3510 with your actual device architecture.")
    endif()
    find_package(ASC REQUIRED)
    enable_language(ASC CXX)
endif()

add_executable(demo_nddma
    ${ASC_SRC_FILE}
)

target_include_directories(demo_nddma PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${REGBASE_INCLUDE_DIR}
)

target_compile_definitions(demo_nddma PRIVATE
    NDDMA_STAGE=${NDDMA_STAGE}
)

target_compile_options(demo_nddma PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:-g -O0>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-x c++>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall>>
)

target_link_directories(demo_nddma PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:$ENV{ASCEND_HOME_PATH}/x86_64-linux/simulator/${SOC_VERSION}/lib>>
)

target_link_libraries(demo_nddma PRIVATE
    tiling_api
    register
    platform
    m
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:npu_drv>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:cpudebug_acl_stub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:${NDDMA_CPU_TARGET}>>
)
