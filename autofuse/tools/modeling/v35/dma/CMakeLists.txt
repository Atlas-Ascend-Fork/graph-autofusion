# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
project(att_dma_calibration LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT NPU_ARCH)
    message(FATAL_ERROR "Set NPU_ARCH to match the installed device")
endif()
get_filename_component(default_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(AUTOFUSE_ROOT "${default_root}" CACHE PATH "Graph-autofusion source checkout providing RegBase headers")
find_package(Threads REQUIRED)
function(add_dma_bench name source)
    add_executable(${name} "src/${source}")
    target_include_directories(${name} PRIVATE "${AUTOFUSE_ROOT}/autofuse/v35/ascendc/api_regbase")
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH};-O2>)
    target_link_libraries(${name} PRIVATE ascendcl ascendc_runtime tiling_api register platform unified_dlog dl m
        graph_base Threads::Threads)
endfunction()
add_dma_bench(att_dma_load_loop loop_load.asc)
add_dma_bench(att_dma_load_stability loop_load.asc)
target_compile_definitions(att_dma_load_stability PRIVATE ATT_DMA_LONG_WINDOW=1)
add_dma_bench(att_dma_load_steady steady_load.asc)
add_dma_bench(att_dma_store store.asc)
add_dma_bench(att_dma_load_nddma steady_load.asc)
target_compile_definitions(att_dma_load_nddma PRIVATE ATT_DMA_SHORT_CALLS=1)
add_dma_bench(att_dma_load_nddma_long steady_load.asc)
target_compile_definitions(att_dma_load_nddma_long PRIVATE ATT_DMA_SHORT_CALLS=2)
