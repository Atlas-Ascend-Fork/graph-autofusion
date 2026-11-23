# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
project(att_gather_calibration LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT NPU_ARCH)
    message(FATAL_ERROR "Set NPU_ARCH to match the installed device")
endif()
get_filename_component(default_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(AUTOFUSE_ROOT "${default_root}" CACHE PATH "Graph-autofusion source checkout providing RegBase headers")
find_package(Threads REQUIRED)
add_executable(att_gather gather.asc)
target_include_directories(att_gather PRIVATE "${AUTOFUSE_ROOT}/autofuse/v35/ascendc/api_regbase")
target_compile_options(att_gather PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH};-O2>)
target_link_libraries(att_gather PRIVATE ascendcl ascendc_runtime tiling_api register platform unified_dlog dl m
    graph_base Threads::Threads)
