# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
project(op_head_cost LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT NPU_ARCH)
    message(FATAL_ERROR "Set NPU_ARCH to match the installed device")
endif()
foreach(bench IN ITEMS bridge head)
    add_executable(${bench} ${bench}.asc)
    target_compile_options(${bench} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH};-O2>)
    target_link_libraries(${bench} PRIVATE ascendcl ascendc_runtime tiling_api register platform unified_dlog dl m graph_base)
endforeach()
