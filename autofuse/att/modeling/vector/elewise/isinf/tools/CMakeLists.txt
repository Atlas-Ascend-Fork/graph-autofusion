# ----------------------------------------------------------------------------------------------------------
# This program is free software; you can redistribute it and/or modify it
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You can redistribute it and/or modify it under the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, cpu, sim")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(isinf_perf_bench LANGUAGES ASC CXX)

add_executable(isinf_perf_bench
    isinf_perf_bench.asc
)

target_link_libraries(isinf_perf_bench PRIVATE
    tiling_api
    register
    platform
    m
    dl
)

target_compile_options(isinf_perf_bench PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
