# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)
project(att_vf_dependency LANGUAGES ASC CXX)
add_executable(att_vf_dependency probe.asc)
target_link_libraries(att_vf_dependency PRIVATE tiling_api register platform m dl)
target_compile_options(att_vf_dependency PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>)
