# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# LLVM must be installed before the main project's configure-time find_package.
cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/../fetch_cann_cmake.cmake")
project(autofuse_llvm21 LANGUAGES C CXX)
init_cann_project()
if(NOT EXISTS "${CANN_CMAKE_DIR}/third_party/llvm-21.cmake")
    message(FATAL_ERROR
        "Selected cann-cmake has no llvm-21.cmake: ${CANN_CMAKE_DIR}. "
        "Update the offline cann-cmake directory/archive to the LLVM 21 fork before retrying.")
endif()
add_cann_third_party(llvm-21)
if(NOT LLVM21_INSTALL_PATH)
    message(FATAL_ERROR "cann-cmake did not define LLVM21_INSTALL_PATH")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/llvm21-install-path.txt" "${LLVM21_INSTALL_PATH}\n")
