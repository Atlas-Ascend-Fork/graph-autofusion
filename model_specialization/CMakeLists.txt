# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(super_kernel_model_specialization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(src)

if(ENABLE_CPP_UTEST OR ENABLE_CPP_STEST)
    set(GRAPH_AUTOFUSION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
    set(RUN_LLT_BINARY ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/tests/aot/cmake/tools/run_llt_binary.sh)
    set(GENERATE_CPP_COV ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/tests/aot/cmake/tools/generate_cpp_cov.sh)
    set(CMAKE_MODULE_PATH
        ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/tests/aot/cmake/module
        ${CMAKE_MODULE_PATH}
    )

    if(NOT TARGET GTest::gtest)
        include(${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/tests/aot/cmake/config.cmake)
    endif()
    if(NOT TARGET intf_llt_options)
        include(${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/tests/aot/cmake/intf.cmake)
    endif()
    if(NOT COMMAND run_llt_test)
        include(${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/tests/aot/cmake/func.cmake)
    endif()
    if(NOT TARGET super_kernel_aot_stub)
        add_subdirectory(
            ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/tests/aot/depends
            ${CMAKE_CURRENT_BINARY_DIR}/super_kernel_aot_depends
        )
    endif()
    if(ENABLE_CPP_UTEST)
        add_subdirectory(tests/ut)
    endif()
    if(ENABLE_CPP_STEST)
        add_subdirectory(tests/st)
    endif()
endif()
