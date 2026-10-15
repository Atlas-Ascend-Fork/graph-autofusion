# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16)
project(super_kernel_aclgraph)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)




if(ENABLE_CPP_UTEST OR ENABLE_CPP_STEST)
    set(GRAPH_AUTOFUSION_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
    set(CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/module
        ${CMAKE_MODULE_PATH}
    )

    include(cmake/config.cmake)
    include(cmake/intf.cmake)
    include(cmake/func.cmake)
    add_subdirectory(depends)
    if(ENABLE_CPP_UTEST)
        add_subdirectory(ut)
    endif()
    if(ENABLE_CPP_STEST)
        add_subdirectory(st)
    endif()
endif()
