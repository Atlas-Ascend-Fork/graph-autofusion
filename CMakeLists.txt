# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16.0)
include(cmake/fetch_cann_cmake.cmake)
project(graph-autofusion)

init_cann_project()
add_cann_target_options()
if(NOT ASCEND_INSTALL_PATH AND DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_INSTALL_PATH "$ENV{ASCEND_INSTALL_PATH}" CACHE PATH "Ascend install path" FORCE)
endif()
if(NOT CANN_3RD_LIB_PATH AND DEFINED ENV{ASCEND_3RD_LIB_PATH})
    set(CANN_3RD_LIB_PATH "$ENV{ASCEND_3RD_LIB_PATH}" CACHE PATH "CANN third-party path" FORCE)
endif()
include(cmake/dependencies.cmake)
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")
SET(CMAKE_C_COMPILER_WORKS TRUE)
SET(CMAKE_CXX_COMPILER_WORKS TRUE)
set(AUTOFUSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/autofuse)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(STATUS "Build graph_autofusion with dependency cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build graph_autofusion with dependency cann pkgs installed" TRUE)
else()
    message(STATUS "Build graph_autofusion with cann source exist")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build graph_autofusion need cann source code" FALSE)
endif()

include(cmake/function.cmake)
include(version.cmake)
check_cann_pkg_build_deps("graph-autofusion")
add_cann_version_info_targets()

set(OPEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../open_source")
if(NOT EXISTS ${OPEN_SOURCE_DIR})
    message(STATUS "Build graph_autofusion with dependency communtiy cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build graph_autofusion with dependency communtiy cann pkgs installed" TRUE)
else()
    message(STATUS "Build graph_autofusion with dependency communtiy cann pkgs not installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build graph_autofusion with dependency communtiy cann pkgs not installed" FALSE)
endif()

set(INSTALL_LIBRARY_DIR graph_autofusion/lib)

add_subdirectory(super_kernel)
add_subdirectory(model_specialization)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/autofuse/CMakeLists.txt")
    option(BUILD_AUTOFUSE "Build autofuse backend modules" ON)
    if(BUILD_AUTOFUSE)
        add_subdirectory(autofuse)
    endif()
endif()

include(cmake/package.cmake)
