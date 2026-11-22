# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(model_specialization_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(MODEL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
find_library(MODEL_SPECIALIZATION_LIBRARY NAMES model_specialization
    HINTS $ENV{ASCEND_HOME_PATH}/lib64 ${MODEL_ROOT}/../build/model_specialization/src)
if(NOT MODEL_SPECIALIZATION_LIBRARY)
    message(FATAL_ERROR "Build model_specialization or set MODEL_SPECIALIZATION_LIBRARY to its full path")
endif()
set(OPP_VENDOR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../build/opp/vendors/customize" CACHE PATH "Custom operator vendor directory")
find_library(CUSTOM_OPAPI_LIBRARY NAMES cust_opapi HINTS ${OPP_VENDOR_PATH}/op_api/lib NO_DEFAULT_PATH)
if(NOT CUSTOM_OPAPI_LIBRARY)
    message(FATAL_ERROR "Build and install ../custom_op first, or set OPP_VENDOR_PATH")
endif()
add_executable(demo main.cpp)
target_compile_definitions(demo PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(demo PRIVATE -Wall -Wextra -Werror)
target_include_directories(demo PRIVATE
    ${MODEL_ROOT}/include/model_specialization
    $ENV{ASCEND_HOME_PATH}/include
    ${OPP_VENDOR_PATH}/op_api/include
)
target_link_directories(demo PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(demo PRIVATE ${MODEL_SPECIALIZATION_LIBRARY} ${CUSTOM_OPAPI_LIBRARY} nnopbase ascendcl acl_rt)
