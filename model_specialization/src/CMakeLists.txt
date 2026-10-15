# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_cann_third_party(boost)

add_library(model_specialization SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/feature_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_compile_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/acl_spec_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aclmodel_specialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../super_kernel/src/aot/common/pattern_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../super_kernel/src/aot/sk_log.cpp
)

target_compile_options(model_specialization PRIVATE
    -std=c++17
    -fPIC
    -fvisibility=hidden
    -Werror
    -Wno-error=deprecated-declarations
    -fno-common
    -ftrapv
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -O0>
)

target_compile_definitions(model_specialization PRIVATE
    FUNC_VISIBILITY
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)

target_include_directories(model_specialization
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../super_kernel/include/super_kernel>
        $<INSTALL_INTERFACE:${CMAKE_SYSTEM_PROCESSOR}-linux/include/model_specialization>
        $<INSTALL_INTERFACE:${CMAKE_SYSTEM_PROCESSOR}-linux/include/super_kernel>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../super_kernel/src/aot
        ${CMAKE_CURRENT_SOURCE_DIR}/../../super_kernel/src/aot/common
)

target_link_libraries(model_specialization
    PRIVATE
        Boost::filesystem
        c_sec_headers
        runtime_headers
    PUBLIC
        acl_rt
        acl_rtc
        runtime
        error_manager
        c_sec
        unified_dlog
)

target_link_options(model_specialization PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)
