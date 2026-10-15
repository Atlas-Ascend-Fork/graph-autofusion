# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

include_guard(GLOBAL)
get_filename_component(ASCENDSK_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(GLOB ASCENDSK_HOST_SOURCES CONFIGURE_DEPENDS
    "${ASCENDSK_ROOT_DIR}/src/aot/*.cpp"
    "${ASCENDSK_ROOT_DIR}/src/aot/common/*.cpp"
)

function(configure_ascendsk_host target)
    target_compile_options(${target} PRIVATE
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
    target_compile_definitions(${target} PRIVATE
        FUNC_VISIBILITY
        _GLIBCXX_USE_CXX11_ABI=0
        $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
    )
    target_include_directories(${target}
        PUBLIC ${ASCENDSK_ROOT_DIR}/include/super_kernel
        PRIVATE ${ASCENDSK_ROOT_DIR} ${ASCENDSK_ROOT_DIR}/src/aot
                ${ASCENDSK_ROOT_DIR}/src/aot/common
    )
    target_link_options(${target} PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )
endfunction()
