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
find_package(Threads REQUIRED)

set(MODEL_SPEC_COMPILE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_protocol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_process_pool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_process_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aclrtc_compile_worker.cpp
)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MODEL_SPEC_COMPILE_SOURCES})
set(MODEL_SPEC_BUILD_CONTENT "${CMAKE_BUILD_TYPE};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION}")
foreach(MODEL_SPEC_SOURCE IN LISTS MODEL_SPEC_COMPILE_SOURCES)
    file(SHA256 ${MODEL_SPEC_SOURCE} MODEL_SPEC_SOURCE_HASH)
    string(APPEND MODEL_SPEC_BUILD_CONTENT ";${MODEL_SPEC_SOURCE_HASH}")
endforeach()
string(SHA256 MODEL_SPEC_WORKER_BUILD_ID "${MODEL_SPEC_BUILD_CONTENT}")

add_library(model_spec_compile_common STATIC compile_protocol.cpp)
set_target_properties(model_spec_compile_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(model_spec_compile_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(model_spec_compile_common PUBLIC
    _GLIBCXX_USE_CXX11_ABI=0
    MODEL_SPEC_WORKER_BUILD_ID="${MODEL_SPEC_WORKER_BUILD_ID}"
)
target_compile_options(model_spec_compile_common PRIVATE
    -Wall -Wextra -Werror -fvisibility=hidden -fstack-protector-strong
)
target_link_libraries(model_spec_compile_common PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

add_executable(aclrtc_compile_worker aclrtc_compile_worker.cpp)
target_include_directories(aclrtc_compile_worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../super_kernel/src/aot)
target_link_libraries(aclrtc_compile_worker PRIVATE model_spec_compile_common acl_rtc unified_dlog)
target_compile_options(aclrtc_compile_worker PRIVATE -Wall -Wextra -Werror -fstack-protector-strong)
target_link_options(aclrtc_compile_worker PRIVATE -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)
set_target_properties(aclrtc_compile_worker PROPERTIES INSTALL_RPATH "$ORIGIN")

set(SPEC_SCOPE_TARGETS)
foreach(SPEC_KERNEL_ARCH IN ITEMS dav-2201 dav-3510)
    string(REPLACE "-" "_" SPEC_KERNEL_ARCH_SUFFIX ${SPEC_KERNEL_ARCH})
    set(SPEC_SCOPE_TARGET spec_scope_${SPEC_KERNEL_ARCH_SUFFIX})
    add_library(${SPEC_SCOPE_TARGET} STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/spec_scope_kernel.asc)
    target_compile_options(${SPEC_SCOPE_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SPEC_KERNEL_ARCH}>)
    target_compile_definitions(${SPEC_SCOPE_TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:SPEC_SCOPE_SYMBOL_SUFFIX=_${SPEC_KERNEL_ARCH_SUFFIX}>
    )
    set_property(TARGET ${SPEC_SCOPE_TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    list(APPEND SPEC_SCOPE_TARGETS ${SPEC_SCOPE_TARGET})
endforeach()

add_library(model_specialization SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/feature_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_compile_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/acl_spec_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_process_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aclmodel_specialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spec_scope_launch.cpp
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
        ${SPEC_SCOPE_TARGETS}
        model_spec_compile_common
        ${CMAKE_DL_LIBS}
        ascendc_runtime
        profapi
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

add_dependencies(model_specialization aclrtc_compile_worker)

# dlog 使用 __FILE__ 记录源码位置，此处去掉本目录源码的路径前缀。
if(CANN_FILE_REMAP_SUPPORTED)
    foreach(MODEL_SPEC_TARGET IN ITEMS model_specialization aclrtc_compile_worker model_spec_compile_common)
        target_compile_options(${MODEL_SPEC_TARGET} PRIVATE
            "-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/="
        )
    endforeach()
endif()

install(TARGETS model_specialization aclrtc_compile_worker
    LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64 COMPONENT graph-autofusion
    RUNTIME DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64 COMPONENT graph-autofusion
)
install(FILES aclmodel_specialization.h
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/model_specialization
    COMPONENT graph-autofusion
)
