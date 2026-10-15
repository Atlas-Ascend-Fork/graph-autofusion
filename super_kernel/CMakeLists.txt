# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE SUPER_KERNEL_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

add_custom_target(superkernel_whl ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl
        COMMAND echo "Start to build superkernel wheel package"
        && cd ${CMAKE_CURRENT_SOURCE_DIR}
        && python3 -m build --wheel --no-isolation --outdir ${CMAKE_CURRENT_BINARY_DIR}
        && rm -rf build
        && echo "Build superkernel wheel package end"
        DEPENDS ${SUPER_KERNEL_SRC} pyproject.toml
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl ${INSTALL_OPTIONAL}
        DESTINATION ${INSTALL_LIBRARY_DIR}
        COMPONENT graph-autofusion
)

find_cann_package(ASC REQUIRED PATHS $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
enable_language(ASC)

set(GENERATED_SK_ENTRY_STUB_CPP "${CMAKE_CURRENT_BINARY_DIR}/kernel/sk_kernel_stub.cpp")
set_source_files_properties(${GENERATED_SK_ENTRY_STUB_CPP} PROPERTIES GENERATED TRUE)

if (NOT TARGET sk_kernel)
    add_subdirectory(kernel)
endif()

include(cmake/aot_host.cmake)

add_library(ascendsk SHARED
    ${ASCENDSK_HOST_SOURCES}
    ${GENERATED_SK_ENTRY_STUB_CPP}
)
add_dependencies(ascendsk generate_entry_stub)
configure_ascendsk_host(ascendsk)
target_include_directories(ascendsk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/aot
)
target_link_libraries(ascendsk PRIVATE c_sec_headers runtime_headers)
target_link_libraries(ascendsk PUBLIC sk_scope_dav_2201 sk_scope_dav_3510 acl_rt acl_rtc ascend_dump runtime error_manager c_sec unified_dlog ascendc_runtime profapi profimpl msprofiler json)

if(ENABLE_CPP_UTEST OR ENABLE_CPP_STEST)
    add_subdirectory(tests/aot)
endif()
