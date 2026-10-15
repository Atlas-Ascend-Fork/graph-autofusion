# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
add_library(super_kernel_aot_stub SHARED
    aclrt_stubs.cpp rt_stub.cpp ut_common_stubs.cpp model_fixture.cpp exception_fixture.cpp profiling_fixture.cpp
)
target_include_directories(super_kernel_aot_stub PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/acl
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/dump
    ${CMAKE_CURRENT_SOURCE_DIR}/base
)
target_include_directories(super_kernel_aot_stub PRIVATE ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot)
target_link_libraries(super_kernel_aot_stub PUBLIC intf_llt_options)
