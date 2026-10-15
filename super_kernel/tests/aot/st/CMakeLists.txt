# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_library(ascendsk_st SHARED ${ASCENDSK_HOST_SOURCES})
configure_ascendsk_host(ascendsk_st)
target_link_libraries(ascendsk_st PRIVATE intf_llt_options super_kernel_aot_stub json)

file(GLOB ST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")
add_executable(super_kernel_aot_profiling_process profiling_process.cpp)
target_link_libraries(super_kernel_aot_profiling_process PRIVATE
    intf_llt_options ascendsk_st super_kernel_aot_stub
)
add_executable(super_kernel_aot_stest ${ST_SOURCES})
add_dependencies(super_kernel_aot_stest super_kernel_aot_profiling_process)
target_compile_definitions(super_kernel_aot_stest PRIVATE
    SK_PROFILING_PROCESS_PATH="$<TARGET_FILE:super_kernel_aot_profiling_process>"
)
target_link_libraries(super_kernel_aot_stest PRIVATE
    intf_llt_options GTest::gtest_main GTest::gtest ascendsk_st super_kernel_aot_stub json
)
run_llt_test(TARGET super_kernel_aot_stest TASK_NUM 1 COVERAGE_SUITE cpp_st)
