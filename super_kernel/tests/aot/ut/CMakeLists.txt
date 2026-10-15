# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16.0)
project(super_kernel_aot_ut)
set(CMAKE_CXX_STANDARD 17)
set(LOCAL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(super_kernel_aot_utest
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model_fixture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pattern_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_options_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_super_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_scope_verify.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_scope_split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_node_scope_tagging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scope_range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_lock_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_notify_wait_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_rt_metadata_stubs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_scope_launch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_dfx_exception_handler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_scope_postprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_resource_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_task_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_profiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_event_recorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_candidate_heap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_scope_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_dump_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_log_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sk_common.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_common.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_options_manager.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_graph.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_scope_launch.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_scope_verify.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_verify_graph.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_verify_scope_split.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_scope_split.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_scope_split_result_reporter.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_lock_detector.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_node.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_optimizer.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_task_builder.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_candidate_heap.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_log.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/super_kernel.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_dfx_exception_handler.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_resource_manager.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_scope_postprocess.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_event_recorder.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/common/pattern_common.cpp
)

target_include_directories(super_kernel_aot_utest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/common/
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/include/super_kernel/
)

target_compile_options(super_kernel_aot_utest PRIVATE
    -g
    -Werror
    -Wno-error=deprecated-declarations
    -Wno-unused-function
)

target_link_libraries(super_kernel_aot_utest PRIVATE
    $<BUILD_INTERFACE:intf_llt_ut>
    json
    super_kernel_aot_stub
)

run_llt_test(
    TARGET super_kernel_aot_utest
    TASK_NUM 1
)
