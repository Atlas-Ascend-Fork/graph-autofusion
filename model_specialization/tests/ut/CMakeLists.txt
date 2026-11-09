# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16.0)
project(super_kernel_model_specialization_utest)

add_executable(model_specialization_utest
    ${CMAKE_CURRENT_SOURCE_DIR}/test_feature_manager.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/feature_manager.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/kernel_compile_features.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/common/pattern_common.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_log.cpp
)
add_dependencies(model_specialization_utest model_specialization)

target_include_directories(model_specialization_utest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/common
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/include/super_kernel
)

target_compile_options(model_specialization_utest PRIVATE
    -g
    -Werror
    -Wno-error=deprecated-declarations
    -Wno-unused-function
)

target_link_libraries(model_specialization_utest PRIVATE
    $<BUILD_INTERFACE:intf_llt_ut>
    GTest::gtest_main
    super_kernel_aot_stub
)

run_llt_test(
    TARGET model_specialization_utest
    TASK_NUM 1
)

add_executable(acl_spec_optimize_ut
    ${CMAKE_CURRENT_SOURCE_DIR}/test_acl_spec_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_spec_scope_launch.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/acl_spec_optimizer.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/compile_process_pool.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/aclmodel_specialization.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/spec_scope_launch.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/feature_manager.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/kernel_compile_features.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/common/pattern_common.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/sk_log.cpp
)

target_include_directories(acl_spec_optimize_ut PRIVATE
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/tests/stub
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot/common
    ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/include/super_kernel
)

target_compile_options(acl_spec_optimize_ut PRIVATE
    -g
    -Werror
    -Wno-error=deprecated-declarations
    -Wno-unused-function
    -include ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/tests/stub/spec_runtime.h
)

target_link_libraries(acl_spec_optimize_ut PRIVATE
    model_spec_compile_common
    ${CMAKE_DL_LIBS}
    Boost::filesystem
    $<BUILD_INTERFACE:intf_llt_ut>
    GTest::gtest_main
    super_kernel_aot_stub
)

set(MODEL_SPEC_TEST_TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp")
file(MAKE_DIRECTORY "${MODEL_SPEC_TEST_TMP_DIR}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/model_spec_test_env.sh"
    "export TMPDIR='${MODEL_SPEC_TEST_TMP_DIR}'\n")
run_llt_test(
    TARGET acl_spec_optimize_ut
    TASK_NUM 1
    ENV_FILE "${CMAKE_CURRENT_BINARY_DIR}/model_spec_test_env.sh"
)

add_dependencies(run_model_specialization_utest run_acl_spec_optimize_ut)

add_executable(compile_pool_fault_worker compile_pool_fault_worker.cpp)
target_link_libraries(compile_pool_fault_worker PRIVATE model_spec_compile_common)

add_executable(aclrtc_compile_worker_stub
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/aclrtc_compile_worker.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/tests/stub/worker_acl_rtc_stub.cpp
)
target_link_libraries(aclrtc_compile_worker_stub PRIVATE model_spec_compile_common acl_rt_headers)
set_target_properties(aclrtc_compile_worker_stub PROPERTIES OUTPUT_NAME aclrtc_compile_worker)
add_dependencies(acl_spec_optimize_ut aclrtc_compile_worker_stub)

add_executable(compile_process_ut
    test_compile_protocol.cpp
    test_compile_artifact.cpp
    test_compile_process_pool.cpp
    test_compile_worker.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/compile_process_pool.cpp
)
target_link_libraries(compile_process_ut PRIVATE model_spec_compile_common GTest::gtest_main GTest::gtest)
target_compile_options(compile_process_ut PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(compile_process_ut PRIVATE
    MODEL_SPEC_FAULT_WORKER_PATH="$<TARGET_FILE:compile_pool_fault_worker>"
    MODEL_SPEC_TEST_WORKER_PATH="$<TARGET_FILE:aclrtc_compile_worker_stub>"
)
add_dependencies(compile_process_ut compile_pool_fault_worker aclrtc_compile_worker_stub)
target_compile_definitions(compile_process_ut PRIVATE MODEL_SPEC_TEST_TMP_DIR="${MODEL_SPEC_TEST_TMP_DIR}")
run_llt_test(TARGET compile_process_ut TASK_NUM 1
    ENV_FILE "${CMAKE_CURRENT_BINARY_DIR}/model_spec_test_env.sh")
add_dependencies(run_model_specialization_utest run_compile_process_ut)
