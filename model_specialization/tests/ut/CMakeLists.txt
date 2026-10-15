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
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/acl_spec_optimizer.cpp
    ${GRAPH_AUTOFUSION_ROOT_DIR}/model_specialization/src/aclmodel_specialization.cpp
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
    Boost::filesystem
    $<BUILD_INTERFACE:intf_llt_ut>
    GTest::gtest_main
    super_kernel_aot_stub
)

run_llt_test(
    TARGET acl_spec_optimize_ut
    TASK_NUM 1
)

add_dependencies(run_model_specialization_utest run_acl_spec_optimize_ut)
