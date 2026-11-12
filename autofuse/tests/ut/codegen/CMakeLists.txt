# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

file(GLOB_RECURSE CODEGEN_TEST_SRCS CONFIGURE_DEPENDS "*.cpp")
add_library(test_codegen OBJECT
        ${CODEGEN_TEST_SRCS}
        ${CODE_ROOT_DIR}/codegen/codegen_tiling_utils.cpp
        ${CODE_ROOT_DIR}/tests/ut/att/testcase/gen_model_info/core/test_brc_buf_graph.cpp
        ${CODE_ROOT_DIR}/tests/ut/att/utils/graph_construct_utils.cpp
)
target_compile_definitions(test_codegen PRIVATE
        CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
        ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH}"
)
target_include_directories(test_codegen PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/att/generator
        ${CODE_ROOT_DIR}/att
        ${CODE_ROOT_DIR}/att/gen_model_info
        ${CODE_ROOT_DIR}/v35
        ${CODE_ROOT_DIR}/tests/framework/share_graph/include
        ${CODE_ROOT_DIR}/tests/depends/runtime/src
        ${ASCEND_INSTALL_PATH}/x86_64-linux/include/acl/error_codes
)
target_link_libraries(test_codegen PRIVATE
    -Wl,--no-as-needed
    optimize
    codegen
    att
    ascir_builtin_ops
    ascir_ops_headers
    aihac_ir
    graph_base_af
    graph_af
    share_graph
    ascir
    ascgen_common
    aihac_symbolizer_af
    json
    ascendc_api_extend
    GTest::gtest)
