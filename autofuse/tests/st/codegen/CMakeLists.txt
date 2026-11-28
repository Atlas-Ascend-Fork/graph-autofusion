# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

add_executable(codegen_st
        test.cpp
        test_codegen_infershape.cpp
        test_codegen_graph_check.cpp
        test_codegen_extend_conv2d.cpp
        test_codegen_tbuf_merge.cpp
        # SIMT 融合组（hoist 点 primary 选择 / fused 调用形态）codegen 用例由 UT 维护，
        # ST 复用同一组用例，保证 ST 增量覆盖率覆盖 codegen_kernel_loop.cpp 的 hoist 路径。
        ${CODE_ROOT_DIR}/tests/ut/codegen/test_codegen_indirect_load_fused_group.cpp
)
target_compile_definitions(codegen_st PRIVATE
        CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
        ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH}"
)
target_link_libraries(codegen_st PRIVATE
    -Wl,--no-as-needed
    atrace
    optimize
    codegen
    aihac_ir
    aihac_ir_register
    graph_base_af
    graph_af
    metadef
    ascir
    ascgen_common
    aihac_symbolizer_af
    json
    ascendc_api_extend
    pyautofuse
    Python3::Python
    GTest::gtest)

add_test(NAME codegen_st COMMAND codegen_st --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/codegen_st.xml)
set_tests_properties(codegen_st PROPERTIES LABELS "st;codegen_st")

add_executable(pgo_collector_st
        ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../ut/codegen/test_pgo_collector.cpp
)
target_compile_definitions(pgo_collector_st PRIVATE
        PGO_PROFILER_STUB_PATH="$<TARGET_FILE:pgo_profiler_stub>"
)
target_link_libraries(pgo_collector_st PRIVATE
        -Wl,--no-as-needed
        codegen
        GTest::gtest
        ${CMAKE_DL_LIBS}
)
add_dependencies(pgo_collector_st pgo_profiler_stub)
add_dependencies(codegen_st pgo_collector_st)
add_test(NAME pgo_collector_st
    COMMAND pgo_collector_st --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/pgo_collector_st.xml
)
set_tests_properties(pgo_collector_st PROPERTIES LABELS "st;codegen_st")

add_test(NAME pgo_timestamp_abi_st
    COMMAND ${Python3_EXECUTABLE} -m pytest
        ${CMAKE_CURRENT_SOURCE_DIR}/../python/test_pgo_timestamp_abi.py -q
)
set_tests_properties(pgo_timestamp_abi_st PROPERTIES
    LABELS "st;codegen_st"
    ENVIRONMENT "CXX=${CMAKE_CXX_COMPILER}"
)

add_test(NAME kernel_tool_st_aggregate
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/kernel_tool/test_kernel
)
set_tests_properties(kernel_tool_st_aggregate PROPERTIES
    LABELS "st;kernel_tool_st"
)
