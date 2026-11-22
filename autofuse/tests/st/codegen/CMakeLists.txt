set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

add_executable(codegen_st
        test.cpp
        test_codegen_infershape.cpp
        test_codegen_graph_check.cpp
        test_codegen_extend_conv2d.cpp
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
