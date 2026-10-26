set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")

file(GLOB_RECURSE CODEGEN_TEST_SRCS CONFIGURE_DEPENDS "*.cpp")
add_library(test_codegen OBJECT
        ${CODEGEN_TEST_SRCS}
)
target_compile_definitions(test_codegen PRIVATE
        CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
        ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH}"
)
target_include_directories(test_codegen PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/att/generator
        ${CODE_ROOT_DIR}/v35
        ${CODE_ROOT_DIR}/tests/framework/share_graph/include
        ${CODE_ROOT_DIR}/tests/depends/runtime/src
        ${ASCEND_INSTALL_PATH}/x86_64-linux/include/acl/error_codes
)
target_link_libraries(test_codegen PRIVATE
    -Wl,--no-as-needed
    optimize
    codegen
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
