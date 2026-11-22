set(OPTIMIZE_ROOT_DIR ${CODE_ROOT_DIR}/optimize)

file(GLOB_RECURSE OPTIMIZE_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER OPTIMIZE_TEST_FILES EXCLUDE REGEX "autofuse_backend_stub.cpp")

add_executable(optimize_ut
        test.cpp
        ${OPTIMIZE_TEST_FILES}
        ${CODE_ROOT_DIR}/common/autofuse_backend_spec_api.cpp
)

target_include_directories(optimize_ut PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/..
        ${CODE_ROOT_DIR}/tests/ut/optimize
        ${CODE_ROOT_DIR}/tests/depends/runtime/src
        ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph
        ${CODE_ROOT_DIR}/autofuse
        ${CODE_ROOT_DIR}/inc
        ${ASCEND_INSTALL_PATH}/x86_64-linux/include/acl/error_codes
)

set(ASCGEN_DT_COMPILE_OPTION
        -O0
        -g
        --coverage -fprofile-arcs -ftest-coverage
#        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
)

set(ASCGEN_COMMON_LINK_OPTION
#        -fsanitize=address -fsanitize=leak -fsanitize-recover=address
        -lrt -ldl -lgcov
)

target_compile_options(optimize_ut PRIVATE ${ASCGEN_DT_COMPILE_OPTION}
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal
        #        -Werror
        -Wno-subobject-linkage
)

target_link_libraries(optimize_ut ${ASCGEN_COMMON_LINK_OPTION}
        -Wl,--no-as-needed
        optimize
        codegen
        ascir
        aihac_ir
        aihac_ir_register
        ascir_builtin_ops
        ascir_ops_headers
        ascgen_common
        asc_slog_stub
        autofuse_runtime_stub
        error_manager
        graph_af
        metadef
        e2e
        easy_asc_graph
        GTest::gtest
        pthread
)

add_test(NAME optimize_ut_aggregate
    COMMAND $<TARGET_FILE:optimize_ut>
)
