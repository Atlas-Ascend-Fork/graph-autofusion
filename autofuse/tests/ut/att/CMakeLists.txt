set (CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set (CMAKE_C_FLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(ATT_DIR ${TOP_DIR}/att)

cmake_minimum_required(VERSION 3.14)
project (Att[CXX])
add_compile_options(-std=c++17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_compile_definitions(TOP_DIR=\"${TOP_DIR}\")
add_compile_definitions(UT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

file(GLOB SOURCES
        # common stub
        ${TOP_DIR}/tests/common/stub/stub_model_info.cpp
        ${TOP_DIR}/tests/common/stub/stub_matmul_modelinfo.cpp
        ${TOP_DIR}/tests/common/stub/stub_solver_model_info.cpp
        # generator/preprocess
        ${TOP_DIR}/tests/ut/att/testcase/preprocess/*.cpp
        # generator/solver_pass
        ${TOP_DIR}/tests/ut/att/testcase/solver_pass/general_solver/*.cpp
        # generator/solver_pass_gen
        ${TOP_DIR}/tests/ut/att/testcase/solver_pass_gen/axes_reorder_gen/*.cpp
        ${TOP_DIR}/tests/ut/att/testcase/solver_pass_gen/general_solver_gen/*.cpp
        ${TOP_DIR}/tests/ut/att/testcase/solver_pass_gen/manager/*.cpp
        # generator/core
        ${TOP_DIR}/tests/ut/att/testcase/generator/core/*.cpp
        # concat helpers/tests
        ${TOP_DIR}/tests/ut/att/testcase/concat/*.cpp
        # gen_model_info
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/*.cpp
        # gen_model_info/parser
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/parser/*.cpp
        # gen_model_info/expr_gen
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/expr_gen/*.cpp
        # gen_model_info/pass
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/pass/*.cpp
        # gen_model_info/utils
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/utils/*.cpp
        # gen_model_info/reuse_group_utils
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/reuse_group_utils/*.cpp
        # gen_model_info/api_perf_register
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/api_perf_register/*.cpp
        ${TOP_DIR}/autofuse/tests/v35/ut/att/gen_model_info/api_perf_register/test_reduce_min_max_api_perf_v2.cpp
        # gen_model_info/core
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/core/*.cpp
        # util
        ${TOP_DIR}/tests/ut/att/testcase/util/*.cpp
        # ascir
        ${TOP_DIR}/tests/ut/att/testcase/ascir/*.cpp
        # check_input
        ${TOP_DIR}/tests/ut/att/testcase/check_input/*.cpp
        ${TOP_DIR}/tests/ut/att/utils/*.cpp
    )

add_executable(att_ut
        ${SOURCES}
        )

target_link_directories(att_ut PRIVATE ${ASCEND_INSTALL_LIB_PATH})

target_include_directories(att_ut PRIVATE
        ${TOP_DIR}/tests/ut/att/testcase
        ${TOP_DIR}/tests/ut/att/utils
        ${TOP_DIR}/tests/common/att
        ${TOP_DIR}/tests/common/stub
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info
        ${TOP_DIR}/tests/ut/att/testcase/gen_model_info/core
        ${TOP_DIR}/
        ${TOP_DIR}
        ${TOP_DIR}/common
        ${TOP_DIR}/tests/depends/common/inc
        ${ATT_DIR}/util
        ${ATT_DIR}/dtree
        ${ATT_DIR}/common_utils
        ${ATT_DIR}/generator
        ${ATT_DIR}/generator/decision_tree
        ${ATT_DIR}/generator/preprocess
        ${ATT_DIR}/generator/solver_pass_gen
        ${ATT_DIR}/generator/decision_tree
        ${ATT_DIR}/gen_model_info
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/x86_64-linux/include/acl/error_codes
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
        ${CMAKE_BINARY_DIR}/autofuse/ascir_builtin_ops
        ${TOP_DIR}/tests/ut/att/testcase/ascir
        ${TOP_DIR}/tests/ut/att/utils
        )

target_compile_options(att_ut PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -fpermissive
        )

target_link_options(att_ut PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(att_ut PRIVATE
        GTest::gtest GTest::gmock GTest::gtest_main -lgcov pthread
        -Wl,--no-as-needed
        att
        asc_slog_stub
        autofuse_runtime_stub
        aihac_ir
        aihac_symbolizer_af
        ascgen_common
        ascir_builtin_ops
        aihac_ir_register
        ascir_ops_headers
        symengine
        Boost::boost
        graph_base_af
        error_manager
        metadef
        graph_af
        c_sec
        json
        ascend_protobuf
        )
add_dependencies(att_ut ascir_builtin_ops)

add_test(NAME att_ut_aggregate
    COMMAND $<TARGET_FILE:att_ut>
)
