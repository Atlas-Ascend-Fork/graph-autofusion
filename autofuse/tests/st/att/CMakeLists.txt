set (CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set (CMAKE_C_FLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fsanitize=address -fsanitize=leak -fsanitize-recover=address")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=leak -fsanitize-recover=address")

message(STATUS "Using environment variable LD_LIBRARY_PATH: $ENV{LD_LIBRARY_PATH}")
message(STATUS "Using environment variable ASCEND_INSTALL_LIB_PATH: ${ASCEND_INSTALL_LIB_PATH}")
set(CMAKE_BUILD_TYPE Debug)

set(ATT_DIR ${CODE_ROOT_DIR}/att)
set(TC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase)

add_compile_definitions(ST_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
add_compile_definitions(TOP_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../../..\")

file(GLOB SOURCES
        ${TC_DIR}/matmul/*.cpp
        ${TC_DIR}/ffn/*.cpp
        ${TC_DIR}/att_st_test/att_test.cpp
        ${TC_DIR}/test_expr/*.cpp
        ${TC_DIR}/source_mirror/generator/solver_pass_gen/axes_reorder_solver_gen/*.cpp
        ${TC_DIR}/solver_pass_gen/general_solver_gen/*.cpp
        ${TC_DIR}/solver_pass/general_solver/*.cpp
        ${TC_DIR}/solver_pass_manager/*.cpp
        ${TC_DIR}/select_model/*.cpp
        ${TC_DIR}/stub/*.cpp
        ${TC_DIR}/utils_test/*.cpp
        ${TC_DIR}/gen_model_info/*.cpp
        ${TC_DIR}/gen_model_info/api_tiling_gen/*.cpp
        ${TC_DIR}/gen_model_info/api_tiling_gen/api/*.cpp
        ${TC_DIR}/test_log/*.cpp
        ${TC_DIR}/add_layer_norm/test_add_layer_norm.cpp
        ${TC_DIR}/add_layer_norm/add_layer_norm.cpp
        ${TC_DIR}/concat/*.cpp
        ${TC_DIR}/reduce_split_penalty/*.cpp
        ${TC_DIR}/e2e_st_test/*.cpp
        ${TC_DIR}/profiler_e2e/*.cpp
        ${TC_DIR}/e2e_brc_buf/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
)

add_executable(att_st
        ${SOURCES}
)
target_link_directories(att_st PRIVATE ${ASCEND_INSTALL_LIB_PATH})
target_include_directories(att_st BEFORE PRIVATE
        ${CODE_ROOT_DIR}/tests/st/common
        ${CODE_ROOT_DIR}/tests/st/att/testcase/scenario
        ${CODE_ROOT_DIR}/tests/depends/runtime/src
        ${CODE_ROOT_DIR}/tests/depends/common/inc
        ${ASCEND_INSTALL_PATH}/x86_64-linux/include/acl/error_codes
        ${CMAKE_SOURCE_DIR}/tests/st/common
        ${CMAKE_SOURCE_DIR}/tests/depends/runtime/src
        ${CMAKE_SOURCE_DIR}/tests/st/att/testcase/scenario
        ${CMAKE_SOURCE_DIR}
        ${TC_DIR}
        ${CMAKE_SOURCE_DIR}/tests/depends/common/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/att
        ${CODE_ROOT_DIR}/common
        ${ATT_DIR}/util
        ${ATT_DIR}/common_utils
        ${ATT_DIR}/generator
        ${ATT_DIR}/generator/preprocess
        ${ATT_DIR}/generator/solver_pass
        ${ATT_DIR}/generator/solver_pass_gen
        ${ATT_DIR}/generator/cache
        ${ATT_DIR}/gen_model_info
        ${ASCEND_INSTALL_PATH}/include
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
        ${CMAKE_BINARY_DIR}/ascir_builtin_ops
)

target_compile_options(att_st PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -fpermissive
        )
target_compile_definitions(att_st PRIVATE ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH}")

target_link_options(att_st PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(att_st
        GTest::gtest GTest::gtest_main pthread -lgcov
        symengine
        Boost::boost
        atrace
        att
        asc_slog_stub
        autofuse_runtime_stub
        aihac_ir
        ascgen_common
        ascir_builtin_ops
        aihac_symbolizer_af
        ascir_ops_headers
        aihac_ir_register
        graph_base_af
        graph_af
        error_manager
        c_sec
        json
        ascend_protobuf
        )
add_dependencies(att_st ascir_builtin_ops)

add_test(NAME att_st COMMAND att_st --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/att_st.xml)
set_tests_properties(att_st PROPERTIES LABELS "st;att_st")
