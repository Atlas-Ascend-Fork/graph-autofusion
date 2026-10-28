cmake_minimum_required(VERSION 3.16.1)
project(autofuse)

option(ENABLE_DEVICE_VALIDATION "Register host device validation contract tests" ON)
option(ENABLE_DEVICE_VALIDATION_RUNNER "Register runner-backed validation tests" OFF)
option(ENABLE_DEVICE_VALIDATION_REAL_CODEGEN "Register real-codegen validation ST" OFF)
option(ENABLE_DEVICE_VALIDATION_HARDWARE "Register real-device validation ST" OFF)

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-all -Wl,-z,now")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-all -Wl,-z,now")

set(CODE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ASCEND_ROOT ${ASCEND_INSTALL_PATH}) # e.g. /usr/local/Ascend/ascend-toolkit/latest

set(ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH})

if (ENABLE_OPEN_SRC)
    if (NOT DEFINED ASCEND_3RD_LIB_PATH)
        set(ASCEND_3RD_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/output/third_party)
    endif ()

    if (NOT EXISTS "${ASCEND_3RD_LIB_PATH}")
        message(FATAL_ERROR "Error: Directory \"${ASCEND_3RD_LIB_PATH}\" does not exist!\n"
                "Please using build_third_party.sh compile first.")
    endif ()
    #    include(../../cmake/test_utils.cmake)
    set(GE_METADEF_DIR ${CODE_ROOT_DIR}/graph_metadef)
    set(GE_METADEF_INC_DIR ${CODE_ROOT_DIR}/inc/graph_metadef)
else ()
    set(GE_METADEF_DIR ${TOP_DIR}/air/graph_metadef)
    set(GE_METADEF_INC_DIR ${TOP_DIR}/air/inc/graph_metadef)
endif ()

link_directories(${ASCEND_ROOT}/lib64)
include_directories(${GE_METADEF_INC_DIR})
include_directories(${GE_METADEF_INC_DIR}/graph)
include_directories(${GE_METADEF_INC_DIR}/inc/graph)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/ascendc_ir_core)
include_directories(${GE_METADEF_INC_DIR}/graph/ascendc_ir/utils)


include_directories(${GE_METADEF_INC_DIR}/graph/debug)
include_directories(${GE_METADEF_INC_DIR}/graph/utils)
include_directories(${GE_METADEF_DIR}/graph)
include_directories(${GE_METADEF_DIR})
include_directories(${GE_METADEF_DIR}/../tests/graph_metadef/ut)
include_directories(${CODE_ROOT_DIR}/common)
include_directories(${CODE_ROOT_DIR}/ascendc/api)
include_directories(${CODE_ROOT_DIR}/compiler/py_module)
include_directories(${CODE_ROOT_DIR}/ascir/meta)
include_directories(${CODE_ROOT_DIR}/att)
include_directories(${CODE_ROOT_DIR}/inc)
include_directories(${CODE_ROOT_DIR}/inc/common)
include_directories(${ASCEND_ROOT}/include)
include_directories(${ASCEND_ROOT}/include/experiment)
include_directories(${ASCEND_ROOT}/include/experiment/runtime)
include_directories(${ASCEND_ROOT}/include/experiment/msprof)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos_af)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc/base)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc/profiling)



find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
if (ENABLE_OPEN_SRC)
    add_cann_third_party(json)
    add_cann_third_party(gtest)
    foreach(gtest_target GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
        if (TARGET ${gtest_target})
            set_property(TARGET ${gtest_target} APPEND PROPERTY INTERFACE_LINK_LIBRARIES pthread)
        endif ()
    endforeach()
    add_cann_third_party(boost)
    add_cann_third_party(symengine)
    add_cann_third_party(protobuf)
    find_cann_package(unified_dlog)
    find_cann_package(mmpa)
    find_cann_package(atrace)
    find_cann_package(metadef)
endif ()

if (DEFINED unified_dlog_INCLUDE_DIR)
    get_filename_component(UNIFIED_DLOG_PKG_INC_DIR "${unified_dlog_INCLUDE_DIR}" DIRECTORY)
    include_directories(${UNIFIED_DLOG_PKG_INC_DIR})
endif ()
if (DEFINED runtime_SHARED_LIBRARY)
    get_filename_component(ASCEND_FOUND_LIB_DIR "${runtime_SHARED_LIBRARY}" DIRECTORY)
    link_directories(${ASCEND_FOUND_LIB_DIR})
endif ()
if (DEFINED runtime_INCLUDE_DIR)
    include_directories(${runtime_INCLUDE_DIR}/profiling)
endif ()

include(GoogleTest)
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

include(${CODE_ROOT_DIR}/cmake/test_funcs.cmake)

# For open source build: provide stub interface targets that replace Ascend internal build targets
if (ENABLE_OPEN_SRC)
    set(AIR_CODE_DIR ${CODE_ROOT_DIR})
    set(RUNTIME_PKG_INC ${ASCEND_INSTALL_PATH}/pkg_inc)

    set(SLOG_INC_DIR ${ASCEND_INSTALL_PATH}/pkg_inc/base)

    if (NOT TARGET intf_llt_pub)
        add_library(intf_llt_pub INTERFACE)
        target_compile_options(intf_llt_pub INTERFACE -g -fPIC -w)
        target_include_directories(intf_llt_pub INTERFACE
            ${RUNTIME_PKG_INC}
            ${RUNTIME_PKG_INC}/runtime
            ${RUNTIME_PKG_INC}/runtime/runtime
            ${RUNTIME_PKG_INC}/profiling
            ${SLOG_INC_DIR}
            ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/acl/error_codes
            ${CMAKE_CURRENT_SOURCE_DIR}/depends/securec
        )
    endif()

    if (NOT TARGET unified_dlog_headers)
        add_library(unified_dlog_headers INTERFACE)
        target_include_directories(unified_dlog_headers INTERFACE
            ${RUNTIME_PKG_INC}
            ${SLOG_INC_DIR}
        )
    endif()

    if (NOT TARGET atrace_headers)
        add_library(atrace_headers INTERFACE)
        target_include_directories(atrace_headers INTERFACE
            ${RUNTIME_PKG_INC}
            ${RUNTIME_PKG_INC}/trace
        )
    endif()

    set(RUNTIME_ACL_ERROR_CODES_DIR "${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/acl/error_codes")

    if (NOT TARGET runtime_headers)
        add_library(runtime_headers INTERFACE)
        target_include_directories(runtime_headers INTERFACE
            ${RUNTIME_PKG_INC}
            ${RUNTIME_PKG_INC}/runtime
            ${RUNTIME_PKG_INC}/runtime/runtime
            ${RUNTIME_PKG_INC}/profiling
            ${SLOG_INC_DIR}
            ${RUNTIME_ACL_ERROR_CODES_DIR}
        )
    else()
        set_property(TARGET runtime_headers APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${RUNTIME_ACL_ERROR_CODES_DIR})
    endif()

    if (NOT TARGET c_sec)
        # Use stub securec.h for open source builds
        add_library(c_sec INTERFACE)
        target_include_directories(c_sec INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/depends/securec
        )
    endif()

    if (NOT TARGET metadef_headers)
        add_library(metadef_headers INTERFACE)
    endif()
endif()

# Community CANN packages (e.g. 9.3.0) no longer ship libintf_llt_pub; fall back to an
# interface stub whenever no real target exists so test stubs keep linking.
if (NOT TARGET intf_llt_pub)
    add_library(intf_llt_pub INTERFACE)
    target_compile_options(intf_llt_pub INTERFACE -g -fPIC -w)
endif()

file(GLOB_RECURSE ASCIR_META_TEST_SRC CONFIGURE_DEPENDS "${CODE_ROOT_DIR}/ascir/meta/*.cpp")
file(GLOB_RECURSE ASCIR_BUILTIN_OPS_TEST_SRC CONFIGURE_DEPENDS
        "${CODE_ROOT_DIR}/ascir/generator/*.cpp"
        "${CODE_ROOT_DIR}/ascir/reg_func/*.cpp"
)
set_source_files_properties(${ASCIR_META_TEST_SRC}
        DIRECTORY ${CODE_ROOT_DIR}/ascir/meta
        PROPERTIES COMPILE_FLAGS "-fvisibility=default"
)
set_source_files_properties(${ASCIR_BUILTIN_OPS_TEST_SRC}
        DIRECTORY ${CODE_ROOT_DIR}/ascir/generator
        PROPERTIES COMPILE_FLAGS "-fvisibility=default"
)

add_subdirectory(depends/slog)
add_subdirectory(depends/trace)
add_subdirectory(depends/runtime)
add_subdirectory(depends/common)
add_subdirectory(framework)
add_subdirectory(st)
add_subdirectory(ut)

include(CMakePrintHelpers)
message(STATUS "Variables in project:")
cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(INSTALL_BASE_DIR)
cmake_print_variables(ENABLE_OPEN_SRC)

####################################common############################################
set(COMMON_DIR ${CODE_ROOT_DIR}/common)
file(GLOB_RECURSE CODEGEN_COMMON_SRCS CONFIGURE_DEPENDS "${COMMON_DIR}/*.cpp" "${COMMON_DIR}/*.cc")
list(REMOVE_ITEM CODEGEN_COMMON_SRCS
        "${COMMON_DIR}/autofuse_backend_spec_api.cpp"
)
add_library(ascgen_common SHARED
        ${CODEGEN_COMMON_SRCS}
)

target_include_directories(ascgen_common PUBLIC
        ${CODE_ROOT_DIR}
        ${COMMON_DIR}
        ${CMAKE_BINARY_DIR}/ascir_builtin_ops
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
)

if (ENABLE_OPEN_SRC)
    target_link_libraries(ascgen_common PUBLIC
            -lgcov
            graph_af
            ascir
            aihac_symbolizer_af
            mmpa
            c_sec
            unified_dlog
            autofuse_runtime_stub
            atrace
            ascend_protobuf
            aihac_ir_register
    )
else ()
    target_link_libraries(ascgen_common PUBLIC
            $<BUILD_INTERFACE:intf_llt_pub>
            $<BUILD_INTERFACE:mmpa_headers>
            $<BUILD_INTERFACE:msprof_headers>
            $<BUILD_INTERFACE:runtime_headers>
            $<BUILD_INTERFACE:atrace_headers>
            mmpa
            c_sec
            unified_dlog
            atrace
            graph_af
            json
            ascend_protobuf
            aihac_ir_register
    )
endif ()

#####################################att###########################################
SET(ENABLE_SYMENGINE "on")

set(ATT_DIR ${CODE_ROOT_DIR}/att)

file(GLOB SOURCES
    ${ATT_DIR}/generator/*.cpp
    ${ATT_DIR}/generator/cache/*.cpp
    ${ATT_DIR}/generator/preprocess/*.cpp
    ${ATT_DIR}/generator/solver_pass/*.cpp
    ${ATT_DIR}/generator/solver_pass/axes_reorder_solver_code/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/axes_reorder_solver/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/general_solver/*.cpp
    ${ATT_DIR}/generator/solver_pass_gen/golden_solver/*.cpp
    ${ATT_DIR}/generator/extra_info_gen/*.cpp
    ${ATT_DIR}/generator/generator_utils/*.cpp
    ${ATT_DIR}/generator/tiling_data_gen/*.cpp
    ${ATT_DIR}/generator/tiling_option_generator/*.cpp
    ${ATT_DIR}/util/*.cpp
    ${ATT_DIR}/gen_model_info/axes_priority_reg/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/v1/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/v2/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/utils/*.cpp
    ${ATT_DIR}/gen_model_info/api_perf_register/*.cpp
    ${ATT_DIR}/gen_model_info/expr_gen/*.cpp
    ${ATT_DIR}/gen_model_info/parser/*.cpp
    ${ATT_DIR}/gen_model_info/pass/*.cpp
    ${ATT_DIR}/gen_model_info/reuse_group_utils/*.cpp
    ${ATT_DIR}/gen_model_info/utils/*.cpp
    ${ATT_DIR}/gen_model_info/*.cpp
    ${ATT_DIR}/gen_model_info/axes_priority_reg/*.cpp
    ${ATT_DIR}/gen_model_info/api_tiling_gen/*.cpp
    ${ATT_DIR}/gen_model_info/api_tiling_gen/api/*.cpp
    ${ATT_DIR}/gen_model_info/sketch_gen/*.cpp
    ${ATT_DIR}/gen_tiling_impl.cpp
    )


list(REMOVE_ITEM SOURCES ${ATT_DIR}/gen_model_info/api_perf_register/soc_perf_config_data.cpp)
list(APPEND SOURCES ${CODE_ROOT_DIR}/tests/common/att/soc_perf_test_config.cpp)

add_library(att SHARED
    ${SOURCES}
)

add_dependencies(att ascir_builtin_ops_header)
target_include_directories(att PRIVATE
        ${ATT_DIR}/../
        ${ATT_DIR}
        ${ATT_DIR}/../common
        ${ATT_DIR}/../inc/
        ${ATT_DIR}/base
        ${ATT_DIR}/util
        ${ATT_DIR}/common_utils
        ${ATT_DIR}/generator
        ${ATT_DIR}/generator/preprocess
        ${ATT_DIR}/generator/solver_pass
        ${ATT_DIR}/generator/solver_pass_gen
        ${ATT_DIR}/ascir
        ${ATT_DIR}/ascir/generator
        ${ATT_DIR}/gen_model_info
        ${ASCEND_INSTALL_PATH}/include
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
        ${CMAKE_BINARY_DIR}/ascir_builtin_ops
        )

add_definitions(-DFORMULAS_PERF)

target_compile_options(att PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -fPIC
        -fno-common
        -Wextra
        -Wfloat-equal
        -Werror
)

target_link_options(att PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(att PRIVATE
        -lgcov
        symengine
        Boost::boost
        optimize
        json
        aihac_ir
        aihac_symbolizer_af
        graph_base_af
        graph_af
        error_manager
        ascend_protobuf
        ge_log_utils
        ascir_builtin_ops
        autofuse_runtime_stub
        ascgen_common
)

if (DEFINED ENABLE_OPEN_SRC)
    target_link_libraries(att PUBLIC
            $<BUILD_INTERFACE:mmpa_headers>
            $<BUILD_INTERFACE:runtime_headers>
            mmpa
            unified_dlog
            atrace
    )
endif ()

########################################codegen########################################
set(CODEGEN_DIR ${CODE_ROOT_DIR}/codegen)
file(GLOB_RECURSE CODEGEN_SOURCES "${CODEGEN_DIR}/*.cpp")
add_library(codegen SHARED
        ${CODEGEN_SOURCES}
)

set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH};${CODE_ROOT_DIR}/att/build")

target_compile_options(codegen PRIVATE
    -Wfloat-equal -Wextra
)

target_include_directories(codegen PUBLIC
    ${CODEGEN_DIR}
    ${CODEGEN_DIR}/api_call
    ${CODE_ROOT_DIR}/common
    ${CODE_ROOT_DIR}/att
    ${CMAKE_CURRENT_SOURCE_DIR}/depends/runtime/src
)

if (ENABLE_OPEN_SRC)
    target_link_libraries(codegen PUBLIC
        -lgcov
        graph_af
        ascir
        ascgen_common
        aihac_symbolizer_af
        error_manager
        att
        json
        ascendc_api_extend
    )
else ()
    target_link_libraries(codegen PUBLIC
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:atrace_headers>
        mmpa
        c_sec
        unified_dlog
        atrace
        ascend_protobuf
        graph_af
        json
        ascendc_api_extend
    )
endif ()

######################################optimize##########################################
set(OPTIMIZE_DIR ${CODE_ROOT_DIR}/optimize)
file(GLOB_RECURSE SOURCES "${OPTIMIZE_DIR}/*.cpp")

add_library(optimize SHARED
        ${SOURCES}
)

target_include_directories(optimize PUBLIC
        ${OPTIMIZE_DIR}
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/autofuse
)

# Host compiler optimization is unnecessary for this test-only library.
# Keep cold, coverage-enabled builds from spending memory on -O2 optimization.
target_compile_options(optimize PRIVATE -fPIC -Werror -fno-common -Wextra -Wfloat-equal -O0 -Wall)

if (ENABLE_OPEN_SRC)
    target_link_libraries(optimize PUBLIC
            -lgcov
            ascir
            ascgen_common
            graph_af
            ascir
            unified_dlog
            autofuse_runtime_stub
            error_manager
            atrace
            graph_base_af
            ascir_builtin_ops
    )
else ()
    target_link_libraries(optimize PUBLIC
            graph_af
            unified_dlog
            atrace
            graph_base_af
            c_sec
    )
endif ()

####################################pyautofuse############################################
set(PYAUTOFUSE_DIR ${CODE_ROOT_DIR}/compiler/py_module/)

Python3_add_library(pyautofuse MODULE
        ${PYAUTOFUSE_DIR}/pyascir.cpp
        ${PYAUTOFUSE_DIR}/pyascir_types.cpp
        ${PYAUTOFUSE_DIR}/pyascir_common_utils.cpp
        ${PYAUTOFUSE_DIR}/pyautofuse.cpp
)
target_compile_options(pyautofuse PRIVATE -Wno-error=variadic-macros)
target_include_directories(pyautofuse PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/optimize
        ${CODE_ROOT_DIR}/codegen
        ${CODE_ROOT_DIR}/autofuse
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
)
target_compile_definitions(pyautofuse PRIVATE
        google=ascend_private
)
if (ENABLE_OPEN_SRC)
    target_link_libraries(pyautofuse PUBLIC
            -Wl,--no-as-needed
            $<BUILD_INTERFACE:ascir_ops_headers>
            $<BUILD_INTERFACE:runtime_headers>
            atrace
            ascgen_common
            att
            codegen
            ascir_builtin_ops
            aihac_ir
            ascend_protobuf
            json
            graph_af
            graph_base_af
            autofuse_runtime_stub
    )
else ()
    target_link_libraries(pyautofuse PRIVATE
            $<BUILD_INTERFACE:atrace_headers>
            $<BUILD_INTERFACE:ascir_ops_headers>
            atrace_share
            ascgen_common
            att
            codegen
            aihac_ir
            ascend_protobuf
            json
            graph_af
            graph_base_af
            unified_dlog
    )
endif ()
if (TARGET pyautofuse_runtime_error_tiling)
    add_dependencies(pyautofuse pyautofuse_runtime_error_tiling)
endif ()
if (ENABLE_OPEN_SRC)
    target_compile_options(att PRIVATE -g --coverage -fprofile-arcs -ftest-coverage)
    target_compile_options(optimize PRIVATE -g --coverage -fprofile-arcs -ftest-coverage)
    target_compile_options(ascgen_common PUBLIC -g --coverage -fprofile-arcs -ftest-coverage)
    target_compile_options(codegen PUBLIC -g --coverage -fprofile-arcs -ftest-coverage)
    target_compile_options(pyautofuse PUBLIC -g --coverage -fprofile-arcs -ftest-coverage)
endif ()

foreach(autofuse_test_target IN ITEMS ascgen_common att codegen optimize pyautofuse test_common)
    target_compile_options(${autofuse_test_target} PRIVATE
        -include${CODE_ROOT_DIR}/inc/graph_metadef/graph/tensor_af.h
    )
    target_compile_definitions(${autofuse_test_target} PRIVATE
        INC_EXTERNAL_GRAPH_TENSOR_H_
        INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
    )
endforeach()

if (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/v35")
    add_subdirectory(v35)
endif()
