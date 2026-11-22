set(INDUCTOR_MATMUL_ELEMWISE_WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
set(INDUCTOR_MATMUL_ELEMWISE_TILING_DEF ${INDUCTOR_MATMUL_ELEMWISE_WORKDIR}/autofuse_tiling_data.h)
set(INDUCTOR_MATMUL_ELEMWISE_HOST_CODE ${INDUCTOR_MATMUL_ELEMWISE_WORKDIR}/inductor_matmul_elemwise_host.cpp)
set(INDUCTOR_MATMUL_ELEMWISE_DEVICE_CODE ${INDUCTOR_MATMUL_ELEMWISE_WORKDIR}/inductor_matmul_elemwise_device.cpp)
set(INDUCTOR_MATMUL_ELEMWISE_OUTPUT_DIR ${INDUCTOR_MATMUL_ELEMWISE_WORKDIR}/split_compile_out)

set(PYAUTOFUSE_DIR ${CMAKE_BINARY_DIR}/autofuse/tests)
set(AUTOFUSE_PYTHON_DIR ${CODE_ROOT_DIR}/compiler/python)
set(INDUCTOR_MATMUL_ELEMWISE_RUNTIME_LD_LIBRARY_PATH
    "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64:$ENV{LD_LIBRARY_PATH}")
string(CONCAT INDUCTOR_MATMUL_ELEMWISE_KERNEL_SRC_LIST
    "${INDUCTOR_MATMUL_ELEMWISE_TILING_DEF}:"
    "${INDUCTOR_MATMUL_ELEMWISE_HOST_CODE}:"
    "${INDUCTOR_MATMUL_ELEMWISE_DEVICE_CODE}"
)

add_executable(inductor_matmul_elemwise_test_codegen inductor_matmul_elemwise_backend_generate.cpp)
target_link_options(inductor_matmul_elemwise_test_codegen PRIVATE -Wl,--no-as-needed)
target_link_libraries(inductor_matmul_elemwise_test_codegen
    atrace optimize att codegen common_stub share_graph ascgen_common
    aihac_symbolizer_af asc_slog_stub error_manager metadef json
    GTest::gtest GTest::gtest_main)
target_compile_definitions(inductor_matmul_elemwise_test_codegen PRIVATE
    ATT_SO_NAME="./libatt.so"
    KERNEL_SRC_LIST="${INDUCTOR_MATMUL_ELEMWISE_KERNEL_SRC_LIST}"
    INDUCTOR_MATMUL_ELEMWISE_DIAG_DIR="/tmp/opencode/diag"
)
add_test(NAME inductor_matmul_elemwise_test_codegen
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${INDUCTOR_MATMUL_ELEMWISE_RUNTIME_LD_LIBRARY_PATH}"
            $<TARGET_FILE:inductor_matmul_elemwise_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_matmul_elemwise_test_codegen.xml)
set_tests_properties(inductor_matmul_elemwise_test_codegen PROPERTIES
    LABELS "st;build_backend_test1;inductor_matmul_elemwise_test_codegen")

set(MATMUL_TILING_DATA_SEARCH_DIRS ${ASCEND_INSTALL_PATH}/opp)
foreach(OPP_ENV ASCEND_OPP_PATH ASCEND_CUSTOM_OPP_PATH)
    if(DEFINED ENV{${OPP_ENV}} AND NOT "$ENV{${OPP_ENV}}" STREQUAL "")
        string(REPLACE ":" ";" OPP_ENV_PATHS "$ENV{${OPP_ENV}}")
        list(APPEND MATMUL_TILING_DATA_SEARCH_DIRS ${OPP_ENV_PATHS})
    endif()
endforeach()

foreach(SEARCH_DIR ${MATMUL_TILING_DATA_SEARCH_DIRS})
    file(GLOB_RECURSE MATMUL_TILING_DATA_HEADERS_IN_DIR
        ${SEARCH_DIR}/**/arch35/mat_mul_tiling_data.h)
    list(APPEND MATMUL_TILING_DATA_HEADERS ${MATMUL_TILING_DATA_HEADERS_IN_DIR})
endforeach()
if(NOT MATMUL_TILING_DATA_HEADERS)
    message(WARNING
        "Skip inductor_matmul_elemwise_test_e2e: "
        "arch35/mat_mul_tiling_data.h not found in Ascend install or OPP paths")
    add_custom_target(inductor_matmul_elemwise_test_e2e)
    return()
endif()
list(GET MATMUL_TILING_DATA_HEADERS 0 MATMUL_TILING_DATA_HEADER)
get_filename_component(MATMUL_TILING_DATA_INCLUDE_DIR ${MATMUL_TILING_DATA_HEADER} DIRECTORY)
get_filename_component(MATMUL_TILING_DATA_INCLUDE_DIR ${MATMUL_TILING_DATA_INCLUDE_DIR} DIRECTORY)

file(MAKE_DIRECTORY ${INDUCTOR_MATMUL_ELEMWISE_OUTPUT_DIR})
add_custom_command(
    OUTPUT ${INDUCTOR_MATMUL_ELEMWISE_TILING_DEF} ${INDUCTOR_MATMUL_ELEMWISE_HOST_CODE} ${INDUCTOR_MATMUL_ELEMWISE_DEVICE_CODE}
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${INDUCTOR_MATMUL_ELEMWISE_RUNTIME_LD_LIBRARY_PATH}"
            $<TARGET_FILE:inductor_matmul_elemwise_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_matmul_elemwise_test_codegen.xml
    DEPENDS inductor_matmul_elemwise_test_codegen
    VERBATIM)
add_custom_target(inductor_matmul_elemwise_test_codegen_outputs
    DEPENDS ${INDUCTOR_MATMUL_ELEMWISE_TILING_DEF} ${INDUCTOR_MATMUL_ELEMWISE_HOST_CODE} ${INDUCTOR_MATMUL_ELEMWISE_DEVICE_CODE})
add_inductor_host_abi1_helper(inductor_matmul_elemwise_host_abi1_helper inductor_matmul_elemwise_test_codegen_outputs
    ${INDUCTOR_MATMUL_ELEMWISE_WORKDIR} EXTRA_INCLUDE_DIRS ${MATMUL_TILING_DATA_INCLUDE_DIR})
add_executable(inductor_matmul_elemwise_test_e2e test_e2e_inductor_matmul_elemwise_split_compile.cpp)
add_dependencies(inductor_matmul_elemwise_test_e2e inductor_matmul_elemwise_host_abi1_helper)
add_inductor_split_compile_test_runtime(inductor_matmul_elemwise_test_e2e)
target_link_libraries(inductor_matmul_elemwise_test_e2e dl GTest::gtest GTest::gtest_main unified_dlog)
target_include_directories(inductor_matmul_elemwise_test_e2e PRIVATE
    ${INDUCTOR_MATMUL_ELEMWISE_WORKDIR} ${MATMUL_TILING_DATA_INCLUDE_DIR})
target_compile_definitions(inductor_matmul_elemwise_test_e2e PRIVATE
    TILING_DEF_FILE="${INDUCTOR_MATMUL_ELEMWISE_TILING_DEF}"
    HOST_CODE_FILE="${INDUCTOR_MATMUL_ELEMWISE_HOST_CODE}"
    DEVICE_CODE_FILE="${INDUCTOR_MATMUL_ELEMWISE_DEVICE_CODE}"
    OUTPUT_DIR="${INDUCTOR_MATMUL_ELEMWISE_OUTPUT_DIR}"
    PYAUTOFUSE_DIR="${PYAUTOFUSE_DIR}"
    AUTOFUSE_PYTHON_DIR="${AUTOFUSE_PYTHON_DIR}"
    ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"
    HOST_HELPER_BIN="$<TARGET_FILE:inductor_matmul_elemwise_host_abi1_helper>"
)
add_test(NAME inductor_matmul_elemwise_test_e2e
    COMMAND inductor_matmul_elemwise_test_e2e --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_matmul_elemwise_test_e2e.xml)
set_tests_properties(inductor_matmul_elemwise_test_e2e PROPERTIES
    LABELS "st;build_backend_test2;inductor_matmul_elemwise_test_e2e"
    ENVIRONMENT "LD_LIBRARY_PATH=${INDUCTOR_MATMUL_ELEMWISE_RUNTIME_LD_LIBRARY_PATH}")
