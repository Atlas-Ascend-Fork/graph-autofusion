# Runtime library search path used by every backend codegen/e2e test executable.
# Defined at file scope so sweep aggregation can reuse it for its variant drivers.
set(BACKEND_E2E_RUNTIME_LD_LIBRARY_PATH
    "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64:$ENV{LD_LIBRARY_PATH}")

function(do_backend_e2e_st_test)
    set(one_value_arg
        WORKDIR # Workdir
        )
    set(mul_value_arg
        CODEGEN # Codegen library source file
        KERNEL_SRC # Kernel source file that codegen will generate
        TEST_SRC # Test case source file
        TILING_KEY
        )

    set(flag_arg
        NO_TEST # 仅构建 codegen/e2e 目标，不注册 ctest 用例（供 sweep 聚合复用）
        )

    set(TEST_NAME ${ARGV0})
    cmake_parse_arguments(PARSE_ARGV 1 ARG "${flag_arg}" "${one_value_arg}" "${mul_value_arg}")

    if(NOT ARG_TILING_KEY)
        set(ARG_TILING_KEY 0)
    endif()

    message(STATUS "ARG_WORKDIR=${ARG_WORKDIR}")

    foreach(file ${ARG_KERNEL_SRC})
        list(APPEND KERNEL_SRC "${ARG_WORKDIR}/${file}")
    endforeach()
    set_source_files_properties(${KERNEL_SRC} PROPERTIES GENERATED TRUE)

    set(E2E_ST1_GENERATOR_EXE_NAME ${TEST_NAME}_codegen_v2)
    set(E2E_ST2_EXE_KERNEL_EXE_NAME ${TEST_NAME}_e2e_v2)

    add_executable(${E2E_ST1_GENERATOR_EXE_NAME} ${ARG_CODEGEN})
    target_include_directories(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
        ${CODE_ROOT_DIR}/tests/st/common)
    target_link_options(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
            -Wl,--no-as-needed
    )
    target_include_directories(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
            ${CODE_ROOT_DIR}/tests/st/common
            ${CODE_ROOT_DIR}/tests/depends/runtime/src
    )
    target_link_libraries(${E2E_ST1_GENERATOR_EXE_NAME}
                          easy_asc_graph
                          atrace
                          asc_slog_stub
                          optimize
                          att
                          codegen
                          common_stub
                          share_graph
                          ascgen_common
                          aihac_symbolizer_af
                          error_manager
                          metadef
                          json
                          GTest::gtest
                          GTest::gtest_main
                          autofuse_runtime_stub)


    list (JOIN ARG_KERNEL_SRC ":" KERNEL_SRC_LIST)
    message(STATUS "KERNEL_SRC_LIST = ${KERNEL_SRC_LIST}")

    target_compile_definitions(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
       ATT_SO_NAME=\"./libatt.so\"
       KERNEL_SRC_LIST=\"${KERNEL_SRC_LIST}\"
    )

    set(BACKEND_RUNTIME_LD_LIBRARY_PATH "${BACKEND_E2E_RUNTIME_LD_LIBRARY_PATH}")

    if(NOT ARG_NO_TEST)
    add_test(NAME ${E2E_ST1_GENERATOR_EXE_NAME}
             COMMAND ${CMAKE_COMMAND} -E env
                     "LD_LIBRARY_PATH=${BACKEND_RUNTIME_LD_LIBRARY_PATH}"
                     $<TARGET_FILE:${E2E_ST1_GENERATOR_EXE_NAME}>
                     --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/v35/st/${E2E_ST1_GENERATOR_EXE_NAME}.xml)
    set_tests_properties(${E2E_ST1_GENERATOR_EXE_NAME} PROPERTIES LABELS "st;build_backend_test1;${E2E_ST1_GENERATOR_EXE_NAME}")
    endif()

    add_custom_target(${TEST_NAME}_generated_sources_v2
                      COMMAND ${CMAKE_COMMAND} -E env
                              "LD_LIBRARY_PATH=${BACKEND_RUNTIME_LD_LIBRARY_PATH}"
                              $<TARGET_FILE:${E2E_ST1_GENERATOR_EXE_NAME}>
                      WORKING_DIRECTORY ${ARG_WORKDIR}
                      BYPRODUCTS ${KERNEL_SRC}
                      DEPENDS ${E2E_ST1_GENERATOR_EXE_NAME})

    add_executable(${E2E_ST2_EXE_KERNEL_EXE_NAME} ${KERNEL_SRC} ${ARG_TEST_SRC})
    add_dependencies(${E2E_ST2_EXE_KERNEL_EXE_NAME} ${TEST_NAME}_generated_sources_v2)
    target_include_directories(${E2E_ST2_EXE_KERNEL_EXE_NAME} PRIVATE ${ARG_WORKDIR})
    target_link_libraries(${E2E_ST2_EXE_KERNEL_EXE_NAME} tikicpulib_ascend950pr_9599 GTest::gtest GTest::gtest_main)
    target_link_libraries(${E2E_ST2_EXE_KERNEL_EXE_NAME} unified_dlog)
    target_compile_options(${E2E_ST2_EXE_KERNEL_EXE_NAME} PRIVATE -DAUTO_FUSE_DEVICE=1  -DTILING_KEY_VAR=${ARG_TILING_KEY})
    #gtest_discover_tests(${E2E_ST2_EXE_KERNEL_EXE_NAME})
    if(NOT ARG_NO_TEST)
    add_test(NAME ${E2E_ST2_EXE_KERNEL_EXE_NAME} COMMAND ${E2E_ST2_EXE_KERNEL_EXE_NAME} --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/v35/st/${E2E_ST2_EXE_KERNEL_EXE_NAME}.xml)
    set_tests_properties(${E2E_ST2_EXE_KERNEL_EXE_NAME} PROPERTIES LABELS "st;build_backend_test2;${E2E_ST2_EXE_KERNEL_EXE_NAME}")
    endif()
endfunction()

macro(backend_e2e_st_test)
    do_backend_e2e_st_test(${ARGV}  WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro()
