# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(AF_AUTOFUSE_MLIR_LIT_SOURCE_DIR ${MLIR_CODE_ROOT_DIR}/test/lit)
set(AF_AUTOFUSE_MLIR_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lit)

configure_lit_site_cfg(
    ${AF_AUTOFUSE_MLIR_LIT_SOURCE_DIR}/lit.site.cfg.py.in
    ${AF_AUTOFUSE_MLIR_LIT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${AF_AUTOFUSE_MLIR_LIT_SOURCE_DIR}/lit.cfg.py
)

if(ENABLE_AUTOFUSE_MLIR_TESTS)
    # Register an MLIR gtest executable, its CTest entry and a check-* driver.
    # Callers supply SOURCES/LIBS and any extra include directories or definitions.
    # Example:
    #   add_afir_gtest(AFIRSymbolicTest
    #     SOURCES ExpressionTest.cpp
    #     LIBS AFIRSymbolicSupport)
    # COMMENT overrides the driver message; LABELS extend the default mlir_ut label.
    function(add_afir_gtest name)
        cmake_parse_arguments(AFIR_GTEST "" "COMMENT" "SOURCES;LIBS;LABELS" ${ARGN})
        if(NOT AFIR_GTEST_SOURCES)
            message(FATAL_ERROR "${name}: SOURCES is required")
        endif()

        add_executable(${name} ${AFIR_GTEST_SOURCES})
        target_link_libraries(${name} PRIVATE
            ${AFIR_GTEST_LIBS}
            GTest::gtest
            GTest::gtest_main
        )

        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES
            LABELS "mlir_ut;${AFIR_GTEST_LABELS}"
            ENVIRONMENT "LD_LIBRARY_PATH=${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
        )

        set(_afir_gtest_check_name "check-${name}")
        string(TOLOWER "${_afir_gtest_check_name}" _afir_gtest_check_name)
        add_custom_target(${_afir_gtest_check_name}
            COMMAND ${CMAKE_COMMAND} -E env
                "LD_LIBRARY_PATH=${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
                $<TARGET_FILE:${name}>
            DEPENDS ${name}
            COMMENT "${AFIR_GTEST_COMMENT}"
        )
        # Register the driver in a global property keyed by the suite (the
        # test/ subdirectory it lives in, e.g. ut or st) so the aggregate
        # targets can pick up every registered test without per-directory
        # glue targets.
        file(RELATIVE_PATH _afir_gtest_rel "${MLIR_CODE_ROOT_DIR}/test" "${CMAKE_CURRENT_SOURCE_DIR}")
        string(REGEX MATCH "^[^/]*" _afir_gtest_suite "${_afir_gtest_rel}")
        set_property(GLOBAL APPEND PROPERTY
            "AFIR_GTEST_CHECK_TARGETS_${_afir_gtest_suite}" "${_afir_gtest_check_name}")
    endfunction()

    # Shared runtime library search path for MLIR C++ tests.  The devlib
    # directory is part of that search path because libruntime.so DT_NEEDs
    # libascend_hal.so (device driver stubs), which never ships next to it in
    # lib64.  CMake only emits -rpath-link for lib64, where libruntime itself
    # was found, so the linker needs the same directory added explicitly --
    # mirroring the devlib entries run_autofuse_test.sh sets for the legacy
    # test runner.
    set(AF_CANN_DEVLIB_DIR "${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib")
    string(CONCAT AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH
        "$<TARGET_FILE_DIR:aihac_codegen>:$<TARGET_FILE_DIR:graph_af>:$<TARGET_FILE_DIR:aihac_symbolizer_af>:"
        "$<TARGET_FILE_DIR:aihac_ir>:$<TARGET_FILE_DIR:aihac_ir_register>:$<TARGET_FILE_DIR:ascend_protobuf>:"
        "$ENV{ASCEND_HOME_PATH}/lib64:$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
    )
    if(EXISTS "${AF_CANN_DEVLIB_DIR}")
        # Link-time always resolves through the devlib stub: it is ABI-identical
        # to the real driver library, so this is equally correct on hosts with
        # an actual NPU and driver installed.
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${AF_CANN_DEVLIB_DIR}")
        # At run time prefer the real driver's libascend_hal.so when present:
        # this path is prepended to LD_LIBRARY_PATH, so the stub would shadow
        # the driver library sourced from set_env.sh and break device-touching
        # st/e2e tests.
        get_filename_component(AF_CANN_ROOT_DIR "${ASCEND_INSTALL_PATH}" DIRECTORY)
        if(NOT EXISTS "${AF_CANN_ROOT_DIR}/driver/lib64/libascend_hal.so")
            string(APPEND AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH ":${AF_CANN_DEVLIB_DIR}")
        endif()
    endif()

    # The MLIR-only test configuration intentionally leaves RUN_TEST unset, so
    # autofuse/tests (which normally defines autofuse_runtime_stub) is not in
    # the build graph. Keep a test-local stub available without changing the
    # production target graph or colliding with the full tests configuration.
    if(TARGET autofuse_runtime_stub)
        set(AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET autofuse_runtime_stub)
    else()
        add_library(autofuse_mlir_runtime_stub SHARED
            ${CODE_ROOT_DIR}/tests/depends/runtime/src/runtime_stub.cc
        )
        target_include_directories(autofuse_mlir_runtime_stub PRIVATE
            ${CODE_ROOT_DIR}/tests/depends/runtime/src
        )
        target_link_libraries(autofuse_mlir_runtime_stub
            PRIVATE c_sec
            PUBLIC runtime_headers
        )
        set(AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET autofuse_mlir_runtime_stub)
    endif()
    add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/ut" ut)
    add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/st" st)
    add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e" e2e)
    # probe 依赖 e2e 的 MlirE2eCpuGraphBuilder，因此在 e2e 后配置；不加入 check-*。
    add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/probe" probe)

    set(AF_AUTOFUSE_MLIR_TEST_DEPENDS af-afir-gen af-opt ascir-translate)

    add_lit_testsuite(check-lit-mlir
        "Running Autofuse MLIR smoke tests"
        ${AF_AUTOFUSE_MLIR_LIT_BINARY_DIR}
        DEPENDS ${AF_AUTOFUSE_MLIR_TEST_DEPENDS}
    )

    add_custom_target(check-lit
        DEPENDS check-lit-mlir
        COMMENT "Running MLIR lit tests"
    )

    add_custom_target(check-mlir
        DEPENDS check-ut check-st check-lit check-e2e
        COMMENT "Running all MLIR tests"
    )
endif()
