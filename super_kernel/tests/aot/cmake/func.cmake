# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
function(run_llt_test)
    cmake_parse_arguments(LLT "" "TARGET;TASK_NUM;ENV_FILE;COVERAGE_SUITE" "" ${ARGN})

    set(_llt_run_target run_${LLT_TARGET})

    if (ENABLE_ASAN)
        execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_MAJOR)
        string(REGEX MATCHALL "[0-9]+" GCC_MAJOR ${GCC_MAJOR})
        if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
            set(LD_PRELOAD_ "/usr/lib/gcc/x86_64-linux-gnu/${GCC_MAJOR}/libasan.so:/usr/lib/gcc/x86_64-linux-gnu/${GCC_MAJOR}/libstdc++.so")
        else()
            set(LD_PRELOAD_ "/usr/lib/gcc/aarch64-linux-gnu/${GCC_MAJOR}/libasan.so:/usr/lib/gcc/aarch64-linux-gnu/${GCC_MAJOR}/libstdc++.so")
        endif()
        # 谨慎修改 ASAN_OPTIONS_ 取值, 当前出现 ASAN 告警会使 UT 失败.
        set(ASAN_OPTIONS_ "detect_leaks=0:halt_on_error=0")
        if(LLT_ENV_FILE)
            set(ENV_FILE_PATH ${LLT_ENV_FILE})
        else()
            set(ENV_FILE_PATH "__EMPTY__")
        endif()
        add_custom_target(
                ${_llt_run_target}
                COMMAND bash ${RUN_LLT_BINARY} "${ENV_FILE_PATH}" "$ENV{LD_LIBRARY_PATH}" "${LD_PRELOAD_}" "${ASAN_OPTIONS_}" $<TARGET_FILE:${LLT_TARGET}> ${GTEST_FILTER}
                DEPENDS ${LLT_TARGET}
                COMMENT "Run ${LLT_TARGET} with asan"
        )
    else()
        if(LLT_ENV_FILE)
            set(ENV_FILE_PATH ${LLT_ENV_FILE})
        else()
            set(ENV_FILE_PATH "__EMPTY__")
        endif()

        add_custom_target(
            ${_llt_run_target}
            COMMAND bash ${RUN_LLT_BINARY} "${ENV_FILE_PATH}" "$ENV{LD_LIBRARY_PATH}" "__EMPTY__" "__EMPTY__" $<TARGET_FILE:${LLT_TARGET}> ${GTEST_FILTER}
            DEPENDS ${LLT_TARGET}
            COMMENT "Run ${LLT_TARGET}"
        )
    endif()

    if(ENABLE_GCOV)
        set(_collect_coverage_data_target collect_coverage_data)
        if(NOT LLT_COVERAGE_SUITE)
            set(LLT_COVERAGE_SUITE cpp_ut)
        else()
            set(_collect_coverage_data_target collect_coverage_data_${LLT_COVERAGE_SUITE})
        endif()

        get_filename_component(_ops_builtin_bin_path ${CMAKE_BINARY_DIR} DIRECTORY)
        set(_cov_capture_dir ${_ops_builtin_bin_path})
        set(_cov_source_filter)
        if(LLT_COVERAGE_SUITE STREQUAL "cpp_st")
            set(_cov_capture_dir ${CMAKE_CURRENT_BINARY_DIR})
            set(_cov_source_filter ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/src/aot)
            get_filename_component(_cov_source_filter "${_cov_source_filter}" ABSOLUTE)
        endif()
        set(_cov_report ${GRAPH_AUTOFUSION_ROOT_DIR}/super_kernel/coverage/${LLT_COVERAGE_SUITE})
        set(_cov_html ${_cov_report}/html)
        set(_cov_data ${_cov_report}/coverage.info)

        if (NOT TARGET ${_collect_coverage_data_target})
            add_custom_target(${_collect_coverage_data_target}
                    COMMAND bash ${GENERATE_CPP_COV} ${_ops_builtin_bin_path} ${_cov_data} ${_cov_html} $ENV{ASCEND_HOME_PATH} ${_cov_capture_dir} ${_cov_source_filter}
                    COMMENT "Run collect coverage data"
            )
        endif()

        add_dependencies(${_collect_coverage_data_target} ${_llt_run_target})
    endif()
endfunction(run_llt_test)

function(run_python_llt_test)
    cmake_parse_arguments(PYTHON "" "TARGET;SRC_FILES_DIR;TEST_FILES_DIR;EXPORT_PYTHONPATH;COVERAGERC_DIR;ENV_FILE;ENV_PARAMS;PYVERSION;PRE_TEST_COMMAND;TASK_NUM" "DEPENDS" ${ARGN})
    add_custom_target(${PYTHON_TARGET} ALL DEPENDS ${CMAKE_INSTALL_PREFIX}/${PYTHON_TARGET}.timestamp)

    if(NOT DEFINED PYTHON_PYVERSION)
        set(PYTHON_PYVERSION python3)
    endif()

    if((NOT DEFINED PYTHON_EXPORT_PYTHONPATH) OR (PYTHON_EXPORT_PYTHONPATH STREQUAL ""))
        set(PY_EXPORT_PYTHONPATH \"\")
    else()
        set(PY_EXPORT_PYTHONPATH ${PYTHON_EXPORT_PYTHONPATH})
    endif()

    if((NOT DEFINED PYTHON_COVERAGERC_DIR) OR (PYTHON_COVERAGERC_DIR STREQUAL ""))
        set(PY_COVERAGERC_DIR \"\")
    else()
        set(PY_COVERAGERC_DIR ${PYTHON_COVERAGERC_DIR})
    endif()

    if((NOT DEFINED PYTHON_ENV_PARAMS) OR (PYTHON_ENV_PARAMS STREQUAL ""))
        set(PY_ENV_PARAMS \"\")
    else()
        set(PY_ENV_PARAMS ${PYTHON_ENV_PARAMS})
    endif()

    if((NOT DEFINED PYTHON_ENV_FILE) OR (PYTHON_ENV_FILE STREQUAL ""))
        set(PY_ENV_FILE \"\")
    else()
        set(PY_ENV_FILE ${PYTHON_ENV_FILE})
    endif()

    if((NOT DEFINED PYTHON_PRE_TEST_COMMAND) OR (PYTHON_PRE_TEST_COMMAND STREQUAL ""))
        set(PY_PRE_TEST_COMMAND "echo>null")
    else()
        set(PY_PRE_TEST_COMMAND ${PYTHON_PRE_TEST_COMMAND})
    endif()

    if((NOT DEFINED PYTHON_SRC_FILES_DIR) OR (PYTHON_SRC_FILES_DIR STREQUAL ""))
        message(FATAL_ERROR "NOT DEFINED PYTHON_SRC_FILES_DIR")
    else()
        set(PY_SRC_FILES_DIR ${PYTHON_SRC_FILES_DIR})
        string(REPLACE "," ";" PYTHON_SRC_FILES_DIR ${PY_SRC_FILES_DIR})
    endif()

    if((NOT DEFINED PYTHON_TEST_FILES_DIR) OR (PYTHON_TEST_FILES_DIR STREQUAL ""))
        message(FATAL_ERROR "NOT DEFINED PYTHON_TEST_FILES_DIR")
    else()
        set(PY_TEST_FILES_DIR ${PYTHON_TEST_FILES_DIR})
        string(REPLACE "," ";" PYTHON_TEST_FILES_DIR ${PY_TEST_FILES_DIR})
    endif()

    if((NOT DEFINED PYTHON_TARGET) OR (PYTHON_TARGET STREQUAL ""))
        message(FATAL_ERROR "NOT DEFINED PYTHON_TARGET")
    endif()

    if((NOT DEFINED LLT_RUN_MOD) OR (LLT_RUN_MOD STREQUAL ""))
        set(LLT_RUN_MOD single)
    endif()

    if(NOT DEFINED PYTHON_TASK_NUM)
        set(PYTHON_TASK_NUM 8)
    endif()
    if(DEFINED SKIP_EXECUTE)
        set(LLT_EXECUTE_COMMAND echo "skip execute ${PYTHON_TARGET}" )
    else()
        set(LLT_EXECUTE_COMMAND bash ${ASCENDC_TOOLS_ROOT_DIR}/tests/cmake/tools/python_llt_run_and_check.sh ${CMAKE_INSTALL_PREFIX} ${PYTHON_TARGET}
        ${PY_SRC_FILES_DIR} ${PY_TEST_FILES_DIR} ${PY_ENV_FILE} ${PY_ENV_PARAMS}
        ${PY_EXPORT_PYTHONPATH} ${PY_COVERAGERC_DIR} ${PYTHON_PYVERSION} "${LLT_RUN_MOD}" "${PYTHON_TASK_NUM}")
    endif()
    add_custom_command(
        OUTPUT ${CMAKE_INSTALL_PREFIX}/${PYTHON_TARGET}.timestamp
        COMMAND echo "PRE_TEST_COMMAND: ${PY_PRE_TEST_COMMAND}"
        COMMAND bash -c "${PY_PRE_TEST_COMMAND}"
        COMMAND echo "execute python_llt_run_and_check.sh starttime:"
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/${PACKAGE}_report
        COMMAND ${LLT_EXECUTE_COMMAND}
        COMMAND echo "execute python_llt_run_and_check.sh end."
        COMMAND echo "build python ${PACKAGE} test successfully"
        COMMAND touch ${CMAKE_INSTALL_PREFIX}/${PYTHON_TARGET}.timestamp
        DEPENDS ${PYTHON_SRC_FILES_DIR} ${PYTHON_TEST_FILES_DIR} ${PYTHON_ENV_FILE} ${PYTHON_DEPENDS}
        WORKING_DIRECTORY ${ASCENDC_TOOLS_ROOT_DIR}
    )

endfunction(run_python_llt_test)
