# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# Pure Python multistream wheel.  Building and installing it needs no CANN
# and no compiled library.  Running a plan does need the host's CANN
# Runtime shared library, which the wheel neither ships nor installs.
#
# The directory is also intentionally usable as a standalone CMake project:
# this is how the package's no-CANN build/install smoke test exercises it.  In
# the graph-autofusion super-project the parent supplies the Python interpreter
# and the product-relative install directory, so do not introduce a second
# dependency discovery path there.
cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(multistream LANGUAGES NONE)
endif()

if(NOT Python3_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()
if(NOT INSTALL_LIBRARY_DIR)
    set(INSTALL_LIBRARY_DIR lib)
endif()

# The package root is this directory, so glob the sub-packages explicitly:
# a recursive glob would also pick up the tests, the CI scripts and any
# generated build directory and rebuild the wheel whenever they change.
file(GLOB MULTISTREAM_PY_SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.py"
     "${CMAKE_CURRENT_SOURCE_DIR}/common/*.py"
     "${CMAKE_CURRENT_SOURCE_DIR}/minidag/*.py"
     "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/*.py")

# Configure the staging script with the source list so additions, removals and
# renames invalidate the wheel, even when the remaining files are unchanged.
set(MULTISTREAM_WHEEL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheel-source")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/stage_wheel.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/stage_wheel.cmake" @ONLY)

# ``python -m build`` is the canonical frontend used by graph-autofusion, but
# a released CANN build image may provide only pip (and an existing source
# ``build/`` directory can shadow the frontend as a namespace package).  Pick
# the frontend at configure time and retain a no-network pip fallback so the
# pure-Python package remains buildable in both environments.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('build.__main__') else 1)"
    RESULT_VARIABLE MULTISTREAM_BUILD_FRONTEND_STATUS
    OUTPUT_QUIET ERROR_QUIET)
if(MULTISTREAM_BUILD_FRONTEND_STATUS EQUAL 0)
    set(MULTISTREAM_WHEEL_COMMAND
        "${Python3_EXECUTABLE}" -m build --wheel --no-isolation --outdir "${CMAKE_CURRENT_BINARY_DIR}"
        "${MULTISTREAM_WHEEL_SOURCE_DIR}")
else()
    set(MULTISTREAM_WHEEL_COMMAND
        "${Python3_EXECUTABLE}" -m pip wheel "${MULTISTREAM_WHEEL_SOURCE_DIR}" --no-deps --no-build-isolation --no-index
        --wheel-dir "${CMAKE_CURRENT_BINARY_DIR}")
endif()
add_custom_target(multistream_whl ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/multistream-0.1.0-py3-none-any.whl)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/multistream-0.1.0-py3-none-any.whl
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/stage_wheel.cmake"
    COMMAND ${MULTISTREAM_WHEEL_COMMAND}
    DEPENDS ${MULTISTREAM_PY_SRC} pyproject.toml "${CMAKE_CURRENT_BINARY_DIR}/stage_wheel.cmake"
    # Build outside the checkout; each invocation gets a clean setuptools tree.
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/multistream-0.1.0-py3-none-any.whl ${INSTALL_OPTIONAL}
        DESTINATION ${INSTALL_LIBRARY_DIR}
        COMPONENT graph-autofusion)
