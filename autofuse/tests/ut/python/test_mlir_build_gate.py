# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the real CMake routing without requiring CANN or LLVM packages."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("mlir_enabled", [False, True])
@pytest.mark.parametrize("run_test", [0, 1])
@pytest.mark.parametrize("processor", ["x86_64", "aarch64"])
def test_mlir_adapters_follow_build_gate(tmp_path, mlir_enabled, run_test, processor):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is required")
    driver = tmp_path / "CMakeLists.txt"
    trace = tmp_path / "trace.txt"
    # Package-path resolution validates the config-file locations before the
    # bridge target is configured. The driver stubs find_package(), so empty
    # markers are sufficient and no installed LLVM/MLIR package is required.
    llvm_build = tmp_path / "llvm-build"
    mlir_cmake = llvm_build / "lib" / "cmake" / "mlir"
    llvm_cmake = llvm_build / "lib" / "cmake" / "llvm"
    mlir_cmake.mkdir(parents=True)
    llvm_cmake.mkdir(parents=True)
    (mlir_cmake / "MLIRConfig.cmake").touch()
    (llvm_cmake / "LLVMConfig.cmake").touch()
    # Keep CMake's condition evaluation and the actual parent/adapter/test files.
    # Stub unrelated subtrees and build commands; this checks routing, not C++ compilation.
    no_op_commands = (
        "add_compile_options add_definitions include_directories link_directories "
        "target_compile_definitions target_include_directories "
        "target_sources "
        "add_dependencies install enable_testing Python3_add_library"
    ).split()
    driver.write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(mlir_gate NONE)\n"
        + "\n".join(f"function({command})\nendfunction()" for command in no_op_commands)
        + r"""
function(project)
endfunction()
function(find_package)
  file(APPEND "${TRACE}" "package;${ARGV}\n")
endfunction()
function(add_cann_third_party)
endfunction()
function(find_cann_package)
endfunction()
function(add_library)
  file(APPEND "${TRACE}" "library;${ARGV0}\n")
endfunction()
function(target_link_libraries)
  file(APPEND "${TRACE}" "link;${ARGV}\n")
endfunction()
function(target_compile_options)
  file(APPEND "${TRACE}" "compile_options;${ARGV}\n")
endfunction()
function(target_link_options)
  file(APPEND "${TRACE}" "link_options;${ARGV}\n")
endfunction()
function(set_source_files_properties)
  file(APPEND "${TRACE}" "source_properties;${ARGV}\n")
endfunction()
function(add_subdirectory path)
  file(APPEND "${TRACE}" "directory;${path}\n")
  if(path STREQUAL "mlir/common" OR path STREQUAL "mlir/bridge/legacy" OR path STREQUAL "tests")
    set(CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
    include("${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
  endif()
endfunction()
function(include path)
  if(path STREQUAL "cmake/af_intf_pub.cmake")
    _include("${ROOT}/autofuse/${path}")
  else()
    _include(${ARGV})
  endif()
endfunction()
set(CMAKE_CURRENT_SOURCE_DIR "${ROOT}/autofuse")
set(CMAKE_SOURCE_DIR "${ROOT}")
set(CMAKE_BINARY_DIR "${BUILD}")
set(CMAKE_CURRENT_BINARY_DIR "${BUILD}/autofuse")
set(CMAKE_SYSTEM_PROCESSOR "${PROCESSOR}")
set(AUTOFUSE_LLD_EXECUTABLE "${CMAKE_COMMAND}")
set(CMAKE_BUILD_TYPE Release)
set(ENABLE_OPEN_SRC OFF)
include("${ROOT}/autofuse/CMakeLists.txt")
"""
    )
    result = subprocess.run(
        [
            cmake,
            f"-DROOT={ROOT}",
            f"-DBUILD={tmp_path}",
            f"-DTRACE={trace}",
            f"-DENABLE_AUTOFUSE_MLIR={'ON' if mlir_enabled else 'OFF'}",
            f"-DRUN_TEST={run_test}",
            f"-DPROCESSOR={processor}",
            f"-DLLVM_BUILD_DIR={llvm_build}",
            "-S",
            str(tmp_path),
            "-B",
            str(tmp_path / "build"),
        ],
        cwd=ROOT / "autofuse",
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [line.split(";") for line in trace.read_text().splitlines()]
    directories = [event[1] for event in events if event[0] == "directory"]
    libraries = [event[1] for event in events if event[0] == "library"]
    links = [event for event in events if event[0] == "link"]

    for directory in ("mlir/common", "mlir/bridge/legacy", "mlir"):
        assert (directory in directories) == mlir_enabled
    for library in (
        "AutofuseMlirScheduleCommon",
        "AutofuseMlirScheduleLegacyAdapter",
        "AutofuseMlirScheduleBridge",
    ):
        assert (library in libraries) == mlir_enabled
    assert ("tests" in directories) == bool(run_test)
    for target in ("aihac_codegen", "codegen"):
        linked = any(
            event[1] == target and "AutofuseMlirScheduleBridge" in event[2:]
            for event in links
        )
        assert linked == (
            mlir_enabled and (target == "aihac_codegen" or bool(run_test))
        )
    if mlir_enabled:
        assert directories.index("graph_metadef/proto") < directories.index(
            "mlir/common"
        )
        assert directories.index("mlir/common") < directories.index(
            "mlir/bridge/legacy"
        )
        assert directories.index("mlir/bridge/legacy") < directories.index("mlir")
        if run_test:
            assert directories.index("mlir") < directories.index("tests")
        assert any(
            event[1:3] == ["AutofuseMlirScheduleLegacyAdapter", "PUBLIC"]
            and "unified_dlog" in event[3:]
            for event in links
        )
    else:
        assert not any(
            event[0] == "package" and event[1] in ("LLVM", "MLIR") for event in events
        )
        assert not any("AutofuseMlir" in item for event in links for item in event[2:])

    uses_lld = any(
        event[0:2] == ["link_options", "aihac_codegen"] and "-fuse-ld=lld" in event
        for event in events
    )
    assert uses_lld == (mlir_enabled and processor == "aarch64")
    test_sections = any(
        event[0:2] == ["compile_options", "codegen"] and "-ffunction-sections" in event
        for event in events
    )
    assert test_sections == (mlir_enabled and bool(run_test))
    codegen_properties = any(
        event[0] == "source_properties"
        and str(ROOT / "autofuse/codegen/codegen.cpp") in event
        and "-Wno-error=unused-parameter" in event
        for event in events
    )
    assert codegen_properties == mlir_enabled
