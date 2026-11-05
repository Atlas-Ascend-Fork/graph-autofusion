# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate the build-local Compact DMA check repair without a CANN install."""

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

SDK_HELPER = (
    Path(__file__).resolve().parents[2] / "e2e/cpu/support/prepare_reg_base_sdk.py"
)
prepare_sdk = runpy.run_path(str(SDK_HELPER))["prepare_sdk"]

LOAD = """template <typename T, PaddingMode mode>
__aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams,
    const DataCopyPadExtParams<T>& padParams)
{
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dst, src, dataCopyParams, padParams)));
    DataCopyPadGm2UBImpl<T, mode>(dst, src, dataCopyParams, padParams);
}
"""
STORE = """template <typename T, PaddingMode mode>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams)
{
    ASCENDC_REPORT_OVERFLOW_MEM((CheckDataCopyPadTensorSizeOverflow(dst, src, dataCopyParams)));
    DataCopyPadUB2GMImpl<T, mode>(dst, src, dataCopyParams);
}
"""


class RegBaseSdkPreparationTest(unittest.TestCase):
    def test_checks_remain_and_sdk_refresh_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory) / "sdk"
            output = Path(directory) / "private"
            relative = Path("impl/basic_api/kernel_operator_data_copy_intf_impl.h")
            (sdk / relative.parent).mkdir(parents=True)
            (sdk / "include").mkdir()
            source = LOAD + STORE
            (sdk / relative).write_text(source)
            other = sdk / "include/kernel_operator.h"
            other.write_text("old SDK header")
            prepare_sdk(sdk, output)
            corrected = output / relative
            text = corrected.read_text()
            self.assertEqual(text.count("DataCopyPad Compact"), 2)
            # Both original Normal bounds checks and the real DMA calls survive.
            for line in source.splitlines():
                if "ASCENDC_REPORT_OVERFLOW_MEM" in line or "Impl<T, mode>" in line:
                    self.assertIn(line.strip(), text)
            self.assertIn("dst.GetSize()", text)
            self.assertIn("src.GetSize()", text)
            self.assertEqual((sdk / relative).read_text(), source)
            os.utime(corrected, ns=(1_000_000_000, 1_000_000_000))
            other.write_text("new SDK header")
            prepare_sdk(sdk, output)
            self.assertEqual(corrected.stat().st_mtime_ns, 1_000_000_000)
            self.assertEqual(
                (output / "include/kernel_operator.h").read_text(), "new SDK header"
            )
            self.assertEqual(corrected.read_text(), text)


class RegBaseSdkCmakeTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_native_and_compact_sdk_configuration(self):
        for architecture, available in (
            ("aarch64-linux", False),
            ("aarch64-linux", True),
            ("x86_64-linux", True),
        ):
            with (
                self.subTest(compact_sdk=available),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                sdk = root / "cann" / architecture / "asc"
                (sdk / "include").mkdir(parents=True)
                if available:
                    header = (
                        sdk / "impl/basic_api/kernel_operator_data_copy_intf_impl.h"
                    )
                    header.parent.mkdir(parents=True)
                    header.write_text(LOAD + STORE)
                (root / "dummy.cpp").write_text("int main() { return 0; }")
                cases = root / "cases"
                cases.mkdir()
                mlir = SDK_HELPER.parents[4]
                (root / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(RegBaseHeaders LANGUAGES CXX)
enable_testing()
set(MLIR_CODE_ROOT_DIR "{mlir}")
set(CODE_ROOT_DIR "{mlir.parent}")
set(Python3_EXECUTABLE "{sys.executable}")
set(ASCEND_INSTALL_PATH "{root / "cann"}")
set(AF_MLIR_E2E_CPU_REG_BASE_NATIVE_SDK "{sdk}")
set(AF_MLIR_E2E_CPU_REG_BASE_SDK "{root / "private"}")
set(AF_MLIR_E2E_CPU_REG_BASE_TIKICPULIB_TARGET tikicpulib)
add_library(tikicpulib INTERFACE)
add_library(GTest::gtest INTERFACE IMPORTED)
add_library(GTest::gtest_main INTERFACE IMPORTED)
add_library(MlirE2eCpuTilingStub SHARED dummy.cpp)
add_custom_target(check-e2e-cpu)
include("{mlir}/test/e2e/cpu/AddCpuCase.cmake")
add_subdirectory(cases)
""")
                (cases / "CMakeLists.txt").write_text(f"""
foreach(case IN ITEMS ordinary compact)
    set(options)
    if(case STREQUAL "compact")
        set(options REQUIRES_COMPACT_SDK)
    endif()
    add_mlir_e2e_cpu_case(${{case}} PLATFORM reg_base ${{options}}
        INPUT_SHAPES "[2,3]" OUTPUT_SHAPE "[3,2]"
        GENERATOR_SOURCES "{root}/dummy.cpp" TEST_SOURCES "{root}/dummy.cpp")
endforeach()
if(NOT TARGET MlirE2eCpu_ordinary)
    message(FATAL_ERROR "ordinary reg case was skipped")
endif()
get_target_property(paths MlirE2eCpu_ordinary INCLUDE_DIRECTORIES)
if(NOT "{sdk}/include" IN_LIST paths OR "{root}/private/include" IN_LIST paths)
    message(FATAL_ERROR "ordinary reg case uses incorrect headers: ${{paths}}")
endif()
if(TARGET MlirE2eCpu_compact)
    if(NOT {"TRUE" if available else "FALSE"})
        message(FATAL_ERROR "compact case should be skipped")
    endif()
    get_target_property(paths MlirE2eCpu_compact INCLUDE_DIRECTORIES)
    if(NOT "{root}/private/include" IN_LIST paths)
        message(FATAL_ERROR "compact case lacks patched headers")
    endif()
elseif({"TRUE" if available else "FALSE"})
    message(FATAL_ERROR "compact case was incorrectly skipped")
endif()
""")
                result = subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if available:
                    self.assertEqual(result.returncode, 0, result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("failed to prepare Compact DMA SDK", result.stdout)
                    self.assertNotIn("Skipping Compact DMA", result.stdout)


if __name__ == "__main__":
    unittest.main()
