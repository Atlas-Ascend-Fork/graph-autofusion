# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pathlib
import shutil
import subprocess
import tempfile
import unittest


class TilingExpressionAllocationTest(unittest.TestCase):
    def test_evaluation_uses_caller_scratch_without_new_allocations(self):
        compiler = shutil.which("g++")
        self.assertIsNotNone(
            compiler, "g++ is required for the native allocation regression"
        )
        root = pathlib.Path(__file__).resolve().parents[3]
        source = (
            pathlib.Path(__file__).parent / "data" / "tiling_expression_allocations.cpp"
        )
        with tempfile.TemporaryDirectory(
            prefix="tiling_expression_allocations_"
        ) as directory:
            binary = pathlib.Path(directory) / "probe"
            command = [
                compiler,
                "-std=c++17",
                "-O2",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-I" + str(root / "att"),
                str(source),
                str(root / "att/program/tiling_expression.cpp"),
                str(root / "att/program/tiling_stage.cpp"),
                str(root / "att/program/tiling_program_value.cpp"),
                "-o",
                str(binary),
            ]
            built = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run(
                [str(binary)], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                result.stdout.strip(),
                "negative_control=1 allocations=0 reads=4000 ok=1",
            )


if __name__ == "__main__":
    unittest.main()
