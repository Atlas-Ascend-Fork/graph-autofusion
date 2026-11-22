# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host C++ checks of the device templates' callback declaration contract.

Compile the actual template/parameter declarations with empty function bodies.
This checks C++ name lookup and callback substitution, not AscendC device logic.
No CANN SDK or pyautofuse import is needed.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


API_ROOT = Path(__file__).resolve().parents[3] / "v35/ascendc/api_cube"
LEGACY_TYPES = """
class AutoFusionVector { public: struct Params {}; };
class AutoFusionVectorDB { public: struct Params {}; };
"""


def template_declarations(header, name):
    declarations = []
    for function in (name, name + "_db"):
        position = header.index("void " + function + "(")
        start = header.rfind("#ifdef INDUCTOR_TILING_DATA", 0, position)
        if start < 0:
            raise AssertionError("Missing template declaration for " + function)
        body = header.index("{", position)
        declarations.append(header[start:body] + "{}\n")
    return "\n".join(declarations)


class TestCvCallbackTemplateDeclarations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++")
        if cls.compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")

    def compile_source(self, source):
        with tempfile.TemporaryDirectory(prefix="cv-callback-declarations-") as root:
            path = Path(root) / "test.cpp"
            path.write_text(source)
            return subprocess.run(
                [self.compiler, "-std=c++17", "-fsyntax-only", str(path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

    def source(self, header, function, inductor, mode):
        prologue = "#include <cstdint>\n#define __aicore__\nusing GM_ADDR = void*;\n"
        if inductor:
            prologue += "#define INDUCTOR_TILING_DATA\n"
        if mode == "legacy_before":
            prologue += LEGACY_TYPES
        source = prologue + template_declarations(header, function)
        if mode == "legacy_after":
            source += LEGACY_TYPES
        source += "struct CvTiling {};\n"
        if mode == "custom":
            source += "class CustomCallback { public: struct Params {}; };\n"
        calls = []
        if mode != "common_only":
            for suffix, legacy_type in (
                ("", "AutoFusionVector"),
                ("_db", "AutoFusionVectorDB"),
            ):
                arguments = (["CvTiling"] if inductor else []) + ["0"] * 7
                callback = "CustomCallback" if mode == "custom" else legacy_type
                # Inductor legacy calls exercise the existing default argument.
                if mode == "custom" or not inductor:
                    arguments.append(callback)
                tiling = "CvTiling{}" if inductor else "nullptr"
                calls.append(
                    f"{function}{suffix}<{','.join(arguments)}>("
                    f"nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {tiling}, nullptr);"
                )
        source += "int main() {\n" + "\n".join(calls) + "\n}\n"
        return source

    def test_common_custom_and_legacy_callbacks(self):
        for filename, function in (
            ("matmul_dynamic.h", "mat_mul_v3_fusion"),
            ("batch_matmul_dynamic.h", "batch_mat_mul_v3_fusion"),
        ):
            header = (API_ROOT / filename).read_text()
            for inductor in (False, True):
                for mode in ("common_only", "custom", "legacy_before", "legacy_after"):
                    with self.subTest(header=filename, inductor=inductor, mode=mode):
                        result = self.compile_source(
                            self.source(header, function, inductor, mode)
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_undeclared_default_type_regression_is_detected(self):
        header = (API_ROOT / "matmul_dynamic.h").read_text()
        # Model the reported pre-fix output: named defaults, without declarations.
        header = header.replace("class AutoFusionVector;", "").replace(
            "class AutoFusionVectorDB;", ""
        )
        result = self.compile_source(
            self.source(header, "mat_mul_v3_fusion", True, "custom")
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AutoFusionVector", result.stderr)


if __name__ == "__main__":
    unittest.main()
