# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("optimization", ["-O2", "-O3"])
def test_timestamp_avoids_new_libgcc_dependency(tmp_path, optimization):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]) or not shutil.which("readelf"):
        pytest.skip("C++ compiler and readelf are required for the ELF ABI check")
    library = tmp_path / "pgo_timestamp.so"
    subprocess.run(
        compiler
        + [
            "-std=c++17",
            optimization,
            "-shared",
            "-fPIC",
            f"-I{ROOT_DIR / 'autofuse/tests/depends/securec'}",
            str(ROOT_DIR / "autofuse/codegen/pgo/pgo_timestamp.cpp"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    symbols = subprocess.run(
        ["readelf", "--wide", "--dyn-syms", str(library)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "__divmodti4" not in symbols, symbols
    assert "GCC_7.0.0" not in symbols, symbols
