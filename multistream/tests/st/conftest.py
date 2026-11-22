# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Wheel fixtures and environment reporting for multistream ST."""

import platform
import subprocess
import sys
from pathlib import Path

import pytest

import multistream as ms


@pytest.fixture(scope="session")
def installed_wheel(tmp_path_factory):
    source = Path(__file__).resolve().parents[2]
    work = tmp_path_factory.mktemp("multistream-wheel")
    wheels = work / "wheels"
    target = work / "installed"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(wheels),
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        cwd=work,
        check=True,
        timeout=120,
    )
    subprocess.run(
        ["cmake", "--build", str(wheels), "--target", "multistream_whl", "-j", "8"],
        cwd=work,
        check=True,
        timeout=600,
    )
    artifacts = list(wheels.glob("multistream-*.whl"))
    assert len(artifacts) == 1
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            str(artifacts[0]),
        ],
        cwd=work,
        check=True,
        timeout=600,
    )
    return artifacts[0], target


@pytest.fixture(scope="session", autouse=True)
def record_environment(record_testsuite_property):
    record_testsuite_property("python", platform.python_version())
    record_testsuite_property("package", ms.__file__)
    record_testsuite_property("runtime", "stubbed rtGetSocSpec")
