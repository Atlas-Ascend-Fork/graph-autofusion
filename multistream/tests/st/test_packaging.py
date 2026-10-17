# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Validate the actual release wheel's contents and dependency metadata."""

import re
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

pytestmark = [pytest.mark.st, pytest.mark.package]
_SOURCE = Path(__file__).resolve().parents[2]


def test_wheel_contains_all_package_modules(installed_wheel):
    wheel, _ = installed_wheel
    sources = list(_SOURCE.glob("*.py"))
    for directory in ("common", "minidag", "algorithm"):
        sources.extend((_SOURCE / directory).rglob("*.py"))
    expected = {
        "multistream/" + path.relative_to(_SOURCE).as_posix()
        for path in sources
        if "__pycache__" not in path.parts
    }
    with zipfile.ZipFile(wheel) as archive:
        actual = {
            name for name in archive.namelist() if name.startswith("multistream/")
        }
    assert actual == expected


def test_wheel_is_pure_python_without_runtime_dependencies(installed_wheel):
    wheel, _ = installed_wheel
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(
            name.endswith((".so", ".dll", ".pyd", ".dylib")) for name in names
        )
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        assert len(metadata_names) == len(wheel_names) == 1
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        wheel_metadata = Parser().parsestr(archive.read(wheel_names[0]).decode("utf-8"))
    assert wheel_metadata["Root-Is-Purelib"] == "true"
    assert metadata.get_all("Provides-Extra") == ["dev"]
    for requirement in metadata.get_all("Requires-Dist", []):
        assert re.fullmatch(
            r"extra\s*==\s*(['\"])dev\1", requirement.partition(";")[2].strip()
        )


@pytest.mark.parametrize("in_source_build", [False, True])
def test_cmake_incremental_wheel_tracks_source_changes(tmp_path, in_source_build):
    source = tmp_path / "package source"
    shutil.copytree(
        _SOURCE,
        source,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", "tests", "coverage"
        ),
    )
    build = source / "build" if in_source_build else tmp_path / "cmake build"
    build.mkdir()
    sentinel = source / "build" / "caller-owned.txt"
    sentinel.parent.mkdir(exist_ok=True)
    sentinel.write_text("do not remove")
    deleted = source / "common" / "deleted_module.py"
    renamed = source / "common" / "original_module.py"
    deleted.write_text("VALUE = 1\n")
    renamed.write_text("VALUE = 2\n")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Unix Makefiles",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    def build_and_check():
        result = subprocess.run(
            ["cmake", "--build", str(build), "-j", "8"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        wheel = next(build.glob("multistream-*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            actual = {name for name in archive.namelist() if name.endswith(".py")}
        expected = {
            "multistream/" + path.relative_to(source).as_posix()
            for directory in (
                source,
                source / "common",
                source / "minidag",
                source / "algorithm",
            )
            for path in directory.glob("*.py")
        }
        assert actual == expected
        assert sentinel.read_text() == "do not remove"
        return wheel.stat().st_mtime_ns

    timestamp = build_and_check()
    assert build_and_check() == timestamp
    deleted.unlink()
    build_and_check()
    renamed = renamed.rename(renamed.with_name("renamed_module.py"))
    build_and_check()
    renamed.write_text("VALUE = 3\n")
    build_and_check()
    with zipfile.ZipFile(next(build.glob("multistream-*.whl"))) as archive:
        assert archive.read("multistream/common/renamed_module.py") == b"VALUE = 3\n"
