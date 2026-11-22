# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

import pytest

from compile_test_utils import load_asc_codegen_compile_module

STRIP_BEFORE_SIZE = 1000
STRIP_AFTER_SIZE = 400


@pytest.fixture(scope="module")
def asc_codegen_compile_module():
    with load_asc_codegen_compile_module() as module:
        yield module


class _LogCapture(object):
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message, *args):
        self.infos.append((message, args))

    def warning(self, message, *args):
        self.warnings.append((message, args))


def test_strip_slim_so_logs_pre_strip_and_post_strip_sizes(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    so_path = tmpdir.join("libdemo_kernel.so")
    so_path.write("a" * STRIP_BEFORE_SIZE)
    log_capture = _LogCapture()

    def fake_strip_run(cmd, capture_output, text):
        assert cmd == ["strip", str(so_path)]
        with open(so_path, "w") as so_file:
            so_file.write("b" * STRIP_AFTER_SIZE)
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(asc_codegen_compile_module.subprocess, "run", fake_strip_run)
    monkeypatch.setattr(asc_codegen_compile_module, "logger", log_capture)

    asc_codegen_compile_module.strip_slim_so(str(so_path))

    assert log_capture.warnings == []
    assert len(log_capture.infos) == 1
    message, args = log_capture.infos[0]
    assert message == "[SOSIZE] strip %s: %d -> %d"
    assert args == (str(so_path), STRIP_BEFORE_SIZE, STRIP_AFTER_SIZE)


def test_strip_slim_so_failure_logs_warning_without_raising(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    so_path = tmpdir.join("libdemo_kernel.so")
    so_path.write("a" * STRIP_BEFORE_SIZE)
    log_capture = _LogCapture()

    def fake_strip_run(cmd, capture_output, text):
        return type("Result", (), {"returncode": 1, "stderr": "strip fail"})()

    monkeypatch.setattr(asc_codegen_compile_module.subprocess, "run", fake_strip_run)
    monkeypatch.setattr(asc_codegen_compile_module, "logger", log_capture)

    asc_codegen_compile_module.strip_slim_so(str(so_path))

    assert len(log_capture.warnings) == 1
    message, args = log_capture.warnings[0]
    assert message == "[SOSIZE] strip %s failed: %s"
    assert args == (str(so_path), "strip fail")
    assert log_capture.infos == []
    assert so_path.size() == STRIP_BEFORE_SIZE


def test_strip_slim_so_missing_strip_tool_warns_without_raising(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    so_path = tmpdir.join("libdemo_kernel.so")
    so_path.write("a" * STRIP_BEFORE_SIZE)
    log_capture = _LogCapture()

    def raise_file_not_found(cmd, capture_output, text):
        raise FileNotFoundError(f"No such file or directory: '{cmd[0]}'")

    monkeypatch.setattr(
        asc_codegen_compile_module.subprocess, "run", raise_file_not_found
    )
    monkeypatch.setattr(asc_codegen_compile_module, "logger", log_capture)

    asc_codegen_compile_module.strip_slim_so(str(so_path), "aarch64-linux-gnu-")

    assert len(log_capture.warnings) == 1
    message, args = log_capture.warnings[0]
    assert message == "[SOSIZE] strip %s failed: %s"
    assert args[0] == str(so_path)
    assert "aarch64-linux-gnu-strip" in args[1]
    assert log_capture.infos == []
    assert so_path.size() == STRIP_BEFORE_SIZE


def test_strip_slim_so_missing_so_warns_without_raising(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    so_path = tmpdir.join("libdemo_kernel.so")
    log_capture = _LogCapture()

    monkeypatch.setattr(asc_codegen_compile_module, "logger", log_capture)

    asc_codegen_compile_module.strip_slim_so(str(so_path))

    assert len(log_capture.warnings) == 1
    message, args = log_capture.warnings[0]
    assert message == "[SOSIZE] strip %s failed: %s"
    assert args[0] == str(so_path)
    assert log_capture.infos == []


class _FakeCompletedProcess(object):
    def __init__(self):
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""


class _FakeScheduleResults(object):
    @staticmethod
    def get_name():
        return "DemoGraph"

    @staticmethod
    def get_input_num():
        return 1

    @staticmethod
    def get_output_num():
        return 1


def test_pgo_kernel_compile_does_not_apply_slim_export(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    temp_dir = tmpdir.mkdir("temp")
    host_build_dir = temp_dir.mkdir("host")
    pgo_dir = asc_codegen_compile_module.get_pgo_dir(str(temp_dir))
    os.makedirs(pgo_dir, exist_ok=True)
    kernel_name = "demo_kernel"
    graph_name = "DemoGraph"
    host_build_dir.join(f"lib{kernel_name}.so").write("fake so")
    kernel_file = temp_dir.join("kernel.o")
    kernel_file.write("fake kernel")
    json_file = temp_dir.join("kernel.json")
    json_file.write("{}")

    monkeypatch.setattr(
        asc_codegen_compile_module,
        "ascbc_kernel_compile",
        lambda *args, **kwargs: (str(kernel_file), str(json_file)),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module.subprocess,
        "run",
        lambda cmd, capture_output, text: _FakeCompletedProcess(),
    )
    fake_code_gen = type(
        "FakeCodeGen",
        (),
        {"get_kernel_and_json_generator": lambda self, k, j: "// stub"},
    )()

    result, temp_files = asc_codegen_compile_module.pgo_kernel_compile(
        "ignored",
        kernel_name,
        temp_dir=str(temp_dir),
        params={"schedule_results": _FakeScheduleResults(), "impl_mode": None},
        op_kernel_src="void demo_kernel() {}",
        code_gen=fake_code_gen,
    )

    assert result is True
    assert os.path.exists(os.path.join(pgo_dir, f"lib{graph_name}.so"))
    cmake_source = host_build_dir.join("CMakeLists.txt").read()
    assert "--version-script" not in cmake_source
    assert not os.path.exists(os.path.join(str(host_build_dir), "export.map"))


def test_generate_cmake_lists_keeps_slim_export_for_delivery_compile(
    tmpdir, asc_codegen_compile_module
):
    host_build_dir = tmpdir.mkdir("host")

    asc_codegen_compile_module.generate_cmake_lists(
        asc_codegen_compile_module.HostCompileContext(
            "DemoGraph", "demo_kernel", str(host_build_dir), True, False, False
        )
    )

    cmake_source = host_build_dir.join("CMakeLists.txt").read()
    assert "-Wl,--version-script=" in cmake_source
    assert host_build_dir.join("export.map").exists()
    export_map_source = host_build_dir.join("export.map").read()
    for symbol in asc_codegen_compile_module.VV_SLIM_EXPORT_SYMBOLS:
        assert f"{symbol};" in export_map_source
