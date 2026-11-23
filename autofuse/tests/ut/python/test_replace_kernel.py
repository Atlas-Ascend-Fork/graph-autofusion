# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import pytest

from compile_test_utils import load_asc_codegen_compile_module

SPLIT_HEADER_NAMES = (
    "autofuse_tiling_func_state.h",
    "autofuse_tiling_func_log.h",
    "autofuse_tiling_func_pgo.h",
    "autofuse_tiling_func_base.h",
    "autofuse_tiling_func_solver.h",
    "autofuse_tiling_func_api.h",
    "autofuse_tiling_func_entry.h",
    "autofuse_tiling_func_tail.h",
    "cube_kernel_tiling_wrapper.h",
)


@pytest.fixture(scope="module")
def asc_codegen_compile_module():
    with load_asc_codegen_compile_module() as module:
        yield module


def test_get_replace_kernel_root_returns_none_when_env_is_missing(
    monkeypatch, asc_codegen_compile_module
):
    monkeypatch.delenv("AUTOFUSE_DFX_FLAGS", raising=False)
    assert asc_codegen_compile_module.get_replace_kernel_root() is None


def test_replace_device_kernel_replaces_exact_kernel_file(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    device_build_dir = tmpdir.ensure("build", "device", dir=True)

    replace_root.join("demo_graph_op_kernel.cpp").write("new device kernel")
    dst = device_build_dir.join("demo_graph_op_kernel.cpp")
    dst.write("old device kernel")

    asc_codegen_compile_module.replace_device_kernel(
        str(replace_root), str(device_build_dir), "demo_graph"
    )

    assert dst.read() == "new device kernel"


def test_replace_host_files_replaces_host_tiling_group_and_removes_stale_files(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.ensure("a", "b", "whatever", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("demo_graph_tiling_func_1.cpp").write("new1")
    host_build_dir.join("demo_graph_tiling_func_old.cpp").write("stale")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    host_files = sorted(
        [
            path.basename
            for path in host_build_dir.visit(fil="demo_graph*tiling_func*.cpp")
        ]
    )
    assert host_files == [
        "demo_graph_tiling_func_0.cpp",
        "demo_graph_tiling_func_1.cpp",
    ]


def test_find_host_replace_source_dir_raises_when_multiple_dirs_match(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    dir1 = replace_root.mkdir("x")
    dir2 = replace_root.mkdir("y")
    dir1.join("demo_graph_tiling_func_a.cpp").write("a")
    dir2.join("demo_graph_tiling_func_b.cpp").write("b")

    with pytest.raises(RuntimeError):
        asc_codegen_compile_module.find_host_replace_source_dir(
            str(replace_root), "demo_graph"
        )


def test_replace_host_files_allows_missing_optional_headers(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "new0"
    assert not host_build_dir.join("autofuse_tiling_data.h").check()
    assert not host_build_dir.join("autofuse_tiling_func_common.h").check()


def test_replace_host_files_skips_same_file_copy_when_source_is_host_dir(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    host_build_dir = replace_root.ensure("host", dir=True)
    host_build_dir.join("demo_graph_tiling_func_0.cpp").write("same cpp")
    host_build_dir.join("autofuse_tiling_data.h").write("same header")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "same cpp"
    assert host_build_dir.join("autofuse_tiling_data.h").read() == "same header"


def test_replace_host_files_copies_autofuse_cube_tiling_data_when_present(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("autofuse_cube_tiling_data.h").write("cube")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("autofuse_cube_tiling_data.h").read() == "cube"


def test_replace_host_files_copies_split_headers_when_present(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_solver_func.cpp").write("new solver")
    for header_name in SPLIT_HEADER_NAMES:
        source_dir.join(header_name).write(header_name)

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    for header_name in SPLIT_HEADER_NAMES:
        assert host_build_dir.join(header_name).read() == header_name


def test_host_miss_device_hit_keeps_device_replacement_independent(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    device_build_dir = tmpdir.ensure("build", "device", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    replace_root.join("demo_graph_op_kernel.cpp").write("new device kernel")
    device_build_dir.join("demo_graph_op_kernel.cpp").write("old device kernel")

    asc_codegen_compile_module.replace_device_kernel(
        str(replace_root), str(device_build_dir), "demo_graph"
    )
    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert (
        device_build_dir.join("demo_graph_op_kernel.cpp").read() == "new device kernel"
    )
    assert host_build_dir.listdir() == []


def test_device_miss_host_hit_keeps_host_replacement_independent(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    device_build_dir = tmpdir.ensure("build", "device", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    device_build_dir.join("demo_graph_op_kernel.cpp").write("old device kernel")

    asc_codegen_compile_module.replace_device_kernel(
        str(replace_root), str(device_build_dir), "demo_graph"
    )
    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert (
        device_build_dir.join("demo_graph_op_kernel.cpp").read() == "old device kernel"
    )
    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "new0"


def _set_module_mocks(module, monkeypatch, mocks):
    for name, mock in mocks.items():
        monkeypatch.setattr(module, name, mock)


def _mock_compile_preparation(asc_codegen_compile_module, monkeypatch, tmpdir):
    mocks = {
        "get_graph_basic_info": lambda params, args: (
            "demo_graph",
            1,
            1,
            True,
            False,
            {},
        ),
        "create_compile_dirs": lambda temp_dir: (
            str(tmpdir.ensure("device", dir=True)),
            str(tmpdir.ensure("host", dir=True)),
        ),
        "generate_device_and_host_code": lambda **kwargs: ("kernel", {"k": "v"}),
        "is_static_compile": lambda params, tiling_func_srcs: True,
    }
    _set_module_mocks(asc_codegen_compile_module, monkeypatch, mocks)


def _mock_compile_execution(asc_codegen_compile_module, monkeypatch, tmpdir, calls):
    def fake_static_shape_compile(**kwargs):
        if "use_cv_common" in kwargs:
            kwargs["use_cv_common"][0] = True
        calls.append(("static_shape_compile", kwargs["graph_name"], kwargs))

    mocks = {
        "ascbc_matmul_kernel_tiling_pro": lambda *args, **kwargs: kwargs[
            "use_cv_common"
        ].__setitem__(0, True),
        "static_shape_compile": fake_static_shape_compile,
        "get_replace_kernel_root": lambda: tmpdir.ensure("replace_root", dir=True),
        "replace_host_files": lambda root, host_dir, graph_name: calls.append(
            ("replace_host", host_dir, graph_name)
        ),
        "replace_kernel": lambda *args, **kwargs: calls.append(("replace_device",)),
        "ascbc_kernel_compile": lambda *args, **kwargs: (
            "kernel.o",
            "kernel.json",
        ),
        "asc_graph_compile_post": lambda *args, **kwargs: calls.append(
            ("post", args[0])
        ),
        "timestamp_set": lambda *args, **kwargs: None,
    }
    _set_module_mocks(asc_codegen_compile_module, monkeypatch, mocks)


def _mock_host_replacement_compile_flow(
    asc_codegen_compile_module, monkeypatch, tmpdir, calls
):
    _mock_compile_preparation(asc_codegen_compile_module, monkeypatch, tmpdir)
    _mock_compile_execution(asc_codegen_compile_module, monkeypatch, tmpdir, calls)


def test_host_replacement_happens_once_before_static_shape_compile_and_uses_final_host_dir(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    calls = []
    host_cv_common_dir = tmpdir.ensure("host", "cv_common", dir=True)
    _mock_host_replacement_compile_flow(
        asc_codegen_compile_module, monkeypatch, tmpdir, calls
    )

    asc_codegen_compile_module.asc_graph_compile(
        "arg0",
        "kernel_name",
        temp_dir=str(tmpdir),
        params={
            # asc_graph_compile入口按is_cube_type分流CV融合预算，非cube桩保持原_compile_impl流程
            "schedule_results": SimpleNamespace(is_cube_type=lambda: False),
            "vector_core_num": 1,
            "impl_mode": None,
        },
    )

    replace_calls = [item for item in calls if item[0] == "replace_host"]
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == str(host_cv_common_dir)
    order = [item[0] for item in calls]
    assert order.index("replace_host") < order.index("static_shape_compile")
    static_shape_calls = [item for item in calls if item[0] == "static_shape_compile"]
    assert static_shape_calls[0][2]["vector_core_num"] == 1


class _LogCapture(object):
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        if args:
            message = message % args
        self.messages.append(message)


def test_replace_host_files_logs_source_target_removed_and_copied(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.ensure("nested", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)
    log_capture = _LogCapture()

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("autofuse_tiling_data.h").write("tiling")
    host_build_dir.join("demo_graph_tiling_func_old.cpp").write("stale")

    monkeypatch.setattr(asc_codegen_compile_module, "logger", log_capture)

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert any(
        "replace host source dir:" in message for message in log_capture.messages
    )
    assert any(
        "replace host target dir:" in message for message in log_capture.messages
    )
    assert any(
        "cleanup stale host tiling files:" in message
        for message in log_capture.messages
    )
    assert any("copied host files:" in message for message in log_capture.messages)


def test_find_host_replace_source_dir_error_message_contains_conflict_dirs(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    dir1 = replace_root.mkdir("x")
    dir2 = replace_root.mkdir("y")
    dir1.join("demo_graph_tiling_func_a.cpp").write("a")
    dir2.join("demo_graph_tiling_func_b.cpp").write("b")

    with pytest.raises(RuntimeError) as exc_info:
        asc_codegen_compile_module.find_host_replace_source_dir(
            str(replace_root), "demo_graph"
        )

    error_message = str(exc_info.value)
    assert str(dir1) in error_message
    assert str(dir2) in error_message


def test_replace_host_files_does_not_touch_infershape_cpp(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("demo_graph_infershape.cpp").write("new infershape")
    host_build_dir.join("demo_graph_infershape.cpp").write("old infershape")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "new0"
    assert host_build_dir.join("demo_graph_infershape.cpp").read() == "old infershape"


def test_ascbc_host_compile_uses_32_jobs_for_tiling_func_split_compile(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    host_build_dir = tmpdir.mkdir("host")
    commands = []

    class FakeCompletedProcess(object):
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *, cwd, capture_output, text):
        assert cwd == str(host_build_dir)
        commands.append(cmd)
        return FakeCompletedProcess()

    monkeypatch.setattr(asc_codegen_compile_module.subprocess, "run", fake_run)

    asc_codegen_compile_module.ascbc_host_compile(
        asc_codegen_compile_module.HostCompileContext(
            "demo_graph", "demo_kernel", str(host_build_dir), True, True, False
        )
    )

    assert len(commands) == 2
    assert commands[0][0] == "cmake"
    assert commands[1] == ["make", "-C", "./", "-j", "32"]
