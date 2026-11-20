# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import sys
import types
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[4]
UT_PYTHON_DIR = ROOT_DIR / "autofuse/tests/ut/python"
sys.path.insert(0, os.fspath(UT_PYTHON_DIR))

from compile_test_utils import PYTHON_DIR, load_compile_module  # noqa: E402


MODULE_NAME = "autofuse.compile_adapter"
MODULE_PATH = os.path.join(PYTHON_DIR, "compile_adapter.py")
ASCENDC_MODULE_NAME = "autofuse.compiler.python.ascendc_compile"
ASCENDC_MODULE_PATH = os.path.join(PYTHON_DIR, "ascendc_compile.py")


@pytest.fixture()
def compile_adapter_module():
    ascendc_compile_module = types.ModuleType("autofuse.ascendc_compile")

    class CompileError(Exception):
        pass

    ascendc_compile_module.CompileError = CompileError
    with load_compile_module(
        MODULE_NAME,
        MODULE_PATH,
        extra_autofuse_attrs={"ascendc_compile": ascendc_compile_module},
        extra_modules={"autofuse.ascendc_compile": ascendc_compile_module},
    ) as loaded_module:
        yield loaded_module


@pytest.fixture()
def ascendc_compile_module():
    platform_info_module = types.ModuleType(
        "asc_op_compile_base.common.platform.platform_info"
    )

    def get_soc_spec(_key):
        return ""

    platform_info_module.get_soc_spec = get_soc_spec
    extra_modules = {
        "asc_op_compile_base": types.ModuleType("asc_op_compile_base"),
        "asc_op_compile_base.common": types.ModuleType("asc_op_compile_base.common"),
        "asc_op_compile_base.common.platform": types.ModuleType(
            "asc_op_compile_base.common.platform"
        ),
        "asc_op_compile_base.common.platform.platform_info": platform_info_module,
    }
    with load_compile_module(
        ASCENDC_MODULE_NAME, ASCENDC_MODULE_PATH, extra_modules=extra_modules
    ) as loaded_module:
        yield loaded_module


def _make_scheme_a_host_impl():
    return "\n".join(
        [
            "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
            "struct CommonType {};",
            "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
            'extern "C" int Solver() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner",
            "int main() { return 0; }",
            "// AUTOFUSE_SPLIT_FILE_END: PgoRunner",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: PgoDeviceSource",
            'extern "C" __global__ __aicore__ void graph() {}',
            "// AUTOFUSE_SPLIT_FILE_END: PgoDeviceSource",
        ]
    )


def test_scheme_a_host_compile_publishes_generation_bundle_last(
    compile_adapter_module, tmp_path, monkeypatch
):
    output_file = tmp_path / "tiling.so"
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    argv = [
        f"--output_file={output_file}",
        f"--output_path={build_dir}",
        "--graph_name=TestGraph",
        "--soc_version=Ascend910B",
    ]
    published = {}

    def fake_main(args):
        assert args.host_files == [
            os.fspath(build_dir / "host" / "test_graph_tiling_func_solver_func.cpp")
        ]
        assert Path(args.pgo_runner_file).name == "test_graph_tiling_func_PgoRunner.cpp"
        assert Path(args.pgo_device_file).name == "test_graph_pgo_device.cpp"
        generation = "stgeneration"
        generation_dir = Path(f"{output_file}.pgo_v2/{generation}")
        generation_dir.parent.mkdir()
        staging_dir = tmp_path / ".staging"
        staging_dir.mkdir()
        (staging_dir / "tiling.so").write_bytes(b"tiling")
        (staging_dir / "tiling.so.pgo_runner").write_bytes(b"runner")
        (staging_dir / "tiling.so.pgo_kernel.aicore_binary_elf_v1").write_bytes(
            b"kernel"
        )
        (staging_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "bundle_schema_version": 2,
                    "cache_root": "pgo_v2",
                    "generation": generation,
                    "result_protocol_version": 1,
                }
            ),
            encoding="utf-8",
        )
        os.replace(staging_dir, generation_dir)
        tmp_tiling = tmp_path / ".tiling.tmp"
        tmp_tiling.write_bytes(b"tiling")
        os.replace(tmp_tiling, output_file)
        published["generation_dir"] = generation_dir

    compile_adapter_module.ascendc_compile.main = fake_main
    compile_adapter_module.host_compile(
        "struct AutofuseTilingData {};", _make_scheme_a_host_impl(), argv
    )

    assert output_file.read_bytes() == b"tiling"
    assert (published["generation_dir"] / "tiling.so").read_bytes() == b"tiling"
    assert (
        published["generation_dir"] / "tiling.so.pgo_runner"
    ).read_bytes() == b"runner"
    assert (
        published["generation_dir"] / "tiling.so.pgo_kernel.aicore_binary_elf_v1"
    ).read_bytes() == b"kernel"


def test_inductor_host_link_includes_acl_runtime(ascendc_compile_module, tmp_path):
    captured = {}
    args = types.SimpleNamespace(
        host_files=[os.fspath(tmp_path / "host.cpp")],
        output_file=os.fspath(tmp_path / "tiling.so"),
        stage="host",
        graph_name="empty_tensor_graph",
        pgo_runner_file=None,
        temp_dir=os.fspath(tmp_path),
    )

    def fake_compile_host_objs(*_args):
        return ["host.o"]

    def fake_is_cv_fusion_compile(_args):
        return False

    ascendc_compile_module.module.compile_host_objs = fake_compile_host_objs
    ascendc_compile_module.module.is_cv_fusion_compile = fake_is_cv_fusion_compile

    def fake_link_shared(
        target_file, obj_files, link_libraries=None, extra_link_options=None
    ):
        captured["link_libraries"] = link_libraries
        return target_file

    ascendc_compile_module.module.link_shared = fake_link_shared

    ascendc_compile_module.build_host_output(args, os.fspath(tmp_path))

    assert "acl_rt" in captured["link_libraries"]


def _long_graph_name(op_count=80, md5="20fd52a523aa877ef4c42f6134c141de"):
    ops = ["op{}".format(i) for i in range(op_count)]
    return "autofused_{}_{}".format("_".join(ops), md5)


def _make_rt_launch_kernel_impl(aiv_key):
    return "\n".join(
        [
            "// common part + MIX entry",
            'extern "C" __global__ void graph() {}',
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {aiv_key}",
            'extern "C" __global__ void graph_aiv() {}',
            f"// AUTOFUSE_SPLIT_FILE_END: {aiv_key}",
        ]
    )


def test_long_graph_name_compile_writes_fitting_source_names(
    compile_adapter_module, tmp_path
):
    graph_name = _long_graph_name()
    aiv_key = f"{graph_name}_op_kernel_aiv.cpp"
    output_file = tmp_path / "tiling.so"
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    argv = [
        f"--output_file={output_file}",
        f"--output_path={build_dir}",
        f"--graph_name={graph_name}",
        "--soc_version=Ascend910B",
    ]
    captured = []

    def fake_main(args):
        captured.append(args)

    compile_adapter_module.ascendc_compile.main = fake_main

    # host 侧：split 源与 PGO device 源文件名都需裁剪后才能落盘。
    compile_adapter_module.host_compile(
        "struct AutofuseTilingData {};", _make_scheme_a_host_impl(), argv
    )
    host_args = captured[-1]
    host_paths = [*host_args.host_files, host_args.pgo_runner_file]
    for path in host_paths:
        base_name = os.path.basename(path)
        assert os.path.exists(path)
        assert len((base_name + ".o").encode()) <= 255
        assert "_cut" in base_name

    # device 侧：含 AIV 双 bin 单元，主/AIV 源文件名都需裁剪后才能落盘。
    compile_adapter_module.kernel_compile(
        "struct AutofuseTilingData {};",
        _make_rt_launch_kernel_impl(aiv_key),
        argv,
    )
    device_args = captured[-1]
    for path in (device_args.device_files, device_args.device_files_aiv):
        base_name = os.path.basename(path)
        assert os.path.exists(path)
        assert len((base_name + ".o").encode()) <= 255
        assert "_cut" in base_name
    assert device_args.device_files.endswith("_op_kernel.cpp")
    assert device_args.device_files_aiv.endswith("_op_kernel_aiv.cpp")
