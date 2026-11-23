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

import inspect
import os
import types

import pytest


def _host_compile_args(tmpdir):
    return type(
        "Args",
        (),
        {
            "stage": "host",
            "temp_dir": str(tmpdir),
            "graph_name": "graph",
            "trace_stage": "host_compile",
        },
    )()


def _device_compile_args(tmpdir):
    return type(
        "Args",
        (),
        {
            "stage": "device",
            "temp_dir": str(tmpdir),
            "graph_name": "graph",
            "trace_stage": "device_compile",
        },
    )()


AIV_SPLIT_KEY = "graph_op_kernel_aiv.cpp"


def _make_rt_launch_kernel_impl(
    aiv_body='extern "C" __global__ void graph_aiv() {}', aiv_key=AIV_SPLIT_KEY
):
    """rtLaunch 双单元形态的 kernel 源：主内容在 marker 外，AIV 入口在 marker 段内。"""
    return "\n".join(
        [
            "// common part + MIX entry",
            'extern "C" __global__ void graph() {}',
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {aiv_key}",
            aiv_body,
            f"// AUTOFUSE_SPLIT_FILE_END: {aiv_key}",
        ]
    )


def test_split_device_sources_extracts_aiv_unit(compile_adapter_module):
    main_code, split_files = compile_adapter_module.split_device_sources(
        _make_rt_launch_kernel_impl()
    )

    assert 'extern "C" __global__ void graph() {}' in main_code
    assert "AUTOFUSE_SPLIT_FILE_BEGIN" not in main_code
    assert "graph_aiv" not in main_code
    assert list(split_files.keys()) == [AIV_SPLIT_KEY]
    assert 'extern "C" __global__ void graph_aiv() {}' in split_files[AIV_SPLIT_KEY]
    assert "AUTOFUSE_SPLIT_FILE_BEGIN" not in split_files[AIV_SPLIT_KEY]


def test_split_device_sources_keeps_legacy_content_intact(compile_adapter_module):
    legacy = 'extern "C" int64_t AutofuseLaunch() { return 0; }\n'
    main_code, split_files = compile_adapter_module.split_device_sources(legacy)

    assert main_code == legacy
    assert split_files == {}


def test_split_device_sources_rejects_nested_and_mismatched_markers(
    compile_adapter_module,
):
    nested = (
        f"// AUTOFUSE_SPLIT_FILE_BEGIN: {AIV_SPLIT_KEY}\n"
        "// AUTOFUSE_SPLIT_FILE_BEGIN: other.cpp\n"
        "// AUTOFUSE_SPLIT_FILE_END: other.cpp\n"
        f"// AUTOFUSE_SPLIT_FILE_END: {AIV_SPLIT_KEY}\n"
    )
    with pytest.raises(compile_adapter_module.ascendc_compile.CompileError):
        compile_adapter_module.split_device_sources(nested)

    mismatched = (
        f"// AUTOFUSE_SPLIT_FILE_BEGIN: {AIV_SPLIT_KEY}\n"
        "// AUTOFUSE_SPLIT_FILE_END: other.cpp\n"
    )
    with pytest.raises(compile_adapter_module.ascendc_compile.CompileError):
        compile_adapter_module.split_device_sources(mismatched)

    unclosed = f"// AUTOFUSE_SPLIT_FILE_BEGIN: {AIV_SPLIT_KEY}\n"
    with pytest.raises(compile_adapter_module.ascendc_compile.CompileError):
        compile_adapter_module.split_device_sources(unclosed)


def test_write_compile_device_sources_writes_dual_units(compile_adapter_module, tmpdir):
    args = _device_compile_args(tmpdir)

    compile_adapter_module.write_compile_device_sources(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": None,
            "kernel_impl_code": _make_rt_launch_kernel_impl(),
        },
        args,
        "autofuse_tiling_data.h",
        "graph_op_kernel.cpp",
    )

    device_dir = os.path.join(str(tmpdir), "device")
    assert args.device_files == os.path.join(device_dir, "graph_op_kernel.cpp")
    assert args.device_files_aiv == os.path.join(device_dir, AIV_SPLIT_KEY)
    assert os.path.exists(args.device_files)
    with open(args.device_files_aiv) as f:
        assert "graph_aiv" in f.read()


def test_write_compile_device_sources_cleans_stale_aiv_unit(
    compile_adapter_module, tmpdir
):
    args = _device_compile_args(tmpdir)
    device_dir = os.path.join(str(tmpdir), "device")
    os.makedirs(device_dir, exist_ok=True)
    stale_aiv = os.path.join(device_dir, AIV_SPLIT_KEY)
    with open(stale_aiv, "w") as f:
        f.write("stale aiv unit")

    # legacy 形态（无 marker）：残留的 stale AIV 源必须被清理，防止被误编进双 bin 链路。
    compile_adapter_module.write_compile_device_sources(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": None,
            "kernel_impl_code": 'extern "C" int64_t AutofuseLaunch() { return 0; }',
        },
        args,
        "autofuse_tiling_data.h",
        "graph_op_kernel.cpp",
    )

    assert not os.path.exists(stale_aiv)
    assert args.device_files == os.path.join(device_dir, "graph_op_kernel.cpp")
    assert args.device_files_aiv is None


def test_write_compile_device_sources_fits_long_aiv_unit(
    compile_adapter_module, tmpdir
):
    graph_name = _long_kernel_name(op_count=80)
    aiv_key = f"{graph_name}_op_kernel_aiv.cpp"
    assert len(aiv_key.encode()) > 255
    args = _device_compile_args(tmpdir)
    args.graph_name = graph_name

    compile_adapter_module.write_compile_device_sources(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": None,
            "kernel_impl_code": _make_rt_launch_kernel_impl(aiv_key=aiv_key),
        },
        args,
        "autofuse_tiling_data.h",
        "graph_op_kernel.cpp",
    )

    aiv_name = os.path.basename(args.device_files_aiv)
    assert os.path.exists(args.device_files_aiv)
    assert len(aiv_name.encode()) <= 255
    assert len((aiv_name + ".o").encode()) <= 255
    assert aiv_name.startswith("autofused_")
    assert aiv_name.endswith("_op_kernel_aiv.cpp")
    assert "_cut" in aiv_name
    assert "_20fd52a523aa877ef4c42f6134c141de" in aiv_name


def test_write_compile_device_sources_cleans_stale_aiv_unit_for_long_name(
    compile_adapter_module, tmpdir
):
    graph_name = _long_kernel_name(op_count=80)
    args = _device_compile_args(tmpdir)
    args.graph_name = graph_name
    device_dir = os.path.join(str(tmpdir), "device")
    os.makedirs(device_dir, exist_ok=True)
    stale_name = compile_adapter_module.fitted_device_aiv_file_name(graph_name)
    stale_aiv = os.path.join(device_dir, stale_name)
    with open(stale_aiv, "w") as f:
        f.write("stale aiv unit")

    compile_adapter_module.write_compile_device_sources(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": None,
            "kernel_impl_code": 'extern "C" int64_t AutofuseLaunch() { return 0; }',
        },
        args,
        "autofuse_tiling_data.h",
        "graph_op_kernel.cpp",
    )

    assert not os.path.exists(stale_aiv)
    assert args.device_files_aiv is None


def test_execute_compile_fits_long_graph_name_for_device_units(
    compile_adapter_module, tmpdir
):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    graph_name = _long_kernel_name(op_count=80)
    aiv_key = f"{graph_name}_op_kernel_aiv.cpp"
    args = _device_compile_args(tmpdir)
    args.graph_name = graph_name

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": None,
            "kernel_impl_code": _make_rt_launch_kernel_impl(aiv_key=aiv_key),
        },
        args,
    )

    for path in (captured["args"].device_files, captured["args"].device_files_aiv):
        base_name = os.path.basename(path)
        assert os.path.exists(path)
        assert len((base_name + ".o").encode()) <= 255
    main_name = os.path.basename(captured["args"].device_files)
    aiv_name = os.path.basename(captured["args"].device_files_aiv)
    assert captured["args"].device_files_aiv.endswith("_op_kernel_aiv.cpp")
    # find_aiv_device_source 的回退约定：主源名去掉 .cpp 加 _aiv.cpp 必须等于 AIV 源名。
    assert main_name[: -len("_op_kernel.cpp")] + "_op_kernel_aiv.cpp" == aiv_name


def _scheme_a_sources():
    return {
        "tiling_struct_code": "struct AutofuseTilingData {};",
        "host_impl_code": _make_scheme_a_split_host_impl(),
        "kernel_impl_code": None,
    }


def _execute_scheme_a_host_compile(compile_adapter_module, tmpdir):
    args = _host_compile_args(tmpdir)
    compile_adapter_module.execute_compile(_scheme_a_sources(), args)
    return args


def _clear_cann_root_envs(monkeypatch):
    for env_name in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_HOME"):
        monkeypatch.delenv(env_name, raising=False)


def test_host_compile_defaults_to_abi_1(compile_adapter_module):
    argv = ["--output_file=host.so", "--compile_options=-Werror"]

    args, temp_dir_ctx, auto_cleanup = compile_adapter_module.prepare_compile_context(
        argv, "host", None
    )

    assert auto_cleanup is True
    assert temp_dir_ctx is not None
    assert "-Werror" in args.compile_options
    assert "-D_GLIBCXX_USE_CXX11_ABI=1" in args.compile_options
    temp_dir_ctx.cleanup()


@pytest.mark.parametrize("abi", ["0", "1"])
def test_host_compile_keeps_explicit_abi(compile_adapter_module, abi):
    option = f"-D_GLIBCXX_USE_CXX11_ABI={abi}"
    argv = ["--output_file=host.so", f"--compile_options=-Werror {option}"]

    args, temp_dir_ctx, _ = compile_adapter_module.prepare_compile_context(
        argv, "host", None
    )

    assert args.compile_options == f"-Werror {option}"
    temp_dir_ctx.cleanup()


def test_device_compile_does_not_add_host_default_abi(compile_adapter_module):
    argv = ["--output_file=device.so", "--compile_options=-Werror"]

    args, temp_dir_ctx, _ = compile_adapter_module.prepare_compile_context(
        argv, "device", None
    )

    assert args.compile_options == "-Werror"
    temp_dir_ctx.cleanup()


def test_pgo_get_top_result_missing_file_returns_failure(
    compile_adapter_module, tmpdir
):
    missing_search = str(tmpdir.join("missing_search.txt"))

    assert compile_adapter_module.pgo_get_top_result(missing_search) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "duration", ["1.79769e+308", "1.7976931348623157e308", "NaN", "inf", "0", "-1"]
)
def test_pgo_cannot_publish_invalid_duration(
    compile_adapter_module, tmp_path, duration
):
    config = tmp_path / "fusion_config.txt"
    config.write_text("1\n1 2 # 23\n")

    assert not compile_adapter_module.pgo_generate_config(config, f"1 3 # {duration}")
    assert compile_adapter_module.pgo_read_config(config) == "1 2 # 23"


def test_pgo_atomic_publish_failure_keeps_the_previous_config(
    compile_adapter_module, monkeypatch, tmp_path
):
    config = tmp_path / "fusion_config.txt"
    config.write_text("1\n1 2 # 23\n")

    def fail_replace(source, target):
        assert target == config
        assert os.path.dirname(source) == str(tmp_path)
        raise OSError("commit failed")

    monkeypatch.setattr(compile_adapter_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="commit failed"):
        compile_adapter_module.pgo_generate_config(config, "1 3 # 22")
    assert compile_adapter_module.pgo_read_config(config) == "1 2 # 23"
    assert list(tmp_path.iterdir()) == [config]


def test_jit_compile_records_atrace_and_reports(compile_adapter_module, tmpdir, capsys):
    output_file = tmpdir.join("jit.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.jit_compile("tiling", "host", "kernel", argv)

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert [
        "InductorCompile",
        "jit_compile",
        "WriteHostSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "jit_compile",
        "WriteDeviceSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "jit_compile",
        "BuildCompiledArtifacts",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "jit_compile",
        "AutoFuseCompileTotal",
        "autofuse",
    ] in labels
    assert compile_adapter_module.duration_reports == [True]
    assert capsys.readouterr().out == ""


def test_host_compile_records_duration_without_stdout(
    compile_adapter_module, tmpdir, capsys
):
    output_file = tmpdir.join("host.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.host_compile("tiling", "host", argv)

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert [
        "InductorCompile",
        "host_compile",
        "WriteHostSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "host_compile",
        "BuildCompiledArtifacts",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "host_compile",
        "AutoFuseCompileTotal",
        "autofuse",
    ] in labels
    assert capsys.readouterr().out == ""


def test_kernel_compile_records_device_stage(compile_adapter_module, tmpdir):
    output_file = tmpdir.join("kernel.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.kernel_compile(
        "tiling", "kernel", argv, tiling_repr="AutofuseTilingData{}"
    )

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert [
        "InductorCompile",
        "kernel_compile",
        "WriteDeviceSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "kernel_compile",
        "BuildCompiledArtifacts",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "kernel_compile",
        "AutoFuseCompileTotal",
        "autofuse",
    ] in labels


def test_execute_compile_keeps_single_host_file_without_marker(
    compile_adapter_module, tmpdir
):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    args = _host_compile_args(tmpdir)

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": 'extern "C" int TilingFunc() { return 0; }',
            "kernel_impl_code": None,
        },
        args,
    )

    host_file = os.path.join(str(tmpdir), "host", "graph_tiling_func.cpp")
    assert captured["args"].host_files == host_file
    assert os.path.exists(host_file)


def test_execute_compile_merges_host_files_with_marker(compile_adapter_module, tmpdir):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    host_impl = "\n".join(
        [
            "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
            "struct CommonType {};",
            "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
            'extern "C" int Solver() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: asc_graph0_schedule_result0_g0",
            '#include "autofuse_tiling_func_common.h"',
            'extern "C" int TilingFunc() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: asc_graph0_schedule_result0_g0",
        ]
    )
    args = _host_compile_args(tmpdir)

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": host_impl,
            "kernel_impl_code": None,
        },
        args,
    )

    host_dir = os.path.join(str(tmpdir), "host")
    # host 编译为单个 cpp 源文件（cpp 段合并），header 段拆出独立 .h 供 include 引用。
    host_file = os.path.join(host_dir, "graph_tiling_func.cpp")
    assert captured["args"].host_files == host_file
    assert os.path.exists(os.path.join(host_dir, "autofuse_tiling_func_common.h"))
    with open(host_file) as f:
        merged = f.read()
    assert merged.count('#include "autofuse_tiling_func_common.h"') == 1
    assert 'extern "C" int Solver()' in merged
    assert 'extern "C" int TilingFunc()' in merged
    assert not os.path.exists(
        os.path.join(host_dir, "graph_tiling_func_solver_func.cpp")
    )


def _make_many_split_host_impl(seg_count):
    """Build host_impl with one header segment and seg_count cpp segments."""
    lines = [
        "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
        "struct CommonType {};",
        "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
    ]
    for i in range(seg_count):
        lines.append(
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: asc_graph{i}_schedule_result{i}_g0"
        )
        lines.append('#include "autofuse_tiling_func_common.h"')
        lines.append(f'extern "C" int TilingFunc{i}() {{ return {i}; }}')
        lines.append(f"// AUTOFUSE_SPLIT_FILE_END: asc_graph{i}_schedule_result{i}_g0")
    return "\n".join(lines)


def test_execute_compile_splits_host_files_when_above_threshold(
    compile_adapter_module, tmpdir
):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    threshold = compile_adapter_module.HOST_SPLIT_COMPILE_THRESHOLD
    host_impl = _make_many_split_host_impl(threshold)
    args = _host_compile_args(tmpdir)

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": host_impl,
            "kernel_impl_code": None,
        },
        args,
    )

    host_dir = os.path.join(str(tmpdir), "host")
    host_files = captured["args"].host_files
    # >= threshold: keep split files instead of merging into one cpp.
    assert isinstance(host_files, list)
    assert len(host_files) == threshold
    for source_file in host_files:
        assert os.path.exists(source_file)
    # the merged single file must NOT exist in the split path.
    assert not os.path.exists(os.path.join(host_dir, "graph_tiling_func.cpp"))


def test_execute_compile_merges_host_files_when_below_threshold(
    compile_adapter_module, tmpdir
):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    threshold = compile_adapter_module.HOST_SPLIT_COMPILE_THRESHOLD
    # one fewer segment than the threshold still merges.
    host_impl = _make_many_split_host_impl(threshold - 1)
    args = _host_compile_args(tmpdir)

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": host_impl,
            "kernel_impl_code": None,
        },
        args,
    )

    host_dir = os.path.join(str(tmpdir), "host")
    host_files = captured["args"].host_files
    assert isinstance(host_files, str)
    assert host_files == os.path.join(host_dir, "graph_tiling_func.cpp")
    assert os.path.exists(host_files)


def _long_kernel_name(op_count=45, md5="20fd52a523aa877ef4c42f6134c141de"):
    ops = ["op{}".format(i) for i in range(op_count)]
    return "autofused_{}_{}".format("_".join(ops), md5)


def test_fit_kernel_name_for_file_name_keeps_fitting_name(compile_adapter_module):
    name = "autofused_pow_sum_885af2aa3e03bf3e76a84c7c178790c9"
    suffix = "_tiling_func_solver_func.cpp"

    assert compile_adapter_module.fit_kernel_name_for_file_name(name, suffix) == name


def test_fit_kernel_name_for_file_name_truncates_ops_only(compile_adapter_module):
    md5 = "20fd52a523aa877ef4c42f6134c141de"
    name = _long_kernel_name(md5=md5)
    suffix = "_tiling_func_asc_graph0_schedule_result0_g0.cpp"
    assert len(name.encode()) + len(suffix.encode()) > 255

    fitted = compile_adapter_module.fit_kernel_name_for_file_name(name, suffix)

    marker = "_" + compile_adapter_module.KERNEL_NAME_TRUNCATE_MARKER
    hash_part = "_" + md5
    prefix_len = len("autofused_")
    hash_len = len(hash_part)
    cut_len = len(marker) + hash_len
    assert len(fitted.encode()) + len(suffix.encode()) <= 255
    assert len(fitted.encode()) + len(suffix.encode()) + len(".o") <= 255
    # autofused_ 前缀和 md5 后缀必须原样保留
    assert fitted.startswith("autofused_")
    assert fitted.endswith(hash_part)
    # 截断处有 _cut 标记
    assert fitted[prefix_len:-hash_len].endswith(marker)
    # 只整段删除算子名，保留部分是原算子名序列的前缀（不会出现半个算子名）
    kept = fitted[prefix_len:-cut_len]
    original = name[prefix_len:-hash_len]
    assert original.startswith(kept + "_") or original == kept


def test_fit_kernel_name_for_file_name_reserves_object_suffix(compile_adapter_module):
    name = _long_kernel_name(op_count=80)
    suffix = "_tiling_func_solver_func.cpp"

    fitted = compile_adapter_module.fit_kernel_name_for_file_name(name, suffix)

    source_name = fitted + suffix
    assert len(source_name.encode()) <= 255
    assert len((source_name + ".o").encode()) <= 255


def test_fit_kernel_name_for_file_name_keeps_no_hash_names_unique(
    compile_adapter_module,
):
    prefix = "kernel_" + "_".join("op{}".format(i) for i in range(80))
    suffix = "_tiling_func_solver_func.cpp"
    assert len((prefix + "_aaaa").encode()) + len(suffix.encode()) > 255

    first = compile_adapter_module.fit_kernel_name_for_file_name(
        prefix + "_aaaa", suffix
    )
    second = compile_adapter_module.fit_kernel_name_for_file_name(
        prefix + "_bbbb", suffix
    )

    assert first != second


def test_write_split_host_sources_fits_long_graph_name(tmpdir, compile_adapter_module):
    threshold = compile_adapter_module.HOST_SPLIT_COMPILE_THRESHOLD
    host_impl = _make_many_split_host_impl(threshold)

    host_files = compile_adapter_module.write_split_host_sources(
        str(tmpdir), _long_kernel_name(), host_impl
    )

    assert len(host_files) == threshold
    for source_file in host_files:
        base_name = os.path.basename(source_file)
        assert os.path.exists(source_file)
        assert len((base_name + ".o").encode()) <= 255
        assert "_cut" in base_name
        assert base_name.startswith("autofused_")
        assert "_20fd52a523aa877ef4c42f6134c141de" in base_name


def test_execute_compile_fits_long_graph_name_for_merged_host(
    compile_adapter_module, tmpdir
):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    host_impl = _make_many_split_host_impl(1)
    args = _host_compile_args(tmpdir)
    # 名字长到连合并路径的 {graph_name}_tiling_func.cpp(229 字节) 都会超限
    args.graph_name = _long_kernel_name(op_count=80)

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": host_impl,
            "kernel_impl_code": None,
        },
        args,
    )

    host_file = captured["args"].host_files
    base_name = os.path.basename(host_file)
    assert os.path.exists(host_file)
    assert len((base_name + ".o").encode()) <= 255
    assert "_cut" in base_name
    assert base_name.endswith("_tiling_func.cpp")


def test_write_split_host_sources_writes_split_headers_without_injecting_common(
    tmpdir, compile_adapter_module
):
    host_code = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingStateHeader
state header
// AUTOFUSE_SPLIT_FILE_END: TilingStateHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: TilingSolverHeader
solver header
// AUTOFUSE_SPLIT_FILE_END: TilingSolverHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: TilingApiHeader
api header
// AUTOFUSE_SPLIT_FILE_END: TilingApiHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: ACubeKernelTilingWrapperHpp
wrapper header
// AUTOFUSE_SPLIT_FILE_END: ACubeKernelTilingWrapperHpp
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
#include "autofuse_tiling_func_state.h"
#include "autofuse_tiling_func_solver.h"
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    host_files = compile_adapter_module.write_split_host_sources(
        str(tmpdir), "demo_graph", host_code
    )

    assert not tmpdir.join("autofuse_tiling_func_common.h").check()
    assert tmpdir.join("autofuse_tiling_func_state.h").read() == "state header\n"
    assert tmpdir.join("autofuse_tiling_func_solver.h").read() == "solver header\n"
    assert tmpdir.join("autofuse_tiling_func_api.h").read() == "api header\n"
    assert tmpdir.join("cube_kernel_tiling_wrapper.h").read() == "wrapper header\n"
    cpp = tmpdir.join("demo_graph_tiling_func_solver_func.cpp").read()
    assert cpp.count('#include "autofuse_tiling_func_common.h"') == 0
    assert '#include "autofuse_tiling_func_state.h"' in cpp
    assert len(host_files) == 1


def test_write_split_host_sources_legacy_marker_injects_common(
    tmpdir, compile_adapter_module
):
    host_code = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead
common header
// AUTOFUSE_SPLIT_FILE_END: TilingHead
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    compile_adapter_module.write_split_host_sources(
        str(tmpdir), "demo_graph", host_code
    )

    cpp = tmpdir.join("demo_graph_tiling_func_solver_func.cpp").read()
    assert cpp.startswith('#include "autofuse_tiling_func_common.h"')


def test_write_split_host_sources_historical_split_keeps_cpp(
    tmpdir, compile_adapter_module
):
    host_code = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingBaseHeader
base
// AUTOFUSE_SPLIT_FILE_END: TilingBaseHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    compile_adapter_module.write_split_host_sources(
        str(tmpdir), "demo_graph", host_code
    )

    cpp = tmpdir.join("demo_graph_tiling_func_solver_func.cpp").read()
    assert not cpp.startswith('#include "autofuse_tiling_func_common.h"')


@pytest.mark.parametrize(
    "header_key", ["TilingSolverHeader", "UnknownHeader", "UnknownHpp"]
)
def test_write_split_host_sources_rejects_undetermined_header_format(
    tmpdir, compile_adapter_module, header_key
):
    host_code = f"""// AUTOFUSE_SPLIT_FILE_BEGIN: {header_key}
header
// AUTOFUSE_SPLIT_FILE_END: {header_key}
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
int Solver() {{ return 0; }}
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    with pytest.raises(Exception):
        compile_adapter_module.write_split_host_sources(
            str(tmpdir), "demo_graph", host_code
        )


def _make_pgo_split_host_impl(runner_key="PgoRunner"):
    return "\n".join(
        [
            "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
            "struct CommonType {};",
            "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
            'extern "C" int Solver() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {runner_key}",
            "int main() { return Solver(); }",
            f"// AUTOFUSE_SPLIT_FILE_END: {runner_key}",
        ]
    )


def _make_scheme_a_split_host_impl(
    runner_key="PgoRunner", device_key="PgoDeviceSource"
):
    return "\n".join(
        [
            "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
            "struct CommonType {};",
            "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
            'extern "C" int Solver() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {runner_key}",
            "int main() { return Solver(); }",
            f"// AUTOFUSE_SPLIT_FILE_END: {runner_key}",
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {device_key}",
            'extern "C" __global__ __aicore__ void graph() {}',
            f"// AUTOFUSE_SPLIT_FILE_END: {device_key}",
        ]
    )


def test_write_split_host_sources_excludes_exact_pgo_runner(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))

    host_files = compile_adapter_module.write_split_host_sources(
        host_dir, "graph", _make_pgo_split_host_impl()
    )

    assert host_files == [os.path.join(host_dir, "graph_tiling_func_solver_func.cpp")]
    assert os.path.exists(os.path.join(host_dir, "graph_tiling_func_PgoRunner.cpp"))


def test_write_split_host_sources_keeps_similar_runner_key(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))

    host_files = compile_adapter_module.write_split_host_sources(
        host_dir, "graph", _make_pgo_split_host_impl("PgoRunnerExtra")
    )

    assert host_files == [
        os.path.join(host_dir, "graph_tiling_func_solver_func.cpp"),
        os.path.join(host_dir, "graph_tiling_func_PgoRunnerExtra.cpp"),
    ]


def test_write_inductor_pgo_sources_classifies_exact_versioned_splits(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))
    device_dir = str(tmpdir.mkdir("device"))

    normal_files, runner_file, device_file = (
        compile_adapter_module.write_inductor_pgo_sources(
            host_dir, device_dir, "graph", _make_scheme_a_split_host_impl()
        )
    )

    assert normal_files == [os.path.join(host_dir, "graph_tiling_func_solver_func.cpp")]
    assert runner_file == os.path.join(host_dir, "graph_tiling_func_PgoRunner.cpp")
    assert device_file == os.path.join(device_dir, "graph_pgo_device.cpp")
    assert all(
        "PgoRunner" not in path and "PgoDeviceSource" not in path
        for path in normal_files
    )
    assert "int main()" in open(runner_file).read()
    assert "__aicore__ void graph()" in open(device_file).read()


def test_write_inductor_pgo_sources_fits_long_graph_name(
    compile_adapter_module, tmpdir
):
    graph_name = _long_kernel_name(op_count=80)
    host_dir = str(tmpdir.mkdir("host"))
    device_dir = str(tmpdir.mkdir("device"))

    host_files, runner_file, device_file = (
        compile_adapter_module.write_inductor_pgo_sources(
            host_dir, device_dir, graph_name, _make_scheme_a_split_host_impl()
        )
    )

    for path in [*host_files, runner_file, device_file]:
        base_name = os.path.basename(path)
        assert os.path.exists(path)
        assert len((base_name + ".o").encode()) <= 255
        assert "_cut" in base_name
    assert os.path.basename(device_file).endswith("_pgo_device.cpp")
    assert "_20fd52a523aa877ef4c42f6134c141de" in os.path.basename(device_file)


def test_write_inductor_pgo_sources_supports_final_split_headers(
    compile_adapter_module, tmpdir
):
    host_impl = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingStateHeader
state header
// AUTOFUSE_SPLIT_FILE_END: TilingStateHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
#include "autofuse_tiling_func_state.h"
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
// AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner
int main() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: PgoRunner
// AUTOFUSE_SPLIT_FILE_BEGIN: PgoDeviceSource
extern "C" __global__ __aicore__ void graph() {}
// AUTOFUSE_SPLIT_FILE_END: PgoDeviceSource
"""
    host_dir = str(tmpdir.mkdir("host"))
    device_dir = str(tmpdir.mkdir("device"))

    normal_files, runner_file, device_file = (
        compile_adapter_module.write_inductor_pgo_sources(
            host_dir, device_dir, "graph", host_impl
        )
    )

    assert not tmpdir.join("host", "autofuse_tiling_func_common.h").check()
    assert (
        tmpdir.join("host", "autofuse_tiling_func_state.h").read() == "state header\n"
    )
    assert (
        '#include "autofuse_tiling_func_common.h"' not in open(normal_files[0]).read()
    )
    assert '#include "autofuse_tiling_func_common.h"' not in open(runner_file).read()
    assert device_file == os.path.join(device_dir, "graph_pgo_device.cpp")


@pytest.mark.parametrize(
    "host_impl",
    [
        _make_pgo_split_host_impl(),
    ],
)
def test_write_inductor_pgo_sources_rejects_missing_pair(
    compile_adapter_module, tmpdir, host_impl
):
    host_dir = str(tmpdir.mkdir("host"))
    device_dir = str(tmpdir.mkdir("device"))

    with pytest.raises(
        compile_adapter_module.ascendc_compile.CompileError, match="PGO|Pgo"
    ):
        compile_adapter_module.write_inductor_pgo_sources(
            host_dir, device_dir, "graph", host_impl
        )


def test_write_split_host_sources_keeps_similar_pgo_keys(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))
    host_impl = _make_scheme_a_split_host_impl("PgoRunnerExtra", "PgoDeviceSourceExtra")

    host_files = compile_adapter_module.write_split_host_sources(
        host_dir, "graph", host_impl
    )

    assert any("PgoRunnerExtra" in path for path in host_files)
    assert any("PgoDeviceSourceExtra" in path for path in host_files)


def test_execute_compile_prepares_scheme_a_sidecars_without_changing_host_api(
    compile_adapter_module, tmpdir, monkeypatch
):
    captured = {}

    def capture_args(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = capture_args
    _execute_scheme_a_host_compile(compile_adapter_module, tmpdir)

    compiled_args = captured["args"]
    assert compiled_args.host_files == [
        os.path.join(str(tmpdir), "host", "graph_tiling_func_solver_func.cpp")
    ]
    assert compiled_args.pgo_runner_file.endswith("graph_tiling_func_PgoRunner.cpp")
    assert compiled_args.pgo_device_file.endswith("graph_pgo_device.cpp")
    assert not hasattr(compiled_args, "pgo_mspti_config")


def test_execute_compile_scheme_a_without_mspti_keeps_pgo_proxy_runtime_linkage(
    compile_adapter_module, tmpdir
):
    captured = {}

    def capture_args(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = capture_args
    _execute_scheme_a_host_compile(compile_adapter_module, tmpdir)

    compiled_args = captured["args"]
    assert compiled_args.pgo_runner_file.endswith("graph_tiling_func_PgoRunner.cpp")
    assert compiled_args.pgo_device_file.endswith("graph_pgo_device.cpp")
    assert not hasattr(compiled_args, "pgo_mspti_config")


def test_execute_compile_scheme_a_supports_stage_all(
    compile_adapter_module, tmpdir, monkeypatch
):
    captured = {}

    def capture_args(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = capture_args
    args = type(
        "Args",
        (),
        {
            "stage": "all",
            "temp_dir": str(tmpdir),
            "graph_name": "graph",
            "trace_stage": "jit_compile",
        },
    )()

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": _make_scheme_a_split_host_impl(),
            "kernel_impl_code": 'extern "C" void kernel() {}',
        },
        args,
    )

    compiled_args = captured["args"]
    assert compiled_args.stage == "all"
    assert compiled_args.host_files == [
        os.path.join(str(tmpdir), "host", "graph_tiling_func_solver_func.cpp")
    ]
    assert compiled_args.pgo_runner_file.endswith("graph_tiling_func_PgoRunner.cpp")
    assert compiled_args.pgo_device_file.endswith("graph_pgo_device.cpp")
    assert compiled_args.device_files.endswith("graph_op_kernel.cpp")


@pytest.mark.parametrize(
    "host_impl",
    [
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "int Solver2() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: ../solver",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: ../solver",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: a/b",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: a/b",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: a\\b",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: a\\b",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: ",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: ",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            ]
        ),
    ],
)
def test_execute_compile_rejects_invalid_split_marker(
    compile_adapter_module, tmpdir, host_impl
):
    def fake_main(args):
        return None

    compile_adapter_module.ascendc_compile.main = fake_main
    args = _host_compile_args(tmpdir)

    with pytest.raises(compile_adapter_module.ascendc_compile.CompileError) as exc_info:
        compile_adapter_module.execute_compile(
            {
                "tiling_struct_code": "struct AutofuseTilingData {};",
                "host_impl_code": host_impl,
                "kernel_impl_code": None,
            },
            args,
        )

    assert "split host source" in str(
        exc_info.value
    ) or "invalid split host source key" in str(exc_info.value)


def test_host_compile_and_jit_compile_signatures_keep_stable(compile_adapter_module):
    assert list(inspect.signature(compile_adapter_module.jit_compile).parameters) == [
        "tiling_def",
        "host_tiling",
        "op_kernel",
        "argv",
    ]
    assert list(inspect.signature(compile_adapter_module.host_compile).parameters) == [
        "tiling_def_code",
        "tiling_impl_code",
        "argv",
    ]


def test_parse_compile_args_reads_const_tiling_baked(compile_adapter_module):
    args, _, _ = compile_adapter_module.prepare_compile_context(
        ["--output_file=x.so", "--const_tiling_baked=True"], "link", None
    )
    assert args.const_tiling_baked is True

    args, _, _ = compile_adapter_module.prepare_compile_context(
        ["--output_file=x.so", "--const_tiling_baked=False"], "link", None
    )
    assert args.const_tiling_baked is False

    args, _, _ = compile_adapter_module.prepare_compile_context(
        ["--output_file=x.so"], "link", None
    )
    assert args.const_tiling_baked is False


def test_compile_result_carries_baked_for_device_obj(compile_adapter_module):
    result = {
        "kernel_obj_path": "/tmp/k.o",
        "kernel_source_path": "/tmp/k.cpp",
        "baked": True,
    }
    out = compile_adapter_module._compile_result(
        types.SimpleNamespace(stage="device_obj"), result
    )
    assert out["baked"] is True

    out = compile_adapter_module._compile_result(
        types.SimpleNamespace(stage="device_obj"),
        {"kernel_obj_path": "/tmp/k.o", "kernel_source_path": "/tmp/k.cpp"},
    )
    assert out["baked"] is False


def test_parse_compile_args_keeps_cv_core_limits_explicit(compile_adapter_module):
    limited, _, _ = compile_adapter_module.prepare_compile_context(
        ["--output_file=x.so", "--aic_num=5", "--aiv_num=7"], "device_obj", None
    )
    assert (limited.aic_num, limited.aiv_num) == (5, 7)
    defaults, _, _ = compile_adapter_module.prepare_compile_context(
        ["--output_file=x.so"], "all", None
    )
    assert (defaults.aic_num, defaults.aiv_num) == (0, 0)
