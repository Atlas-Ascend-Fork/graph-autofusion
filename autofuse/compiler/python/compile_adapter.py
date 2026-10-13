#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import tempfile
import argparse
import time
from typing import List
from autofuse import ascendc_compile
import re
import tbe.common.utils.log as logger


def _log_warning(message, *args):
    warning = getattr(logger, "warning", None)
    if warning is None:
        warning = getattr(logger, "warn")
    warning(message, *args)


HOST_DEFAULT_CXX11_ABI = "-D_GLIBCXX_USE_CXX11_ABI=1"
HOST_CXX11_ABI_PREFIX = "-D_GLIBCXX_USE_CXX11_ABI="
INDUCTOR_COMPILE_TRACE_LABEL = "InductorCompile"
SOURCES_TILING_STRUCT = "tiling_struct_code"
SOURCES_HOST_IMPL = "host_impl_code"
SOURCES_KERNEL_IMPL = "kernel_impl_code"
SPLIT_BEGIN_PREFIX = "// AUTOFUSE_SPLIT_FILE_BEGIN:"
SPLIT_END_PREFIX = "// AUTOFUSE_SPLIT_FILE_END:"
SPLIT_HEADER_KEY = "TilingHead"
SPLIT_HEADER_INCLUDE = '#include "autofuse_tiling_func_common.h"'
SPLIT_HEADER_FILES = {
    "TilingHead": "autofuse_tiling_func_common.h",
    "TilingStateHeader": "autofuse_tiling_func_state.h",
    "TilingLogHeader": "autofuse_tiling_func_log.h",
    "TilingPgoHeader": "autofuse_tiling_func_pgo.h",
    "TilingBaseHeader": "autofuse_tiling_func_base.h",
    "TilingSolverHeader": "autofuse_tiling_func_solver.h",
    "TilingApiHeader": "autofuse_tiling_func_api.h",
    "TilingEntryHeader": "autofuse_tiling_func_entry.h",
    "TilingTailHeader": "autofuse_tiling_func_tail.h",
    "ACubeKernelTilingWrapperHpp": "cube_kernel_tiling_wrapper.h",
}
TILING_HEADER_FILES = dict(SPLIT_HEADER_FILES)
TILING_HEADER_FILES["CubeKernelTilingWrapperHpp"] = "cube_kernel_tiling_wrapper.h"
FINAL_SPLIT_DISCRIMINATOR_KEYS = {"TilingStateHeader"}
HISTORICAL_SPLIT_DISCRIMINATOR_KEYS = {
    "TilingBaseHeader",
    "TilingEntryHeader",
    "TilingTailHeader",
}
SPLIT_PGO_RUNNER_KEY = "PgoRunner"
SPLIT_PGO_DEVICE_SOURCE_KEY = "PgoDeviceSource"
CANN_ROOT_ENV_NAMES = ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_HOME")
# 当 host 拆分后的 cpp 段数量达到该阈值时，保留多文件拆分并并发编译，
# 否则合并为单个 cpp 文件串行编译（默认路径）。
HOST_SPLIT_COMPILE_THRESHOLD = 10


def str2bool(v):
    v_lower = v.lower()
    if v_lower in ["true", "1", "yes", "y"]:
        return True
    elif v_lower in ["false", "0", "no", "n"]:
        return False
    else:
        raise ValueError(f"Invalid boolean value: '{v}'")


def camel_to_snake(camel_str):
    # 使用正则表达式匹配大写字母
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    # 使用正则表达式匹配小写字母后跟大写字母的情况
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def gen_valid_name(t_name):
    result = []
    last_was_underscore = False

    for c in t_name:
        if c.isalnum():
            result.append(c)
            last_was_underscore = False
        else:
            if not last_was_underscore:
                result.append("_")
                last_was_underscore = True

    ret_name = "".join(result)

    # 删除开头的下划线
    if ret_name and ret_name[0] == "_":
        ret_name = ret_name[1:]

    # 如果以数字开头，添加前缀
    if ret_name and ret_name[0].isdigit():
        ret_name = "t_" + ret_name

    return ret_name


def parse_compile_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_name", default="autofuse", type=str, help="Graph name."
    )
    parser.add_argument(
        "--output_file", required=True, type=str, help="Destination directory."
    )
    parser.add_argument("--output_path", default="", type=str, help="Output directory.")
    parser.add_argument(
        "--force_unknown", default=False, type=str2bool, help="force unknown shape."
    )
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="PGO tiling config file after turning.",
    )
    parser.add_argument(
        "--soc_version", default="Ascend910B", type=str, help="chip soc version."
    )
    parser.add_argument(
        "--compile_options",
        default="",
        type=str,
        help="Compile options of tiling and kernel.",
    )
    parser.add_argument("--tiling_obj_paths", default="", type=str)
    parser.add_argument("--tiling_source_paths", default="", type=str)
    parser.add_argument("--kernel_obj_path", default="", type=str)
    parser.add_argument("--kernel_source_path", default="", type=str)
    parser.add_argument("--shared_cv_wrapper_so", default="", type=str)
    # 使用 parse_known_args 容忍上层透传的未声明参数，避免触发 SystemExit 终止编译。
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.info("[CompileArgs] ignored unrecognized arguments: %s", unknown)
    return args


def generate_file(dst_dir, file_name, text):
    os.makedirs(dst_dir, exist_ok=True)
    file_path = os.path.join(dst_dir, file_name)
    with open(file_path, "w") as file:
        file.write(text)


def has_split_host_marker(host_impl_code):
    return SPLIT_BEGIN_PREFIX in host_impl_code or SPLIT_END_PREFIX in host_impl_code


def validate_split_key(key):
    if not key:
        raise ascendc_compile.CompileError(f"invalid split host source key: {key}")
    if any(token in key for token in ("/", "\\", "..")):
        raise ascendc_compile.CompileError(f"invalid split host source key: {key}")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", key) is None:
        raise ascendc_compile.CompileError(f"invalid split host source key: {key}")


def parse_split_marker(line, prefix):
    stripped = line.strip()
    if not stripped.startswith(prefix):
        return None
    key = stripped[len(prefix) :].strip()
    validate_split_key(key)
    return key


def finish_split_source(key, lines, headers, cpp_sources, seen_keys):
    if key in seen_keys:
        raise ascendc_compile.CompileError(
            f"split host source key is duplicated: {key}"
        )
    seen_keys.add(key)
    content = "".join(lines)
    if key in SPLIT_HEADER_FILES:
        headers[key] = content
        return headers, cpp_sources
    if key.endswith("Header") or key.endswith("Hpp"):
        raise ascendc_compile.CompileError(f"unknown split host header key: {key}")
    cpp_sources.append((key, content))
    return headers, cpp_sources


def validate_split_sources(headers, cpp_sources):
    discriminator_keys = (
        FINAL_SPLIT_DISCRIMINATOR_KEYS | HISTORICAL_SPLIT_DISCRIMINATOR_KEYS
    )
    is_split_format = bool(discriminator_keys & set(headers))
    if not is_split_format and SPLIT_HEADER_KEY not in headers:
        raise ascendc_compile.CompileError("split host source has no TilingHead")
    if not cpp_sources:
        raise ascendc_compile.CompileError("split host source has no cpp source")


def parse_split_host_sources(host_impl_code):
    current_key = None
    current_lines = []
    headers = {}
    cpp_sources = []
    seen_keys = set()
    for line in host_impl_code.splitlines(keepends=True):
        begin_key = parse_split_marker(line, SPLIT_BEGIN_PREFIX)
        end_key = parse_split_marker(line, SPLIT_END_PREFIX)
        if begin_key is not None and end_key is not None:
            raise ascendc_compile.CompileError("split host source marker is invalid")
        if begin_key is not None:
            if current_key is not None:
                raise ascendc_compile.CompileError(
                    f"split host source marker is nested: {begin_key}"
                )
            current_key = begin_key
            current_lines = []
            continue
        if end_key is not None:
            if current_key is None:
                raise ascendc_compile.CompileError(
                    f"split host source marker end without begin: {end_key}"
                )
            if current_key != end_key:
                raise ascendc_compile.CompileError(
                    f"split host source marker mismatch: begin={current_key}, end={end_key}"
                )
            headers, cpp_sources = finish_split_source(
                current_key, current_lines, headers, cpp_sources, seen_keys
            )
            current_key = None
            current_lines = []
            continue
        if current_key is None:
            if line.strip():
                raise ascendc_compile.CompileError(
                    "split host source has content outside marker"
                )
            continue
        current_lines.append(line)
    if current_key is not None:
        raise ascendc_compile.CompileError(
            f"split host source marker is not closed: {current_key}"
        )
    validate_split_sources(headers, cpp_sources)
    return headers, cpp_sources


def add_split_header_include(cpp_content):
    if SPLIT_HEADER_INCLUDE in cpp_content:
        return cpp_content
    return SPLIT_HEADER_INCLUDE + "\n" + cpp_content


def is_versioned_split_format(headers):
    return bool(
        (FINAL_SPLIT_DISCRIMINATOR_KEYS | HISTORICAL_SPLIT_DISCRIMINATOR_KEYS)
        & set(headers)
    )


def write_split_source_files(
    host_file_path, graph_name, headers, cpp_sources, excluded_keys
):
    for key, content in headers.items():
        generate_file(host_file_path, SPLIT_HEADER_FILES[key], content)
    source_files = {}
    is_split_format = is_versioned_split_format(headers)
    for key, cpp_content in cpp_sources:
        if key in excluded_keys:
            continue
        file_name = f"{graph_name}_tiling_func_{key}.cpp"
        content = (
            cpp_content if is_split_format else add_split_header_include(cpp_content)
        )
        generate_file(host_file_path, file_name, content)
        source_files[key] = os.path.join(host_file_path, file_name)
    return source_files


def write_split_host_source_groups(host_file_path, graph_name, host_impl_code):
    headers, cpp_sources = parse_split_host_sources(host_impl_code)
    source_files = write_split_source_files(
        host_file_path,
        graph_name,
        headers,
        cpp_sources,
        {SPLIT_PGO_DEVICE_SOURCE_KEY},
    )
    runner_file = source_files.pop(SPLIT_PGO_RUNNER_KEY, None)
    return list(source_files.values()), runner_file


def write_split_host_sources(host_file_path, graph_name, host_impl_code):
    host_files, _ = write_split_host_source_groups(
        host_file_path, graph_name, host_impl_code
    )
    return host_files


def write_inductor_pgo_sources(host_dir, device_dir, graph_name, host_impl_code):
    headers, cpp_sources = parse_split_host_sources(host_impl_code)
    source_map = dict(cpp_sources)
    runner_content = source_map.get(SPLIT_PGO_RUNNER_KEY)
    device_content = source_map.get(SPLIT_PGO_DEVICE_SOURCE_KEY)
    if runner_content is None or device_content is None:
        raise ascendc_compile.CompileError(
            "PgoRunner and PgoDeviceSource must be generated together"
        )
    source_files = write_split_source_files(
        host_dir,
        graph_name,
        headers,
        cpp_sources,
        {SPLIT_PGO_DEVICE_SOURCE_KEY},
    )
    runner_file = source_files.pop(SPLIT_PGO_RUNNER_KEY)
    device_name = f"{graph_name}_pgo_device.cpp"
    generate_file(device_dir, device_name, device_content)
    return (
        list(source_files.values()),
        runner_file,
        os.path.join(device_dir, device_name),
    )


def has_inductor_pgo_split(host_impl_code):
    _, cpp_sources = parse_split_host_sources(host_impl_code)
    keys = {key for key, _ in cpp_sources}
    return bool(keys & {SPLIT_PGO_RUNNER_KEY, SPLIT_PGO_DEVICE_SOURCE_KEY})


def write_single_host_source(host_file_path, base_host_file, host_impl_code):
    generate_file(host_file_path, base_host_file, host_impl_code)
    return os.path.join(host_file_path, base_host_file)


def write_merged_host_sources(host_file_path, base_host_file, host_impl_code):
    """Generate one host source file while preserving split headers.

    cpp 段合并为单个源文件（单文件编译），header 段仍拆出为独立 .h 文件，
    供 cpp 段中间的 #include "autofuse_tiling_func_*.h" 引用。
    """
    if not has_split_host_marker(host_impl_code):
        return write_single_host_source(host_file_path, base_host_file, host_impl_code)

    headers, cpp_sources = parse_split_host_sources(host_impl_code)
    for key, content in headers.items():
        generate_file(host_file_path, SPLIT_HEADER_FILES[key], content)

    is_split_format = is_versioned_split_format(headers)
    merged = []
    for _key, cpp_content in cpp_sources:
        if is_split_format:
            content = cpp_content
        else:
            content = add_split_header_include(cpp_content)
            if merged and content.startswith(SPLIT_HEADER_INCLUDE):
                content = content[len(SPLIT_HEADER_INCLUDE):]  # fmt: skip
        merged.append(content.rstrip("\n"))
    generate_file(host_file_path, base_host_file, "\n".join(merged) + "\n")
    return os.path.join(host_file_path, base_host_file)


def write_host_sources(host_file_path, base_host_file, graph_name, host_impl_code):
    # host 编译默认合并为单个 cpp 源文件（cpp 段合并），header 段拆出独立 .h 供 include 引用。
    # 当拆分后的 cpp 段数量较多（>= HOST_SPLIT_COMPILE_THRESHOLD）时，保留多文件拆分，
    # 交由 compile_host_objs 并发编译，缩短整体 host 编译耗时。
    if has_split_host_marker(host_impl_code):
        _, cpp_sources = parse_split_host_sources(host_impl_code)
        if len(cpp_sources) >= HOST_SPLIT_COMPILE_THRESHOLD:
            return write_split_host_sources(host_file_path, graph_name, host_impl_code)
    return write_merged_host_sources(host_file_path, base_host_file, host_impl_code)


def parse_env_flags(env_name):
    result = {}
    flags = os.getenv(env_name)
    if not flags:
        return result
    params = flags.split(";")
    for param in params:
        if "=" in param:
            key_part, value_part = param.split("=", 1)
            key = key_part.lstrip("-")
            result[key] = value_part
    return result


def get_dfx_env_result():
    return parse_env_flags("AUTOFUSE_DFX_FLAGS")


def get_debug_flag():
    dfx_dict = get_dfx_env_result()
    return dfx_dict.get("codegen_compile_debug", "false").lower() == "true"


def record_inductor_compile_duration(stage, step, graph_name, start, duration):
    from autofuse.pyautofuse import ascir

    labels = [INDUCTOR_COMPILE_TRACE_LABEL, stage, step, graph_name]
    ascir.utils.duration_record(labels, int(start), int(duration))


def report_inductor_compile_durations():
    from autofuse.pyautofuse import ascir

    ascir.utils.report_durations()


class InductorCompileDuration:
    def __init__(self, stage, step, graph_name):
        self.stage = stage
        self.step = step
        self.graph_name = graph_name
        self.start = None

    def __enter__(self):
        self.start = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = time.time_ns()
        record_inductor_compile_duration(
            self.stage, self.step, self.graph_name, self.start, end - self.start
        )
        return False


def get_pgo_topn():
    default_topn = 5
    dfx_dict = get_dfx_env_result()
    topn_str = dfx_dict.get("autofuse_pgo_topn", str(default_topn))
    try:
        topn = int(topn_str)
        if topn < 0:
            return default_topn
        return topn
    except ValueError:
        return default_topn


def get_pgo_env_flag():
    result = parse_env_flags("AUTOFUSE_FLAGS")
    return result.get("autofuse_enable_pgo", "false").lower() == "true"


def prepare_compile_context(argv, stage, tiling_repr, bin_mode=False):
    args = parse_compile_args(argv)
    args.stage = stage
    args.tiling_repr = tiling_repr
    args.bin_mode = bin_mode
    get_pgo_cache_root = getattr(ascendc_compile, "get_pgo_cache_root", None)
    args.pgo_cache_root = (
        get_pgo_cache_root(args.output_file)
        if callable(get_pgo_cache_root)
        else f"{os.path.realpath(args.output_file)}.pgo_v2"
    )
    if (
        stage in ("host", "host_obj")
        and HOST_CXX11_ABI_PREFIX not in args.compile_options
    ):
        args.compile_options = (
            args.compile_options + " " + HOST_DEFAULT_CXX11_ABI
        ).strip()

    args.graph_name = camel_to_snake(gen_valid_name(args.graph_name))
    auto_cleanup = not args.output_path and not get_debug_flag()

    if auto_cleanup:
        temp_dir_ctx = tempfile.TemporaryDirectory()
        args.temp_dir = temp_dir_ctx.name
        return args, temp_dir_ctx, True
    args.temp_dir = args.output_path if args.output_path else tempfile.mkdtemp()
    return args, None, False


def write_compile_host_sources(sources, args, tiling_def_file, base_host_file):
    host_file_path = os.path.join(args.temp_dir, "host")
    generate_file(host_file_path, tiling_def_file, sources[SOURCES_TILING_STRUCT])
    host_impl_code = sources[SOURCES_HOST_IMPL]
    if not (
        has_split_host_marker(host_impl_code) and has_inductor_pgo_split(host_impl_code)
    ):
        args.host_files = write_host_sources(
            host_file_path, base_host_file, args.graph_name, host_impl_code
        )
        return
    if args.stage not in ("all", "host", "host_obj"):
        raise ascendc_compile.CompileError(
            "Inductor PGO sidecar is supported only in all/host_compile stages"
        )
    device_file_path = os.path.join(args.temp_dir, "device")
    generate_file(device_file_path, tiling_def_file, sources[SOURCES_TILING_STRUCT])
    args.host_files, args.pgo_runner_file, args.pgo_device_file = (
        write_inductor_pgo_sources(
            host_file_path, device_file_path, args.graph_name, host_impl_code
        )
    )


def write_compile_device_sources(sources, args, tiling_def_file, base_device_file):
    device_file_path = os.path.join(args.temp_dir, "device")
    generate_file(device_file_path, tiling_def_file, sources[SOURCES_TILING_STRUCT])
    generate_file(device_file_path, base_device_file, sources[SOURCES_KERNEL_IMPL])
    args.device_files = os.path.join(device_file_path, base_device_file)


def execute_compile(sources, args):
    tiling_def_file = "autofuse_tiling_data.h"
    base_host_file = args.graph_name + "_tiling_func.cpp"
    base_device_file = args.graph_name + "_op_kernel.cpp"
    if args.stage in ["all", "host", "host_obj"]:
        with InductorCompileDuration(
            args.trace_stage, "WriteHostSource", args.graph_name
        ):
            write_compile_host_sources(sources, args, tiling_def_file, base_host_file)
    if args.stage in ["all", "device", "device_obj"]:
        with InductorCompileDuration(
            args.trace_stage, "WriteDeviceSource", args.graph_name
        ):
            write_compile_device_sources(
                sources, args, tiling_def_file, base_device_file
            )

    with InductorCompileDuration(
        args.trace_stage, "BuildCompiledArtifacts", args.graph_name
    ):
        result = ascendc_compile.main(args)
    if result is not None:
        return _compile_result(args, result)
    if args.stage == "link":
        return args.output_file
    return args.temp_dir


def _compile_result(args, result):
    if args.stage == "host_obj":
        return {
            "version": 1,
            "stage": args.stage,
            "tiling_obj_paths": result["tiling_obj_paths"],
            "tiling_source_paths": result["tiling_source_paths"],
            "shared_cv_wrapper_so": result.get("shared_cv_wrapper_so"),
        }
    if args.stage == "device_obj":
        return {
            "version": 1,
            "stage": args.stage,
            "kernel_obj_path": result["kernel_obj_path"],
            "kernel_source_path": result["kernel_source_path"],
        }
    return result


def compile_core(
    sources,
    argv: List[str],
    stage="all",
    tiling_repr=None,
    trace_stage=None,
    bin_mode=False,
):
    args = None
    total_start = time.time_ns()
    try:
        prepare_start = time.time_ns()
        args, temp_dir_ctx, auto_cleanup = prepare_compile_context(
            argv, stage, tiling_repr, bin_mode
        )
        args.trace_stage = trace_stage if trace_stage is not None else stage
        prepare_end = time.time_ns()
        record_inductor_compile_duration(
            args.trace_stage,
            "PrepareCompileContext",
            args.graph_name,
            prepare_start,
            prepare_end - prepare_start,
        )
        if not auto_cleanup:
            return execute_compile(sources, args)
        with temp_dir_ctx:
            return execute_compile(sources, args)
    finally:
        if args is not None:
            total_end = time.time_ns()
            record_inductor_compile_duration(
                args.trace_stage,
                "AutoFuseCompileTotal",
                args.graph_name,
                total_start,
                total_end - total_start,
            )
            report_inductor_compile_durations()


def jit_compile(tiling_def, host_tiling, op_kernel, argv: List[str]):
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def,
            SOURCES_HOST_IMPL: host_tiling,
            SOURCES_KERNEL_IMPL: op_kernel,
        },
        argv,
        trace_stage="jit_compile",
    )


def jit_compile_bin(tiling_def, host_tiling, op_kernel, argv: List[str]):
    """jit 编译的 rtLaunch 变体：链接产物导出 GetKernelBin 而非 AutofuseLaunch。

    与 ``jit_compile`` 的区别仅在链接阶段——device .o 不再直接链入 so，而是抽出
    aicore 二进制并由 ``GetKernelBin`` 导出。旧 ``jit_compile`` 保持不变，供 CV 融合
    等尚未迁移到 rtKernelLaunchWithHandleV2 的场景继续使用。
    """
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def,
            SOURCES_HOST_IMPL: host_tiling,
            SOURCES_KERNEL_IMPL: op_kernel,
        },
        argv,
        trace_stage="jit_compile_bin",
        bin_mode=True,
    )


def host_compile(tiling_def_code, tiling_impl_code, argv: List[str]):
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def_code,
            SOURCES_HOST_IMPL: tiling_impl_code,
            SOURCES_KERNEL_IMPL: None,
        },
        argv,
        "host",
        trace_stage="host_compile",
    )


def kernel_compile(
    tiling_def_code, kernel_impl_code, argv: List[str], *, tiling_repr=None
):
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def_code,
            SOURCES_HOST_IMPL: None,
            SOURCES_KERNEL_IMPL: kernel_impl_code,
        },
        argv,
        "device",
        tiling_repr,
        trace_stage="kernel_compile",
    )


def kernel_compile_bin(
    tiling_def_code, kernel_impl_code, argv: List[str], *, tiling_repr=None
):
    """静态 shape（TopN）kernel 编译的 rtLaunch 变体：链接产物导出 GetKernelBin。

    与 ``kernel_compile`` 的区别仅在链接阶段——静态 kernel.so 仅导出 GetKernelBin，
    tiling 已由 ``tiling_repr`` 固化进 device 代码。旧 ``kernel_compile`` 保持不变。
    """
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def_code,
            SOURCES_HOST_IMPL: None,
            SOURCES_KERNEL_IMPL: kernel_impl_code,
        },
        argv,
        "device",
        tiling_repr,
        trace_stage="kernel_compile_bin",
        bin_mode=True,
    )


def build_tiling_obj(tiling_def_code, tiling_impl_code, argv: List[str]):
    """编译 host 侧 tiling 函数为 .o，不链接。供外层（torchair）并行调度。"""
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def_code,
            SOURCES_HOST_IMPL: tiling_impl_code,
            SOURCES_KERNEL_IMPL: None,
        },
        argv,
        "host_obj",
        trace_stage="build_tiling_obj",
    )


def build_kernel_obj(
    tiling_def_code, kernel_impl_code, argv: List[str], *, tiling_repr=None
):
    """编译 device 侧 kernel 代码为 .o，不链接。供外层（torchair）并行调度。"""
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def_code,
            SOURCES_HOST_IMPL: None,
            SOURCES_KERNEL_IMPL: kernel_impl_code,
        },
        argv,
        "device_obj",
        tiling_repr,
        trace_stage="build_kernel_obj",
    )


def build_kernel_so(tiling_def_code, argv: List[str], *, tiling_repr=None):
    """链接 host.o + device.o，产出最终 kernel.so。"""
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def_code,
            SOURCES_HOST_IMPL: None,
            SOURCES_KERNEL_IMPL: None,
        },
        argv,
        "link",
        tiling_repr,
        trace_stage="build_kernel_so",
    )


def build_kernel_so_bin(tiling_def_code, argv: List[str], *, tiling_repr=None):
    """链接 host.o + GetKernelBin host.o，产出导出 GetKernelBin 的 kernel.so。

    对应 ``build_kernel_so`` 的 rtLaunch 变体：链接阶段把 device .o 抽出 aicore 二进制
    并生成 GetKernelBin host 对象后链入，不再链接 AutofuseLaunch。旧 ``build_kernel_so``
    保持不变，供 CV 融合等尚未迁移的场景继续使用。
    """
    return compile_core(
        {
            SOURCES_TILING_STRUCT: tiling_def_code,
            SOURCES_HOST_IMPL: None,
            SOURCES_KERNEL_IMPL: None,
        },
        argv,
        "link",
        tiling_repr,
        trace_stage="build_kernel_so_bin",
        bin_mode=True,
    )


def get_current_cann_root():
    module_file = getattr(ascendc_compile, "__file__", "")
    if not module_file:
        return None
    return os.path.realpath(
        os.path.join(os.path.dirname(module_file), "..", "..", "..")
    )


def get_inductor_pgo_cann_root_candidates():
    candidates = []

    def add_candidate(path):
        if not path:
            return
        real_path = os.path.realpath(path)
        if real_path not in candidates:
            candidates.append(real_path)

    add_candidate(get_current_cann_root())
    for env_name in CANN_ROOT_ENV_NAMES:
        add_candidate(os.getenv(env_name))
    return candidates


def extract_time(line):
    try:
        time_str = line.split("#")[-1].strip()
        if time_str == "1.79769e+308":  # 采样失败的返回值
            return float("inf")
        return float(time_str)
    except (ValueError, IndexError):
        return float("inf")


def pgo_get_top_result(search_path, top_n=5):
    try:
        with open(search_path, "r") as file:
            lines = [line.strip() for line in file if line.strip()]
    except OSError:
        return None, None, None

    if not lines:
        return None, None, None

    origin_line = lines[-1]
    solution_set_line = lines[:-1]

    sorted_lines = sorted(solution_set_line, key=extract_time)
    if top_n == 0 or top_n > len(sorted_lines):
        top_lines = sorted_lines
    else:
        top_lines = sorted_lines[:top_n]

    return top_lines, origin_line, top_n


def pgo_write_config(config_path, tiling_data, is_last_result=False):
    # 写入配置文件
    # 只有调优结束后，才写1标记内存复用，否则写0强制每次读文件
    with open(config_path, "w") as file:
        if is_last_result:
            file.write("1\n")
        else:
            file.write("0\n")
        file.write(f"{tiling_data}\n")
        file.flush()


def pgo_generate_config(search_path, config_path, topn=5):
    with open(search_path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    target_lines = lines[-(topn + 1) :]

    result = min(target_lines, key=extract_time)
    if extract_time(result) == float("inf"):
        result = lines[-1]
    pgo_write_config(config_path, result, is_last_result=True)
