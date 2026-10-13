#!/usr/bin/env python3
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
import ctypes
import fcntl
import hashlib
import json
import os
import re
import sys
import tbe.common.utils.log as logger
import shutil
import argparse
import subprocess
import platform
import tempfile
import uuid
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from dataclasses import dataclass
from functools import wraps
from typing import List
from asc_op_compile_base.common.platform.platform_info import get_soc_spec


def _log_warning(message, *args):
    warning = getattr(logger, "warning", None)
    if warning is None:
        warning = getattr(logger, "warn")
    warning(message, *args)


PYF_PATH = os.path.dirname(os.path.realpath(__file__))
ASCEND_PATH = os.path.join(PYF_PATH, "..", "..", "..")
machine = platform.machine()
HOST_LINK_LIBRARIES = ["tiling_api", "platform", "graph_base", "register", "acl_rt"]
CV_HOST_LINK_LIBRARIES = HOST_LINK_LIBRARIES + ["nnopbase"]
INDUCTOR_COMPILE_TRACE_LABEL = "InductorCompile"
HOST_COMPILE_MAX_WORKERS = 32
HOST_CPP_STANDARD = "-std=c++17"
CV_WRAPPER_SOURCE_NAME = "cube_kernel_tiling_wrapper.cpp"
CV_WRAPPER_SPLIT_KEY = "BCubeKernelTilingWrapperCpp"
CV_WRAPPER_CACHE_DIR_NAME = "cv_tiling_wrapper_cache"
CV_WRAPPER_SO_BASENAME = "libautofuse_cv_tiling_wrapper"
CV_WRAPPER_RPATH_OPTION = f"-Wl,-rpath,$ORIGIN/{CV_WRAPPER_CACHE_DIR_NAME}"
PGO_BUNDLE_SCHEMA_VERSION = 2
PGO_RESULT_PROTOCOL_VERSION = 2
PGO_KERNEL_FORMAT = "aicore_binary_elf_v1"
if not os.path.exists(ASCEND_PATH):
    ASCEND_PATH = os.getenv("ASCEND_HOME_PATH", ASCEND_PATH)

PCH_FILENAME = "autofuse_tiling_pch.h"
PCH_OUTPUT_NAME = "autofuse_tiling_pch.h.gch"
PCH_CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "autofuse_pch_cache")
COMPILE_TRACE_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache", "autofuse_compile_trace"
)
PCH_COMMON_HEADERS = [
    "<algorithm>",
    "<cmath>",
    "<cstdint>",
    "<cstring>",
    "<functional>",
    "<iostream>",
    "<map>",
    "<memory>",
    "<sstream>",
    "<string>",
    "<type_traits>",
    "<unordered_map>",
    "<utility>",
    "<vector>",
]


def get_dfx_flag(name):
    flags = os.getenv("AUTOFUSE_DFX_FLAGS", "")
    for flag in flags.split(";"):
        if "=" not in flag:
            continue
        key, value = flag.split("=", 1)
        if key.lstrip("-") == name:
            return value
    return ""


class CompileError(Exception):
    """Compile failed exception."""


@dataclass(frozen=True)
class PgoBundle:
    tiling_file: str
    runner_file: str
    kernel_file: str
    output_file: str
    generation: str
    ld_preload: str = ""


def record_inductor_compile_duration(stage, step, graph_name, start, duration):
    from autofuse.pyautofuse import ascir

    labels = [INDUCTOR_COMPILE_TRACE_LABEL, stage, step, graph_name]
    ascir.utils.duration_record(labels, int(start), int(duration))


class InductorCompileDuration:
    def __init__(self, args, step):
        self.stage = getattr(args, "trace_stage", getattr(args, "stage", "unknown"))
        self.step = step
        self.graph_name = getattr(args, "graph_name", "unknown")
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


def inductor_compile_duration(step, args_index=0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            compile_args = (
                args[args_index] if len(args) > args_index else kwargs.get("args")
            )
            with InductorCompileDuration(compile_args, step):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_soc_type(args):
    """根据 soc_version 返回对应的类型"""
    if args.soc_version.startswith("Ascend910B"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend910_93"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend950"):
        return "dav-3510"
    else:
        raise ValueError(f"Unsupported soc_version: {args.soc_version}")


def get_compile_diagnostic_flags(output_file):
    """Return opt-in Bisheng flags for compiler diagnosis.

    ``codegen_compile_debug=true`` enables the per-pass timing report and a
    compiler timeline trace.  Per-pass timing is used instead of per-pass-run
    timing to limit diagnostic overhead and output volume.
    """
    if get_dfx_flag("codegen_compile_debug").lower() != "true":
        return []

    os.makedirs(COMPILE_TRACE_ROOT, exist_ok=True)
    trace_file = os.path.join(
        COMPILE_TRACE_ROOT,
        f"{os.path.basename(output_file)}.{uuid.uuid4().hex}.json",
    )
    logger.info("[CompileTrace] %s", trace_file)
    return [
        "-ftime-report=per-pass",
        f"-ftime-trace={trace_file}",
    ]


def run_compile_command(cmd: List[str], stage_name):
    diagnostics_enabled = get_dfx_flag("codegen_compile_debug").lower() == "true"
    result = subprocess.run(
        cmd,
        capture_output=not diagnostics_enabled,
        text=True,
    )
    if result.returncode != 0:
        error_msg = f"{stage_name} compile failed with code {result.returncode}"
        if result.stderr:
            error_msg += f"\nstderr: {result.stderr}"
        raise CompileError(error_msg)
    if result.stdout:
        logger.info("[%s] %s", stage_name, result.stdout)


def link_shared(target_file, obj_files, link_libraries=None, extra_link_options=None):
    link_command = [f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng"]
    link_command.extend(obj_files)
    link_command.extend(["-fPIC", "--shared", "-o", target_file])
    if extra_link_options:
        link_command.extend(extra_link_options)
    if link_libraries:
        link_command.extend(["-L", f"{ASCEND_PATH}/lib64"])
        link_command.extend(["-L", f"{ASCEND_PATH}/{machine}-linux/lib64"])
        link_command.extend([f"-l{link_library}" for link_library in link_libraries])
    run_compile_command(link_command, "LinkObj")
    return target_file


def is_cv_wrapper_source(source_file):
    base_name = os.path.basename(source_file)
    return base_name == CV_WRAPPER_SOURCE_NAME or base_name.endswith(
        f"_tiling_func_{CV_WRAPPER_SPLIT_KEY}.cpp"
    )


def get_shared_cv_wrapper_cache_dir(args: argparse.Namespace, temp_dir):
    output_file = getattr(args, "output_file", None)
    output_file_dir = (
        os.path.dirname(os.path.realpath(output_file)) if output_file else None
    )
    cache_root = (
        temp_dir
        or output_file_dir
        or os.getenv("RUN_DIR")
        or os.getenv("TORCHINDUCTOR_NPU_EXT_CACHE_DIR")
    )
    return os.path.join(os.path.realpath(cache_root), CV_WRAPPER_CACHE_DIR_NAME)


def read_file_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()


def get_shared_cv_wrapper_so_path(args: argparse.Namespace, temp_dir, source_file):
    digest = hashlib.sha256()
    digest.update(read_file_bytes(source_file))
    digest.update(str(ASCEND_PATH).encode("utf-8"))
    digest.update(str(machine).encode("utf-8"))
    digest.update(str(getattr(args, "soc_version", "")).encode("utf-8"))
    digest.update(str(getattr(args, "compile_options", "")).encode("utf-8"))
    digest.update(str(getattr(args, "stage", "")).encode("utf-8"))
    so_name = f"{CV_WRAPPER_SO_BASENAME}_{digest.hexdigest()[:16]}.so"
    return os.path.join(get_shared_cv_wrapper_cache_dir(args, temp_dir), so_name)


def get_shared_cv_wrapper_soname_options(so_path):
    return [f"-Wl,-soname,{os.path.basename(so_path)}"]


def get_shared_cv_wrapper_rpath_options(args: argparse.Namespace):
    return (
        [CV_WRAPPER_RPATH_OPTION]
        if getattr(args, "shared_cv_wrapper_so", None)
        else None
    )


def build_shared_cv_wrapper_so(
    args: argparse.Namespace, temp_dir, source_file, so_path
):
    tmp_so_path = f"{so_path}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        wrapper_obj = compile_host_obj_file(args, temp_dir, source_file)
        link_shared(
            tmp_so_path,
            [wrapper_obj],
            link_libraries=CV_HOST_LINK_LIBRARIES,
            extra_link_options=get_shared_cv_wrapper_soname_options(so_path),
        )
        os.replace(tmp_so_path, so_path)
    finally:
        if os.path.exists(tmp_so_path):
            os.remove(tmp_so_path)


def ensure_shared_cv_wrapper_so(args: argparse.Namespace, temp_dir, source_file):
    so_path = get_shared_cv_wrapper_so_path(args, temp_dir, source_file)
    if os.path.exists(so_path):
        return so_path
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    lock_path = f"{so_path}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if os.path.exists(so_path):
                return so_path
            build_shared_cv_wrapper_so(args, temp_dir, source_file, so_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return so_path


def append_shared_cv_wrapper_so(args: argparse.Namespace, obj_files):
    shared_cv_wrapper_so = getattr(args, "shared_cv_wrapper_so", None)
    if shared_cv_wrapper_so and is_cv_fusion_compile(args):
        return obj_files + [shared_cv_wrapper_so]
    return obj_files


def prepare_shared_cv_wrapper(args: argparse.Namespace, temp_dir, host_files):
    if not is_cv_fusion_compile(args):
        return host_files
    regular_host_files = []
    wrapper_sources = []
    for source_file in host_files:
        if is_cv_wrapper_source(source_file):
            wrapper_sources.append(source_file)
        else:
            regular_host_files.append(source_file)
    if wrapper_sources:
        args.shared_cv_wrapper_so = ensure_shared_cv_wrapper_so(
            args, temp_dir, wrapper_sources[0]
        )
    return regular_host_files


def link_pgo_executable(target_file, obj_files, mspti_link_flags):
    link_command = [f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng", *obj_files]
    link_command.extend(["-fPIC", "-o", target_file])
    link_command.extend(["-L", f"{ASCEND_PATH}/lib64"])
    link_command.extend(["-L", f"{ASCEND_PATH}/{machine}-linux/lib64"])
    link_command.extend([f"-l{link_library}" for link_library in HOST_LINK_LIBRARIES])
    link_command.extend(
        ["-lascendcl", "-lruntime", "-lunified_dlog", "-lascendalog", "-lc_sec", "-lm"]
    )
    link_command.extend(
        f"-Wl,-rpath,{option[2:]}"
        for option in mspti_link_flags
        if option.startswith("-L") and len(option) > 2
    )
    link_command.extend(mspti_link_flags)
    link_command.extend(["-lstdc++", "-ldl", "-lpthread"])
    run_compile_command(link_command, "LinkPgoExecutable")
    return target_file


def extract_aicore_binary(kernel_obj_path, output_file):
    objcopy = shutil.which("llvm-objcopy")
    if objcopy is None:
        objcopy = os.path.join(
            ASCEND_PATH, "tools", "bisheng_compiler", "bin", "llvm-objcopy"
        )
    if not os.path.isfile(objcopy):
        raise CompileError("llvm-objcopy is required for Inductor PGO device binary")
    run_compile_command(
        [objcopy, "--dump-section", f".aicore_binary={output_file}", kernel_obj_path],
        "ExtractPgoDeviceBinary",
    )
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        raise CompileError("extracted Inductor PGO device binary is empty")
    return output_file


def build_pgo_sidecars(args, temp_dir):
    mspti_dir, preload_files, link_flags = args.pgo_mspti_config
    args.pgo_mspti_dir = mspti_dir
    args.pgo_ld_preload = ":".join(preload_files)
    runner_obj = compile_host_obj_file(args, temp_dir, args.pgo_runner_file)
    runner_file = os.path.join(temp_dir, "pgo_runner")
    link_pgo_executable(runner_file, [runner_obj], link_flags)
    args.device_files = args.pgo_device_file
    device_obj = compile_device_obj(args, temp_dir)
    kernel_file = os.path.join(temp_dir, f"pgo_kernel.{PGO_KERNEL_FORMAT}")
    extract_aicore_binary(device_obj, kernel_file)
    return runner_file, kernel_file


def get_pgo_sidecar_paths(output_file, generation):
    generation_dir = f"{os.path.realpath(output_file)}.pgo.{generation}"
    output_name = os.path.basename(output_file)
    return {
        "generation_dir": generation_dir,
        "tiling_so": os.path.join(generation_dir, output_name),
        "runner": os.path.join(generation_dir, f"{output_name}.pgo_runner"),
        "kernel": os.path.join(
            generation_dir, f"{output_name}.pgo_kernel.{PGO_KERNEL_FORMAT}"
        ),
        "manifest": os.path.join(generation_dir, "manifest.json"),
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pgo_manifest(bundle):
    return {
        "bundle_schema_version": PGO_BUNDLE_SCHEMA_VERSION,
        "generation": bundle.generation,
        "result_protocol_version": PGO_RESULT_PROTOCOL_VERSION,
        "ld_preload": bundle.ld_preload,
        "artifacts": {
            "tiling_so": {
                "file": os.path.basename(bundle.output_file),
                "sha256": file_sha256(bundle.tiling_file),
            },
            "runner": {
                "file": os.path.basename(bundle.runner_file),
                "sha256": file_sha256(bundle.runner_file),
            },
            "kernel": {
                "file": os.path.basename(bundle.kernel_file),
                "sha256": file_sha256(bundle.kernel_file),
            },
        },
    }


def write_pgo_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())


def cleanup_stale_pgo_generations(output_file, current_generation_dir):
    output_file = os.path.realpath(output_file)
    output_dir = os.path.dirname(output_file)
    generation_prefix = os.path.basename(output_file) + ".pgo."
    previous_generations = []
    try:
        for entry in os.scandir(output_dir):
            if not entry.name.startswith(generation_prefix) or not entry.is_dir(
                follow_symlinks=False
            ):
                continue
            if os.path.realpath(entry.path) == os.path.realpath(current_generation_dir):
                continue
            previous_generations.append(
                (entry.stat(follow_symlinks=False).st_mtime_ns, entry.name, entry.path)
            )
    except OSError:
        return
    previous_generations.sort(reverse=True)
    for _, _, generation_dir in previous_generations[1:]:
        shutil.rmtree(generation_dir, ignore_errors=True)


def publish_pgo_bundle(bundle):
    output_file = os.path.realpath(bundle.output_file)
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    paths = get_pgo_sidecar_paths(output_file, bundle.generation)
    staging_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(output_file)}.pgo.", dir=output_dir
    )
    staged_tiling = os.path.join(
        output_dir, f".{os.path.basename(output_file)}.{bundle.generation}.tmp"
    )
    generation_published = False
    try:
        staged_generation_tiling = os.path.join(
            staging_dir, os.path.basename(paths["tiling_so"])
        )
        staged_runner = os.path.join(staging_dir, os.path.basename(paths["runner"]))
        staged_kernel = os.path.join(staging_dir, os.path.basename(paths["kernel"]))
        shutil.copy2(bundle.tiling_file, staged_generation_tiling)
        shutil.copy2(bundle.runner_file, staged_runner)
        shutil.copy2(bundle.kernel_file, staged_kernel)
        shutil.copy2(staged_generation_tiling, staged_tiling)
        manifest = build_pgo_manifest(
            PgoBundle(
                staged_generation_tiling,
                staged_runner,
                staged_kernel,
                output_file,
                bundle.generation,
                bundle.ld_preload,
            )
        )
        write_pgo_manifest(os.path.join(staging_dir, "manifest.json"), manifest)
        os.replace(staging_dir, paths["generation_dir"])
        generation_published = True
        os.replace(staged_tiling, output_file)
        cleanup_stale_pgo_generations(output_file, paths["generation_dir"])
        return paths
    except OSError as ex:
        if generation_published:
            shutil.rmtree(paths["generation_dir"], ignore_errors=True)
        raise CompileError(f"publish Inductor PGO bundle failed: {ex}") from ex
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if os.path.exists(staged_tiling):
            os.remove(staged_tiling)


def iter_compile_source_files(args: argparse.Namespace):
    for source_files in (
        getattr(args, "host_files", None),
        getattr(args, "device_files", None),
    ):
        for source_file in normalize_to_list(source_files):
            yield source_file


def contains_cv_tiling_data(source_file):
    if not source_file or not os.path.exists(source_file):
        return False
    with open(source_file, "r") as f:
        return "CVAutofuseTilingData" in f.read()


def is_cv_fusion_compile(args: argparse.Namespace):
    return any(
        contains_cv_tiling_data(source_file)
        for source_file in iter_compile_source_files(args)
    )


def get_host_abi_compile_options(args: argparse.Namespace):
    if is_cv_fusion_compile(args):
        return [
            option
            for option in args.compile_options.split()
            if "_GLIBCXX_USE_CXX11_ABI" not in option
        ], ["-D", "_GLIBCXX_USE_CXX11_ABI=0"]
    return args.compile_options.split(), (
        []
        if "-D_GLIBCXX_USE_CXX11_ABI=" in args.compile_options
        else ["-D", "_GLIBCXX_USE_CXX11_ABI=0"]
    )


def build_host_include_options(temp_dir):
    options = [
        "-I",
        f"{ASCEND_PATH}/include",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/pkg_inc",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/pkg_inc/base",
        "-I",
        f"{ASCEND_PATH}/pkg_inc",
        "-I",
        f"{ASCEND_PATH}/pkg_inc/base",
        "-I",
        f"{ASCEND_PATH}/include/base",
        "-I",
        f"{ASCEND_PATH}/include/experiment",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/include",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/include/aclnn",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api/tiling/platform",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/conv2d_v2",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/conv2d_v2",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
    ]
    if temp_dir is not None:
        options[0:0] = ["-I", f"{temp_dir}/host"]
    return options


def build_host_base_options(args: argparse.Namespace, temp_dir):
    soc_version = get_soc_type(args)
    options = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-D",
        "kernel_EXPORTS",
        *build_host_include_options(temp_dir),
        "-fPIC",
        f"--npu-arch={soc_version}",
        HOST_CPP_STANDARD,
        "-O2",
        "-fno-common",
        "-Wextra",
        "-Wfloat-equal",
        "-fvisibility=default",
    ]
    mspti_dir = getattr(args, "pgo_mspti_dir", None)
    if mspti_dir:
        options.extend(["-I", os.path.join(mspti_dir, "include")])
    pgo_generation = getattr(args, "pgo_generation", None)
    if pgo_generation:
        options.extend(["-D", f'AUTOFUSE_PGO_GENERATION="{pgo_generation}"'])
    return options


def build_host_output_options(source_file, obj_file):
    return [
        "-o",
        obj_file,
        "-c",
        "-x",
        "c++",
        source_file,
    ]


def generate_pch_source(host_dir):
    pch_src_path = os.path.join(host_dir, PCH_FILENAME)
    lines = ["// Auto-generated PCH source for autofuse tiling host compile"]
    lines.append("// Stable headers shared by generated Host tiling sources")
    for header in PCH_COMMON_HEADERS:
        lines.append(f"#include {header}")
    with open(pch_src_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return pch_src_path


def build_pch_command(args, pch_src_path, pch_out_path):
    compile_options, host_abi_option = get_host_abi_compile_options(args)
    pch_args = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-D",
        "kernel_EXPORTS",
        *build_host_include_options(None),
        "-fPIC",
        f"--npu-arch={get_soc_type(args)}",
        HOST_CPP_STANDARD,
        "-O2",
        "-fno-common",
        "-Wextra",
        "-Wfloat-equal",
        "-fvisibility=default",
        *compile_options,
        "-D",
        "LOG_CPP",
        *host_abi_option,
        "-x",
        "c++-header",
        pch_src_path,
        "-o",
        pch_out_path,
    ]
    return pch_args + get_compile_diagnostic_flags(pch_out_path)


def get_host_pch_key(args):
    compile_options, host_abi_option = get_host_abi_compile_options(args)
    key_data = "\0".join(
        [
            f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
            get_soc_type(args),
            HOST_CPP_STANDARD,
            *build_host_include_options(None),
            *compile_options,
            "-D",
            "kernel_EXPORTS",
            "-D",
            "LOG_CPP",
            *host_abi_option,
            *PCH_COMMON_HEADERS,
        ]
    )
    return hashlib.sha256(key_data.encode("utf-8")).hexdigest()


def get_host_pch_paths(args):
    pch_key = get_host_pch_key(args)
    root = os.path.join(PCH_CACHE_ROOT, pch_key)
    return (
        root,
        os.path.join(root, PCH_FILENAME),
        os.path.join(root, PCH_OUTPUT_NAME),
        os.path.join(root, ".lock"),
    )


def prepare_host_pch(args, paths):
    root, pch_source, pch_output, _ = paths
    os.makedirs(root, exist_ok=True)
    if os.path.isfile(pch_output) and os.path.getsize(pch_output) > 0:
        return paths
    generate_pch_source(root)
    with InductorCompileDuration(args, "CompileHostPCH"):
        run_compile_command(build_pch_command(args, pch_source, pch_output), "PCH")
    if not os.path.isfile(pch_output) or os.path.getsize(pch_output) == 0:
        raise CompileError("PCH compiler did not produce a valid output")
    return paths


def cleanup_host_pch_files(paths):
    _, pch_source, pch_output, _ = paths
    for path in (pch_source, pch_output):
        if os.path.exists(path):
            os.remove(path)


def prepare_host_pch_with_cleanup(args, paths):
    try:
        # 该打点包含缓存检查和 PCH 准备；缓存命中时不会产生 CompileHostPCH。
        with InductorCompileDuration(args, "PrepareHostPCH"):
            return prepare_host_pch(args, paths)
    except (CompileError, OSError, ValueError):
        try:
            cleanup_host_pch_files(paths)
        except OSError:
            pass
        return None


@contextmanager
def global_pch_build(args):
    """获取一份按编译上下文复用的只读 PCH。"""
    try:
        paths = get_host_pch_paths(args)
        root, _, _, lock_path = paths
        os.makedirs(root, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            # 记录等待其他 AutoFuseCompile 释放 PCH 生成锁的耗时。
            with InductorCompileDuration(args, "WaitHostPCHLock"):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                pch_entry = prepare_host_pch_with_cleanup(args, paths)
            finally:
                # 只在检查/生成 PCH 期间持有排他锁，不阻塞其他请求的 Host 编译。
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError):
        pch_entry = None
    # PCH 已经生成完成且不会在本上下文中删除，多个请求可以并行使用。
    yield pch_entry


def invalidate_host_pch(args, pch_path):
    """删除已被编译器拒绝的当前上下文 PCH 缓存。"""
    root, _, pch_output, lock_path = get_host_pch_paths(args)
    if pch_path != pch_output:
        return
    os.makedirs(root, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(pch_output):
                os.remove(pch_output)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def host_compile_batch(args):
    """在一批 Host 编译期间获取一份可复用的 PCH。"""
    # global_pch_build 在全局锁保护下检查或生成缓存 PCH，随后释放锁。
    host_files = getattr(args, "host_files", None)
    pch_context = (
        global_pch_build(args) if host_files is not None else nullcontext(None)
    )
    with pch_context as pch_entry:
        # pch_entry 保存 PCH 的相关路径，其中第 3 个元素是 .gch 文件路径。
        # 将该路径交给外层的 compile_host_objs()，让所有 Host 文件共享它。
        # 如果 PCH 生成失败，返回 None，Host 编译会自动回退到普通 C++ 编译。
        yield pch_entry[2] if pch_entry else None


def build_host_compile_cmd(
    args: argparse.Namespace, temp_dir, source_file, obj_file, pch_path=None
):
    compile_options, host_abi_option = get_host_abi_compile_options(args)
    cmd = [
        *build_host_base_options(args, temp_dir),
        *compile_options,
        "-D",
        "LOG_CPP",
        *host_abi_option,
    ]
    if pch_path:
        cmd += ["-include-pch", pch_path]
    cmd += build_host_output_options(source_file, obj_file)
    cmd += get_compile_diagnostic_flags(obj_file)
    return cmd


def get_host_obj_path(source_file, temp_dir):
    return os.path.join(temp_dir, "host", os.path.basename(source_file) + ".o")


def compile_host_obj_file(
    args: argparse.Namespace, temp_dir, source_file, pch_state=None
):
    obj_file = get_host_obj_path(source_file, temp_dir)
    if pch_state:
        with pch_state["lock"]:
            pch_path = pch_state["path"]
    else:
        pch_path = None
    host_compile_cmd = build_host_compile_cmd(
        args, temp_dir, source_file, obj_file, pch_path
    )
    try:
        run_compile_command(host_compile_cmd, "Host")
    except CompileError as ex:
        if pch_path:
            with pch_state["lock"]:
                pch_state["path"] = None
            try:
                invalidate_host_pch(args, pch_path)
            except OSError:
                pass
            fallback_cmd = build_host_compile_cmd(args, temp_dir, source_file, obj_file)
            try:
                run_compile_command(fallback_cmd, "Host")
            except CompileError as fallback_ex:
                raise CompileError(
                    f"Host compile failed for {source_file}: {fallback_ex}"
                ) from fallback_ex
        else:
            raise CompileError(f"Host compile failed for {source_file}: {ex}") from ex
    return obj_file


def get_host_compile_worker_count(cpp_file_count):
    return min(HOST_COMPILE_MAX_WORKERS, cpp_file_count, os.cpu_count() or 1)


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@inductor_compile_duration("CompileHostObj")
def compile_host_objs(args: argparse.Namespace, temp_dir, pch_path=None):
    host_files = prepare_shared_cv_wrapper(
        args, temp_dir, normalize_to_list(args.host_files)
    )
    if not host_files:
        return []
    pch_state = {"path": pch_path, "lock": Lock()}
    if len(host_files) == 1:
        return [compile_host_obj_file(args, temp_dir, host_files[0], pch_state)]

    obj_files = [None] * len(host_files)
    worker_count = get_host_compile_worker_count(len(host_files))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                compile_host_obj_file, args, temp_dir, source_file, pch_state
            ): index
            for index, source_file in enumerate(host_files)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            obj_files[index] = future.result()
    return obj_files


@inductor_compile_duration("CompileHostObj")
def compile_host_obj(args: argparse.Namespace, temp_dir):
    host_files = normalize_to_list(args.host_files)
    if len(host_files) != 1:
        raise CompileError("compile_host_obj expects exactly one host source")
    return compile_host_obj_file(args, temp_dir, host_files[0])


@inductor_compile_duration("CompileDeviceObj")
def compile_device_obj(args: argparse.Namespace, temp_dir):
    base_device_file = os.path.basename(args.device_files)
    soc_version = get_soc_type(args)
    device_compile_cmd = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-I",
        f"{temp_dir}/device",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/asc/include",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/batch_mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/batch_mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common/cmct",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct",
        "-fPIC",
        "-D",
        "HAVE_TILING",
        "-D",
        "AUTO_FUSE_DEVICE=1",
        f"--npu-arch={soc_version}",
        "-o",
        f"{temp_dir}/device/{base_device_file}.o",
        "-c",
        "-x",
        "asc",
        f"{temp_dir}/device/{base_device_file}",
        *get_compile_diagnostic_flags(f"{temp_dir}/device/{base_device_file}.o"),
    ]
    run_compile_command(device_compile_cmd, "Device")
    return f"{temp_dir}/device/{base_device_file}.o"


@inductor_compile_duration("LinkKernelSo")
def link_kernel_so(
    args: argparse.Namespace, tiling_obj_paths, temp_dir, kernel_obj_path
):
    if not kernel_obj_path:
        raise ValueError("kernel_obj_path is required for linking")

    target_file = os.path.join(temp_dir, os.path.basename(args.output_file))
    obj_files = [kernel_obj_path]
    tiling_obj_paths = normalize_to_list(tiling_obj_paths)
    if tiling_obj_paths:
        obj_files = tiling_obj_paths + obj_files
    obj_files = append_shared_cv_wrapper_so(args, obj_files)
    link_libraries = (
        CV_HOST_LINK_LIBRARIES
        if tiling_obj_paths and is_cv_fusion_compile(args)
        else (HOST_LINK_LIBRARIES if tiling_obj_paths else None)
    )
    return link_shared(
        target_file,
        obj_files,
        link_libraries=link_libraries,
        extra_link_options=get_shared_cv_wrapper_rpath_options(args),
    )


def clean_before_modify(temp_dir):
    src_directory = os.getcwd()
    keep_dirs = {"host", "device", CV_WRAPPER_CACHE_DIR_NAME}
    for entry in os.listdir(temp_dir):
        entry_path = os.path.join(temp_dir, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)
            logger.info("delete file: %s", entry_path)
        elif entry not in keep_dirs:
            shutil.rmtree(entry_path)
            logger.info("delete dir: %s", entry_path)
    os.chdir(src_directory)


def remove_tiling_data_from_launch(lines, start_index, launch_pattern):
    launch_lines = []
    index = start_index
    while index < len(lines):
        launch_lines.append(lines[index])
        if ");" in lines[index]:
            break
        index += 1

    launch_code = "".join(launch_lines)
    launch_code, replace_count = launch_pattern.subn("", launch_code, count=1)
    return launch_code.splitlines(keepends=True), index + 1, replace_count > 0


def remove_tiling_data_from_kernel_definition(
    lines, start_index, kernel_param_pattern, tiling_repr
):
    definition_lines = []
    index = start_index
    while index < len(lines):
        definition_lines.append(lines[index])
        if "{" in lines[index]:
            break
        index += 1

    definition_code = "".join(definition_lines)
    definition_code, replace_count = kernel_param_pattern.subn(
        "", definition_code, count=1
    )
    if replace_count == 0:
        return definition_lines, index + 1, False

    definition_lines = definition_code.splitlines(keepends=True)
    if tiling_repr is None:
        definition_lines.append("  const AutofuseTilingData t;\n")
    else:
        definition_lines.append(f"  constexpr AutofuseTilingData t = {tiling_repr};\n")
    return definition_lines, index + 1, True


def should_select_inductor_const_tiling_branch(
    args: argparse.Namespace, tiling_repr=None
):
    if tiling_repr is not None:
        return True
    return getattr(args, "stage", None) != "device" and is_cv_fusion_compile(args)


def prepare_static_shape_kernel(args: argparse.Namespace, temp_dir, tiling_repr=None):
    base_device_files = os.path.basename(args.device_files)
    kernel_file = os.path.join(temp_dir, "device", base_device_files)
    with open(kernel_file, "r") as f:
        lines = f.readlines()
    has_inductor_const_tiling = any(
        "INDUCTOR_CONST_TILING_DATA" in line for line in lines
    )
    if has_inductor_const_tiling:
        lines = expand_inductor_const_tiling_data(args, lines, tiling_repr)
    return kernel_file, lines


def write_kernel_lines(kernel_file, lines):
    with open(kernel_file, "w") as f:
        f.writelines(lines)


def build_static_shape_patterns():
    return (
        re.compile(
            r'^\s*(extern\s+"C"\s+)?__global__\s+__aicore__\s+void\s+(\w+)\s*\('
        ),
        re.compile(r",\s*AutofuseTilingData\s+t(?=\s*\)\s*\{)", re.DOTALL),
        re.compile(r"\b(\w+)\s*(?:<[^<>]*>)?\s*<<<"),
        re.compile(r",\s*\*\s*tiling_data(?=\s*\);)", re.DOTALL),
    )


def rewrite_static_shape_kernel_definitions(
    lines, kernel_start_pattern, kernel_param_pattern, tiling_repr=None
):
    result = []
    rewritten_kernels = set()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        kernel_match = kernel_start_pattern.match(line)
        if kernel_match:
            definition_lines, line_index, is_rewritten = (
                remove_tiling_data_from_kernel_definition(
                    lines, line_index, kernel_param_pattern, tiling_repr
                )
            )
            result.extend(definition_lines)
            if is_rewritten:
                rewritten_kernels.add(kernel_match.group(2))
            continue

        result.append(line)
        line_index += 1
    return result, rewritten_kernels


def rewrite_static_shape_kernel_launches(
    lines, rewritten_kernels, launch_start_pattern, launch_pattern
):
    result = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        launch_match = launch_start_pattern.search(line)
        if launch_match and launch_match.group(1) in rewritten_kernels:
            launch_lines, line_index, is_rewritten = remove_tiling_data_from_launch(
                lines, line_index, launch_pattern
            )
            if not is_rewritten:
                raise CompileError(
                    f"Failed to remove tiling data from launch of {launch_match.group(1)}"
                )
            result.extend(launch_lines)
            continue
        result.append(line)
        line_index += 1
    return result


def static_shape_kernel_proc(args: argparse.Namespace, temp_dir, tiling_repr=None):
    with InductorCompileDuration(args, "RewriteDeviceKernelForStaticTiling"):
        clean_before_modify(temp_dir)
        kernel_file, lines = prepare_static_shape_kernel(args, temp_dir, tiling_repr)
        if tiling_repr is None and getattr(args, "stage", None) == "device":
            write_kernel_lines(kernel_file, lines)
            return

        patterns = build_static_shape_patterns()
        definition_result, rewritten_kernels = rewrite_static_shape_kernel_definitions(
            lines, patterns[0], patterns[1], tiling_repr
        )
        result = rewrite_static_shape_kernel_launches(
            definition_result, rewritten_kernels, patterns[2], patterns[3]
        )
        write_kernel_lines(kernel_file, result)


def expand_inductor_const_tiling_data(
    args: argparse.Namespace, lines, tiling_repr=None
):
    result = []
    index = 0
    pending_const_tiling = False
    select_const_branch = should_select_inductor_const_tiling_branch(args, tiling_repr)
    while index < len(lines):
        line = lines[index]
        if is_inductor_dynamic_tiling_param(lines, index):
            if tiling_repr is None and result:
                result[-1] = result[-1].rstrip("\n") + ", AutofuseTilingData t"
            elif tiling_repr is not None:
                pending_const_tiling = True
            index += 3
            continue
        if is_inductor_const_tiling_branch(lines, index):
            result.append(lines[index + 1] if select_const_branch else lines[index + 3])
            index += 5
            continue
        result.append(line)
        if pending_const_tiling and line.strip() == ") {":
            result.append(f"  constexpr AutofuseTilingData t = {tiling_repr};\n")
            pending_const_tiling = False
        index += 1
    return result


def is_inductor_dynamic_tiling_param(lines, index):
    if index + 2 >= len(lines):
        return False
    return (
        lines[index].strip() == "#ifndef INDUCTOR_CONST_TILING_DATA"
        and lines[index + 1].strip() == ", AutofuseTilingData t"
        and lines[index + 2].strip() == "#endif"
    )


def is_inductor_const_tiling_branch(lines, index):
    if index + 4 >= len(lines):
        return False
    return (
        lines[index].strip() == "#ifdef INDUCTOR_CONST_TILING_DATA"
        and lines[index + 2].strip() == "#else"
        and lines[index + 4].strip() == "#endif"
    )


def has_inductor_const_tiling_data(args: argparse.Namespace, temp_dir):
    base_device_files = os.path.basename(args.device_files)
    kernel_file = os.path.join(temp_dir, "device", base_device_files)
    with open(kernel_file, "r") as f:
        return "INDUCTOR_CONST_TILING_DATA" in f.read()


def init_torch_npu_for_const_tiling():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.synchronize()


@inductor_compile_duration("PrepareStaticShapeRecompile")
def try_static_shape_compile(args: argparse.Namespace, temp_dir, so_path):
    if args.force_unknown:
        return False
    lib = ctypes.CDLL(so_path)
    lib.AutofuseIsStaticShape.argtypes = []
    lib.AutofuseIsStaticShape.restype = ctypes.c_bool
    if not bool(lib.AutofuseIsStaticShape()):
        return False
    logger.info("static shape detected, recompile kernel with const tiling data")
    static_shape_kernel_proc(args, temp_dir)
    with InductorCompileDuration(args, "InitTorchNpu"):
        init_torch_npu_for_const_tiling()
    with InductorCompileDuration(args, "GenerateConstTilingData"):
        lib.GenConstTilingData.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.GenConstTilingData.restype = ctypes.c_char_p
        config_file = ctypes.c_char_p(args.config_file.encode("utf-8"))
        aiv_num = int(get_soc_spec("vector_core_cnt"))
        ub_size = int(get_soc_spec("ub_size"))
        result = lib.GenConstTilingData(
            config_file, ctypes.c_int(aiv_num), ctypes.c_int(ub_size)
        )
    if not result:
        raise CompileError("GenConstTilingData returned empty const tiling data")
    const_tiling_data = result.decode("utf-8")
    tiling_data = os.path.join(temp_dir, "device", "autofuse_tiling_data.h")
    tiling_data_bak = os.path.join(temp_dir, "device", "autofuse_tiling_data_bak.h")
    shutil.copy(tiling_data, tiling_data_bak)
    with open(tiling_data, "w") as file:
        file.write(const_tiling_data)
    return True


@inductor_compile_duration("LinkTilingSo")
def link_tiling_so(args, tiling_obj_paths, temp_dir):
    """Link an existing set of tiling objects into the tiling shared library."""
    if not tiling_obj_paths:
        raise ValueError("tiling object files are required for linking")

    tiling_obj_paths = append_shared_cv_wrapper_so(
        args, normalize_to_list(tiling_obj_paths)
    )
    so_file = os.path.join(temp_dir, os.path.basename(args.output_file))
    link_libraries = (
        CV_HOST_LINK_LIBRARIES if is_cv_fusion_compile(args) else HOST_LINK_LIBRARIES
    )
    if getattr(args, "pgo_runner_file", None) is not None:
        link_libraries = link_libraries + ["ascendcl", "runtime"]
    return link_shared(
        so_file,
        tiling_obj_paths,
        link_libraries=link_libraries,
        extra_link_options=get_shared_cv_wrapper_rpath_options(args),
    )


@inductor_compile_duration("BuildKernelTarget")
def build_kernel_target(args, tiling_obj_paths, temp_dir):
    """Compile device objects and build the final kernel shared library."""
    if args.stage == "device" and (
        args.tiling_repr is not None or has_inductor_const_tiling_data(args, temp_dir)
    ):
        if args.tiling_repr is not None:
            logger.info("process static shape kernel with tiling_repr")
        static_shape_kernel_proc(args, temp_dir, args.tiling_repr)

    kernel_obj_path = compile_device_obj(args, temp_dir)
    so_file = link_kernel_so(args, tiling_obj_paths, temp_dir, kernel_obj_path)
    if args.stage == "device" or not try_static_shape_compile(args, temp_dir, so_file):
        return so_file

    # 静态 shape 重编译后，重新生成 device.o 并链接最终产物。
    kernel_obj_path = compile_device_obj(args, temp_dir)
    return link_kernel_so(args, tiling_obj_paths, temp_dir, kernel_obj_path)


@inductor_compile_duration("BuildKernelTargetFromObjects")
def build_kernel_target_from_objects(args, tiling_obj_paths, kernel_obj_path, temp_dir):
    """Build the final kernel from precompiled host and device objects."""
    if not tiling_obj_paths:
        raise ValueError("host object files are required")
    if not kernel_obj_path:
        raise ValueError("device object path is required")

    so_file = link_kernel_so(args, tiling_obj_paths, temp_dir, kernel_obj_path)
    if not try_static_shape_compile(args, temp_dir, so_file):
        return so_file

    # 静态 shape 重编译会修改 device 源码，需生成新的 device.o。
    kernel_obj_path = compile_device_obj(args, temp_dir)
    return link_kernel_so(args, tiling_obj_paths, temp_dir, kernel_obj_path)


@inductor_compile_duration("CopyOutput", args_index=1)
def copy_so_to_output(so_file, args, src_directory):
    dst_file = os.path.realpath(args.output_file)
    dst_dir_path = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    shutil.copy(so_file, dst_file)
    shared_cv_wrapper_so = getattr(args, "shared_cv_wrapper_so", None)
    if shared_cv_wrapper_so:
        wrapper_dst_dir = os.path.join(dst_dir_path, CV_WRAPPER_CACHE_DIR_NAME)
        os.makedirs(wrapper_dst_dir, exist_ok=True)
        wrapper_dst_file = os.path.join(
            wrapper_dst_dir, os.path.basename(shared_cv_wrapper_so)
        )
        if os.path.realpath(shared_cv_wrapper_so) != os.path.realpath(wrapper_dst_file):
            shutil.copy(shared_cv_wrapper_so, wrapper_dst_file)
    logger.info("copy file %s to %s", so_file, dst_file)
    os.chdir(src_directory)


def build_host_output(args, pch_path=None):
    should_build_sidecars = (
        getattr(args, "pgo_runner_file", None) is not None
        and getattr(args, "pgo_mspti_config", None) is not None
    )
    if should_build_sidecars:
        args.pgo_generation = uuid.uuid4().hex
    if pch_path is None:
        tiling_obj_paths = compile_host_objs(args, args.temp_dir)
    else:
        tiling_obj_paths = compile_host_objs(args, args.temp_dir, pch_path)
    so_file = link_tiling_so(args, tiling_obj_paths, args.temp_dir)
    if not should_build_sidecars:
        return so_file
    try:
        runner_file, kernel_file = build_pgo_sidecars(args, args.temp_dir)
        publish_pgo_bundle(
            PgoBundle(
                so_file,
                runner_file,
                kernel_file,
                args.output_file,
                args.pgo_generation,
                getattr(args, "pgo_ld_preload", ""),
            )
        )
        return None
    except CompileError as ex:
        _log_warning("[PGO] Inductor PGO sidecar build failed, skip PGO: %s", ex)
        return so_file


def parse_object_paths(value):
    if not value:
        return []
    return [path for path in value.split(";") if path]


def validate_artifact_paths(paths, kind):
    if not paths:
        raise CompileError(f"{kind} artifact paths are empty")
    missing = [
        path for path in paths if not isinstance(path, str) or not os.path.isfile(path)
    ]
    if missing:
        raise CompileError(f"{kind} artifact files are missing: {missing}")


def main(args):
    logger.info("compile args: %s", args)
    src_directory = os.getcwd()
    os.chdir(args.temp_dir)
    logger.info("change work dir: %s", os.getcwd())
    # 原子编译 stage 返回结构化 artifact，避免进程间共享 Python 状态。
    try:
        if args.stage == "host":
            with host_compile_batch(args) as pch_path:
                so_file = build_host_output(args, pch_path)
        elif args.stage == "host_obj":
            with host_compile_batch(args) as pch_path:
                tiling_obj_paths = compile_host_objs(args, args.temp_dir, pch_path)
            return {
                "version": 1,
                "stage": "host_obj",
                "tiling_obj_paths": tiling_obj_paths,
                "tiling_source_paths": normalize_to_list(args.host_files),
                "shared_cv_wrapper_so": getattr(args, "shared_cv_wrapper_so", None),
            }
        elif args.stage == "device_obj":
            kernel_obj_path = compile_device_obj(args, args.temp_dir)
            return {
                "version": 1,
                "stage": "device_obj",
                "kernel_obj_path": kernel_obj_path,
                "kernel_source_path": args.device_files,
            }
        elif args.stage == "link":
            tiling_obj_paths = parse_object_paths(args.tiling_obj_paths)
            kernel_obj_path = args.kernel_obj_path
            validate_artifact_paths(tiling_obj_paths, "host object")
            validate_artifact_paths(
                [kernel_obj_path, args.kernel_source_path], "device artifact"
            )
            args.host_files = parse_object_paths(args.tiling_source_paths)
            args.device_files = args.kernel_source_path
            args.shared_cv_wrapper_so = args.shared_cv_wrapper_so or None
            so_file = build_kernel_target_from_objects(
                args, tiling_obj_paths, kernel_obj_path, args.temp_dir
            )
        elif args.stage == "device":
            so_file = build_kernel_target(args, None, args.temp_dir)
        else:  # all
            with host_compile_batch(args) as pch_path:
                if pch_path is None:
                    tiling_obj_paths = compile_host_objs(args, args.temp_dir)
                else:
                    tiling_obj_paths = compile_host_objs(args, args.temp_dir, pch_path)
            so_file = build_kernel_target(args, tiling_obj_paths, args.temp_dir)
        if so_file is not None:
            copy_so_to_output(so_file, args, src_directory)
    finally:
        os.chdir(src_directory)


def main_with_except(argv: List[str]):
    """Main process with except exceptions."""
    try:
        logger.info("Enter main func")
        return main(argv)
    except argparse.ArgumentError as ex:
        logger.error("error: check arguments error, %s", ex)
        return False


if __name__ == "__main__":
    if not main_with_except(sys.argv[1:]):
        sys.exit(1)
