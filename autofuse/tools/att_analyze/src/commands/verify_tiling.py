#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import os
import re
import json
import subprocess
import ctypes
import tempfile
import shutil
import ast
from ctypes import byref, c_uint32, c_uint64, c_size_t, c_void_p, c_int64
from typing import Dict, Tuple
from core.file_utils import ensure_output_dir


# ──── 场景检测 ────────────────────────────────────────────────


def detect_scene(source_dir: str) -> str:
    """返回 'tf' 或 'inductor'"""
    files = os.listdir(source_dir)
    if any(f == "output_code.py" for f in files):
        return "inductor"
    if any("tiling_func" in f and f.endswith(".cpp") for f in files):
        return "tf"
    raise ValueError(
        f"Unable to detect scene automatically: neither *tiling_func*.cpp nor output_code.py was found in {source_dir}"
    )


# ──── 输入参数加载 ────────────────────────────────────────────


def load_input_params(args) -> Dict:
    """加载输入参数，优先级：--input-json > --preset"""
    if getattr(args, "input_json", None):
        with open(args.input_json) as f:
            params = json.load(f)
    else:
        preset_dir = os.path.join(os.path.dirname(__file__), "presets")
        preset_file = os.path.join(preset_dir, f"preset_{args.preset}.json")
        with open(preset_file) as f:
            params = json.load(f)
    override = getattr(args, "aiv_num", None)
    if isinstance(override, int):
        params["aiv_num"] = override
    return params


def print_input_config(input_params: Dict, source: str) -> None:
    """Print the effective tiling inputs so users can check hardware assumptions."""
    print(
        "[verify-tiling] input-config "
        f"source={source} "
        f"aiv_num={input_params.get('aiv_num', 48)} "
        f"ub_size={input_params.get('ub_size', 196608)} "
        f"dynamic_dims={input_params.get('dynamic_dims', [])}"
    )
    print(
        "[verify-tiling] aiv_num is the tiling configuration value; check it against the device core count;"
        "use --aiv-num or --input-json to override it if they differ"
    )


# ──── 编译配置加载 ────────────────────────────────────────────

_DEFAULT_FLAGS = (
    "-O0 -g -fno-common -Werror -Wextra -Wfloat-equal -fvisibility=default -DLOG_CPP"
)


def validate_input_params(input_params: Dict) -> None:
    """Validate values passed to the native tiling entrypoint."""
    dynamic_dims = input_params.get("dynamic_dims", [])
    if not isinstance(dynamic_dims, list) or len(dynamic_dims) > 32:
        raise ValueError("dynamic_dims must be a list with at most 32 dimensions")
    if any(
        not isinstance(dim, int) or not 0 < dim <= 0x7FFFFFFF for dim in dynamic_dims
    ):
        raise ValueError("dynamic_dims values must be in [1, 2147483647]")
    aiv_num = input_params.get("aiv_num", 48)
    ub_size = input_params.get("ub_size", 196608)
    if not isinstance(aiv_num, int) or not 0 < aiv_num <= 0xFFFF:
        raise ValueError("aiv_num must be in [1, 65535]")
    if not isinstance(ub_size, int) or not 0 < ub_size <= 0x7FFFFFFF:
        raise ValueError("ub_size must be in [1, 2147483647]")
    abi = input_params.get("abi")
    if not isinstance(abi, dict) or abi.get("kind") not in (
        "tf_static",
        "tf_dynamic",
        "inductor",
    ):
        raise ValueError("abi.kind must be tf_static, tf_dynamic, or inductor")
    if abi.get("block_dim_width") not in (32, 64):
        raise ValueError("abi.block_dim_width must be 32 or 64")
    if not isinstance(abi.get("shape_dims"), int) or abi["shape_dims"] < 0:
        raise ValueError("abi.shape_dims must be a non-negative integer")
    if abi["kind"] == "tf_static" and abi["shape_dims"] != 0:
        raise ValueError("tf_static requires shape_dims=0")
    if abi["kind"] in ("tf_dynamic", "inductor") and abi["shape_dims"] == 0:
        raise ValueError(f"{abi['kind']} requires shape_dims>0")


def load_compile_config(args) -> Dict:
    """加载编译配置，返回 {'flags': str, 'extra_includes': [], 'extra_links': []}"""
    config_path = getattr(args, "compile_config", None)
    if config_path is None:
        config_path = os.path.expanduser("~/.att_analyze/compile.toml")
    if config_path and os.path.exists(config_path):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        if tomllib:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            return data.get("compile", {})
    return {"flags": _DEFAULT_FLAGS, "extra_includes": [], "extra_links": []}


# ──── inductor 场景：提取 artifacts ──────────────────────────


def extract_inductor_artifacts(output_code_py: str) -> Tuple[str, str]:
    """从 output_code.py 提取 tiling_def 和 host_impl 字符串"""
    with open(output_code_py) as f:
        src = f.read()
    artifacts_match = re.search(r"(\w+_artifacts)\s*=\s*\{", src)
    if not artifacts_match:
        raise ValueError("*_artifacts dictionary not found in output_code.py")
    var_name = artifacts_match.group(1)
    tree = ast.parse(src, filename=output_code_py)
    artifacts = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == var_name
            for target in node.targets
        ):
            try:
                artifacts = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise ValueError(
                    f"{var_name} must be a parseable literal dictionary"
                ) from exc
            break
    if not isinstance(artifacts, dict):
        raise ValueError(f"{var_name} is not a dictionary")
    return artifacts["tiling_def"], artifacts["host_impl"]


# ──── 准备编译目录 ────────────────────────────────────────────

_CMAKELISTS_TEMPLATE = """\
cmake_minimum_required(VERSION 3.10)
project({kernel_name})
set(CMAKE_CXX_STANDARD 17)
set(ASCEND_PATH $ENV{{ASCEND_HOME_PATH}})
file(GLOB ALL_CPP_SRCS *tiling_func*.cpp *infershape*.cpp)
add_library({kernel_name} SHARED ${{ALL_CPP_SRCS}})
target_compile_options({kernel_name} PRIVATE {flags})
target_include_directories({kernel_name} PRIVATE
    ${{ASCEND_PATH}}/include
    ${{ASCEND_PATH}}/aarch64-linux/include
    ${{ASCEND_PATH}}/x86_64-linux/include
    {extra_includes}
)
target_link_libraries({kernel_name}
    c_sec ascendalog platform error_manager tiling_api graph_base register
    {extra_links}
)
"""


def prepare_build_dir(
    source_dir: str, scene: str, tmp_dir: str, compile_cfg: Dict
) -> Tuple[str, str]:
    """返回 (build_dir, kernel_name)"""
    flags = compile_cfg.get("flags", _DEFAULT_FLAGS)
    extra_includes = "\n    ".join(compile_cfg.get("extra_includes", []))
    extra_links = "\n    ".join(compile_cfg.get("extra_links", []))

    if scene == "inductor":
        output_code = os.path.join(source_dir, "output_code.py")
        tiling_def, host_impl = extract_inductor_artifacts(output_code)
        build_dir = tmp_dir
        with open(os.path.join(build_dir, "autofuse_tiling_data.h"), "w") as f:
            f.write(tiling_def)
        with open(os.path.join(build_dir, "tiling_func.cpp"), "w") as f:
            f.write(host_impl)
    else:
        build_dir = tmp_dir
        for filename in os.listdir(source_dir):
            if (
                filename.endswith(".cpp")
                and ("tiling_func" in filename or "infershape" in filename)
            ) or filename.endswith((".h", ".hpp", ".inc")):
                shutil.copy2(
                    os.path.join(source_dir, filename),
                    os.path.join(build_dir, filename),
                )

    cpp_files = [
        f for f in os.listdir(build_dir) if "tiling_func" in f and f.endswith(".cpp")
    ]
    kernel_name = (
        re.sub(r"_tiling_func.*", "", cpp_files[0]) if cpp_files else "KernelUnknown"
    )

    cmake_path = os.path.join(build_dir, "CMakeLists.txt")
    if not os.path.exists(cmake_path):
        with open(cmake_path, "w") as f:
            f.write(
                _CMAKELISTS_TEMPLATE.format(
                    kernel_name=kernel_name,
                    flags=flags,
                    extra_includes=extra_includes,
                    extra_links=extra_links,
                )
            )
    return build_dir, kernel_name


# ──── 编译 ───────────────────────────────────────────────────


def compile_tiling(build_dir: str, kernel_name: str) -> Tuple[bool, str]:
    """执行 cmake + make，返回 (success, so_path_or_error_msg)"""
    cmake_build = os.path.join(build_dir, "build")
    os.makedirs(cmake_build, exist_ok=True)

    cmake_cmd = [
        "cmake",
        "-S",
        build_dir,
        "-B",
        cmake_build,
        "-DCMAKE_C_COMPILER=gcc",
        "-DCMAKE_CXX_COMPILER=g++",
    ]
    r = subprocess.run(cmake_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"cmake 失败:\n{r.stderr}"

    make_cmd = ["make", "-C", cmake_build, "-j8"]
    r = subprocess.run(make_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"make 失败:\n{r.stderr}"

    so_path = os.path.join(cmake_build, f"lib{kernel_name}.so")
    if not os.path.exists(so_path):
        return False, f"编译成功但未找到 {so_path}"
    return True, so_path


# ──── ctypes 执行 ────────────────────────────────────────────


def execute_tiling(so_path: str, input_params: Dict, scene: str) -> Dict:
    """按场景调用 AutofuseTiling，返回 {block_dim, workspace_size}."""
    validate_input_params(input_params)
    lib = ctypes.CDLL(so_path)
    lib.GetTilingDataSize.restype = c_size_t
    tiling_size = lib.GetTilingDataSize()
    if not isinstance(tiling_size, int) or not 0 < tiling_size <= 1 << 30:
        raise ValueError("GetTilingDataSize returned an invalid size")

    aiv_num = input_params.get("aiv_num", 48)
    ub_size = input_params.get("ub_size", 196608)
    dynamic_dims = [c_uint32(dim) for dim in input_params.get("dynamic_dims", [])]

    tiling_buf = ctypes.create_string_buffer(tiling_size)
    ws = c_uint64(0)
    abi = input_params["abi"]
    bd_type = c_uint64 if abi["block_dim_width"] == 64 else c_uint32
    bd = bd_type(0)

    dims = input_params.get("dynamic_dims", [])
    if len(dims) != abi["shape_dims"]:
        raise ValueError("dynamic_dims count does not match abi.shape_dims")
    common_args = [c_void_p, ctypes.POINTER(c_uint64), ctypes.POINTER(bd_type)]
    if scene == "inductor" and abi["kind"] == "inductor":
        lib.AutofuseTiling.argtypes = (
            [c_uint32] * len(dynamic_dims) + common_args + [c_void_p]
        )
        call_args = dynamic_dims + [tiling_buf, byref(ws), byref(bd), None]
    elif scene == "tf" and abi["kind"] in ("tf_static", "tf_dynamic"):
        lib.AutofuseTiling.argtypes = (
            [c_uint32] * len(dynamic_dims) + common_args + [c_uint32, c_uint32]
        )
        call_args = dynamic_dims + [
            tiling_buf,
            byref(ws),
            byref(bd),
            c_uint32(aiv_num),
            c_uint32(ub_size),
        ]
    else:
        raise ValueError("scene does not match abi.kind")
    lib.AutofuseTiling.restype = c_int64
    status = lib.AutofuseTiling(*call_args)
    if status != 0:
        raise RuntimeError(f"AutofuseTiling failed with status {status}")

    block_dim_val = bd.value
    result = {"block_dim": block_dim_val, "workspace_size": ws.value}
    if (
        not 0 <= result["block_dim"] <= 0xFFFFFFFF
        or not 0 <= result["workspace_size"] <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("AutofuseTiling returned an invalid value")
    return result


# ──── 主入口 ─────────────────────────────────────────────────


def run(args):
    source_dir = os.path.abspath(args.source_dir)
    if not os.path.isdir(source_dir):
        print(f"✗ Source directory does not exist: {source_dir}")
        return 2
    try:
        scene = args.scene or detect_scene(source_dir)
    except ValueError as exc:
        print(f"✗ {exc}")
        return 2
    input_params = load_input_params(args)
    compile_cfg = load_compile_config(args)
    if args.log and not os.path.exists(args.log):
        print(f"✗ Log does not exist: {args.log}")
        return 2

    print(f"[verify-tiling] scene={scene}")
    source = (
        f"input-json:{args.input_json}" if args.input_json else f"preset_{args.preset}"
    )
    if isinstance(getattr(args, "aiv_num", None), int):
        source += "+--aiv-num"
    print_input_config(input_params, source)
    holder = (
        tempfile.TemporaryDirectory(prefix="att-verify-")
        if not args.keep_build
        else None
    )
    tmp_dir = holder.name if holder else tempfile.mkdtemp(prefix="att-verify-")
    record = {
        "scene": scene,
        "source_dir": source_dir,
        "case": args.case,
        "log": args.log,
    }
    try:
        build_dir, kernel_name = prepare_build_dir(
            source_dir, scene, tmp_dir, compile_cfg
        )
        print("\n=== Compile Check ===")
        ok, result = compile_tiling(build_dir, kernel_name)
        if not ok:
            print(f"✗ Compilation failed:\n{result}")
            record.update(status="COMPILE_FAILED", error=result)
            return_code = 1
        else:
            print(f"✓ cmake OK\n✓ make OK  → {result}")
            record["so_path"] = result
            print("\n=== Tiling Execution ===")
            try:
                fields = execute_tiling(result, input_params, scene)
                for k, v in fields.items():
                    print(f"  {k:<20} = {v}")
                record.update(status="SUCCEEDED", fields=fields)
                return_code = 0
            except Exception as exc:
                print(f"✗ Runtime Error: {exc}")
                record.update(status="RUNTIME_FAILED", error=str(exc))
                return_code = 1
        ensure_output_dir(args.output)
        with open(
            os.path.join(args.output, "result.json"), "w", encoding="utf-8"
        ) as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        return return_code
    finally:
        if holder:
            holder.cleanup()
