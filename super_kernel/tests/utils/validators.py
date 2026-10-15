#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------


"""Validation helpers for system tests."""

import os
import sys
from pathlib import Path
from typing import Iterable, Tuple
import difflib


def _resolve_kernel_paths(kernel_root: Path, kernel_name: str) -> Tuple[Path, Path]:
    kernel_root = Path(kernel_root)
    generated_dir = kernel_root / "kernel_meta"
    if not generated_dir.is_dir():
        raise AssertionError(f"Expected kernel_meta directory missing: {generated_dir}")

    pid = os.getpid()
    generated_file = generated_dir / f"{kernel_name}_{pid}_kernel.cpp"
    log_file = generated_dir / f"{kernel_name}_{pid}.log"
    return generated_file, log_file


def _strip_backend_dfx_section(lines):
    """Drop the mix core DFX section appended by the device compile backend.

    The backend appends metadata sections to the generated kernel source. They
    are produced outside of this repository and are not part of what the
    SuperKernel code generator emits, so they are ignored when comparing the
    generated source with the golden source.
    """
    for index, line in enumerate(lines):
        if not line.startswith("#if TILING_KEY_VAR"):
            continue
        rest = [item.strip() for item in lines[index:] if item.strip()]
        if all(
            item.startswith("#if")
            or item.startswith("#endif")
            or item.startswith("static const struct FunLevel")
            for item in rest
        ):
            return lines[:index]
    return lines


def _drop_trailing_blank_lines(lines):
    """Ignore trailing blank lines.

    Golden sources cannot keep them because the pre-commit hooks strip them,
    while the generated kernel source ends with a blank line.
    """
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    return lines[:end]


def compare_files(golden_path, codegen_path, encoding="utf-8"):
    """
    比较两个文件的内容并显示差异

    参数:
        golden_path: 预期结果文件(golden file)的路径
        codegen_path: 代码生成文件(codegen file)的路径
        encoding: 文件编码格式，默认为utf-8

    返回:
        bool: 两个文件是否相同
    """
    try:
        # 读取文件内容
        with open(golden_path, "r", encoding=encoding) as f:
            golden_lines = f.readlines()

        with open(codegen_path, "r", encoding=encoding) as f:
            codegen_lines = f.readlines()

        golden_lines = _drop_trailing_blank_lines(
            _strip_backend_dfx_section(golden_lines)
        )
        codegen_lines = _drop_trailing_blank_lines(codegen_lines)

    except FileNotFoundError as e:
        print(f"Error: file not found - {e.filename}", file=sys.stderr)
        return False
    except UnicodeDecodeError as e:
        print(f"Error: file encoding error - {e}", file=sys.stderr)
        return False

    # 比较文件内容
    differ = difflib.Differ()
    diff = list(differ.compare(golden_lines, codegen_lines))

    # 检查是否有差异
    has_diff = any(line.startswith(("+", "-", "?")) for line in diff)

    if has_diff:
        print(f"File contents differ: {golden_path} and {codegen_path}")
        print("=" * 80)

        # 打印差异，使用颜色区分（如果终端支持）
        for line in diff:
            if line.startswith("+"):
                # 新增内容（绿色）
                print(f"\033[92m{line}\033[0m", end="")
            elif line.startswith("-"):
                # 删除内容（红色）
                print(f"\033[91m{line}\033[0m", end="")
            elif line.startswith("?"):
                # 差异标记（黄色）
                print(f"\033[93m{line}\033[0m", end="")
            else:
                # 相同内容（默认颜色）
                print(line, end="")

        print("=" * 80)
    else:
        print(f"File contents are identical: {golden_path} and {codegen_path}")

    return not has_diff


def validate_codegen_output(
    kernel_root: Path, kernel_name: str, expected_source: Path
) -> None:
    """Validate generated code matches expected source code.

    Only the source produced by the SuperKernel code generator is compared. The
    device compile backend appends extra metadata sections (mix core DFX
    sections) to this file, which are produced outside of this repository and
    are not part of the golden files.
    """
    generated_file, _ = _resolve_kernel_paths(kernel_root, kernel_name)

    if not generated_file.is_file():
        generated_dir = generated_file.parent
        available = sorted(p.name for p in generated_dir.iterdir())
        raise AssertionError(
            "Generated kernel file missing: "
            f"expected {generated_file.name}; available={available}"
        )

    expected_source = Path(expected_source)
    if not expected_source.is_file():
        raise AssertionError(f"Expected source template missing: {expected_source}")

    assert compare_files(expected_source, generated_file) is True


def validate_compile_options(
    compile_info,
    expected_options: Iterable[str],
) -> None:
    """Ensure the compile options of the generated kernel are all present."""
    options = [str(option) for option in compile_info.get("compile_option", [])]

    missing = [
        option
        for option in expected_options
        if not any(option in actual for actual in options)
    ]
    if missing:
        raise AssertionError(
            "Missing expected compile options: "
            + ", ".join(missing)
            + f"\nreceived: {options}"
        )
