#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AFIR ODS 文档门禁。

稳定性分层与分档判据见 docs/design/2026-09-09-afir-public-dsl-architecture-design.md 2.5。
``stable`` 与 ``experimental`` 档的 Op 都对外公开，其 ODS ``summary`` / ``description``
会直接进入 ``-gen-dialect-doc`` 产出的对外文档，因此强制要求非空；缺失即拦截 PR。
``internal`` 档不对外，只统计不拦截。

门禁覆盖 ``experimental`` 而不只是 ``stable``，是因为分档判据里「文档必填」这一条正是
在 ``experimental`` 这一档生效的：一个 Op 升到 ``experimental`` 的那一刻就已经对外可见，
等到升 ``stable`` 再补文档，中间那段时间公开的是一个没有说明的接口。

档位由 ``AFIROpBase.td`` 中 ``AFIR_Op`` 的 ``afirStability`` 字段声明，默认 ``internal``。
本脚本通过 ``llvm-tblgen --dump-json`` 读取解析后的记录，而不是正则匹配 .td 文本，
因此 class 继承、``defvar`` 与多级 base class 都能被正确解析。

用法::

    python3 check_afir_ods_docs.py                 # 门禁：stable/experimental 档缺文档则失败
    python3 check_afir_ods_docs.py --summary       # 追加各档位文档覆盖率统计
    python3 check_afir_ods_docs.py --require-tblgen  # 找不到 tblgen 也判失败（CI 用）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AFIR_IR_DIR = ROOT / "dialect" / "AFIR" / "IR"
AFIR_OPS_TD = AFIR_IR_DIR / "AFIROps.td"

STABILITY_FIELD = "afirStability"
TIER_STABLE = "stable"
TIER_EXPERIMENTAL = "experimental"
TIER_INTERNAL = "internal"
KNOWN_TIERS = (TIER_STABLE, TIER_EXPERIMENTAL, TIER_INTERNAL)

# 对外公开的两档强制要求的 ODS 字段。
REQUIRED_DOC_FIELDS = ("summary", "description")
# 强制要求文档的档位。
PUBLIC_TIERS = (TIER_STABLE, TIER_EXPERIMENTAL)


def create_stream_logger(name: str, stream) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


OUTPUT_LOGGER = create_stream_logger(f"{__name__}.stdout", sys.stdout)
ERROR_LOGGER = create_stream_logger(f"{__name__}.stderr", sys.stderr)


@dataclass(frozen=True)
class OpRecord:
    """一条 ODS Op 记录中门禁关心的部分。"""

    record: str
    mnemonic: str
    stability: str
    missing: tuple[str, ...]

    @property
    def documented(self) -> bool:
        return not self.missing


def llvm_root_candidates() -> list[Path]:
    """按 cmake/AutofuseMlirDeps.cmake 的同一顺序给出 LLVM 根目录候选。"""
    candidates: list[Path] = []
    build_dir = os.environ.get("LLVM_BUILD_DIR")
    if build_dir:
        candidates.append(Path(build_dir))
    for var in ("MLIR_DIR", "LLVM_DIR"):
        cmake_dir = os.environ.get(var)
        if cmake_dir:
            # <root>/lib/cmake/<mlir|llvm> -> <root>
            candidates.append(Path(cmake_dir).parents[2])
    candidates.append(ROOT / ".deps" / "llvm")
    candidates.append(Path("/opt/llvm"))
    return candidates


def find_tblgen(explicit: str | None) -> Path | None:
    """定位 llvm-tblgen；顺序与构建系统查找 MLIR 的顺序一致。"""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    for root in llvm_root_candidates():
        candidate = root / "bin" / "llvm-tblgen"
        if candidate.is_file():
            return candidate
    found = shutil.which("llvm-tblgen")
    return Path(found) if found else None


def find_include_dir(tblgen: Path) -> Path | None:
    """定位随 tblgen 一同安装的 MLIR .td 头目录（mlir/IR/OpBase.td 所在处）。"""
    root = tblgen.parent.parent
    for relative in ("include", "../include"):
        candidate = (root / relative).resolve()
        if (candidate / "mlir" / "IR" / "OpBase.td").is_file():
            return candidate
    return None


def dump_ods_records(tblgen: Path, include_dir: Path) -> dict:
    """用 tblgen 把 AFIROps.td 解析成 JSON 记录表。"""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "afir_ods.json"
        command = [
            str(tblgen),
            "--dump-json",
            "-I",
            str(AFIR_IR_DIR),
            "-I",
            str(include_dir),
            str(AFIR_OPS_TD),
            "-o",
            str(out),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"llvm-tblgen 解析 {AFIR_OPS_TD} 失败：\n{completed.stderr.strip()}"
            )
        return json.loads(out.read_text(encoding="utf-8"))


def collect_ops(records: dict) -> list[OpRecord]:
    """从 tblgen JSON 中抽出全部 Op 记录及其档位与文档状态。"""
    collected: list[OpRecord] = []
    for name in records.get("!instanceof", {}).get("Op", []):
        record = records[name]
        stability = record.get(STABILITY_FIELD) or TIER_INTERNAL
        missing = tuple(
            field
            for field in REQUIRED_DOC_FIELDS
            if not str(record.get(field) or "").strip()
        )
        collected.append(
            OpRecord(
                record=name,
                mnemonic=str(record.get("opName") or name),
                stability=str(stability),
                missing=missing,
            )
        )
    return sorted(collected, key=lambda op: op.mnemonic)


def report_summary(ops: list[OpRecord]) -> None:
    """打印各档位的文档覆盖率，使重构过程中的文档进展可见。"""
    OUTPUT_LOGGER.info("AFIR ODS 文档覆盖率（按稳定性档位）：")
    for tier in KNOWN_TIERS:
        tier_ops = [op for op in ops if op.stability == tier]
        if not tier_ops:
            continue
        documented = sum(1 for op in tier_ops if op.documented)
        mandatory = "强制" if tier in PUBLIC_TIERS else "不强制"
        OUTPUT_LOGGER.info(
            f"  {tier:<12} {documented:>3}/{len(tier_ops):<3} 已补齐 summary+description（{mandatory}）"
        )
    OUTPUT_LOGGER.info(
        f"  {'合计':<10} {sum(1 for op in ops if op.documented):>3}/{len(ops):<3}"
    )


def report_violations(violations: list[OpRecord]) -> None:
    ERROR_LOGGER.error(
        f"AFIR ODS 文档门禁失败：{len(violations)} 个对外公开档位的 Op 缺少文档。"
    )
    for op in violations:
        ERROR_LOGGER.error(
            f"  {op.mnemonic} ({op.record}) 缺少：{', '.join(op.missing)}"
        )
    ERROR_LOGGER.error("")
    ERROR_LOGGER.error(
        "stable 与 experimental 档都对外公开，其 summary / description 会进入生成的对外文档。"
    )
    ERROR_LOGGER.error(
        "请补齐文档，或在该 Op 上改回 `let afirStability = AFIR_Internal;`（分档判据见设计文档 2.5）。"
    )


def report_unknown_tiers(unknown: list[OpRecord]) -> None:
    ERROR_LOGGER.error(
        f"AFIR ODS 文档门禁失败：{len(unknown)} 个 Op 的稳定性档位取值非法。"
    )
    for op in unknown:
        ERROR_LOGGER.error(
            f"  {op.mnemonic} ({op.record}) 的 {STABILITY_FIELD} = {op.stability!r}"
        )
    ERROR_LOGGER.error(f"合法取值：{', '.join(KNOWN_TIERS)}（见 AFIROpBase.td）。")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 AFIR 对外公开档位（stable / experimental）Op 的 ODS 文档是否齐全。"
    )
    parser.add_argument(
        "--tblgen",
        help="llvm-tblgen 路径；默认按 LLVM_BUILD_DIR / MLIR_DIR / LLVM_DIR 顺序查找。",
    )
    parser.add_argument(
        "--summary", action="store_true", help="额外打印各档位的文档覆盖率统计。"
    )
    parser.add_argument(
        "--require-tblgen",
        action="store_true",
        help="找不到 llvm-tblgen 时判为失败，而不是跳过；CI 门禁应打开。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    tblgen = find_tblgen(args.tblgen)
    if tblgen is None:
        message = "未找到 llvm-tblgen，跳过 AFIR ODS 文档检查（设置 LLVM_BUILD_DIR 或用 --tblgen 指定）。"
        if args.require_tblgen:
            ERROR_LOGGER.error(message.replace("跳过", "无法执行"))
            return 1
        ERROR_LOGGER.warning(message)
        return 0

    include_dir = find_include_dir(tblgen)
    if include_dir is None:
        ERROR_LOGGER.error(
            f"在 {tblgen.parent.parent} 下未找到 MLIR 的 .td 头目录（mlir/IR/OpBase.td）。"
        )
        return 1

    try:
        records = dump_ods_records(tblgen, include_dir)
    except (RuntimeError, json.JSONDecodeError) as error:
        ERROR_LOGGER.error(str(error))
        return 1

    ops = collect_ops(records)
    if not ops:
        ERROR_LOGGER.error(
            f"未从 {AFIR_OPS_TD} 解析到任何 Op 记录，检查 tblgen 的 include 路径。"
        )
        return 1

    unknown = [op for op in ops if op.stability not in KNOWN_TIERS]
    violations = [
        op for op in ops if op.stability in PUBLIC_TIERS and not op.documented
    ]

    if args.summary:
        report_summary(ops)

    if unknown:
        report_unknown_tiers(unknown)
    if violations:
        report_violations(violations)
    if unknown or violations:
        return 1

    public_count = sum(1 for op in ops if op.stability in PUBLIC_TIERS)
    OUTPUT_LOGGER.info(
        f"AFIR ODS 文档门禁通过：{len(ops)} 个 Op，其中对外公开 {public_count} 个，文档齐全。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
