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
"""AFIR 分层方向门禁。

设计文档 2.2 的依赖方向是**单向**的::

    afir            <- 不知道 transform.afir 存在
    transform.afir  -> 依赖 afir
    afplan          -> 挂在 afir 的 func 上，但不被 afir 语义依赖
    adapter         -> 依赖以上三者，反之不成立

验收 1 要求这条方向「有测试保证」。今天成立的是其中一条——``afir`` 不依赖
``afplan``——因为另外两个命名空间尚未注册（随 Schedule / ATT 迁移落地）。

还有一条同样的方向事实：``afir`` 不依赖**后端方言**。AFIR 是可对外公开的 kernel DSL，
它的分析与变换不该认识 ``ascendc`` / ``emitasc`` / ``emitc``；能搬进 ``dialect/AFIR/``
的判据正是「产物是不是 AFIR」（分层重构设计 1.1）。

**为什么是脚本而不是单元测试**　方向是一条关于**依赖**的事实，运行期测不出来：一个
新加的 ``#include "dialect/AFPlan/..."`` 只要那条路径没被某个用例走到，任何运行期断言
都不会变红。能看见依赖的只有源码与构建描述，所以判据落在这两处：

1. ``dialect/AFIR/`` 下的代码（注释与 ODS 文档串之外）不出现被禁的记号；
2. ``dialect/AFIR/`` 的 CMake 目标不链接被禁的库。

第二条不是第一条的重复：今天这些库一个都不链接，所以哪怕有人写了 ``#include`` 也会在
链接期失败——真正需要防住的回归是**有人往 LINK_LIBS 里加一行**，那一刻第一条的保护就
没了，而它不会让任何用例变红。后端这一组尤其如此：搬运做完之后 ``dialect/AFIR/`` 里会
住着一批曾经与后端同目录的 pass，那条隐式保护是这次重构唯一的看门人。

运行期那一半由 ``test/ut/dialect/AFIRLayeringTest.cpp`` 补上：只加载 ``afir`` 的
context 能解析并校验一个 kernel body，解析过程不会把 ``afplan`` 拉进来。

用法::

    python3 check_afir_layering.py            # 门禁：有违例则退出码非零
    python3 check_afir_layering.py --summary  # 追加扫描到的文件数
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AFIR_DIR = ROOT / "dialect" / "AFIR"

SOURCE_SUFFIXES = (".h", ".cpp", ".td")


@dataclass(frozen=True)
class Rule:
    """一组被禁的记号，连同它被禁的理由。"""

    tokens: tuple[str, ...]
    link_libs: tuple[str, ...]
    reason: str


RULES = (
    # AFIR 不得依赖的下游命名空间。新增 transform.afir 的 C++ 命名空间或 adapter 时同步补充。
    Rule(
        tokens=("afplan", "AFPlan"),
        link_libs=("AFPlanDialect",),
        reason="afir 不得依赖 afplan（设计文档 2.2 依赖方向单向）",
    ),
    # 后端方言。`ascir/` 是 AscendC/EmitAsc 方言头文件的路径前缀，单独列出是因为
    # `#include "ascir/Dialect/Asc/IR/Asc.h"` 这一行里没有 `ascendc` 这个词。
    Rule(
        tokens=("ascendc", "AscendC", "emitasc", "EmitAsc", "emitc", "EmitC", "ascir/"),
        link_libs=("MLIRAsc", "MLIREmitAsc", "MLIREmitCDialect", "MLIRTargetAsc"),
        reason="afir 不得依赖后端方言（分层重构设计 1.1：能搬进 dialect/AFIR/ 的判据是「产物是不是 AFIR」）",
    ),
)

COMMENT_LOGGER_NAME = f"{__name__}.stdout"


def create_stream_logger(name: str, stream) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


OUTPUT_LOGGER = create_stream_logger(COMMENT_LOGGER_NAME, sys.stdout)
ERROR_LOGGER = create_stream_logger(f"{__name__}.stderr", sys.stderr)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    text: str
    reason: str


INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)')


def blank_comments(source: str, tablegen: bool) -> str:
    """把注释、字符串字面量与 ODS 文档串替换成等长空白。

    保留行结构（换行不动），这样违例行号仍然对得上原文件。提到另一个命名空间的**说明**
    是应该鼓励的——归属判据本来就要解释「这个字段为什么归 afplan」；要拦的是代码。

    ``#include`` 的路径也是字符串字面量，会被一并抹掉，所以它由 :data:`INCLUDE_PATTERN`
    在抹之前单独扫一遍。漏掉这一条，门禁就拦不住依赖进门的**主要**方式。
    """
    out = list(source)
    index = 0
    length = len(source)
    while index < length:
        rest = source[index:]
        if rest.startswith("//"):
            end = source.find("\n", index)
            end = length if end < 0 else end
            for position in range(index, end):
                out[position] = " "
            index = end
            continue
        if rest.startswith("/*"):
            end = source.find("*/", index + 2)
            end = length if end < 0 else end + 2
            for position in range(index, end):
                if out[position] != "\n":
                    out[position] = " "
            index = end
            continue
        if tablegen and rest.startswith("[{"):
            end = source.find("}]", index + 2)
            end = length if end < 0 else end + 2
            for position in range(index, end):
                if out[position] != "\n":
                    out[position] = " "
            index = end
            continue
        if rest.startswith('"'):
            end = index + 1
            while end < length and source[end] != '"':
                end += 2 if source[end] == "\\" else 1
            end = min(end + 1, length)
            for position in range(index, end):
                if out[position] != "\n":
                    out[position] = " "
            index = end
            continue
        index += 1
    return "".join(out)


def scan_sources() -> tuple[list[Violation], int]:
    violations: list[Violation] = []
    scanned = 0
    patterns = [
        (re.compile("|".join(re.escape(token) for token in rule.tokens)), rule.reason)
        for rule in RULES
    ]
    for path in sorted(AFIR_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        scanned += 1
        source = path.read_text(encoding="utf-8")
        code = blank_comments(source, tablegen=path.suffix == ".td")
        original = source.splitlines()
        for number, line in enumerate(original, start=1):
            included = INCLUDE_PATTERN.match(line)
            if not included:
                continue
            for pattern, reason in patterns:
                if pattern.search(included.group(1)):
                    violations.append(
                        Violation(
                            path=path, line=number, text=line.strip(), reason=reason
                        )
                    )
        for number, line in enumerate(code.splitlines(), start=1):
            for pattern, reason in patterns:
                if pattern.search(line):
                    violations.append(
                        Violation(
                            path=path,
                            line=number,
                            text=original[number - 1].strip(),
                            reason=reason,
                        )
                    )
    return violations, scanned


def scan_build_files() -> tuple[list[Violation], int]:
    violations: list[Violation] = []
    scanned = 0
    for path in sorted(AFIR_DIR.rglob("CMakeLists.txt")):
        scanned += 1
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.split("#", 1)[0]
            for rule in RULES:
                for library in rule.link_libs:
                    if re.search(rf"\b{re.escape(library)}\b", stripped):
                        violations.append(
                            Violation(
                                path=path,
                                line=number,
                                text=line.strip(),
                                reason=f"afir 的构建目标不得链接 {library}（{rule.reason}）",
                            )
                        )
    return violations, scanned


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--summary", action="store_true", help="追加扫描到的文件数")
    arguments = parser.parse_args()

    if not AFIR_DIR.is_dir():
        ERROR_LOGGER.error("找不到 AFIR 方言目录：%s", AFIR_DIR)
        return 1

    source_violations, sources = scan_sources()
    build_violations, builds = scan_build_files()
    violations = source_violations + build_violations

    if arguments.summary:
        OUTPUT_LOGGER.info(
            "AFIR 分层方向门禁：扫描 %d 个源文件、%d 个 CMakeLists.txt", sources, builds
        )

    if violations:
        ERROR_LOGGER.error("AFIR 分层方向门禁失败：")
        for violation in violations:
            ERROR_LOGGER.error(
                "  %s:%d: %s\n    %s",
                violation.path.relative_to(ROOT),
                violation.line,
                violation.reason,
                violation.text,
            )
        ERROR_LOGGER.error(
            "依赖方向见 docs/design/2026-09-09-afir-public-dsl-architecture-design.md 2.2 与 "
            "docs/design/2026-09-14-afir-layering-refactor.md 1.1。\n"
            "  需要把某条信息从 afir 递给 afplan 时，方向是反的：属性挂在 afir 的容器上，由 afplan 自己校验。\n"
            "  需要在 AFIR 层的 pass 里发射后端算子时，拆开：分析留在 dialect/AFIR/，发射留在 conversion/。\n"
            "  两者之间用什么接，看通用的那半是不是一个算法（分层重构设计 2.1 的判据）：是算法就上 "
            "provider 回调（LoopGenerate.h 的 SymbolBindingProvider 是现成的形状）；只是一串目标决策，"
            "就让分析产出一张表、由 conversion 侧读表（Analysis/CanonicalBindings.h 是现成的形状），"
            "不要为此造一组 std::function。"
        )
        return 1

    OUTPUT_LOGGER.info("AFIR 分层方向门禁通过：afir 不依赖 afplan，也不依赖后端方言。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
