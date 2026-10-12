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
from typing import Dict, List, Tuple
from core.file_utils import ensure_output_dir, find_log_files
from commands.case_filter import parse_case_arg


# ──── 正则模式 ───────────────────────────────────────────────

_OUTER_BEGIN = re.compile(
    r"\[DFX\]Begin to gen model info for asc graph (\d+), schedule result (\d+), schedule group (\d+)"
)
_OUTER_END = re.compile(
    r"\[DFX\]End to gen model info for graph(\d+)_result(\d+)_g(\d+)"
)
_INNER_BEGIN = re.compile(
    r"\[DFX\]Begin to generate model info for graph \S+ of tiling case id (\d+)"
)
_INNER_END = re.compile(
    r"\[DFX\]End to generate model info for graph \S+ of tiling case id (\d+)"
)
_PROF_CASE = re.compile(r"graph(\d+)_result(\d+)_g(\d+)_(R?\d+)")


class SlogSplitter:
    @staticmethod
    def _case_id(content: str, fallback: int) -> int:
        match = re.search(r"tiling case id\s+(?:R)?(\d+)", content, re.I)
        if match:
            return int(match.group(1))
        if not match:
            match = _PROF_CASE.search(content)
            if match:
                return int(re.sub(r"^R", "", match.group(4)))
        return fallback

    def parse_compiler_model_info(self, log_content: str) -> Dict[Tuple, List[str]]:
        """返回 {(graph, result, group): [case1_content, case2_content, ...]}"""
        result: Dict[Tuple, List[str]] = {}
        current_group = None
        in_inner = False
        current_case_lines: List[str] = []
        case_list: List[str] = []

        for line in log_content.splitlines(keepends=True):
            m = _OUTER_BEGIN.search(line)
            if m:
                current_group = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                case_list = []
                continue

            m = _OUTER_END.search(line)
            if m and current_group:
                result[current_group] = case_list
                current_group = None
                continue

            m = _INNER_BEGIN.search(line)
            if m and current_group:
                in_inner = True
                current_case_lines = [line]
                continue

            m = _INNER_END.search(line)
            if m and in_inner:
                in_inner = False
                case_list.append("".join(current_case_lines))
                current_case_lines = []
                continue

            if in_inner:
                current_case_lines.append(line)

        return result

    def parse_runtime_cases(
        self, log_content: str, operator_name: str
    ) -> Dict[Tuple, List[str]]:
        """返回 {(graph, result, group): [case1_content, case2_content, ...]}"""
        pending: Dict[Tuple, List[str]] = {}
        for line in log_content.splitlines(keepends=True):
            if f"[{operator_name}]" not in line:
                continue
            m = _PROF_CASE.search(line)
            if m:
                key = (
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(re.sub(r"^R", "", m.group(4))),
                )
                pending.setdefault(key, []).append(line)

        # 按 (graph, result, group) 分组，按 case 顺序排列
        result: Dict[Tuple, List[str]] = {}
        for (g, r, grp, c), lines in sorted(pending.items()):
            group_key = (g, r, grp)
            result.setdefault(group_key, []).append("".join(lines))
        return result

    def write_compiler_output(
        self, groups: Dict[Tuple, List[str]], op: str, out_base: str
    ):
        for (g, r, grp), cases in groups.items():
            dir_path = os.path.join(
                out_base, op, "compiler", f"graph{g}_result{r}", f"g{grp}"
            )
            ensure_output_dir(dir_path)
            for i, content in enumerate(cases):
                case_id = self._case_id(content, i)
                with open(
                    os.path.join(dir_path, f"case{case_id}.log"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(content)

    def write_runtime_output(
        self, groups: Dict[Tuple, List[str]], op: str, out_base: str
    ):
        for (g, r, grp), cases in groups.items():
            dir_path = os.path.join(
                out_base, op, "runtime", f"graph{g}_result{r}", f"g{grp}"
            )
            ensure_output_dir(dir_path)
            for i, content in enumerate(cases):
                case_id = self._case_id(content, i)
                with open(
                    os.path.join(dir_path, f"case{case_id}.log"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(content)


def run(args):
    log_files = find_log_files(args.log_path)
    full_content = ""
    for path in log_files:
        with open(path, encoding="utf-8", errors="replace") as f:
            full_content += f.read()

    from core.log_parser import LogParser

    parser = LogParser()
    op_names = [args.op] if args.op else parser.extract_operator_names(full_content)

    splitter = SlogSplitter()
    out_base = args.output

    for op in op_names:
        compiler_groups = splitter.parse_compiler_model_info(full_content)
        runtime_groups = splitter.parse_runtime_cases(full_content, op)
        case_filter = parse_case_arg(args.case)
        if case_filter:
            compiler_groups = {
                key: [
                    content
                    for index, content in enumerate(values)
                    if case_filter.match(
                        key[1], key[2], splitter._case_id(content, index)
                    )
                ]
                for key, values in compiler_groups.items()
            }
            runtime_groups = {
                key: [
                    content
                    for index, content in enumerate(values)
                    if case_filter.match(
                        key[1], key[2], splitter._case_id(content, index)
                    )
                ]
                for key, values in runtime_groups.items()
            }
        splitter.write_compiler_output(compiler_groups, op, out_base)
        splitter.write_runtime_output(runtime_groups, op, out_base)
        print(
            f"[split-slog] {op}: compiler={len(compiler_groups)} groups, runtime={len(runtime_groups)} groups"
        )
