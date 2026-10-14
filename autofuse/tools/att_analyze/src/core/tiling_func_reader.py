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
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class SubItem:
    name: str
    value: float
    contribution: float = 0.0  # value / total


@dataclass
class NodePerfInfo:
    node_name: str
    node_type: str
    pipe_type: str
    sub_items: List[SubItem] = field(default_factory=list)
    total: float = 0.0
    is_bottleneck: bool = False


@dataclass
class CasePerfInfo:
    result_id: int
    group_id: int
    case_id: int
    nodes: List[NodePerfInfo] = field(default_factory=list)
    bottleneck_pipe: str = ""
    graph_id: int = 0


# 匹配 [PERF] 行，可选四维坐标标签
# [PERF][graph0_result1_g0_R0] Load_0[Load]: AIV_MTE2 = ...
# [PERF] Load_0[Load]: AIV_MTE2 = ...
_LINE_RE = re.compile(
    r"\[PERF\](?:\[graph(\d+)_result(\d+)_g(\d+)_(R?\d+)\])?\s*"
    r"(\w+)\[(\w+)\]:\s*(\w+)\s*=\s*(.+)"
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TOTAL_RE = re.compile(rf"=\s*({_NUMBER})\s*$")
_ITEM_RE = re.compile(rf"(\w+)\s*=\s*({_NUMBER})")


def _parse_rhs(rhs: str) -> Tuple[List[SubItem], float]:
    """解析右侧表达式，返回 (sub_items, total)"""
    total_match = _TOTAL_RE.search(rhs)
    total = float(total_match.group(1)) if total_match else 0.0

    items = []
    for m in _ITEM_RE.finditer(rhs):
        name, val = m.group(1), float(m.group(2))
        items.append(SubItem(name=name, value=val))

    # 若只有一个 item 且 total 未单独列出，用 item.value 作 total
    if not total_match and items:
        total = items[0].value

    for item in items:
        item.contribution = (item.value / total) if total > 0 else 0.0

    return items, total


def parse_perf_lines(
    log_content: str,
    operator_name: str,
    result_id: int = 0,
    group_id: int = 0,
    case_id: int = 0,
    graph_id: int = 0,
) -> List[CasePerfInfo]:
    """
    解析 [PERF] 行，返回 CasePerfInfo 列表。
    日志行若含 [graph*_result*_g*_*] 标签则按标签分组；否则全部归属到指定 (result_id, group_id, case_id)。
    """
    cases: Dict[tuple, CasePerfInfo] = {}

    for line in log_content.splitlines():
        if f"[{operator_name}]" not in line or "[PERF]" not in line:
            continue
        # 去掉算子前缀后解析
        idx = line.index("[PERF]")
        inner = line[idx:]
        m = _LINE_RE.match(inner)
        if not m:
            continue

        gr_id = int(m.group(1)) if m.group(1) else graph_id
        r_id = int(m.group(2)) if m.group(2) else result_id
        grp_id = int(m.group(3)) if m.group(3) else group_id
        raw_case = m.group(4)
        c_id = int(re.sub(r"^R", "", raw_case)) if raw_case else case_id

        node_name = m.group(5)
        node_type = m.group(6)
        pipe_type = m.group(7)
        rhs = m.group(8)

        sub_items, total = _parse_rhs(rhs)
        node = NodePerfInfo(
            node_name=node_name,
            node_type=node_type,
            pipe_type=pipe_type,
            sub_items=sub_items,
            total=total,
        )

        key = (gr_id, r_id, grp_id, c_id)
        if key not in cases:
            cases[key] = CasePerfInfo(
                result_id=r_id, group_id=grp_id, case_id=c_id, graph_id=gr_id
            )
        cases[key].nodes.append(node)

    # 计算瓶颈管道
    for info in cases.values():
        if not info.nodes:
            continue
        max_node = max(info.nodes, key=lambda n: n.total)
        max_node.is_bottleneck = True
        info.bottleneck_pipe = max_node.pipe_type

    return list(cases.values())
