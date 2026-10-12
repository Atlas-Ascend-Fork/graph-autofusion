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
import statistics
from html import escape
from typing import List, Optional, Tuple
from core.tiling_func_reader import CasePerfInfo, NodePerfInfo, parse_perf_lines
from core.log_parser import LogParser
from core.file_utils import find_log_files, ensure_output_dir
from commands.case_filter import parse_case_arg

# ──── SVG 布局常数 ────────────────────────────────────────────
_CANVAS_W = 900
_FONT_SIZE = 12
_LINE_H = 20
_BAR_H = 14
_OP_GAP = 40
_GROUP_GAP = 16
_OP_TITLE_H = 40
_CASE_TAB_H = 28
_MARGIN = 20


def _bar(x: int, y: int, ratio: float, fill: str, max_w: int) -> str:
    bar_w = max(1, int(ratio * max_w))
    return f'<rect x="{x}" y="{y}" width="{bar_w}" height="{_BAR_H}" fill="{fill}" opacity="0.7"/>'


def _text(
    x: int,
    y: int,
    content: str,
    fill: str = "#222",
    anchor: str = "start",
    size: Optional[int] = None,
) -> str:
    sz = size or _FONT_SIZE
    return f'<text x="{x}" y="{y}" font-size="{sz}" fill="{escape(fill, quote=True)}" text-anchor="{anchor}" font-family="monospace">{escape(str(content))}</text>'


def _render_node(node: NodePerfInfo, x: int, y: int, col_w: int) -> Tuple[str, int]:
    """渲染单个节点，返回 (fragment, height_used)"""
    parts = []
    pipe_color = "red" if node.is_bottleneck else "#333"
    pipe_label = f"{node.pipe_type} = {node.total:.2f}"
    if node.is_bottleneck:
        pipe_label += " \U0001f534"
    parts.append(_text(x, y + _FONT_SIZE, pipe_label, fill=pipe_color))
    h = _LINE_H

    bar_x = x + 130
    bar_max_w = max(col_w - 140, 10)
    for item in node.sub_items:
        if item.value == 0:
            continue
        label = f"{item.name:<16} {item.contribution * 100:.1f}%"
        parts.append(_text(x, y + h + _FONT_SIZE, label))
        parts.append(_bar(bar_x, y + h, item.contribution, "#4a90d9", bar_max_w))
        h += _LINE_H

    return "\n".join(parts), h + 4


def _render_group(
    cases: List[CasePerfInfo],
    group_id: int,
    x: int,
    y: int,
    col_w: int,
    selected_case_id: Optional[int],
) -> Tuple[str, int]:
    """渲染单个 group 列，返回 (fragment, height)"""
    parts = [_text(x, y + _FONT_SIZE, f"group{group_id}", size=_FONT_SIZE + 1)]
    h = _LINE_H + 4

    # case 标签行
    tab_y = y + h
    tab_x = x
    for c in cases:
        is_sel = c.case_id == selected_case_id
        label = f"[case{c.case_id}{'★' if is_sel else ''}]"
        fill = "#1a6fb5" if is_sel else "#888"
        parts.append(_text(tab_x, tab_y + _FONT_SIZE, label, fill=fill))
        tab_x += len(label) * 7 + 4
    h += _CASE_TAB_H

    # 显示选中 case 的节点
    display_case = next(
        (c for c in cases if c.case_id == selected_case_id), cases[0] if cases else None
    )
    if display_case:
        for node in display_case.nodes:
            frag, node_h = _render_node(node, x + 4, y + h, col_w - 8)
            parts.append(frag)
            h += node_h + 4

    # 跨 case 方差对比
    if len(cases) > 1:
        item_vals: dict = {}
        for c in cases:
            for node in c.nodes:
                for si in node.sub_items:
                    item_vals.setdefault(si.name, []).append(si.value)
        if item_vals:
            max_var_name = max(
                item_vals,
                key=lambda k: statistics.variance(item_vals[k])
                if len(item_vals[k]) > 1
                else 0,
            )
            vals = item_vals[max_var_name]
            parts.append(
                _text(
                    x,
                    y + h + _FONT_SIZE,
                    f"敏感参数: {max_var_name}（跨 case 方差最大）",
                    fill="#b85c00",
                )
            )
            h += _LINE_H
            # A valid profiling record may contain an all-zero metric (for
            # example, when a pipe was not used).  Keep the chart renderable
            # in that case instead of dividing by zero below.
            max_v = max(vals) if vals and max(vals) != 0 else 1
            for c, v in zip(cases, vals):
                bar_w = int((v / max_v) * (col_w - 80))
                bar_y = y + h
                parts.append(_text(x, bar_y + _FONT_SIZE, f"case{c.case_id}"))
                parts.append(
                    f'<rect x="{x + 50}" y="{bar_y}" width="{bar_w}" height="{_BAR_H}" fill="#e07b39" opacity="0.7"/>'
                )
                parts.append(_text(x + 55 + bar_w, bar_y + _FONT_SIZE, f"{v:.2f}"))
                h += _LINE_H

    return "\n".join(parts), h


def render_svg(
    op_name: str,
    cases: List[CasePerfInfo],
    selected_case: Optional[Tuple[int, int, int]],
) -> Tuple[str, int]:
    """
    生成单算子的 SVG <g> 片段，返回 (fragment, height)。
    由 build_full_svg 负责包裹为完整 <svg> 根元素。
    """
    groups: dict = {}
    for c in cases:
        groups.setdefault(c.group_id, []).append(c)

    group_ids = sorted(groups.keys())
    n_groups = len(group_ids)
    cols_per_row = min(n_groups, 2) if n_groups > 3 else max(n_groups, 1)
    col_w = (_CANVAS_W - _MARGIN * 2 - _GROUP_GAP * (cols_per_row - 1)) // cols_per_row

    selected_str = ""
    if selected_case:
        r, g, c = selected_case
        selected_str = f"Selected: graph0_result{r}_g{g}_case{c}"

    inner_parts = [
        f'<rect width="{_CANVAS_W}" height="{_OP_TITLE_H}" fill="#f0f4f8" rx="4"/>',
        _text(_MARGIN, _FONT_SIZE + 6, op_name, size=_FONT_SIZE + 4, fill="#1a1a1a"),
    ]
    if selected_str:
        inner_parts.append(
            _text(_MARGIN, _FONT_SIZE * 2 + 14, selected_str, fill="#555")
        )
    inner_parts.append(
        f'<line x1="{_MARGIN}" y1="{_OP_TITLE_H - 4}" x2="{_CANVAS_W - _MARGIN}" y2="{_OP_TITLE_H - 4}" stroke="#ccc"/>'
    )

    body_y = _OP_TITLE_H
    max_row_h = 0
    for col_idx, gid in enumerate(group_ids):
        col = col_idx % cols_per_row
        if col == 0 and col_idx > 0:
            body_y += max_row_h + _GROUP_GAP
            max_row_h = 0
        x = _MARGIN + col * (col_w + _GROUP_GAP)
        y = body_y

        sel_case_id = (
            selected_case[2] if selected_case and selected_case[1] == gid else None
        )
        frag, gh = _render_group(groups[gid], gid, x, y, col_w, sel_case_id)
        inner_parts.append(frag)
        max_row_h = max(max_row_h, gh)

    total_h = body_y + max_row_h + _OP_GAP

    fragment = "<g>\n" + "\n".join(inner_parts) + "\n</g>"
    return fragment, total_h


def build_full_svg(sections: List[Tuple[str, int]]) -> str:
    """将多个算子 section 垂直堆叠，包裹为合法单 <svg> 根元素"""
    total_h = sum(h for _, h in sections)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_CANVAS_W}" height="{total_h}">'
    ]
    y_offset = 0
    for frag, h in sections:
        svg_parts.append(f'<g transform="translate(0,{y_offset})">')
        svg_parts.append(frag)
        svg_parts.append("</g>")
        y_offset += h
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def run(args):
    if not os.path.isdir(args.source_dir):
        print(f"[perf-formula] 源目录不存在: {args.source_dir}")
        return 2
    log_files = find_log_files(args.log_path)
    full_content = ""
    for path in log_files:
        with open(path, encoding="utf-8", errors="replace") as f:
            full_content += f.read()

    log_parser = LogParser()
    op_names = log_parser.extract_operator_names(full_content)
    case_filter = parse_case_arg(args.case)
    out_dir = ensure_output_dir(args.output)
    sections: List[Tuple[str, int]] = []

    for op in op_names:
        graph_id, result_id = log_parser.extract_graph_result(full_content, op)
        group_cases = log_parser.extract_all_group_cases(
            full_content, op, graph_id, result_id
        )
        perf_cases = parse_perf_lines(full_content, op)

        if case_filter:
            perf_cases = [
                c
                for c in perf_cases
                if case_filter.match(c.result_id, c.group_id, c.case_id)
            ]
        elif group_cases:
            selected_keys = {(result_id, gid, cid) for gid, cid in group_cases.items()}
            filtered = [
                c
                for c in perf_cases
                if (c.result_id, c.group_id, c.case_id) in selected_keys
            ]
            perf_cases = filtered or perf_cases

        if not perf_cases:
            print(f"[perf-formula] {op}: 未找到 [PERF] 日志行，跳过")
            continue

        selected_case = None
        if group_cases:
            first_gid = min(group_cases.keys())
            selected_case = (result_id, first_gid, group_cases[first_gid])

        frag, h = render_svg(op, perf_cases, selected_case)
        sections.append((frag, h))

    out_path = os.path.join(out_dir, "perf_formula.svg")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_full_svg(sections))
    print(f"[perf-formula] 输出: {out_path}")
    return 0 if sections else 1
