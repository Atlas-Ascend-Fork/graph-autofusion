#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Render SuperKernel round-report JSON into Chinese Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _as_round_report(data, source):
    if isinstance(data, dict) and isinstance(data.get("round_report"), dict):
        data = data["round_report"]
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a JSON object")
    required = ["effective_fusion", "per_sk_fusion_table", "non_fusion_operator_table"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{source}: missing round-report fields: {', '.join(missing)}")
    return data


def load_round_report(path):
    return _as_round_report(json.loads(Path(path).read_text()), path)


def _escape(value):
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple)):
        return ", ".join(_escape(item) for item in value) or "-"
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    return text.replace("|", r"\|").replace("\n", "<br>")


def _portable_path(value):
    if value is None:
        return None
    path = Path(str(value))
    return path.name if path.is_absolute() else path.as_posix()


def _histogram(value):
    if not value:
        return "-"

    def key(item):
        raw = item[0]
        try:
            return (0, int(raw))
        except (TypeError, ValueError):
            return (1, str(raw))

    return ", ".join(
        f"{count}:{total}" for count, total in sorted(value.items(), key=key)
    )


def _percent(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}%"


def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape(cell) for cell in row) + " |")
    return lines


def _sort_sk_rows(rows):
    return sorted(
        rows or [],
        key=lambda item: (
            not bool(item.get("counts_as_effective_fusion")),
            -(item.get("child_count") or 0),
            str(item.get("function") or ""),
        ),
    )


def _op_sequence(row, limit=12):
    sequence = row.get("op_sequence") or row.get("child_functions") or []
    if not sequence:
        return "-"
    sequence = [str(item) for item in sequence if item]
    if len(sequence) > limit:
        sequence = sequence[:limit] + [f"... 共{len(sequence)}个"]
    return " -> ".join(sequence)


def _scope_text(row):
    parts = []
    source_scope = row.get("source_scope")
    if source_scope:
        parts.append(str(source_scope))
    boundary = row.get("boundary") or {}
    start_op = boundary.get("start_op")
    end_op = boundary.get("end_op")
    if start_op or end_op:
        parts.append(f"{start_op or '?'} -> {end_op or '?'}")
    scope_id = row.get("scope_id")
    if scope_id is not None:
        parts.append(f"scope_id={scope_id}")
    return "; ".join(parts) or "-"


def _example_text(row):
    parts = []
    function = row.get("example_function")
    if function:
        parts.append(str(function))
    node_id = row.get("example_node_id")
    if node_id is not None:
        parts.append(f"node={node_id}")
    stream_id = row.get("example_stream_id")
    if stream_id is not None:
        parts.append(f"stream={stream_id}")
    scope_names = row.get("example_scope_names")
    if scope_names:
        parts.append("scope=" + ",".join(str(item) for item in scope_names))
    return "; ".join(parts) or "-"


def _overview_rows(round_report):
    effective = round_report.get("effective_fusion") or {}
    single_child = round_report.get("single_child_exclusion") or {}
    return [
        ("结构描述阈值", f"child_count >= {round_report.get('min_child_nodes', 5)}"),
        ("融合组总数", effective.get("total_group_count", 0)),
        ("有效融合组", effective.get("effective_group_count", 0)),
        ("浅融合组", effective.get("shallow_group_count", 0)),
        ("单子算子 SK 数量", single_child.get("single_child_group_count", 0)),
        ("性能动作来源", "fresh profiling analysis"),
        (
            "2..4 子算子浅融合数量",
            single_child.get("shallow_non_single_child_group_count", 0),
        ),
        ("有效 child 总数", effective.get("effective_child_node_total", 0)),
        ("浅融合 child 总数", effective.get("shallow_child_node_total", 0)),
        (
            "有效 child 分布",
            _histogram(effective.get("effective_child_count_histogram")),
        ),
        (
            "浅融合 child 分布",
            _histogram(effective.get("shallow_child_count_histogram")),
        ),
        ("有效预计 launch 减少", effective.get("effective_launch_reduction", 0)),
        ("全部融合预计 launch 减少", effective.get("all_fused_launch_reduction", 0)),
    ]


def _single_child_rows(round_report, limit):
    rows = []
    for row in (round_report.get("single_child_exclusion") or {}).get("candidates", [])[
        :limit
    ]:
        rows.append(
            [
                row.get("function"),
                _scope_text(row),
                row.get("child_functions") or row.get("op_sequence"),
                _portable_path(row.get("path")),
                "pending profiling analysis",
            ]
        )
    return rows


def _sk_rows(round_report, limit):
    rows = []
    for row in _sort_sk_rows(round_report.get("per_sk_fusion_table"))[:limit]:
        rows.append(
            [
                row.get("function"),
                _scope_text(row),
                row.get("layer"),
                row.get("segments"),
                row.get("child_count"),
                row.get("counts_as_effective_fusion"),
                row.get("stream_ids"),
                f"{row.get('launch_count_without_sk', '-')}/{row.get('launch_count_with_sk', '-')}",
                row.get("estimated_launch_reduction"),
                row.get("control_core_mode"),
                _op_sequence(row),
                row.get("path"),
            ]
        )
    return rows


def _non_fusion_rows(round_report, limit):
    rows = []
    for row in (round_report.get("non_fusion_operator_table") or [])[:limit]:
        rows.append(
            [
                row.get("source"),
                row.get("reason"),
                row.get("count"),
                _percent(row.get("percent")),
                row.get("op_type"),
                row.get("kernel_type"),
                _example_text(row),
                row.get("example_detail") or row.get("example_break_reason"),
                row.get("action"),
            ]
        )
    return rows


def render_markdown(
    reports, *, title="SuperKernel 单轮融合报告", top_sk=20, top_non_fusion=20
):
    lines = [
        f"# {title}",
        "",
        "> 该报告由 `round-report.child-ge-5.json` 或 `sk-meta-summary.json` 渲染，仅解释 metadata。profiler interval、clean performance 和 promotion decision 仍以 `round-evidence.json` 或最终 `REPORT.md` 为准。",
    ]

    for source, round_report in reports:
        source_path = Path(source)
        round_name = (
            round_report.get("round_name") or source_path.parent.name or "unknown"
        )
        lines.extend(
            ["", f"## {round_name}", "", f"- 来源: `{_portable_path(source_path)}`"]
        )

        lines.extend(["", "### 融合概览", ""])
        lines.extend(_table(["指标", "值"], _overview_rows(round_report)))

        lines.extend(["", "### Child count 结构描述", ""])
        lines.append(
            "child_count、深度和预计 launch 减少只描述 metadata。所有可靠 mapped SK "
            "都进入 fresh profiling；keep/prune/reprofile/block 只由 profiling 分析决定。"
        )
        single_child_rows = _single_child_rows(round_report, top_sk)
        if single_child_rows:
            lines.extend(
                _table(
                    ["SK function", "Scope / 边界", "单子算子", "Path", "性能状态"],
                    single_child_rows,
                )
            )
        else:
            lines.append("未发现 child_count == 1 的 SK。")

        lines.extend(["", f"### SK 融合范围明细（最多 {top_sk} 条）", ""])
        sk_rows = _sk_rows(round_report, top_sk)
        if sk_rows:
            lines.extend(
                _table(
                    [
                        "SK function",
                        "Scope / 边界",
                        "Layer",
                        "Segments",
                        "Child",
                        "有效",
                        "Streams",
                        "Launch 前/后",
                        "预计减少",
                        "Control core",
                        "Child op sequence",
                        "Path",
                    ],
                    sk_rows,
                )
            )
        else:
            lines.append("未发现 SK 融合组。")

        lines.extend(["", f"### 未融合或断裂原因（最多 {top_non_fusion} 条）", ""])
        non_fusion_rows = _non_fusion_rows(round_report, top_non_fusion)
        if non_fusion_rows:
            lines.extend(
                _table(
                    [
                        "Source",
                        "Reason",
                        "Count",
                        "Percent",
                        "Op type",
                        "Kernel type",
                        "Example",
                        "Detail",
                        "Action",
                    ],
                    non_fusion_rows,
                )
            )
        else:
            lines.append("未发现未融合或 scope 断裂记录。")

    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render SuperKernel round-report JSON into Chinese Markdown."
    )
    parser.add_argument(
        "json",
        nargs="+",
        type=Path,
        help="round-report.child-ge-5.json or sk-meta-summary.json",
    )
    parser.add_argument("--markdown-out", type=Path, help="write Markdown to this path")
    parser.add_argument("--title", default="SuperKernel 单轮融合报告")
    parser.add_argument("--top-sk", type=int, default=20)
    parser.add_argument("--top-non-fusion", type=int, default=20)
    args = parser.parse_args(argv)

    if args.top_sk < 1:
        parser.error("--top-sk must be >= 1")
    if args.top_non_fusion < 1:
        parser.error("--top-non-fusion must be >= 1")

    reports = [(path, load_round_report(path)) for path in args.json]
    markdown = render_markdown(
        reports,
        title=args.title,
        top_sk=args.top_sk,
        top_non_fusion=args.top_non_fusion,
    )
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown + "\n")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
