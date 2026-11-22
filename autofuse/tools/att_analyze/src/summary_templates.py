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
# 空壳转发：LogParser 和 OperatorSummary 已迁移到 core/log_parser.py
import os
import csv
import json
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.log_parser import LogParser, OperatorSummary  # noqa: F401
from core.evidence_schema import FinalTilingRecord, normalize_pipe_estimates


_FINAL_COLUMNS = [
    "Final Source",
    "Tiling Key",
    "Score",
    "Pipe Estimate",
    "Tiling Repr",
    "Final Parse Status",
]
_TOPN_COLUMNS = ["Output Index", "Source Path", "Source Line"]


def _compact_json(value):
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _final_row(
    record: FinalTilingRecord,
    summary: Optional[OperatorSummary] = None,
    parse_status: Optional[str] = None,
):
    row = {
        "Operator": record.op or (summary.operator_name if summary else ""),
        "Graph": record.graph
        if record.graph is not None
        else (summary.graph if summary else ""),
        "Result": record.result
        if record.result is not None
        else (summary.result if summary else ""),
        "Group": record.group
        if record.group is not None
        else (summary.group if summary else ""),
        "Case": record.case
        if record.case is not None
        else (summary.case if summary else ""),
        "AIV_MTE2": summary.aiv_mte2
        if summary and summary.aiv_mte2 is not None
        else "",
        "AIV_MTE3": summary.aiv_mte3
        if summary and summary.aiv_mte3 is not None
        else "",
        "Objective Value": summary.objective_value
        if summary and summary.objective_value is not None
        else "",
        "Result Perf": summary.result_performance
        if summary and summary.result_performance is not None
        else "",
        "Final Source": record.source or "",
        "Tiling Key": record.tiling_key if record.tiling_key is not None else "",
        "Score": record.score if record.score is not None else "",
        "Pipe Estimate": _compact_json(normalize_pipe_estimates(record.pipe_est)),
        "Tiling Repr": _compact_json(record.tiling_repr),
        "Final Parse Status": parse_status or record.parse_status or "",
    }
    if summary:
        # FINAL_TILING carries the final identity/repr while historical CSV
        # columns still come from the legacy PROF observations.
        row.update(summary.tiling_values)
    return row


def _topn_row(record: FinalTilingRecord):
    return {
        **_final_row(record),
        "Output Index": record.output_index,
        "Source Path": record.source_path,
        "Source Line": record.source_line,
    }


def find_log_files(path):
    """递归查找目录下的所有日志文件"""
    log_files = []
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith(".log"):
                    log_files.append(os.path.join(root, file))
        return sorted(log_files)
    else:
        return []


def print_summary_table(summaries: List[OperatorSummary]):
    if not summaries:
        print("No data to display.")
        return

    all_tiling_keys = set()
    for summary in summaries:
        all_tiling_keys.update(summary.tiling_values.keys())

    fixed_tiling_keys = ["ub_size", "block_dim"]
    performance_keys = ["AIV_MTE2", "AIV_MTE3"]
    dynamic_tiling_keys = sorted(
        all_tiling_keys - set(fixed_tiling_keys) - set(performance_keys)
    )
    all_columns = [
        "Operator",
        "Graph",
        "Result",
        "Group",
        "Case",
        "AIV_MTE2",
        "AIV_MTE3",
        "Objective Value",
        "Result Perf",
    ]
    all_columns.extend(dynamic_tiling_keys)
    all_columns.extend(fixed_tiling_keys)

    col_widths = {}
    for col in all_columns:
        col_widths[col] = max(len(col), 15)

    for summary in summaries:
        col_widths["Operator"] = max(col_widths["Operator"], len(summary.operator_name))

    header = "  ".join(col.ljust(col_widths[col]) for col in all_columns)
    print(header)
    print("-" * len(header))

    for summary in summaries:
        row = []
        row.append(summary.operator_name.ljust(col_widths["Operator"]))
        row.append(str(summary.graph).ljust(col_widths["Graph"]))
        row.append(str(summary.result).ljust(col_widths["Result"]))
        row.append(str(summary.group).ljust(col_widths["Group"]))
        row.append(str(summary.case).ljust(col_widths["Case"]))
        row.append(
            str(summary.aiv_mte2 if summary.aiv_mte2 is not None else "N/A").ljust(
                col_widths["AIV_MTE2"]
            )
        )
        row.append(
            str(summary.aiv_mte3 if summary.aiv_mte3 is not None else "N/A").ljust(
                col_widths["AIV_MTE3"]
            )
        )
        row.append(
            str(
                summary.objective_value
                if summary.objective_value is not None
                else "N/A"
            ).ljust(col_widths["Objective Value"])
        )
        row.append(
            str(
                summary.result_performance
                if summary.result_performance is not None
                else "N/A"
            ).ljust(col_widths["Result Perf"])
        )

        for key in dynamic_tiling_keys:
            value = summary.tiling_values.get(key, "N/A")
            row.append(str(value).ljust(col_widths[key]))

        for key in fixed_tiling_keys:
            value = summary.tiling_values.get(key, "N/A")
            row.append(str(value).ljust(col_widths[key]))

        print("  ".join(row))


def print_tiling_table(records: Sequence[FinalTilingRecord], *, topn: bool = False):
    """Print structured tiling records."""
    if not records:
        return

    columns = [
        "Operator",
        "Graph",
        "Result",
        "Group",
        "Case",
        "Final Source",
        "Tiling Key",
        "Score",
        "Pipe Estimate",
        "Tiling Repr",
        "Final Parse Status",
    ]
    if topn:
        print("TopN candidates")
        columns.extend(_TOPN_COLUMNS)
    print("  ".join(columns))
    print("-" * (len("  ".join(columns)) + 1))
    for record in records:
        row = _topn_row(record) if topn else _final_row(record)
        print("  ".join(str(row[column]) for column in columns))


def export_to_csv(
    summaries: List[OperatorSummary],
    output_file: str,
    final_records: Optional[Sequence[FinalTilingRecord]] = None,
    *,
    topn_records: Sequence[FinalTilingRecord] = (),
):
    final_records = list(final_records or [])
    if not summaries and not final_records and not topn_records:
        print("No data to export.")
        return

    all_tiling_keys = set()
    for summary in summaries:
        all_tiling_keys.update(summary.tiling_values.keys())

    fixed_tiling_keys = ["ub_size", "block_dim"]
    performance_keys = ["AIV_MTE2", "AIV_MTE3"]
    dynamic_tiling_keys = sorted(
        all_tiling_keys - set(fixed_tiling_keys) - set(performance_keys)
    )

    fieldnames = [
        "Operator",
        "Graph",
        "Result",
        "Group",
        "Case",
        "AIV_MTE2",
        "AIV_MTE3",
        "Objective Value",
        "Result Perf",
    ]
    fieldnames.extend(dynamic_tiling_keys)
    fieldnames.extend(fixed_tiling_keys)
    fieldnames.extend(_FINAL_COLUMNS)
    if topn_records:
        fieldnames.extend(_TOPN_COLUMNS)

    summary_by_key = {
        (item.operator_name, item.graph, item.result, item.group): item
        for item in summaries
    }

    with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        seen_final_keys = set()
        export_items = []
        for record in final_records:
            identity = (record.op, record.graph, record.result, record.group)
            duplicate = identity in seen_final_keys and record.parse_status == "ok"
            seen_final_keys.add(identity)
            export_items.append(
                _final_row(
                    record,
                    summary_by_key.get(identity),
                    "duplicate_final_tiling" if duplicate else None,
                )
            )
        if not final_records:
            export_items = []
            for summary in summaries:
                row = {
                    "Operator": summary.operator_name,
                    "Graph": summary.graph,
                    "Result": summary.result,
                    "Group": summary.group,
                    "Case": summary.case,
                    "AIV_MTE2": summary.aiv_mte2
                    if summary.aiv_mte2 is not None
                    else "",
                    "AIV_MTE3": summary.aiv_mte3
                    if summary.aiv_mte3 is not None
                    else "",
                    "Objective Value": summary.objective_value
                    if summary.objective_value is not None
                    else "",
                    "Result Perf": summary.result_performance
                    if summary.result_performance is not None
                    else "",
                    **{
                        key: summary.tiling_values.get(key, "")
                        for key in dynamic_tiling_keys
                    },
                    **{
                        key: summary.tiling_values.get(key, "")
                        for key in fixed_tiling_keys
                    },
                    **{key: "" for key in _FINAL_COLUMNS},
                }
                export_items.append(row)

        writer.writerows(export_items)
        writer.writerows(_topn_row(record) for record in topn_records)


def export_to_excel(
    summaries: List[OperatorSummary],
    output_file: str,
    final_records: Optional[Sequence[FinalTilingRecord]] = None,
    *,
    topn_records: Sequence[FinalTilingRecord] = (),
):
    final_records = list(final_records or [])
    try:
        import openpyxl
        from openpyxl.styles import Font, Alignment, PatternFill
    except ImportError:
        print("Warning: openpyxl not installed. Falling back to CSV format.")
        csv_file = output_file.replace(".xlsx", ".csv").replace(".xls", ".csv")
        export_to_csv(
            summaries, csv_file, final_records=final_records, topn_records=topn_records
        )
        print(f"Data exported to {csv_file}")
        return

    if not summaries and not final_records and not topn_records:
        print("No data to export.")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"

    all_tiling_keys = set()
    for summary in summaries:
        all_tiling_keys.update(summary.tiling_values.keys())

    fixed_tiling_keys = ["ub_size", "block_dim"]
    performance_keys = ["AIV_MTE2", "AIV_MTE3"]
    dynamic_tiling_keys = sorted(
        all_tiling_keys - set(fixed_tiling_keys) - set(performance_keys)
    )

    fieldnames = [
        "Operator",
        "Graph",
        "Result",
        "Group",
        "Case",
        "AIV_MTE2",
        "AIV_MTE3",
        "Objective Value",
        "Result Perf",
    ]
    fieldnames.extend(dynamic_tiling_keys)
    fieldnames.extend(fixed_tiling_keys)
    if final_records or topn_records:
        fieldnames.extend(_FINAL_COLUMNS)
    if topn_records:
        fieldnames.extend(_TOPN_COLUMNS)

    header_fill = PatternFill(
        start_color="4472C4", end_color="4472C4", fill_type="solid"
    )
    header_font = Font(bold=True, color="FFFFFF")

    for col_num, header in enumerate(fieldnames, 1):
        cell = ws.cell(row=1, column=col_num, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    summary_by_key = {
        (summary.operator_name, summary.graph, summary.result, summary.group): summary
        for summary in summaries
    }
    if final_records:
        rows = [
            _final_row(
                record,
                summary_by_key.get(
                    (record.op, record.graph, record.result, record.group)
                ),
            )
            for record in final_records
        ]
    else:
        rows = []
        for summary in summaries:
            rows.append(
                {
                    "Operator": summary.operator_name,
                    "Graph": summary.graph,
                    "Result": summary.result,
                    "Group": summary.group,
                    "Case": summary.case,
                    "AIV_MTE2": summary.aiv_mte2
                    if summary.aiv_mte2 is not None
                    else "",
                    "AIV_MTE3": summary.aiv_mte3
                    if summary.aiv_mte3 is not None
                    else "",
                    "Objective Value": summary.objective_value
                    if summary.objective_value is not None
                    else "",
                    "Result Perf": summary.result_performance
                    if summary.result_performance is not None
                    else "",
                    **{
                        key: summary.tiling_values.get(key, "")
                        for key in dynamic_tiling_keys
                    },
                    **{
                        key: summary.tiling_values.get(key, "")
                        for key in fixed_tiling_keys
                    },
                }
            )

    rows.extend(_topn_row(record) for record in topn_records)
    for row_num, data in enumerate(rows, 2):
        for col_num, header in enumerate(fieldnames, 1):
            cell = ws.cell(row=row_num, column=col_num, value=data.get(header, ""))
            cell.alignment = Alignment(horizontal="center", vertical="center")

    for col_num in range(1, len(fieldnames) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_num)].width = 15

    wb.save(output_file)
    print(f"Data exported to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse log files and generate summary tables"
    )
    parser.add_argument(
        "log_path", help="Path to log file or directory containing log files"
    )
    parser.add_argument(
        "-o", "--output", help="Output file path (optional, default: print to console)"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Summarize best group/case tiling data for all results",
    )
    parser.add_argument(
        "--topn", action="store_true", help="Include exported TopN candidate groups"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["console", "csv", "excel"],
        default="console",
        help="Output format: console (default), csv, or excel",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively find all .log files in directory",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print(f"Error: Path '{args.log_path}' not found.")
        return 1

    if os.path.isdir(args.log_path) or args.recursive:
        log_files = find_log_files(args.log_path)
        if not log_files:
            print(f"No .log files found in '{args.log_path}'")
            return 1
        print(f"Found {len(log_files)} log file(s)")
    else:
        log_files = [args.log_path]

    log_parser = LogParser()
    all_summaries = []
    all_final_records = []
    all_topn_records = []

    for log_file in log_files:
        print(f"Parsing: {log_file}")
        try:
            final_records = []
            if os.path.isfile(log_file):
                with open(log_file, "r", encoding="utf-8") as stream:
                    content = stream.read()
                if "[ATT][FINAL_TILING" in content:
                    final_records = log_parser.extract_final_tiling_records(
                        content, log_file
                    )
                if args.topn and "[ATT][TOPN_TILING" in content:
                    all_topn_records.extend(
                        log_parser.extract_topn_tiling_records(content, log_file)
                    )
            if any(record.source != "legacy" for record in final_records):
                all_final_records.extend(
                    record for record in final_records if record.source != "legacy"
                )
            summary_mode = (
                "all_results_all_groups" if args.all else "best_result_all_groups"
            )
            summaries = log_parser.parse_log_file(log_file, summary_mode=summary_mode)
            all_summaries.extend(summaries)
        except Exception as e:
            print(f"Warning: Failed to parse {log_file}: {e}")

    print(f"Total operators extracted: {len(all_summaries)}")

    if not all_summaries and not all_final_records and not all_topn_records:
        print("No data to export.")
        return 1

    if args.format == "console":
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                original_stdout = sys.stdout
                sys.stdout = f
                print_summary_table(all_summaries)
                print_tiling_table(all_final_records)
                print_tiling_table(all_topn_records, topn=True)
                sys.stdout = original_stdout
            print(f"Summary table has been written to '{args.output}'")
        else:
            print_summary_table(all_summaries)
            print_tiling_table(all_final_records)
            print_tiling_table(all_topn_records, topn=True)
    elif args.format == "csv":
        output_file = args.output if args.output else "output.csv"
        export_to_csv(
            all_summaries,
            output_file,
            final_records=all_final_records,
            topn_records=all_topn_records,
        )
        print(f"Data exported to {output_file}")
    elif args.format == "excel":
        output_file = args.output if args.output else "output.xlsx"
        export_to_excel(
            all_summaries,
            output_file,
            final_records=all_final_records,
            topn_records=all_topn_records,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
