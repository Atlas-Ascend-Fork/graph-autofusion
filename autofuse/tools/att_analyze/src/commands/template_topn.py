#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Extract static Top1/TopN template evidence from an Autofuse worker log."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


CSV_FILE = "topn_templates.csv"
MARKDOWN_FILE = "topn_templates.md"

_CSV_FIELDS = [
    "case_id",
    "visit_id",
    "shape_id",
    "operator_name",
    "call_index",
    "association_status",
    "top1_template_names",
    "top1_group_count",
    "top1_name_status",
    "top1_modeled_perf",
    "top1_is_default",
    "topn_candidate_count",
    "topn_candidate_template_names",
    "topn_modeled_perf_values",
    "measured_task_duration_us",
    "performance_source",
]

_TEMPLATE_LINE = re.compile(r"\[ATT\]\[TEMPLATE\]")
_OUTPUT_INDEX = re.compile(r"output\[(\d+)\]")
_OUTPUT_PERF = re.compile(r"output\[(\d+)\]:\s+perf=([0-9.+-]+)\s+is_default=(\d+)")
_VISIT_START = re.compile(r"^\[paired-visit-start\]\s+(\S+)\s+(\d+)\s+(\S+)")
_TOPN_ENTER = re.compile(r"GenerateTopnSolutions enter:.*?operator=(\S+)")
_LOG_EXECUTION = re.compile(r"\b[A-Z_]+\((\d+),[^)]*\):.*?\[[^]\n]+:\d+\](\d+)\s")
_TEMPLATE_OPERATOR = re.compile(r"EmitFinalTilingLines:\[([^\]]+)\]\[ATT\]\[TEMPLATE\]")


def _field(line: str, key: str) -> str:
    match = re.search(rf'{key}=(?:"([^"]*)"|([^\s]+))', line)
    if not match:
        return ""
    return match.group(1) if match.group(1) is not None else match.group(2)


def _template_record(line: str) -> dict | None:
    if not _TEMPLATE_LINE.search(line) or _field(line, "stage") != "topn_output":
        return None
    output_match = _OUTPUT_INDEX.search(line)
    if not output_match:
        return None
    semantic_name = _field(line, "semantic_name")
    has_name = "semantic_name=" in line
    is_core_summary = (
        " fields " not in line
        and " detail " not in line
        and bool(_field(line, "line_hash"))
        and bool(_field(line, "line_len"))
    )
    if not has_name and not is_core_summary and " detail " not in line:
        return None
    record = {
        "output_index": output_match.group(1),
        "group": _field(line, "group"),
        "semantic_index": _field(line, "semantic_index"),
        "graph": _field(line, "graph"),
        "result": _field(line, "result"),
    }
    if has_name or is_core_summary:
        record.update(
            semantic_name=semantic_name if has_name else "Unknown",
            name_status=_field(line, "name_status") if has_name else "unavailable",
        )
    return record


def _unique_names(rows: Iterable[dict]) -> List[str]:
    names: List[str] = []
    for row in rows:
        name = row["semantic_name"]
        if name and name not in names:
            names.append(name)
    return names


def build_rows(log_text: str) -> List[dict]:
    grouped: Dict[tuple, dict] = {}
    context = ("", "", "")
    calls: Dict[tuple, tuple] = {}
    call_counts: Dict[str, int] = {}
    for line in log_text.splitlines():
        visit_match = _VISIT_START.match(line)
        if visit_match:
            context = visit_match.groups()
            calls = {}
            call_counts = {}
            continue
        execution_match = _LOG_EXECUTION.search(line)
        execution = execution_match.groups() if execution_match else ("", "")
        enter_match = _TOPN_ENTER.search(line)
        if enter_match:
            operator_name = enter_match.group(1)
            call_counts[operator_name] = call_counts.get(operator_name, 0) + 1
            calls[execution] = (operator_name, call_counts[operator_name])
            continue
        operator_name, call_index = calls.get(execution, ("", 0))
        confirmed = execution in calls
        record = _template_record(line)
        if record is not None:
            template_operator = _TEMPLATE_OPERATOR.search(line)
            if template_operator and template_operator.group(1) != operator_name:
                operator_name, call_index = template_operator.group(1), 0
                if "semantic_name" in record:
                    calls.pop(execution, None)
                confirmed = False
            key = context + (operator_name, call_index, execution)
            detail_key = (
                record["output_index"],
                record["group"],
                record["semantic_index"],
                template_operator.group(1) if template_operator else "",
            )
            if "semantic_name" not in record:
                pending = grouped.get(key, {}).get("pending", {})
                target = pending.get(detail_key)
                if target is not None:
                    for field in ("graph", "result"):
                        if record[field]:
                            target[field] = record[field]
                    if target["graph"] and target["result"]:
                        pending.pop(detail_key)
                continue
            grouped.setdefault(
                key, {"templates": [], "perf": {}, "association_status": "confirmed"}
            )
            grouped[key]["templates"].append(record)
            grouped[key].setdefault("pending", {})[detail_key] = record
            if not confirmed:
                grouped[key]["association_status"] = "unconfirmed"
            continue
        perf_match = _OUTPUT_PERF.search(line)
        if perf_match and context != ("", "", ""):
            output_index, perf, is_default = perf_match.groups()
            key = context + (operator_name, call_index, execution)
            grouped.setdefault(
                key, {"templates": [], "perf": {}, "association_status": "confirmed"}
            )
            if not confirmed:
                grouped[key]["association_status"] = "unconfirmed"
            grouped[key]["perf"][output_index] = {
                "perf": perf,
                "is_default": is_default,
            }

    rows: List[dict] = []
    for (
        case_id,
        visit_id,
        shape_id,
        operator_name,
        call_index,
        _,
    ), item in grouped.items():
        templates = item["templates"]
        if not templates:
            continue
        templates.sort(
            key=lambda row: (int(row["output_index"]), int(row["group"] or 0))
        )
        top1_rows = [row for row in templates if row["output_index"] == "0"]
        top1_groups = {(row["graph"], row["result"], row["group"]) for row in top1_rows}
        if any(not row["graph"] or not row["result"] for row in top1_rows):
            # Historical or truncated logs only establish a local-group lower bound.
            top1_groups = {(row["group"],) for row in top1_rows}
        top1_perf = item["perf"].get("0", {})
        top1_statuses: List[str] = []
        for row in top1_rows:
            if row["name_status"] and row["name_status"] not in top1_statuses:
                top1_statuses.append(row["name_status"])
        rows.append(
            {
                "case_id": case_id,
                "visit_id": visit_id,
                "shape_id": shape_id,
                "operator_name": operator_name,
                "call_index": str(call_index),
                "association_status": item.get("association_status", "unconfirmed"),
                "top1_template_names": ";".join(_unique_names(top1_rows)),
                "top1_group_count": str(len(top1_groups)),
                "top1_name_status": ";".join(top1_statuses),
                "top1_modeled_perf": top1_perf.get("perf", ""),
                "top1_is_default": top1_perf.get("is_default", ""),
                "topn_candidate_count": str(
                    len({row["output_index"] for row in templates})
                ),
                "topn_candidate_template_names": ";".join(_unique_names(templates)),
                "topn_modeled_perf_values": ";".join(
                    f"{index}:{item['perf'][index]['perf']}"
                    for index in sorted(item["perf"], key=int)
                ),
                "measured_task_duration_us": "",
                "performance_source": "autofuse_log",
            }
        )
    return rows


def _markdown_cell(value: str) -> str:
    if not value:
        return ""
    if ";" in value:
        items = [item.replace("|", "\\|") for item in value.split(";")]
        return "<br>".join(f"`{item}`" for item in items)
    return f"`{value.replace('|', chr(92) + '|')}`"


def _markdown(rows: Sequence[dict]) -> str:
    lines = [
        "### Static Top1 and TopN template matrix",
        "",
        "Top1 is `topn_output output[0]`. The command reads only the Autofuse worker log; "
        "`top1_modeled_perf` is available only when the log contains `output[0]: perf=`. "
        "Measured Task Duration is not inferred from this log and is left empty. "
        "Group counts use graph/result/group when all Top1 identity details are available; "
        "otherwise distinct local group IDs provide only a lower bound.",
        "",
        "| Case | Visit | Operator | Call | Association | Top1 template | Top1 group count | Top1 modeled perf | Top1 default | "
        "TopN count | TopN candidate templates | TopN modeled perf | Measured Task Duration(us) |",
        "|---|---:|---|---:|---|---|---:|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {visit} | {operator} | {call} | {association} | {top1} | {groups} | {perf} | {default} | {count} | "
            "{candidates} | {candidate_perf} | {measured} |".format(
                case=f"`{row['case_id']}`",
                visit=row["visit_id"],
                operator=_markdown_cell(row["operator_name"]),
                call=row["call_index"],
                association=row["association_status"],
                top1=_markdown_cell(row["top1_template_names"]),
                groups=row["top1_group_count"],
                perf=row["top1_modeled_perf"] or "-",
                default=row["top1_is_default"] or "-",
                count=row["topn_candidate_count"],
                candidates=_markdown_cell(row["topn_candidate_template_names"]),
                candidate_perf=_markdown_cell(row["topn_modeled_perf_values"]),
                measured=row["measured_task_duration_us"] or "-",
            )
        )
    return "\n".join(lines) + "\n"


def run(args) -> int:
    log_path = Path(args.log_path)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"topn-templates: {exc}", file=getattr(args, "stderr", sys.stderr))
        return 1

    rows = build_rows(log_text)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILE
    markdown_path = output_dir / MARKDOWN_FILE
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(_markdown(rows), encoding="utf-8")
    print(f"topn-templates: wrote {csv_path} and {markdown_path}")
    return 0
