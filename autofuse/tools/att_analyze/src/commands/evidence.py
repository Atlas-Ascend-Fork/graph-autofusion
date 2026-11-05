#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Export ATT log observations as canonical JSONL evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.evidence_schema import make_record
from core.log_parser import LogParser


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
# ATT output is sometimes prefixed with an additional severity tag, for
# example ``[INFO] [Add] [PROF] ...``.  Consume all bracketed tags and retain
# the tag immediately before ``[PROF]`` as the operator name.  This keeps the
# parser from incorrectly reporting ``INFO`` as the operator.
_PROF_PREFIX = r"(?:\[[^\]]+\]\s*)*\[([^\]]+)\]\s*\[PROF\]"
_MESSAGE_PREFIX = (
    r"(?:\[(?!PROF\])[^\]]+\]\s*)*"
    r"\[(?!PROF\])([^\]]+)\]\s*(?:\[PROF\])?"
)

# Tiling case labels are emitted both as ``_0`` and as ``_R0`` in profiling
# logs.  The latter is a sub-case notation used by the PERF records; both map
# to the same integer case id in the evidence schema.
_KEY = r"graph(\d+)_result(\d+)_g(-?\d+)_(?:R)?(\d+)"
_VALUE_RE = re.compile(
    rf"{_PROF_PREFIX}The value of\s+(\w+)\s+is\s+({_NUMBER})\s+in\s+{_KEY}"
)
_OBJECTIVE_RE = re.compile(
    rf"{_PROF_PREFIX}The objective value of the tiling data is\s+({_NUMBER})\s+in\s+{_KEY}"
)
_TEMPLATE_RE = re.compile(
    rf"{_PROF_PREFIX}Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph(\d+)_result(\d+)_g(-?\d+)(?:_(?:R)?\d+)?\s+is the best choice"
)
_GRAPH_SELECTION_RE = re.compile(
    rf"{_PROF_PREFIX}Among all schedule results,\s*graph(\d+)_result(\d+)\s+is the best choice"
)
_GRAPH_RE = re.compile(
    rf"{_MESSAGE_PREFIX}The value of\s+graph(\d+)_result(\d+)\s+is\s+({_NUMBER})"
)
_PIPE_NAMES = {"M2": "AIV_MTE2", "M3": "AIV_MTE3", "V": "AIV_VEC"}
_PIPE_ORDER = ("AIV_MTE2", "AIV_MTE3", "AIV_VEC")


def _final_record_json(record: Any) -> Dict[str, Any]:
    pipe = record.pipe_est or {}
    pipe_est = (
        {name: pipe.get(name, pipe.get(alias)) for alias, name in _PIPE_NAMES.items()}
        if pipe
        else {name: None for name in _PIPE_ORDER}
    )
    return {
        "schema_version": "att-evidence/v1",
        "operator": record.op,
        "graph_id": record.graph,
        "result_id": record.result,
        "group_id": record.group,
        "case_id": record.case,
        "source": record.source,
        "selection_mode": record.selection_mode,
        "tiling_key": record.tiling_key,
        "score": record.score,
        "sub_case_tag": record.sub_case_tag,
        "template_name": record.template_name,
        "pipe_est": pipe_est,
        "tiling_repr": record.tiling_repr,
        "repr_kind": record.repr_kind,
        "repr_hash": record.repr_hash,
        "parse_status": record.parse_status,
        "source_path": record.source_path,
        "source_line": record.source_line,
    }


def _final_summary_json(summary: Any) -> Dict[str, Any]:
    return {
        "schema_version": "att-evidence/v1",
        "record_type": "final_tiling_summary",
        "operator": summary.op,
        "graph_id": summary.graph,
        "result_id": summary.result,
        "source": summary.source,
        "selection_mode": summary.selection_mode,
        "groups": summary.groups,
        "parse_status": summary.parse_status,
        "source_path": summary.source_path,
        "source_line": summary.source_line,
    }


def _log_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(root, name)
        for root, dirs, files in os.walk(path)
        for name in sorted(files)
        if name.endswith(".log")
    )


def _key(
    operator: str, graph: int, result: int, group: int, case: int
) -> Tuple[str, int, int, int, int]:
    return operator, graph, result, group, case


def _scan_log(path: str) -> List[Dict[str, Any]]:
    """Scan all candidate records while retaining first source locations."""

    records: Dict[Tuple[str, int, int, int, int], Dict[str, Any]] = {}
    graph_perf: Dict[Tuple[str, int, int], float] = {}
    selected: Dict[Tuple[str, int, int, int], int] = {}
    explicit_graph_results = set()
    with open(path, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            match = _VALUE_RE.search(line)
            if match:
                operator, name, value, graph, result, group, case = match.groups()
                key = _key(operator, int(graph), int(result), int(group), int(case))
                item = records.setdefault(
                    key, {"tiling_values": {}, "source_lines": {}}
                )
                item["tiling_values"][name] = float(value)
                item["source_lines"].setdefault(name, line_no)
                item.setdefault("source_line", line_no)
                continue

            match = _OBJECTIVE_RE.search(line)
            if match:
                operator, objective, graph, result, group, case = match.groups()
                key = _key(operator, int(graph), int(result), int(group), int(case))
                item = records.setdefault(
                    key, {"tiling_values": {}, "source_lines": {}}
                )
                item["objective"] = float(objective)
                item["source_lines"].setdefault("objective", line_no)
                item.setdefault("source_line", line_no)
                continue

            match = _TEMPLATE_RE.search(line)
            if match:
                operator, case, graph, result, group = match.groups()
                graph_i, result_i, group_i, case_i = map(
                    int, (graph, result, group, case)
                )
                selected[(operator, graph_i, result_i, group_i)] = case_i
                key = _key(operator, graph_i, result_i, group_i, case_i)
                item = records.setdefault(
                    key, {"tiling_values": {}, "source_lines": {}}
                )
                item["source_lines"].setdefault("template_selection", line_no)
                item.setdefault("source_line", line_no)
                continue

            match = _GRAPH_SELECTION_RE.search(line)
            if match:
                operator, graph, result = match.groups()
                explicit_graph_results.add((operator, int(graph), int(result)))
                continue

            match = _GRAPH_RE.search(line)
            if match:
                operator, graph, result, perf = match.groups()
                graph_perf[(operator, int(graph), int(result))] = float(perf)

    result: List[Dict[str, Any]] = []
    for (operator, graph, schedule_result, group, case), item in records.items():
        selection_key = (operator, graph, schedule_result, group)
        is_selected = selected.get(selection_key) == case
        if is_selected:
            status = (
                "ok"
                if (operator, graph, schedule_result) in explicit_graph_results
                else "inferred_graph_result"
            )
        elif selection_key in selected:
            # This group has a recorded winner, but the current line describes
            # another candidate case.
            status = "candidate"
        else:
            # Tiling/objective lines without a template choice are incomplete
            # evidence; do not silently mark them as complete.
            status = "missing_group_case"
        result.append(
            make_record(
                operator=operator,
                graph_id=graph,
                result_id=schedule_result,
                group_id=group,
                case_id=case,
                tiling_values=item.get("tiling_values"),
                objective=item.get("objective"),
                result_performance=graph_perf.get((operator, graph, schedule_result)),
                source_path=os.path.abspath(path),
                source_line=item.get("source_line"),
                source_lines=item.get("source_lines"),
                parse_status=status,
            )
        )

    # Logs containing only selection lines still need one record per selected
    # group/case.  The scanner above already creates those records; parser
    # output is used solely to expose explicit missing-data statuses.
    parser = LogParser()
    summaries = parser.parse_log_file(path, summary_mode="all_results_all_groups")
    by_key = {
        (s.operator_name, s.graph, s.result, s.group, s.case): s for s in summaries
    }
    for item in result:
        summary = by_key.get(
            (
                item["operator"],
                item["graph_id"],
                item["result_id"],
                item["group_id"],
                item["case_id"],
            )
        )
        if summary and summary.parse_status != "ok":
            current_status = item["parse_status"]
            if current_status == "missing_group_case":
                # Keep the specific missing template diagnosis even when a
                # secondary metric (for example result performance) is absent.
                continue
            if (
                summary.parse_status == "missing_result_performance"
                and item.get("result_performance") is not None
            ):
                # The evidence scanner understands prefixed PROF performance
                # lines that the legacy parser does not.  Do not downgrade a
                # complete record merely because the compatibility parser
                # could not recognize the same metric.
                continue
            if current_status != summary.parse_status:
                item["parse_status"] = f"{current_status};{summary.parse_status}"
    return sorted(
        result,
        key=lambda x: (
            x["source_path"],
            x["operator"],
            x["graph_id"],
            x["result_id"],
            x["group_id"],
            x["case_id"],
        ),
    )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifacts(output_dir: str) -> List[Dict[str, str]]:
    artifacts = []
    for root, dirs, files in os.walk(output_dir):
        dirs.sort()
        for name in sorted(files):
            if name == "tool-manifest.json":
                continue
            path = os.path.join(root, name)
            artifacts.append(
                {"path": os.path.relpath(path, output_dir), "sha256": _sha256(path)}
            )
    return artifacts


def export(
    log_path: str, output_dir: str, command: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    files = _log_files(log_path)
    if not files:
        raise FileNotFoundError(f"No log files found at '{log_path}'")
    os.makedirs(output_dir, exist_ok=True)
    evidence_path = os.path.join(output_dir, "att-evidence.jsonl")
    count = 0
    with open(evidence_path, "w", encoding="utf-8") as stream:
        for path in files:
            with open(path, "r", encoding="utf-8") as input_stream:
                content = input_stream.read()
            final_records = []
            final_summaries = []
            if "[ATT][FINAL_TILING" in content:
                parser = LogParser()
                source_path = os.path.abspath(path)
                final_records = parser.extract_final_tiling_records(
                    content, source_path
                )
                final_summaries = parser.extract_final_tiling_summaries(
                    content, source_path
                )
            records = (
                []
                if any(record.source != "legacy" for record in final_records)
                else _scan_log(path)
            )
            if final_records and any(
                record.source != "legacy" for record in final_records
            ):
                for final_record in final_records:
                    records.append(_final_record_json(final_record))
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                count += 1
            if not records and final_summaries:
                for summary in final_summaries:
                    stream.write(
                        json.dumps(
                            _final_summary_json(summary),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    count += 1

    manifest = {
        "manifest_version": "att-tool/v1",
        "schema_version": "att-evidence/v1",
        "tool": "att_analyze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command or []),
        "inputs": [
            {"path": os.path.abspath(path), "sha256": _sha256(path)} for path in files
        ],
        "record_count": count,
        "artifacts": _artifacts(output_dir),
    }
    manifest_path = os.path.join(output_dir, "tool-manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def run(args: Any) -> None:
    try:
        manifest = export(
            args.log_path, args.output, command=getattr(args, "_argv", None) or sys.argv
        )
    except (OSError, UnicodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return
    print(f"Exported {manifest['record_count']} evidence record(s) to '{args.output}'")
