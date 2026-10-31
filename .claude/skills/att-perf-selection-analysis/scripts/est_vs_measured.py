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

"""Compare run-local ATT final pipe estimates with uniquely associated AIV profiles.

Candidate observations remain separate: they cannot prove dispatch or supply the
estimate for a final tiling with potentially different shape/key/subcase values.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "autofuse/tools/att_analyze/src"))
from core.log_parser import LogParser  # noqa: E402


PIPES = {
    "AIV_MTE2": ("M2", "aiv_mte2_ratio"),
    "AIV_MTE3": ("M3", "aiv_mte3_ratio"),
    "AIV_VEC": ("V", "aiv_vec_ratio"),
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# FINAL_TILING parsing (including framing, hashes and invalid statuses) belongs
# to LogParser. Only candidate observations missing from its public API are
# scanned here; raw R labels and every source location are retained.
CANDIDATE = re.compile(
    rf"\[([^\[\]]+)\]\s*\[PROF\](?:The value of "
    rf"(?P<pipe>AIV_MTE2|AIV_MTE3|AIV_VEC) is|"
    rf"The objective value of the tiling data is)\s+(?P<value>{NUMBER})\s+"
    r"in graph(?P<graph>\d+)_result(?P<result>\d+)_g(?P<group>-?\d+)_(?P<case>R?\d+)(?!\w)"
)
FIELDS = [
    "Run",
    "RecordKind",
    "Operator",
    "Graph",
    "Result",
    "Group",
    "Case",
    "CaseLabel",
    "TilingKey",
    "SubCaseTag",
    "Template",
    "FinalSource",
    "SelectionMode",
    "ParseStatus",
    "Pipe",
    "Estimate",
    "Measured",
    "Ratio",
    "Status",
    "LogEvidence",
    "FinalSummaryEvidence",
    "ProfileEvidence",
]


def log_files(path):
    path = Path(path).resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix in (".log", ".slog") and p.is_file()
        )
        if files:
            return files
    raise ValueError(f"未找到日志: {path}")


def read_logs(path, run, filters):
    candidates, finals, summaries = [], [], []
    parser = LogParser()
    for log in log_files(path):
        content = log.read_text(encoding="utf-8")
        finals.extend(parser.extract_final_tiling_records(content, str(log)))
        summaries.extend(parser.extract_final_tiling_summaries(content, str(log)))
        for line_no, line in enumerate(content.splitlines(), 1):
            match = CANDIDATE.search(line)
            if not match:
                continue
            op = match.group(1)
            if filters and not any(f in op for f in filters):
                continue
            case = match["case"]
            candidates.append(
                {
                    "Run": run,
                    "RecordKind": "candidate",
                    "Operator": op,
                    "Graph": int(match["graph"]),
                    "Result": int(match["result"]),
                    "Group": int(match["group"]),
                    "Case": int(case.lstrip("R")),
                    "CaseLabel": case,
                    "Pipe": match["pipe"] or "Objective",
                    "Estimate": float(match["value"]),
                    "Status": "candidate_only"
                    if math.isfinite(float(match["value"]))
                    else "invalid_candidate_estimate",
                    "LogEvidence": f"{log}:{line_no}",
                }
            )
    return (
        candidates,
        [f for f in finals if not filters or any(s in (f.op or "") for s in filters)],
        summaries,
    )


def read_profile(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Op Name", "Block Num", "aiv_total_cycles"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path}: 缺少列 {sorted(required - set(reader.fieldnames or []))}"
            )
        for row in reader:
            # line_num is the ending physical line, including quoted multiline cells.
            rows.append((row, f"{Path(path).resolve()}:{reader.line_num}"))
    return rows


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def profile_match(op, rows, mapping):
    if not op:
        return None, "missing_profile"
    if op in mapping:
        hits = [
            (row, anchor) for row, anchor in rows if row.get("Op Name") == mapping[op]
        ]
    else:
        hits = [
            (row, anchor)
            for row, anchor in rows
            if row.get("Op Name") and normalized(row["Op Name"]) == normalized(op)
        ]
    if len(hits) != 1:
        return None, "ambiguous_profile" if hits else "missing_profile"
    return hits[0], "ok"


def number(value):
    if isinstance(value, bool):
        raise ValueError("需要数值而非布尔值")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("需要有限非负数")
    return result


def measured_pipe(row, ratio_column):
    """Ratios accept fractions or explicit percentages, never implicit percent units."""
    total = number(row.get("aiv_total_cycles"))
    blocks = number(row.get("Block Num"))
    raw_ratio = str(row.get(ratio_column) or "").strip()
    ratio = number(raw_ratio.rstrip("%")) / (100 if raw_ratio.endswith("%") else 1)
    if total <= 0 or blocks <= 0 or not blocks.is_integer() or ratio > 1:
        raise ValueError("AIV cycle/block/ratio 口径无效")
    return total * ratio / blocks


def summary_status(final, summaries):
    if not summaries:
        return "ok"
    if len(summaries) != 1:
        return "ambiguous_final"
    summary = summaries[0]
    if summary.parse_status != "ok":
        return summary.parse_status
    if len(summary.groups) != 1:
        return "ambiguous_final"
    group = summary.groups.get(str(final.group))
    if (
        (summary.graph, summary.result, summary.source, summary.selection_mode)
        != (final.graph, final.result, final.source, final.selection_mode)
        or not isinstance(group, dict)
        or any(
            group.get(k) != v
            for k, v in {
                "case_id": final.case,
                "tiling_key": final.tiling_key,
            }.items()
        )
        or any(
            k in group and group[k] != v
            for k, v in {
                "sub_case_tag": final.sub_case_tag,
                "template": final.template_name,
            }.items()
        )
    ):
        return "conflicting_final_summary"
    return "ok"


def build_rows(plog, profile, run, filters, mapping):
    candidates, finals, summaries = read_logs(plog, run, filters)
    profiles = read_profile(profile)
    counts = Counter(f.op for f in finals)
    # Prevent two log operators from consuming the same physical profiling row.
    matches = {op: profile_match(op, profiles, mapping) for op in counts}
    owners = Counter(hit[1] for hit, status in matches.values() if status == "ok")
    rows = list(candidates)
    for final in finals:
        hit, status = matches[final.op]
        op_summaries = [s for s in summaries if s.op == final.op]
        result_status = summary_status(final, op_summaries)
        if final.parse_status != "ok":
            status = final.parse_status
        elif (final.source, final.selection_mode) not in {
            ("runtime", "default"),
            ("runtime", "explicit"),
            ("pgo", "pgo"),
        }:
            status = "unsupported_final_source"
        elif counts[final.op] != 1:
            # A whole kernel profile cannot be attributed to individual groups,
            # subcases or repeated invocations without further call-level evidence.
            status = "ambiguous_final"
        elif result_status != "ok":
            status = result_status
        elif status == "ok" and owners[hit[1]] != 1:
            status = "ambiguous_profile"
        for pipe, (alias, ratio_col) in PIPES.items():
            estimates = final.pipe_est if isinstance(final.pipe_est, dict) else {}
            estimate = estimates.get(pipe, estimates.get(alias))
            row = {
                "Run": run,
                "RecordKind": "final",
                "Operator": final.op,
                "Graph": final.graph,
                "Result": final.result,
                "Group": final.group,
                "Case": final.case,
                "TilingKey": final.tiling_key,
                "SubCaseTag": final.sub_case_tag,
                "Template": final.template_name,
                "FinalSource": final.source,
                "SelectionMode": final.selection_mode,
                "ParseStatus": final.parse_status,
                "Pipe": pipe,
                "Estimate": estimate,
                "Status": status,
                "LogEvidence": f"{final.source_path}:{final.source_line}",
                "FinalSummaryEvidence": ";".join(
                    f"{s.source_path}:{s.source_line}" for s in op_summaries
                ),
            }
            if status == "ok":
                row["ProfileEvidence"] = hit[1]
                try:
                    measured = measured_pipe(hit[0], ratio_col)
                except (TypeError, ValueError):
                    row["Status"] = "invalid_or_missing_metric"
                else:
                    row["Measured"] = measured
                    if estimate is None:
                        row["Status"] = "missing_estimate"
                    else:
                        try:
                            estimate = number(estimate)
                        except (TypeError, ValueError):
                            row["Status"] = "invalid_estimate"
                        else:
                            if measured == 0:
                                row["Status"] = "zero_measured"
                            else:
                                row["Ratio"] = estimate / measured
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plog", action="append", required=True, help="每 run 的独立日志文件或目录"
    )
    parser.add_argument(
        "--op-summary",
        action="append",
        required=True,
        help="与第 N 个 --plog 对应的 CSV",
    )
    parser.add_argument(
        "--op-filter",
        action="append",
        default=[],
        help="按日志算子名子串筛选（区分大小写；可重复，任一命中）",
    )
    parser.add_argument(
        "--op-map",
        help="可选 JSON 对象：日志算子名到 profiling 完整 Op Name 的显式映射",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="新的 CSV 文件路径，不覆盖已有文件"
    )
    args = parser.parse_args(argv)
    if len(args.plog) != len(args.op_summary):
        parser.error("--plog 与 --op-summary 必须按顺序一一成对")
    try:
        mapping = (
            json.loads(Path(args.op_map).read_text(encoding="utf-8"))
            if args.op_map
            else {}
        )
        if not isinstance(mapping, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in mapping.items()
        ):
            raise ValueError("--op-map 必须为非空算子名到非空 Op Name 的 JSON 对象")
        rows = []
        for run, (plog, profile) in enumerate(zip(args.plog, args.op_summary), 1):
            rows.extend(build_rows(plog, profile, run, args.op_filter, mapping))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        print(f"[est-vs-measured] {exc}", file=sys.stderr)
        return 2
    comparable = sum("Ratio" in row for row in rows)
    print(
        f"[est-vs-measured] {len(args.plog)} run, {len(rows)} 行, {comparable} 项可比较: {output}"
    )
    if not comparable:
        print(
            "[est-vs-measured] 证据不足以比较估值与实测；请检查 Status 列。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
