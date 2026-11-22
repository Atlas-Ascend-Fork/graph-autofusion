#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


BASE_FIELDS = [
    "Op Name",
    "OP Type",
    "Task Type",
    "Task Duration(us)",
    "Task Start Time(us)",
    "Task Wait Time(us)",
    "Task ID",
    "Stream ID",
    "Block Dim",
    "aicore_time(us)",
    "aiv_time(us)",
    "total_cycles",
    "aic_total_cycles",
    "aiv_total_cycles",
    "memory_bound",
    "cube_utilization(%)",
]

FIELD_ALIASES = {
    "Block Dim": [
        "Block Dim",
        "BlockDim",
        "block_dim",
        "blockdim",
        "Block Num",
        "BlockNum",
        "block_num",
        "blocknum",
    ],
}

PIPE_ALIASES = {
    "mte2": [
        "mte2_exe_time(us)",
        "mte2_exe_ratio",
        "aic_mte2_time(us)",
        "aic_mte2_ratio",
        "aiv_mte2_time(us)",
        "aiv_mte2_ratio",
    ],
    "mte3": [
        "mte3_exe_time(us)",
        "mte3_exe_ratio",
        "aic_mte3_time(us)",
        "aic_mte3_ratio",
        "aiv_mte3_time(us)",
        "aiv_mte3_ratio",
    ],
    "vector": [
        "vec_exe_time(us)",
        "vec_exe_ratio",
        "aic_vec_time(us)",
        "aic_vec_ratio",
        "aiv_vec_time(us)",
        "aiv_vec_ratio",
    ],
}

PIPE_RATIO_FIELDS = {
    "mte2": ["aiv_mte2_ratio", "mte2_exe_ratio", "aic_mte2_ratio"],
    "mte3": ["mte3_exe_ratio", "aiv_mte3_ratio", "aic_mte3_ratio"],
    "vector": ["vec_exe_ratio", "aiv_vec_ratio", "aic_vec_ratio"],
}

PIPE_TIME_RATIO_FIELDS = {
    "mte2": [
        ("aiv_mte2_time(us)", "aiv_time(us)"),
        ("mte2_exe_time(us)", "aiv_time(us)"),
        ("mte2_exe_time(us)", "Task Duration(us)"),
    ],
    "mte3": [
        ("aiv_mte3_time(us)", "aiv_time(us)"),
        ("aic_mte3_time(us)", "aicore_time(us)"),
        ("mte3_exe_time(us)", "Task Duration(us)"),
    ],
    "vector": [
        ("aiv_vec_time(us)", "aiv_time(us)"),
        ("aic_vec_time(us)", "aicore_time(us)"),
        ("vec_exe_time(us)", "Task Duration(us)"),
    ],
}

AGGREGATE_CLEAR_FIELDS = {
    "execution_repeat_index",
    "experiment_order",
    "experiment_total",
    "profiling_parse_index",
    "profiling_source_row",
    "profiling_match_index",
}

AGGREGATE_FIRST_FIELDS = {
    "experiment_idx",
    "token",
    "round_id",
    "sample_id",
    "execution_repeat_count",
    "sensitivity_id",
    "sensitivity_key",
    "sensitivity_name",
    "stage_define",
    "group_id",
    "group_name",
    "model_target",
    "metric_target",
    "scan_variable",
    "controlled_variables",
    "dtype",
    "dtype_size",
    "dim",
    "block_dim",
    "repeat",
    "output_dims",
    "output_stride",
    "input_stride",
    "input_stride_axis",
    "input_stride_multiplier",
    "output_stride_axis",
    "output_stride_multiplier",
    "src_offset_elem",
    "dst_offset_elem",
    "src_align_mod32",
    "dst_align_mod32",
    "inner_elems",
    "inner_bytes",
    "total_elems",
    "total_bytes",
    "bytes_per_core",
    "logical_total_bytes",
    "model_x_bytes_per_core",
    "gm_span_elems",
    "ub_span_elems",
    "input_stride_pattern",
    "output_stride_pattern",
    "layout_pattern",
    "notes",
    "fit_role",
    "bytes_region",
    "shape_policy",
    "model_family",
    "source_file",
    "profiling_match_strategy",
    "block_dim_field",
    "profiler_block_dim",
    "profiler_block_dim_field",
    "nddma_metric_field",
}


def normalize_field_name(field_name: str) -> str:
    return "".join(ch.lower() for ch in str(field_name).strip() if ch.isalnum())


def build_normalized_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in raw_row.items():
        normalized[normalize_field_name(key)] = value
    return normalized


def get_with_aliases(
    raw_row: Dict[str, Any],
    normalized_row: Dict[str, Any],
    field: str,
    default: str = "N/A",
) -> Any:
    candidates = FIELD_ALIASES.get(field, [field])
    for candidate in candidates:
        if candidate in raw_row and raw_row[candidate] not in ("", None):
            return raw_row[candidate]
        normalized_key = normalize_field_name(candidate)
        if normalized_key in normalized_row and normalized_row[normalized_key] not in (
            "",
            None,
        ):
            return normalized_row[normalized_key]
    normalized_field = normalize_field_name(field)
    if normalized_field in normalized_row and normalized_row[normalized_field] not in (
        "",
        None,
    ):
        return normalized_row[normalized_field]
    return default


def resolve_profiler_block_dim(entry: Dict[str, Any]) -> tuple[float, str]:
    candidates = [
        "Block Dim",
        "BlockDim",
        "Block Num",
        "BlockNum",
        "block_num",
        "blocknum",
    ]
    for field in candidates:
        value = try_numeric(entry.get(field))
        if value is not None:
            return value, field
    raise ValueError("missing numeric field from Block Dim aliases")


def resolve_block_dim(entry: Dict[str, Any]) -> tuple[float, str]:
    for field in ["block_dim", "factor_block_dim"]:
        value = try_numeric(entry.get(field))
        if value is not None:
            return value, field
    return resolve_profiler_block_dim(entry)


def resolve_model_x_bytes_per_core(entry: Dict[str, Any], block_dim: float) -> str:
    logical_total_bytes = try_numeric(entry.get("logical_total_bytes"))
    bytes_per_core = try_numeric(entry.get("bytes_per_core"))
    if logical_total_bytes is not None and block_dim > 0.0:
        return f"{logical_total_bytes / block_dim:.12g}"
    if bytes_per_core is not None:
        return f"{bytes_per_core:.12g}"
    total_bytes = try_numeric(entry.get("total_bytes"))
    if total_bytes is not None:
        return f"{total_bytes:.12g}"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze NDDMA msprof op_summary and merge experiment params."
    )
    parser.add_argument("--profiling-dir", required=True)
    parser.add_argument("--experiment-log", required=True)
    parser.add_argument("--factor-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--analysis-name",
        default="analysis_result",
        help="Basename of the summary JSON, written as <analysis-name>.<analysis-suffix>.",
    )
    parser.add_argument(
        "--analysis-suffix",
        default="json",
        help="Suffix of the summary JSON.",
    )
    parser.add_argument("--pipes", default="mte2")
    parser.add_argument(
        "--op-name-filter",
        default="nddma",
        help="Case-insensitive op-name substring. Empty value keeps all op_summary rows.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow positional matching when profiling and experiment row counts differ.",
    )
    return parser.parse_args()


def parse_numeric(value: Any) -> float:
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        raise ValueError("missing numeric value")
    if text.endswith("%"):
        text = text[:-1].strip()
    return float(text)


def try_numeric(value: Any) -> float | None:
    try:
        return parse_numeric(value)
    except (TypeError, ValueError):
        return None


def normalize_ratio(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def first_numeric(entry: Dict[str, Any], fields: Sequence[str]) -> tuple[str, float]:
    for field in fields:
        value = try_numeric(entry.get(field))
        if value is not None:
            return field, value
    raise ValueError(f"missing numeric field from {fields}")


def resolve_total_cycles(entry: Dict[str, Any], pipe: str) -> tuple[str, float]:
    task_type = str(entry.get("Task Type", "")).upper()
    if pipe == "mte2":
        return first_numeric(entry, ["aiv_total_cycles"])
    if pipe in {"mte3", "vector"}:
        if task_type == "AI_VECTOR_CORE":
            return first_numeric(entry, ["aiv_total_cycles", "total_cycles"])
        if task_type == "AI_CORE":
            return first_numeric(entry, ["aic_total_cycles", "total_cycles"])
    return first_numeric(
        entry, ["total_cycles", "aiv_total_cycles", "aic_total_cycles"]
    )


def resolve_pipe_ratio(entry: Dict[str, Any], pipe: str) -> tuple[str, float]:
    zero_candidate: tuple[str, float] | None = None
    for field in PIPE_RATIO_FIELDS.get(pipe, []):
        raw_ratio = try_numeric(entry.get(field))
        if raw_ratio is None:
            continue
        ratio = normalize_ratio(raw_ratio)
        if ratio > 0.0:
            return field, ratio
        zero_candidate = (field, ratio)

    for pipe_time_field, total_time_field in PIPE_TIME_RATIO_FIELDS.get(pipe, []):
        pipe_time = try_numeric(entry.get(pipe_time_field))
        total_time = try_numeric(entry.get(total_time_field))
        if pipe_time is None or total_time is None or total_time <= 0.0:
            continue
        ratio = pipe_time / total_time
        if ratio > 0.0:
            return f"{pipe_time_field}/{total_time_field}", ratio

    if zero_candidate is not None:
        return zero_candidate
    raise ValueError(f"missing numeric ratio field for pipe={pipe}")


def profiling_order_key(entry: Dict[str, Any]) -> tuple[int, float | str, int]:
    start_time = try_numeric(entry.get("Task Start Time(us)"))
    parse_index = int(entry.get("profiling_parse_index", 0) or 0)
    if start_time is not None:
        return (0, start_time, parse_index)
    return (1, str(entry.get("source_file", "")), parse_index)


def op_summary_files(profiling_dir: Path) -> List[Path]:
    def is_op_summary(path: Path) -> bool:
        name = path.name.lower()
        return path.suffix.lower() == ".csv" and (
            name == "op_summary.csv" or name.startswith("op_summary_")
        )

    return sorted(
        (path for path in profiling_dir.rglob("*.csv") if is_op_summary(path)),
        key=lambda path: path.relative_to(profiling_dir).as_posix(),
    )


def parse_op_summary(
    profiling_dir: Path, pipes: Sequence[str], op_name_filter: str
) -> List[Dict[str, Any]]:
    op_summary_files_list = op_summary_files(profiling_dir)
    rows: List[Dict[str, Any]] = []
    lowered_filter = op_name_filter.lower()
    parse_index = 0
    for csv_path in op_summary_files_list:
        with open(csv_path, newline="", encoding="utf-8-sig") as file_obj:
            reader = csv.DictReader(file_obj)
            for raw_row in reader:
                normalized_row = build_normalized_row(raw_row)
                op_name = raw_row.get("Op Name", "")
                if lowered_filter and lowered_filter not in op_name.lower():
                    continue
                parse_index += 1
                entry: Dict[str, Any] = {"source_file": str(csv_path)}
                entry["profiling_parse_index"] = str(parse_index)
                entry["profiling_source_row"] = str(reader.line_num)
                for field in BASE_FIELDS:
                    entry[field] = get_with_aliases(
                        raw_row, normalized_row, field, "N/A"
                    )
                for pipe in pipes:
                    for field in PIPE_ALIASES.get(pipe, []):
                        entry[field] = get_with_aliases(
                            raw_row, normalized_row, field, "N/A"
                        )
                rows.append(entry)
    return sorted(rows, key=profiling_order_key)


def sample_op_names(profiling_dir: Path, limit: int = 20) -> List[str]:
    names: List[str] = []
    seen = set()
    for csv_path in op_summary_files(profiling_dir):
        with open(csv_path, newline="", encoding="utf-8-sig") as file_obj:
            reader = csv.DictReader(file_obj)
            for raw_row in reader:
                name = str(raw_row.get("Op Name", "")).strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
                    if len(names) >= limit:
                        return names
    return names


def parse_experiment_log(experiment_log: Path) -> List[Dict[str, str]]:
    line_pattern = re.compile(r"\[(\d+)/(\d+)\]\s+(\S+)")
    rows: List[Dict[str, str]] = []
    with open(experiment_log, encoding="utf-8") as file_obj:
        for line in file_obj:
            match = line_pattern.search(line)
            if not match:
                continue
            token = match.group(3)
            repeat_match = re.search(r"_exec(\d+)$", token)
            rows.append(
                {
                    "experiment_order": match.group(1),
                    "experiment_total": match.group(2),
                    "token": token,
                    "execution_repeat_index": repeat_match.group(1)
                    if repeat_match
                    else "",
                }
            )
    return rows


def read_factor_rows(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as file_obj:
        return {row["token"]: row for row in csv.DictReader(file_obj)}


def lookup_factor_row(
    factor_lookup: Dict[str, Dict[str, str]], token: str
) -> Dict[str, str]:
    if token in factor_lookup:
        return factor_lookup[token]
    base_token = re.sub(r"_exec\d+$", "", token)
    return factor_lookup.get(base_token, {})


def base_factor_token(token: str) -> str:
    return re.sub(r"_exec\d+$", "", token)


def read_factor_row_list(path: Path | None) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def summarize_count_mismatch(
    profiling_rows: Sequence[Dict[str, Any]],
    experiment_rows: Sequence[Dict[str, str]],
    factor_rows: Sequence[Dict[str, str]],
    factor_csv: Path,
) -> str:
    details = [
        f"profiling_rows={len(profiling_rows)}",
        f"experiment_rows={len(experiment_rows)}",
        f"factor_rows={len(factor_rows)}",
        f"factor_csv={factor_csv}",
    ]
    experiment_tokens = [
        base_factor_token(row.get("token", ""))
        for row in experiment_rows
        if row.get("token")
    ]
    factor_tokens = [row.get("token", "") for row in factor_rows if row.get("token")]
    factor_token_set = set(factor_tokens)
    experiment_token_set = set(experiment_tokens)
    missing_factor_tokens = sorted(
        {token for token in experiment_tokens if token not in factor_token_set}
    )
    extra_factor_tokens = [
        token for token in factor_tokens if token not in experiment_token_set
    ]
    if missing_factor_tokens:
        details.append(
            "experiment base tokens missing from factor csv: "
            + ", ".join(missing_factor_tokens[:5])
        )
    if extra_factor_tokens:
        details.append(
            "factor csv tokens not present in experiment log: "
            + ", ".join(extra_factor_tokens[:5])
        )
    if len(factor_rows) > len(experiment_rows):
        details.append(
            "factor csv has more rows than experiment log; this usually means you are "
            "using a newer generated template to analyze an older run."
        )
    if len(experiment_rows) > len(profiling_rows):
        details.append(
            "experiment log has more rows than profiling output; some tasks may not have "
            "been captured in op_summary."
        )
    if len(profiling_rows) > len(experiment_rows):
        details.append(
            "profiling output has more rows than experiment log; the op-name filter may be too broad "
            "or the experiment log may be incomplete."
        )
    return " ".join(details)


def unmatched_experiment_tokens(
    profiling_rows: Sequence[Dict[str, Any]],
    experiment_rows: Sequence[Dict[str, str]],
) -> List[str]:
    count = min(len(profiling_rows), len(experiment_rows))
    return [row.get("token", "") for row in experiment_rows[count:] if row.get("token")]


def enrich_pipe_metrics(entry: Dict[str, Any], pipes: Sequence[str]) -> Dict[str, Any]:
    enriched = dict(entry)
    try:
        profiler_block_dim, profiler_block_dim_field = resolve_profiler_block_dim(
            enriched
        )
        if profiler_block_dim > 0.0:
            enriched["profiler_block_dim"] = f"{profiler_block_dim:.12g}"
            enriched["profiler_block_dim_field"] = profiler_block_dim_field
    except ValueError:
        enriched["profiler_block_dim"] = ""
        enriched["profiler_block_dim_field"] = ""
    try:
        block_dim, block_dim_field = resolve_block_dim(enriched)
        if block_dim <= 0.0:
            block_dim = 1.0
        enriched["block_dim_field"] = block_dim_field
        enriched["block_dim"] = f"{block_dim:.12g}"
    except ValueError:
        block_dim = 1.0
        enriched["block_dim_field"] = ""
        enriched["block_dim"] = ""
    enriched["model_x_bytes_per_core"] = resolve_model_x_bytes_per_core(
        enriched, block_dim
    )
    for pipe in pipes:
        try:
            total_cycles_field, total_cycles = resolve_total_cycles(enriched, pipe)
            ratio_field, ratio = resolve_pipe_ratio(enriched, pipe)
            cycles = total_cycles * ratio
            enriched[f"{pipe}_total_cycles_field"] = total_cycles_field
            enriched[f"{pipe}_ratio_field"] = ratio_field
            enriched[f"{pipe}_ratio_normalized"] = f"{ratio:.12g}"
            enriched[f"{pipe}_cycles"] = f"{cycles:.12g}"
            enriched[f"{pipe}_cycles_per_block"] = f"{cycles / block_dim:.12g}"
        except ValueError:
            enriched[f"{pipe}_total_cycles_field"] = ""
            enriched[f"{pipe}_ratio_field"] = ""
            enriched[f"{pipe}_ratio_normalized"] = "N/A"
            enriched[f"{pipe}_cycles"] = "N/A"
            enriched[f"{pipe}_cycles_per_block"] = "N/A"

    enriched["nddma_metric_field"] = "nddma_mte2_cycles_per_block"
    enriched["nddma_mte2_cycles"] = enriched.get("mte2_cycles", "N/A")
    enriched["nddma_mte2_cycles_per_block"] = enriched.get(
        "mte2_cycles_per_block", "N/A"
    )
    return enriched


def match_with_params(
    profiling_rows: Sequence[Dict[str, Any]],
    experiment_rows: Sequence[Dict[str, str]],
    factor_lookup: Dict[str, Dict[str, str]],
    pipes: Sequence[str],
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    count = min(len(profiling_rows), len(experiment_rows))
    for index in range(count):
        exp = experiment_rows[index]
        factor = lookup_factor_row(factor_lookup, exp["token"])
        merged: Dict[str, Any] = {}
        merged.update(factor)
        merged.update(exp)
        if exp.get("execution_repeat_index"):
            merged["execution_repeat_index"] = exp["execution_repeat_index"]
        embedded_factor = {
            key: profiling_rows[index].get(key, "")
            for key in factor.keys()
            if profiling_rows[index].get(key, "") not in ("", None)
        }
        merged.update(embedded_factor)
        merged.update(profiling_rows[index])
        merged["profiling_match_index"] = str(index + 1)
        merged["profiling_match_strategy"] = "positional_after_task_start_time_sort"
        matched.append(enrich_pipe_metrics(merged, pipes))
    return matched


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_value(value: float) -> str:
    return f"{value:.12g}"


def aggregation_key(row: Dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return base_factor_token(str(row.get("token", "")).strip())


def aggregate_duplicate_samples(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for row in rows:
        key = aggregation_key(row)
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)

    aggregated: List[Dict[str, Any]] = []
    for key in order:
        group_rows = grouped[key]
        first = group_rows[0]
        output: Dict[str, Any] = {}
        seen_fields: List[str] = []
        seen = set()
        for row in group_rows:
            for field in row:
                if field not in seen:
                    seen.add(field)
                    seen_fields.append(field)

        for field in seen_fields:
            if field in AGGREGATE_CLEAR_FIELDS:
                output[field] = ""
                continue
            if field in AGGREGATE_FIRST_FIELDS:
                output[field] = first.get(field, "")
                continue

            values = [row.get(field, "") for row in group_rows]
            numeric = [try_numeric(value) for value in values]
            if all(value is not None for value in numeric):
                output[field] = format_mean_value(
                    mean(value for value in numeric if value is not None)
                )
            else:
                output[field] = first.get(field, "")

        output["sample_id"] = key
        output["token"] = base_factor_token(str(first.get("token", "")))
        output["profiling_mean_sample_count"] = str(len(group_rows))
        output["profiling_mean_metric_count"] = str(
            len(numeric_values(group_rows, "nddma_mte2_cycles"))
        )
        aggregated.append(output)
    return aggregated


def numeric_values(rows: Iterable[Dict[str, Any]], field: str) -> List[float]:
    return [value for row in rows if (value := try_numeric(row.get(field))) is not None]


def summarize_by(
    rows: Sequence[Dict[str, Any]], key: str, metric: str
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    result: Dict[str, Dict[str, Any]] = {}
    for group_value, group_rows in grouped.items():
        values = numeric_values(group_rows, metric)
        result[group_value] = {
            "count": len(group_rows),
            "metric_count": len(values),
            "mean": mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return result


def zero_metric_summary(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    metric_rows = [row for row in rows if try_numeric(row.get(metric)) is not None]
    zero_rows = [row for row in metric_rows if try_numeric(row.get(metric)) == 0.0]
    by_sensitivity_key: Dict[str, int] = defaultdict(int)
    by_group_id: Dict[str, int] = defaultdict(int)
    by_ratio_field: Dict[str, int] = defaultdict(int)
    examples: List[Dict[str, Any]] = []
    for row in zero_rows:
        by_sensitivity_key[str(row.get("sensitivity_key", ""))] += 1
        by_group_id[str(row.get("group_id", ""))] += 1
        by_ratio_field[str(row.get("mte2_ratio_field", ""))] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "token": row.get("token", ""),
                    "sample_id": row.get("sample_id", ""),
                    "sensitivity_key": row.get("sensitivity_key", ""),
                    "dtype": row.get("dtype", ""),
                    "block_dim": row.get("block_dim", ""),
                    metric: row.get(metric, ""),
                    "aiv_total_cycles": row.get("aiv_total_cycles", ""),
                    "mte2_ratio_field": row.get("mte2_ratio_field", ""),
                    "mte2_ratio_normalized": row.get("mte2_ratio_normalized", ""),
                    "aiv_mte2_ratio": row.get("aiv_mte2_ratio", ""),
                    "mte2_exe_ratio": row.get("mte2_exe_ratio", ""),
                    "aiv_mte2_time(us)": row.get("aiv_mte2_time(us)", ""),
                    "mte2_exe_time(us)": row.get("mte2_exe_time(us)", ""),
                }
            )
    return {
        "metric": metric,
        "metric_row_count": len(metric_rows),
        "zero_count": len(zero_rows),
        "zero_fraction": (len(zero_rows) / len(metric_rows)) if metric_rows else 0.0,
        "by_sensitivity_key": dict(by_sensitivity_key),
        "by_group_id": dict(by_group_id),
        "by_mte2_ratio_field": dict(by_ratio_field),
        "examples": examples,
    }


def main() -> int:
    args = parse_args()
    profiling_dir = Path(args.profiling_dir).resolve()
    experiment_log = Path(args.experiment_log).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipes = [pipe.strip() for pipe in args.pipes.split(",") if pipe.strip()]
    factor_csv = (
        Path(args.factor_csv).resolve()
        if args.factor_csv
        else experiment_log.parent.parent / f"{experiment_log.parent.parent.name}.csv"
    )

    profiling_rows = parse_op_summary(profiling_dir, pipes, args.op_name_filter)
    if not profiling_rows:
        print(f"[ERROR] no matched op_summary rows under {profiling_dir}")
        names = sample_op_names(profiling_dir)
        if names:
            print(
                f"[ERROR] op-name filter was {args.op_name_filter!r}; sample Op Name values: {', '.join(names)}"
            )
            print(
                "[ERROR] rerun with --op-name-filter <one of the above substrings>, or --op-name-filter '' to keep all rows."
            )
        return 1

    experiment_rows = parse_experiment_log(experiment_log)
    if not experiment_rows:
        print(f"[ERROR] no experiment rows parsed from {experiment_log}")
        return 1
    factor_rows = read_factor_row_list(factor_csv)
    count_mismatch = len(profiling_rows) != len(experiment_rows)
    mismatch_details = ""
    missing_tokens: List[str] = []
    if count_mismatch:
        if factor_rows:
            mismatch_details = summarize_count_mismatch(
                profiling_rows, experiment_rows, factor_rows, factor_csv
            )
        else:
            mismatch_details = f"factor_csv not found or empty: {factor_csv}"
        missing_tokens = unmatched_experiment_tokens(profiling_rows, experiment_rows)
        level = "[ERROR]" if not args.allow_count_mismatch else "[WARN]"
        print(
            f"{level} profiling row count does not match experiment row count: "
            f"profiling_rows={len(profiling_rows)} experiment_rows={len(experiment_rows)}."
        )
        print(f"{level} mismatch diagnostics: {mismatch_details}")
        if missing_tokens:
            print(
                f"{level} unmatched experiment tokens: {', '.join(missing_tokens[:10])}"
            )
        if not args.allow_count_mismatch:
            print(
                "[ERROR] profiling and experiment row counts differ, so positional matching can silently mislabel "
                "or truncate later rows. Use --allow-count-mismatch only for manual debugging."
            )
            return 1

    factor_lookup = read_factor_rows(factor_csv)
    matched = match_with_params(profiling_rows, experiment_rows, factor_lookup, pipes)
    if not matched:
        print("[ERROR] no profiling rows matched with experiment rows")
        return 1

    write_csv(output_dir / "profiling_with_params.csv", matched)
    matched_mean = aggregate_duplicate_samples(matched)
    write_csv(output_dir / "profiling_with_params_mean.csv", matched_mean)
    write_csv(
        output_dir / "experiment_params.csv",
        [
            dict(lookup_factor_row(factor_lookup, row["token"]), **row)
            for row in experiment_rows
        ],
    )
    summary = {
        "metric_field": "nddma_mte2_cycles_per_block",
        "pipes": pipes,
        "profiling_row_count": len(profiling_rows),
        "experiment_row_count": len(experiment_rows),
        "matched_row_count": len(matched),
        "matched_mean_row_count": len(matched_mean),
        "match_strategy": "positional_after_task_start_time_sort",
        "count_mismatch_allowed": bool(args.allow_count_mismatch),
        "count_mismatch_detected": count_mismatch,
        "count_mismatch_details": mismatch_details,
        "unmatched_experiment_tokens": missing_tokens,
        "factor_csv": str(factor_csv),
        "by_sensitivity_key": summarize_by(
            matched, "sensitivity_key", "nddma_mte2_cycles"
        ),
        "by_group_id": summarize_by(matched, "group_id", "nddma_mte2_cycles"),
        "mean_by_sensitivity_key": summarize_by(
            matched_mean, "sensitivity_key", "nddma_mte2_cycles"
        ),
        "mean_by_group_id": summarize_by(matched_mean, "group_id", "nddma_mte2_cycles"),
        "zero_metric_summary": zero_metric_summary(matched, "nddma_mte2_cycles"),
        "zero_metric_mean_summary": zero_metric_summary(
            matched_mean, "nddma_mte2_cycles"
        ),
    }
    (output_dir / f"{args.analysis_name}.{args.analysis_suffix}").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[INFO] wrote {output_dir / 'profiling_with_params.csv'}")
    print(f"[INFO] wrote {output_dir / 'profiling_with_params_mean.csv'}")
    print(
        f"[INFO] metric_field=nddma_mte2_cycles_per_block matched_rows={len(matched)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
