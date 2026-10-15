#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Summarize TorchAir SuperKernel metadata and prove whether fusion occurred."""

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path


REASONS = re.compile(
    r"\b(NOT_IN_SCOPE|IN_UNFUSIBLE_SCOPE|EXCEED_CORE_MAX|EXCEED_SCOPE_MAX|"
    r"OP_UNSUPPORT|DYNAMIC_TASK_UNSUPPORT|SIMT_OP_UNSUPPORT|UNFUSIBLE_NODE|"
    r"DEADLOCK_DETECTED|SYNCALL_OP_DROP|DEBUG_PER_OP_MAX_CORE|"
    r"RESOURCE_INSUFFICIENT|EXTERNAL_DEPEND|EXIST_DEADLOCK|ISOLATED_EVENT|"
    r"KERNEL_ATTR_GET_FAILED|NO_TARGET_NODE|UNRECOVERABLE_FAIL|SUCCESS)\b"
)
STATIC_KERNEL_OP = re.compile(r"(?:^|_)static_kernel_([A-Za-z0-9]+)_")
SK_START_END = re.compile(
    r"^sk_\d+_(?P<scope>.+?)_start_(?P<start>.+)_end_(?P<end>.+)$"
)
MODEL_DIRECTORY = re.compile(r"^model_(?P<model_id>\d+)(?:_|$)")
LOG_NAMES = {
    "sk_fused_nodes.log",
    "sk_fusion_fail_reasons.log",
    "sk_scope_split.log",
}
EXPECTED_REASONS = {"NOT_IN_SCOPE", "IN_UNFUSIBLE_SCOPE", "SUCCESS"}
REPLAY_MIN_CHILD_NODES = 1
REPLAY_MIN_EFFECTIVE_CHILD_NODES = 5
FUSED_HEADER = re.compile(
    r"SK Function:\s*(?P<name>.*?),\s*scope id:\s*(?P<scope_id>\d+),\s*Node Count:\s*(?P<count>\d+)"
)
FUSED_NODE = re.compile(
    r"nodeId:(?P<node_id>\d+),\s*streamId:(?P<stream_id>\d+).*?"
    r"KernelInfos\{funcName:(?P<func_name>[^,}]+),\s*kernelType:(?P<kernel_type>[^,}]+)"
)
FAIL_NODE = re.compile(
    r"Node \[nodeId:(?P<node_id>\d+),\s*streamId:(?P<stream_id>\d+).*?"
    r"KernelInfos\{funcName:(?P<func_name>[^,}]+),\s*kernelType:(?P<kernel_type>[^,}]+).*?"
    r"\}: reason: (?P<reason>.*)$"
)
SCOPE_LINE = re.compile(
    r"Scope \d+ \(scopeId=(?P<scope_id>\d+)\): (?P<node_count>\d+) nodes, "
    r"(?P<stream_count>\d+) streams.*scopeNames=\[(?P<scope_names>[^\]]*)\]"
)
BREAK_LINE = re.compile(
    r"BreakInfo: breakReason=(?P<break_reason>.*?), triggerNode=(?P<trigger_node>\d+), "
    r"triggerStream=(?P<trigger_stream>\d+), fusionFailReason=(?P<fusion_fail_reason>.*?), "
    r"detail=\"(?P<detail>.*)\""
)
LAYER_PATTERNS = (
    re.compile(r"(?:decoder\.layer|layers?|blocks?)\.?[_-]?(?P<layer>\d+)"),
    re.compile(r"layer[_-](?P<layer>\d+)"),
)
NODE_RUNTIME_FIELDS = {
    "num_blocks": re.compile(r"\bnumBlocks:(?P<value>\d+)"),
    "cube_num": re.compile(r"\bcubeNum:(?P<value>\d+)"),
    "vector_num": re.compile(r"\bvecNum:(?P<value>\d+)"),
    "schedule_mode": re.compile(r"\bisScheModeOn:(?P<value>\d+)"),
    "mix_split": re.compile(r"\bneedMixKernelSplit:(?P<value>\d+)"),
    "resolved_num": re.compile(r"\bresolvedNum:(?P<value>\d+)"),
}
TASK_RATIO = re.compile(r"\btaskRatio:\[(?P<cube>\d+),(?P<vector>\d+)\]")


def _relative_artifact_path(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _model_id_from_artifact(path, root):
    root = Path(root).resolve()
    for parent in Path(path).resolve().parents:
        match = MODEL_DIRECTORY.match(parent.name)
        if match:
            return match.group("model_id")
        if parent == root:
            break
    return None


def _trusted_child_count(group):
    if group.get("node_count_mismatch"):
        return len(group.get("nodes") or [])
    return group.get("node_count") or 0


def _infer_scope_kind(round_name, scope_kind):
    if scope_kind is not None:
        if scope_kind not in {"manual", "automatic_aot"}:
            raise ValueError("scope_kind must be manual or automatic_aot")
        return scope_kind
    normalized = (round_name or "").strip().lower().replace("_", "-")
    if normalized in {"s1", "s1-auto", "automatic-aot"}:
        return "automatic_aot"
    return "manual"


def _classify_reason(text):
    if not text:
        return "UNKNOWN"
    explicit = REASONS.search(text)
    if explicit:
        return explicit.group(1)
    phrase_map = [
        ("there exists unfusible node in scope", "UNFUSIBLE_NODE"),
        ("there exists deadlock in scope", "DEADLOCK_DETECTED"),
        ("core count of syncall", "SYNCALL_OP_DROP"),
        ("per-op debug mode", "DEBUG_PER_OP_MAX_CORE"),
        ("does not support the operation of fusing SuperKernel", "OP_UNSUPPORT"),
        ("refresh task information at runtime", "DYNAMIC_TASK_UNSUPPORT"),
        ("actively marked that this operator is not fused", "IN_UNFUSIBLE_SCOPE"),
        ("not within user-marked", "NOT_IN_SCOPE"),
        ("Insufficient resources", "RESOURCE_INSUFFICIENT"),
        ("deadlock", "EXIST_DEADLOCK"),
        ("external dependency", "EXTERNAL_DEPEND"),
        ("Exceeded maximum scope", "EXCEED_SCOPE_MAX"),
        ("SIMT", "SIMT_OP_UNSUPPORT"),
    ]
    for phrase, reason in phrase_map:
        if phrase.lower() in text.lower():
            return reason
    return "UNKNOWN"


def _sample(items, limit):
    return items[:limit]


def _top_counter(counter, limit):
    return [
        {"name": str(name), "count": count}
        for name, count in counter.most_common(limit)
    ]


def _quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "avg": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p50": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "avg": sum(values) / len(values),
        "max": max(values),
    }


def _counter_table(counter):
    total = sum(counter.values())
    return [
        {
            "name": str(name),
            "count": count,
            "percent": (count / total * 100) if total else None,
        }
        for name, count in counter.most_common()
    ]


def _op_from_function(name):
    if not name:
        return None
    match = STATIC_KERNEL_OP.search(name)
    if match:
        return match.group(1)
    for marker in ("aclnn", "aclnnInplace"):
        if name.startswith(marker):
            tail = name[len(marker) :]
            return tail.split("_", 1)[0] or None
    return name.split("_", 1)[0]


def _parse_sk_function(name):
    match = SK_START_END.match(name or "")
    if not match:
        return None
    start = match.group("start")
    end = match.group("end")
    return {
        "scope": match.group("scope").strip(),
        "start_function": start,
        "end_function": end,
        "start_op": _op_from_function(start),
        "end_op": _op_from_function(end),
    }


def _layer_from_name(name):
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name or "")
        if match:
            return int(match.group("layer"))
    return None


def _scope_dimensions(scope_names):
    names = [
        stripped
        for name in scope_names
        if isinstance(name, str) and (stripped := name.strip())
    ]
    layer = next(
        (value for name in names if (value := _layer_from_name(name)) is not None),
        None,
    )
    segments = []
    for name in names:
        for pattern in LAYER_PATTERNS:
            match = pattern.search(name)
            if not match:
                continue
            suffix = name[match.end() :].strip().strip("._-/").strip()
            if suffix:
                segment = re.split(r"[._/\-]", suffix, maxsplit=1)[0].strip()
                if segment:
                    segments.append(segment)
            break
    return layer, sorted(set(segments))


def _node_runtime_fields(line):
    fields = {}
    for name, pattern in NODE_RUNTIME_FIELDS.items():
        match = pattern.search(line)
        fields[name] = int(match.group("value")) if match else None
    task_ratio = TASK_RATIO.search(line)
    fields["task_ratio"] = (
        [int(task_ratio.group("cube")), int(task_ratio.group("vector"))]
        if task_ratio
        else None
    )
    return fields


def _decorate_group(group):
    boundary = group.get("boundary") or {}
    scope = boundary.get("scope")
    layer, segments = _scope_dimensions([scope] if scope else [])
    group["layer"] = layer
    group["segments"] = segments
    schedule_values = [
        node["schedule_mode"]
        for node in group["nodes"]
        if node.get("schedule_mode") is not None
    ]
    if not schedule_values:
        group["control_core_mode"] = "unknown"
    elif any(schedule_values):
        group["control_core_mode"] = "enabled"
    else:
        group["control_core_mode"] = "disabled"
    group["parsed_node_count"] = len(group["nodes"])
    group["node_count_mismatch"] = group["node_count"] != len(group["nodes"])
    group["declared_child_count"] = group["node_count"]
    group["count_reliable"] = not group["node_count_mismatch"]
    group["trusted_child_count"] = _trusted_child_count(group)
    return group


def _scope_key(group):
    boundary = group.get("boundary") or {}
    scope = boundary.get("scope")
    if scope:
        return scope
    return f"scope_id:{group['scope_id']}"


def _histogram(values):
    return {
        str(value): count
        for value, count in sorted(collections.Counter(values).items())
    }


def _parse_fused_groups(log_paths, root):
    groups = []
    for path in log_paths:
        current = None
        for line in path.read_text(errors="replace").splitlines():
            header = FUSED_HEADER.search(line)
            if header:
                current = {
                    "function": header.group("name"),
                    "model_id": _model_id_from_artifact(path, root),
                    "scope_id": int(header.group("scope_id")),
                    "node_count": int(header.group("count")),
                    "boundary": _parse_sk_function(header.group("name")),
                    "nodes": [],
                    "path": _relative_artifact_path(path, root),
                }
                groups.append(current)
                continue
            node = FUSED_NODE.search(line)
            if current and node:
                current["nodes"].append(
                    {
                        "node_id": int(node.group("node_id")),
                        "stream_id": int(node.group("stream_id")),
                        "func_name": node.group("func_name"),
                        "op_type": _op_from_function(node.group("func_name")),
                        "kernel_type": node.group("kernel_type"),
                        "static_kernel_symbol": node.group("func_name").startswith(
                            "static_kernel_"
                        ),
                        **_node_runtime_fields(line),
                    }
                )
    return [_decorate_group(group) for group in groups]


def _parse_failures(log_paths, root):
    failures = []
    for path in log_paths:
        for line in path.read_text(errors="replace").splitlines():
            match = FAIL_NODE.search(line)
            if not match:
                continue
            reason_text = match.group("reason").strip()
            failures.append(
                {
                    "node_id": int(match.group("node_id")),
                    "stream_id": int(match.group("stream_id")),
                    "func_name": match.group("func_name"),
                    "op_type": _op_from_function(match.group("func_name")),
                    "kernel_type": match.group("kernel_type"),
                    "static_kernel_symbol": match.group("func_name").startswith(
                        "static_kernel_"
                    ),
                    **_node_runtime_fields(line),
                    "reason": _classify_reason(reason_text),
                    "detail": reason_text,
                    "path": _relative_artifact_path(path, root),
                }
            )
    return failures


def _parse_scope_breaks(log_paths, root):
    breaks = []
    current_scope = None
    for path in log_paths:
        for line in path.read_text(errors="replace").splitlines():
            scope = SCOPE_LINE.search(line)
            if scope:
                current_scope = {
                    "scope_id": int(scope.group("scope_id")),
                    "node_count": int(scope.group("node_count")),
                    "stream_count": int(scope.group("stream_count")),
                    "scope_names": [
                        item.strip()
                        for item in scope.group("scope_names").split(",")
                        if item.strip()
                    ],
                    "path": _relative_artifact_path(path, root),
                }
                layer, segments = _scope_dimensions(current_scope["scope_names"])
                current_scope["layer"] = layer
                current_scope["segments"] = segments
                continue
            match = BREAK_LINE.search(line)
            if current_scope and match:
                fail_text = match.group("fusion_fail_reason").strip()
                breaks.append(
                    {
                        **current_scope,
                        "trigger_node": int(match.group("trigger_node")),
                        "trigger_stream": int(match.group("trigger_stream")),
                        "break_reason": _classify_reason(match.group("break_reason")),
                        "break_detail": match.group("break_reason").strip(),
                        "fusion_fail_reason": _classify_reason(fail_text),
                        "fusion_fail_detail": fail_text,
                        "detail": match.group("detail"),
                    }
                )
    return breaks


def _deduplicate_scope_breaks(scope_breaks):
    unique = []
    seen = set()
    for item in scope_breaks:
        key = (
            item["path"],
            item["scope_id"],
            item["node_count"],
            item["stream_count"],
            tuple(item.get("scope_names", [])),
            item["trigger_node"],
            item["trigger_stream"],
            item["break_reason"],
            item["fusion_fail_reason"],
            item.get("detail"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _summarize_fused_groups(fused_groups, sample_limit):
    child_counts = [_trusted_child_count(group) for group in fused_groups]
    node_records = [node for group in fused_groups for node in group["nodes"]]
    by_scope = collections.defaultdict(list)
    for group in fused_groups:
        by_scope[_scope_key(group)].append(group)

    top_scopes = []
    for scope_name, groups in by_scope.items():
        scope_child_counts = [_trusted_child_count(group) for group in groups]
        scope_nodes = [node for group in groups for node in group["nodes"]]
        top_scopes.append(
            {
                "scope": scope_name,
                "group_count": len(groups),
                "total_child_nodes": sum(scope_child_counts),
                "child_count_stats": _stats(scope_child_counts),
                "single_child_group_count": sum(
                    count <= 1 for count in scope_child_counts
                ),
                "stream_count_stats": _stats(
                    len({node["stream_id"] for node in group["nodes"]})
                    for group in groups
                ),
                "top_child_op_types": _top_counter(
                    collections.Counter(
                        node.get("op_type") or node.get("func_name")
                        for node in scope_nodes
                    ),
                    sample_limit,
                ),
                "sample_boundaries": _sample(
                    [
                        {
                            "function": group["function"],
                            "boundary": group["boundary"],
                            "node_count": _trusted_child_count(group),
                            "declared_node_count": group.get("node_count"),
                            "node_count_mismatch": group.get("node_count_mismatch"),
                        }
                        for group in groups
                    ],
                    min(sample_limit, 10),
                ),
            }
        )

    top_scopes.sort(
        key=lambda item: (
            item["total_child_nodes"],
            item["group_count"],
            item["child_count_stats"]["max"] or 0,
        ),
        reverse=True,
    )

    return {
        "group_count": len(fused_groups),
        "total_child_nodes": sum(child_counts),
        "child_count_stats": _stats(child_counts),
        "child_count_histogram": _histogram(child_counts),
        "single_child_group_count": sum(count <= 1 for count in child_counts),
        "single_child_group_ratio": (
            sum(count <= 1 for count in child_counts) / len(child_counts)
            if child_counts
            else None
        ),
        "multi_child_group_count": sum(count > 1 for count in child_counts),
        "max_child_group_samples": _sample(
            sorted(
                [
                    {
                        "function": group["function"],
                        "scope_id": group["scope_id"],
                        "boundary": group["boundary"],
                        "node_count": _trusted_child_count(group),
                        "declared_node_count": group.get("node_count"),
                        "node_count_mismatch": group.get("node_count_mismatch"),
                        "stream_ids": sorted(
                            {node["stream_id"] for node in group["nodes"]}
                        ),
                        "child_op_types": _top_counter(
                            collections.Counter(
                                node.get("op_type") or node.get("func_name")
                                for node in group["nodes"]
                            ),
                            12,
                        ),
                    }
                    for group in fused_groups
                ],
                key=lambda item: item["node_count"],
                reverse=True,
            ),
            min(sample_limit, 20),
        ),
        "top_scopes": _sample(top_scopes, sample_limit),
        "top_child_op_types": _top_counter(
            collections.Counter(
                node.get("op_type") or node.get("func_name") for node in node_records
            ),
            sample_limit,
        ),
        "kernel_type_counts": _top_counter(
            collections.Counter(node.get("kernel_type") for node in node_records),
            sample_limit,
        ),
    }


def _candidate_from_fused_group(group):
    nodes = group.get("nodes") or []
    boundary = group.get("boundary") or {}
    op_sequence = [node.get("op_type") or node.get("func_name") for node in nodes]
    return {
        "model_id": group.get("model_id"),
        "scope_id": group.get("scope_id"),
        "function": group.get("function"),
        "source_scope": boundary.get("scope"),
        "boundary": boundary,
        "child_count": _trusted_child_count(group),
        "declared_child_count": group.get("node_count"),
        "parsed_child_count": len(nodes),
        "node_count_mismatch": group.get("node_count_mismatch"),
        "count_reliable": not group.get("node_count_mismatch"),
        "op_sequence": op_sequence,
        "kernel_type_sequence": [node.get("kernel_type") for node in nodes],
        "stream_ids": sorted(
            stream_id
            for stream_id in {node.get("stream_id") for node in nodes}
            if stream_id is not None
        ),
        "control_core_mode": group.get("control_core_mode"),
        "layer": group.get("layer"),
        "segments": group.get("segments") or [],
        "path": group.get("path"),
        "nodes": nodes,
        "performance_action": "pending_profiling_analysis",
    }


def build_deep_auto_scope_plan(summary, min_child_nodes=5, sample_limit=20):
    """Inventory reliable S1_auto groups for profiling-driven manual scope work."""
    groups = summary.get("sk_inventory") or []
    selected_groups = [
        group
        for group in groups
        if group.get("count_reliable", not group.get("node_count_mismatch"))
    ]
    skipped_groups = [
        group
        for group in groups
        if not group.get("count_reliable", not group.get("node_count_mismatch"))
    ]
    selected_groups.sort(
        key=lambda group: (
            -_trusted_child_count(group),
            str(group.get("function") or ""),
            group.get("scope_id") or 0,
        )
    )
    candidates = [_candidate_from_fused_group(group) for group in selected_groups]
    by_sequence = collections.Counter(
        tuple(candidate["op_sequence"]) for candidate in candidates
    )
    by_boundary = collections.Counter(
        (
            candidate["boundary"].get("start_op")
            if candidate.get("boundary")
            else None,
            candidate["boundary"].get("end_op") if candidate.get("boundary") else None,
            candidate["child_count"],
        )
        for candidate in candidates
    )
    return {
        "strategy": "manual_scope_inventory_pending_profiling",
        "source": "S1_auto sk_inventory",
        "min_child_nodes": min_child_nodes,
        "min_child_nodes_role": "descriptive_only",
        "selected_group_count": len(candidates),
        "skipped_group_count": len(skipped_groups),
        "selected_child_node_total": sum(
            candidate["child_count"] or 0 for candidate in candidates
        ),
        "skipped_child_node_total": sum(
            _trusted_child_count(group) for group in skipped_groups
        ),
        "child_count_histogram": _histogram(
            candidate["child_count"] for candidate in candidates
        ),
        "skipped_child_count_histogram": _histogram(
            _trusted_child_count(group) for group in skipped_groups
        ),
        "manual_scope_candidates": candidates,
        "top_op_sequence_patterns": [
            {"count": count, "op_sequence": list(sequence)}
            for sequence, count in by_sequence.most_common(sample_limit)
        ],
        "top_boundary_patterns": [
            {
                "count": count,
                "start_op": boundary[0],
                "end_op": boundary[1],
                "child_count": boundary[2],
            }
            for boundary, count in by_boundary.most_common(sample_limit)
        ],
        "guardrail": (
            "Translate these ranges back to source-level scope markers before running "
            "a manual candidate. Child count is descriptive and must not select or reject "
            "a range; fresh profiling analysis decides performance action. Do not cross "
            "collectives, dynamic-task operators, cache mutation, or dependencies unless "
            "the source mapping proves the same ordering as S1_auto."
        ),
    }


def _summarize_failures(fusion_failures, sample_limit):
    reason_counter = collections.Counter(item["reason"] for item in fusion_failures)
    top_op_types_by_reason = {}
    top_kernel_types_by_reason = {}
    examples_by_reason = {}
    for reason in sorted(reason_counter):
        reason_items = [item for item in fusion_failures if item["reason"] == reason]
        top_op_types_by_reason[reason] = _top_counter(
            collections.Counter(
                item.get("op_type") or item.get("func_name") for item in reason_items
            ),
            sample_limit,
        )
        top_kernel_types_by_reason[reason] = _top_counter(
            collections.Counter(item.get("kernel_type") for item in reason_items),
            sample_limit,
        )
        examples_by_reason[reason] = _sample(reason_items, min(sample_limit, 10))
    return {
        "record_count": len(fusion_failures),
        "reason_counts": dict(sorted(reason_counter.items())),
        "reason_table": _counter_table(reason_counter),
        "top_op_types_by_reason": top_op_types_by_reason,
        "top_kernel_types_by_reason": top_kernel_types_by_reason,
        "examples_by_reason": examples_by_reason,
    }


def _summarize_scope_breaks(scope_breaks, sample_limit, raw_record_count=None):
    break_counter = collections.Counter(item["break_reason"] for item in scope_breaks)
    fail_counter = collections.Counter(
        item["fusion_fail_reason"] for item in scope_breaks
    )
    scope_names_by_reason = collections.defaultdict(collections.Counter)
    details_by_reason = collections.defaultdict(collections.Counter)
    examples_by_reason = collections.defaultdict(list)
    for item in scope_breaks:
        reason = item["fusion_fail_reason"]
        scope_name = (
            ",".join(item.get("scope_names") or []) or f"scope_id:{item['scope_id']}"
        )
        scope_names_by_reason[reason][scope_name] += 1
        detail = (
            item.get("fusion_fail_detail")
            or item.get("detail")
            or item.get("break_detail")
        )
        details_by_reason[reason][detail] += 1
        if len(examples_by_reason[reason]) < min(sample_limit, 10):
            examples_by_reason[reason].append(item)
    return {
        "record_count": len(scope_breaks),
        "raw_record_count": (
            raw_record_count if raw_record_count is not None else len(scope_breaks)
        ),
        "duplicate_record_count": (
            raw_record_count - len(scope_breaks) if raw_record_count is not None else 0
        ),
        "break_reason_counts": dict(sorted(break_counter.items())),
        "fusion_fail_reason_counts": dict(sorted(fail_counter.items())),
        "fusion_fail_reason_table": _counter_table(fail_counter),
        "top_scope_names_by_reason": {
            reason: _top_counter(counter, sample_limit)
            for reason, counter in sorted(scope_names_by_reason.items())
        },
        "top_details_by_reason": {
            reason: _top_counter(counter, sample_limit)
            for reason, counter in sorted(details_by_reason.items())
        },
        "examples_by_reason": dict(sorted(examples_by_reason.items())),
    }


def _disconnect_metrics(groups, breaks):
    child_counts = [_trusted_child_count(group) for group in groups]
    single_ratio = (
        sum(count <= 1 for count in child_counts) / len(child_counts)
        if child_counts
        else None
    )
    actionable_breaks = [
        item for item in breaks if item["fusion_fail_reason"] not in EXPECTED_REASONS
    ]
    breaks_per_group = len(actionable_breaks) / len(groups) if groups else None
    groups_per_100_children = (
        len(groups) / sum(child_counts) * 100 if sum(child_counts) else None
    )
    if not groups:
        level = "no_fusion"
    elif (single_ratio or 0) >= 0.5 or (breaks_per_group or 0) >= 2:
        level = "severe"
    elif (single_ratio or 0) >= 0.2 or (breaks_per_group or 0) >= 1:
        level = "moderate"
    else:
        level = "low"
    return {
        "level": level,
        "fused_group_count": len(groups),
        "total_child_nodes": sum(child_counts),
        "single_child_group_ratio": single_ratio,
        "actionable_break_count": len(actionable_breaks),
        "breaks_per_fused_group": breaks_per_group,
        "fused_groups_per_100_child_nodes": groups_per_100_children,
        "interpretation": (
            "This is a fragmentation indicator, not fusion coverage. Scope logs may "
            "include markers and repeated fragments, so do not divide by model op count."
        ),
    }


def _control_core_summary(groups, additional_nodes=()):
    fused_nodes = [node for group in groups for node in group["nodes"]]
    additional_nodes = list(additional_nodes)
    nodes = [*fused_nodes, *additional_nodes]
    known = [node for node in nodes if node.get("schedule_mode") is not None]
    enabled = sum(node["schedule_mode"] != 0 for node in known)
    if not known:
        status = "unknown"
    elif enabled == len(known):
        status = "enabled"
    elif enabled:
        status = "mixed"
    else:
        status = "disabled"
    group_counter = collections.Counter(group["control_core_mode"] for group in groups)
    return {
        "status": status,
        "evidence_field": "KernelInfos.isScheModeOn",
        "known_child_count": len(known),
        "enabled_child_count": enabled,
        "disabled_child_count": len(known) - enabled,
        "group_mode_counts": dict(sorted(group_counter.items())),
        "fused_child_evidence_count": sum(
            node.get("schedule_mode") is not None for node in fused_nodes
        ),
        "unfused_child_evidence_count": sum(
            node.get("schedule_mode") is not None for node in additional_nodes
        ),
        "guardrail": (
            "isScheModeOn is the AOT metadata evidence for scheduling/control-core mode. "
            "Confirm the field semantics against the active CANN revision before changing "
            "runtime scheduling policy."
        ),
    }


def _summarize_layers(fused_groups, scope_breaks, sample_limit):
    layers = sorted(
        {
            item["layer"]
            for item in [*fused_groups, *scope_breaks]
            if item.get("layer") is not None
        }
    )
    result = {}
    for layer in layers:
        groups = [group for group in fused_groups if group.get("layer") == layer]
        breaks = [item for item in scope_breaks if item.get("layer") == layer]
        nodes = [node for group in groups for node in group["nodes"]]
        result[str(layer)] = {
            "sk_count": len(groups),
            "total_child_nodes": sum(_trusted_child_count(group) for group in groups),
            "child_op_count_stats": _stats(
                _trusted_child_count(group) for group in groups
            ),
            "child_op_count_histogram": _histogram(
                _trusted_child_count(group) for group in groups
            ),
            "single_child_sk_count": sum(
                _trusted_child_count(group) <= 1 and group.get("count_reliable")
                for group in groups
            ),
            "scope_fragment_count": len(breaks),
            "declared_scope_node_count": sum(item["node_count"] for item in breaks),
            "break_reason_counts": dict(
                sorted(
                    collections.Counter(
                        item["fusion_fail_reason"] for item in breaks
                    ).items()
                )
            ),
            "segments": sorted(
                {
                    segment
                    for item in [*groups, *breaks]
                    for segment in item.get("segments", [])
                }
            ),
            "stream_count_stats": _stats(
                len({node["stream_id"] for node in group["nodes"]}) for group in groups
            ),
            "top_child_op_types": _top_counter(
                collections.Counter(
                    node.get("op_type") or node["func_name"] for node in nodes
                ),
                sample_limit,
            ),
            "kernel_type_counts": _top_counter(
                collections.Counter(node["kernel_type"] for node in nodes), sample_limit
            ),
            "control_core": _control_core_summary(groups),
            "disconnect": _disconnect_metrics(groups, breaks),
            "sk_functions": [group["function"] for group in groups],
        }
    return {
        "layer_count": len(result),
        "layers": result,
        "coverage_limit": (
            "Only deterministic layer names found in scope metadata are counted. "
            "Missing layers are evidence gaps, not zero-SK layers."
        ),
    }


def _break_action(reason, items):
    non_static = sum(not item.get("static_kernel_symbol", False) for item in items)
    base = {
        "reason": reason,
        "count": len(items),
        "non_static_symbol_or_unknown_count": non_static,
        "option_candidates": [],
        "can_reduce_before_operator_adaptation": False,
    }
    if reason in {"NOT_IN_SCOPE", "IN_UNFUSIBLE_SCOPE", "EXCEED_SCOPE_MAX"}:
        return {
            **base,
            "category": "scope_configuration",
            "can_reduce_before_operator_adaptation": True,
            "action": "fix marker placement, intentional exclusions, or scope-name cardinality",
        }
    if reason == "RESOURCE_INSUFFICIENT":
        return {
            **base,
            "category": "scope_resource",
            "can_reduce_before_operator_adaptation": True,
            "action": "split at dependency/resource boundaries before testing a wider scope",
        }
    if reason in {
        "EXIST_DEADLOCK",
        "DEADLOCK_DETECTED",
        "EXTERNAL_DEPEND",
        "ISOLATED_EVENT",
    }:
        details = " ".join(
            str(item.get("detail") or item.get("fusion_fail_detail") or "")
            for item in items
        ).lower()
        options = []
        if "valuewait" in details or "value wait" in details:
            options.append("aggressive_opt_strategies.value_breaker_bypass")
        if "event" in details:
            options.append("aggressive_opt_strategies.event_breaker_bypass")
        if "task" in details:
            options.append("aggressive_opt_strategies.task_breaker_bypass")
        return {
            **base,
            "category": "dependency",
            "option_candidates": options,
            "can_reduce_before_operator_adaptation": bool(options),
            "action": (
                "preserve dependency ordering; test only matching accepted aggressive options "
                "after owner confirmation and deadlock/correctness checks"
            ),
        }
    if reason in {"OP_UNSUPPORT", "SIMT_OP_UNSUPPORT", "KERNEL_ATTR_GET_FAILED"}:
        return {
            **base,
            "category": "deferred_operator_adaptation",
            "action": "move outside scope now; adapt the Ascend C/non-SK child in the next phase",
        }
    if reason == "DYNAMIC_TASK_UNSUPPORT":
        return {
            **base,
            "category": "deferred_runtime_or_static_kernel",
            "action": "exclude now; enable/adapt static task information in the next phase",
        }
    if reason in {"EXCEED_CORE_MAX", "SYNCALL_OP_DROP"}:
        return {
            **base,
            "category": "deferred_core_synchronization",
            "action": "validate child core count/full-core synchronization before deeper fusion",
        }
    if reason in EXPECTED_REASONS:
        return {
            **base,
            "category": "expected_or_intentional",
            "action": "verify this exclusion matches the candidate manifest",
        }
    return {
        **base,
        "category": "unresolved",
        "action": "inspect the first trigger, node detail, and active AOT source before tuning",
    }


def _build_break_action_plan(fusion_failures, scope_breaks):
    source = list(fusion_failures)
    if not source:
        source = [
            {
                **item,
                "reason": item["fusion_fail_reason"],
                "static_kernel_symbol": False,
            }
            for item in scope_breaks
        ]
    by_reason = collections.defaultdict(list)
    for item in source:
        by_reason[item["reason"]].append(item)
    actions = [_break_action(reason, items) for reason, items in by_reason.items()]
    priority = {
        "scope_configuration": 0,
        "scope_resource": 1,
        "dependency": 2,
        "expected_or_intentional": 3,
        "deferred_runtime_or_static_kernel": 4,
        "deferred_operator_adaptation": 5,
        "deferred_core_synchronization": 6,
        "unresolved": 7,
    }
    actions.sort(key=lambda item: (priority.get(item["category"], 99), -item["count"]))
    return {
        "ordered_actions": actions,
        "minimal_break_sequence": [
            "remove scope/marker mistakes and split resource-heavy regions",
            "test only evidence-matched options accepted by the active wrapper",
            "re-run fusion and correctness to prove the break count decreased",
            "defer OP_UNSUPPORT, non-static/unknown symbols, and operator adaptation",
        ],
    }


def _build_per_sk_fusion_table(fused_groups, min_child_nodes):
    rows = []
    for group in fused_groups:
        child_count = _trusted_child_count(group)
        nodes = group.get("nodes") or []
        node_count_mismatch = bool(group.get("node_count_mismatch"))
        stream_ids = sorted(
            stream_id
            for stream_id in {node.get("stream_id") for node in nodes}
            if stream_id is not None
        )
        row = {
            "function": group.get("function"),
            "model_id": group.get("model_id"),
            "scope_id": group.get("scope_id"),
            "boundary": group.get("boundary"),
            "source_scope": (group.get("boundary") or {}).get("scope"),
            "layer": group.get("layer"),
            "segments": group.get("segments") or [],
            "child_count": child_count,
            "declared_child_count": group.get("node_count"),
            "parsed_child_count": len(nodes),
            "node_count_mismatch": node_count_mismatch,
            "count_reliable": not node_count_mismatch,
            "counts_as_effective_fusion": (
                not node_count_mismatch and child_count >= min_child_nodes
            ),
            "launch_count_without_sk": child_count,
            "launch_count_with_sk": 1 if child_count else 0,
            "estimated_launch_reduction": max(child_count - 1, 0),
            "stream_ids": stream_ids,
            "stream_count": len(stream_ids),
            "control_core_mode": group.get("control_core_mode"),
            "op_sequence": [
                node.get("op_type") or node.get("func_name") for node in nodes
            ],
            "child_functions": [node.get("func_name") for node in nodes],
            "kernel_type_counts": _top_counter(
                collections.Counter(node.get("kernel_type") for node in nodes),
                len(nodes) or 1,
            ),
            "path": group.get("path"),
        }
        rows.append(row)
    rows.sort(
        key=lambda item: (
            not item["counts_as_effective_fusion"],
            -(item["child_count"] or 0),
            str(item.get("function") or ""),
        )
    )
    return rows


def _build_non_fusion_operator_table(fusion_failures, scope_breaks):
    rows = []
    total_records = len(fusion_failures) + len(scope_breaks)
    by_reason = collections.defaultdict(list)
    for item in fusion_failures:
        by_reason[item["reason"]].append(item)
    for item in scope_breaks:
        by_reason[item["fusion_fail_reason"]].append(item)
    action_by_reason = {
        reason: _break_action(reason, items)["action"]
        for reason, items in by_reason.items()
    }

    failure_groups = collections.defaultdict(list)
    for item in fusion_failures:
        key = (
            "sk_fusion_fail_reasons.log",
            item["reason"],
            item.get("op_type"),
            item.get("kernel_type"),
        )
        failure_groups[key].append(item)
    for (source, reason, op_type, kernel_type), items in failure_groups.items():
        example = items[0]
        count = len(items)
        rows.append(
            {
                "source": source,
                "reason": reason,
                "count": count,
                "percent": count / total_records * 100 if total_records else None,
                "op_type": op_type,
                "kernel_type": kernel_type,
                "example_function": example.get("func_name"),
                "example_node_id": example.get("node_id"),
                "example_stream_id": example.get("stream_id"),
                "example_scope_names": None,
                "example_detail": example.get("detail"),
                "action": action_by_reason.get(reason),
            }
        )

    scope_groups = collections.defaultdict(list)
    for item in scope_breaks:
        key = (
            "sk_scope_split.log",
            item["fusion_fail_reason"],
            tuple(item.get("scope_names") or []),
            item.get("break_reason"),
        )
        scope_groups[key].append(item)
    for (source, reason, scope_names, break_reason), items in scope_groups.items():
        example = items[0]
        count = len(items)
        rows.append(
            {
                "source": source,
                "reason": reason,
                "count": count,
                "percent": count / total_records * 100 if total_records else None,
                "op_type": None,
                "kernel_type": None,
                "example_function": None,
                "example_node_id": example.get("trigger_node"),
                "example_stream_id": example.get("trigger_stream"),
                "example_scope_names": list(scope_names),
                "example_break_reason": break_reason,
                "example_detail": example.get("detail")
                or example.get("fusion_fail_detail")
                or example.get("break_detail"),
                "action": action_by_reason.get(reason),
            }
        )

    rows.sort(
        key=lambda item: (
            -item["count"],
            item["reason"],
            item["source"],
            str(item.get("op_type") or ""),
        )
    )
    return rows


def _build_round_report(
    fused_groups,
    fusion_failures,
    scope_breaks,
    *,
    min_child_nodes=5,
    round_name=None,
    scope_kind="manual",
):
    per_sk = _build_per_sk_fusion_table(fused_groups, min_child_nodes)
    effective = [item for item in per_sk if item["counts_as_effective_fusion"]]
    shallow = [item for item in per_sk if not item["counts_as_effective_fusion"]]
    single_child = [
        item
        for item in per_sk
        if item["child_count"] == 1 and item.get("count_reliable")
    ]
    shallow_non_single_child = [
        item for item in per_sk if 1 < item["child_count"] < min_child_nodes
    ]
    single_child_exclusion = {
        "policy": (
            "Compatibility-only child-count inventory. It never adjusts scope, blocks "
            "profiling, or controls promotion; fresh profiling analysis decides action."
        ),
        "scope_kind": scope_kind,
        "requires_scope_adjustment": False,
        "next_round_policy": "profiling_decides",
        "performance_action": "none",
        "single_child_group_count": len(single_child),
        "shallow_non_single_child_group_count": len(shallow_non_single_child),
        "candidates": [
            {
                "function": item.get("function"),
                "scope_id": item.get("scope_id"),
                "source_scope": item.get("source_scope"),
                "boundary": item.get("boundary"),
                "layer": item.get("layer"),
                "child_count": item.get("child_count"),
                "child_functions": item.get("child_functions") or [],
                "op_sequence": item.get("op_sequence") or [],
                "stream_ids": item.get("stream_ids") or [],
                "path": item.get("path"),
                "exclusion_status": "not_applicable",
                "performance_action": "pending_profiling_analysis",
            }
            for item in single_child
        ],
        "s1_auto_action": "none",
    }
    return {
        "round_name": round_name,
        "min_child_nodes": min_child_nodes,
        "filter_policy": (
            "Child-count thresholds and effective/shallow labels are descriptive metadata "
            "only. Every reliably mapped SK proceeds to profiling; fresh profiling "
            "analysis alone decides keep, prune, reprofile, or block."
        ),
        "effective_fusion": {
            "total_group_count": len(per_sk),
            "effective_group_count": len(effective),
            "shallow_group_count": len(shallow),
            "effective_child_node_total": sum(
                item["child_count"] for item in effective
            ),
            "shallow_child_node_total": sum(item["child_count"] for item in shallow),
            "effective_child_count_histogram": _histogram(
                item["child_count"] for item in effective
            ),
            "shallow_child_count_histogram": _histogram(
                item["child_count"] for item in shallow
            ),
            "effective_launch_reduction": sum(
                item["estimated_launch_reduction"] for item in effective
            ),
            "all_fused_launch_reduction": sum(
                item["estimated_launch_reduction"] for item in per_sk
            ),
        },
        "single_child_exclusion": single_child_exclusion,
        "per_sk_fusion_table": per_sk,
        "non_fusion_operator_table": _build_non_fusion_operator_table(
            fusion_failures, scope_breaks
        ),
        "required_per_round_report_fields": [
            "descriptive child-count threshold and effective/shallow labels",
            "all reliable SK groups proceed to fresh profiling analysis",
            "single-child and shallow groups carry no metadata-derived action",
            "per-SK child operator count and child list",
            "unfused operator/reason table",
            "launch reduction estimate from metadata",
            (
                "external fresh deep-fusion replay status and artifact link in "
                "round-evidence.json or the final report"
            ),
            (
                "external profiler comparison status and artifact link in "
                "round-evidence.json or the final report"
            ),
            (
                "external clean-performance and promotion status and artifact link in "
                "round-evidence.json or the final report"
            ),
        ],
    }


def _nonempty_records(path):
    return [
        line
        for line in path.read_text(errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _model_directory(path, root):
    for parent in path.parents:
        if parent.name.startswith("model_"):
            return parent
        if parent == root:
            break
    return None


def analyze_metadata(
    root,
    sample_limit=20,
    min_effective_child_nodes=5,
    round_name=None,
    scope_kind=None,
):
    root = Path(root).resolve()
    scope_kind = _infer_scope_kind(round_name, scope_kind)
    log_paths = sorted(path for path in root.rglob("*.log") if path.name in LOG_NAMES)
    model_dirs = {
        model_dir
        for path in log_paths
        if (model_dir := _model_directory(path, root)) is not None
    }
    fused_logs = [path for path in log_paths if path.name == "sk_fused_nodes.log"]
    fused_groups = _parse_fused_groups(fused_logs, root)
    fused_node_records = sum(_trusted_child_count(group) for group in fused_groups)
    fusion_failures = _parse_failures(
        (path for path in log_paths if path.name == "sk_fusion_fail_reasons.log"),
        root,
    )
    raw_scope_breaks = _parse_scope_breaks(
        (path for path in log_paths if path.name == "sk_scope_split.log"),
        root,
    )
    scope_breaks = _deduplicate_scope_breaks(raw_scope_breaks)

    raw_reason_counts = collections.Counter()
    for path in log_paths:
        raw_reason_counts.update(REASONS.findall(path.read_text(errors="replace")))
    if fusion_failures:
        reason_count_source = "fusion_failure_records"
        reason_counter = collections.Counter(item["reason"] for item in fusion_failures)
    else:
        reason_count_source = "deduplicated_scope_break_records"
        reason_counter = collections.Counter(
            item["fusion_fail_reason"] for item in scope_breaks
        )
    reason_counts = dict(sorted(reason_counter.items()))
    raw_reason_counts = dict(sorted(raw_reason_counts.items()))
    expected_reason_count = sum(
        count for reason, count in reason_counts.items() if reason in EXPECTED_REASONS
    )
    actionable_reason_count = sum(reason_counts.values()) - expected_reason_count
    layer_summary = _summarize_layers(fused_groups, scope_breaks, sample_limit)
    control_core = _control_core_summary(fused_groups, fusion_failures)
    break_action_plan = _build_break_action_plan(fusion_failures, scope_breaks)

    round_report = _build_round_report(
        fused_groups,
        fusion_failures,
        scope_breaks,
        min_child_nodes=min_effective_child_nodes,
        round_name=round_name,
        scope_kind=scope_kind,
    )

    return {
        "root": ".",
        "model_count": len(model_dirs),
        "files": {
            name: sum(path.name == name for path in log_paths)
            for name in sorted(LOG_NAMES)
        },
        "fused_node_records": fused_node_records,
        "fused_group_count": len(fused_groups),
        "sk_inventory": fused_groups,
        "layer_summary": layer_summary,
        "control_core": control_core,
        "disconnect": _disconnect_metrics(fused_groups, scope_breaks),
        "break_action_plan": break_action_plan,
        "round_report": round_report,
        "metadata_quality": {
            "declared_child_node_count": sum(
                group.get("node_count") or 0 for group in fused_groups
            ),
            "trusted_child_node_count": fused_node_records,
            "parsed_child_node_count": sum(
                len(group["nodes"]) for group in fused_groups
            ),
            "node_count_mismatch_group_count": sum(
                group["node_count_mismatch"] for group in fused_groups
            ),
            "layer_named_sk_count": sum(
                group.get("layer") is not None for group in fused_groups
            ),
            "layer_named_break_count": sum(
                item.get("layer") is not None for item in scope_breaks
            ),
            "raw_scope_break_record_count": len(raw_scope_breaks),
            "unique_scope_break_record_count": len(scope_breaks),
        },
        "fusion_depth": {
            "total_child_nodes": fused_node_records,
            "effective_min_child_nodes": min_effective_child_nodes,
            "effective_group_count": round_report["effective_fusion"][
                "effective_group_count"
            ],
            "shallow_group_count": round_report["effective_fusion"][
                "shallow_group_count"
            ],
            "effective_launch_reduction": round_report["effective_fusion"][
                "effective_launch_reduction"
            ],
            "avg_child_nodes_per_group": (
                fused_node_records / len(fused_groups) if fused_groups else 0
            ),
            "child_count_stats": _stats(
                _trusted_child_count(group) for group in fused_groups
            ),
            "child_count_histogram": _histogram(
                _trusted_child_count(group) for group in fused_groups
            ),
            "max_child_nodes_per_group": max(
                (_trusted_child_count(group) for group in fused_groups), default=0
            ),
            "single_child_group_count": sum(
                _trusted_child_count(group) <= 1 and group.get("count_reliable")
                for group in fused_groups
            ),
            "single_child_group_ratio": (
                sum(
                    _trusted_child_count(group) <= 1 and group.get("count_reliable")
                    for group in fused_groups
                )
                / len(fused_groups)
                if fused_groups
                else None
            ),
        },
        "reason_counts": reason_counts,
        "reason_count_source": reason_count_source,
        "raw_reason_token_counts": raw_reason_counts,
        "fused_group_summary": _summarize_fused_groups(fused_groups, sample_limit),
        "fusion_failure_summary": _summarize_failures(fusion_failures, sample_limit),
        "scope_break_summary": _summarize_scope_breaks(
            scope_breaks, sample_limit, raw_record_count=len(raw_scope_breaks)
        ),
        "expected_reason_count": expected_reason_count,
        "actionable_reason_count": actionable_reason_count,
        "fused_group_samples": _sample(fused_groups, sample_limit),
        "failure_samples": _sample(fusion_failures, sample_limit),
        "scope_break_samples": _sample(scope_breaks, sample_limit),
        "fusion_proven": fused_node_records > 0,
    }


def _canonical_signature_value(value):
    if isinstance(value, dict):
        return {
            str(key): _canonical_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_signature_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_signature_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _replay_signature(item):
    boundary = item.get("boundary") or {}
    signature = {
        "model_id": item.get("model_id"),
        "source_scope": item.get("source_scope"),
        "segments": _canonical_signature_value(item.get("segments") or []),
        "boundary": {
            "start_op": boundary.get("start_op"),
            "end_op": boundary.get("end_op"),
        },
        "op_sequence": _canonical_signature_value(item.get("op_sequence") or []),
        "child_count": item.get("child_count"),
    }
    if "graph_occurrence_fingerprint" in item:
        signature["graph_occurrence_fingerprint"] = item.get(
            "graph_occurrence_fingerprint"
        )
    return signature, json.dumps(
        signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _replay_group(item):
    signature, signature_key = _replay_signature(item)
    group = {
        "signature": signature,
        "model_id": signature["model_id"],
        "source_scope": signature["source_scope"],
        "segments": signature["segments"],
        "boundary": signature["boundary"],
        "ordered_child_op_sequence": signature["op_sequence"],
        "child_count": item.get("child_count"),
        "count_reliable": item.get("count_reliable") is True,
        "_signature_key": signature_key,
    }
    if "graph_occurrence_fingerprint" in signature:
        group["graph_occurrence_fingerprint"] = signature[
            "graph_occurrence_fingerprint"
        ]
    return group


def _replay_identity_complete(
    *,
    model_id,
    source_scope,
    segments,
    boundary,
    sequence,
    child_count,
    count_reliable,
):
    if not isinstance(model_id, str) or not model_id.isdigit():
        return False
    if not isinstance(source_scope, str) or not source_scope.strip():
        return False
    if (
        not isinstance(segments, list)
        or any(
            not isinstance(segment, str) or not segment.strip() for segment in segments
        )
        or segments != sorted(set(segments))
    ):
        return False
    if not isinstance(boundary, dict):
        return False
    start_op = boundary.get("start_op")
    end_op = boundary.get("end_op")
    if not isinstance(start_op, str) or not start_op.strip():
        return False
    if not isinstance(end_op, str) or not end_op.strip():
        return False
    if (
        not isinstance(sequence, list)
        or not sequence
        or any(
            not isinstance(op_type, str) or not op_type.strip() for op_type in sequence
        )
    ):
        return False
    if (
        isinstance(child_count, bool)
        or not isinstance(child_count, int)
        or child_count != len(sequence)
    ):
        return False
    return count_reliable is True and start_op == sequence[0] and end_op == sequence[-1]


def _replay_row_identity_complete(item):
    return _replay_identity_complete(
        model_id=item.get("model_id"),
        source_scope=item.get("source_scope"),
        segments=item.get("segments"),
        boundary=item.get("boundary"),
        sequence=item.get("op_sequence"),
        child_count=item.get("child_count"),
        count_reliable=item.get("count_reliable"),
    )


def _matched_replay_group_identity_complete(group):
    child_count = group.get("child_count")
    return _replay_identity_complete(
        model_id=group.get("model_id"),
        source_scope=group.get("source_scope"),
        segments=group.get("segments"),
        boundary=group.get("boundary"),
        sequence=group.get("ordered_child_op_sequence"),
        child_count=child_count,
        count_reliable=group.get("count_reliable"),
    ) and (
        group.get("verify_count_reliable") is True
        and isinstance(group.get("verify_child_count"), int)
        and not isinstance(group.get("verify_child_count"), bool)
        and group.get("verify_child_count") == child_count
    )


def _match_replay_groups(compat_groups, verify_groups):
    matched_groups = []
    compat_only_groups = []
    verify_only_groups = []
    compat_by_signature = collections.defaultdict(list)
    verify_by_signature = collections.defaultdict(list)
    for group in compat_groups:
        compat_by_signature[group["_signature_key"]].append(group)
    for group in verify_groups:
        verify_by_signature[group["_signature_key"]].append(group)

    for signature_key in sorted(set(compat_by_signature) | set(verify_by_signature)):
        compat_matches = compat_by_signature.get(signature_key, [])
        verify_matches = verify_by_signature.get(signature_key, [])
        if len(compat_matches) == 1 and len(verify_matches) == 1:
            compat_group = compat_matches[0]
            verify_group = verify_matches[0]
            matched_group = {
                "signature": compat_group["signature"],
                "model_id": compat_group["model_id"],
                "source_scope": compat_group["source_scope"],
                "segments": compat_group["segments"],
                "boundary": compat_group["boundary"],
                "ordered_child_op_sequence": compat_group["ordered_child_op_sequence"],
                "child_count": compat_group["child_count"],
                "verify_child_count": verify_group["child_count"],
                "count_reliable": compat_group["count_reliable"],
                "verify_count_reliable": verify_group["count_reliable"],
            }
            if "graph_occurrence_fingerprint" in compat_group:
                matched_group["graph_occurrence_fingerprint"] = compat_group[
                    "graph_occurrence_fingerprint"
                ]
            matched_groups.append(matched_group)
        else:
            compat_only_groups.extend(compat_matches)
            verify_only_groups.extend(verify_matches)

    for group in compat_only_groups + verify_only_groups:
        group.pop("_signature_key", None)
    return matched_groups, compat_only_groups, verify_only_groups


def _portable_path(path, base_dir):
    return os.path.relpath(Path(path).resolve(), Path(base_dir).resolve()).replace(
        os.sep, "/"
    )


def _common_artifact_root(paths):
    resolved_paths = [str(Path(path).resolve()) for path in paths if path is not None]
    if not resolved_paths:
        return Path.cwd()
    return Path(os.path.commonpath(resolved_paths))


def _config_manifest(path, artifact_base=None):
    result = {"path": str(path) if path is not None else None, "fingerprint": None}
    if path is None:
        result.update({"valid": False, "failure_reason": "config_manifest_missing"})
        return result
    path = Path(path)
    result["path"] = (
        _portable_path(path, artifact_base)
        if artifact_base is not None
        else str(path.resolve())
    )
    try:
        data = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        result.update({"valid": False, "failure_reason": "config_manifest_invalid"})
        return result
    if not isinstance(data, dict):
        result.update({"valid": False, "failure_reason": "config_manifest_not_object"})
        return result
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    result.update(
        {
            "valid": True,
            "failure_reason": None,
            "fingerprint": hashlib.sha256(canonical).hexdigest(),
        }
    )
    return result


def _report_artifact_path(report, field, base_dir=None):
    value = report.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        path = Path(value)
        return (
            (Path(base_dir) / path)
            if base_dir is not None and not path.is_absolute()
            else path
        )
    except (TypeError, ValueError, OSError):
        return None


def validate_replay_report(
    path,
    *,
    expected_candidate=None,
    expected_round_id=None,
    expected_source_revision=None,
):
    """Recompute a replay proof from its on-disk artifacts without trusting its claims."""
    result = {
        "path": str(path) if path is not None else None,
        "valid": False,
        "checks": {
            "candidate_name": False,
            "candidate_name_matches": False,
            "round_id": False,
            "round_id_matches": False,
            "source_revision": False,
            "source_revision_matches": False,
            "artifact_paths_valid": False,
            "fusion_reproducible": False,
            "config_fingerprint_match": False,
            "distinct_metadata_roots": False,
            "matched_fusion_group_count": False,
            "matched_reliable_signature": False,
            "report_matches_recomputed": False,
        },
        "failure_reason": None,
    }
    if path is None:
        result["failure_reason"] = "replay_evidence_missing"
        return result
    if not isinstance(path, (str, Path)):
        result["failure_reason"] = "replay_evidence_path_invalid"
        return result
    try:
        path = Path(path)
    except (TypeError, ValueError, OSError):
        result["failure_reason"] = "replay_evidence_path_invalid"
        return result
    try:
        path = path.resolve()
        result["path"] = str(path)
        report = json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError, json.JSONDecodeError):
        result["failure_reason"] = "replay_evidence_invalid"
        return result
    if not isinstance(report, dict):
        result["failure_reason"] = "replay_evidence_not_object"
        return result

    candidate_name = report.get("candidate_name")
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        result["failure_reason"] = "replay_candidate_name_invalid"
        return result
    candidate_name = candidate_name.strip()
    result["candidate_name"] = candidate_name
    result["checks"]["candidate_name"] = True
    result["checks"]["candidate_name_matches"] = (
        expected_candidate is None or candidate_name == expected_candidate
    )
    if not result["checks"]["candidate_name_matches"]:
        result["failure_reason"] = "replay_candidate_name_mismatch"
        return result

    round_id = report.get("round_id")
    if (
        not isinstance(round_id, str)
        or not round_id.strip()
        or round_id != round_id.strip()
        or not round_id.startswith(f"{candidate_name}-")
    ):
        result["failure_reason"] = "replay_round_id_invalid"
        return result
    result["round_id"] = round_id
    result["checks"]["round_id"] = True
    result["checks"]["round_id_matches"] = (
        expected_round_id is None or round_id == expected_round_id
    )
    if not result["checks"]["round_id_matches"]:
        result["failure_reason"] = "replay_round_id_mismatch"
        return result

    source_revision = report.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or not source_revision.strip()
        or source_revision != source_revision.strip()
    ):
        result["failure_reason"] = "replay_source_revision_invalid"
        return result
    result["source_revision"] = source_revision
    result["checks"]["source_revision"] = True
    result["checks"]["source_revision_matches"] = (
        expected_source_revision is None or source_revision == expected_source_revision
    )
    if not result["checks"]["source_revision_matches"]:
        result["failure_reason"] = "replay_source_revision_mismatch"
        return result

    report_base = path.parent
    compat_root = _report_artifact_path(report, "compat_metadata_root", report_base)
    verify_root = _report_artifact_path(report, "verify_metadata_root", report_base)
    compat_config = _report_artifact_path(
        report, "compat_config_manifest_path", report_base
    )
    verify_config = _report_artifact_path(
        report, "verify_config_manifest_path", report_base
    )
    if None in (compat_root, verify_root, compat_config, verify_config):
        result["failure_reason"] = "replay_artifact_path_invalid"
        return result
    try:
        paths_valid = (
            compat_root.is_dir()
            and verify_root.is_dir()
            and compat_config.is_file()
            and verify_config.is_file()
        )
    except OSError:
        paths_valid = False
    result["checks"]["artifact_paths_valid"] = paths_valid
    if not paths_valid:
        result["failure_reason"] = "replay_artifacts_invalid"
        return result

    replay_min_child_nodes = report.get("replay_min_child_nodes")
    if (
        isinstance(replay_min_child_nodes, bool)
        or not isinstance(replay_min_child_nodes, int)
        or replay_min_child_nodes < 1
    ):
        result["failure_reason"] = "replay_report_stale_or_tampered"
        return result

    try:
        recomputed = compare_fusion_replay(
            compat_root,
            verify_root,
            replay_min_child_nodes=replay_min_child_nodes,
            compat_config_manifest=compat_config,
            verify_config_manifest=verify_config,
            candidate_name=candidate_name,
            round_id=round_id,
            source_revision=source_revision,
            artifact_base=report_base,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        result["failure_reason"] = "replay_artifacts_invalid"
        return result
    result["recomputed_replay"] = recomputed
    report_matches_recomputed = all(
        field in report and report[field] == value
        for field, value in recomputed.items()
    )
    matched_groups = recomputed["matched_fusion_groups"]
    matched_reliable_signature = bool(matched_groups) and all(
        _matched_replay_group_identity_complete(group) for group in matched_groups
    )
    result["checks"] = {
        **result["checks"],
        "fusion_reproducible": recomputed["fusion_reproducible"],
        "config_fingerprint_match": recomputed["config_fingerprint_match"],
        "distinct_metadata_roots": (
            (report_base / recomputed["compat_metadata_root"]).resolve()
            != (report_base / recomputed["verify_metadata_root"]).resolve()
        ),
        "matched_fusion_group_count": recomputed["matched_fusion_group_count"] >= 1,
        "matched_reliable_signature": matched_reliable_signature,
        "report_matches_recomputed": report_matches_recomputed,
    }
    result["valid"] = all(result["checks"].values())
    result["fusion_reproducible"] = recomputed["fusion_reproducible"]
    result["deep_fusion_reproducible"] = recomputed["deep_fusion_reproducible"]
    if not result["valid"]:
        if not report_matches_recomputed:
            result["failure_reason"] = "replay_report_stale_or_tampered"
        elif not recomputed["fusion_reproducible"]:
            result["failure_reason"] = recomputed["failure_reason"]
        else:
            result["failure_reason"] = "replay_report_stale_or_tampered"
    return result


def compare_fusion_replay(
    compat_root,
    verify_root,
    *,
    replay_min_child_nodes=REPLAY_MIN_CHILD_NODES,
    compat_config_manifest=None,
    verify_config_manifest=None,
    candidate_name=None,
    round_id=None,
    source_revision=None,
    artifact_base=None,
):
    if (
        isinstance(replay_min_child_nodes, bool)
        or not isinstance(replay_min_child_nodes, int)
        or replay_min_child_nodes < 1
    ):
        raise ValueError("replay_min_child_nodes must be an integer >= 1")
    compat_root = Path(compat_root).resolve()
    verify_root = Path(verify_root).resolve()
    artifact_base = (
        Path(artifact_base).resolve()
        if artifact_base is not None
        else _common_artifact_root(
            [
                compat_root,
                verify_root,
                compat_config_manifest,
                verify_config_manifest,
            ]
        )
    )
    compat_config = _config_manifest(compat_config_manifest, artifact_base)
    verify_config = _config_manifest(verify_config_manifest, artifact_base)
    config_fingerprint_match = (
        compat_config["valid"]
        and verify_config["valid"]
        and compat_config["fingerprint"] == verify_config["fingerprint"]
    )
    compat_report = analyze_metadata(
        compat_root, min_effective_child_nodes=REPLAY_MIN_CHILD_NODES
    )
    verify_report = analyze_metadata(
        verify_root, min_effective_child_nodes=REPLAY_MIN_CHILD_NODES
    )
    compat_rows = compat_report["round_report"]["per_sk_fusion_table"]
    verify_rows = verify_report["round_report"]["per_sk_fusion_table"]

    def eligible_groups(rows, min_child_nodes):
        return [
            _replay_group(item)
            for item in rows
            if _replay_row_identity_complete(item)
            and isinstance(item.get("child_count"), int)
            and not isinstance(item.get("child_count"), bool)
            and item["child_count"] >= min_child_nodes
        ]

    compat_groups = eligible_groups(compat_rows, replay_min_child_nodes)
    verify_groups = eligible_groups(verify_rows, replay_min_child_nodes)
    compat_groups.sort(key=lambda item: (item["_signature_key"], item["child_count"]))
    verify_groups.sort(key=lambda item: (item["_signature_key"], item["child_count"]))
    matched_groups, compat_only_groups, verify_only_groups = _match_replay_groups(
        compat_groups, verify_groups
    )

    compat_deep_groups = eligible_groups(compat_rows, REPLAY_MIN_EFFECTIVE_CHILD_NODES)
    verify_deep_groups = eligible_groups(verify_rows, REPLAY_MIN_EFFECTIVE_CHILD_NODES)
    compat_deep_groups.sort(
        key=lambda item: (item["_signature_key"], item["child_count"])
    )
    verify_deep_groups.sort(
        key=lambda item: (item["_signature_key"], item["child_count"])
    )
    matched_deep_groups, _, _ = _match_replay_groups(
        compat_deep_groups, verify_deep_groups
    )

    if compat_root.resolve() == verify_root.resolve():
        failure_reason = "verify_metadata_not_fresh"
    elif not compat_config["valid"]:
        failure_reason = compat_config["failure_reason"]
    elif not verify_config["valid"]:
        failure_reason = verify_config["failure_reason"]
    elif not config_fingerprint_match:
        failure_reason = "config_fingerprint_mismatch"
    elif not compat_groups:
        failure_reason = "compat_has_no_effective_fusion"
    elif not verify_groups:
        failure_reason = "verify_has_no_effective_fusion"
    elif not matched_groups:
        failure_reason = "no_matching_effective_fusion_signature"
    else:
        failure_reason = None

    fusion_reproducible = failure_reason is None
    deep_fusion_reproducible = (
        compat_root != verify_root
        and config_fingerprint_match
        and bool(matched_deep_groups)
    )

    return {
        "candidate_name": candidate_name,
        "round_id": round_id,
        "source_revision": source_revision,
        "replay_min_child_nodes": replay_min_child_nodes,
        "min_child_nodes": replay_min_child_nodes,
        "artifact_base": ".",
        "compat_metadata_root": _portable_path(compat_root, artifact_base),
        "verify_metadata_root": _portable_path(verify_root, artifact_base),
        "compat_config_manifest_path": compat_config["path"],
        "verify_config_manifest_path": verify_config["path"],
        "compat_config_fingerprint": compat_config["fingerprint"],
        "verify_config_fingerprint": verify_config["fingerprint"],
        "config_fingerprint_match": config_fingerprint_match,
        "compat_config_manifest": compat_config,
        "verify_config_manifest": verify_config,
        "compat_effective_group_count": len(compat_groups),
        "verify_effective_group_count": len(verify_groups),
        "matched_effective_group_count": len(matched_groups),
        "matched_effective_groups": matched_groups,
        "compat_only_effective_groups": compat_only_groups,
        "verify_only_effective_groups": verify_only_groups,
        "compat_fusion_group_count": len(compat_groups),
        "verify_fusion_group_count": len(verify_groups),
        "matched_fusion_group_count": len(matched_groups),
        "matched_fusion_groups": matched_groups,
        "compat_only_fusion_groups": compat_only_groups,
        "verify_only_fusion_groups": verify_only_groups,
        "fusion_reproducible": fusion_reproducible,
        "deep_fusion_reproducible": deep_fusion_reproducible,
        "failure_reason": failure_reason,
    }


def compare_deep_fusion_replay(
    compat_root,
    verify_root,
    *,
    compat_config_manifest=None,
    verify_config_manifest=None,
    candidate_name=None,
    round_id=None,
    source_revision=None,
    artifact_base=None,
):
    """Compatibility wrapper for callers that require a five-child replay gate."""
    return compare_fusion_replay(
        compat_root,
        verify_root,
        replay_min_child_nodes=REPLAY_MIN_EFFECTIVE_CHILD_NODES,
        compat_config_manifest=compat_config_manifest,
        verify_config_manifest=verify_config_manifest,
        candidate_name=candidate_name,
        round_id=round_id,
        source_revision=source_revision,
        artifact_base=artifact_base,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="metadata directory to scan recursively"
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--verify-root", type=Path)
    parser.add_argument("--replay-report-out", type=Path)
    parser.add_argument("--replay-min-child-nodes", type=int, default=1)
    parser.add_argument("--compat-config-manifest", type=Path)
    parser.add_argument("--verify-config-manifest", type=Path)
    parser.add_argument("--candidate-name")
    parser.add_argument("--round-id")
    parser.add_argument("--source-revision")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument(
        "--round-name", help="candidate or test-round name such as S1/S2/S4c"
    )
    parser.add_argument(
        "--scope-kind",
        choices=("manual", "automatic_aot"),
        help="scope ownership for the single-child next-round policy",
    )
    parser.add_argument(
        "--round-report-out",
        type=Path,
        help=(
            "write the per-round fusion report section, including descriptive "
            "child-count buckets"
        ),
    )
    parser.add_argument(
        "--effective-min-child-nodes",
        type=int,
        default=5,
        help=(
            "descriptive threshold separating effective and shallow fusion summary "
            "buckets; it never filters profiling, replay, scope candidates, or "
            "performance analysis"
        ),
    )
    parser.add_argument(
        "--deep-auto-scope-plan-out",
        type=Path,
        help=(
            "write all reliable S1_auto fused groups as manual-scope candidates; "
            "--deep-auto-min-child-nodes remains descriptive only"
        ),
    )
    parser.add_argument(
        "--deep-auto-min-child-nodes",
        type=int,
        default=5,
        help="descriptive child-count threshold; never excludes a reliable candidate",
    )
    args = parser.parse_args(argv)
    if args.replay_report_out and not args.verify_root:
        parser.error("--replay-report-out requires --verify-root")
    if args.verify_root and (
        args.compat_config_manifest is None or args.verify_config_manifest is None
    ):
        parser.error(
            "--verify-root requires --compat-config-manifest and --verify-config-manifest"
        )
    if args.verify_root and (
        not args.candidate_name or not args.candidate_name.strip()
    ):
        parser.error("--verify-root requires --candidate-name NAME")
    if args.verify_root and (not args.round_id or not args.round_id.strip()):
        parser.error("--verify-root requires --round-id ROUND")
    if args.verify_root and (
        not args.source_revision or not args.source_revision.strip()
    ):
        parser.error("--verify-root requires --source-revision REVISION")
    if args.verify_root and not args.round_id.strip().startswith(
        f"{args.candidate_name.strip()}-"
    ):
        parser.error("--round-id must belong to --candidate-name family")
    if args.verify_root and args.replay_min_child_nodes < 1:
        parser.error("--replay-min-child-nodes must be >= 1")
    if not args.verify_root and (
        args.compat_config_manifest is not None
        or args.verify_config_manifest is not None
    ):
        parser.error("config manifests require --verify-root")
    if not args.verify_root and args.candidate_name is not None:
        parser.error("--candidate-name requires --verify-root")
    if not args.verify_root and (
        args.round_id is not None or args.source_revision is not None
    ):
        parser.error("--round-id and --source-revision require --verify-root")

    report = analyze_metadata(
        args.root,
        sample_limit=args.sample_limit,
        min_effective_child_nodes=args.effective_min_child_nodes,
        round_name=args.round_name,
        scope_kind=args.scope_kind,
    )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n")
    if args.round_report_out:
        args.round_report_out.parent.mkdir(parents=True, exist_ok=True)
        args.round_report_out.write_text(
            json.dumps(report["round_report"], indent=2, sort_keys=True) + "\n"
        )
    if args.deep_auto_scope_plan_out:
        plan = build_deep_auto_scope_plan(
            report,
            min_child_nodes=args.deep_auto_min_child_nodes,
            sample_limit=args.sample_limit,
        )
        args.deep_auto_scope_plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.deep_auto_scope_plan_out.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
    if args.verify_root:
        replay = compare_fusion_replay(
            args.root,
            args.verify_root,
            replay_min_child_nodes=args.replay_min_child_nodes,
            compat_config_manifest=args.compat_config_manifest,
            verify_config_manifest=args.verify_config_manifest,
            candidate_name=args.candidate_name.strip(),
            round_id=args.round_id.strip(),
            source_revision=args.source_revision.strip(),
            artifact_base=(
                args.replay_report_out.parent if args.replay_report_out else None
            ),
        )
        replay_output = json.dumps(replay, indent=2, sort_keys=True)
        print(replay_output)
        if args.replay_report_out:
            args.replay_report_out.parent.mkdir(parents=True, exist_ok=True)
            args.replay_report_out.write_text(replay_output + "\n")
        return 0 if replay["fusion_reproducible"] else 1
    print(output)
    if report["model_count"] == 0:
        return 2
    return 0 if report["fusion_proven"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
