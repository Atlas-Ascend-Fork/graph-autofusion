#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Registry-first update library extraction for SK network analysis."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TypeVar


RE_SCOPE_SUMMARY = re.compile(
    r"Scope (\d+)(?: \(scopeId=(\d+)\))?: (\d+) nodes, (\d+) streams, "
    r"scopeBitFlags=([0-9a-fA-F]+), scopeNames=\[(.*)\]"
)
RE_SCOPE_NODE = re.compile(
    r"\[(\d+)\] \[nodeId:(\d+), streamId:(-?\d+), "
    r"streamIdxInGraph:(\d+), nodeIdxInStream:(\d+)"
    r"(?:, ([^\]]+))?\](?: - (.+))?"
)
RE_SCOPE_NODE_HEADER = re.compile(r"Scope (\d+) nodes: (\d+)")
RE_SCOPE_STREAM = re.compile(
    r"Scope (\d+) StreamInfo\[(\d+)\]: streamIdx=(\d+), headNode=(\d+), tailNode=(\d+), nodeSize=(\d+)"
)
RE_SCOPE_BATCH_HEADER = re.compile(r"Printing scope split results, total scopes: (\d+)")
RE_SCOPE_BATCH_BEGIN = re.compile(
    r"Scope split results begin: pass=([^,]+), totalScopes=(\d+)"
)
RE_FUSED_HEADER = re.compile(r"SK Function:\s*(.+), scope id: (\d+), Node Count: (\d+)")
RE_TASK_SECTION = re.compile(r"(AIC|AIV) TaskQue: cap=(\d+), tasks=(\d+)")
RE_TASK_ENTRY_HEAD = re.compile(r"\[(\d+)\]\s*type=([A-Z_]+),\s*(.*)")
RE_TASK_ENTRY_ADDR = re.compile(r"entry\[(\d+)\]=([0-9a-zA-Zx]+)")
RE_HEADER_INFO = re.compile(
    r"SkHeaderInfo: aicOff=(\d+), aivOff=(\d+), counterOff=(\d+), (?:wsOff=(\d+), )?"
    r"dfxOff=(\d+), eventConfigOff=(\d+), nodeCnt=(\d+), totalSize=(\d+)"
)
RE_DFX_INFO = re.compile(
    r"dfx\[(\d+)\]: bin=([0-9a-zA-Zx]+), ori=([0-9a-zA-Zx]+), aicSize=([0-9a-zA-Zx]+), aivSize=([0-9a-zA-Zx]+)"
    r"(?:,\s*numBlocks=(\d+),\s*cubeNum=(\d+),\s*vecNum=(\d+))?"
)
RE_DFXINFO_HEADER = re.compile(r"=== SkDfxInfo \(offset=(\d+), nodeCnt=(\d+)\) ===")
RE_COUNTERINFO_HEADER = re.compile(r"=== SkCounterInfo \(offset=(\d+)\) ===")
RE_COUNTERINFO_CORE = re.compile(
    r"\[core (\d+)\] index=(\d+), launch=(\d+), exit=(\d+)"
)
RE_DFXINFO_NODE = re.compile(
    r"\[node (\d+)\] binHdl=0x([0-9a-zA-Zx]+), funcHdlOri=0x([0-9a-zA-Zx]+), "
    r"aicSize=0x([0-9a-zA-Zx]+), aivSize=0x([0-9a-zA-Zx]+)"
)
RE_DFXINFO_FUNC_NAME = re.compile(r"Function name:\s*(.+)")
RE_DFXINFO_ENTRY_AIC = re.compile(r"entryAic\[(\d+)\]=0x([0-9a-zA-Zx]+)")
RE_DFXINFO_ENTRY_AIV = re.compile(r"entryAiv\[(\d+)\]=0x([0-9a-zA-Zx]+)")
RE_EXCEPTION_FROM_SUPERKERNEL = re.compile(
    r"Exception is from superkernel function ['\"]([^'\"]+)['\"],\s*op_trace=([a-zA-Z]+)"
)
RE_EXCEPTION_CORE_MATCH = re.compile(
    r"\[Core (\d+)\] Found in node\[(\d+)\], entry\[(\d+)\]"
)
RE_EXCEPTION_CORE_FUNC = re.compile(r"\[Core (\d+)\] Function name:\s*(.+)")
RE_CORETYPE = re.compile(r"\[Core (\d+)\] CoreType:\s*([A-Z]+)")
RE_EXCEPTION_START_PC = re.compile(
    r"\[Core (\d+)\]\s*start\s*pc:\s*(0x[0-9a-fA-F]+)", re.IGNORECASE
)
RE_CURRENT_PC = re.compile(
    r"\[Core (\d+)\]\s*current\s*pc:\s*(0x[0-9a-fA-F]+)", re.IGNORECASE
)
RE_ENTRY_ADDRESS = re.compile(r"\[Core (\d+)\] Entry address:\s*(0x[0-9a-fA-F]+)")
RE_END_ADDRESS = re.compile(r"\[Core (\d+)\] End address:\s*(0x[0-9a-fA-F]+)")
RE_FUNCTION_SIZE = re.compile(
    r"\[Core (\d+)\] Function size:\s*(0x[0-9a-fA-F]+)\s+\((\d+)\s+bytes\)"
)
RE_NO_SUB_KERNEL_MATCHED = re.compile(
    r"\[Core (\d+)\] No sub kernel matched, aicore error occurred in sk entry\."
)
RE_MODEL_RI = re.compile(r"model_(\d+)")
RE_SCOPE_NAME = re.compile(r"scopeName:\s*(.*?)__skId:\s*(\d+)")
RE_FUNCTION_CONTEXT = re.compile(
    r"scopeName:\s*(.*?)__skId:\s*(\d+)(?:__startNodeName:\s*(.*?)__endNodeName:\s*(.*))?$"
)
RE_FUNC_NAME = re.compile(r"funcName:([^,}]+)")
RE_KERNEL_TYPE = re.compile(r"kernelType:([^,}]+)")
RE_SCOPE_UPDATE_BEGIN = re.compile(
    r"scope update begin:\s*(?:scopeName=([^,]+),\s*)?streamCount=(\d+)"
)
RE_SCHEDULED_SK_FUNCTION = re.compile(
    r"schedule scope: build finished,.*?\bskFuncName=([^,\s]+)"
)
RE_UPDATE_STREAM_BEGIN = re.compile(
    r"update stream begin:\s*(?:scopeName=([^,]+),\s*)?"
    r"streamId=(\d+), headNodeId=(\d+), tailNodeId=(\d+), "
    r"nodeSize=(\d+), customParamSize=(\d+)"
)
RE_UPDATING_NODE_TASK = re.compile(r"Updating node for task\s*:\s*(\d+)")
RE_UPDATED_KERNEL_NODE = re.compile(
    r"Updated kernel node for task (\d+) with argsHandle"
)
RE_UPDATE_STREAM_END = re.compile(
    r"update stream end:\s*(?:scopeName=([^,]+),\s*)?streamId=(\d+), visitedNodes=(\d+)"
)
RE_SCOPE_UPDATE_FINISHED = re.compile(
    r"scope update finished:\s*(?:scopeName=([^,]+),\s*)?(?:updateTotalNodes|update total nodes)=(\d+)"
)
RE_UPDATE_GRAPH_START = re.compile(r"Start update graph")
RE_UPDATE_GRAPH_END = re.compile(r"End update graph")
RE_NODE_UPDATE_RESULT = re.compile(
    r"node update result:\s*nodeId=(\d+),\s*(?:updateTargetType|type)=([A-Z_]+)(?:,\s*(.*))?$"
)
RE_EVENT_NODE_ADDR = re.compile(
    r"Updated (notify|wait|reset) node addrValue:\s*nodeId=(\d+),\s*addr=([^,\s]+)"
)
RE_EVENT_MEMORY_RESOURCE = re.compile(
    r"event memory allocated end:\s*eventId=(0x[0-9a-fA-F]+),\s*addr=([^,\s]+)"
)
RE_NODEID = re.compile(r"nodeId:(\d+)")
RE_STREAMID = re.compile(r"streamId:(-?\d+)")
RE_STREAM_IDX = re.compile(r"streamIdxInGraph:(\d+)")
RE_NODE_IDX = re.compile(r"nodeIdxInStream:(\d+)")
RE_EVENT_BLOCK = re.compile(
    r"(EventNotify|EventWait|EventReset|MemoryWrite|MemoryWait)\(([^)]*)\)"
)
RE_EVENT_ID = re.compile(r"eventId:(0x[0-9a-fA-F]+)")
RE_EVENT_FLAG = re.compile(r"eventFlag:(0x[0-9a-fA-F]+)")
RE_DEV_ARGS = re.compile(r"devArgs:(0x[0-9a-zA-Z]+)")
RE_TYPE = re.compile(r"type=(\w+)")
RE_LINE_TIMESTAMP = re.compile(
    r"^\s*(?:\[(?P<bracket_ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]|"
    r"(?P<plain_ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?))"
)
NON_FUSED_NODE_TYPES = {
    "EVENT_NOTIFY",
    "EVENT_WAIT",
    "EVENT_RESET",
    "MEMORY_WRITE",
    "MEMORY_WAIT",
}

PHASE_PATTERNS = [
    ("optimize_begin", "Begin aclskOptimize"),
    ("update_node_scope_flags_begin", "Starting UpdateNodeScopeBitFlags"),
    ("deadlock_refine", "DeadlockRefine"),
    ("build_tasks_begin", "start build tasks for super kernel"),
    ("build_tasks_finish", "Finish build tasks for super kernel"),
    ("shape_profiling", "[sk shape profiling]"),
    ("time_profiling", "[sk time profiling]"),
    ("optimize_end", "End aclskOptimize"),
]


class ModelInstancePartitionError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        partial_result: list[dict[str, Any]] | None = None,
        failed_model_instance_ids: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result or []
        self.failed_model_instance_ids = failed_model_instance_ids or []


SYNC_TYPE_MAP: dict[int, tuple[str, str, str]] = {
    0: ("ALL_SYNC", "ALL", "ALL"),
    1: ("AIC_TO_AIC", "CROSS", "AIC_TO_AIC"),
    2: ("AIV_TO_AIV", "CROSS", "AIV_TO_AIV"),
    3: ("SET_AIC_TO_AIV", "SET", "AIC_TO_AIV"),
    4: ("SET_AIV_TO_AIC", "SET", "AIV_TO_AIC"),
    5: ("WAIT_AIC_TO_AIV", "WAIT", "AIC_TO_AIV"),
    6: ("WAIT_AIV_TO_AIC", "WAIT", "AIV_TO_AIC"),
    7: ("SYNC_NONE", "NONE", "NONE"),
}

GRAPH_BOUND_TASK_TYPES = {"KERNEL", "FUNC", "PRELOAD"}
EVENT_TASK_TYPES = {
    "EVENT_NOTIFY",
    "EVENT_WAIT",
    "EVENT_RESET",
    "NOTIFY",
    "WAIT",
    "RESET",
}
EVENT_TASK_TO_NODE_TYPE = {
    "EVENT_NOTIFY": "EVENT_NOTIFY",
    "NOTIFY": "EVENT_NOTIFY",
    "EVENT_WAIT": "EVENT_WAIT",
    "WAIT": "EVENT_WAIT",
    "EVENT_RESET": "EVENT_RESET",
    "RESET": "EVENT_RESET",
}

COUNTER_LAUNCH_STATE_MAP: dict[int, tuple[str, str]] = {
    0: ("ORIGIN", "No SK entry executed yet"),
    1: ("SK_ENTRY_LAUNCHED", "SK started but no sub-kernel executed yet"),
    2: ("OP_LAUNCHED", "Sub-kernel is currently running"),
    3: ("OP_FINISHED", "Sub-kernel finished; next operator may be pending"),
    4: ("SK_ENTRY_FINISHED", "SK entry execution completed"),
}


class ParseContext:
    """Per-process source line cache for one collection run."""

    def __init__(
        self, model_dir: Path | None = None, *, validate_json_shadow: bool = True
    ) -> None:
        self.model_dir = model_dir
        self.validate_json_shadow = validate_json_shadow
        self.line_cache: dict[str, list[str]] = {}
        self.read_counts: dict[str, int] = {}

    def read_lines(self, path: Path) -> list[str]:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in self.line_cache:
            self.line_cache[key] = _read_lines_uncached(path)
            self.read_counts[key] = self.read_counts.get(key, 0) + 1
        return self.line_cache[key]


_CURRENT_PARSE_CONTEXT: ContextVar[ParseContext | None] = ContextVar(
    "sk_parse_context", default=None
)


@contextmanager
def use_parse_context(context: ParseContext):
    token = _CURRENT_PARSE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_PARSE_CONTEXT.reset(token)


def _read_lines_uncached(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()


def _read_lines(path: Path) -> list[str]:
    context = _CURRENT_PARSE_CONTEXT.get()
    if context is not None:
        return context.read_lines(path)
    return _read_lines_uncached(path)


def _pick_latest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    existing.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return existing[0]


def _resolve_input_dir(input_path: str | os.PathLike[str]) -> Path:
    path = Path(input_path).resolve()
    if path.is_file():
        return path.parent
    return path


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "super_kernel.log").is_file()


def _dedupe_order_model_dirs(candidates: list[Path]) -> list[Path]:
    unique_existing: dict[str, Path] = {}
    for candidate in candidates:
        if candidate.exists() and _is_model_dir(candidate):
            resolved = candidate.resolve()
            unique_existing[str(resolved)] = resolved
    ordered = list(unique_existing.values())
    ordered.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return ordered


def _direct_child_model_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return _dedupe_order_model_dirs([candidate for candidate in path.glob("model_*")])


def _structured_model_dir_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(_direct_child_model_dirs(path))

    for sk_meta_root in (path / "sk_meta", path / "runs" / "sk_meta"):
        if sk_meta_root.is_dir():
            for process_dir in sk_meta_root.iterdir():
                candidates.extend(_direct_child_model_dirs(process_dir))

    if path.name == "sk_meta":
        for process_dir in path.iterdir():
            candidates.extend(_direct_child_model_dirs(process_dir))

    return _dedupe_order_model_dirs(candidates)


def _fallback_model_dir_candidates(path: Path) -> list[Path]:
    ignored_names = {"reports", ".cache", "__pycache__"}
    candidates: list[Path] = []
    for root, dir_names, _file_names in os.walk(path):
        dir_names[:] = [
            name
            for name in dir_names
            if name not in ignored_names and not name.startswith(".")
        ]
        root_path = Path(root)
        for name in dir_names:
            if name.startswith("model_"):
                candidates.append(root_path / name)
    return _dedupe_order_model_dirs(candidates)


def find_model_dirs(input_path: str | os.PathLike[str]) -> list[Path]:
    path = _resolve_input_dir(input_path)
    if _is_model_dir(path):
        return [path]
    structured = _structured_model_dir_candidates(path)
    if structured:
        return structured
    return _fallback_model_dir_candidates(path)


def _resolve_node_library_source(model_dir: Path) -> tuple[Path, str]:
    primary = model_dir / "sk_node_detail.log"
    fallback = model_dir / "super_kernel.log"
    if primary.is_file():
        return primary, "sk_node_detail.log"
    return fallback, "super_kernel.log"


def infer_model_asset_root(
    input_path: str | os.PathLike[str],
    model_dirs: list[Path] | None = None,
) -> Path | None:
    input_dir = _resolve_input_dir(input_path)
    resolved_model_dirs = []
    for item in model_dirs or find_model_dirs(input_dir):
        resolved_model_dirs.append(item.resolve())
    if not resolved_model_dirs:
        return None

    common = Path(os.path.commonpath([str(item) for item in resolved_model_dirs]))
    if common.name.startswith("model_"):
        common = common.parent
    if input_dir == common:
        return common
    if all(item.parent == common for item in resolved_model_dirs):
        return common.parent if common.parent != common else common
    return common


def infer_result_root(
    input_path: str | os.PathLike[str],
    model_asset_root: Path | None = None,
    model_dirs: list[Path] | None = None,
) -> Path:
    input_dir = _resolve_input_dir(input_path)
    markers = ("log", "kernel_meta", "reports")
    if any((input_dir / marker).exists() for marker in markers):
        return input_dir
    if model_asset_root is not None and model_asset_root.parent != model_asset_root:
        return model_asset_root.parent
    resolved_model_dirs = []
    for item in model_dirs or find_model_dirs(input_dir):
        resolved_model_dirs.append(item.resolve())
    if resolved_model_dirs:
        first_model_dir = resolved_model_dirs[0]
        parent = first_model_dir.parent.parent
        if parent != first_model_dir.parent:
            return parent
    return input_dir


T = TypeVar("T")


def _emit(message: object = "", *, file: Any = None, end: str = "\n") -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")


def _partition_consecutive(items: list[T], count: int) -> list[list[T]]:
    if count <= 0:
        return []
    if not items:
        return [[] for _ in range(count)]
    base, remainder = divmod(len(items), count)
    groups: list[list[T]] = []
    index = 0
    for group_index in range(count):
        size = base + (1 if group_index < remainder else 0)
        end = index + size
        groups.append(items[index:end])
        index += size
    return groups


def _read_indexed_lines(
    path: Path, line_range: tuple[int, int] | None = None
) -> list[tuple[int, str]]:
    context = _CURRENT_PARSE_CONTEXT.get()
    if line_range and context is None:
        begin_line, end_line = line_range
        begin = max(1, begin_line)
        end = max(begin, end_line)
        if not path.is_file():
            return []
        selected: list[tuple[int, str]] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                if line_no < begin:
                    continue
                if line_no > end:
                    break
                selected.append((line_no, line))
        return selected
    lines = _read_lines(path)
    if not line_range:
        return list(enumerate(lines, start=1))
    begin_line, end_line = line_range
    begin = max(1, begin_line)
    end = max(begin, end_line)
    start = begin - 1
    sliced = lines[start:end]
    return [(begin + index, line) for index, line in enumerate(sliced)]


def _extract_line_timestamp(line: str) -> str | None:
    match = RE_LINE_TIMESTAMP.search(line)
    if not match:
        return None
    value = match.group("bracket_ts") or match.group("plain_ts")
    return value.replace(",", ".") if value else None


def _is_terminal_final_pass(batch: dict[str, Any]) -> bool:
    pass_name = str(batch.get("pass") or "").strip().lower()
    detail = str(batch.get("detail") or "").strip().lower()
    return "final" in pass_name or "final" in detail


def _fallback_node_key(
    node_id: int, stream_idx_in_graph: int, node_idx_in_stream: int
) -> tuple[Any, ...]:
    return ("fallback", node_id, stream_idx_in_graph, node_idx_in_stream)


def _full_node_key(
    node_id: int,
    stream_id: int | None,
    stream_idx_in_graph: int,
    node_idx_in_stream: int,
) -> tuple[Any, ...] | None:
    if not isinstance(stream_id, int) or stream_id < 0:
        return None
    return ("full", node_id, stream_id, stream_idx_in_graph, node_idx_in_stream)


def _expected_node_keys(
    node: dict[str, Any],
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...]]:
    node_id = int(node.get("node_id"))
    stream_idx_in_graph = int(node.get("stream_idx_in_graph"))
    node_idx_in_stream = int(node.get("node_idx_in_stream"))
    stream_id = node.get("stream_id")
    return (
        _full_node_key(
            node_id,
            stream_id if isinstance(stream_id, int) else None,
            stream_idx_in_graph,
            node_idx_in_stream,
        ),
        _fallback_node_key(node_id, stream_idx_in_graph, node_idx_in_stream),
    )


def _entry_node_keys(
    entry: dict[str, Any],
) -> tuple[tuple[Any, ...] | None, tuple[Any, ...]]:
    return (
        _full_node_key(
            int(entry.get("node_id")),
            entry.get("stream_id") if isinstance(entry.get("stream_id"), int) else None,
            int(entry.get("stream_idx_in_graph")),
            int(entry.get("node_idx_in_stream")),
        ),
        _fallback_node_key(
            int(entry.get("node_id")),
            int(entry.get("stream_idx_in_graph")),
            int(entry.get("node_idx_in_stream")),
        ),
    )


def _group_signature_key(node: dict[str, Any]) -> tuple[Any, ...]:
    return _fallback_node_key(
        int(node.get("node_id")),
        int(node.get("stream_idx_in_graph")),
        int(node.get("node_idx_in_stream")),
    )


def _node_key_sample(nodes: list[dict[str, Any]], limit: int = 10) -> list[str]:
    sample: list[str] = []
    for node in nodes[:limit]:
        primary, fallback = _expected_node_keys(node)
        sample.append(str(primary or fallback))
    return sample


def _signature_sample(
    keys: list[tuple[Any, ...]] | frozenset[tuple[Any, ...]], limit: int = 10
) -> list[str]:
    ordered = list(keys)
    ordered.sort(key=str)
    return [str(item) for item in ordered[:limit]]


def _timestamp_outside_range(
    timestamp: str | None, start: str | None, end: str | None
) -> bool:
    if not timestamp or not start or not end:
        return False
    return timestamp < start or timestamp > end


def _normalize_model_instance_id(value: Any, *, default: str = "mi01") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    match = re.match(r"^mi(\d+)$", text)
    if match:
        return f"mi{int(match.group(1)):02d}"
    return text


def _model_instance_id(payload: dict[str, Any] | None, *, default: str = "mi01") -> str:
    if not isinstance(payload, dict):
        return default
    return _normalize_model_instance_id(
        payload.get("model_instance_id"), default=default
    )


def _model_instance_index(payload: dict[str, Any] | None, *, default: int = 1) -> int:
    if not isinstance(payload, dict):
        return default
    return int(payload.get("model_instance_index") or default)


def _model_instance_count(payload: dict[str, Any] | None, *, default: int = 1) -> int:
    if not isinstance(payload, dict):
        return default
    return int(payload.get("model_instance_count") or default)


def _model_instance_range(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    model_instance_range = payload.get("model_instance_range")
    return model_instance_range if isinstance(model_instance_range, dict) else None


def _normalize_model_instance_partition_mode(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text


def _model_instance_partition_mode(
    stats: dict[str, Any] | None, *, default: str = "unknown"
) -> str:
    if not isinstance(stats, dict):
        return default
    return _normalize_model_instance_partition_mode(
        stats.get("model_instance_partition_mode") or default
    )


def _model_instance_partition_verified(
    stats: dict[str, Any] | None, *, default: bool = False
) -> bool:
    if not isinstance(stats, dict):
        return default
    return bool(stats.get("model_instance_partition_verified", default))


def _parser_log(
    level: str,
    code: str,
    *,
    model_dir: Path | None = None,
    model_instance_id: str | None = None,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    parts = [f"[PARSE][{level}]"]
    if code:
        parts.append(code)
    if model_dir is not None:
        parts.append(f"model_dir={model_dir}")
    if model_instance_id:
        parts.append(f"model_instance={model_instance_id}")
    parts.append(message)
    _emit(" ".join(str(part) for part in parts if part), file=sys.stderr)
    if details:
        _emit(
            "  details="
            + json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in details.items()
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )


def _scope_graph_alignment_diagnostics(
    scope_library: dict[str, Any], graph_library: dict[str, Any]
) -> dict[str, Any]:
    scopes = scope_library.get("scopes", []) if isinstance(scope_library, dict) else []
    node_library = (
        graph_library.get("node_library", {}) if isinstance(graph_library, dict) else {}
    )
    node_rows = node_library.get("nodes", []) if isinstance(node_library, dict) else []
    node_ids_in_library = {
        node.get("node_id")
        for node in node_rows
        if isinstance(node.get("node_id"), int)
    }
    scope_node_ids: set[int] = set()
    empty_scope_ids: list[int] = []
    partial_scope_ids: list[int] = []
    per_scope: list[dict[str, Any]] = []
    for scope in scopes:
        scope_id = scope.get("scope_id")
        node_ids = []
        for node_id in scope.get("node_ids", []):
            if isinstance(node_id, int):
                node_ids.append(node_id)
        scope_node_ids.update(node_ids)
        matched_count = sum(1 for node_id in node_ids if node_id in node_ids_in_library)
        missing_count = len(node_ids) - matched_count
        if node_ids and matched_count == 0:
            empty_scope_ids.append(scope_id)
        elif missing_count > 0:
            partial_scope_ids.append(scope_id)
        per_scope.append(
            {
                "scope_id": scope_id,
                "node_id_count": len(node_ids),
                "matched_node_count": matched_count,
                "missing_node_count": missing_count,
            }
        )
    missing_node_ids = sorted(scope_node_ids - node_ids_in_library)
    node_update_rows = (
        graph_library.get("node_update_registry", {}).get("rows", [])
        if isinstance(graph_library, dict)
        else []
    )
    node_library_stats = (
        node_library.get("stats", {}) if isinstance(node_library, dict) else {}
    )
    return {
        "scope_count": len(scopes),
        "scope_node_count": len(scope_node_ids),
        "node_library_count": len(node_ids_in_library),
        "node_update_row_count": len(node_update_rows)
        if isinstance(node_update_rows, list)
        else 0,
        "missing_node_id_count": len(missing_node_ids),
        "missing_node_ids_sample": missing_node_ids[:20],
        "empty_scope_ids": empty_scope_ids,
        "partial_scope_ids": partial_scope_ids,
        "model_instance_partition_mode": _model_instance_partition_mode(
            node_library_stats
        ),
        "model_instance_partition_verified": _model_instance_partition_verified(
            node_library_stats
        ),
        "per_scope": per_scope,
        "node_library_stats": node_library_stats,
    }


def _validate_model_instance_report(
    model_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    model_instance_id = _model_instance_id(report)
    scope_library = (
        report.get("scope_library", {})
        if isinstance(report.get("scope_library"), dict)
        else {}
    )
    graph_library = (
        report.get("graph_library", {})
        if isinstance(report.get("graph_library"), dict)
        else {}
    )
    diagnostics = _scope_graph_alignment_diagnostics(scope_library, graph_library)
    graph_library["parser_diagnostics"] = diagnostics

    _parser_log(
        "INFO",
        "scope_graph_alignment",
        model_dir=model_path,
        model_instance_id=model_instance_id,
        message="collected scope/graph consistency snapshot",
        details={
            "scope_count": diagnostics["scope_count"],
            "scope_node_count": diagnostics["scope_node_count"],
            "node_library_count": diagnostics["node_library_count"],
            "node_update_row_count": diagnostics["node_update_row_count"],
            "missing_node_id_count": diagnostics["missing_node_id_count"],
            "empty_scope_ids": diagnostics["empty_scope_ids"][:10],
            "partial_scope_ids": diagnostics["partial_scope_ids"][:10],
        },
    )

    node_library_stats = diagnostics.get("node_library_stats", {})
    model_instance_count = _model_instance_count(report)
    partition_mode = _model_instance_partition_mode(node_library_stats)
    partition_verified = _model_instance_partition_verified(node_library_stats)
    if model_instance_count > 1 and (
        partition_mode
        not in {
            "model_instance_line_range",
            "model_instance_timestamp_range",
            "model_instance_partitioned",
        }
        or not partition_verified
    ):
        raise ValueError(
            "node_library is not safely model-instance partitioned under multi-model-instance parsing"
        )

    if diagnostics["scope_node_count"] > 0 and diagnostics["node_library_count"] == 0:
        _parser_log(
            "WARN",
            "scope_node_library_missing",
            model_dir=model_path,
            model_instance_id=model_instance_id,
            message=(
                "scope library contains node ids, but graph node library is empty; "
                "scope graph rendering may be unavailable"
            ),
            details={
                "scope_count": diagnostics["scope_count"],
                "scope_node_count": diagnostics["scope_node_count"],
            },
        )

    if diagnostics["empty_scope_ids"] and diagnostics["node_library_count"] > 0:
        empty_scope_id_sample = []
        for scope_id in diagnostics["empty_scope_ids"][:10]:
            empty_scope_id_sample.append(str(scope_id))
        raise ValueError(
            "some scopes contain node ids but none of them can be matched in graph node library: "
            + ",".join(empty_scope_id_sample)
        )

    if diagnostics["missing_node_id_count"] > 0:
        _parser_log(
            "WARN",
            "scope_node_partial_mismatch",
            model_dir=model_path,
            model_instance_id=model_instance_id,
            message="some scope node ids cannot be found in graph node library",
            details={
                "missing_node_id_count": diagnostics["missing_node_id_count"],
                "missing_node_ids_sample": diagnostics["missing_node_ids_sample"],
                "partial_scope_ids": diagnostics["partial_scope_ids"][:10],
            },
        )

    if (
        diagnostics["scope_node_count"] > 0
        and diagnostics["node_update_row_count"] == 0
    ):
        _parser_log(
            "WARN",
            "node_update_registry_empty",
            model_dir=model_path,
            model_instance_id=model_instance_id,
            message=(
                "node_update_registry is empty; scope nodes can still render, "
                "but update annotations will be unavailable"
            ),
            details={
                "scope_node_count": diagnostics["scope_node_count"],
                "scope_count": diagnostics["scope_count"],
            },
        )

    return report


def detect_model_instances(model_dir: Path) -> list[dict[str, Any]]:
    log_path = model_dir / "super_kernel.log"
    indexed_lines = _read_indexed_lines(log_path)
    line_count = indexed_lines[-1][0] if indexed_lines else 0
    model_instances: list[dict[str, Any]] = []
    current_begin: int | None = None
    current_begin_timestamp: str | None = None

    for line_no, line in indexed_lines:
        if "Begin aclskOptimize" in line:
            if current_begin is not None:
                model_instance_index = len(model_instances) + 1
                model_instances.append(
                    {
                        "model_instance_index": model_instance_index,
                        "model_instance_id": f"mi{model_instance_index:02d}",
                        "begin_line": current_begin,
                        "end_line": max(current_begin, line_no - 1),
                        "begin_timestamp": current_begin_timestamp,
                        "end_timestamp": None,
                    }
                )
            current_begin = line_no
            current_begin_timestamp = _extract_line_timestamp(line)
            continue
        if "End aclskOptimize" in line and current_begin is not None:
            model_instance_index = len(model_instances) + 1
            model_instances.append(
                {
                    "model_instance_index": model_instance_index,
                    "model_instance_id": f"mi{model_instance_index:02d}",
                    "begin_line": current_begin,
                    "end_line": line_no,
                    "begin_timestamp": current_begin_timestamp,
                    "end_timestamp": _extract_line_timestamp(line),
                }
            )
            current_begin = None
            current_begin_timestamp = None

    if current_begin is not None:
        model_instance_index = len(model_instances) + 1
        model_instances.append(
            {
                "model_instance_index": model_instance_index,
                "model_instance_id": f"mi{model_instance_index:02d}",
                "begin_line": current_begin,
                "end_line": line_count or current_begin,
                "begin_timestamp": current_begin_timestamp,
                "end_timestamp": None,
            }
        )

    if not model_instances:
        model_instances.append(
            {
                "model_instance_index": 1,
                "model_instance_id": "mi01",
                "begin_line": 1,
                "end_line": line_count or 1,
                "begin_timestamp": None,
                "end_timestamp": None,
            }
        )
    return model_instances


def find_model_dir(input_path: str | os.PathLike[str]) -> Path:
    model_dirs = find_model_dirs(input_path)
    if not model_dirs:
        raise FileNotFoundError(
            f"Unable to find model directory under: {Path(input_path).resolve()}"
        )
    return model_dirs[0]


def write_update_libraries(
    report: dict[str, Any],
    data_dir: Path,
    scope_library_output: Path | None = None,
    graph_library_output: Path | None = None,
    dfx_library_output: Path | None = None,
) -> tuple[Path, Path, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    scope_library_output = scope_library_output or (data_dir / "scope-library.json")
    graph_library_output = graph_library_output or (data_dir / "graph-library.json")
    dfx_library_output = dfx_library_output or (data_dir / "dfx-library.json")
    scope_library_output = scope_library_output.resolve()
    graph_library_output = graph_library_output.resolve()
    dfx_library_output = dfx_library_output.resolve()
    scope_library_output.parent.mkdir(parents=True, exist_ok=True)
    graph_library_output.parent.mkdir(parents=True, exist_ok=True)
    dfx_library_output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(report.get("dfx_library"), dict):
        report["dfx_library"]["path"] = str(dfx_library_output)

    scope_library_payload = _annotate_scope_library_for_output(
        report.get("scope_library", {})
    )
    graph_library_payload = _annotate_graph_library_for_output(
        report.get("graph_library", {})
    )
    dfx_library_payload = _annotate_dfx_library_for_output(
        report.get("dfx_library", {})
    )
    fusion_analysis_payload = report.get("fusion_analysis")
    if isinstance(dfx_library_payload, dict) and isinstance(
        fusion_analysis_payload, dict
    ):
        # Keep this optional evidence beside runtime DFX data in the persisted
        # artifact while retaining the sibling field in the in-memory report.
        dfx_library_payload["fusion_analysis"] = copy.deepcopy(fusion_analysis_payload)

    scope_library_output.write_text(
        json.dumps(scope_library_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    graph_library_output.write_text(
        json.dumps(graph_library_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dfx_library_output.write_text(
        json.dumps(dfx_library_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return scope_library_output, graph_library_output, dfx_library_output


def _annotate_scope_library_for_output(scope_library: Any) -> Any:
    if not isinstance(scope_library, dict):
        return scope_library
    payload = copy.deepcopy(scope_library)
    payload["$comment"] = (
        "Scope library is the primary interpretation layer for final scopes, fused functions, "
        "device-task sections, and scope-local update payloads."
    )
    payload["$field_comments"] = {
        "synthesized_custom_binding_diagnostics": (
            "Binding summary for synthesized custom nodes. Inspect this block when a custom node "
            "is present in device tasks but missing from a scope update view."
        ),
        "fused_library": "Function-level grouping used to correlate scopes, fused functions, and device sections.",
        "device_task_library": "Device-side task sections used for task-queue rendering and synthesized custom nodes.",
    }
    fused_library = payload.get("fused_library")
    if isinstance(fused_library, dict):
        fused_library["$field_comments"] = {
            "functions": (
                "Exported fused functions. node_details carry the structured rows used for signature "
                "matching and task correlation."
            ),
        }
    device_task_library = payload.get("device_task_library")
    if isinstance(device_task_library, dict):
        device_task_library["$field_comments"] = {
            "sections": "Device-task bundles grouped by function or scope context.",
            "bound_scope_id": (
                "Resolved scope binding result for synthesized custom nodes. When missing, rely on "
                "synthesized_custom_binding_diagnostics instead of assuming a scope match."
            ),
        }
    scopes = payload.get("scopes")
    if isinstance(scopes, list):
        for scope in scopes:
            if not isinstance(scope, dict):
                continue
            scope["$field_comments"] = {
                "scope_export_ordinal": "Stable display order inside the current scope-library file.",
                "update": (
                    "Update-phase payload for this scope. Null means no update "
                    "or synthesized custom payload was observed."
                ),
            }
            update_payload = scope.get("update")
            if isinstance(update_payload, dict):
                update_payload["$comment"] = (
                    "Update payload mixes two domains: real graph-node updates and synthesized custom "
                    "nodes that only exist in task/update semantics."
                )
                update_payload["$field_comments"] = {
                    "graph_backed_updates": "Structured update rows for real graph nodes touched during update.",
                    "synthesized_custom_nodes": (
                        "Custom nodes that do not exist in graph membership and must be rendered as "
                        "update/task-side synthesized objects."
                    ),
                    "node_details": (
                        "Raw update detail lines. Besides text inspection, current scope-graph rendering still "
                        "uses these lines to recover explicit notify/wait/reset addrValue facts."
                    ),
                    "diagnostics": "Counts describing update completeness, fallback behavior, and binding quality.",
                }
    return payload


def _annotate_graph_library_for_output(graph_library: Any) -> Any:
    if not isinstance(graph_library, dict):
        return graph_library
    payload = copy.deepcopy(graph_library)
    payload["$comment"] = (
        "Graph library records graph-node membership and update-execution facts for one model or model-instance."
    )
    payload["$field_comments"] = {
        "node_library": "Graph-node membership view for the current model-instance.",
        "node_update_registry": "Structured update rows parsed from update execution.",
        "parser_diagnostics": (
            "Cross-check between scope-library and graph-library. Non-zero counts here usually "
            "mean partial alignment, not automatic hard failure."
        ),
    }
    node_library = payload.get("node_library")
    if isinstance(node_library, dict):
        stats = node_library.get("stats")
        if isinstance(stats, dict):
            for noisy_key in (
                "duplicate_node_ids_sample",
                "scope_node_missing_sample",
                "processed_node_extra_sample",
            ):
                stats.pop(noisy_key, None)
        node_library["$comment"] = (
            "Graph-backed node catalog. Synthesized custom nodes are intentionally excluded from this object."
        )
        if isinstance(stats, dict):
            stats["$field_comments"] = {
                "duplicate_node_id_count": (
                    "How many repeated nodeId rows were merged. Non-zero means the source logged the "
                    "same nodeId more than once inside the current extraction window."
                ),
                "model_instance_partition_mode": (
                    "How the current model-instance was partitioned, for example by line range or timestamp window."
                ),
                "scope_node_missing_count": (
                    "How many scope-declared nodes could not be found in the exported node library."
                ),
                "processed_node_extra_count": (
                    "How many extracted graph nodes are outside the current scope-node set. This is "
                    "the main signal that graph nodes are broader than final scope nodes."
                ),
                "membership_source_kind": "Which file actually decided graph-node membership.",
                "membership_partition_basis": (
                    "Which slicing basis was used for multi-instance partitioning, for example "
                    "timestamp_window or line_range."
                ),
                "detail_overlay_source_kind": (
                    "Which file, if any, was used only to enrich node rows after membership was fixed."
                ),
                "detail_overlay_missing_count": (
                    "How many graph-membership rows could not find a matching enrichment row."
                ),
            }
    node_update_registry = payload.get("node_update_registry")
    if isinstance(node_update_registry, dict):
        node_update_registry["$field_comments"] = {
            "rows": (
                "Structured graph-backed node update rows. Scope-level update boundaries are currently "
                "carried by scope_library.scopes[].update instead of this object."
            ),
        }
    parser_diagnostics = payload.get("parser_diagnostics")
    if isinstance(parser_diagnostics, dict):
        parser_diagnostics["$field_comments"] = {
            "missing_node_id_count": "How many scope node ids were not found in node_library.",
            "empty_scope_ids": "Scopes whose node ids could not be matched at all.",
            "partial_scope_ids": "Scopes that matched only partially.",
        }
    return payload


def _annotate_dfx_library_for_output(dfx_library: Any) -> Any:
    if not isinstance(dfx_library, dict):
        return dfx_library
    payload = copy.deepcopy(dfx_library)
    payload["$comment"] = (
        "DFX library records runtime-first debugging evidence: phase marks, payload facts, exception "
        "facts, counters, PC localization, and evidence summaries."
    )
    payload["$field_comments"] = {
        "payload_registry": "Device payload evidence correlated from runtime dumps and task sections.",
        "diagnostic_pc_registry": "Post-processed diagnostic PC events used by hang and crash analysis.",
        "evidence": "Human-facing confidence summary explaining why each DFX conclusion is considered trustworthy.",
        "fusion_analysis": "Optional fusion-boundary evidence. Direct boundary counts and same-region failure evidence are kept separate.",
    }
    return payload


def _extract_model_ri(model_dir: Path) -> str | None:
    match = RE_MODEL_RI.search(model_dir.name)
    return match.group(1) if match else None


def _normalize_scope_name(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "(none)"
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text or "(none)"


def _is_missing_scope_name(value: str | None) -> bool:
    text = (value or "").strip().lower()
    return text in {"", "(none)", "none", "-"}


def _extract_function_context(function_text: str) -> dict[str, Any]:
    text = str(function_text or "").strip()
    raw_scope_name = ""
    start_node_name = ""
    end_node_name = ""
    sk_id = None

    match = RE_FUNCTION_CONTEXT.search(text)
    if match:
        raw_scope_name = (match.group(1) or "").strip()
        sk_id = int(match.group(2))
        start_node_name = (match.group(3) or "").strip()
        end_node_name = (match.group(4) or "").strip()
    else:
        match = RE_SCOPE_NAME.search(text)
        if match:
            raw_scope_name = (match.group(1) or "").strip()
            sk_id = int(match.group(2))

    scope_name_text = ""
    if "scopeName:" in text:
        scope_name_text = text.split("scopeName:", 1)[1].strip()

    scope_name = _normalize_scope_name(scope_name_text or raw_scope_name)

    return {
        "function_text": text,
        "raw_scope_name": scope_name,
        "scope_name": scope_name,
        "sk_id": sk_id,
        "start_node_name": start_node_name,
        "end_node_name": end_node_name,
    }


def _decode_sync_args(raw_args: str) -> dict[str, Any]:
    text = str(raw_args or "").strip()
    value = None
    try:
        value = int(text, 16) if text.lower().startswith("0x") else int(text)
    except Exception:
        value = None

    sync_type = "UNKNOWN"
    sync_kind = "UNKNOWN"
    sync_direction = "UNKNOWN"
    if value is not None and value in SYNC_TYPE_MAP:
        sync_type, sync_kind, sync_direction = SYNC_TYPE_MAP[value]

    return {
        "sync_raw_args": text,
        "sync_value": value,
        "sync_type": sync_type,
        "sync_kind": sync_kind,
        "sync_direction": sync_direction,
    }


def _parse_loose_kv_fields(text: str | None) -> dict[str, str]:
    raw = (text or "").strip()
    if not raw:
        return {}
    result: dict[str, str] = {}
    for part in raw.split(","):
        piece = part.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_device_task_entry_line(
    line: str, active_queue: str
) -> dict[str, Any] | None:
    match = RE_TASK_ENTRY_HEAD.search(line)
    if not match:
        return None
    fields = _parse_loose_kv_fields(match.group(3))
    task_index = _safe_parse_int(fields.get("idx"))
    block_count = _safe_parse_int(fields.get("blk"))
    entry_count = _safe_parse_int(fields.get("entries"))
    missing_required_fields = (
        task_index is None or block_count is None or entry_count is None
    )
    if missing_required_fields or "args" not in fields:
        return None

    task_type = match.group(2)
    task = {
        "queue": active_queue,
        "task_id": int(match.group(1)),
        "task_index": task_index,
        "node_id": _safe_parse_int(fields.get("nodeId")),
        "task_type": task_type,
        "block_count": block_count,
        "entry_count": entry_count,
        "args": fields["args"],
        "entries": [],
        "detail": line.strip(),
    }
    if fields.get("debugOptions") is not None:
        task["debug_options"] = fields["debugOptions"]
    if fields.get("relatedType") is not None:
        task["related_type"] = fields["relatedType"]
    if fields.get("extraInfo") is not None:
        task["extra_info"] = fields["extraInfo"]
    if task_type == "SYNC":
        task.update(_decode_sync_args(fields["args"]))
    return task


def _safe_parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text)
    except Exception:
        return None


def _append_limited_sample(
    samples: list[dict[str, Any]], item: dict[str, Any], *, limit: int = 10
) -> None:
    if len(samples) < limit:
        samples.append(item)


def _read_json_payload(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return {"_parse_error": str(exc)}


def _normalize_compare_hex(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    parsed = _safe_parse_int(text)
    if parsed is not None:
        return hex(parsed)
    return text


def _normalize_compare_scalar(value: Any) -> Any:
    parsed = _safe_parse_int(value)
    return parsed if parsed is not None else str(value or "").strip()


def _normalize_compare_enum(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_compare_entries(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_normalize_compare_hex(item) for item in values if item is not None]


def _task_queue_json_by_scope_id(
    model_dir: Path,
) -> tuple[
    str | None, dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None
]:
    path = model_dir / "sk_task_queue.json"
    payload = _read_json_payload(path)
    if payload is None:
        return None, {}, [], None
    if isinstance(payload, dict) and payload.get("_parse_error"):
        return (
            str(path),
            {},
            [],
            {"status": "parse_error", "error": payload.get("_parse_error")},
        )
    if not isinstance(payload, dict):
        return str(path), {}, [], {"status": "invalid", "reason": "root_not_object"}
    scopes_by_id: dict[int, dict[str, Any]] = {}
    unscoped_payloads: list[dict[str, Any]] = []
    if isinstance(payload.get("taskQueues"), dict):
        scope_id = _safe_parse_int(payload.get("scopeId"))
        if scope_id is not None:
            scopes_by_id[scope_id] = payload
        else:
            unscoped_payloads.append(payload)
    for scope in (
        payload.get("scopes", []) if isinstance(payload.get("scopes"), list) else []
    ):
        if not isinstance(scope, dict):
            continue
        scope_id = _safe_parse_int(scope.get("scopeId"))
        if scope_id is None:
            continue
        scopes_by_id[scope_id] = scope
    return str(path), scopes_by_id, unscoped_payloads, None


def _json_queue_tasks(
    scope_payload: dict[str, Any], queue_name: str
) -> list[dict[str, Any]]:
    queues = scope_payload.get("taskQueues", {})
    if not isinstance(queues, dict):
        return []
    queue_payload = queues.get(queue_name.lower(), {})
    if not isinstance(queue_payload, dict):
        return []
    task_que = queue_payload.get("taskQue", {})
    if not isinstance(task_que, dict):
        return []
    task_infos = task_que.get("taskInfos", [])
    return (
        [item for item in task_infos if isinstance(item, dict)]
        if isinstance(task_infos, list)
        else []
    )


def _compare_task_queue_json(
    section: dict[str, Any],
    json_scope: dict[str, Any],
    section_index: int,
    samples: list[dict[str, Any]],
) -> int:
    mismatch_count = 0

    def _record(
        queue: str, task_index: int | None, field: str, log_value: Any, json_value: Any
    ) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        _append_limited_sample(
            samples,
            {
                "section_index": section_index,
                "scope_id": section.get("sk_id"),
                "queue": queue,
                "task_index": task_index,
                "field": field,
                "log_value": log_value,
                "json_value": json_value,
            },
        )

    for queue_name in ("AIC", "AIV"):
        log_tasks = section.get("queues", {}).get(queue_name, []) or []
        json_tasks = _json_queue_tasks(json_scope, queue_name)
        if len(log_tasks) != len(json_tasks):
            _record(queue_name, None, "task_count", len(log_tasks), len(json_tasks))
        for task_ordinal, (log_task, json_task) in enumerate(
            zip(log_tasks, json_tasks)
        ):
            scalar_pairs = [
                (
                    "block_count",
                    log_task.get("block_count"),
                    json_task.get("numBlocks"),
                ),
                ("entry_count", log_task.get("entry_count"), json_task.get("entryCnt")),
            ]
            if _task_json_node_index_comparable(log_task):
                scalar_pairs.append(
                    (
                        "task_index",
                        log_task.get("task_index"),
                        json_task.get("nodeIndex"),
                    )
                )
            for field, log_value, json_value in scalar_pairs:
                if _normalize_compare_scalar(log_value) != _normalize_compare_scalar(
                    json_value
                ):
                    _record(queue_name, task_ordinal, field, log_value, json_value)
            if _normalize_compare_enum(
                log_task.get("task_type")
            ) != _normalize_compare_enum(json_task.get("type")):
                _record(
                    queue_name,
                    task_ordinal,
                    "task_type",
                    log_task.get("task_type"),
                    json_task.get("type"),
                )
            if _normalize_compare_hex(log_task.get("args")) != _normalize_compare_hex(
                json_task.get("args")
            ):
                _record(
                    queue_name,
                    task_ordinal,
                    "args",
                    log_task.get("args"),
                    json_task.get("args"),
                )
            if "debug_options" in log_task:
                if _normalize_compare_hex(
                    log_task.get("debug_options")
                ) != _normalize_compare_hex(json_task.get("debugOptions")):
                    _record(
                        queue_name,
                        task_ordinal,
                        "debug_options",
                        log_task.get("debug_options"),
                        json_task.get("debugOptions"),
                    )
            log_entries = _normalize_compare_entries(log_task.get("entries", []))
            json_entries = _normalize_compare_entries(json_task.get("entries", []))
            if log_entries != json_entries:
                _record(
                    queue_name,
                    task_ordinal,
                    "entries",
                    log_task.get("entries", []),
                    json_task.get("entries", []),
                )
    return mismatch_count


def _queue_task_counts(section: dict[str, Any]) -> dict[str, int]:
    queues = section.get("queues", {}) if isinstance(section, dict) else {}
    return {
        queue_name: len(queues.get(queue_name, []) or [])
        if isinstance(queues, dict)
        else 0
        for queue_name in ("AIC", "AIV")
    }


def _json_queue_task_counts(json_scope: dict[str, Any]) -> dict[str, int]:
    return {
        queue_name: len(_json_queue_tasks(json_scope, queue_name))
        for queue_name in ("AIC", "AIV")
    }


def _task_json_node_index_comparable(log_task: dict[str, Any]) -> bool:
    return (
        str(log_task.get("task_type") or "").strip().upper() in GRAPH_BOUND_TASK_TYPES
    )


def _match_unscoped_task_queue_json(
    sections: list[dict[str, Any]],
    unscoped_payload: dict[str, Any],
    excluded_section_indexes: set[int],
    samples: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    json_counts = _json_queue_task_counts(unscoped_payload)
    candidates = []
    for index, section in enumerate(sections):
        if index in excluded_section_indexes:
            continue
        if _queue_task_counts(section) == json_counts:
            candidates.append((index, section))
    if len(candidates) != 1:
        return None, None
    section_index, section = candidates[0]
    mismatch_count = _compare_task_queue_json(
        section, unscoped_payload, section_index, samples
    )
    return section_index, mismatch_count


def _validate_device_task_library_with_json(
    model_dir: Path, sections: list[dict[str, Any]]
) -> dict[str, Any]:
    json_path, scopes_by_id, unscoped_payloads, invalid = _task_queue_json_by_scope_id(
        model_dir
    )
    if json_path is None:
        return {
            "status": "missing",
            "path": None,
            "matched_scope_count": 0,
            "matched_unscoped_count": 0,
            "ambiguous_unscoped_count": 0,
            "mismatch_count": 0,
            "missing_in_log": [],
            "missing_in_json": [],
            "mismatch_samples": [],
        }
    if invalid is not None:
        return {
            **invalid,
            "path": json_path,
            "matched_scope_count": 0,
            "matched_unscoped_count": 0,
            "ambiguous_unscoped_count": 0,
            "mismatch_count": 0,
            "missing_in_log": [],
            "missing_in_json": [],
            "mismatch_samples": [],
        }

    sections_by_scope_id: dict[int, tuple[int, dict[str, Any]]] = {}
    for index, section in enumerate(sections):
        scope_id = _safe_parse_int(section.get("sk_id"))
        if scope_id is not None:
            sections_by_scope_id[scope_id] = (index, section)
    samples: list[dict[str, Any]] = []
    mismatch_count = 0
    matched_scope_ids = sorted(set(sections_by_scope_id) & set(scopes_by_id))
    matched_section_indexes: set[int] = set()
    for scope_id in matched_scope_ids:
        section_index, section = sections_by_scope_id[scope_id]
        matched_section_indexes.add(section_index)
        mismatch_count += _compare_task_queue_json(
            section, scopes_by_id[scope_id], section_index, samples
        )

    matched_unscoped_count = 0
    ambiguous_unscoped_count = 0
    for unscoped_payload in unscoped_payloads:
        section_index, unscoped_mismatch_count = _match_unscoped_task_queue_json(
            sections,
            unscoped_payload,
            matched_section_indexes,
            samples,
        )
        if section_index is None or unscoped_mismatch_count is None:
            ambiguous_unscoped_count += 1
            continue
        matched_unscoped_count += 1
        matched_section_indexes.add(section_index)
        mismatch_count += unscoped_mismatch_count

    missing_in_log = []
    for scope_id in scopes_by_id:
        if scope_id not in sections_by_scope_id:
            missing_in_log.append(scope_id)
    missing_in_log.sort()
    missing_scope_ids = set()
    for section_index, section in enumerate(sections):
        scope_id = _safe_parse_int(section.get("sk_id"))
        if section_index not in matched_section_indexes and scope_id is not None:
            missing_scope_ids.add(scope_id)
    missing_in_json = sorted(missing_scope_ids)
    status = "passed"
    if mismatch_count:
        status = "mismatch"
    elif missing_in_log or missing_in_json or ambiguous_unscoped_count:
        status = "partial"
    return {
        "status": status,
        "path": json_path,
        "matched_scope_count": len(matched_scope_ids),
        "matched_unscoped_count": matched_unscoped_count,
        "ambiguous_unscoped_count": ambiguous_unscoped_count,
        "mismatch_count": mismatch_count,
        "missing_in_log": missing_in_log,
        "missing_in_json": missing_in_json,
        "mismatch_samples": samples,
        "unchecked_fields": [
            "nodeId",
            "relatedType",
            "extraInfo",
            "header_info",
            "dfx",
        ],
    }


def _match_device_section_to_fused_function(
    section: dict[str, Any],
    fused_functions: list[dict[str, Any]],
    section_index: int,
) -> dict[str, Any] | None:
    function_text = str(section.get("function_text") or "").strip()
    if function_text:
        for fused in fused_functions:
            if str(fused.get("function_text") or "").strip() == function_text:
                return fused

    scope_name = str(section.get("scope_name") or "").strip()
    sk_id = section.get("sk_id")
    if scope_name or sk_id is not None:
        for fused in fused_functions:
            if scope_name and str(fused.get("scope_name") or "").strip() != scope_name:
                continue
            if sk_id is not None and fused.get("sk_id") != sk_id:
                continue
            return fused

    if 0 <= section_index < len(fused_functions):
        return fused_functions[section_index]
    return None


def _task_identity_sample(
    *,
    section: dict[str, Any],
    queue_name: str,
    task: dict[str, Any],
    reason: str,
    resolved_graph_node_id: int | None = None,
    identity_kind: str | None = None,
    custom_instance_key: str | None = None,
) -> dict[str, Any]:
    return {
        "scope_name": section.get("scope_name"),
        "sk_id": section.get("sk_id"),
        "queue": queue_name,
        "task_id": task.get("task_id"),
        "task_type": task.get("task_type"),
        "task_index": task.get("task_index"),
        "node_id": task.get("node_id"),
        "resolved_graph_node_id": resolved_graph_node_id,
        "identity_kind": identity_kind,
        "custom_instance_key": custom_instance_key,
        "reason": reason,
    }


def _graph_node_identity_payload(
    node: dict[str, Any] | None, fallback_node_id: int | None = None
) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        if not isinstance(fallback_node_id, int):
            return None
        return {"node_id": fallback_node_id}
    node_id = node.get("node_id")
    if not isinstance(node_id, int):
        node_id = fallback_node_id
    if not isinstance(node_id, int):
        return None
    payload: dict[str, Any] = {"node_id": node_id}
    if isinstance(node.get("stream_idx_in_graph"), int):
        payload["stream_idx_in_graph"] = int(node.get("stream_idx_in_graph"))
    if isinstance(node.get("node_idx_in_stream"), int):
        payload["node_idx_in_stream"] = int(node.get("node_idx_in_stream"))
    if isinstance(node.get("stream_id"), int):
        payload["stream_id"] = int(node.get("stream_id"))
    return payload


def _event_node_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in EVENT_TASK_TO_NODE_TYPE:
        return EVENT_TASK_TO_NODE_TYPE[text]
    if text in {"EVENTNOTIFY", "EVENT_NOTIFY"}:
        return "EVENT_NOTIFY"
    if text in {"EVENTWAIT", "EVENT_WAIT"}:
        return "EVENT_WAIT"
    if text in {"EVENTRESET", "EVENT_RESET"}:
        return "EVENT_RESET"
    return ""


def _resolve_task_graph_identity(
    task: dict[str, Any],
    *,
    scope_node_ids: set[int],
    node_ids_in_library: set[int],
    ordinal_node_map: dict[int, int],
    node_identity_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    task_type = str(task.get("task_type") or "").upper()
    raw_task_index = _safe_parse_int(task.get("task_index"))
    raw_node_id = _safe_parse_int(task.get("node_id"))

    def _valid_node_id(candidate: int | None) -> bool:
        return (
            candidate is not None
            and candidate in scope_node_ids
            and candidate in node_ids_in_library
        )

    if task_type in GRAPH_BOUND_TASK_TYPES and _valid_node_id(raw_node_id):
        return {
            "identity_kind": "graph_node",
            "identity_status": "resolved",
            "raw_task_index": raw_task_index,
            "raw_node_id": raw_node_id,
            "resolved_graph_node_id": raw_node_id,
            "graph_node_key": _graph_node_identity_payload(
                node_identity_by_id.get(raw_node_id), raw_node_id
            ),
            "graph_identity_valid": True,
            "graph_identity_source": "explicit_node_id",
            "graph_identity_reason": "matched_explicit_node_id",
            "exclude_from_duplicate_check": False,
            "custom_task_candidate": False,
            "custom_instance_key": None,
        }

    if task_type in EVENT_TASK_TYPES and _valid_node_id(raw_node_id):
        node_identity = node_identity_by_id.get(raw_node_id)
        task_event_type = _event_node_type(task_type)
        graph_event_type = _event_node_type(
            node_identity.get("node_type") if isinstance(node_identity, dict) else ""
        )
        if task_event_type and graph_event_type == task_event_type:
            return {
                "identity_kind": "graph_node",
                "identity_status": "resolved",
                "raw_task_index": raw_task_index,
                "raw_node_id": raw_node_id,
                "resolved_graph_node_id": raw_node_id,
                "graph_node_key": _graph_node_identity_payload(
                    node_identity_by_id.get(raw_node_id), raw_node_id
                ),
                "graph_identity_valid": True,
                "graph_identity_source": "event_explicit_node_id",
                "graph_identity_reason": "matched_event_explicit_node_id",
                "exclude_from_duplicate_check": False,
                "custom_task_candidate": False,
                "custom_instance_key": None,
            }

    if task_type in GRAPH_BOUND_TASK_TYPES and raw_task_index is not None:
        ordinal_node_id = ordinal_node_map.get(raw_task_index)
        if _valid_node_id(ordinal_node_id):
            return {
                "identity_kind": "graph_node",
                "identity_status": "resolved",
                "raw_task_index": raw_task_index,
                "raw_node_id": raw_node_id,
                "resolved_graph_node_id": ordinal_node_id,
                "graph_node_key": _graph_node_identity_payload(
                    node_identity_by_id.get(ordinal_node_id), ordinal_node_id
                ),
                "graph_identity_valid": True,
                "graph_identity_source": "fused_ordinal",
                "graph_identity_reason": "matched_fused_ordinal",
                "exclude_from_duplicate_check": False,
                "custom_task_candidate": False,
                "custom_instance_key": None,
            }

    if task_type in EVENT_TASK_TYPES and raw_task_index is not None:
        ordinal_node_id = ordinal_node_map.get(raw_task_index)
        if _valid_node_id(ordinal_node_id):
            node_identity = node_identity_by_id.get(ordinal_node_id)
            task_event_type = _event_node_type(task_type)
            graph_event_type = _event_node_type(
                node_identity.get("node_type")
                if isinstance(node_identity, dict)
                else ""
            )
            if task_event_type and graph_event_type == task_event_type:
                return {
                    "identity_kind": "graph_node",
                    "identity_status": "resolved",
                    "raw_task_index": raw_task_index,
                    "raw_node_id": raw_node_id,
                    "resolved_graph_node_id": ordinal_node_id,
                    "graph_node_key": _graph_node_identity_payload(
                        node_identity, ordinal_node_id
                    ),
                    "graph_identity_valid": True,
                    "graph_identity_source": "event_fused_ordinal",
                    "graph_identity_reason": "matched_event_fused_ordinal",
                    "exclude_from_duplicate_check": False,
                    "custom_task_candidate": False,
                    "custom_instance_key": None,
                }

    if raw_node_id is not None and raw_node_id not in node_ids_in_library:
        reason = "node_id_not_in_graph_library"
    elif raw_node_id is not None and raw_node_id not in scope_node_ids:
        reason = "node_id_not_in_scope"
    elif task_type == "SYNC":
        reason = "sync_task_not_graph_bound"
    elif task_type in EVENT_TASK_TYPES:
        reason = "event_task_without_valid_graph_identity"
    elif task_type in GRAPH_BOUND_TASK_TYPES and raw_task_index is None:
        reason = "missing_task_index"
    elif task_type in GRAPH_BOUND_TASK_TYPES:
        reason = "task_index_out_of_scope_ordinal_range"
    else:
        reason = "task_not_graph_bound"

    identity_kind = (
        "synthesized_custom" if task_type in EVENT_TASK_TYPES else "unresolved"
    )
    identity_status = (
        "resolved" if identity_kind == "synthesized_custom" else "unresolved"
    )
    return {
        "identity_kind": identity_kind,
        "identity_status": identity_status,
        "raw_task_index": raw_task_index,
        "raw_node_id": raw_node_id,
        "resolved_graph_node_id": None,
        "graph_node_key": None,
        "graph_identity_valid": False,
        "graph_identity_source": "none",
        "graph_identity_reason": reason,
        "exclude_from_duplicate_check": True,
        "custom_task_candidate": task_type in EVENT_TASK_TYPES,
        "custom_instance_key": None,
    }


def annotate_device_task_graph_identity(
    device_task_library: dict[str, Any],
    fused_library: dict[str, Any],
    node_library: dict[str, Any],
) -> dict[str, Any]:
    annotated_sections: list[dict[str, Any]] = []
    fused_functions = (
        fused_library.get("functions", []) if isinstance(fused_library, dict) else []
    )
    node_ids_in_library = set()
    node_identity_by_id = {}
    for node in node_library.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if node_id is not None:
            node_ids_in_library.add(int(node_id))
        if isinstance(node_id, int):
            node_identity_by_id[int(node_id)] = node
    duplicate_samples_by_identity_key: dict[str, list[dict[str, Any]]] = {}
    excluded_reason_counts: dict[str, int] = {}
    valid_source_counts = {"explicit_node_id": 0, "fused_ordinal": 0}
    identity_kind_counts = {"graph_node": 0, "synthesized_custom": 0, "unresolved": 0}
    excluded_samples: list[dict[str, Any]] = []
    total_task_count = 0
    graph_identity_valid_count = 0
    excluded_task_count = 0
    custom_invalid_count = 0

    for section_index, section in enumerate(device_task_library.get("sections", [])):
        fused = _match_device_section_to_fused_function(
            section, fused_functions, section_index
        )
        fused_nodes = (fused or {}).get("node_details") or []
        scope_node_ids = set()
        ordinal_node_map = {}
        for node in fused_nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("node_id")
            if node_id is not None:
                scope_node_ids.add(int(node_id))
            ordinal = node.get("ordinal")
            if ordinal is not None and node_id is not None:
                ordinal_node_map[int(ordinal)] = int(node_id)
        section_queues: dict[str, list[dict[str, Any]]] = {}
        section_excluded_reason_counts: dict[str, int] = {}
        section_excluded_samples: list[dict[str, Any]] = []
        section_valid_count = 0
        section_excluded_count = 0
        section_custom_invalid_count = 0
        section_valid_source_counts = {"explicit_node_id": 0, "fused_ordinal": 0}
        section_identity_kind_counts = {
            "graph_node": 0,
            "synthesized_custom": 0,
            "unresolved": 0,
        }
        queue_custom_ordinals: dict[str, int] = {}

        for queue_name, tasks in (section.get("queues", {}) or {}).items():
            annotated_tasks: list[dict[str, Any]] = []
            for task in tasks or []:
                total_task_count += 1
                resolved = _resolve_task_graph_identity(
                    task,
                    scope_node_ids=scope_node_ids,
                    node_ids_in_library=node_ids_in_library,
                    ordinal_node_map=ordinal_node_map,
                    node_identity_by_id=node_identity_by_id,
                )
                annotated_task = dict(task)
                annotated_task.update(resolved)
                identity_kind = str(resolved.get("identity_kind") or "unresolved")
                if identity_kind not in identity_kind_counts:
                    identity_kind = "unresolved"
                    annotated_task["identity_kind"] = identity_kind
                identity_kind_counts[identity_kind] = (
                    identity_kind_counts.get(identity_kind, 0) + 1
                )
                section_identity_kind_counts[identity_kind] = (
                    section_identity_kind_counts.get(identity_kind, 0) + 1
                )

                if identity_kind == "synthesized_custom":
                    queue_key = str(queue_name).upper()
                    queue_custom_ordinal = queue_custom_ordinals.get(queue_key, 0)
                    queue_custom_ordinals[queue_key] = queue_custom_ordinal + 1
                    custom_instance_key = (
                        f"section:{section_index}:queue:{queue_key}:"
                        f"kind:{str(task.get('task_type') or '').upper()}:ord:{queue_custom_ordinal}"
                    )
                    annotated_task["custom_instance_key"] = custom_instance_key
                    annotated_task["custom_descriptor"] = {
                        "scope_name": section.get("scope_name"),
                        "sk_id": section.get("sk_id"),
                        "section_ordinal": section_index,
                        "queue_name": queue_key,
                        "queue_custom_ordinal": queue_custom_ordinal,
                        "custom_kind": str(task.get("task_type") or "").upper(),
                        "task_id": task.get("task_id"),
                        "task_index_raw": task.get("task_index"),
                        "node_id_raw": task.get("node_id"),
                        "args": task.get("args"),
                        "entries": list(task.get("entries", [])),
                    }
                annotated_tasks.append(annotated_task)

                if resolved["graph_identity_valid"]:
                    graph_identity_valid_count += 1
                    section_valid_count += 1
                    source = str(resolved["graph_identity_source"])
                    valid_source_counts[source] = valid_source_counts.get(source, 0) + 1
                    section_valid_source_counts[source] = (
                        section_valid_source_counts.get(source, 0) + 1
                    )
                    resolved_node_id = int(resolved["resolved_graph_node_id"])
                    task_type = str(task.get("task_type") or "").upper()
                    identity_key = (
                        f"{str(queue_name).upper()}:{task_type}:{resolved_node_id}"
                    )
                    sample = _task_identity_sample(
                        section=section,
                        queue_name=str(queue_name),
                        task=task,
                        reason=str(resolved["graph_identity_reason"]),
                        resolved_graph_node_id=resolved_node_id,
                        identity_kind=identity_kind,
                    )
                    sample["identity_key"] = identity_key
                    duplicate_samples_by_identity_key.setdefault(
                        identity_key, []
                    ).append(sample)
                else:
                    excluded_task_count += 1
                    section_excluded_count += 1
                    reason = str(resolved["graph_identity_reason"])
                    excluded_reason_counts[reason] = (
                        excluded_reason_counts.get(reason, 0) + 1
                    )
                    section_excluded_reason_counts[reason] = (
                        section_excluded_reason_counts.get(reason, 0) + 1
                    )
                    sample = _task_identity_sample(
                        section=section,
                        queue_name=str(queue_name),
                        task=task,
                        reason=reason,
                        identity_kind=identity_kind,
                        custom_instance_key=annotated_task.get("custom_instance_key"),
                    )
                    _append_limited_sample(excluded_samples, sample)
                    _append_limited_sample(section_excluded_samples, sample)
                    if resolved["custom_task_candidate"]:
                        custom_invalid_count += 1
                        section_custom_invalid_count += 1
            section_queues[str(queue_name)] = annotated_tasks

        annotated_section = dict(section)
        annotated_section["queues"] = section_queues
        bound_scope_id = (fused or {}).get("scope_id")
        bound_scope_binding_source = (
            "matched_fused_scope_id" if bound_scope_id is not None else "none"
        )
        annotated_section["bound_scope_id"] = bound_scope_id
        annotated_section["bound_scope_binding_source"] = bound_scope_binding_source
        annotated_section["task_identity_diagnostics"] = {
            "matched_fused_scope_id": (fused or {}).get("scope_id"),
            "bound_scope_id": bound_scope_id,
            "bound_scope_binding_source": bound_scope_binding_source,
            "scope_node_count": len(scope_node_ids),
            "task_count": sum(len(tasks or []) for tasks in section_queues.values()),
            "graph_identity_valid_count": section_valid_count,
            "identity_kind_counts": section_identity_kind_counts,
            "excluded_task_count": section_excluded_count,
            "custom_invalid_task_count": section_custom_invalid_count,
            "valid_source_counts": section_valid_source_counts,
            "excluded_reason_counts": section_excluded_reason_counts,
            "excluded_samples": section_excluded_samples,
        }
        annotated_sections.append(annotated_section)

    duplicate_identity_keys = sorted(
        identity_key
        for identity_key, samples in duplicate_samples_by_identity_key.items()
        if len(samples) > 1
    )
    duplicate_details = {
        identity_key: duplicate_samples_by_identity_key[identity_key][:4]
        for identity_key in duplicate_identity_keys[:10]
    }
    diagnostics = {
        "task_count": total_task_count,
        "graph_identity_valid_count": graph_identity_valid_count,
        "identity_kind_counts": identity_kind_counts,
        "excluded_task_count": excluded_task_count,
        "custom_invalid_task_count": custom_invalid_count,
        "valid_source_counts": valid_source_counts,
        "excluded_reason_counts": excluded_reason_counts,
        "excluded_samples": excluded_samples,
        "duplicate_graph_identity_count": len(duplicate_identity_keys),
        "duplicate_graph_identity_keys_sample": duplicate_identity_keys[:20],
        "duplicate_graph_identity_samples": duplicate_details,
        "graph_identity_validation_status": "duplicate"
        if duplicate_identity_keys
        else "passed",
    }

    return {
        "path": device_task_library.get("path"),
        "source_kind": device_task_library.get("source_kind"),
        "sections": annotated_sections,
        "task_queue_json_validation": device_task_library.get(
            "task_queue_json_validation", {}
        ),
        "task_identity_diagnostics": diagnostics,
    }


def _parse_super_kernel_line_items(
    line_items: list[tuple[int, str]], log_path: Path
) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    matched_keys: set[str] = set()
    exception_events: list[dict[str, Any]] = []
    counter_info: dict[str, Any] = {
        "offset": None,
        "cores": [],
    }
    dfx_info: dict[str, Any] = {
        "offset": None,
        "node_cnt": None,
        "entries": [],
    }
    exception_symbol_events: list[dict[str, Any]] = []
    pc_localization_events: list[dict[str, Any]] = []

    current_dfx_node = None
    current_pc_localizations: dict[int, dict[str, Any]] = {}

    for line_no, line in line_items:
        for key, pattern in PHASE_PATTERNS:
            if pattern in line:
                phases.append({"key": key, "line": line_no, "message": line.strip()})
                matched_keys.add(key)
                break

        match = RE_EXCEPTION_FROM_SUPERKERNEL.search(line)
        if match:
            exception_events.append(
                {
                    "line": line_no,
                    "function": match.group(1),
                    "op_trace": match.group(2).lower() == "true",
                    "text": line.strip(),
                }
            )

        match = RE_COUNTERINFO_HEADER.search(line)
        if match:
            counter_info["offset"] = int(match.group(1))
            continue

        match = RE_COUNTERINFO_CORE.search(line)
        if match:
            counter_info["cores"].append(
                {
                    "core_id": int(match.group(1)),
                    "index": int(match.group(2)),
                    "launch": int(match.group(3)),
                    "exit": int(match.group(4)),
                    "line": line_no,
                    "text": line.strip(),
                }
            )
            continue

        match = RE_DFXINFO_HEADER.search(line)
        if match:
            dfx_info["offset"] = int(match.group(1))
            dfx_info["node_cnt"] = int(match.group(2))
            current_dfx_node = None
            continue

        match = RE_DFXINFO_NODE.search(line)
        if match:
            current_dfx_node = {
                "node_index": int(match.group(1)),
                "bin_handle": f"0x{match.group(2)}",
                "original_handle": f"0x{match.group(3)}",
                "aic_size": f"0x{match.group(4)}",
                "aiv_size": f"0x{match.group(5)}",
                "entry_aic": {},
                "entry_aiv": {},
                "function_name": None,
                "line": line_no,
                "text": line.strip(),
            }
            dfx_info["entries"].append(current_dfx_node)
            continue

        match = RE_EXCEPTION_CORE_MATCH.search(line)
        if match:
            core_id = int(match.group(1))
            exception_symbol_events.append(
                {
                    "kind": "matched_core_node",
                    "line": line_no,
                    "core_id": core_id,
                    "node_index": int(match.group(2)),
                    "entry_index": int(match.group(3)),
                    "text": line.strip(),
                }
            )
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["node_index"] = int(match.group(2))
            item["entry_index"] = int(match.group(3))
            item["matched"] = True
            item.setdefault("pc_match_status", "matched_sub_kernel")
            item["line"] = item.get("line") or line_no
            continue

        match = RE_EXCEPTION_CORE_FUNC.search(line)
        if match:
            core_id = int(match.group(1))
            exception_symbol_events.append(
                {
                    "kind": "symbolized_function",
                    "line": line_no,
                    "core_id": core_id,
                    "function_name": match.group(2).strip(),
                    "text": line.strip(),
                }
            )
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["function_name"] = match.group(2).strip()
            item["line"] = item.get("line") or line_no
            continue

        match = RE_NO_SUB_KERNEL_MATCHED.search(line)
        if match:
            core_id = int(match.group(1))
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["pc_match_status"] = "no_sub_kernel_matched"
            item["matched"] = False
            item["line"] = line_no
            item["detail"] = line.strip()
            continue

        match = RE_CORETYPE.search(line)
        if match:
            core_id = int(match.group(1))
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["core_type"] = match.group(2).strip()
            item.setdefault("pc_match_status", "matched_sub_kernel")
            item["line"] = item.get("line") or line_no
            continue

        match = RE_EXCEPTION_START_PC.search(line)
        if match:
            core_id = int(match.group(1))
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["exception_start_pc"] = match.group(2)
            item["line"] = item.get("line") or line_no
            continue

        match = RE_CURRENT_PC.search(line)
        if match:
            core_id = int(match.group(1))
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["current_pc"] = match.group(2)
            item["line"] = item.get("line") or line_no
            continue

        match = RE_ENTRY_ADDRESS.search(line)
        if match:
            core_id = int(match.group(1))
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["entry_start_pc"] = match.group(2)
            item["line"] = item.get("line") or line_no
            continue

        match = RE_END_ADDRESS.search(line)
        if match:
            core_id = int(match.group(1))
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["entry_end_pc"] = match.group(2)
            item["line"] = item.get("line") or line_no
            continue

        match = RE_FUNCTION_SIZE.search(line)
        if match:
            core_id = int(match.group(1))
            item = current_pc_localizations.setdefault(core_id, {"core_id": core_id})
            item["function_size"] = match.group(1 + 1)
            item["function_size_bytes"] = int(match.group(3))
            item["line"] = item.get("line") or line_no
            continue

        match = RE_DFXINFO_FUNC_NAME.search(line)
        if match and current_dfx_node is not None:
            current_dfx_node["function_name"] = match.group(1).strip()
            continue

        match = RE_DFXINFO_ENTRY_AIC.search(line)
        if match and current_dfx_node is not None:
            current_dfx_node["entry_aic"][int(match.group(1))] = f"0x{match.group(2)}"
            continue

        match = RE_DFXINFO_ENTRY_AIV.search(line)
        if match and current_dfx_node is not None:
            current_dfx_node["entry_aiv"][int(match.group(1))] = f"0x{match.group(2)}"
            continue

    dfx_info["entries"] = sorted(
        dfx_info["entries"], key=lambda item: item["node_index"]
    )
    for item in current_pc_localizations.values():
        if (
            "current_pc" in item
            or "exception_start_pc" in item
            or "entry_start_pc" in item
        ):
            pc_localization_events.append(item)
    pc_localization_events.sort(
        key=lambda item: (item.get("line", 0), item.get("core_id", -1))
    )
    return {
        "path": str(log_path),
        "phases": phases,
        "phase_keys": sorted(matched_keys),
        "exception_events": exception_events,
        "sk_counter_info": counter_info,
        "exception_symbol_events": exception_symbol_events,
        "pc_localization_events": pc_localization_events,
        "sk_dfx_dump": dfx_info,
    }


def parse_super_kernel_log(
    model_dir: Path, line_range: tuple[int, int] | None = None
) -> dict[str, Any]:
    log_path = model_dir / "super_kernel.log"
    return _parse_super_kernel_line_items(
        _read_indexed_lines(log_path, line_range), log_path
    )


def _strip_op_trace_suffix(name: str | None) -> str:
    if not name:
        return ""
    if name.endswith("_op_trace"):
        suffix_len = len("_op_trace")
        return name[:-suffix_len]
    return name


def _to_evidence_level(has_explicit: bool, has_inferred: bool = False) -> str:
    if has_explicit:
        return "explicit"
    if has_inferred:
        return "inferred"
    return "unknown"


def _index_fused_functions(
    functions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    function_index: dict[str, list[dict[str, Any]]] = {}
    for function in functions:
        function_text = function.get("function_text", "")
        if function_text:
            function_index.setdefault(function_text, []).append(function)
            function_index.setdefault(_strip_op_trace_suffix(function_text), []).append(
                function
            )
    return function_index


def _section_node_count(section: dict[str, Any]) -> int | None:
    header = section.get("header_info", {})
    if isinstance(header, dict) and isinstance(header.get("node_count"), int):
        return header.get("node_count")
    dfx_rows = section.get("dfx", [])
    if isinstance(dfx_rows, list):
        return len(dfx_rows)
    return None


def _normalize_function_key(value: str | None) -> str:
    return _strip_op_trace_suffix(value).strip() if value else ""


def _find_runtime_payload_section(
    function_sections: list[dict[str, Any]],
    dfx_dump: dict[str, Any],
    exception_events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not function_sections:
        return None, []

    normalized_section_index: dict[str, list[dict[str, Any]]] = {}
    for section in function_sections:
        key = _normalize_function_key(section.get("function_text"))
        if key:
            normalized_section_index.setdefault(key, []).append(section)

    for event in exception_events if isinstance(exception_events, list) else []:
        key = _normalize_function_key(event.get("function"))
        candidates = normalized_section_index.get(key, [])
        if len(candidates) == 1:
            return candidates[0], candidates
        if candidates:
            return None, candidates

    node_cnt = dfx_dump.get("node_cnt") if isinstance(dfx_dump, dict) else None
    if isinstance(node_cnt, int):
        count_matches = [
            section
            for section in function_sections
            if _section_node_count(section) == node_cnt
        ]
        if len(count_matches) == 1:
            return count_matches[0], count_matches
        if count_matches:
            return None, count_matches

    return None, []


def _build_dfx_payload(
    function_sections: list[dict[str, Any]],
    dfx_dump: dict[str, Any] | None,
    exception_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dfx_dump = dfx_dump if isinstance(dfx_dump, dict) else {}
    runtime_entries = dfx_dump.get("entries", [])
    runtime_entries = runtime_entries if isinstance(runtime_entries, list) else []
    has_runtime_header = isinstance(dfx_dump.get("offset"), int) and isinstance(
        dfx_dump.get("node_cnt"), int
    )

    summary = {
        "device_args_section_count": len(function_sections),
        "function_count": 0,
        "matched_function_count": 0,
        "has_runtime_dump": bool(has_runtime_header and runtime_entries),
        "runtime_entry_count": len(runtime_entries),
        "runtime_node_cnt": dfx_dump.get("node_cnt"),
        "runtime_offset": dfx_dump.get("offset"),
    }
    if not has_runtime_header:
        summary["runtime_status"] = "missing_exception_handler_dump"
        return [], summary
    if not runtime_entries:
        summary["runtime_status"] = "missing_exception_handler_rows"
        return [], summary

    matched_section, candidates = _find_runtime_payload_section(
        function_sections, dfx_dump, exception_events
    )
    summary["function_count"] = 1
    if matched_section is not None:
        summary["matched_function_count"] = 1
        summary["runtime_status"] = "matched_device_args_section"
    elif candidates:
        summary["runtime_status"] = "ambiguous_device_args_section"
    else:
        summary["runtime_status"] = "unresolved_device_args_section"

    device_rows_by_index: dict[int, dict[str, Any]] = {}
    if matched_section is not None:
        for row in (
            matched_section.get("dfx", [])
            if isinstance(matched_section.get("dfx"), list)
            else []
        ):
            node_index = row.get("node_index")
            if isinstance(node_index, int):
                device_rows_by_index[node_index] = row

    mapped_rows: list[dict[str, Any]] = []
    for entry in runtime_entries:
        node_index = entry.get("node_index")
        mapped_row = {
            "node_index": node_index,
            "exception_handler": {
                "bin_handle": entry.get("bin_handle"),
                "original_handle": entry.get("original_handle"),
                "aic_size": entry.get("aic_size"),
                "aiv_size": entry.get("aiv_size"),
                "function_name": entry.get("function_name"),
                "entry_aic": entry.get("entry_aic", {}),
                "entry_aiv": entry.get("entry_aiv", {}),
                "line": entry.get("line"),
                "detail": entry.get("text", ""),
            },
        }
        device_row = (
            device_rows_by_index.get(node_index)
            if isinstance(node_index, int)
            else None
        )
        if device_row:
            mapped_row["device_args_compare"] = {
                "bin_handle": device_row.get("bin_handle"),
                "original_handle": device_row.get("original_handle"),
                "aic_size": device_row.get("aic_size"),
                "aiv_size": device_row.get("aiv_size"),
                "detail": device_row.get("detail", ""),
            }
        mapped_rows.append(mapped_row)

    payload_reasons = ["has_exception_handler_dump", "has_exception_handler_rows"]
    if matched_section is not None:
        payload_reasons.append("has_device_args_comparison")
    elif candidates:
        payload_reasons.append("ambiguous_device_args_section")
    else:
        payload_reasons.append("no_device_args_section_match")

    payload_row: dict[str, Any] = {
        "dfx_entry_count": len(runtime_entries),
        "dfx_count_status": (
            "ok"
            if matched_section is not None
            and _section_node_count(matched_section) == len(runtime_entries)
            else "unverified"
        ),
        "rows": mapped_rows,
        "evidence_level": _to_evidence_level(bool(runtime_entries), has_runtime_header),
        "evidence_reasons": payload_reasons,
    }

    if matched_section is not None:
        payload_row["function_text"] = matched_section.get("function_text", "")
        payload_row["sk_id"] = matched_section.get("sk_id")
        payload_row["scope_id"] = matched_section.get("scope_id")
        if matched_section.get("scope_name") not in (None, ""):
            payload_row["scope_name"] = matched_section.get("scope_name")
        header = matched_section.get("header_info", {})
        if isinstance(header, dict) and header:
            payload_row["header_info"] = header
    elif candidates:
        payload_row["candidate_sections"] = [
            {
                "function_text": item.get("function_text", ""),
                "sk_id": item.get("sk_id"),
                "scope_id": item.get("scope_id"),
            }
            for item in candidates
        ]

    return [payload_row], summary


def _build_counter_registry(
    counter_info: dict[str, Any] | None,
    dfx_dump: dict[str, Any] | None,
    payload_rows: list[dict[str, Any]],
    exception_events: list[dict[str, Any]],
) -> dict[str, Any]:
    counter_info = counter_info if isinstance(counter_info, dict) else {}
    dfx_dump = dfx_dump if isinstance(dfx_dump, dict) else {}
    cores = counter_info.get("cores", [])
    cores = cores if isinstance(cores, list) else []
    runtime_entries = dfx_dump.get("entries", [])
    runtime_entries = runtime_entries if isinstance(runtime_entries, list) else []
    runtime_by_index = {
        item.get("node_index"): item
        for item in runtime_entries
        if isinstance(item.get("node_index"), int)
    }
    matched_payload = payload_rows[0] if payload_rows else {}
    matched_context = (
        {
            "function_text": matched_payload.get("function_text"),
            "scope_id": matched_payload.get("scope_id"),
            "sk_id": matched_payload.get("sk_id"),
        }
        if isinstance(matched_payload, dict)
        else {}
    )

    resolved_cores: list[dict[str, Any]] = []
    active_cores: list[dict[str, Any]] = []
    used_cores: list[dict[str, Any]] = []
    finished_cores: list[dict[str, Any]] = []
    unused_cores: list[dict[str, Any]] = []
    has_exception_events = any(
        isinstance(item, dict)
        for item in exception_events
        if isinstance(exception_events, list)
    )
    op_trace_enabled = any(
        bool(item.get("op_trace"))
        for item in exception_events
        if isinstance(item, dict)
    )
    needs_op_trace_hint = has_exception_events and any(
        not bool(item.get("op_trace"))
        for item in exception_events
        if isinstance(item, dict)
    )
    running_op_counts: dict[int, list[int]] = {}

    def _core_type_for_id(core_id: Any) -> str:
        parsed = _safe_parse_int(core_id)
        if isinstance(parsed, int) and parsed >= 25:
            return "AIV"
        return "AIC"

    def _select_entry_pc(
        entry: dict[str, Any], core_id: Any, core_type: str
    ) -> tuple[int | None, str | None, int]:
        entry_key = "entry_aiv" if core_type == "AIV" else "entry_aic"
        entry_map = (
            entry.get(entry_key, {}) if isinstance(entry.get(entry_key), dict) else {}
        )
        normalized_map = {
            _safe_parse_int(slot): str(value)
            for slot, value in entry_map.items()
            if _safe_parse_int(slot) is not None and value not in (None, "", 0, "0x0")
        }
        valid_slots = sorted(slot for slot in normalized_map if isinstance(slot, int))
        if not valid_slots:
            return None, None, 0
        entry_cnt = len(valid_slots)
        parsed_core_id = _safe_parse_int(core_id)
        modulo_slot = (
            (parsed_core_id % entry_cnt) if isinstance(parsed_core_id, int) else 0
        )
        if modulo_slot in normalized_map:
            return modulo_slot, normalized_map[modulo_slot], entry_cnt
        fallback_slot = valid_slots[modulo_slot % entry_cnt]
        return fallback_slot, normalized_map[fallback_slot], entry_cnt

    for core in cores:
        launch = _safe_parse_int(core.get("launch"))
        launch_state, launch_desc = COUNTER_LAUNCH_STATE_MAP.get(
            launch if isinstance(launch, int) else -1,
            ("UNKNOWN", "Unknown launch status"),
        )
        index = _safe_parse_int(core.get("index"))
        resolved = {
            "core_id": core.get("core_id"),
            "index": index,
            "launch": launch,
            "exit": _safe_parse_int(core.get("exit")),
            "launch_state": launch_state,
            "launch_desc": launch_desc,
            "line": core.get("line"),
            "detail": core.get("text", ""),
            "core_type": _core_type_for_id(core.get("core_id")),
        }
        entry = runtime_by_index.get(index) if isinstance(index, int) else None
        if entry:
            resolved["function_name"] = entry.get("function_name")
            resolved["node_index"] = entry.get("node_index")
            selected_slot, selected_entry_pc, entry_count = _select_entry_pc(
                entry,
                core.get("core_id"),
                resolved["core_type"],
            )
            resolved["entry_count"] = entry_count
            resolved["selected_entry_slot"] = selected_slot
            resolved["derived_entry_start_pc"] = selected_entry_pc
            if entry_count > 0:
                resolved["entry_slot_rule"] = f"core_id % {entry_count}"
        if matched_context.get("function_text"):
            resolved.update(
                {
                    key: value
                    for key, value in matched_context.items()
                    if value is not None
                }
            )

        is_unused_core = launch == 0 and index == 0
        resolved["is_unused_core"] = is_unused_core
        if is_unused_core:
            unused_cores.append(resolved)
        else:
            used_cores.append(resolved)

        if launch == 2 and isinstance(index, int):
            running_op_counts.setdefault(index, []).append(core.get("core_id"))
            active_cores.append(resolved)
        elif launch == 1:
            resolved["target_kind"] = "next_op"
            active_cores.append(resolved)
        elif launch == 3:
            resolved["target_kind"] = "last_finished_op"
            finished_cores.append(resolved)

        resolved_cores.append(resolved)

    dominant_active_op_id = None
    dominant_active_core_ids: list[int] = []
    if running_op_counts:
        dominant_active_op_id, dominant_active_core_ids = max(
            running_op_counts.items(),
            key=lambda item: (len(item[1]), -item[0]),
        )
    dominant_entry = (
        runtime_by_index.get(dominant_active_op_id)
        if isinstance(dominant_active_op_id, int)
        else None
    )
    dominant_function_name = (
        dominant_entry.get("function_name")
        if isinstance(dominant_entry, dict)
        else None
    )
    has_running_subkernel = bool(active_cores)
    all_used_cores_finished = (
        bool(used_cores)
        and len(finished_cores) == len(used_cores)
        and not has_running_subkernel
    )
    counter_contract_violation = (not op_trace_enabled) and any(
        (_safe_parse_int(core.get("launch")) not in (None, 0))
        or (_safe_parse_int(core.get("index")) not in (None, 0))
        for core in cores
    )
    if has_exception_events and not op_trace_enabled:
        active_cores = []
        used_cores = []
        finished_cores = []
        has_running_subkernel = False
        all_used_cores_finished = False
        dominant_active_op_id = None
        dominant_active_core_ids = []
        dominant_function_name = None
        counter_problem_kind = "op_trace_disabled"
    elif has_running_subkernel:
        counter_problem_kind = "subkernel_running_issue"
    elif all_used_cores_finished:
        counter_problem_kind = "sk_level_issue"
    elif used_cores:
        counter_problem_kind = "mixed_or_unclear"
    else:
        counter_problem_kind = "no_used_core"

    return {
        "summary": {
            "offset": counter_info.get("offset"),
            "core_count": len(resolved_cores),
            "used_core_count": len(used_cores),
            "unused_core_count": len(unused_cores),
            "active_core_count": len(active_cores),
            "finished_core_count": len(finished_cores),
            "op_trace_enabled": op_trace_enabled,
            "needs_op_trace_hint": needs_op_trace_hint,
            "counter_contract_violation": counter_contract_violation,
            "has_running_subkernel": has_running_subkernel,
            "all_used_cores_finished": all_used_cores_finished,
            "counter_problem_kind": counter_problem_kind,
            "dominant_active_op_id": dominant_active_op_id,
            "dominant_active_core_count": len(dominant_active_core_ids),
            "dominant_active_core_ids": dominant_active_core_ids,
            "dominant_active_function_name": dominant_function_name,
        },
        "cores": resolved_cores,
        "used_cores": used_cores,
        "unused_cores": unused_cores,
        "finished_cores": finished_cores,
        "active_cores": active_cores,
    }


def _build_diagnostic_pc_registry(
    counter_registry: dict[str, Any],
    pc_localization_registry: dict[str, Any],
) -> dict[str, Any]:
    counter_summary = (
        counter_registry.get("summary", {})
        if isinstance(counter_registry, dict)
        else {}
    )
    counter_kind = counter_summary.get("counter_problem_kind")
    pc_events = (
        pc_localization_registry.get("events", [])
        if isinstance(pc_localization_registry.get("events"), list)
        else []
    )
    pc_by_core = {
        item.get("core_id"): item
        for item in pc_events
        if isinstance(item, dict) and item.get("core_id") is not None
    }
    events: list[dict[str, Any]] = []

    if counter_kind == "subkernel_running_issue":
        for core in (
            counter_registry.get("active_cores", [])
            if isinstance(counter_registry.get("active_cores"), list)
            else []
        ):
            core_id = core.get("core_id")
            pc_item = pc_by_core.get(core_id, {})
            entry_start_pc = pc_item.get("entry_start_pc") or core.get(
                "derived_entry_start_pc"
            )
            events.append(
                {
                    "issue_kind": "subkernel",
                    "core_id": core_id,
                    "core_type": core.get("core_type"),
                    "op_id": core.get("index"),
                    "function_name": pc_item.get("function_name")
                    or core.get("function_name"),
                    "reported_start_pc": entry_start_pc,
                    "reported_current_pc": pc_item.get("current_pc"),
                    "reported_pc_basis": (
                        "entry_start_pc_from_pc_localization"
                        if pc_item.get("entry_start_pc")
                        else "entry_selected_by_core_id_mod_entry_cnt"
                    ),
                    "exception_start_pc": pc_item.get("exception_start_pc"),
                    "current_pc": pc_item.get("current_pc"),
                    "entry_start_pc": entry_start_pc,
                    "entry_end_pc": pc_item.get("entry_end_pc"),
                    "node_index": pc_item.get("node_index")
                    if pc_item.get("node_index") is not None
                    else core.get("node_index"),
                    "entry_index": pc_item.get("entry_index")
                    if pc_item.get("entry_index") is not None
                    else core.get("selected_entry_slot"),
                    "entry_slot_rule": core.get("entry_slot_rule"),
                }
            )
    elif counter_kind == "sk_level_issue":
        for item in pc_events:
            if not isinstance(item, dict):
                continue
            events.append(
                {
                    "issue_kind": "sk",
                    "core_id": item.get("core_id"),
                    "core_type": item.get("core_type"),
                    "op_id": None,
                    "function_name": item.get("function_name"),
                    "reported_start_pc": item.get("exception_start_pc"),
                    "reported_current_pc": item.get("current_pc"),
                    "reported_pc_basis": "exception_register_start_pc",
                    "exception_start_pc": item.get("exception_start_pc"),
                    "current_pc": item.get("current_pc"),
                    "entry_start_pc": item.get("entry_start_pc"),
                    "entry_end_pc": item.get("entry_end_pc"),
                    "node_index": item.get("node_index"),
                    "entry_index": item.get("entry_index"),
                    "entry_slot_rule": None,
                }
            )

    has_reported_start_pc = False
    has_reported_current_pc = False
    for item in events:
        if item.get("reported_start_pc"):
            has_reported_start_pc = True
        if item.get("reported_current_pc"):
            has_reported_current_pc = True

    return {
        "summary": {
            "issue_kind": counter_kind,
            "target_count": len(events),
            "has_reported_start_pc": has_reported_start_pc,
            "has_reported_current_pc": has_reported_current_pc,
        },
        "events": events,
    }


def _build_pc_localization_registry(super_kernel: dict[str, Any]) -> dict[str, Any]:
    items = (
        super_kernel.get("pc_localization_events", [])
        if isinstance(super_kernel, dict)
        else []
    )
    items = items if isinstance(items, list) else []
    matched_count = 0
    unresolved_count = 0
    for item in items:
        if item.get("pc_match_status") == "matched_sub_kernel":
            matched_count += 1
        if item.get("pc_match_status") == "no_sub_kernel_matched":
            unresolved_count += 1
    return {
        "summary": {
            "event_count": len(items),
            "matched_count": matched_count,
            "unresolved_count": unresolved_count,
        },
        "events": items,
    }


def _build_phase_registry(super_kernel: dict[str, Any]) -> dict[str, Any]:
    phases = super_kernel.get("phases", []) if isinstance(super_kernel, dict) else []
    phase_keys = (
        super_kernel.get("phase_keys", []) if isinstance(super_kernel, dict) else []
    )
    stats_by_key: dict[str, dict[str, Any]] = {}
    markers: list[dict[str, Any]] = []

    for item in phases if isinstance(phases, list) else []:
        key = item.get("key")
        line = item.get("line")
        message = item.get("message", "")
        if not key or not isinstance(line, int):
            continue
        if key not in stats_by_key:
            stats_by_key[key] = {
                "key": key,
                "count": 0,
                "first_line": line,
                "last_line": line,
                "sample_message": message,
            }
            markers.append(
                {
                    "key": key,
                    "line": line,
                    "message": message,
                }
            )
        stats = stats_by_key[key]
        stats["count"] += 1
        stats["last_line"] = line

    return {
        "phase_keys": phase_keys if isinstance(phase_keys, list) else [],
        "phase_markers": markers,
        "phase_stats": [stats_by_key[key] for key in phase_keys if key in stats_by_key],
    }


def _compact_dfx_evidence(dfx_evidence: dict[str, Any]) -> dict[str, Any]:
    payload = dfx_evidence.get("payload", {}) if isinstance(dfx_evidence, dict) else {}
    exception = (
        dfx_evidence.get("exception", {}) if isinstance(dfx_evidence, dict) else {}
    )
    counter = dfx_evidence.get("counter", {}) if isinstance(dfx_evidence, dict) else {}
    pc_localization = (
        dfx_evidence.get("pc_localization", {})
        if isinstance(dfx_evidence, dict)
        else {}
    )
    diagnostic_pc = (
        dfx_evidence.get("diagnostic_pc", {}) if isinstance(dfx_evidence, dict) else {}
    )
    update = dfx_evidence.get("update", {}) if isinstance(dfx_evidence, dict) else {}

    return {
        "payload": {
            "evidence_level": payload.get("evidence_level"),
            "evidence_reasons": payload.get("evidence_reasons", []),
            "summary": payload.get("summary", {}),
        },
        "exception": {
            "has_superkernel_exception": exception.get(
                "has_superkernel_exception", False
            ),
            "event_count": len(exception.get("events", []))
            if isinstance(exception.get("events"), list)
            else 0,
            "core_symbol_event_count": len(exception.get("core_symbol_events", []))
            if isinstance(exception.get("core_symbol_events"), list)
            else 0,
            "evidence_level": exception.get("evidence_level"),
            "core_symbol_evidence_level": exception.get("core_symbol_evidence_level"),
            "evidence_reasons": exception.get("evidence_reasons", []),
        },
        "counter": {
            "evidence_level": counter.get("evidence_level"),
            "evidence_reasons": counter.get("evidence_reasons", []),
            "summary": counter.get("summary", {}),
        },
        "pc_localization": {
            "evidence_level": pc_localization.get("evidence_level"),
            "evidence_reasons": pc_localization.get("evidence_reasons", []),
            "summary": pc_localization.get("summary", {}),
        },
        "diagnostic_pc": {
            "evidence_level": diagnostic_pc.get("evidence_level"),
            "evidence_reasons": diagnostic_pc.get("evidence_reasons", []),
            "summary": diagnostic_pc.get("summary", {}),
        },
        "update": {
            "node_update_total": update.get("node_update_total", 0),
            "value_write_wait_total": update.get("value_write_wait_total", 0),
            "value_update_types": update.get("value_update_types", {}),
            "evidence_level": update.get("evidence_level"),
            "evidence_reasons": update.get("evidence_reasons", []),
        },
    }


def build_dfx_evidence(
    update_execution: dict[str, Any],
    device_task_library: dict[str, Any],
    fused_library: dict[str, Any],
    super_kernel: dict[str, Any],
) -> dict[str, Any]:
    function_sections = (
        device_task_library.get("sections", [])
        if isinstance(device_task_library, dict)
        else []
    )
    dfx_dump = (
        super_kernel.get("sk_dfx_dump", {}) if isinstance(super_kernel, dict) else {}
    )
    counter_info = (
        super_kernel.get("sk_counter_info", {})
        if isinstance(super_kernel, dict)
        else {}
    )
    exception_events = (
        super_kernel.get("exception_events", [])
        if isinstance(super_kernel, dict)
        else []
    )
    payload_rows, payload_summary = _build_dfx_payload(
        function_sections,
        dfx_dump if isinstance(dfx_dump, dict) else {},
        exception_events if isinstance(exception_events, list) else [],
    )

    function_index = _index_fused_functions(
        fused_library.get("functions", []) if isinstance(fused_library, dict) else []
    )
    mapped_exception_events = []
    for event in (
        super_kernel.get("exception_events", [])
        if isinstance(super_kernel, dict)
        else []
    ):
        function = event.get("function", "")
        candidates = []
        for key in (function, _strip_op_trace_suffix(function)):
            for function_entry in function_index.get(key, []):
                candidates.append(
                    {
                        "function_text": function_entry.get("function_text"),
                        "scope_id": function_entry.get("scope_id"),
                        "sk_id": function_entry.get("sk_id"),
                        "scope_name": function_entry.get("scope_name", ""),
                        "node_ids": function_entry.get("node_ids", []),
                    }
                )

        seen_keys = set()
        unique_candidates = []
        for item in candidates:
            key = (item.get("function_text"), item.get("scope_id"), item.get("sk_id"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_candidates.append(item)

        mapped_exception_events.append(
            {
                "line": event.get("line"),
                "function": function,
                "normalized_function": _strip_op_trace_suffix(function),
                "op_trace": event.get("op_trace"),
                "raw_line": event.get("text", ""),
                "candidate_count": len(unique_candidates),
                "candidates": unique_candidates,
                "evidence_level": _to_evidence_level(
                    bool(unique_candidates),
                    bool(event.get("op_trace")) and bool(event.get("line")),
                ),
                "evidence_reasons": [
                    "candidate_functions_found"
                    if unique_candidates
                    else "unresolved_function_mapping",
                    "exception_from_superkernel_function"
                    if event.get("function")
                    else "missing_exception_function",
                ],
            }
        )

    has_explicit_exception = False
    has_inferred_exception = False
    for item in mapped_exception_events:
        if item.get("evidence_level") == "explicit":
            has_explicit_exception = True
        if item.get("evidence_level") == "inferred":
            has_inferred_exception = True
    exception_level = _to_evidence_level(
        has_explicit_exception,
        has_inferred_exception,
    )
    core_symbol_level = _to_evidence_level(
        bool(super_kernel.get("exception_symbol_events"))
    )
    counter_registry = _build_counter_registry(
        counter_info if isinstance(counter_info, dict) else {},
        dfx_dump if isinstance(dfx_dump, dict) else {},
        payload_rows,
        exception_events if isinstance(exception_events, list) else [],
    )
    counter_summary = (
        counter_registry.get("summary", {})
        if isinstance(counter_registry, dict)
        else {}
    )
    pc_localization_registry = _build_pc_localization_registry(
        super_kernel if isinstance(super_kernel, dict) else {}
    )
    pc_localization_summary = (
        pc_localization_registry.get("summary", {})
        if isinstance(pc_localization_registry, dict)
        else {}
    )
    diagnostic_pc_registry = _build_diagnostic_pc_registry(
        counter_registry, pc_localization_registry
    )
    diagnostic_pc_summary = (
        diagnostic_pc_registry.get("summary", {})
        if isinstance(diagnostic_pc_registry, dict)
        else {}
    )

    node_update_results = (
        update_execution.get("node_update_results", [])
        if isinstance(update_execution, dict)
        else []
    )
    value_update_types = {
        "VALUE_WRITE": 0,
        "VALUE_WAIT": 0,
        "VALUE_READ": 0,
        "VALUE_RESET": 0,
        "INVALID": 0,
        "OTHER": 0,
    }
    for row in node_update_results if isinstance(node_update_results, list) else []:
        update_type = row.get("type", "OTHER")
        if update_type == "VALUE_WRITE":
            value_update_types["VALUE_WRITE"] += 1
        elif update_type == "VALUE_WAIT":
            value_update_types["VALUE_WAIT"] += 1
        elif update_type in ("MEMORY_WRITE", "MEMORY_WAIT"):
            value_update_types["OTHER"] += 1
        else:
            value_update_types.setdefault(update_type, 0)
            value_update_types[update_type] = value_update_types.get(update_type, 0) + 1

    value_update_total = (
        value_update_types["VALUE_WRITE"] + value_update_types["VALUE_WAIT"]
    )
    return {
        "source": {
            "super_kernel_log": super_kernel.get("path")
            if isinstance(super_kernel, dict)
            else None,
            "sk_device_args": device_task_library.get("path")
            if isinstance(device_task_library, dict)
            else None,
            "sk_fused_nodes": fused_library.get("path")
            if isinstance(fused_library, dict)
            else None,
        },
        "payload": {
            "functions": payload_rows,
            "summary": payload_summary,
            "evidence_level": _to_evidence_level(
                payload_summary.get("matched_function_count", 0) > 0,
                bool(payload_summary.get("has_runtime_dump")),
            ),
            "evidence_reasons": [
                "has_exception_handler_dump"
                if payload_summary.get("has_runtime_dump")
                else payload_summary.get(
                    "runtime_status", "missing_exception_handler_dump"
                ),
                "has_device_args_sections"
                if payload_summary.get("device_args_section_count", 0) > 0
                else "no_device_args_sections",
            ],
        },
        "exception": {
            "events": mapped_exception_events,
            "core_symbol_events": super_kernel.get("exception_symbol_events", [])
            if isinstance(super_kernel, dict)
            else [],
            "has_superkernel_exception": bool(mapped_exception_events),
            "evidence_level": exception_level,
            "core_symbol_evidence_level": core_symbol_level,
            "evidence_reasons": [
                "superkernel_exception_event"
                if mapped_exception_events
                else "no_exception_event",
                "core_symbol_events"
                if super_kernel.get("exception_symbol_events")
                else "no_core_symbol_events",
            ],
        },
        "counter": {
            "summary": counter_summary,
            "cores": counter_registry.get("cores", [])
            if isinstance(counter_registry, dict)
            else [],
            "active_cores": counter_registry.get("active_cores", [])
            if isinstance(counter_registry, dict)
            else [],
            "used_cores": counter_registry.get("used_cores", [])
            if isinstance(counter_registry, dict)
            else [],
            "unused_cores": counter_registry.get("unused_cores", [])
            if isinstance(counter_registry, dict)
            else [],
            "finished_cores": counter_registry.get("finished_cores", [])
            if isinstance(counter_registry, dict)
            else [],
            "evidence_level": _to_evidence_level(
                bool(counter_summary.get("dominant_active_function_name")),
                bool(counter_summary.get("active_core_count")),
            ),
            "evidence_reasons": [
                "has_counter_cores"
                if counter_summary.get("core_count", 0) > 0
                else "no_counter_cores",
                "has_running_op_localization"
                if counter_summary.get("dominant_active_function_name")
                else "no_running_op_localization",
                "op_trace_disabled"
                if counter_summary.get("needs_op_trace_hint")
                else "op_trace_enabled_or_unknown",
            ],
        },
        "pc_localization": {
            "summary": pc_localization_summary,
            "events": pc_localization_registry.get("events", [])
            if isinstance(pc_localization_registry, dict)
            else [],
            "evidence_level": _to_evidence_level(
                pc_localization_summary.get("matched_count", 0) > 0,
                pc_localization_summary.get("event_count", 0) > 0,
            ),
            "evidence_reasons": [
                "matched_sub_kernel_pc_range"
                if pc_localization_summary.get("matched_count", 0) > 0
                else "no_matched_sub_kernel_pc_range",
                "pc_localization_events"
                if pc_localization_summary.get("event_count", 0) > 0
                else "no_pc_localization_events",
            ],
        },
        "diagnostic_pc": {
            "summary": diagnostic_pc_summary,
            "events": diagnostic_pc_registry.get("events", [])
            if isinstance(diagnostic_pc_registry, dict)
            else [],
            "evidence_level": _to_evidence_level(
                diagnostic_pc_summary.get("target_count", 0) > 0
                and diagnostic_pc_summary.get("has_reported_start_pc", False),
                diagnostic_pc_summary.get("target_count", 0) > 0,
            ),
            "evidence_reasons": [
                "has_diagnostic_pc_targets"
                if diagnostic_pc_summary.get("target_count", 0) > 0
                else "no_diagnostic_pc_targets",
                "has_reported_current_pc"
                if diagnostic_pc_summary.get("has_reported_current_pc", False)
                else "no_reported_current_pc",
            ],
        },
        "update": {
            "node_update_total": len(node_update_results),
            "value_write_wait_total": value_update_total,
            "value_update_types": value_update_types,
            "raw_node_updates": node_update_results,
            "evidence_level": _to_evidence_level(
                bool(node_update_results),
                bool(value_update_total),
            ),
            "evidence_reasons": [
                "has_node_update_results"
                if node_update_results
                else "no_node_update_results",
                "has_value_write_wait" if value_update_total else "no_value_write_wait",
            ],
        },
    }


def build_dfx_library(
    model_dir: Path,
    device_task_library: dict[str, Any],
    fused_library: dict[str, Any],
    super_kernel: dict[str, Any],
    dfx_evidence: dict[str, Any],
) -> dict[str, Any]:
    payload = dfx_evidence.get("payload", {}) if isinstance(dfx_evidence, dict) else {}
    exception = (
        dfx_evidence.get("exception", {}) if isinstance(dfx_evidence, dict) else {}
    )
    counter = dfx_evidence.get("counter", {}) if isinstance(dfx_evidence, dict) else {}
    pc_localization = (
        dfx_evidence.get("pc_localization", {})
        if isinstance(dfx_evidence, dict)
        else {}
    )
    diagnostic_pc = (
        dfx_evidence.get("diagnostic_pc", {}) if isinstance(dfx_evidence, dict) else {}
    )
    phase_registry = _build_phase_registry(
        super_kernel if isinstance(super_kernel, dict) else {}
    )
    payload_summary = dict(
        payload.get("summary", {}) if isinstance(payload, dict) else {}
    )

    return {
        "path": str(model_dir / "dfx-library.json"),
        "source_kind": "sk_meta",
        "model_ri": _extract_model_ri(model_dir),
        "source": {
            "super_kernel_log": super_kernel.get("path")
            if isinstance(super_kernel, dict)
            else None,
            "sk_device_args": device_task_library.get("path")
            if isinstance(device_task_library, dict)
            else None,
            "sk_fused_nodes": fused_library.get("path")
            if isinstance(fused_library, dict)
            else None,
        },
        "phase_registry": phase_registry,
        "payload_registry": {
            "functions": payload.get("functions", [])
            if isinstance(payload, dict)
            else [],
            "summary": payload_summary,
        },
        "exception_registry": {
            "events": exception.get("events", [])
            if isinstance(exception, dict)
            else [],
            "core_symbol_events": exception.get("core_symbol_events", [])
            if isinstance(exception, dict)
            else [],
            "has_superkernel_exception": exception.get(
                "has_superkernel_exception", False
            )
            if isinstance(exception, dict)
            else False,
        },
        "counter_registry": {
            "summary": counter.get("summary", {}) if isinstance(counter, dict) else {},
            "cores": counter.get("cores", []) if isinstance(counter, dict) else [],
            "used_cores": counter.get("used_cores", [])
            if isinstance(counter, dict)
            else [],
            "unused_cores": counter.get("unused_cores", [])
            if isinstance(counter, dict)
            else [],
            "finished_cores": counter.get("finished_cores", [])
            if isinstance(counter, dict)
            else [],
            "active_cores": counter.get("active_cores", [])
            if isinstance(counter, dict)
            else [],
        },
        "pc_localization_registry": {
            "summary": pc_localization.get("summary", {})
            if isinstance(pc_localization, dict)
            else {},
            "events": pc_localization.get("events", [])
            if isinstance(pc_localization, dict)
            else [],
        },
        "diagnostic_pc_registry": {
            "summary": diagnostic_pc.get("summary", {})
            if isinstance(diagnostic_pc, dict)
            else {},
            "events": diagnostic_pc.get("events", [])
            if isinstance(diagnostic_pc, dict)
            else [],
        },
        "evidence": _compact_dfx_evidence(
            dfx_evidence if isinstance(dfx_evidence, dict) else {}
        ),
    }


def _parse_update_execution_line_items(
    line_items: list[tuple[int, str]], path: Path
) -> dict[str, Any]:
    scope_updates: list[dict[str, Any]] = []
    node_update_results: list[dict[str, Any]] = []
    event_addr_updates: list[dict[str, Any]] = []
    event_memory_resources: list[dict[str, Any]] = []
    graph_update = {"graph_update_begin_line": None, "graph_update_end_line": None}
    current_scope: dict[str, Any] | None = None
    current_stream: dict[str, Any] | None = None
    scheduled_function: str | None = None

    for line_no, line in line_items:
        match = RE_SCHEDULED_SK_FUNCTION.search(line)
        if match:
            scheduled_function = match.group(1)
        if graph_update[
            "graph_update_begin_line"
        ] is None and RE_UPDATE_GRAPH_START.search(line):
            graph_update["graph_update_begin_line"] = line_no
        if graph_update["graph_update_end_line"] is None and RE_UPDATE_GRAPH_END.search(
            line
        ):
            graph_update["graph_update_end_line"] = line_no

        match = RE_SCOPE_UPDATE_BEGIN.search(line)
        if match:
            current_scope = {
                "scope_update_order": len(scope_updates),
                "begin_line": line_no,
                "begin_detail": line.strip(),
                "scope_name": _normalize_scope_name(match.group(1)),
                "sk_function": scheduled_function,
                "stream_count": int(match.group(2)),
                "end_line": None,
                "finish_detail": "",
                "update_total_nodes": None,
                "streams": [],
            }
            scope_updates.append(current_scope)
            scheduled_function = None
            current_stream = None
            continue

        match = RE_UPDATE_STREAM_BEGIN.search(line)
        if match and current_scope is not None:
            current_stream = {
                "scope_name": _normalize_scope_name(
                    match.group(1) or current_scope.get("scope_name")
                ),
                "stream_id": int(match.group(2)),
                "stream_idx_in_graph": int(match.group(2)),
                "head_node_id": int(match.group(3)),
                "tail_node_id": int(match.group(4)),
                "node_size": int(match.group(5)),
                "custom_param_size": int(match.group(6)),
                "begin_line": line_no,
                "begin_detail": line.strip(),
                "end_line": None,
                "end_detail": "",
                "visited_nodes": None,
                "task_updates": [],
            }
            current_scope["streams"].append(current_stream)
            continue

        match = RE_UPDATING_NODE_TASK.search(line)
        if match and current_stream is not None:
            current_stream["task_updates"].append(
                {
                    "task_index": int(match.group(1)),
                    "task_line": line_no,
                    "updated_kernel": False,
                    "updated_line": None,
                }
            )
            continue

        match = RE_UPDATED_KERNEL_NODE.search(line)
        if match and current_stream is not None:
            task_index = int(match.group(1))
            for item in reversed(current_stream["task_updates"]):
                if item["task_index"] == task_index and not item["updated_kernel"]:
                    item["updated_kernel"] = True
                    item["updated_line"] = line_no
                    break
            continue

        match = RE_UPDATE_STREAM_END.search(line)
        if match and current_stream is not None:
            if match.group(1):
                current_stream["scope_name"] = _normalize_scope_name(match.group(1))
            current_stream["end_line"] = line_no
            current_stream["end_detail"] = line.strip()
            current_stream["visited_nodes"] = int(match.group(3))
            continue

        match = RE_SCOPE_UPDATE_FINISHED.search(line)
        if match and current_scope is not None:
            if match.group(1):
                current_scope["scope_name"] = _normalize_scope_name(match.group(1))
            current_scope["end_line"] = line_no
            current_scope["finish_detail"] = line.strip()
            current_scope["update_total_nodes"] = int(match.group(2))
            current_stream = None
            current_scope = None
            continue

        match = RE_NODE_UPDATE_RESULT.search(line)
        if match:
            extra = _parse_loose_kv_fields(match.group(3))
            update_type = match.group(2)
            node_update_results.append(
                {
                    "node_id": int(match.group(1)),
                    "type": update_type,
                    "op_info_ptr": extra.get("opInfoPtr"),
                    "op_info_size": _safe_parse_int(extra.get("opInfoSize")),
                    "func_handle": extra.get("funcHandle"),
                    "args": extra.get("args"),
                    "args_size": _safe_parse_int(extra.get("argsSize")),
                    "num_blocks": _safe_parse_int(extra.get("numBlocks")),
                    "addr": extra.get("addr"),
                    "value": extra.get("value"),
                    "flag": extra.get("flag"),
                    "detail": line.strip(),
                    "line": line_no,
                    "sk_function": current_scope.get("sk_function")
                    if current_scope
                    else None,
                }
            )
            continue

        match = RE_EVENT_NODE_ADDR.search(line)
        if match:
            event_addr_updates.append(
                {
                    "kind": match.group(1),
                    "node_id": int(match.group(2)),
                    "addr": match.group(3),
                    "detail": line.strip(),
                    "line": line_no,
                }
            )
            continue

        match = RE_EVENT_MEMORY_RESOURCE.search(line)
        if match:
            event_memory_resources.append(
                {
                    "event_id": match.group(1),
                    "addr": match.group(2),
                    "detail": line.strip(),
                    "line": line_no,
                }
            )
            continue

    flat_task_updates = []
    for scope_update in scope_updates:
        for stream_update in scope_update["streams"]:
            for item in stream_update["task_updates"]:
                flat_task_updates.append(
                    {
                        "scope_update_order": scope_update["scope_update_order"],
                        "scope_name": scope_update.get("scope_name"),
                        "stream_id": stream_update["stream_id"],
                        "stream_idx_in_graph": stream_update["stream_idx_in_graph"],
                        "head_node_id": stream_update["head_node_id"],
                        "tail_node_id": stream_update["tail_node_id"],
                        "custom_param_size": stream_update["custom_param_size"],
                        **item,
                    }
                )

    return {
        "path": str(path),
        "graph_update": {
            **graph_update,
            "graph_update_seen": bool(
                graph_update["graph_update_begin_line"]
                and graph_update["graph_update_end_line"]
            ),
        },
        "scope_updates": scope_updates,
        "task_updates": flat_task_updates,
        "node_update_results": node_update_results,
        "event_addr_updates": event_addr_updates,
        "event_memory_resources": event_memory_resources,
    }


def parse_update_execution(
    model_dir: Path, line_range: tuple[int, int] | None = None
) -> dict[str, Any]:
    path = model_dir / "super_kernel.log"
    return _parse_update_execution_line_items(
        _read_indexed_lines(path, line_range), path
    )


def _finalize_scope_batch(scopes: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for scope_id in sorted(scopes):
        scope = scopes[scope_id]
        node_seen: set[tuple[int, int, int]] = set()
        dedup_nodes = []
        for node in scope["nodes"]:
            node_sig = (
                node["node_id"],
                node["stream_idx_in_graph"],
                node["node_idx_in_stream"],
            )
            if node_sig in node_seen:
                continue
            node_seen.add(node_sig)
            dedup_nodes.append(node)
        scope["nodes"] = dedup_nodes

        latest_stream_by_idx: dict[int, dict[str, Any]] = {}
        for stream in scope["streams"]:
            latest_stream_by_idx[stream["stream_idx"]] = stream
        dedup_streams = []
        for idx in sorted(latest_stream_by_idx):
            dedup_streams.append(latest_stream_by_idx[idx])
        expected_stream_count = scope.get("stream_count", 0)
        if isinstance(expected_stream_count, int) and expected_stream_count > 0:
            dedup_streams = dedup_streams[:expected_stream_count]
        scope["streams"] = dedup_streams
        ordered.append(scope)
    return ordered


def _parse_scope_split_file(path: Path) -> dict[str, Any]:
    scopes: dict[int, dict[str, Any]] = {}
    current_scope_id: int | None = None
    batches: list[dict[str, Any]] = []
    scope_display_to_id: dict[int, int] = {}
    current_batch_header = ""
    current_batch_pass: str | None = None
    current_batch_begin_line: int | None = None
    current_batch_end_line: int | None = None
    current_batch_begin_timestamp: str | None = None
    current_batch_end_timestamp: str | None = None
    for line_no, line in _read_indexed_lines(path):
        begin_match = RE_SCOPE_BATCH_BEGIN.search(line)
        if begin_match:
            if scopes:
                batches.append(
                    {
                        "detail": current_batch_header,
                        "pass": current_batch_pass,
                        "scopes": _finalize_scope_batch(scopes),
                        "begin_line": current_batch_begin_line,
                        "end_line": current_batch_end_line,
                        "begin_timestamp": current_batch_begin_timestamp,
                        "end_timestamp": current_batch_end_timestamp,
                    }
                )
            scopes = {}
            scope_display_to_id = {}
            current_scope_id = None
            current_batch_header = line.strip()
            current_batch_pass = begin_match.group(1).strip()
            current_batch_begin_line = line_no
            current_batch_end_line = line_no
            current_batch_begin_timestamp = _extract_line_timestamp(line)
            current_batch_end_timestamp = current_batch_begin_timestamp
            continue

        if RE_SCOPE_BATCH_HEADER.search(line):
            if scopes:
                batches.append(
                    {
                        "detail": current_batch_header,
                        "pass": current_batch_pass,
                        "scopes": _finalize_scope_batch(scopes),
                        "begin_line": current_batch_begin_line,
                        "end_line": current_batch_end_line,
                        "begin_timestamp": current_batch_begin_timestamp,
                        "end_timestamp": current_batch_end_timestamp,
                    }
                )
            scopes = {}
            scope_display_to_id = {}
            current_scope_id = None
            current_batch_header = line.strip()
            current_batch_pass = None
            current_batch_begin_line = line_no
            current_batch_end_line = line_no
            current_batch_begin_timestamp = _extract_line_timestamp(line)
            current_batch_end_timestamp = current_batch_begin_timestamp
            continue

        if current_batch_begin_line is not None:
            current_batch_end_line = line_no
            current_batch_end_timestamp = (
                _extract_line_timestamp(line) or current_batch_end_timestamp
            )

        match = RE_SCOPE_SUMMARY.search(line)
        if match:
            display_scope_id = int(match.group(1))
            scope_id = int(match.group(2) or display_scope_id)
            scope_display_to_id[display_scope_id] = scope_id
            scopes.setdefault(
                scope_id,
                {
                    "scope_id": scope_id,
                    "node_count": int(match.group(3)),
                    "stream_count": int(match.group(4)),
                    "scope_bit_flags": match.group(5),
                    "scope_names": match.group(6),
                    "nodes": [],
                    "streams": [],
                },
            )
            current_scope_id = None
            continue

        match = RE_SCOPE_NODE_HEADER.search(line)
        if match:
            display_scope_id = int(match.group(1))
            current_scope_id = scope_display_to_id.get(
                display_scope_id, display_scope_id
            )
            continue

        match = RE_SCOPE_NODE.search(line)
        if match:
            node = {
                "ordinal": int(match.group(1)),
                "node_id": int(match.group(2)),
                "stream_id": int(match.group(3)),
                "stream_idx_in_graph": int(match.group(4)),
                "node_idx_in_stream": int(match.group(5)),
                "detail": line.strip(),
            }
            if current_scope_id is not None and current_scope_id in scopes:
                scopes[current_scope_id]["nodes"].append(node)
            continue

        match = RE_SCOPE_STREAM.search(line)
        if match:
            display_scope_id = int(match.group(1))
            scope_id = scope_display_to_id.get(display_scope_id, display_scope_id)
            scope = scopes.setdefault(
                scope_id,
                {
                    "scope_id": scope_id,
                    "node_count": 0,
                    "stream_count": 0,
                    "scope_bit_flags": "",
                    "scope_names": "",
                    "nodes": [],
                    "streams": [],
                },
            )
            scope["streams"].append(
                {
                    "info_idx": int(match.group(2)),
                    "stream_idx": int(match.group(3)),
                    "head_node": int(match.group(4)),
                    "tail_node": int(match.group(5)),
                    "node_size": int(match.group(6)),
                    "detail": line.strip(),
                }
            )
            current_scope_id = None

    if scopes:
        batches.append(
            {
                "detail": current_batch_header,
                "pass": current_batch_pass,
                "scopes": _finalize_scope_batch(scopes),
                "begin_line": current_batch_begin_line,
                "end_line": current_batch_end_line,
                "begin_timestamp": current_batch_begin_timestamp,
                "end_timestamp": current_batch_end_timestamp,
            }
        )

    final_scopes = batches[-1]["scopes"] if batches else []
    transitions_by_scope_id: dict[int, list[dict[str, Any]]] = {}
    for batch in batches[:-1]:
        detail = batch.get("detail", "")
        pass_name = batch.get("pass")
        for scope in batch.get("scopes", []):
            scope_id = scope.get("scope_id")
            if scope_id is None:
                continue
            entry = {
                "detail": detail,
                "node_ids": [
                    node.get("node_id")
                    for node in scope.get("nodes", [])
                    if node.get("node_id") is not None
                ],
            }
            if pass_name:
                entry["pass"] = pass_name
            transitions_by_scope_id.setdefault(scope_id, []).append(entry)
    return {
        "scopes": final_scopes,
        "transitions_by_scope_id": transitions_by_scope_id,
        "batches": batches,
    }


def parse_scope_library(model_dir: Path) -> dict[str, Any]:
    primary = model_dir / "sk_scope_split.log"
    fallback = model_dir / "super_kernel.log"
    if primary.is_file():
        path = primary
        source_kind = "sk_scope_split.log"
    else:
        path = fallback
        source_kind = "super_kernel.log"
    parsed = _parse_scope_split_file(path)
    return {
        "path": str(path),
        "source_kind": source_kind,
        "scopes": parsed["scopes"],
        "transitions_by_scope_id": parsed["transitions_by_scope_id"],
        "batches": parsed["batches"],
    }


def _build_scope_library_from_batches(
    *,
    path: str,
    source_kind: str,
    batch_group: list[dict[str, Any]],
) -> dict[str, Any]:
    final_scopes = batch_group[-1]["scopes"] if batch_group else []
    transitions_by_scope_id: dict[int, list[dict[str, Any]]] = {}
    for batch in batch_group[:-1]:
        detail = batch.get("detail", "")
        pass_name = batch.get("pass")
        for scope in batch.get("scopes", []):
            scope_id = scope.get("scope_id")
            if scope_id is None:
                continue
            entry = {
                "detail": detail,
                "node_ids": [
                    node.get("node_id")
                    for node in scope.get("nodes", [])
                    if node.get("node_id") is not None
                ],
            }
            if pass_name:
                entry["pass"] = pass_name
            transitions_by_scope_id.setdefault(scope_id, []).append(entry)
    return {
        "path": path,
        "source_kind": source_kind,
        "scopes": final_scopes,
        "transitions_by_scope_id": transitions_by_scope_id,
        "batches": batch_group,
    }


def _build_empty_scope_library(path: str, source_kind: str) -> dict[str, Any]:
    return {
        "path": path,
        "source_kind": source_kind,
        "scopes": [],
        "transitions_by_scope_id": {},
        "batches": [],
    }


def _group_scope_batches_by_terminal_final(
    *,
    model_dir: Path,
    batches: list[dict[str, Any]],
    model_instance_count: int,
    source_kind: str,
) -> list[list[dict[str, Any]]]:
    if not batches:
        return [[] for _ in range(model_instance_count)]

    final_indices = []
    for index, batch in enumerate(batches):
        if _is_terminal_final_pass(batch):
            final_indices.append(index)
    if len(final_indices) != model_instance_count:
        reason = (
            "scope_library cannot be safely model-instance isolated: "
            "terminal final batch count does not match optimize/model-instance count"
        )
        _parser_log(
            "ERROR",
            "scope_library_model_instance_partition_failed",
            model_dir=model_dir,
            message=reason,
            details={
                "source_kind": source_kind,
                "model_instance_count": model_instance_count,
                "terminal_final_batch_count": len(final_indices),
                "terminal_final_passes": [
                    str(batches[index].get("pass") or "")
                    for index in final_indices[:20]
                ],
            },
        )
        raise ValueError(reason)

    groups: list[list[dict[str, Any]]] = []
    start = 0
    for final_index in final_indices:
        end = final_index + 1
        groups.append(batches[start:end])
        start = final_index + 1

    if start < len(batches):
        ignored_end = start + 20
        ignored_passes = []
        for batch in batches[start:ignored_end]:
            ignored_passes.append(str(batch.get("pass") or ""))
        _parser_log(
            "WARN",
            "scope_library_trailing_batches_ignored",
            model_dir=model_dir,
            message=(
                "scope split batches remain after the terminal final pass grouping; trailing transitions are ignored"
            ),
            details={
                "source_kind": source_kind,
                "ignored_batch_count": len(batches) - start,
                "ignored_passes": ignored_passes,
            },
        )

    return groups


def parse_scope_library_model_instances(
    model_dir: Path, model_instances: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    base = parse_scope_library(model_dir)
    model_instance_count = len(model_instances)
    if model_instance_count <= 1:
        return [base]
    batches = base.get("batches", [])
    source_kind = str(base.get("source_kind") or "")
    if source_kind == "super_kernel.log":
        batch_groups: list[list[dict[str, Any]]] = []
        for model_instance in model_instances:
            begin_line = int(model_instance.get("begin_line") or 1)
            end_line = int(model_instance.get("end_line") or begin_line)
            group: list[dict[str, Any]] = []
            partial_batch_count = 0
            for batch in batches:
                batch_begin = batch.get("begin_line")
                batch_end = batch.get("end_line")
                if not isinstance(batch_begin, int) or not isinstance(batch_end, int):
                    continue
                fully_contained = begin_line <= batch_begin and batch_end <= end_line
                overlaps = not (batch_end < begin_line or batch_begin > end_line)
                if fully_contained:
                    group.append(batch)
                elif overlaps:
                    partial_batch_count += 1
            if partial_batch_count:
                _parser_log(
                    "WARN",
                    "scope_library_partial_batch_assignment",
                    model_dir=model_dir,
                    model_instance_id=_model_instance_id(model_instance),
                    message="scope split transitions only partially overlap the current model instance line range",
                    details={
                        "source_kind": source_kind,
                        "begin_line": begin_line,
                        "end_line": end_line,
                        "partial_batch_count": partial_batch_count,
                    },
                )
            batch_groups.append(group)
    else:
        batch_groups = _group_scope_batches_by_terminal_final(
            model_dir=model_dir,
            batches=batches,
            model_instance_count=model_instance_count,
            source_kind=source_kind,
        )

    return [
        (
            _build_scope_library_from_batches(
                path=base.get("path"),
                source_kind=base.get("source_kind"),
                batch_group=batch_group,
            )
            if batch_group
            else _build_empty_scope_library(
                str(base.get("path") or ""), str(base.get("source_kind") or "")
            )
        )
        for batch_group in batch_groups
    ]


def _infer_node_type(line: str) -> str:
    match = RE_TYPE.search(line)
    if match:
        mapping = {
            "KERNEL": "KERNEL",
            "NOTIFY": "EVENT_NOTIFY",
            "WAIT": "EVENT_WAIT",
            "RESET": "EVENT_RESET",
            "MEMORY_WRITE": "MEMORY_WRITE",
            "MEMORY_WAIT": "MEMORY_WAIT",
        }
        return mapping.get(match.group(1), match.group(1))
    event_match = RE_EVENT_BLOCK.search(line)
    if event_match:
        mapping = {
            "EventNotify": "EVENT_NOTIFY",
            "EventWait": "EVENT_WAIT",
            "EventReset": "EVENT_RESET",
            "MemoryWrite": "MEMORY_WRITE",
            "MemoryWait": "MEMORY_WAIT",
        }
        return mapping.get(event_match.group(1), event_match.group(1).upper())
    if "KernelInfos{" in line or "Kernel name:" in line:
        return "KERNEL"
    return "UNKNOWN"


def _extract_node_entry_from_line(
    line: str, *, line_no: int | None = None
) -> dict[str, Any] | None:
    node_id_match = RE_NODEID.search(line)
    stream_idx_match = RE_STREAM_IDX.search(line)
    node_idx_match = RE_NODE_IDX.search(line)
    if not node_id_match or not stream_idx_match or not node_idx_match:
        return None
    if (
        "Processed node" not in line
        and "PrintScopeNodes" not in line
        and "PrintSKNodesDetail" not in line
    ):
        return None

    stream_id_match = RE_STREAMID.search(line)
    func_name_match = RE_FUNC_NAME.search(line)
    kernel_type_match = RE_KERNEL_TYPE.search(line)
    event_match = RE_EVENT_BLOCK.search(line)
    event_id = None
    event_flag = None
    event_type = None
    if event_match:
        event_type = event_match.group(1)
        inner = event_match.group(2)
        event_id_match = RE_EVENT_ID.search(inner)
        event_flag_match = RE_EVENT_FLAG.search(inner)
        event_id = event_id_match.group(1) if event_id_match else None
        event_flag = event_flag_match.group(1) if event_flag_match else None
    detail = ""
    if "] - " in line:
        detail = line.split("] - ", 1)[1].strip()
    elif "Processed node " in line:
        detail = line.split("Processed node ", 1)[1].strip()
    dev_args_match = RE_DEV_ARGS.search(line)

    return {
        "node_id": int(node_id_match.group(1)),
        "stream_id": int(stream_id_match.group(1)) if stream_id_match else -1,
        "stream_idx_in_graph": int(stream_idx_match.group(1)),
        "node_idx_in_stream": int(node_idx_match.group(1)),
        "node_type": _infer_node_type(line),
        "detail": detail,
        "func_name": func_name_match.group(1) if func_name_match else "",
        "kernel_type": kernel_type_match.group(1) if kernel_type_match else "",
        "event_type": event_type,
        "event_id": event_id,
        "event_flag": event_flag,
        "dev_args": dev_args_match.group(1) if dev_args_match else "",
        "_line_no": line_no,
        "_timestamp": _extract_line_timestamp(line),
    }


def _merge_node_entry(
    base: dict[str, Any] | None, overlay: dict[str, Any]
) -> dict[str, Any]:
    if base is None:
        return dict(overlay)
    merged = dict(base)
    for key, value in overlay.items():
        if value not in ("", None, "UNKNOWN", -1):
            merged[key] = value
    return merged


def _collect_node_library_entries(
    path: Path, line_range: tuple[int, int] | None = None
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line_no, line in _read_indexed_lines(path, line_range):
        entry = _extract_node_entry_from_line(line, line_no=line_no)
        if entry is None:
            continue
        entries.append(entry)
    return entries


def _sanitize_node_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value for key, value in entry.items() if not str(key).startswith("_")
    }
    context = _CURRENT_PARSE_CONTEXT.get()
    if context is not None and not context.validate_json_shadow:
        result["line"] = entry.get("_line_no")
    return result


def _entry_membership_key_sets(
    entries: list[dict[str, Any]],
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]], list[dict[str, Any]]]:
    primary_keys: set[tuple[Any, ...]] = set()
    fallback_keys: set[tuple[Any, ...]] = set()
    unique_entries: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in entries:
        primary_key, fallback_key = _entry_node_keys(entry)
        if primary_key is not None:
            primary_keys.add(primary_key)
        fallback_keys.add(fallback_key)
        dedupe_key = primary_key or fallback_key
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique_entries.append(entry)
    return primary_keys, fallback_keys, unique_entries


def _node_scope_coverage_stats(
    *,
    entries: list[dict[str, Any]],
    raw_scope_library: dict[str, Any],
) -> dict[str, Any]:
    expected_nodes = _expected_model_instance_universe_nodes(raw_scope_library)
    if not expected_nodes:
        expected_nodes = _expected_model_instance_nodes(raw_scope_library)

    entry_primary_keys, entry_fallback_keys, unique_entries = (
        _entry_membership_key_sets(entries)
    )
    missing_nodes: list[dict[str, Any]] = []
    for node in expected_nodes:
        primary_key, fallback_key = _expected_node_keys(node)
        if (
            primary_key is not None and primary_key in entry_primary_keys
        ) or fallback_key in entry_fallback_keys:
            continue
        missing_nodes.append(node)

    expected_primary_keys, expected_fallback_keys = _node_membership_key_sets(
        expected_nodes
    )
    extra_entries: list[dict[str, Any]] = []
    for entry in unique_entries:
        primary_key, fallback_key = _entry_node_keys(entry)
        if (
            primary_key is not None and primary_key in expected_primary_keys
        ) or fallback_key in expected_fallback_keys:
            continue
        extra_entries.append(entry)

    return {
        "entry_count_in_window": len(entries),
        "scope_node_count": len(expected_nodes),
        "scope_node_missing_count": len(missing_nodes),
        "scope_node_missing_sample": _node_key_sample(missing_nodes),
        "processed_node_extra_count": len(extra_entries),
        "processed_node_extra_sample": [
            str(_entry_node_keys(entry)[0] or _entry_node_keys(entry)[1])
            for entry in extra_entries[:10]
        ],
    }


def _build_node_library_from_entries(
    *,
    entries: list[dict[str, Any]],
    path: Path,
    source_kind: str,
    model_dir: Path,
    model_instance_partition_mode: str,
    model_instance_partition_verified: bool,
    model_instance_id: str | None = None,
    stats_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node_by_id: dict[int, dict[str, Any]] = {}
    duplicate_node_ids: list[int] = []
    for entry in entries:
        if entry["node_id"] in node_by_id:
            duplicate_node_ids.append(entry["node_id"])
        node_by_id[entry["node_id"]] = _merge_node_entry(
            node_by_id.get(entry["node_id"]), entry
        )
    stats = {
        "total_entries": len(entries),
        "unique_node_count": len(node_by_id),
        "duplicate_node_id_count": len(duplicate_node_ids),
        "duplicate_node_ids_sample": sorted(set(duplicate_node_ids))[:20],
        "model_instance_partition_mode": model_instance_partition_mode,
        "model_instance_partition_verified": model_instance_partition_verified,
    }
    if stats_overrides:
        stats.update(stats_overrides)
    if model_instance_id:
        stats["model_instance_id"] = model_instance_id
    if not entries:
        _parser_log(
            "WARN",
            "node_library_empty",
            model_dir=model_dir,
            model_instance_id=model_instance_id,
            message="node library parser did not extract any processed nodes",
            details={"source_kind": source_kind, "path": path},
        )
    elif duplicate_node_ids:
        _parser_log(
            "WARN",
            "node_library_duplicate_node_ids",
            model_dir=model_dir,
            model_instance_id=model_instance_id,
            message="node library contains duplicated nodeId entries; later entries overwrite earlier ones",
            details=stats,
        )
    sanitized_nodes = []
    for node_id in sorted(node_by_id):
        sanitized_nodes.append(_sanitize_node_entry(node_by_id[node_id]))
    return {
        "path": str(path),
        "source_kind": source_kind,
        "nodes": sanitized_nodes,
        "stats": stats,
    }


def _expected_model_instance_nodes(
    scope_library: dict[str, Any],
) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for scope in (
        scope_library.get("scopes", []) if isinstance(scope_library, dict) else []
    ):
        for node in (
            scope.get("nodes", []) if isinstance(scope.get("nodes"), list) else []
        ):
            node_id = node.get("node_id")
            stream_idx_in_graph = node.get("stream_idx_in_graph")
            node_idx_in_stream = node.get("node_idx_in_stream")
            if (
                isinstance(node_id, int)
                and isinstance(stream_idx_in_graph, int)
                and isinstance(node_idx_in_stream, int)
            ):
                sequence.append(node)
    return sequence


def _expected_model_instance_universe_nodes(
    scope_library: dict[str, Any],
) -> list[dict[str, Any]]:
    universe: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for batch in (
        scope_library.get("batches", []) if isinstance(scope_library, dict) else []
    ):
        for scope in (
            batch.get("scopes", []) if isinstance(batch.get("scopes"), list) else []
        ):
            for node in (
                scope.get("nodes", []) if isinstance(scope.get("nodes"), list) else []
            ):
                node_id = node.get("node_id")
                stream_idx_in_graph = node.get("stream_idx_in_graph")
                node_idx_in_stream = node.get("node_idx_in_stream")
                if (
                    not isinstance(node_id, int)
                    or not isinstance(stream_idx_in_graph, int)
                    or not isinstance(node_idx_in_stream, int)
                ):
                    continue
                primary_key, fallback_key = _expected_node_keys(node)
                dedupe_key = primary_key or fallback_key
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                universe.append(node)
    return universe


def parse_node_library(model_dir: Path) -> dict[str, Any]:
    model_instances = detect_model_instances(model_dir)
    raw_scope_model_instances = parse_scope_library_model_instances(
        model_dir, model_instances
    )
    libraries = parse_node_library_model_instances(
        model_dir, model_instances, raw_scope_model_instances
    )
    if libraries:
        return libraries[0]
    path, source_kind = _resolve_node_library_source(model_dir)
    return _build_node_library_from_entries(
        entries=[],
        path=path,
        source_kind=source_kind,
        model_dir=model_dir,
        model_instance_partition_mode="single",
        model_instance_partition_verified=True,
    )


def _build_model_instance_node_library(
    *,
    model_dir: Path,
    path: Path,
    source_kind: str,
    entries: list[dict[str, Any]],
    raw_scope_library: dict[str, Any],
    model_instance_partition_mode: str,
    model_instance_partition_verified: bool,
    model_instance_id: str | None = None,
    stats_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = _node_scope_coverage_stats(
        entries=entries, raw_scope_library=raw_scope_library
    )
    if stats_overrides:
        stats.update(stats_overrides)
    return _build_node_library_from_entries(
        entries=entries,
        path=path,
        source_kind=source_kind,
        model_dir=model_dir,
        model_instance_partition_mode=model_instance_partition_mode,
        model_instance_partition_verified=model_instance_partition_verified,
        model_instance_id=model_instance_id,
        stats_overrides=stats,
    )


def _pop_next_available_entry(
    queue: deque[int],
    entries: list[dict[str, Any]],
    used_entry_ids: set[int],
) -> int | None:
    while queue:
        entry_id = queue[0]
        if entry_id in used_entry_ids:
            queue.popleft()
            continue
        return queue.popleft()
    return None


def _build_entry_indexes(
    entries: list[dict[str, Any]],
) -> tuple[dict[tuple[Any, ...], deque[int]], dict[tuple[Any, ...], deque[int]]]:
    primary_index: dict[tuple[Any, ...], deque[int]] = defaultdict(deque)
    fallback_index: dict[tuple[Any, ...], deque[int]] = defaultdict(deque)
    for entry_id, entry in enumerate(entries):
        primary_key, fallback_key = _entry_node_keys(entry)
        if primary_key is not None:
            primary_index[primary_key].append(entry_id)
        fallback_index[fallback_key].append(entry_id)
    return primary_index, fallback_index


def _can_timestamp_partition_entries(
    model_instances: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> bool:
    if not model_instances or not entries:
        return False
    if not any(entry.get("_timestamp") for entry in entries):
        return False
    return all(
        isinstance(model_instance.get("begin_timestamp"), str)
        and model_instance.get("begin_timestamp")
        for model_instance in model_instances
    )


def _entry_matches_model_instance_timestamp(
    entry: dict[str, Any],
    model_instance: dict[str, Any],
) -> bool:
    entry_timestamp = entry.get("_timestamp")
    if not isinstance(entry_timestamp, str) or not entry_timestamp:
        return False
    begin_timestamp = model_instance.get("begin_timestamp")
    end_timestamp = model_instance.get("end_timestamp")
    if (
        isinstance(begin_timestamp, str)
        and begin_timestamp
        and entry_timestamp < begin_timestamp
    ):
        return False
    if (
        isinstance(end_timestamp, str)
        and end_timestamp
        and entry_timestamp > end_timestamp
    ):
        return False
    return True


def _partition_entries_by_model_instance_timestamps(
    *,
    model_dir: Path,
    source_kind: str,
    model_instances: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> tuple[
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    assigned_entries = [[] for _ in model_instances]
    ambiguous_entries: list[dict[str, Any]] = []
    unassigned_entries: list[dict[str, Any]] = []
    untimestamped_entries: list[dict[str, Any]] = []

    for entry in entries:
        if not entry.get("_timestamp"):
            untimestamped_entries.append(entry)
            continue
        candidate_indexes = [
            index
            for index, model_instance in enumerate(model_instances)
            if _entry_matches_model_instance_timestamp(entry, model_instance)
        ]
        if len(candidate_indexes) == 1:
            assigned_entries[candidate_indexes[0]].append(entry)
            continue
        if len(candidate_indexes) > 1:
            ambiguous_entries.append(entry)
            continue
        unassigned_entries.append(entry)

    if ambiguous_entries:
        _parser_log(
            "WARN",
            "node_library_ambiguous_timestamp_entries",
            model_dir=model_dir,
            message=(
                "some node detail entries match multiple optimize/model-instance "
                "timestamp windows and remain unassigned"
            ),
            details={
                "source_kind": source_kind,
                "ambiguous_entry_count": len(ambiguous_entries),
                "ambiguous_key_sample": [
                    str(_entry_node_keys(entry)[0] or _entry_node_keys(entry)[1])
                    for entry in ambiguous_entries[:10]
                ],
            },
        )

    if unassigned_entries:
        _parser_log(
            "WARN",
            "node_library_unassigned_timestamp_entries",
            model_dir=model_dir,
            message=(
                "some timestamped node detail entries fall outside every "
                "optimize/model-instance window and remain unassigned"
            ),
            details={
                "source_kind": source_kind,
                "unassigned_entry_count": len(unassigned_entries),
                "unassigned_key_sample": [
                    str(_entry_node_keys(entry)[0] or _entry_node_keys(entry)[1])
                    for entry in unassigned_entries[:10]
                ],
            },
        )

    if untimestamped_entries:
        _parser_log(
            "WARN",
            "node_library_untimestamped_entries",
            model_dir=model_dir,
            message=(
                "some node detail entries do not carry timestamps and cannot "
                "participate in timestamp-based model-instance partitioning"
            ),
            details={
                "source_kind": source_kind,
                "untimestamped_entry_count": len(untimestamped_entries),
                "untimestamped_key_sample": [
                    str(_entry_node_keys(entry)[0] or _entry_node_keys(entry)[1])
                    for entry in untimestamped_entries[:10]
                ],
            },
        )

    return (
        assigned_entries,
        ambiguous_entries,
        unassigned_entries,
        untimestamped_entries,
    )


def _overlay_membership_entries_with_detail_entries(
    *,
    model_dir: Path,
    model_instance: dict[str, Any],
    detail_source_kind: str,
    membership_entries: list[dict[str, Any]],
    detail_entries: list[dict[str, Any]],
    detail_primary_index: dict[tuple[Any, ...], deque[int]],
    detail_fallback_index: dict[tuple[Any, ...], deque[int]],
    used_detail_entry_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not membership_entries or not detail_entries:
        return (
            [dict(entry) for entry in membership_entries],
            {
                "detail_overlay_source_kind": detail_source_kind,
                "detail_overlay_match_count": 0,
                "detail_overlay_missing_count": len(membership_entries),
            },
        )

    merged_entries: list[dict[str, Any]] = []
    matched_detail_entries: list[dict[str, Any]] = []
    overlay_match_count = 0
    overlay_missing_count = 0

    for membership_entry in membership_entries:
        primary_key, fallback_key = _entry_node_keys(membership_entry)
        detail_entry_id = None
        if primary_key is not None:
            detail_entry_id = _pop_next_available_entry(
                detail_primary_index[primary_key],
                detail_entries,
                used_detail_entry_ids,
            )
        if detail_entry_id is None:
            detail_entry_id = _pop_next_available_entry(
                detail_fallback_index[fallback_key],
                detail_entries,
                used_detail_entry_ids,
            )
        if detail_entry_id is None:
            overlay_missing_count += 1
            merged_entries.append(dict(membership_entry))
            continue
        used_detail_entry_ids.add(detail_entry_id)
        detail_entry = detail_entries[detail_entry_id]
        matched_detail_entries.append(detail_entry)
        overlay_match_count += 1
        merged_entries.append(_merge_node_entry(dict(membership_entry), detail_entry))

    _warn_model_instance_timestamp_drift(
        model_dir=model_dir,
        model_instance=model_instance,
        source_kind=detail_source_kind,
        assigned_entries=matched_detail_entries,
    )
    return (
        merged_entries,
        {
            "detail_overlay_source_kind": detail_source_kind,
            "detail_overlay_match_count": overlay_match_count,
            "detail_overlay_missing_count": overlay_missing_count,
        },
    )


def _warn_model_instance_timestamp_drift(
    *,
    model_dir: Path,
    model_instance: dict[str, Any],
    source_kind: str,
    assigned_entries: list[dict[str, Any]],
) -> None:
    begin_timestamp = model_instance.get("begin_timestamp")
    end_timestamp = model_instance.get("end_timestamp")
    assigned_timestamps = []
    for entry in assigned_entries:
        if entry.get("_timestamp"):
            assigned_timestamps.append(entry.get("_timestamp"))
    if not assigned_timestamps:
        return
    min_timestamp = min(assigned_timestamps)
    max_timestamp = max(assigned_timestamps)
    if _timestamp_outside_range(
        min_timestamp, begin_timestamp, end_timestamp
    ) or _timestamp_outside_range(max_timestamp, begin_timestamp, end_timestamp):
        _parser_log(
            "WARN",
            "node_library_timestamp_drift",
            model_dir=model_dir,
            model_instance_id=_model_instance_id(model_instance),
            message="node detail timestamps drift outside the optimize/model-instance window",
            details={
                "source_kind": source_kind,
                "optimize_begin_timestamp": begin_timestamp,
                "optimize_end_timestamp": end_timestamp,
                "assigned_min_timestamp": min_timestamp,
                "assigned_max_timestamp": max_timestamp,
            },
        )


def _node_membership_key_sets(
    nodes: list[dict[str, Any]],
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    primary_keys: set[tuple[Any, ...]] = set()
    fallback_keys: set[tuple[Any, ...]] = set()
    for node in nodes:
        primary_key, fallback_key = _expected_node_keys(node)
        if primary_key is not None:
            primary_keys.add(primary_key)
        fallback_keys.add(fallback_key)
    return primary_keys, fallback_keys


def parse_node_library_model_instances(
    model_dir: Path,
    model_instances: list[dict[str, Any]],
    raw_scope_model_instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not model_instances:
        model_instances = detect_model_instances(model_dir)
        raw_scope_model_instances = parse_scope_library_model_instances(
            model_dir, model_instances
        )

    path, source_kind = _resolve_node_library_source(model_dir)
    model_instance_count = len(model_instances)
    membership_path = model_dir / "super_kernel.log"
    detail_entries = (
        _collect_node_library_entries(path) if source_kind != "super_kernel.log" else []
    )
    model_instance_libraries: list[dict[str, Any]] = []

    if source_kind != "super_kernel.log" and model_instance_count <= 1:
        return [
            _build_model_instance_node_library(
                model_dir=model_dir,
                path=path,
                source_kind=source_kind,
                entries=detail_entries,
                raw_scope_library=raw_scope_model_instances[0]
                if raw_scope_model_instances
                else {},
                model_instance_partition_mode="single",
                model_instance_partition_verified=True,
                model_instance_id=_model_instance_id(model_instances[0]),
                stats_overrides={
                    "membership_source_kind": source_kind,
                    "membership_partition_basis": "direct_source",
                    "membership_path": str(path),
                    "detail_overlay_source_kind": None,
                    "detail_overlay_match_count": 0,
                    "detail_overlay_missing_count": 0,
                },
            )
        ]

    if (
        source_kind != "super_kernel.log"
        and model_instance_count > 1
        and _can_timestamp_partition_entries(model_instances, detail_entries)
    ):
        (
            assigned_entries,
            ambiguous_entries,
            unassigned_entries,
            untimestamped_entries,
        ) = _partition_entries_by_model_instance_timestamps(
            model_dir=model_dir,
            source_kind=source_kind,
            model_instances=model_instances,
            entries=detail_entries,
        )
        for index, model_instance in enumerate(model_instances):
            raw_scope_library = (
                raw_scope_model_instances[index]
                if index < len(raw_scope_model_instances)
                else raw_scope_model_instances[-1]
            )
            model_instance_libraries.append(
                _build_model_instance_node_library(
                    model_dir=model_dir,
                    path=path,
                    source_kind=source_kind,
                    entries=assigned_entries[index],
                    raw_scope_library=raw_scope_library,
                    model_instance_partition_mode="model_instance_timestamp_range",
                    model_instance_partition_verified=True,
                    model_instance_id=_model_instance_id(model_instance),
                    stats_overrides={
                        "membership_source_kind": source_kind,
                        "membership_partition_basis": "timestamp_window",
                        "membership_path": str(path),
                        "detail_overlay_source_kind": None,
                        "detail_overlay_match_count": 0,
                        "detail_overlay_missing_count": 0,
                    },
                )
            )

        leftover_detail_entries = (
            ambiguous_entries + unassigned_entries + untimestamped_entries
        )
        if leftover_detail_entries:
            assigned_detail_entry_count = 0
            for item in assigned_entries:
                assigned_detail_entry_count += len(item)
            _parser_log(
                "WARN",
                "node_library_leftover_detail_entries",
                model_dir=model_dir,
                message="unassigned node detail entries remain after model-instance timestamp partitioning",
                details={
                    "detail_source_kind": source_kind,
                    "assigned_detail_entry_count": assigned_detail_entry_count,
                    "total_detail_entry_count": len(detail_entries),
                    "leftover_detail_entry_count": len(leftover_detail_entries),
                    "leftover_detail_key_sample": [
                        str(_entry_node_keys(entry)[0] or _entry_node_keys(entry)[1])
                        for entry in leftover_detail_entries[:10]
                    ],
                },
            )
        return model_instance_libraries

    detail_primary_index, detail_fallback_index = (
        _build_entry_indexes(detail_entries) if detail_entries else ({}, {})
    )
    used_detail_entry_ids: set[int] = set()

    for index, model_instance in enumerate(model_instances):
        line_range = (model_instance["begin_line"], model_instance["end_line"])
        membership_entries = (
            _collect_node_library_entries(membership_path, line_range=line_range)
            if membership_path.exists()
            else []
        )
        raw_scope_library = (
            raw_scope_model_instances[index]
            if index < len(raw_scope_model_instances)
            else raw_scope_model_instances[-1]
        )

        if model_instance_count > 1 and not membership_entries:
            reason = (
                f"node_library cannot be safely model-instance partitioned for {_model_instance_id(model_instance)}: "
                "no processed node entries were found inside the super_kernel optimize/model-instance window"
            )
            _parser_log(
                "ERROR",
                "node_library_model_instance_partition_failed",
                model_dir=model_dir,
                model_instance_id=_model_instance_id(model_instance),
                message=reason,
                details={
                    "membership_source_kind": "super_kernel.log",
                    "membership_path": str(membership_path),
                    "begin_line": line_range[0],
                    "end_line": line_range[1],
                },
            )
            raise ValueError(reason)

        entries = membership_entries
        stats_overrides: dict[str, Any] = {
            "membership_source_kind": "super_kernel.log"
            if membership_entries
            else "none",
            "membership_partition_basis": "line_range",
            "membership_path": str(membership_path),
        }
        library_path = path
        library_source_kind = source_kind
        partition_mode = (
            "single" if model_instance_count <= 1 else "model_instance_line_range"
        )

        if membership_entries and source_kind != "super_kernel.log":
            entries, overlay_stats = _overlay_membership_entries_with_detail_entries(
                model_dir=model_dir,
                model_instance=model_instance,
                detail_source_kind=source_kind,
                membership_entries=membership_entries,
                detail_entries=detail_entries,
                detail_primary_index=detail_primary_index,
                detail_fallback_index=detail_fallback_index,
                used_detail_entry_ids=used_detail_entry_ids,
            )
            stats_overrides.update(overlay_stats)
        elif membership_entries:
            stats_overrides.update(
                {
                    "detail_overlay_source_kind": None,
                    "detail_overlay_match_count": 0,
                    "detail_overlay_missing_count": 0,
                }
            )
            library_path = membership_path
            library_source_kind = "super_kernel.log"
        elif model_instance_count <= 1:
            entries = detail_entries
            used_detail_entry_ids.update(range(len(detail_entries)))
            stats_overrides.update(
                {
                    "detail_overlay_source_kind": source_kind,
                    "detail_overlay_match_count": 0,
                    "detail_overlay_missing_count": 0,
                }
            )
        model_instance_libraries.append(
            _build_model_instance_node_library(
                model_dir=model_dir,
                path=library_path,
                source_kind=library_source_kind,
                entries=entries,
                raw_scope_library=raw_scope_library,
                model_instance_partition_mode=partition_mode,
                model_instance_partition_verified=True,
                model_instance_id=_model_instance_id(model_instance),
                stats_overrides=stats_overrides,
            )
        )

    if detail_entries:
        leftover_detail_entries = [
            entry
            for entry_id, entry in enumerate(detail_entries)
            if entry_id not in used_detail_entry_ids
        ]
        if leftover_detail_entries:
            _parser_log(
                "WARN",
                "node_library_leftover_detail_entries",
                model_dir=model_dir,
                message="unassigned node detail overlay entries remain after model-instance line-range partitioning",
                details={
                    "detail_source_kind": source_kind,
                    "assigned_detail_entry_count": len(used_detail_entry_ids),
                    "total_detail_entry_count": len(detail_entries),
                    "leftover_detail_entry_count": len(leftover_detail_entries),
                    "leftover_detail_key_sample": [
                        str(_entry_node_keys(entry)[0] or _entry_node_keys(entry)[1])
                        for entry in leftover_detail_entries[:10]
                    ],
                },
            )

    return model_instance_libraries


def parse_fused_library(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "sk_fused_nodes.log"
    functions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_no, line in _read_indexed_lines(path):
        match = RE_FUSED_HEADER.search(line)
        if match:
            function_text = match.group(1).strip()
            context = _extract_function_context(function_text)
            current = {
                "function_text": context["function_text"],
                "detail": line.strip(),
                "scope_id": int(match.group(2)),
                "node_count": int(match.group(3)),
                "scope_name": context["scope_name"],
                "raw_scope_name": context["raw_scope_name"],
                "sk_id": context["sk_id"],
                "start_node_name": context["start_node_name"],
                "end_node_name": context["end_node_name"],
                "node_ids": [],
                "node_details": [],
                "_line_no": line_no,
                "_timestamp": _extract_line_timestamp(line),
            }
            functions.append(current)
            continue

        match = RE_SCOPE_NODE.search(line)
        if current and match:
            node_id = int(match.group(2))
            current["node_ids"].append(node_id)
            current["node_details"].append(
                {
                    "ordinal": int(match.group(1)),
                    "node_id": node_id,
                    "stream_id": int(match.group(3)),
                    "stream_idx_in_graph": int(match.group(4)),
                    "node_idx_in_stream": int(match.group(5)),
                    "detail": line.strip(),
                }
            )

    for function in functions:
        function["node_ids"] = list(dict.fromkeys(function["node_ids"]))
    return {
        "path": str(path),
        "source_kind": "sk_fused_nodes.log",
        "functions": functions,
    }


def parse_device_task_library(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "sk_device_args.log"
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_queue: str | None = None
    last_task: dict[str, Any] | None = None
    last_dfx: dict[str, Any] | None = None

    for line_no, line in _read_indexed_lines(path):
        if "Dumping device args for function:" in line:
            function_text = line.split("Dumping device args for function:", 1)[
                1
            ].strip()
            context = _extract_function_context(function_text)
            current = {
                "function_text": context["function_text"],
                "detail": line.strip(),
                "scope_name": context["scope_name"],
                "raw_scope_name": context["raw_scope_name"],
                "sk_id": context["sk_id"],
                "start_node_name": context["start_node_name"],
                "end_node_name": context["end_node_name"],
                "header_info": {},
                "header_info_detail": "",
                "queue_details": {},
                "queues": {"AIC": [], "AIV": []},
                "dfx": [],
                "_line_no": line_no,
                "_timestamp": _extract_line_timestamp(line),
            }
            sections.append(current)
            active_queue = None
            last_task = None
            last_dfx = None
            continue

        if current is None:
            continue

        match = RE_HEADER_INFO.search(line)
        if match:
            current["header_info"] = {
                "aic_offset": int(match.group(1)),
                "aiv_offset": int(match.group(2)),
                "counter_offset": int(match.group(3)),
                "ws_offset": int(match.group(4))
                if match.group(4) is not None
                else None,
                "dfx_offset": int(match.group(5)),
                "event_config_offset": int(match.group(6)),
                "node_count": int(match.group(7)),
                "total_size": int(match.group(8)),
            }
            current["header_info_detail"] = line.strip()
            continue

        match = RE_TASK_SECTION.search(line)
        if match:
            active_queue = match.group(1)
            current["queue_details"][active_queue] = line.strip()
            continue

        if active_queue:
            parsed_task = _parse_device_task_entry_line(line, active_queue)
        else:
            parsed_task = None
        if parsed_task is not None and active_queue:
            last_task = parsed_task
            current["queues"][active_queue].append(last_task)
            continue

        match = RE_TASK_ENTRY_ADDR.search(line)
        if match and last_task is not None:
            last_task["entries"].append(match.group(2))
            continue

        match = RE_DFX_INFO.search(line)
        if match:
            last_dfx = {
                "node_index": int(match.group(1)),
                "bin_handle": match.group(2),
                "original_handle": match.group(3),
                "aic_size": match.group(4),
                "aiv_size": match.group(5),
                "entry_aic": {},
                "entry_aiv": {},
                "detail": line.strip(),
            }
            if match.group(6) is not None:
                last_dfx.update(
                    {
                        "num_blocks": int(match.group(6)),
                        "cube_num": int(match.group(7)),
                        "vec_num": int(match.group(8)),
                    }
                )
            current["dfx"].append(last_dfx)
            continue

        if last_dfx is not None:
            for entry_match in RE_DFXINFO_ENTRY_AIC.finditer(line):
                last_dfx["entry_aic"][int(entry_match.group(1))] = (
                    f"0x{entry_match.group(2)}"
                )
            for entry_match in RE_DFXINFO_ENTRY_AIV.finditer(line):
                last_dfx["entry_aiv"][int(entry_match.group(1))] = (
                    f"0x{entry_match.group(2)}"
                )

    parse_context = _CURRENT_PARSE_CONTEXT.get()
    json_validation = (
        _validate_device_task_library_with_json(model_dir, sections)
        if parse_context is None or parse_context.validate_json_shadow
        else {"status": "disabled"}
    )
    return {
        "path": str(path),
        "source_kind": "sk_device_args.log",
        "sections": sections,
        "task_queue_json_validation": json_validation,
    }


def _build_fused_library_subset(
    fused_library: dict[str, Any], functions: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "path": fused_library.get("path"),
        "source_kind": fused_library.get("source_kind"),
        "functions": functions,
    }


def _build_fused_library_subsets(
    fused_library: dict[str, Any], groups: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    subsets = []
    for group in groups:
        subsets.append(_build_fused_library_subset(fused_library, group))
    return subsets


def _build_device_task_library_subset_json_validation(
    device_task_library: dict[str, Any],
) -> dict[str, Any]:
    validation = device_task_library.get("task_queue_json_validation", {})
    if not validation:
        return {}
    return {
        **validation,
        "status": "not_partitioned",
        "base_status": validation.get("status"),
        "reason": "task_queue_json_validation is computed at model-directory scope and is not split per model instance",
    }


def _build_device_task_library_subset(
    device_task_library: dict[str, Any], sections: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "path": device_task_library.get("path"),
        "source_kind": device_task_library.get("source_kind"),
        "sections": sections,
        "task_queue_json_validation": _build_device_task_library_subset_json_validation(
            device_task_library
        ),
    }


def _build_device_task_library_subsets(
    device_task_library: dict[str, Any], groups: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    subsets = []
    for group in groups:
        subsets.append(_build_device_task_library_subset(device_task_library, group))
    return subsets


def _empty_fused_model_instance_libraries(
    model_dir: Path,
    model_instance_count: int,
) -> list[dict[str, Any]]:
    base = parse_fused_library(model_dir)
    return [_build_fused_library_subset(base, []) for _ in range(model_instance_count)]


def _empty_device_model_instance_libraries(
    model_dir: Path,
    model_instance_count: int,
) -> list[dict[str, Any]]:
    base = parse_device_task_library(model_dir)
    reports = []
    for _ in range(model_instance_count):
        reports.append(_build_device_task_library_subset(base, []))
    return reports


def _scope_signatures(
    scope_library: dict[str, Any],
) -> list[frozenset[tuple[Any, ...]]]:
    signatures: list[frozenset[tuple[Any, ...]]] = []
    for scope in (
        scope_library.get("scopes", []) if isinstance(scope_library, dict) else []
    ):
        keys = [
            _group_signature_key(node)
            for node in scope.get("nodes", [])
            if isinstance(node, dict)
        ]
        signatures.append(frozenset(keys))
    return signatures


def _fused_function_signature(function: dict[str, Any]) -> frozenset[tuple[Any, ...]]:
    keys = [
        _group_signature_key(node)
        for node in function.get("node_details", [])
        if isinstance(node, dict)
    ]
    return frozenset(keys)


def _scope_universe_signature(
    scope_library: dict[str, Any],
) -> frozenset[tuple[Any, ...]]:
    return frozenset(
        _group_signature_key(node)
        for node in _expected_model_instance_universe_nodes(scope_library)
    )


def _node_type_signature_map(
    node_library: dict[str, Any],
) -> dict[tuple[Any, ...], str]:
    mapping: dict[tuple[Any, ...], str] = {}
    for node in node_library.get("nodes", []) if isinstance(node_library, dict) else []:
        if not isinstance(node, dict):
            continue
        key = _group_signature_key(node)
        node_type = str(node.get("node_type") or "").strip().upper()
        if node_type:
            mapping[key] = node_type
    return mapping


def _scope_signature_entries(
    scope_library: dict[str, Any],
    node_type_map: dict[tuple[Any, ...], str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for scope in (
        scope_library.get("scopes", []) if isinstance(scope_library, dict) else []
    ):
        scope_signature = frozenset(
            _group_signature_key(node)
            for node in scope.get("nodes", [])
            if isinstance(node, dict)
        )
        scope_nodes = []
        for node in (
            scope.get("nodes", []) if isinstance(scope.get("nodes"), list) else []
        ):
            if not isinstance(node, dict):
                continue
            key = _group_signature_key(node)
            scope_nodes.append(
                {
                    "node_id": node.get("node_id"),
                    "node_type": node_type_map.get(key),
                    "signature": key,
                }
            )
        entries.append(
            {
                "scope_id": scope.get("scope_id"),
                "scope_signature": scope_signature,
                "scope_nodes": scope_nodes,
            }
        )
    return entries


def _filtered_scope_signature_for_fused_function(
    scope_entry: dict[str, Any],
    function_signature: frozenset[tuple[Any, ...]],
) -> tuple[frozenset[tuple[Any, ...]], list[dict[str, Any]], list[dict[str, Any]]]:
    scope_nodes = scope_entry.get("scope_nodes", [])
    scope_signature = scope_entry.get("scope_signature", frozenset())
    extra_keys = scope_signature - function_signature
    filtered_nodes: list[dict[str, Any]] = []
    blocking_nodes: list[dict[str, Any]] = []
    for node in scope_nodes:
        if node["signature"] not in extra_keys:
            continue
        if node.get("node_type") in NON_FUSED_NODE_TYPES:
            filtered_nodes.append(node)
        else:
            blocking_nodes.append(node)
    if blocking_nodes:
        return frozenset(), filtered_nodes, blocking_nodes
    return (
        frozenset(key for key in scope_signature if key not in extra_keys),
        filtered_nodes,
        [],
    )


def _build_fused_function_descriptor(function: dict[str, Any]) -> dict[str, Any]:
    return {
        "function_text": str(function.get("function_text") or ""),
        "sk_id": function.get("sk_id"),
        "start_node_name": str(function.get("start_node_name") or ""),
        "end_node_name": str(function.get("end_node_name") or ""),
        "signature": _fused_function_signature(function),
    }


def _build_device_section_descriptor(section: dict[str, Any]) -> dict[str, Any]:
    queue_node_id_set = set()
    for queue in section.get("queues", {}).values():
        if not isinstance(queue, list):
            continue
        for task in queue:
            if isinstance(task, dict) and isinstance(task.get("node_id"), int):
                queue_node_id_set.add(int(task.get("node_id")))
    queue_node_ids = sorted(queue_node_id_set)
    return {
        "function_text": str(section.get("function_text") or ""),
        "sk_id": section.get("sk_id"),
        "start_node_name": str(section.get("start_node_name") or ""),
        "end_node_name": str(section.get("end_node_name") or ""),
        "queue_node_ids": queue_node_ids,
        "header_node_count": int(section.get("header_info", {}).get("node_count") or 0),
    }


def parse_fused_library_model_instances(
    model_dir: Path,
    model_instances: list[dict[str, Any]],
    raw_scope_model_instances: list[dict[str, Any]],
    node_libraries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = parse_fused_library(model_dir)
    model_instance_count = len(model_instances)
    if model_instance_count <= 1:
        return [base]
    functions = list(base.get("functions", []))
    if not functions:
        empty_groups = [[] for _ in range(model_instance_count)]
        return _build_fused_library_subsets(base, empty_groups)

    expected_signatures = []
    universe_signatures = []
    for index in range(model_instance_count):
        scope_library = (
            raw_scope_model_instances[index]
            if index < len(raw_scope_model_instances)
            else raw_scope_model_instances[-1]
        )
        node_library = (
            node_libraries[index] if index < len(node_libraries) else node_libraries[-1]
        )
        node_type_map = _node_type_signature_map(node_library)
        expected_signatures.append(
            _scope_signature_entries(scope_library, node_type_map)
        )
        universe_signatures.append(_scope_universe_signature(scope_library))

    groups: list[list[dict[str, Any]]] = [[] for _ in range(model_instance_count)]
    used_indices: set[int] = set()

    for index, model_instance in enumerate(model_instances):
        for scope_entry in expected_signatures[index]:
            scope_signature = scope_entry["scope_signature"]
            if scope_signature and all(
                node.get("node_type") in NON_FUSED_NODE_TYPES
                for node in scope_entry.get("scope_nodes", [])
            ):
                _parser_log(
                    "WARN",
                    "fused_library_empty_filtered_scope_signature",
                    model_dir=model_dir,
                    model_instance_id=_model_instance_id(model_instance),
                    message="filtered final-scope signature is empty; no fused function is required for this scope",
                    details={
                        "scope_id": scope_entry.get("scope_id"),
                        "raw_scope_signature_count": len(scope_signature),
                        "filtered_non_fused_node_count": len(scope_signature),
                    },
                )
                continue
            matched_candidates: list[
                tuple[int, frozenset[tuple[Any, ...]], list[dict[str, Any]]]
            ] = []
            for function_index, function in enumerate(functions):
                if function_index in used_indices:
                    continue
                function_signature = _fused_function_signature(function)
                if not function_signature.issubset(scope_signature):
                    continue
                filtered_signature, filtered_nodes, blocking_nodes = (
                    _filtered_scope_signature_for_fused_function(
                        scope_entry,
                        function_signature,
                    )
                )
                if blocking_nodes:
                    continue
                matched_candidates.append(
                    (function_index, filtered_signature, filtered_nodes)
                )
            if not matched_candidates:
                reason = (
                    "fused_library cannot be safely model-instance partitioned "
                    f"for {_model_instance_id(model_instance)}: "
                    "no fused function matches the expected final-scope signature"
                )
                _parser_log(
                    "ERROR",
                    "fused_library_model_instance_partition_failed",
                    model_dir=model_dir,
                    model_instance_id=_model_instance_id(model_instance),
                    message=reason,
                    details={
                        "expected_scope_signature_sample": _signature_sample(
                            scope_signature
                        ),
                        "expected_scope_count": len(expected_signatures[index]),
                        "scope_id": scope_entry.get("scope_id"),
                    },
                )
                raise ModelInstancePartitionError(
                    reason,
                    partial_result=_build_fused_library_subsets(base, groups),
                    failed_model_instance_ids=[_model_instance_id(model_instance)],
                )
            if len(matched_candidates) > 1:
                candidate_signatures = {
                    candidate_signature
                    for _, candidate_signature, _ in matched_candidates
                }
                if len(candidate_signatures) == 1:
                    _parser_log(
                        "WARN",
                        "fused_library_equivalent_candidates",
                        model_dir=model_dir,
                        model_instance_id=_model_instance_id(model_instance),
                        message=(
                            "multiple equivalent fused functions match the current "
                            "final-scope signature; using the earliest unused function"
                        ),
                        details={
                            "scope_id": scope_entry.get("scope_id"),
                            "candidate_function_count": len(matched_candidates),
                            "candidate_function_sample": [
                                str(
                                    functions[candidate_index].get("function_text")
                                    or ""
                                )
                                for candidate_index, _, _ in matched_candidates[:10]
                            ],
                        },
                    )
                else:
                    reason = (
                        "fused_library cannot be safely model-instance partitioned "
                        f"for {_model_instance_id(model_instance)}: "
                        "multiple fused functions match the expected final-scope signature"
                    )
                    _parser_log(
                        "ERROR",
                        "fused_library_model_instance_partition_failed",
                        model_dir=model_dir,
                        model_instance_id=_model_instance_id(model_instance),
                        message=reason,
                        details={
                            "scope_id": scope_entry.get("scope_id"),
                            "candidate_function_count": len(matched_candidates),
                            "candidate_function_sample": [
                                str(
                                    functions[candidate_index].get("function_text")
                                    or ""
                                )
                                for candidate_index, _, _ in matched_candidates[:10]
                            ],
                        },
                    )
                    raise ModelInstancePartitionError(
                        reason,
                        partial_result=_build_fused_library_subsets(base, groups),
                        failed_model_instance_ids=[_model_instance_id(model_instance)],
                    )
            matched_index, filtered_signature, filtered_nodes = matched_candidates[0]
            if filtered_nodes:
                type_counts = Counter(node["node_type"] for node in filtered_nodes)
                _parser_log(
                    "WARN",
                    "fused_library_scope_signature_filtered",
                    model_dir=model_dir,
                    model_instance_id=_model_instance_id(model_instance),
                    message="filtered non-fused node types from final-scope signature before fused matching",
                    details={
                        "scope_id": scope_entry.get("scope_id"),
                        "raw_scope_signature_count": len(scope_signature),
                        "filtered_scope_signature_count": len(filtered_signature),
                        "filtered_non_fused_node_count": len(filtered_nodes),
                        "filtered_non_fused_type_counts": dict(type_counts),
                        "filtered_non_fused_node_sample": filtered_nodes[:10],
                    },
                )
            used_indices.add(matched_index)
            groups[index].append(functions[matched_index])

    leftover_function_indices = []
    for index in range(len(functions)):
        if index not in used_indices:
            leftover_function_indices.append(index)
    ambiguous_functions: list[dict[str, Any]] = []
    unmatched_functions: list[dict[str, Any]] = []
    for function_index in leftover_function_indices:
        function = functions[function_index]
        signature = _fused_function_signature(function)
        candidates = [
            index
            for index, universe_signature in enumerate(universe_signatures)
            if signature and signature.issubset(universe_signature)
        ]
        if len(candidates) == 1:
            groups[candidates[0]].append(function)
            used_indices.add(function_index)
            _parser_log(
                "WARN",
                "fused_library_non_final_assignment",
                model_dir=model_dir,
                model_instance_id=_model_instance_id(model_instances[candidates[0]]),
                message="assigned extra fused function using model-instance universe signature fallback",
                details={
                    "function_text": function.get("function_text"),
                    "signature_sample": _signature_sample(signature),
                },
            )
            continue
        if len(candidates) > 1:
            ambiguous_functions.append(function)
            continue
        unmatched_functions.append(function)

    if ambiguous_functions:
        reason = (
            "fused_library cannot be safely model-instance partitioned: "
            "extra fused functions match multiple model-instance universes"
        )
        _parser_log(
            "ERROR",
            "fused_library_model_instance_partition_failed",
            model_dir=model_dir,
            message=reason,
            details={
                "ambiguous_function_count": len(ambiguous_functions),
                "ambiguous_function_sample": [
                    str(function.get("function_text") or "")
                    for function in ambiguous_functions[:10]
                ],
            },
        )
        raise ModelInstancePartitionError(
            reason,
            partial_result=_build_fused_library_subsets(base, groups),
        )

    if unmatched_functions:
        reason = (
            "fused_library cannot be safely model-instance partitioned: "
            "extra fused functions do not match any model-instance universe"
        )
        _parser_log(
            "ERROR",
            "fused_library_model_instance_partition_failed",
            model_dir=model_dir,
            message=reason,
            details={
                "unmatched_function_count": len(unmatched_functions),
                "unmatched_function_sample": [
                    str(function.get("function_text") or "")
                    for function in unmatched_functions[:10]
                ],
            },
        )
        raise ModelInstancePartitionError(
            reason,
            partial_result=_build_fused_library_subsets(base, groups),
        )

    return _build_fused_library_subsets(base, groups)


def parse_device_task_library_model_instances(
    model_dir: Path,
    model_instances: list[dict[str, Any]],
    fused_model_instances: list[dict[str, Any]],
    raw_scope_model_instances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = parse_device_task_library(model_dir)
    model_instance_count = len(model_instances)
    if model_instance_count <= 1:
        return [base]
    sections = list(base.get("sections", []))
    if not sections:
        return [
            _build_device_task_library_subset(base, [])
            for _ in range(model_instance_count)
        ]

    instance_universe_node_ids = []
    for index in range(model_instance_count):
        raw_scope_model_instance = (
            raw_scope_model_instances[index]
            if index < len(raw_scope_model_instances)
            else raw_scope_model_instances[-1]
        )
        node_ids = set()
        for node in _expected_model_instance_universe_nodes(raw_scope_model_instance):
            node_id = node.get("node_id")
            if isinstance(node_id, int):
                node_ids.add(int(node_id))
        instance_universe_node_ids.append(node_ids)

    remaining_expected: list[list[dict[str, Any]]] = [
        [
            _build_fused_function_descriptor(function)
            for function in item.get("functions", [])
        ]
        for item in fused_model_instances
    ]
    groups: list[list[dict[str, Any]]] = [[] for _ in range(model_instance_count)]
    leftover_sections: list[dict[str, Any]] = []

    for section in sections:
        descriptor = _build_device_section_descriptor(section)
        matched_instance = None
        matched_index = None
        matched_reason = None

        for index in range(model_instance_count):
            for expected_index, expected in enumerate(remaining_expected[index]):
                if (
                    descriptor["function_text"]
                    and descriptor["function_text"] == expected["function_text"]
                ):
                    matched_instance = index
                    matched_index = expected_index
                    matched_reason = "function_text"
                    break
            if matched_instance is not None:
                break

        if matched_instance is None:
            tuple_matches = []
            for index in range(model_instance_count):
                for expected_index, expected in enumerate(remaining_expected[index]):
                    has_start_end = bool(
                        descriptor["start_node_name"] and descriptor["end_node_name"]
                    )
                    sk_id_matches = (
                        descriptor["sk_id"] is not None
                        and descriptor["sk_id"] == expected["sk_id"]
                    )
                    start_end_matches = (
                        descriptor["start_node_name"] == expected["start_node_name"]
                        and descriptor["end_node_name"] == expected["end_node_name"]
                    )
                    if has_start_end and sk_id_matches and start_end_matches:
                        tuple_matches.append((index, expected_index))
            if len(tuple_matches) == 1:
                matched_instance, matched_index = tuple_matches[0]
                matched_reason = "sk_id_start_end"

        if matched_instance is None:
            sk_id_matches = []
            for index in range(model_instance_count):
                for expected_index, expected in enumerate(remaining_expected[index]):
                    if (
                        descriptor["sk_id"] is not None
                        and descriptor["sk_id"] == expected["sk_id"]
                    ):
                        sk_id_matches.append((index, expected_index))
            if len(sk_id_matches) == 1:
                matched_instance, matched_index = sk_id_matches[0]
                matched_reason = "sk_id"

        if matched_instance is None and descriptor["queue_node_ids"]:
            queue_node_id_set = set(descriptor["queue_node_ids"])
            node_universe_matches = []
            for index, universe_node_ids in enumerate(instance_universe_node_ids):
                if queue_node_id_set.issubset(universe_node_ids):
                    node_universe_matches.append(index)
            if len(node_universe_matches) == 1:
                candidate_index = node_universe_matches[0]
                if len(remaining_expected[candidate_index]) == 1:
                    matched_instance = candidate_index
                    matched_index = 0
                    matched_reason = "queue_node_ids_unique_instance"
                elif not remaining_expected[candidate_index]:
                    matched_instance = candidate_index
                    matched_reason = "queue_node_ids_extra_section"

        if matched_instance is None:
            leftover_sections.append(section)
            continue

        groups[matched_instance].append(section)
        if matched_index is not None:
            remaining_expected[matched_instance].pop(matched_index)
        if matched_reason not in {None, "function_text"}:
            _parser_log(
                "WARN",
                "device_task_library_fallback_assignment",
                model_dir=model_dir,
                model_instance_id=_model_instance_id(model_instances[matched_instance]),
                message="assigned device task section using fallback anchors",
                details={
                    "fallback_reason": matched_reason,
                    "section_function_text": descriptor["function_text"],
                    "section_sk_id": descriptor["sk_id"],
                    "queue_node_ids": descriptor["queue_node_ids"][:10],
                },
            )

    missing_expectations = [
        index
        for index, expected_items in enumerate(remaining_expected)
        if expected_items
    ]
    if missing_expectations:
        reason = (
            "device_task_library cannot be safely model-instance partitioned: "
            "some expected device sections could not be matched to any model instance"
        )
        _parser_log(
            "ERROR",
            "device_task_library_model_instance_partition_failed",
            model_dir=model_dir,
            message=reason,
            details={
                "missing_model_instances": [
                    _model_instance_id(model_instances[index])
                    for index in missing_expectations
                ],
                "missing_function_text_sample": [
                    str(item.get("function_text") or "")
                    for index in missing_expectations
                    for item in remaining_expected[index][:5]
                ],
            },
        )
        raise ModelInstancePartitionError(
            reason,
            partial_result=_build_device_task_library_subsets(base, groups),
            failed_model_instance_ids=[
                _model_instance_id(model_instances[index])
                for index in missing_expectations
            ],
        )

    if leftover_sections:
        reason = (
            "device_task_library cannot be safely model-instance partitioned: "
            "some device task sections could not be uniquely assigned"
        )
        _parser_log(
            "ERROR",
            "device_task_library_model_instance_partition_failed",
            model_dir=model_dir,
            message=reason,
            details={
                "leftover_section_count": len(leftover_sections),
                "leftover_function_text_sample": [
                    str(section.get("function_text") or "")
                    for section in leftover_sections[:10]
                ],
            },
        )
        raise ModelInstancePartitionError(
            reason,
            partial_result=_build_device_task_library_subsets(base, groups),
        )

    return _build_device_task_library_subsets(base, groups)


def _match_scope_update(
    scope: dict[str, Any], scope_updates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    scope_streams = scope.get("streams", [])
    best_match: tuple[int, dict[str, Any]] | None = None
    for scope_update in scope_updates:
        matched = 0
        for update_stream in scope_update.get("streams", []):
            for scope_stream in scope_streams:
                if (
                    scope_stream.get("stream_idx")
                    == update_stream.get(
                        "stream_idx_in_graph", update_stream.get("stream_id")
                    )
                    and scope_stream.get("head_node")
                    == update_stream.get("head_node_id")
                    and scope_stream.get("tail_node")
                    == update_stream.get("tail_node_id")
                ):
                    matched += 1
                    break
        if matched and (best_match is None or matched > best_match[0]):
            best_match = (matched, scope_update)
    return best_match[1] if best_match else None


def _scope_graph_backed_updates(
    scope: dict[str, Any],
    node_update_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scope_nodes = [node for node in scope.get("nodes", []) if isinstance(node, dict)]
    scope_nodes_by_id = {
        int(node.get("node_id")): node
        for node in scope_nodes
        if isinstance(node.get("node_id"), int)
    }
    graph_backed_updates: list[dict[str, Any]] = []
    for row in node_update_rows:
        node_id = row.get("node_id")
        if not isinstance(node_id, int) or node_id not in scope_nodes_by_id:
            continue
        scope_node = scope_nodes_by_id[node_id]
        graph_backed_updates.append(
            {
                "identity_kind": "graph_node",
                "graph_node_key": _graph_node_identity_payload(scope_node, node_id),
                "node_id": node_id,
                "stream_id": scope_node.get("stream_id"),
                "stream_idx_in_graph": scope_node.get("stream_idx_in_graph"),
                "node_idx_in_stream": scope_node.get("node_idx_in_stream"),
                "node_type": scope_node.get("node_type"),
                "kernel_name": scope_node.get("func_name"),
                "update_type": row.get("type"),
                "addr": row.get("addr"),
                "value": row.get("value"),
                "flag": row.get("flag"),
                "op_info_ptr": row.get("op_info_ptr"),
                "op_info_size": row.get("op_info_size"),
                "func_handle": row.get("func_handle"),
                "args": row.get("args"),
                "args_size": row.get("args_size"),
                "num_blocks": row.get("num_blocks"),
                "detail": row.get("detail"),
                "line": row.get("line"),
            }
        )
    return graph_backed_updates


def _match_scope_device_sections(
    scope: dict[str, Any], device_sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scope_name_set = {
        _normalize_scope_name(name)
        for name in (scope.get("scope_names") or [])
        if isinstance(name, str) and name.strip()
    }
    if not scope_name_set:
        scope_name_set = {"(none)"}
    matches: list[dict[str, Any]] = []
    for section in device_sections:
        section_name_candidates = {
            _normalize_scope_name(section.get("scope_name")),
            _normalize_scope_name(section.get("raw_scope_name")),
        }
        function_text = section.get("function_text")
        if isinstance(function_text, str) and function_text.strip():
            context = _extract_function_context(function_text)
            section_name_candidates.add(
                _normalize_scope_name(context.get("scope_name"))
            )
            section_name_candidates.add(
                _normalize_scope_name(context.get("raw_scope_name"))
            )
        section_blob = " ".join(
            str(section.get(key) or "")
            for key in ("function_text", "detail", "scope_name", "raw_scope_name")
        )
        if scope_name_set & section_name_candidates or any(
            f"scopeName: {scope_name}" in section_blob
            or f"{scope_name}__skId:" in section_blob
            for scope_name in scope_name_set
        ):
            matches.append(section)
    return matches


def _bind_scope_device_sections(
    scope_library: dict[str, Any],
    device_sections: list[dict[str, Any]],
) -> tuple[dict[Any, list[dict[str, Any]]], dict[str, Any]]:
    scopes = scope_library.get("scopes", []) if isinstance(scope_library, dict) else []
    scope_ids = {
        scope.get("scope_id")
        for scope in scopes
        if isinstance(scope, dict) and scope.get("scope_id") is not None
    }
    sections_by_scope_id: dict[Any, list[dict[str, Any]]] = {}
    direct_bound_count = 0
    fallback_bound_count = 0
    unbound_samples: list[dict[str, Any]] = []

    for section in device_sections:
        bound_scope_id = section.get("bound_scope_id")
        if bound_scope_id in scope_ids:
            direct_bound_count += 1
            section_with_binding = dict(section)
            section_with_binding["_scope_binding_reason"] = str(
                section.get("bound_scope_binding_source") or "bound_scope_id"
            )
            sections_by_scope_id.setdefault(bound_scope_id, []).append(
                section_with_binding
            )
            continue

        fallback_matches = []
        for scope in scopes:
            if _match_scope_device_sections(scope, [section]):
                fallback_matches.append(scope)
        if len(fallback_matches) == 1:
            fallback_bound_count += 1
            fallback_scope_id = fallback_matches[0].get("scope_id")
            section_with_binding = dict(section)
            section_with_binding["_scope_binding_reason"] = "fallback_scope_name"
            sections_by_scope_id.setdefault(fallback_scope_id, []).append(
                section_with_binding
            )
            continue

        _append_limited_sample(
            unbound_samples,
            {
                "scope_name": section.get("scope_name"),
                "raw_scope_name": section.get("raw_scope_name"),
                "sk_id": section.get("sk_id"),
                "bound_scope_id": bound_scope_id,
                "bound_scope_binding_source": section.get("bound_scope_binding_source"),
                "function_text": section.get("function_text"),
                "match_state": "ambiguous" if fallback_matches else "no_match",
                "fallback_match_scope_ids": [
                    scope.get("scope_id")
                    for scope in fallback_matches
                    if isinstance(scope, dict)
                ],
            },
        )

    return sections_by_scope_id, {
        "direct_bound_section_count": direct_bound_count,
        "fallback_bound_section_count": fallback_bound_count,
        "unbound_section_count": len(unbound_samples),
        "unbound_section_samples": unbound_samples,
    }


def _scope_synthesized_custom_nodes(
    scope: dict[str, Any],
    *,
    scope_export_ordinal: int,
    device_sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    synthesized_nodes: list[dict[str, Any]] = []
    for section in device_sections:
        for queue_name, tasks in (section.get("queues", {}) or {}).items():
            for task in tasks or []:
                if str(task.get("identity_kind") or "") != "synthesized_custom":
                    continue
                base_key = str(task.get("custom_instance_key") or "").strip()
                custom_descriptor = (
                    task.get("custom_descriptor", {})
                    if isinstance(task.get("custom_descriptor"), dict)
                    else {}
                )
                synthesized_nodes.append(
                    {
                        "identity_kind": "synthesized_custom",
                        "custom_instance_key": (
                            f"scope:{scope_export_ordinal}:{base_key}"
                            if base_key
                            else (
                                f"scope:{scope_export_ordinal}:"
                                f"queue:{str(queue_name).upper()}:"
                                f"task:{task.get('task_id')}"
                            )
                        ),
                        "scope_export_ordinal": scope_export_ordinal,
                        "scope_id": scope.get("scope_id"),
                        "scope_name": section.get("scope_name"),
                        "sk_id": section.get("sk_id"),
                        "queue_name": str(queue_name).upper(),
                        "custom_kind": str(task.get("task_type") or "").upper(),
                        "task_id": task.get("task_id"),
                        "task_index_raw": task.get("task_index"),
                        "node_id_raw": task.get("node_id"),
                        "args": task.get("args"),
                        "entries": list(task.get("entries", [])),
                        "section_ordinal": custom_descriptor.get("section_ordinal"),
                        "queue_custom_ordinal": custom_descriptor.get(
                            "queue_custom_ordinal"
                        ),
                        "binding_reason": str(
                            section.get("_scope_binding_reason") or "bound_scope_id"
                        ),
                        "detail": task.get("detail"),
                    }
                )
    return synthesized_nodes


def build_scope_library_export(
    scope_library: dict[str, Any],
    update_execution: dict[str, Any],
    device_task_library: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exported_scopes: list[dict[str, Any]] = []
    node_update_rows = update_execution.get("node_update_results", [])
    device_sections = (
        device_task_library.get("sections", [])
        if isinstance(device_task_library, dict)
        else []
    )
    sections_by_scope_id, synthesized_custom_binding_diagnostics = (
        _bind_scope_device_sections(
            scope_library,
            device_sections,
        )
    )
    for scope_export_ordinal, scope in enumerate(scope_library.get("scopes", [])):
        node_ids = [
            node.get("node_id")
            for node in scope.get("nodes", [])
            if node.get("node_id") is not None
        ]
        details = [
            node.get("detail", "")
            for node in scope.get("nodes", [])
            if node.get("detail", "")
        ]
        scope_update = _match_scope_update(
            scope, update_execution.get("scope_updates", [])
        )
        scope_node_id_set = set(node_ids)
        scope_node_details = [
            row.get("detail", "")
            for row in node_update_rows
            if row.get("node_id") in scope_node_id_set and row.get("detail")
        ]
        graph_backed_updates = _scope_graph_backed_updates(scope, node_update_rows)
        matched_device_sections = list(
            sections_by_scope_id.get(scope.get("scope_id"), [])
        )
        synthesized_custom_nodes = _scope_synthesized_custom_nodes(
            scope,
            scope_export_ordinal=scope_export_ordinal,
            device_sections=matched_device_sections,
        )
        has_any_update_payload = bool(
            scope_update is not None
            or scope_node_details
            or graph_backed_updates
            or synthesized_custom_nodes
        )
        update_payload = None
        if has_any_update_payload:
            update_streams = []
            update_stream_items = (
                scope_update.get("streams", []) if scope_update else []
            )
            for item in update_stream_items:
                update_streams.append(
                    {
                        "stream_id": item.get("stream_id"),
                        "stream_idx_in_graph": item.get(
                            "stream_idx_in_graph", item.get("stream_id")
                        ),
                        "head_node_id": item.get("head_node_id"),
                        "tail_node_id": item.get("tail_node_id"),
                        "node_size": item.get("node_size"),
                        "custom_param_size": item.get("custom_param_size"),
                        "visited_nodes": item.get("visited_nodes"),
                        "begin_detail": item.get("begin_detail", ""),
                        "end_detail": item.get("end_detail", ""),
                    }
                )
            fallback_bound_section_count = 0
            for section in matched_device_sections:
                if (
                    str(section.get("_scope_binding_reason") or "")
                    == "fallback_scope_name"
                ):
                    fallback_bound_section_count += 1
            update_payload = {
                "begin_detail": scope_update.get("begin_detail", "")
                if scope_update
                else "",
                "finish_detail": scope_update.get("finish_detail", "")
                if scope_update
                else "",
                "stream_count": scope_update.get("stream_count") if scope_update else 0,
                "update_total_nodes": scope_update.get("update_total_nodes")
                if scope_update
                else None,
                "node_details": scope_node_details,
                "graph_backed_updates": graph_backed_updates,
                "synthesized_custom_nodes": synthesized_custom_nodes,
                "diagnostics": {
                    "graph_backed_update_count": len(graph_backed_updates),
                    "synthesized_custom_count": len(synthesized_custom_nodes),
                    "matched_device_section_count": len(matched_device_sections),
                    "fallback_bound_section_count": fallback_bound_section_count,
                    "has_scope_update": scope_update is not None,
                },
                "streams": update_streams,
            }
        exported_scopes.append(
            {
                "scope_id": scope.get("scope_id"),
                "scope_export_ordinal": scope_export_ordinal,
                "node_count": scope.get("node_count"),
                "stream_count": scope.get("stream_count"),
                "scope_bit_flags": scope.get("scope_bit_flags"),
                "scope_names": scope.get("scope_names"),
                "node_ids": node_ids,
                "details": details,
                "streams": scope.get("streams", []),
                "update": update_payload,
            }
        )
    transitions_by_scope_id = scope_library.get("transitions_by_scope_id", {})
    return {
        "path": scope_library.get("path"),
        "source_kind": scope_library.get("source_kind"),
        "synthesized_custom_binding_diagnostics": synthesized_custom_binding_diagnostics,
        "scopes": [
            {
                **scope,
                "transitions": transitions_by_scope_id.get(scope.get("scope_id"), []),
            }
            for scope in exported_scopes
        ],
    }


def _export_fused_library(fused_library: dict[str, Any]) -> dict[str, Any]:
    exported_functions: list[dict[str, Any]] = []
    for function in (
        fused_library.get("functions", []) if isinstance(fused_library, dict) else []
    ):
        exported_functions.append(
            {
                "detail": function.get("detail"),
                "scope_id": function.get("scope_id"),
                "node_count": function.get("node_count"),
                "scope_name": function.get("scope_name"),
                "sk_id": function.get("sk_id"),
                "start_node_name": function.get("start_node_name"),
                "end_node_name": function.get("end_node_name"),
                "node_ids": function.get("node_ids", []),
                "node_details": function.get("node_details", []),
            }
        )
    return {
        "path": fused_library.get("path") if isinstance(fused_library, dict) else None,
        "source_kind": fused_library.get("source_kind")
        if isinstance(fused_library, dict)
        else None,
        "functions": exported_functions,
    }


def _export_device_task_library(device_task_library: dict[str, Any]) -> dict[str, Any]:
    exported_sections: list[dict[str, Any]] = []
    for section in (
        device_task_library.get("sections", [])
        if isinstance(device_task_library, dict)
        else []
    ):
        exported_queues: dict[str, list[dict[str, Any]]] = {}
        for queue_name, tasks in (section.get("queues", {}) or {}).items():
            exported_queues[str(queue_name)] = []
            for task in tasks or []:
                exported_task = dict(task)
                if "custom_descriptor" in exported_task and isinstance(
                    exported_task["custom_descriptor"], dict
                ):
                    exported_task["custom_descriptor"] = dict(
                        exported_task["custom_descriptor"]
                    )
                exported_queues[str(queue_name)].append(exported_task)
        exported_sections.append(
            {
                "detail": section.get("detail"),
                "scope_name": section.get("scope_name"),
                "sk_id": section.get("sk_id"),
                "bound_scope_id": section.get("bound_scope_id"),
                "bound_scope_binding_source": section.get("bound_scope_binding_source"),
                "start_node_name": section.get("start_node_name"),
                "end_node_name": section.get("end_node_name"),
                "header_info": section.get("header_info", {}),
                "header_info_detail": section.get("header_info_detail", ""),
                "queue_details": section.get("queue_details", {}),
                "queues": exported_queues
                or section.get("queues", {"AIC": [], "AIV": []}),
                "dfx": section.get("dfx", []),
                "task_identity_diagnostics": section.get(
                    "task_identity_diagnostics", {}
                ),
            }
        )
    return {
        "path": device_task_library.get("path")
        if isinstance(device_task_library, dict)
        else None,
        "source_kind": device_task_library.get("source_kind")
        if isinstance(device_task_library, dict)
        else None,
        "sections": exported_sections,
        "task_queue_json_validation": device_task_library.get(
            "task_queue_json_validation", {}
        )
        if isinstance(device_task_library, dict)
        else {},
        "task_identity_diagnostics": device_task_library.get(
            "task_identity_diagnostics", {}
        )
        if isinstance(device_task_library, dict)
        else {},
    }


def build_node_update_registry(update_execution: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in update_execution.get("node_update_results", []):
        rows.append(dict(row))
    return {
        "path": update_execution.get("path"),
        "source_kind": "super_kernel.log",
        "rows": rows,
    }


_FUSION_ANALYSIS_MODULE: Any | None = None


def _load_fusion_analysis_module() -> Any:
    global _FUSION_ANALYSIS_MODULE
    if _FUSION_ANALYSIS_MODULE is not None:
        return _FUSION_ANALYSIS_MODULE
    try:
        import fusion_analysis as module
    except ModuleNotFoundError:
        module_path = Path(__file__).with_name("fusion_analysis.py")
        spec = importlib.util.spec_from_file_location(
            "sk_tools_fusion_analysis", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to load optional fusion analysis module: {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    _FUSION_ANALYSIS_MODULE = module
    return module


def _empty_fusion_analysis(model_path: Path, error: str = "") -> dict[str, Any]:
    source_paths = {
        name: model_path / name
        for name in ("sk_scope_split.log", "sk_fusion_fail_reasons.log")
    }
    return {
        "schema_version": "1",
        "status": "unavailable",
        "summary": {
            "final_pass": None,
            "final_region_count": 0,
            "successful_sk_count": 0,
            "boundary_count": 0,
            "operator_triggered_boundary_count": 0,
            "control_boundary_count": 0,
            "unclassified_boundary_count": 0,
            "failure_record_count": 0,
            "unique_failure_node_count": 0,
        },
        "operator_impacts": [],
        "boundaries": [],
        "node_failures": [],
        "sources": {
            name: {
                "path": str(path),
                "status": "unavailable",
                "line_range": None,
                "line_count": 0,
                "record_count": 0,
                "error_count": 0,
                "unknown_line_count": 0,
                "sample_lines": [],
            }
            for name, path in source_paths.items()
        },
        "coverage": {"parser_error": error} if error else {},
    }


def _build_optional_fusion_analysis(
    *,
    model_path: Path,
    model_instance: dict[str, Any],
    model_instance_count: int,
    raw_scope_library: dict[str, Any],
    node_library: dict[str, Any],
    fused_library: dict[str, Any],
) -> dict[str, Any]:
    try:
        module = _load_fusion_analysis_module()
        line_range = (
            (model_instance["begin_line"], model_instance["end_line"])
            if model_instance_count > 1
            else None
        )
        return module.build_fusion_analysis(
            model_path,
            node_library=node_library,
            scope_library=raw_scope_library,
            fused_library=fused_library,
            model_instance_range=line_range,
            model_instance_id=_model_instance_id(model_instance),
        )
    except Exception as exc:
        _parser_log(
            "WARN",
            "fusion_analysis_unavailable",
            model_dir=model_path,
            model_instance_id=_model_instance_id(model_instance),
            message="optional fusion-boundary analysis failed; existing graph data remains available",
            details={"error": str(exc)},
        )
        return _empty_fusion_analysis(model_path, str(exc))


def _build_model_instance_report_from_parts(
    *,
    model_path_str: str,
    model_instance: dict[str, Any],
    model_instance_count: int,
    raw_scope_library: dict[str, Any],
    node_library: dict[str, Any],
    fused_library: dict[str, Any],
    device_task_library: dict[str, Any],
    collection_error: str = "",
    super_kernel_line_items: list[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path_str)
    device_task_library = annotate_device_task_graph_identity(
        device_task_library,
        fused_library,
        node_library,
    )
    line_range = (
        (model_instance["begin_line"], model_instance["end_line"])
        if model_instance_count > 1
        else None
    )
    log_path = model_path / "super_kernel.log"
    if super_kernel_line_items is None:
        update_execution = parse_update_execution(model_path, line_range=line_range)
        super_kernel = parse_super_kernel_log(model_path, line_range=line_range)
    else:
        update_execution = _parse_update_execution_line_items(
            super_kernel_line_items, log_path
        )
        super_kernel = _parse_super_kernel_line_items(super_kernel_line_items, log_path)
    scope_library = build_scope_library_export(
        raw_scope_library, update_execution, device_task_library
    )
    exported_fused_library = _export_fused_library(fused_library)
    exported_device_task_library = _export_device_task_library(device_task_library)
    node_update_registry = build_node_update_registry(update_execution)
    dfx_evidence = build_dfx_evidence(
        update_execution, device_task_library, fused_library, super_kernel
    )
    dfx_library = build_dfx_library(
        model_path,
        device_task_library,
        fused_library,
        super_kernel,
        dfx_evidence,
    )
    fusion_analysis = _build_optional_fusion_analysis(
        model_path=model_path,
        model_instance=model_instance,
        model_instance_count=model_instance_count,
        raw_scope_library=raw_scope_library,
        node_library=node_library,
        fused_library=fused_library,
    )
    return _validate_model_instance_report(
        model_path,
        {
            "model_dir": str(model_path),
            "model_ri": _extract_model_ri(model_path),
            "model_instance_id": _model_instance_id(model_instance),
            "model_instance_index": _model_instance_index(model_instance),
            "model_instance_count": model_instance_count,
            "model_instance_range": (
                {
                    "begin_line": model_instance["begin_line"],
                    "end_line": model_instance["end_line"],
                    "begin_timestamp": model_instance.get("begin_timestamp"),
                    "end_timestamp": model_instance.get("end_timestamp"),
                }
                if model_instance_count > 1
                else None
            ),
            "scope_library": {
                "path": scope_library.get("path"),
                "source_kind": scope_library.get("source_kind"),
                "synthesized_custom_binding_diagnostics": scope_library.get(
                    "synthesized_custom_binding_diagnostics", {}
                ),
                "scopes": scope_library.get("scopes", []),
                "fused_library": exported_fused_library,
                "device_task_library": exported_device_task_library,
            },
            "graph_library": {
                "node_library": node_library,
                "node_update_registry": node_update_registry,
            },
            "dfx_library": dfx_library,
            "fusion_analysis": fusion_analysis,
            "collection_error": collection_error,
        },
    )


def _build_model_instance_report_worker(payload: dict[str, Any]) -> dict[str, Any]:
    return _build_model_instance_report_from_parts(**payload)


def collect_update_model_instance_reports(
    model_dir: str | os.PathLike[str],
    *,
    model_instances: list[dict[str, Any]] | None = None,
    instance_workers: int = 1,
) -> list[dict[str, Any]]:
    model_path = find_model_dir(model_dir)
    context = _CURRENT_PARSE_CONTEXT.get()
    if context is None:
        with use_parse_context(ParseContext(model_path)):
            return collect_update_model_instance_reports(
                model_path,
                model_instances=model_instances,
                instance_workers=instance_workers,
            )
    model_instances = model_instances or detect_model_instances(model_path)
    model_instance_count = len(model_instances)

    if model_instance_count <= 1:
        raw_scope_library = parse_scope_library(model_path)
        fused_library = parse_fused_library(model_path)
        node_library = parse_node_library(model_path)
        return [
            _build_model_instance_report_from_parts(
                model_path_str=str(model_path),
                model_instance=model_instances[0],
                model_instance_count=1,
                raw_scope_library=raw_scope_library,
                node_library=node_library,
                fused_library=fused_library,
                device_task_library=parse_device_task_library(model_path),
                collection_error="",
            )
        ]

    raw_scope_model_instances = parse_scope_library_model_instances(
        model_path, model_instances
    )
    node_libraries = parse_node_library_model_instances(
        model_path, model_instances, raw_scope_model_instances
    )
    collection_errors: dict[str, list[str]] = {}
    for model_instance in model_instances:
        collection_errors[_model_instance_id(model_instance)] = []

    try:
        fused_model_instances = parse_fused_library_model_instances(
            model_path, model_instances, raw_scope_model_instances, node_libraries
        )
    except ModelInstancePartitionError as exc:
        fused_model_instances = list(exc.partial_result) if exc.partial_result else []
        while len(fused_model_instances) < model_instance_count:
            fused_model_instances.append(
                _build_fused_library_subset(parse_fused_library(model_path), [])
            )
        failed_ids = exc.failed_model_instance_ids or list(collection_errors.keys())
        for model_instance_id in failed_ids:
            collection_errors.setdefault(model_instance_id, []).append(str(exc))
        fused_model_instances = fused_model_instances[:model_instance_count]
        device_model_instances = _empty_device_model_instance_libraries(
            model_path, model_instance_count
        )
    else:
        try:
            device_model_instances = parse_device_task_library_model_instances(
                model_path,
                model_instances,
                fused_model_instances,
                raw_scope_model_instances,
            )
        except ModelInstancePartitionError as exc:
            device_model_instances = (
                list(exc.partial_result) if exc.partial_result else []
            )
            while len(device_model_instances) < model_instance_count:
                device_model_instances.append(
                    _build_device_task_library_subset(
                        parse_device_task_library(model_path), []
                    )
                )
            failed_ids = exc.failed_model_instance_ids or list(collection_errors.keys())
            for model_instance_id in failed_ids:
                collection_errors.setdefault(model_instance_id, []).append(str(exc))
            device_model_instances = device_model_instances[:model_instance_count]

    payloads: list[dict[str, Any]] = []
    for index, model_instance in enumerate(model_instances):
        line_range = (model_instance["begin_line"], model_instance["end_line"])
        payloads.append(
            {
                "model_path_str": str(model_path),
                "model_instance": model_instance,
                "model_instance_count": model_instance_count,
                "raw_scope_library": raw_scope_model_instances[index]
                if index < len(raw_scope_model_instances)
                else raw_scope_model_instances[-1],
                "node_library": node_libraries[index]
                if index < len(node_libraries)
                else node_libraries[-1],
                "fused_library": fused_model_instances[index]
                if index < len(fused_model_instances)
                else fused_model_instances[-1],
                "device_task_library": device_model_instances[index]
                if index < len(device_model_instances)
                else device_model_instances[-1],
                "collection_error": "; ".join(
                    collection_errors.get(_model_instance_id(model_instance), [])
                ),
                "super_kernel_line_items": _read_indexed_lines(
                    model_path / "super_kernel.log", line_range
                ),
            }
        )
    if instance_workers > 1 and len(payloads) > 1:
        reports: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=min(instance_workers, len(payloads))
        ) as executor:
            futures = [
                executor.submit(_build_model_instance_report_worker, payload)
                for payload in payloads
            ]
            for future in as_completed(futures):
                reports.append(future.result())
        reports.sort(key=lambda item: int(item.get("model_instance_index") or 0))
        return reports

    reports = []
    for payload in payloads:
        reports.append(_build_model_instance_report_from_parts(**payload))
    reports.sort(key=lambda item: int(item.get("model_instance_index") or 0))
    return reports


def collect_update_report(model_dir: str | os.PathLike[str]) -> dict[str, Any]:
    reports = collect_update_model_instance_reports(model_dir)
    return reports[0] if reports else {}


def _normalize_output_dir(
    input_path: str | os.PathLike[str], output_dir: str | None
) -> Path:
    base_dir = Path(output_dir).resolve() if output_dir else None
    if base_dir is not None:
        return base_dir
    input_path_obj = _resolve_input_dir(input_path)
    try:
        model_dirs = find_model_dirs(input_path_obj)
    except FileNotFoundError:
        model_dirs = []
    model_asset_root = (
        infer_model_asset_root(input_path_obj, model_dirs) if model_dirs else None
    )
    result_root = infer_result_root(input_path_obj, model_asset_root, model_dirs)
    return result_root / "reports" / "data"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract scope-library.json 和 graph-library.json from SK model artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", help="model 目录、模型资产目录、结果目录或包含 model_* 的目录"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出目录（默认按 input 目录推导为 <input>/reports/data）",
    )
    parser.add_argument("--scope-library", help="显式指定 scope-library.json 输出路径")
    parser.add_argument("--graph-library", help="显式指定 graph-library.json 输出路径")
    parser.add_argument("--dfx-library", help="显式指定 dfx-library.json 输出路径")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if bool(args.scope_library) ^ bool(args.graph_library):
        parser.error("--scope-library 与 --graph-library 必须同时指定，或都不指定。")

    try:
        model_dir = find_model_dir(args.input)
    except FileNotFoundError as exc:
        _emit(f"[ERROR] 无法定位 model 目录: {exc}")
        return 1

    update_report = collect_update_report(model_dir)
    output_dir = _normalize_output_dir(args.input, args.output_dir)
    scope_library_output = Path(args.scope_library) if args.scope_library else None
    graph_library_output = Path(args.graph_library) if args.graph_library else None
    dfx_library_output = Path(args.dfx_library) if args.dfx_library else None
    scope_path, graph_path, dfx_path = write_update_libraries(
        update_report,
        output_dir,
        scope_library_output=scope_library_output,
        graph_library_output=graph_library_output,
        dfx_library_output=dfx_library_output,
    )
    _emit(f"[OK] 已生成 scope 库: {scope_path}")
    _emit(f"[OK] 已生成 graph 库: {graph_path}")
    _emit(f"[OK] 已生成 dfx 库: {dfx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
