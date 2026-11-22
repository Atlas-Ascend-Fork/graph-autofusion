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

"""Generate run-level SK diagnostics and report indexes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import hashlib
import heapq
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, NamedTuple

try:
    import ijson
except ImportError:  # pragma: no cover - exercised by fallback tests via monkeypatch
    ijson = None

try:
    from tqdm import tqdm
except (
    ImportError
):  # pragma: no cover - exercised by fallback behavior when tqdm is not installed
    tqdm = None

from sk_scope_visualizer import (
    ScopeGraphModel,
    ScopeGraphRenderer,
    ScopeLibrarySource,
)
from sk_task_queue_visualizer import (
    TaskQueueLibrarySource,
    TaskQueueModel,
    TaskQueueRenderer,
)
from sk_visualizer_shared import (
    COMMON_EMPTY_STATE_CSS,
    COMMON_GRAPH_FRAME_CSS,
    COMMON_GRAPH_TOOLBAR_CSS,
    COMMON_SCOPE_SECTION_CSS,
    COMMON_STANDARD_TABLE_JS,
    COMMON_TOOLBAR_CSS,
    COMMON_VISUALIZER_CSS,
    build_report_styles,
    render_empty_note,
    render_graph_toolbar,
    render_page_header,
    render_paginated_table_shell,
    render_report_header,
    render_report_intro_section,
    render_report_section,
    render_report_summary,
    render_standard_table_panel,
    render_scope_section_block,
    render_view_nav,
)
from sk_library_extractor import (
    collect_update_model_instance_reports,
    find_model_dirs,
    infer_model_asset_root,
    infer_result_root,
    write_update_libraries,
)


RE_RUN_NAME = re.compile(r"^\d{8}-\d{6}-(.+)$")
EVENT_FILE_PATTERNS = ("sk_event_dev_device_*.json", "sk_prof_device_*.json")
EVENT_FILE_PREFIXES = ("sk_event_dev_device_", "sk_prof_device_")
EVENT_ASSET_KEY = "sk_event_dev_device_*.json"
EVENT_ASSET_NAME = "sk_event_dev_device_*.json / sk_prof_device_*.json"
LOG_FILE_COUNT_LIMIT = 1000
PARSE_CACHE_VERSION = "sk-model-analysis-parse-cache-v1"


def _emit(message: object = "", *, file: Any = None, end: str = "\n") -> None:
    stream = sys.stdout if file is None else file
    stream.write(f"{message}{end}")


PRESENTATION_LIMITS = {
    "scope_nodes": 24,
    "scope_scopes": 4,
    "task_rows": 24,
    "task_sections": 4,
    "update_secondary": 20,
    "hang_primary": 5,
    "hang_secondary": 8,
    "performance_matrix": 8,
    "portal_primary_views": 5,
    "portal_next_steps": 5,
    "portal_signal_rows": 2,
}
EXCEPTION_RULES = [
    ("aicore_exception", re.compile(r"aicore exception", re.IGNORECASE)),
    ("sk_counter_info", re.compile(r"SkCounterInfo", re.IGNORECASE)),
    ("sk_dfx_info", re.compile(r"SkDfxInfo", re.IGNORECASE)),
    ("function_name", re.compile(r"Function name", re.IGNORECASE)),
    ("core_id", re.compile(r"\bcoreId\s*=", re.IGNORECASE)),
    ("exception_core_symbol", re.compile(r"exception core", re.IGNORECASE)),
    ("runtime_warning_or_error", re.compile(r"\[(WARNING|ERROR|ERR)\]", re.IGNORECASE)),
]
HARD_FAILURE_SIGNAL_KEYS = {
    "aicore_exception",
    "sk_counter_info",
    "sk_dfx_info",
    "exception_core_symbol",
}
HARD_FAILURE_TEXT = re.compile(
    r"(aicore exception|segmentation fault|core dumped|fatal|abort|exception core|skcounterinfo|skdfxinfo)",
    re.IGNORECASE,
)
RUNTIME_WARNING_TEXT = re.compile(r"\[(WARNING|ERROR|ERR)\]", re.IGNORECASE)
ENVIRONMENT_NOISE_TEXT = re.compile(
    r"(profiling|dump|callback|register|plugin|heartbeat|keepalive|upload|download|path not found|permission denied)",
    re.IGNORECASE,
)
ASSET_GUIDANCE = {
    "model_dir": {
        "diagnosis_value": "是所有基于 SK meta 的分析锚点，包括 update、scope、queue、trace 以及大部分 hang/performance 视图。",
        "acquisition_hint": "采集 SK 结果时，保留包含 `<模型资产目录>/<process>/model_*` 的结果目录。",
        "fallback_if_missing": "你仍然可以查看原始 `log/` 和 `kernel_meta/`，但依赖 SK meta 的报告会不可用。",
    },
    "log_dir": {
        "diagnosis_value": "提供 plog 和设备侧运行时上下文，用于 warning、stream、callback 和运行期排查。",
        "acquisition_hint": "保留运行期日志目录，并与 SK 结果根目录一起采集。",
        "fallback_if_missing": "如果 `super_kernel.log` 还在，仍可继续排查，但运行期/设备侧上下文会更弱。",
    },
    "super_kernel.log": {
        "diagnosis_value": "展示优化阶段、update 执行过程以及 SK 侧高层异常上下文。",
        "acquisition_hint": "保留解析出的 `model_*` 目录中的 `super_kernel.log`。",
        "fallback_if_missing": "你仍可查看 plog 或 device log，但阶段和 update 执行证据会不完整。",
    },
    "sk_scope_split.log": {
        "diagnosis_value": "解释 scope 划分结果、流分组以及 scope/node 布局。",
        "acquisition_hint": "把 `model_*` 下的 scope split 日志作为 SK meta 输出的一部分保留下来。",
        "fallback_if_missing": "你仍可查看 fused node 和 queue，但 scope 结构和 scope 报告会受限。",
    },
    "sk_fused_nodes.log": {
        "diagnosis_value": "展示哪些节点进入了每个 SK function，以及哪些 kernel 被融合到一起。",
        "acquisition_hint": "保留 model 目录下的 fused-node dump 日志。",
        "fallback_if_missing": "你仍可查看 task queue 和 phase，但对 fused function 的解释会更弱。",
    },
    "sk_device_args.log": {
        "diagnosis_value": "提供队列布局、任务索引、header 信息和 DFX 相关证据。",
        "acquisition_hint": "保留 model 目录下的 device-args dump 日志。",
        "fallback_if_missing": "你仍可查看 scope 和 fused-node 数据，但 queue/task 诊断与 update 映射会明显退化。",
    },
    "sk_node_detail.log": {
        "diagnosis_value": "提供 node 级结构，供 node tracing 和更细粒度的 scope 诊断使用。",
        "acquisition_hint": "保留 `UpdateNodeScopeBitFlags` 期间生成的 node-detail dump。",
        "fallback_if_missing": "你仍可查看 scope 和 update 报告，但 node tracing 以及部分 node 级映射会不可用。",
    },
    EVENT_ASSET_KEY: {
        "diagnosis_value": "提供 SK function、node 和设备侧性能关联所需的 time-event 覆盖。",
        "acquisition_hint": (
            "采集结果目录时，保留 `sk_event_dev_device_*.json` 或 `sk_prof_device_*.json` 这类 SK event recorder 输出。"
        ),
        "fallback_if_missing": "你仍可做结构和 queue 分析，但时间覆盖和性能判断会更弱。",
    },
}


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


def _entry_model_instance_id(
    entry: dict[str, Any] | None, *, default: str = "mi01"
) -> str:
    if not isinstance(entry, dict):
        return default
    return _normalize_model_instance_id(entry.get("model_instance_id"), default=default)


def _entry_model_instance_index(
    entry: dict[str, Any] | None, *, default: int = 1
) -> int:
    if not isinstance(entry, dict):
        return default
    return int(entry.get("model_instance_index") or default)


def _entry_model_instance_count(
    entry: dict[str, Any] | None, *, default: int = 1
) -> int:
    if not isinstance(entry, dict):
        return default
    return int(entry.get("model_instance_count") or default)


def _entry_collection_error(entry: dict[str, Any] | None) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("collection_error") or "")


REPORT_GUIDANCE = {
    "scope_graph": {
        "title": "Scope View",
        "diagnosis_value": "用原始交互式 scope 图直接查看 scope round、节点和事件关系。",
        "viewer_hint": "在浏览器中打开 `scope-graph.html`。",
        "required_assets": ["model_dir", "sk_scope_split.log"],
        "view_kind": "interactive_graph",
        "recommended_for": "scope_debug",
    },
    "task_queue_graph": {
        "title": "TaskQue View",
        "diagnosis_value": "用原始交互式 queue 图直接查看 AIC/AIV task 布局和派发结构。",
        "viewer_hint": "在浏览器中打开 `task-queue-graph.html`。",
        "required_assets": ["model_dir", "sk_device_args.log"],
        "view_kind": "interactive_graph",
        "recommended_for": "queue_debug",
    },
    "node_trace": {
        "title": "节点追踪",
        "diagnosis_value": "以 tracing 形式查看 node/stream 关系，并和 update/queue 视图交叉定位。",
        "viewer_hint": "在 `edge://tracing` 或 `chrome://tracing` 中加载 `node-trace.json`。",
        "required_assets": ["model_dir", "sk_node_detail.log"],
        "structured_output": "node-trace_meta.json",
        "view_kind": "structured_report",
        "recommended_for": "trace_debug",
    },
    "hang_crash_report": {
        "title": "DFX View",
        "diagnosis_value": "查看模型资产中的阶段推进、runtime payload、异常函数映射与 core symbol 线索。",
        "viewer_hint": "在浏览器中打开 `hang-crash-report.html`。",
        "required_assets": ["model_dir", "super_kernel.log"],
        "view_kind": "diagnostic_summary",
        "recommended_for": "hang_debug",
    },
    "performance_report": {
        "title": "Analysis View",
        "diagnosis_value": "以事件主视角查看时间证据、结构锚点、队列关联和后续深链入口。",
        "viewer_hint": "在浏览器中打开 `performance-report.html`。",
        "required_assets": [
            "model_dir",
            "sk_fused_nodes.log",
            "sk_device_args.log",
            "sk_scope_split.log",
            EVENT_ASSET_KEY,
        ],
        "view_kind": "diagnostic_summary",
        "recommended_for": "performance_debug",
    },
    "run_portal": {
        "title": "Run Portal",
        "diagnosis_value": "只保留四个 View 的入口，不在这里展开诊断细节。",
        "viewer_hint": "在浏览器中打开 `run-portal.html`。",
        "required_assets": [],
        "view_kind": "diagnostic_summary",
        "recommended_for": "diagnostic_overview",
    },
}


def _resolve_input_dir(input_path: str | os.PathLike[str]) -> Path:
    path = Path(input_path).resolve()
    if path.is_file():
        return path.parent
    return path


def _find_model_asset_root(input_dir: Path, model_dirs: list[Path]) -> Path | None:
    if not model_dirs:
        return None
    return infer_model_asset_root(input_dir, model_dirs)


def _find_result_root(
    input_dir: Path, model_asset_root: Path | None, model_dirs: list[Path]
) -> Path:
    return infer_result_root(input_dir, model_asset_root, model_dirs)


def _classify_input(
    input_dir: Path,
    result_root: Path,
    model_asset_root: Path | None,
    model_dir: Path | None,
) -> str:
    if model_dir is not None and input_dir == model_dir:
        return "model_dir"
    if model_asset_root is not None and input_dir == model_asset_root:
        return "model_asset_root"
    if input_dir == result_root:
        return "result_root"
    if (input_dir / "log").is_dir() and input_dir != result_root:
        return "log_root"
    return "generic_dir"


def _discover_context(input_path: str | os.PathLike[str]) -> dict[str, Any]:
    input_dir = _resolve_input_dir(input_path)
    model_dirs: list[Path] = []
    try:
        model_dirs = find_model_dirs(input_dir)
    except FileNotFoundError:
        model_dirs = []
    model_dir = model_dirs[0] if model_dirs else None

    model_asset_root = _find_model_asset_root(input_dir, model_dirs)
    result_root = _find_result_root(input_dir, model_asset_root, model_dirs)
    reports_dir = result_root / "reports"
    log_dir = result_root / "log"
    kernel_meta_dir = result_root / "kernel_meta"

    return {
        "input_dir": input_dir,
        "input_classification": _classify_input(
            input_dir, result_root, model_asset_root, model_dir
        ),
        "result_root": result_root,
        "reports_dir": reports_dir,
        "log_dir": log_dir,
        "model_asset_root": model_asset_root,
        "sk_meta_dir": model_asset_root,
        "kernel_meta_dir": kernel_meta_dir,
        "model_dir": model_dir,
        "model_dirs": model_dirs,
    }


def _capped_log_file_count(log_dir: Path, limit: int = LOG_FILE_COUNT_LIMIT) -> int:
    if not log_dir.is_dir():
        return 0
    count = 0
    for _path in log_dir.glob("**/*.log"):
        count += 1
        if count >= limit:
            return limit
    return count


def _discover_shallow_event_files(context: dict[str, Any]) -> list[str]:
    model_dirs = context.get("model_dirs", [])
    model_asset_root = context.get("model_asset_root")
    search_dirs: dict[str, Path] = {}
    for model_dir in model_dirs:
        if not isinstance(model_dir, Path):
            continue
        search_dirs[str(model_dir.resolve())] = model_dir
        if model_dir.parent != model_dir:
            search_dirs[str(model_dir.parent.resolve())] = model_dir.parent
    if isinstance(model_asset_root, Path) and model_asset_root.is_dir():
        search_dirs[str(model_asset_root.resolve())] = model_asset_root

    collected: list[str] = []
    for directory in search_dirs.values():
        for pattern in EVENT_FILE_PATTERNS:
            for path in directory.glob(pattern):
                if path.is_file():
                    collected.append(str(path))
    return sorted(set(collected))


def _discover_assets(
    context: dict[str, Any], event_files: list[str] | None = None
) -> dict[str, Any]:
    input_dir = context["input_dir"]
    result_root = context["result_root"]
    log_dir = context["log_dir"]
    model_asset_root = context["model_asset_root"]
    kernel_meta_dir = context["kernel_meta_dir"]
    model_dir = context["model_dir"]
    model_dirs = context.get("model_dirs", [])
    discovered_event_files = sorted(
        set(
            event_files
            if event_files is not None
            else _discover_shallow_event_files(context)
        )
    )
    log_file_count = _capped_log_file_count(log_dir)

    def _any_model_file(name: str) -> bool:
        return any((item / name).is_file() for item in model_dirs)

    inventory = {
        "input_path": str(input_dir),
        "input_classification": context["input_classification"],
        "directories": {
            "result_root": {"path": str(result_root), "exists": result_root.is_dir()},
            "log_dir": {"path": str(log_dir), "exists": log_dir.is_dir()},
            "model_asset_root": {
                "path": str(context["model_asset_root"])
                if context.get("model_asset_root")
                else None,
                "exists": bool(
                    context.get("model_asset_root")
                    and context["model_asset_root"].is_dir()
                ),
            },
            "sk_meta_dir": {
                "path": str(model_asset_root) if model_asset_root else None,
                "exists": bool(model_asset_root and model_asset_root.is_dir()),
            },
            "kernel_meta_dir": {
                "path": str(kernel_meta_dir),
                "exists": kernel_meta_dir.is_dir(),
            },
            "model_dir": {
                "path": str(model_dir) if model_dir else None,
                "exists": bool(model_dir and model_dir.is_dir()),
            },
            "model_dirs": {
                "paths": [str(item) for item in model_dirs],
                "count": len(model_dirs),
            },
            "reports_dir": {
                "path": str(context["reports_dir"]),
                "exists": context["reports_dir"].is_dir(),
            },
        },
        "files": {
            "super_kernel.log": _any_model_file("super_kernel.log"),
            "sk_scope_split.log": _any_model_file("sk_scope_split.log"),
            "sk_fused_nodes.log": _any_model_file("sk_fused_nodes.log"),
            "sk_node_detail.log": _any_model_file("sk_node_detail.log"),
            "sk_device_args.log": _any_model_file("sk_device_args.log"),
        },
        "event_files": discovered_event_files,
        "event_file_count": len(discovered_event_files),
        "log_file_count": log_file_count,
        "log_file_count_capped": bool(
            log_dir.is_dir() and log_file_count >= LOG_FILE_COUNT_LIMIT
        ),
    }
    return inventory


def _asset_status(asset_inventory: dict[str, Any], asset_name: str) -> bool:
    directories = asset_inventory["directories"]
    files = asset_inventory["files"]
    if asset_name == "model_dir":
        return directories["model_dir"]["exists"]
    if asset_name == "log_dir":
        return directories["log_dir"]["exists"]
    if asset_name == EVENT_ASSET_KEY:
        return asset_inventory["event_file_count"] > 0
    return bool(files.get(asset_name))


def _build_asset_advice(asset_inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    advice: dict[str, dict[str, Any]] = {}
    for asset_name, guidance in ASSET_GUIDANCE.items():
        advice[asset_name] = {
            "status": "available"
            if _asset_status(asset_inventory, asset_name)
            else "missing",
            "diagnosis_value": guidance["diagnosis_value"],
            "acquisition_hint": guidance["acquisition_hint"],
            "fallback_if_missing": guidance["fallback_if_missing"],
        }
    return advice


def _build_missing_assets(asset_inventory: dict[str, Any]) -> list[dict[str, str]]:
    missing = []
    asset_advice = _build_asset_advice(asset_inventory)
    directories = asset_inventory["directories"]
    files = asset_inventory["files"]
    if not directories["model_dir"]["exists"]:
        missing.append(
            {
                "name": "model_dir",
                "reason": "在输入路径下没有找到 model_* 目录。",
                "expected": "请提供包含 SK 日志的结果根目录、模型资产目录或 model_* 目录。",
                "diagnosis_value": asset_advice["model_dir"]["diagnosis_value"],
                "acquisition_hint": asset_advice["model_dir"]["acquisition_hint"],
                "fallback_if_missing": asset_advice["model_dir"]["fallback_if_missing"],
                "related_reports": [
                    "scope_graph",
                    "task_queue_graph",
                    "node_trace",
                    "hang_crash_report",
                    "performance_report",
                ],
            }
        )
        return missing

    report_requirements = {
        "super_kernel.log": ["hang_crash_report"],
        "sk_scope_split.log": ["scope_graph", "performance_report"],
        "sk_fused_nodes.log": ["performance_report"],
        "sk_node_detail.log": ["node_trace"],
        "sk_device_args.log": ["task_queue_graph", "performance_report"],
        EVENT_ASSET_KEY: ["performance_report"],
    }
    for name, present in files.items():
        if present:
            continue
        guidance = asset_advice[name]
        missing.append(
            {
                "name": name,
                "reason": f"在解析出的 model 目录下缺少 `{name}`。",
                "expected": "请提供包含标准 SK meta 日志集合的 model_* 目录。",
                "diagnosis_value": guidance["diagnosis_value"],
                "acquisition_hint": guidance["acquisition_hint"],
                "fallback_if_missing": guidance["fallback_if_missing"],
                "related_reports": report_requirements.get(name, []),
            }
        )
    if asset_inventory["event_file_count"] == 0:
        guidance = asset_advice[EVENT_ASSET_KEY]
        missing.append(
            {
                "name": EVENT_ASSET_NAME,
                "canonical_name": EVENT_ASSET_KEY,
                "reason": "在结果根目录下没有找到 SK time-event JSON 文件。",
                "expected": f"请提供保留了 `{EVENT_ASSET_NAME}` 输出的结果目录。",
                "diagnosis_value": guidance["diagnosis_value"],
                "acquisition_hint": guidance["acquisition_hint"],
                "fallback_if_missing": guidance["fallback_if_missing"],
                "related_reports": report_requirements[EVENT_ASSET_KEY],
            }
        )
    return missing


def _can_run(
    asset_inventory: dict[str, Any], report_name: str
) -> tuple[bool, str | None]:
    files = asset_inventory["files"]
    directories = asset_inventory["directories"]
    if report_name == "scope_graph":
        if not directories["model_dir"]["exists"]:
            return False, "model directory is missing"
        if not files["sk_scope_split.log"]:
            return False, "missing sk_scope_split.log"
        return True, None
    if report_name == "task_queue_graph":
        if not directories["model_dir"]["exists"]:
            return False, "model directory is missing"
        if not files["sk_device_args.log"]:
            return False, "missing sk_device_args.log"
        return True, None
    if report_name == "node_trace":
        if not directories["model_dir"]["exists"]:
            return False, "model directory is missing"
        if not files["sk_node_detail.log"]:
            return False, "missing sk_node_detail.log"
        return True, None
    if report_name == "hang_crash_report":
        if not directories["model_dir"]["exists"]:
            return False, "model directory is missing"
        if not files["super_kernel.log"]:
            return False, "missing super_kernel.log"
        return True, None
    if report_name == "performance_report":
        if not directories["model_dir"]["exists"]:
            return False, "model directory is missing"
        has_structure_asset = False
        for name in ("sk_fused_nodes.log", "sk_device_args.log", "sk_scope_split.log"):
            if files[name]:
                has_structure_asset = True
                break
        if asset_inventory["event_file_count"] == 0 and not has_structure_asset:
            return (
                False,
                "missing timing and structural assets for performance diagnosis",
            )
        return True, None
    return False, f"unknown report type: {report_name}"


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _infer_entry_name(run_dir: Path) -> str:
    match = RE_RUN_NAME.match(run_dir.name)
    if match:
        return match.group(1)
    return run_dir.name


def _display_optional(value: Any, default: str = "无") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"none", "unknown", "(none)"}:
        return default
    return text


def _split_href_suffix(href: str) -> tuple[str, str]:
    for marker in ("?", "#"):
        if marker in href:
            idx = href.index(marker)
            return href[:idx], href[idx:]
    return href, ""


def _portal_href(href: str) -> str:
    path, suffix = _split_href_suffix(href)
    if not path:
        return href
    if path.startswith(("http://", "https://", "edge://", "chrome://")):
        return href
    if path.startswith("reports/"):
        reports_prefix_len = len("reports/")
        path = path[reports_prefix_len:]
    if path == "run-portal.html":
        return path + suffix
    if path.endswith(".html"):
        if not path.startswith("views/"):
            path = f"views/{path}"
        return path + suffix
    if path.endswith((".json", ".md")):
        if not path.startswith("data/"):
            path = f"data/{path}"
        return path + suffix
    return path + suffix


def _display_join(values: list[Any], default: str = "无") -> str:
    rendered = [
        _display_optional(item, "").strip()
        for item in values
        if _display_optional(item, "").strip()
    ]
    return "、".join(rendered) if rendered else default


def _event_group_label(
    group_type: str, sk_id: int | None, node_id: int | None, device: str
) -> str:
    if group_type == "sk" and sk_id is not None:
        return f"SK {sk_id}"
    if group_type == "node" and node_id is not None:
        return f"节点 {node_id}"
    return f"设备 {device}"


def _event_display_name(
    raw_name: str, sk_id: int | None, node_id: int | None, device: str
) -> str:
    if sk_id is not None and node_id is not None:
        return f"SK {sk_id} / 节点 {node_id}"
    if sk_id is not None:
        return f"SK {sk_id} 事件"
    if node_id is not None:
        return f"节点 {node_id} 事件"
    if device:
        return f"设备 {device} 事件"
    return _display_optional(raw_name, "未命名事件")


def _copy_raw_event_files(event_file_paths: list[str], data_dir: Path) -> list[str]:
    copied: list[str] = []
    data_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in event_file_paths:
        src = Path(raw_path)
        if not src.is_file():
            continue
        dst = data_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def _event_files_by_process(
    event_file_paths: list[str], model_asset_root: Path | None
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for raw_path in event_file_paths:
        src = Path(raw_path)
        if not src.is_file():
            continue
        process_label = src.parent.name
        if model_asset_root is not None:
            try:
                relative = src.relative_to(model_asset_root)
                if len(relative.parts) >= 2:
                    process_label = relative.parts[0]
            except ValueError:
                pass
        grouped.setdefault(process_label, []).append(str(src))
    return grouped


def _empty_event_stats() -> dict[str, Any]:
    return {
        "event_file_count": 0,
        "event_count": 0,
        "parse_error_count": 0,
        "devices": [],
        "node_ids": [],
        "sk_ids": [],
        "event_groups": [],
        "top_events": [],
    }


def _event_stats_parser_name() -> str:
    return "ijson" if ijson is not None else "json"


def _ijson_parse_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [OSError, ValueError]
    if ijson is None:
        return tuple(errors)

    for owner in (ijson, getattr(ijson, "common", None)):
        json_error = getattr(owner, "JSONError", None)
        if isinstance(json_error, type) and issubclass(json_error, BaseException):
            errors.append(json_error)
    return tuple(dict.fromkeys(errors))


@contextmanager
def _suppress_streams(*, stdout: bool = True, stderr: bool = False):
    with ExitStack() as stack:
        if stdout:
            stdout_target = stack.enter_context(
                Path(os.devnull).open("w", encoding="utf-8")
            )
            stack.enter_context(redirect_stdout(stdout_target))
        if stderr:
            stderr_target = stack.enter_context(
                Path(os.devnull).open("w", encoding="utf-8")
            )
            stack.enter_context(redirect_stderr(stderr_target))
        yield


@contextmanager
def _suppress_stdout():
    with _suppress_streams(stdout=True):
        yield


def _iter_event_file_events(path: Path) -> Iterable[dict[str, Any]]:
    if ijson is not None:
        prefix = "traceEvents.item"
        parse_errors = _ijson_parse_errors()
        with path.open("rb") as handle:
            first_byte = handle.read(1)
            while first_byte and first_byte in b" \t\r\n":
                first_byte = handle.read(1)
            if first_byte == b"[":
                prefix = "item"
        with path.open("rb") as handle:
            try:
                yield from ijson.items(handle, prefix)
                return
            except parse_errors:
                fallback_to_json = True
            else:
                fallback_to_json = False
        if not fallback_to_json:
            return
    payload = json.loads(_read_text(path))
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        return
    for event in events:
        if isinstance(event, dict):
            yield event


def _event_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return int(value)
    return None


def _event_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    return 0.0


def _update_event_stats_from_event(
    *,
    stats: dict[str, Any],
    event: dict[str, Any],
    device: str,
    node_ids: set[int],
    sk_ids: set[int],
    groups: dict[str, dict[str, Any]],
    top_events_heap: list[tuple[float, str, int, dict[str, Any]]],
    event_order: int,
) -> int:
    stats["event_count"] += 1
    args = event.get("args", {})
    node_id = _event_int(args.get("nodeId")) if isinstance(args, dict) else None
    sk_id = _event_int(args.get("skId")) if isinstance(args, dict) else None
    model_ri = _event_int(args.get("modelRI")) if isinstance(args, dict) else None
    duration = event.get("dur")
    duration_value = _event_float(duration)
    pid = str(event.get("pid")) if event.get("pid") is not None else ""
    tid = str(event.get("tid")) if event.get("tid") is not None else ""
    event_name = str(event.get("name") or "(unnamed)")
    if node_id is not None:
        node_ids.add(node_id)
    if sk_id is not None:
        sk_ids.add(sk_id)

    if sk_id is not None:
        group_key = f"sk:{sk_id}"
        group_label = _event_group_label("sk", sk_id, None, device)
        group_type = "sk"
    elif node_id is not None:
        group_key = f"node:{node_id}"
        group_label = _event_group_label("node", None, node_id, device)
        group_type = "node"
    else:
        group_key = f"device:{device}"
        group_label = _event_group_label("device", None, None, device)
        group_type = "device"

    group = groups.setdefault(
        group_key,
        {
            "group_key": group_key,
            "group_label": group_label,
            "group_type": group_type,
            "sk_id": sk_id,
            "node_ids": set(),
            "devices": set(),
            "pid_labels": set(),
            "tid_labels": set(),
            "model_ri_values": set(),
            "event_count": 0,
            "total_duration": 0.0,
            "max_duration": 0.0,
            "sample_names": [],
        },
    )
    if node_id is not None:
        group["node_ids"].add(node_id)
    group["devices"].add(device)
    if pid:
        group["pid_labels"].add(pid)
    if tid:
        group["tid_labels"].add(tid)
    if model_ri is not None:
        group["model_ri_values"].add(model_ri)
    group["event_count"] += 1
    group["total_duration"] += duration_value
    group["max_duration"] = max(group["max_duration"], duration_value)
    if len(group["sample_names"]) < 4 and event_name not in group["sample_names"]:
        group["sample_names"].append(event_name)

    top_event = {
        "name": event_name,
        "display_name": _event_display_name(
            event_name,
            sk_id,
            node_id,
            device,
        ),
        "raw_name": event_name,
        "device": device,
        "pid": pid,
        "tid": tid,
        "duration": round(duration_value, 3),
        "sk_id": sk_id,
        "node_id": node_id,
    }
    heap_item = (duration_value, event_name, event_order, top_event)
    if len(top_events_heap) < 12:
        heapq.heappush(top_events_heap, heap_item)
    elif heap_item > top_events_heap[0]:
        heapq.heapreplace(top_events_heap, heap_item)
    return event_order + 1


def _collect_event_stats(
    run_dir: Path, event_file_paths: list[str] | None = None
) -> dict[str, Any]:
    if event_file_paths is not None:
        event_files = [Path(path) for path in event_file_paths if Path(path).is_file()]
    else:
        event_files = sorted(
            path
            for pattern in EVENT_FILE_PATTERNS
            for path in run_dir.glob(f"**/{pattern}")
        )
    stats = _empty_event_stats()
    node_ids: set[int] = set()
    sk_ids: set[int] = set()
    devices: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    top_events_heap: list[tuple[float, str, int, dict[str, Any]]] = []
    event_order = 0

    for path in event_files:
        if not path.name.startswith(EVENT_FILE_PREFIXES):
            continue
        stats["event_file_count"] += 1
        device = path.stem.rsplit("_", 1)[-1]
        devices.append(device)
        try:
            events_iter = _iter_event_file_events(path)
            for event in events_iter:
                if not isinstance(event, dict):
                    continue
                event_order = _update_event_stats_from_event(
                    stats=stats,
                    event=event,
                    device=device,
                    node_ids=node_ids,
                    sk_ids=sk_ids,
                    groups=groups,
                    top_events_heap=top_events_heap,
                    event_order=event_order,
                )
        except (OSError, json.JSONDecodeError):
            stats["parse_error_count"] += 1
            continue

    stats["devices"] = sorted(set(devices))
    stats["node_ids"] = sorted(node_ids)
    stats["sk_ids"] = sorted(sk_ids)
    event_groups = []
    for group in groups.values():
        is_empty_device_group = (
            group["group_type"] == "device"
            and group["event_count"] <= 1
            and not group["node_ids"]
            and not group["pid_labels"]
            and group["max_duration"] == 0.0
            and group["sample_names"] == ["(unnamed)"]
        )
        if is_empty_device_group:
            continue
        event_groups.append(
            {
                **group,
                "node_ids": sorted(group["node_ids"]),
                "devices": sorted(group["devices"]),
                "pid_labels": sorted(group["pid_labels"]),
                "tid_labels": sorted(group["tid_labels"]),
                "model_ri_values": sorted(group["model_ri_values"]),
                "total_duration": round(group["total_duration"], 3),
                "max_duration": round(group["max_duration"], 3),
            }
        )
    stats["event_groups"] = sorted(
        event_groups,
        key=lambda item: (item["total_duration"], item["event_count"]),
        reverse=True,
    )
    top_events = sorted(
        top_events_heap, key=lambda current: (current[0], current[1]), reverse=True
    )
    stats["top_events"] = []
    for item in top_events:
        stats["top_events"].append(item[-1])
    return stats


def _merge_event_stats(stats_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats_items:
        return _empty_event_stats()
    merged = _empty_event_stats()
    devices: set[str] = set()
    node_ids: set[int] = set()
    sk_ids: set[int] = set()
    groups: dict[str, dict[str, Any]] = {}
    top_events_heap: list[tuple[float, str, int, dict[str, Any]]] = []
    event_order = 0

    for stats in stats_items:
        merged["event_file_count"] += int(stats.get("event_file_count", 0) or 0)
        merged["event_count"] += int(stats.get("event_count", 0) or 0)
        merged["parse_error_count"] += int(stats.get("parse_error_count", 0) or 0)
        for item in stats.get("devices", []):
            if item is not None:
                devices.add(str(item))
        for item in stats.get("node_ids", []):
            if isinstance(item, int):
                node_ids.add(item)
        sk_ids.update(item for item in stats.get("sk_ids", []) if isinstance(item, int))
        for group in stats.get("event_groups", []):
            if not isinstance(group, dict):
                continue
            group_key = str(group.get("group_key") or "")
            if not group_key:
                continue
            target = groups.setdefault(
                group_key,
                {
                    "group_key": group_key,
                    "group_label": group.get("group_label"),
                    "group_type": group.get("group_type"),
                    "sk_id": group.get("sk_id"),
                    "node_ids": set(),
                    "devices": set(),
                    "pid_labels": set(),
                    "tid_labels": set(),
                    "model_ri_values": set(),
                    "event_count": 0,
                    "total_duration": 0.0,
                    "max_duration": 0.0,
                    "sample_names": [],
                },
            )
            for item in group.get("node_ids", []):
                if isinstance(item, int):
                    target["node_ids"].add(item)
            for item in group.get("devices", []):
                if item is not None:
                    target["devices"].add(str(item))
            for item in group.get("pid_labels", []):
                if item is not None:
                    target["pid_labels"].add(str(item))
            for item in group.get("tid_labels", []):
                if item is not None:
                    target["tid_labels"].add(str(item))
            target["model_ri_values"].update(
                item
                for item in group.get("model_ri_values", [])
                if isinstance(item, int)
            )
            target["event_count"] += int(group.get("event_count", 0) or 0)
            target["total_duration"] += float(group.get("total_duration", 0.0) or 0.0)
            target["max_duration"] = max(
                float(target["max_duration"]),
                float(group.get("max_duration", 0.0) or 0.0),
            )
            for name in group.get("sample_names", []):
                if len(target["sample_names"]) >= 4:
                    break
                if name not in target["sample_names"]:
                    target["sample_names"].append(name)
        for item in stats.get("top_events", []):
            if not isinstance(item, dict):
                continue
            duration = float(item.get("duration", 0.0) or 0.0)
            name = str(item.get("name") or "")
            heap_item = (duration, name, event_order, item)
            event_order += 1
            if len(top_events_heap) < 12:
                heapq.heappush(top_events_heap, heap_item)
            elif heap_item > top_events_heap[0]:
                heapq.heapreplace(top_events_heap, heap_item)

    merged["devices"] = sorted(devices)
    merged["node_ids"] = sorted(node_ids)
    merged["sk_ids"] = sorted(sk_ids)
    merged["event_groups"] = sorted(
        [
            {
                **group,
                "node_ids": sorted(group["node_ids"]),
                "devices": sorted(group["devices"]),
                "pid_labels": sorted(group["pid_labels"]),
                "tid_labels": sorted(group["tid_labels"]),
                "model_ri_values": sorted(group["model_ri_values"]),
                "total_duration": round(group["total_duration"], 3),
                "max_duration": round(group["max_duration"], 3),
            }
            for group in groups.values()
        ],
        key=lambda item: (item["total_duration"], item["event_count"]),
        reverse=True,
    )
    top_events = sorted(
        top_events_heap, key=lambda current: (current[0], current[1]), reverse=True
    )
    merged["top_events"] = []
    for item in top_events:
        merged["top_events"].append(item[-1])
    return merged


def _collect_process_event_stats_worker(
    *,
    process_label: str,
    run_dir_str: str,
    event_file_paths: list[str],
    data_dir_str: str | None,
) -> dict[str, Any]:
    paths = list(event_file_paths)
    if data_dir_str:
        paths = _copy_raw_event_files(paths, Path(data_dir_str))
    return {
        "process_label": process_label,
        "copied_paths": paths,
        "stats": _collect_event_stats(Path(run_dir_str), paths),
    }


class EventStatsProvider:
    def __init__(self, run_dir: Path, profile: Any | None = None) -> None:
        self.run_dir = run_dir
        self.profile = profile
        self._process_paths: dict[str, list[str]] = {}
        self._process_data_dirs: dict[str, Path] = {}
        self._process_copied_paths: dict[str, list[str]] = {}
        self._process_stats: dict[str, dict[str, Any]] = {}
        self._fallback_paths: list[str] = []
        self._fallback_data_dir: Path | None = None
        self._fallback_copied_paths: list[str] | None = None
        self._fallback_stats: dict[str, Any] | None = None
        self._global_stats: dict[tuple[str, ...], dict[str, Any]] = {}

    def register_process(
        self,
        process_label: str,
        event_file_paths: list[str],
        data_dir: Path | None = None,
    ) -> None:
        label = str(process_label)
        self._process_paths[label] = list(event_file_paths)
        if data_dir is not None:
            self._process_data_dirs[label] = data_dir
        self._process_copied_paths.pop(label, None)
        self._process_stats.pop(label, None)
        self._global_stats.clear()

    def register_fallback(
        self, event_file_paths: list[str], data_dir: Path | None = None
    ) -> None:
        self._fallback_paths = list(event_file_paths)
        self._fallback_data_dir = data_dir
        self._fallback_copied_paths = None
        self._fallback_stats = None
        self._global_stats.clear()

    def copied_paths(self) -> list[str]:
        paths: list[str] = []
        for copied in self._process_copied_paths.values():
            paths.extend(copied)
        if self._fallback_copied_paths:
            paths.extend(self._fallback_copied_paths)
        return paths

    def for_process(self, process_label: str) -> dict[str, Any]:
        label = str(process_label)
        if label not in self._process_paths:
            return _empty_event_stats()
        if label not in self._process_stats:
            paths = self._paths_for_process_stats(label)
            with self._profile_section(
                "collect_process_event_stats",
                process_label=label,
                event_file_count=len(paths),
                parser=_event_stats_parser_name(),
            ):
                self._process_stats[label] = _collect_event_stats(self.run_dir, paths)
            self._global_stats.clear()
        return self._process_stats[label]

    def pending_process_labels(self, process_labels: list[str]) -> list[str]:
        labels = sorted(set(str(item) for item in process_labels))
        return [
            label
            for label in labels
            if label in self._process_paths and label not in self._process_stats
        ]

    def for_processes(
        self,
        process_labels: list[str],
        *,
        workers: int = 1,
        on_process_done: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        labels = sorted(set(str(item) for item in process_labels))
        missing_labels = self.pending_process_labels(labels)
        active_workers = min(max(1, workers), len(missing_labels))
        if active_workers > 1:
            event_file_count = 0
            for label in missing_labels:
                event_file_count += len(self._process_paths[label])
            with self._profile_section(
                "collect_process_event_stats_batch",
                process_count=len(missing_labels),
                event_file_count=event_file_count,
                workers=active_workers,
                parser=_event_stats_parser_name(),
            ):
                with ProcessPoolExecutor(max_workers=active_workers) as executor:
                    futures = [
                        executor.submit(
                            _collect_process_event_stats_worker,
                            process_label=label,
                            run_dir_str=str(self.run_dir),
                            event_file_paths=list(self._process_paths[label]),
                            data_dir_str=str(self._process_data_dirs[label])
                            if label in self._process_data_dirs
                            else None,
                        )
                        for label in missing_labels
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        label = str(result.get("process_label") or "")
                        if not label:
                            continue
                        self._process_copied_paths[label] = list(
                            result.get("copied_paths") or []
                        )
                        stats = result.get("stats")
                        self._process_stats[label] = (
                            stats if isinstance(stats, dict) else _empty_event_stats()
                        )
                        if on_process_done is not None:
                            on_process_done(label)
                self._global_stats.clear()
        else:
            for label in missing_labels:
                self.for_process(label)
                if on_process_done is not None:
                    on_process_done(label)
        return {
            label: self._process_stats.get(label, _empty_event_stats())
            for label in labels
        }

    def global_stats(
        self, process_labels: list[str] | None = None, *, workers: int = 1
    ) -> dict[str, Any]:
        if self._process_paths:
            labels = tuple(
                sorted(
                    set(
                        str(item)
                        for item in (process_labels or self._process_paths.keys())
                    )
                )
            )
            if labels not in self._global_stats:
                process_stats = self.for_processes(list(labels), workers=workers)
                stats_items = []
                for label in labels:
                    stats_items.append(process_stats.get(label, _empty_event_stats()))
                with self._profile_section(
                    "merge_event_stats",
                    process_count=len(labels),
                    cached_process_count=len(self._process_stats),
                ):
                    self._global_stats[labels] = _merge_event_stats(stats_items)
            return self._global_stats[labels]
        if self._fallback_stats is None:
            paths = self._paths_for_fallback_stats()
            with self._profile_section(
                "collect_event_stats",
                event_file_count=len(paths),
                parser=_event_stats_parser_name(),
            ):
                self._fallback_stats = _collect_event_stats(self.run_dir, paths)
        return self._fallback_stats

    @contextmanager
    def _profile_section(self, name: str, **metadata: Any):
        if self.profile is None:
            yield
            return
        with self.profile.section(name, **metadata):
            yield

    def _paths_for_process_stats(self, process_label: str) -> list[str]:
        label = str(process_label)
        raw_paths = self._process_paths.get(label, [])
        data_dir = self._process_data_dirs.get(label)
        if data_dir is None:
            return list(raw_paths)
        if label not in self._process_copied_paths:
            self._process_copied_paths[label] = _copy_raw_event_files(
                raw_paths, data_dir
            )
        return list(self._process_copied_paths[label] or raw_paths)

    def _paths_for_fallback_stats(self) -> list[str]:
        if self._fallback_data_dir is None:
            return list(self._fallback_paths)
        if self._fallback_copied_paths is None:
            self._fallback_copied_paths = _copy_raw_event_files(
                self._fallback_paths, self._fallback_data_dir
            )
        return list(self._fallback_copied_paths or self._fallback_paths)


def _update_scope_library(update_report: dict[str, Any]) -> dict[str, Any]:
    if "scope_library" in update_report and isinstance(
        update_report["scope_library"], dict
    ):
        return update_report["scope_library"]
    return {}


def _update_graph_library(update_report: dict[str, Any]) -> dict[str, Any]:
    if "graph_library" in update_report and isinstance(
        update_report["graph_library"], dict
    ):
        return update_report["graph_library"]
    return {}


def _update_dfx_library(update_report: dict[str, Any]) -> dict[str, Any]:
    if "dfx_library" in update_report and isinstance(
        update_report["dfx_library"], dict
    ):
        return update_report["dfx_library"]
    return {}


def _update_phase_summary_data(update_report: dict[str, Any]) -> dict[str, Any]:
    dfx_library = _update_dfx_library(update_report)
    if dfx_library.get("phase_registry"):
        return dfx_library["phase_registry"]
    if dfx_library.get("phase_summary"):
        return dfx_library["phase_summary"]
    graph_library = _update_graph_library(update_report)
    if graph_library.get("phase_summary"):
        return graph_library["phase_summary"]
    return update_report.get("phase_summary", {})


def _phase_key_for_line(phase_registry: dict[str, Any], line: int | None) -> str:
    if not isinstance(line, int):
        return "external_log"
    phase_markers = phase_registry.get("phase_markers", [])
    phase_key = "before_first_phase"
    for item in phase_markers if isinstance(phase_markers, list) else []:
        marker_line = item.get("line")
        marker_key = item.get("key")
        if not isinstance(marker_line, int) or not marker_key:
            continue
        if marker_line <= line:
            phase_key = marker_key
        else:
            break
    return phase_key


def _build_dfx_phase_correlated_signals(
    update_report: dict[str, Any],
) -> list[dict[str, Any]]:
    dfx_library = _update_dfx_library(update_report)
    phase_registry = (
        dfx_library.get("phase_registry", {}) if isinstance(dfx_library, dict) else {}
    )
    exception_registry = (
        dfx_library.get("exception_registry", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    source = dfx_library.get("source", {}) if isinstance(dfx_library, dict) else {}
    super_kernel_log = source.get("super_kernel_log", "super_kernel.log")

    correlated: list[dict[str, Any]] = []
    for item in (
        exception_registry.get("events", [])
        if isinstance(exception_registry.get("events"), list)
        else []
    ):
        correlated.append(
            {
                "bucket": "hard_failure_signals"
                if item.get("candidate_count", 0)
                else "runtime_warnings",
                "phase_key": _phase_key_for_line(phase_registry, item.get("line")),
                "file": super_kernel_log,
                "line": item.get("line"),
                "kind": "function_name",
                "text": item.get("raw_line", ""),
            }
        )
    for item in (
        exception_registry.get("core_symbol_events", [])
        if isinstance(exception_registry.get("core_symbol_events"), list)
        else []
    ):
        correlated.append(
            {
                "bucket": "hard_failure_signals",
                "phase_key": _phase_key_for_line(phase_registry, item.get("line")),
                "file": super_kernel_log,
                "line": item.get("line"),
                "kind": "exception_core_symbol",
                "text": item.get("text", ""),
            }
        )
    return correlated[:16]


def _collect_super_kernel_runtime_messages(
    update_report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    dfx_library = _update_dfx_library(update_report)
    source = dfx_library.get("source", {}) if isinstance(dfx_library, dict) else {}
    super_kernel_log = source.get("super_kernel_log")
    if not super_kernel_log:
        return {"errors": [], "warnings": []}

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    path = Path(super_kernel_log)
    for line_no, line in enumerate(_read_text(path).splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        if re.search(r"\[(ERROR|ERR)\]", text, re.IGNORECASE):
            errors.append({"file": str(path), "line": line_no, "text": text})
            continue
        if re.search(r"\[(WARN|WARNING)\]", text, re.IGNORECASE):
            warnings.append({"file": str(path), "line": line_no, "text": text})
    return {"errors": errors[:12], "warnings": warnings[:12]}


def _build_dfx_phase_assertions(update_report: dict[str, Any]) -> dict[str, bool]:
    phase_keys = set(_update_phase_summary_data(update_report).get("phase_keys", []))
    return {
        "entered_aclsk_optimize": "optimize_begin" in phase_keys,
        "updated_scope_flags": "update_node_scope_flags_begin" in phase_keys,
        "deadlock_refine_seen": "deadlock_refine" in phase_keys,
        "built_tasks_begin": "build_tasks_begin" in phase_keys,
        "built_tasks_finish": "build_tasks_finish" in phase_keys,
        "ended_aclsk_optimize": "optimize_end" in phase_keys,
        "has_scope_evidence": _update_scope_count(update_report) > 0,
        "has_function_evidence": _update_function_count(update_report) > 0,
        "has_task_evidence": bool(_update_task_indices(update_report)),
        "has_time_event": False,
    }


def _build_dfx_hang_summary(update_report: dict[str, Any]) -> dict[str, Any]:
    dfx_library = _update_dfx_library(update_report)
    payload_registry = (
        dfx_library.get("payload_registry", {}) if isinstance(dfx_library, dict) else {}
    )
    exception_registry = (
        dfx_library.get("exception_registry", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    counter_registry = (
        dfx_library.get("counter_registry", {}) if isinstance(dfx_library, dict) else {}
    )
    pc_localization_registry = (
        dfx_library.get("pc_localization_registry", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    evidence = (
        dfx_library.get("evidence", {})
        if isinstance(dfx_library.get("evidence", {}), dict)
        else {}
    )
    exception_evidence = (
        evidence.get("exception", {}) if isinstance(evidence, dict) else {}
    )
    counter_evidence = evidence.get("counter", {}) if isinstance(evidence, dict) else {}
    payload_summary = (
        payload_registry.get("summary", {})
        if isinstance(payload_registry, dict)
        else {}
    )
    counter_summary = (
        counter_registry.get("summary", {})
        if isinstance(counter_registry, dict)
        else {}
    )
    pc_localization_summary = (
        pc_localization_registry.get("summary", {})
        if isinstance(pc_localization_registry, dict)
        else {}
    )
    exception_events = (
        exception_registry.get("events", [])
        if isinstance(exception_registry.get("events"), list)
        else []
    )
    core_symbol_events = (
        exception_registry.get("core_symbol_events", [])
        if isinstance(exception_registry.get("core_symbol_events"), list)
        else []
    )
    runtime_messages = _collect_super_kernel_runtime_messages(update_report)
    runtime_errors = runtime_messages["errors"]
    runtime_warnings = runtime_messages["warnings"]

    phase_assertions = _build_dfx_phase_assertions(update_report)
    phase_correlated_signals = _build_dfx_phase_correlated_signals(update_report)

    exception_level = str(exception_evidence.get("evidence_level", "unknown"))
    has_core_symbols = bool(core_symbol_events)
    has_pc_localization = pc_localization_summary.get("matched_count", 0) > 0
    has_counter_localization = bool(
        counter_summary.get("dominant_active_function_name")
    )
    counter_problem_kind = str(counter_summary.get("counter_problem_kind", "unknown"))
    counter_disabled_by_op_trace = counter_problem_kind == "op_trace_disabled"
    needs_op_trace_rerun = bool(
        counter_summary.get("needs_op_trace_hint")
        and (exception_events or runtime_errors)
        and (
            counter_disabled_by_op_trace
            or (
                not has_core_symbols
                and not has_pc_localization
                and not has_counter_localization
            )
        )
    )

    actionable_signals: list[dict[str, Any]] = []
    super_kernel_log = dfx_library.get("source", {}).get(
        "super_kernel_log", "super_kernel.log"
    )

    for event in exception_events[:4]:
        priority = "high" if event.get("candidate_count", 0) else "medium"
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": event.get("line"),
                "kind": "function_name",
                "text": event.get("raw_line", ""),
                "phase_key": _phase_key_for_line(
                    _update_phase_summary_data(update_report), event.get("line")
                ),
                "priority": priority,
            }
        )
    for event in core_symbol_events[:2]:
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": event.get("line"),
                "kind": "exception_core_symbol",
                "text": event.get("text", ""),
                "phase_key": _phase_key_for_line(
                    _update_phase_summary_data(update_report), event.get("line")
                ),
                "priority": "high",
            }
        )
    if counter_disabled_by_op_trace:
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": (exception_events[0].get("line") if exception_events else "?"),
                "kind": "op_trace_hint",
                "text": "op_trace=false; 当前禁止使用 counter 做子核诊断，请先 export ASCEND_SK_OP_TRACE_ON=1 后重新复现",
                "phase_key": _phase_key_for_line(
                    _update_phase_summary_data(update_report),
                    exception_events[0].get("line") if exception_events else None,
                ),
                "priority": "high",
            }
        )
    elif counter_summary.get("dominant_active_function_name"):
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": "?",
                "kind": "counter_localization",
                "text": (
                    "SkCounterInfo shows {count} cores still running opId={op_id} -> {func}"
                ).format(
                    count=counter_summary.get("dominant_active_core_count", 0),
                    op_id=counter_summary.get("dominant_active_op_id"),
                    func=counter_summary.get("dominant_active_function_name"),
                ),
                "phase_key": "external_log",
                "priority": "high",
            }
        )
    elif counter_summary.get("counter_problem_kind") == "sk_level_issue":
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": "?",
                "kind": "counter_localization",
                "text": (
                    "SkCounterInfo shows no launch=2 on used cores; all used "
                    "cores are already at launch=3, so the hang is more likely "
                    "in SK-level logic"
                ),
                "phase_key": "external_log",
                "priority": "high",
            }
        )
    if needs_op_trace_rerun and not counter_disabled_by_op_trace:
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": (exception_events[0].get("line") if exception_events else "?"),
                "kind": "op_trace_hint",
                "text": (
                    "op_trace=false; if the issue reproduces, export "
                    "ASCEND_SK_OP_TRACE_ON=1 for stronger per-core symbol evidence"
                ),
                "phase_key": _phase_key_for_line(
                    _update_phase_summary_data(update_report),
                    exception_events[0].get("line") if exception_events else None,
                ),
                "priority": "medium",
            }
        )
    for item in runtime_errors[:2]:
        actionable_signals.append(
            {
                "file": item["file"],
                "line": item["line"],
                "kind": "runtime_warning_or_error",
                "text": item["text"],
                "phase_key": _phase_key_for_line(
                    _update_phase_summary_data(update_report), item.get("line")
                ),
                "priority": "high",
            }
        )
    for item in runtime_warnings[:2]:
        actionable_signals.append(
            {
                "file": item["file"],
                "line": item["line"],
                "kind": "runtime_warning_or_error",
                "text": item["text"],
                "phase_key": _phase_key_for_line(
                    _update_phase_summary_data(update_report), item.get("line")
                ),
                "priority": "low",
            }
        )
    runtime_status = payload_summary.get("runtime_status")
    has_dfx_signal = bool(exception_events or core_symbol_events or runtime_errors)
    if has_dfx_signal and runtime_status == "missing_exception_handler_dump":
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": payload_summary.get("runtime_offset") or "?",
                "kind": "sk_dfx_info",
                "text": "missing exception-handler runtime dump in super_kernel.log",
                "phase_key": "external_log",
                "priority": "medium",
            }
        )
    elif has_dfx_signal and runtime_status in {
        "missing_exception_handler_rows",
        "ambiguous_device_args_section",
        "unresolved_device_args_section",
    }:
        actionable_signals.append(
            {
                "file": super_kernel_log,
                "line": payload_summary.get("runtime_offset") or "?",
                "kind": "sk_dfx_info",
                "text": f"runtime payload status: {runtime_status}",
                "phase_key": "external_log",
                "priority": "medium",
            }
        )

    if has_core_symbols or exception_events or runtime_errors:
        signal_confidence = "high"
    elif runtime_warnings:
        signal_confidence = "low"
    elif exception_level in {"explicit", "inferred"}:
        signal_confidence = "medium"
    elif phase_correlated_signals or phase_assertions["entered_aclsk_optimize"]:
        signal_confidence = "low"
    else:
        signal_confidence = "none"

    if exception_events or core_symbol_events or runtime_errors:
        analysis_status = "abnormal"
        conclusion = "维测分析发现异常信号，建议优先围绕 Exception Registry 与 Core Symbol Events 继续定位。"
    elif runtime_warnings:
        analysis_status = "warning"
        conclusion = (
            "维测分析未发现明确异常，但存在 warning，需要结合 Warning 明细继续关注。"
        )
    else:
        analysis_status = "clean"
        conclusion = (
            "维测分析未见异常，当前模型资产证据显示阶段推进和异常链路都没有明显问题。"
        )

    if not phase_assertions["entered_aclsk_optimize"]:
        likely_failure_stage = "before_aclsk_optimize"
    elif not phase_assertions["updated_scope_flags"]:
        likely_failure_stage = "update_node_scope_flags"
    elif not phase_assertions["built_tasks_begin"]:
        likely_failure_stage = "before_task_build"
    elif (exception_events or runtime_errors) and not phase_assertions[
        "built_tasks_finish"
    ]:
        likely_failure_stage = "during_task_build_with_hard_failure"
    elif not phase_assertions["built_tasks_finish"]:
        likely_failure_stage = "during_task_build"
    elif exception_events or has_core_symbols or runtime_errors:
        likely_failure_stage = "after_build_with_hard_failure"
    elif runtime_warnings and not phase_assertions["ended_aclsk_optimize"]:
        likely_failure_stage = "before_optimize_end_with_runtime_warnings"
    elif not phase_assertions["ended_aclsk_optimize"]:
        likely_failure_stage = "before_optimize_end"
    else:
        likely_failure_stage = "no_clear_failure_signal"

    if counter_disabled_by_op_trace and analysis_status == "abnormal":
        conclusion = (
            "维测分析发现异常信号，但当前 `op_trace=false`，因此不能启用 counter 侧子核诊断；"
            "请先使用 `export ASCEND_SK_OP_TRACE_ON=1` 重新复现。"
        )
    elif (
        counter_summary.get("dominant_active_function_name")
        and analysis_status == "abnormal"
    ):
        conclusion = (
            "维测分析发现异常信号，并且 SkCounterInfo 已将当前运行热点收敛到 "
            f"opId={counter_summary.get('dominant_active_op_id')} / "
            f"{counter_summary.get('dominant_active_function_name')}。"
        )
    elif counter_problem_kind == "sk_level_issue" and analysis_status == "abnormal":
        conclusion = (
            "维测分析发现异常信号，并且 SkCounterInfo 显示所有已使用 core 都处于 launch=3、没有 launch=2；"
            "当前更像是 SK 级逻辑/同步问题，而不是子算子正在执行的问题。"
        )
    elif needs_op_trace_rerun and analysis_status == "abnormal":
        conclusion = (
            "维测分析发现异常信号，但当前对子核的定位证据仍不足；"
            "建议使用 `export ASCEND_SK_OP_TRACE_ON=1` 重新复现，以补齐 per-core 子核定位线索。"
        )

    return {
        "analysis_status": analysis_status,
        "conclusion": conclusion,
        "likely_failure_stage": likely_failure_stage,
        "signal_confidence": signal_confidence,
        "hard_failure_count": len(exception_events)
        + len(core_symbol_events)
        + len(runtime_errors),
        "runtime_warning_count": len(runtime_warnings),
        "environment_noise_count": 0,
        "top_actionable_signals": actionable_signals[:6],
        "recommended_triage_entry": "hang-crash-report.html",
        "recommended_triage_order": [
            "phase_registry",
            "counter_registry",
            "payload_registry",
            "exception_registry",
        ],
        "phase_correlated_signals": phase_correlated_signals,
        "phase_assertions": phase_assertions,
        "warning_signals": runtime_warnings,
        "counter_summary": counter_summary,
        "counter_evidence": counter_evidence,
        "needs_op_trace_rerun": needs_op_trace_rerun,
    }


def _update_task_indices(update_report: dict[str, Any]) -> list[int]:
    scope_library = _update_scope_library(update_report)
    sections = scope_library.get("device_task_library", {}).get("sections", [])
    task_indices = set()

    def _task_is_graph_identity_valid(task: dict[str, Any]) -> bool:
        if "graph_identity_valid" in task:
            return bool(task.get("graph_identity_valid"))
        return isinstance(task.get("task_index"), int)

    for section in sections:
        for task in section.get("queues", {}).get("AIC", []):
            task_index = task.get("task_index")
            if isinstance(task_index, int) and _task_is_graph_identity_valid(task):
                task_indices.add(task_index)
        for task in section.get("queues", {}).get("AIV", []):
            task_index = task.get("task_index")
            if isinstance(task_index, int) and _task_is_graph_identity_valid(task):
                task_indices.add(task_index)
    if task_indices:
        return sorted(task_indices)
    return update_report.get("task_indices", [])


def _update_scope_ids(update_report: dict[str, Any]) -> list[int]:
    scope_library = _update_scope_library(update_report)
    scopes = scope_library.get("scopes", [])
    if scopes:
        return sorted(
            {
                scope.get("scope_id")
                for scope in scopes
                if isinstance(scope.get("scope_id"), int)
            }
        )
    return update_report.get("scope_ids", [])


def _update_node_ids(update_report: dict[str, Any]) -> list[int]:
    scope_library = _update_scope_library(update_report)
    node_ids: set[int] = set()
    for scope in scope_library.get("scopes", []):
        for node_id in scope.get("node_ids", []):
            if isinstance(node_id, int):
                node_ids.add(node_id)
    if node_ids:
        return sorted(node_ids)
    return update_report.get("node_ids", [])


def _update_stream_ids(update_report: dict[str, Any]) -> list[int]:
    scope_library = _update_scope_library(update_report)
    scopes = scope_library.get("scopes", [])
    stream_ids = set()
    for scope in scopes:
        for stream in scope.get("streams", []):
            stream_idx = stream.get("stream_idx")
            if isinstance(stream_idx, int):
                stream_ids.add(stream_idx)
    if stream_ids:
        return sorted(stream_ids)
    return update_report.get("stream_ids", [])


def _update_scope_count(update_report: dict[str, Any]) -> int:
    scope_library = _update_scope_library(update_report)
    scopes = scope_library.get("scopes", [])
    if scopes:
        return len(scopes)
    return len(update_report.get("scopes", []))


def _update_function_count(update_report: dict[str, Any]) -> int:
    fused_functions = (
        _update_scope_library(update_report)
        .get("fused_library", {})
        .get("functions", [])
    )
    if fused_functions:
        return len(fused_functions)
    return len(update_report.get("functions", []))


def _summarize_queue_dfx(update_report: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "function_count": _update_function_count(update_report),
        "functions": [],
        "total_task_count": 0,
        "total_dfx_count": 0,
    }
    scope_library = _update_scope_library(update_report)
    device_sections_by_key = {
        section.get("scope_name"): section
        for section in scope_library.get("device_task_library", {}).get("sections", [])
        if section.get("scope_name")
    }
    fused_functions = scope_library.get("fused_library", {}).get(
        "functions", []
    ) or update_report.get("functions", [])
    for function in fused_functions:
        device_section = device_sections_by_key.get(function.get("scope_name")) or {}
        aic_tasks = device_section.get("queues", {}).get("AIC", [])
        aiv_tasks = device_section.get("queues", {}).get("AIV", [])
        dfx = device_section.get("dfx", [])
        item = {
            "scope_id": function.get("scope_id"),
            "sk_id": function.get("sk_id"),
            "scope_name": function.get("scope_name"),
            "node_ids": function.get("node_ids", []),
            "aic_task_count": len(aic_tasks),
            "aiv_task_count": len(aiv_tasks),
            "task_count": len(aic_tasks) + len(aiv_tasks),
            "dfx_count": len(dfx),
            "header_info": device_section.get("header_info", {}),
        }
        summary["functions"].append(item)
        summary["total_task_count"] += item["task_count"]
        summary["total_dfx_count"] += item["dfx_count"]
    return summary


def _correlate_performance(
    update_report: dict[str, Any], event_stats: dict[str, Any]
) -> list[dict[str, Any]]:
    functions_by_sk: dict[int, list[dict[str, Any]]] = {}
    functions_by_node: dict[int, list[dict[str, Any]]] = {}
    fused_functions = _update_scope_library(update_report).get("fused_library", {}).get(
        "functions", []
    ) or update_report.get("functions", [])
    for function in fused_functions:
        sk_id = function.get("sk_id")
        if isinstance(sk_id, int):
            functions_by_sk.setdefault(sk_id, []).append(function)
        for node_id in function.get("node_ids", []):
            if isinstance(node_id, int):
                functions_by_node.setdefault(node_id, []).append(function)

    device_sections_by_key = {}
    device_sections = (
        _update_scope_library(update_report)
        .get("device_task_library", {})
        .get("sections", [])
    )
    for section in device_sections:
        scope_name = section.get("scope_name")
        if scope_name:
            device_sections_by_key[scope_name] = section
    correlated: list[dict[str, Any]] = []
    for group in event_stats.get("event_groups", []):
        matched_functions: list[dict[str, Any]] = []
        seen_function_keys: set[tuple[Any, Any]] = set()

        sk_id = group.get("sk_id")
        if isinstance(sk_id, int):
            for function in functions_by_sk.get(sk_id, []):
                key = (function.get("scope_id"), function.get("sk_id"))
                if key not in seen_function_keys:
                    matched_functions.append(function)
                    seen_function_keys.add(key)

        for node_id in group.get("node_ids", []):
            for function in functions_by_node.get(node_id, []):
                key = (function.get("scope_id"), function.get("sk_id"))
                if key not in seen_function_keys:
                    matched_functions.append(function)
                    seen_function_keys.add(key)

        matched_scope_ids = sorted(
            {
                function.get("scope_id")
                for function in matched_functions
                if function.get("scope_id") is not None
            }
        )
        matched_node_id_set = set()
        group_node_ids = set(group.get("node_ids", []))
        for function in matched_functions:
            for node_id in function.get("node_ids", []):
                if isinstance(node_id, int) and node_id in group_node_ids:
                    matched_node_id_set.add(node_id)
        matched_node_ids = sorted(matched_node_id_set)
        function_task_indices = set()
        function_queue_types = set()

        def _task_is_graph_identity_valid(task: dict[str, Any]) -> bool:
            if "graph_identity_valid" in task:
                return bool(task.get("graph_identity_valid"))
            return isinstance(task.get("task_index"), int)

        for function in matched_functions:
            section = device_sections_by_key.get(function.get("scope_name")) or {}
            for task in section.get("queues", {}).get("AIC", []):
                if isinstance(
                    task.get("task_index"), int
                ) and _task_is_graph_identity_valid(task):
                    function_task_indices.add(task.get("task_index"))
                queue_value = task.get("task_type") or task.get("queue") or "AIC"
                if queue_value:
                    function_queue_types.add(str(queue_value))
            for task in section.get("queues", {}).get("AIV", []):
                if isinstance(
                    task.get("task_index"), int
                ) and _task_is_graph_identity_valid(task):
                    function_task_indices.add(task.get("task_index"))
                queue_value = task.get("task_type") or task.get("queue") or "AIV"
                if queue_value:
                    function_queue_types.add(str(queue_value))
        task_indices = sorted(function_task_indices)
        queue_types = sorted(function_queue_types)
        correlated.append(
            {
                "row_kind": "event_primary",
                "group_key": group.get("group_key"),
                "group_label": group.get("group_label"),
                "group_type": group.get("group_type"),
                "sk_id": group.get("sk_id"),
                "node_ids": group.get("node_ids", []),
                "devices": group.get("devices", []),
                "pid_labels": group.get("pid_labels", []),
                "tid_labels": group.get("tid_labels", []),
                "event_count": group.get("event_count", 0),
                "total_duration": group.get("total_duration", 0.0),
                "max_duration": group.get("max_duration", 0.0),
                "sample_names": group.get("sample_names", []),
                "matched_event_node_ids": matched_node_ids,
                "has_event_match": True,
                "sk_id_seen_in_events": group.get("sk_id") is not None,
                "task_indices": task_indices,
                "queue_types": queue_types,
                "matched_scope_ids": matched_scope_ids,
                "linked_update_scope": matched_scope_ids[0]
                if matched_scope_ids
                else None,
                "has_structure_match": bool(matched_functions),
                "has_queue_link": bool(queue_types),
                "has_task_link": bool(task_indices),
            }
        )
    return correlated


def _performance_judgment(item: dict[str, Any], event_stats: dict[str, Any]) -> str:
    if event_stats["event_file_count"] == 0:
        return "structure-first"
    if item.get("has_structure_match") and (
        item.get("has_queue_link") or item.get("has_task_link")
    ):
        return "time-covered"
    if item.get("has_structure_match"):
        return "queue-first"
    return "timing-coverage-gap"


def _performance_judgment_label(judgment: str) -> str:
    return {
        "structure-first": "先看结构",
        "time-covered": "时间证据已覆盖",
        "queue-first": "先看队列",
        "timing-coverage-gap": "时间覆盖仍有缺口",
    }.get(judgment, judgment)


def _performance_focus_label(value: str) -> str:
    return {
        "structure_first": "先从结构入手",
        "timing_coverage_gap": "时间覆盖仍有缺口",
        "queue_mapping_gap": "队列映射仍有缺口",
        "structure_queue_time_correlation": "结构、队列与时间证据已可关联",
    }.get(value, value)


def _signal_confidence_label(value: str) -> str:
    return {
        "none": "暂无有效信号",
        "low": "信号较弱",
        "medium": "信号中等",
        "high": "信号较强",
    }.get(value, value)


def _priority_label(value: str) -> str:
    return {
        "high": "高",
        "medium": "中",
        "low": "低",
    }.get(value, value)


def _signal_kind_label(value: str) -> str:
    return {
        "runtime_warning_or_error": "运行期告警/错误",
        "aicore_exception": "AICore 异常",
        "sk_counter_info": "SK 计数器异常",
        "counter_localization": "计数器定位",
        "op_trace_hint": "op_trace 提示",
        "sk_dfx_info": "SK DFX 异常",
        "function_name": "函数名线索",
        "core_id": "Core ID 线索",
        "exception_core_symbol": "异常 Core 线索",
        "signal": "信号",
    }.get(value, value)


def _phase_key_label(value: str) -> str:
    return {
        "external_log": "外部日志",
        "before_first_phase": "首个阶段前",
        "optimize_begin": "开始优化",
        "update_node_scope_flags_begin": "更新 Scope 标记",
        "deadlock_refine": "死锁细化",
        "build_tasks_begin": "开始构建任务",
        "build_tasks_finish": "完成构建任务",
        "shape_profiling": "形状 Profiling",
        "time_profiling": "时间 Profiling",
        "optimize_end": "结束优化",
    }.get(value, value)


def _signal_bucket_label(value: str) -> str:
    return {
        "hard_failure_signals": "硬失败信号",
        "runtime_warnings": "运行期告警",
        "environment_noise": "环境噪声",
    }.get(value, value)


def _failure_stage_label(value: str) -> str:
    return {
        "before_aclsk_optimize": "进入 aclskOptimize 之前",
        "update_node_scope_flags": "UpdateNodeScopeBitFlags 阶段",
        "before_task_build": "构建任务之前",
        "task_build": "构建任务阶段",
        "after_task_build": "构建任务之后",
        "during_task_build": "构建任务阶段",
        "during_task_build_with_hard_failure": "构建任务阶段（已观察到硬失败）",
        "after_build_with_hard_failure": "构建任务之后（已观察到硬失败）",
        "before_optimize_end_with_runtime_warnings": "优化结束前（伴随运行期告警）",
        "before_optimize_end": "优化结束前",
        "no_clear_failure_signal": "未观察到清晰失败信号",
        "unknown": "当前无法确认",
    }.get(value, value)


def _evidence_level_label(value: str) -> str:
    return {
        "explicit": "显式证据",
        "inferred": "推断证据",
        "unknown": "证据不足",
        "none": "无",
    }.get(value, value)


def _analysis_status_label(value: str) -> str:
    return {
        "clean": "维测分析未见异常",
        "warning": "存在告警需要关注",
        "abnormal": "发现异常信号",
    }.get(value, value)


def _counter_launch_state_label(value: str) -> str:
    return {
        "ORIGIN": "未执行 SK",
        "SK_ENTRY_LAUNCHED": "SK 已启动，子核未执行",
        "OP_LAUNCHED": "子核运行中",
        "OP_FINISHED": "子核已结束",
        "SK_ENTRY_FINISHED": "SK 已结束",
        "UNKNOWN": "未知",
    }.get(value, value)


def _diagnostic_completeness_label(value: str) -> str:
    return {
        "complete": "完整",
        "limited": "有限",
        "insufficient": "不足",
    }.get(value, value)


def _capability_mode_label(value: str) -> str:
    return {
        "graph-capable": "可图形诊断",
        "summary-only": "仅摘要引导",
    }.get(value, value)


def _presentation_mode_label(value: str) -> str:
    return {
        "focused": "默认",
        "expanded": "展开",
    }.get(value, value)


def _generation_status_label(value: str) -> str:
    return {
        "ok": "已生成",
        "failed": "生成失败",
    }.get(value, value)


def _view_kind_label(value: str) -> str:
    return {
        "interactive_graph": "View",
        "structured_report": "结构化视图",
        "diagnostic_summary": "View",
    }.get(value, value)


def _recommended_for_label(value: str) -> str:
    return {
        "scope_debug": "作用域排查",
        "queue_debug": "队列排查",
        "trace_debug": "追踪排查",
        "update_debug": "更新排查",
        "hang_debug": "故障排查",
        "performance_debug": "性能排查",
        "diagnostic_overview": "总览诊断",
        "structured_index": "结构化索引",
    }.get(value, value)


def _validation_status_label(value: str) -> str:
    return {
        "ok": "正常",
        "failed": "失败",
        "warning": "有告警",
        "unknown": "未知",
    }.get(value, value)


def _asset_directory_label(value: str) -> str:
    return {
        "result_root": "运行结果目录",
        "log_dir": "日志目录",
        "model_asset_root": "模型资产目录",
        "sk_meta_dir": "SK 元数据目录",
        "kernel_meta_dir": "Kernel 元数据目录",
        "model_dir": "模型目录",
        "reports_dir": "报告目录",
    }.get(value, value)


def _asset_file_label(value: str) -> str:
    return {
        "super_kernel.log": "super_kernel.log",
        "sk_scope_split.log": "sk_scope_split.log",
        "sk_fused_nodes.log": "sk_fused_nodes.log",
        "sk_node_detail.log": "sk_node_detail.log",
        "sk_device_args.log": "sk_device_args.log",
    }.get(value, value)


def _capability_label(value: str) -> str:
    mapping = {
        "diagnostic_overview_ready": "Analysis View 可用",
        "summary_guidance_ready": "摘要引导可用",
        "scope_structure_ready": "作用域结构可读",
        "queue_structure_ready": "队列结构可读",
        "graph_capable_mode": "图形诊断可用",
        "summary_only_mode": "仅摘要模式",
        "node_trace_ready": "节点追踪可用",
        "hang_triage_actionable": "故障排查信号可执行",
        "hang_triage_limited": "故障排查信号偏弱",
        "performance_time_covered": "性能时间证据已覆盖",
        "performance_structure_only": "当前仅有性能结构视角",
        "scope_graph_unavailable": "作用域图不可用",
        "queue_graph_unavailable": "队列图不可用",
        "interactive_graph_navigation": "交互式图形导航仍缺失",
        "graph_writeback_payload": "缺少完整图写回载荷",
        "node_trace_unavailable": "节点追踪不可用",
        "high_confidence_failure_signal": "缺少高置信故障信号",
        "timing_coverage_gap": "时间覆盖仍有缺口",
    }
    if value.startswith("performance_focus:"):
        return "性能焦点：" + _performance_focus_label(value.split(":", 1)[1])
    return mapping.get(value, value)


def _linked_objects_summary(linked_objects: dict[str, Any]) -> str:
    if not linked_objects:
        return "无"
    parts: list[str] = []
    if linked_objects.get("scope_ids"):
        parts.append(
            "作用域 "
            + _display_join([str(item) for item in linked_objects["scope_ids"]])
        )
    if linked_objects.get("node_ids"):
        parts.append(
            "节点 " + _display_join([str(item) for item in linked_objects["node_ids"]])
        )
    if linked_objects.get("task_indices"):
        parts.append(
            "任务 "
            + _display_join([str(item) for item in linked_objects["task_indices"]])
        )
    if linked_objects.get("likely_failure_stage"):
        parts.append(
            "失败阶段："
            + _failure_stage_label(str(linked_objects["likely_failure_stage"]))
        )
    return "；".join(parts) if parts else "无"


def _status_from_ok(ok: bool) -> str:
    return "ok" if ok else "failed"


def _build_validation(
    expected_reports: list[str],
    report_links: dict[str, str],
    tool_runs: dict[str, dict[str, Any]],
    blocked_reports: dict[str, str],
) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    for key in expected_reports:
        if key in blocked_reports:
            reports[key] = {
                "status": "failed",
                "path": None,
                "message": blocked_reports[key],
            }
            continue

        if key in tool_runs:
            tool_run = tool_runs[key]
            path = report_links.get(key)
            reports[key] = {
                "status": "ok"
                if tool_run["ok"] and path
                else _status_from_ok(tool_run["ok"]),
                "path": path,
                "message": tool_run["output"][-400:] if tool_run["output"] else "",
            }
            if tool_run["ok"] and path is None:
                reports[key]["status"] = "partial"
                reports[key]["message"] = (
                    "tool exited successfully but expected report file was not found"
                )
            continue

        reports[key] = {
            "status": "ok" if key in report_links else "failed",
            "path": report_links.get(key),
            "message": "" if key in report_links else "report was not generated",
        }

    statuses = {item["status"] for item in reports.values()}
    overall = "ok"
    if "failed" in statuses and "ok" in statuses:
        overall = "partial"
    elif "failed" in statuses and "ok" not in statuses:
        overall = "failed"
    elif "partial" in statuses:
        overall = "partial"
    return {"overall_status": overall, "reports": reports}


def _run_tool(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, str(exc)
    return completed.returncode == 0, completed.stdout


def _default_parallel_workers(model_count: int) -> int:
    return max(1, min(model_count, os.cpu_count() or 1))


def _resolve_parallel_workers(model_count: int, requested_jobs: int | None) -> int:
    if requested_jobs is None:
        return _default_parallel_workers(model_count)
    return max(1, min(requested_jobs, model_count))


class ProgressTracker:
    def __init__(
        self,
        *,
        stage_index: int,
        total_stages: int,
        label: str,
        total_items: int,
        worker_count: int | None = None,
    ) -> None:
        self.stage_index = stage_index
        self.total_stages = total_stages
        self.label = label
        self.total_items = max(total_items, 1)
        self.worker_count = worker_count
        self.completed = 0
        self.started_at = time.monotonic()
        self._bar = None
        self._started = False
        self._use_tqdm = bool(tqdm is not None and sys.stderr.isatty())

    def __enter__(self) -> "ProgressTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(complete=exc_type is None)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._use_tqdm:
            self._bar = tqdm(
                total=self.total_items,
                desc=self._description(),
                unit="item",
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            )
        else:
            _emit(self._fallback_line("start"), file=sys.stderr)

    def step(self, amount: int = 1) -> None:
        if not self._started:
            self.start()
        previous = self.completed
        self.completed += max(1, amount)
        if self.completed > self.total_items:
            self.completed = self.total_items
        delta = self.completed - previous
        if self._bar is not None and delta > 0:
            self._bar.update(delta)
        if self.completed >= self.total_items:
            self.close()

    def close(self, *, complete: bool = True) -> None:
        if not self._started:
            return
        if complete and self.completed < self.total_items:
            remaining = self.total_items - self.completed
            self.completed = self.total_items
            if self._bar is not None and remaining > 0:
                self._bar.update(remaining)
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        else:
            _emit(
                self._fallback_line("done" if complete else "stopped"), file=sys.stderr
            )
        self._started = False

    def _description(self) -> str:
        worker_text = (
            f" · workers {self.worker_count}" if self.worker_count is not None else ""
        )
        return (
            f"Stage {self.stage_index}/{self.total_stages}: {self.label}{worker_text}"
        )

    def _fallback_line(self, state: str) -> str:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        return f"{self._description()} · {state} ({self.completed}/{self.total_items}) · elapsed {elapsed:.1f}s"


class ProfileRecorder:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.started_at = time.perf_counter()
        self.sections: list[dict[str, Any]] = []

    @contextmanager
    def section(self, name: str, **metadata: Any):
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                self.sections.append(
                    {
                        "name": name,
                        "elapsed_seconds": round(time.perf_counter() - start, 6),
                        **metadata,
                    }
                )

    def write(self, path: Path, **metadata: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 6),
            "sections": self.sections,
            **metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _collect_model_bundle_worker(
    *,
    index: int,
    model_dir_str: str,
    model_asset_root_str: str | None,
    reports_dir_str: str,
    multi_model_mode: bool,
    instance_workers: int = 1,
    use_cache: bool = True,
) -> dict[str, Any]:
    current_model_dir = Path(model_dir_str)
    model_asset_root = Path(model_asset_root_str) if model_asset_root_str else None
    reports_dir = Path(reports_dir_str)
    process_label = _model_process_label(current_model_dir, model_asset_root)

    try:
        if use_cache:
            cached = _load_cached_model_entries(
                reports_dir=reports_dir,
                model_dir=current_model_dir,
                model_asset_root=model_asset_root,
                multi_model_mode=multi_model_mode,
            )
            if cached is not None:
                return {
                    "index": index,
                    "entries": cached["entries"],
                    "model_instance_count": cached["model_instance_count"],
                    "cache_hit": True,
                }
        with _suppress_streams(stdout=False, stderr=True):
            model_instance_reports = collect_update_model_instance_reports(
                current_model_dir,
                instance_workers=instance_workers,
            )
        model_instance_count = len(model_instance_reports) or 1
        bundle_mode = multi_model_mode or model_instance_count > 1
        entries: list[dict[str, Any]] = []
        for model_instance_report in model_instance_reports:
            model_instance_id = _entry_model_instance_id(model_instance_report)
            model_ri = model_instance_report.get("model_ri")
            bundle_dir_name = (
                current_model_dir.name
                if model_instance_count == 1
                else _model_instance_bundle_dirname(
                    model_ri, model_instance_id, current_model_dir
                )
            )
            model_key = (
                f"{process_label}__{bundle_dir_name}"
                if bundle_mode
                else _model_bundle_key(current_model_dir, model_asset_root)
            )
            model_label = (
                f"{process_label}/{bundle_dir_name}"
                if bundle_mode
                else _model_bundle_label(current_model_dir, model_asset_root)
            )
            model_ir_label = (
                current_model_dir.name
                if model_instance_count == 1
                else f"{current_model_dir.name}-{model_instance_id}"
            )
            model_bundle_root = (
                reports_dir / process_label / bundle_dir_name
                if bundle_mode
                else reports_dir
            )
            model_data_dir = (
                model_bundle_root / "data" if bundle_mode else reports_dir / "data"
            )
            model_views_dir = (
                model_bundle_root / "views" if bundle_mode else reports_dir / "views"
            )
            model_views_dir.mkdir(parents=True, exist_ok=True)
            (
                model_scope_library_json,
                model_graph_library_json,
                model_dfx_library_json,
            ) = write_update_libraries(
                model_instance_report,
                model_data_dir,
            )
            entries.append(
                {
                    "index": index,
                    "model_dir": str(current_model_dir),
                    "model_key": model_key,
                    "model_label": model_label,
                    "process_label": process_label,
                    "model_ir_label": model_ir_label,
                    "model_ri": model_ri,
                    "model_instance_id": model_instance_id,
                    "model_instance_index": _entry_model_instance_index(
                        model_instance_report
                    ),
                    "model_instance_count": _entry_model_instance_count(
                        model_instance_report, default=model_instance_count
                    ),
                    "scope_count": _update_scope_count(model_instance_report),
                    "function_count": _update_function_count(model_instance_report),
                    "has_update": _model_has_update(model_instance_report),
                    "scope_graph_available": _scope_graph_has_content(
                        model_instance_report
                    ),
                    "task_queue_available": _task_queue_graph_has_content(
                        model_instance_report
                    ),
                    "is_unfused": _model_is_unfused(model_instance_report),
                    "dfx_status": "clean",
                    "dfx_conclusion": "",
                    "bundle_root": str(model_bundle_root),
                    "scope_ids": _update_scope_ids(model_instance_report),
                    "node_ids": _update_node_ids(model_instance_report),
                    "stream_ids": _update_stream_ids(model_instance_report),
                    "task_indices": _update_task_indices(model_instance_report),
                    "phase_keys": _update_phase_summary_data(model_instance_report).get(
                        "phase_keys", []
                    ),
                    "scope_library_json": str(model_scope_library_json),
                    "graph_library_json": str(model_graph_library_json),
                    "dfx_library_json": str(model_dfx_library_json),
                    "views_dir": str(model_views_dir),
                    "report_paths": {},
                    "collection_error": _entry_collection_error(model_instance_report),
                    "processing_error": _entry_collection_error(model_instance_report),
                }
            )
            model_instance_report.clear()
        _write_cached_model_entries(
            reports_dir=reports_dir,
            model_dir=current_model_dir,
            entries=entries,
            model_instance_count=model_instance_count,
        )
        return {
            "index": index,
            "entries": entries,
            "model_instance_count": model_instance_count,
            "cache_hit": False,
        }
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        bundle_dir_name = _model_instance_bundle_dirname(
            None, "mi01", current_model_dir
        )
        model_bundle_root = (
            reports_dir / process_label / bundle_dir_name
            if multi_model_mode
            else reports_dir
        )
        return {
            "index": index,
            "entries": [
                {
                    "index": index,
                    "model_dir": str(current_model_dir),
                    "model_key": f"{process_label}__{bundle_dir_name}",
                    "model_label": f"{process_label}/{bundle_dir_name}",
                    "process_label": process_label,
                    "model_ir_label": f"{current_model_dir.name}-mi01",
                    "model_ri": None,
                    "model_instance_id": "mi01",
                    "model_instance_index": 1,
                    "model_instance_count": 1,
                    "scope_count": 0,
                    "function_count": 0,
                    "has_update": False,
                    "scope_graph_available": False,
                    "task_queue_available": False,
                    "is_unfused": False,
                    "dfx_status": "clean",
                    "dfx_conclusion": "",
                    "bundle_root": str(model_bundle_root),
                    "update_report": {
                        "model_ri": None,
                        "model_instance_id": "mi01",
                        "model_instance_index": 1,
                        "model_instance_count": 1,
                        "scope_ids": [],
                        "node_ids": [],
                        "stream_ids": [],
                        "task_indices": [],
                        "phase_summary": {"phase_keys": []},
                        "functions": [],
                        "scopes": [],
                    },
                    "scope_library_json": "",
                    "graph_library_json": "",
                    "dfx_library_json": "",
                    "views_dir": str(model_bundle_root / "views"),
                    "report_paths": {},
                    "collection_error": "",
                    "processing_error": str(exc),
                }
            ],
            "model_instance_count": 1,
        }


def _normalize_model_report_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: int(item.get("index", 0))):
        item = dict(entry)
        item["model_dir"] = Path(str(item["model_dir"]))
        item["bundle_root"] = Path(str(item["bundle_root"]))
        item["views_dir"] = Path(str(item["views_dir"]))
        item["scope_library_json"] = (
            Path(str(item["scope_library_json"]))
            if item.get("scope_library_json")
            else None
        )
        item["graph_library_json"] = (
            Path(str(item["graph_library_json"]))
            if item.get("graph_library_json")
            else None
        )
        item["dfx_library_json"] = (
            Path(str(item["dfx_library_json"]))
            if item.get("dfx_library_json")
            else None
        )
        normalized.append(item)
    return normalized


def _model_collection_failure_summary(
    entries: list[dict[str, Any]], *, limit: int = 5
) -> str:
    if not entries:
        return "no model report entries were collected"

    details: list[str] = []
    for entry in entries[:limit]:
        label = str(
            entry.get("model_label")
            or entry.get("model_ir_label")
            or entry.get("model_dir")
            or f"model_index={entry.get('index', '?')}"
        )
        reason = str(
            entry.get("processing_error") or entry.get("collection_error") or ""
        ).strip()
        if not reason:
            missing_outputs = []
            for key in (
                "scope_library_json",
                "graph_library_json",
                "dfx_library_json",
            ):
                if not entry.get(key):
                    missing_outputs.append(key)
            if missing_outputs:
                reason = "missing " + ", ".join(missing_outputs)
            else:
                reason = "library outputs were not produced"
        details.append(f"{label}: {reason}")

    remaining = len(entries) - limit
    if remaining > 0:
        details.append(f"... {remaining} more model entries failed")
    return "; ".join(details)


def _source_file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _model_parse_fingerprint(model_dir: Path) -> dict[str, Any]:
    source_names = (
        "super_kernel.log",
        "sk_scope_split.log",
        "sk_node_detail.log",
        "sk_fused_nodes.log",
        "sk_device_args.log",
        "sk_task_queue.json",
    )
    source_fingerprints = []
    for name in source_names:
        source_fingerprints.append(_source_file_fingerprint(model_dir / name))
    return {
        "cache_version": PARSE_CACHE_VERSION,
        "model_dir": str(model_dir),
        "sources": source_fingerprints,
    }


def _cache_key_for_fingerprint(fingerprint: dict[str, Any]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:24]


def _model_cache_dir(reports_dir: Path, model_dir: Path) -> Path:
    fingerprint = _model_parse_fingerprint(model_dir)
    return (
        reports_dir
        / ".cache"
        / "sk-model-analysis"
        / PARSE_CACHE_VERSION
        / _cache_key_for_fingerprint(fingerprint)
    )


def _load_cached_model_entries(
    *,
    reports_dir: Path,
    model_dir: Path,
    model_asset_root: Path | None,
    multi_model_mode: bool,
) -> dict[str, Any] | None:
    cache_dir = _model_cache_dir(reports_dir, model_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8", errors="replace")
        )
    except Exception:
        return None
    if manifest.get("fingerprint") != _model_parse_fingerprint(model_dir):
        return None
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return None
    rebuilt_entries: list[dict[str, Any]] = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        if not isinstance(entry, dict):
            return None
        for key in ("scope_library_json", "graph_library_json", "dfx_library_json"):
            path = Path(str(entry.get(key) or ""))
            if not path.is_file():
                return None
        model_instance_count = int(
            entry.get("model_instance_count")
            or manifest.get("model_instance_count")
            or len(entries)
            or 1
        )
        model_instance_id = _entry_model_instance_id(entry)
        model_ri = entry.get("model_ri")
        process_label = _model_process_label(model_dir, model_asset_root)
        bundle_mode = multi_model_mode or model_instance_count > 1
        bundle_dir_name = (
            model_dir.name
            if model_instance_count == 1
            else _model_instance_bundle_dirname(model_ri, model_instance_id, model_dir)
        )
        model_bundle_root = (
            reports_dir / process_label / bundle_dir_name
            if bundle_mode
            else reports_dir
        )
        entry["model_key"] = (
            f"{process_label}__{bundle_dir_name}"
            if bundle_mode
            else _model_bundle_key(model_dir, model_asset_root)
        )
        entry["model_label"] = (
            f"{process_label}/{bundle_dir_name}"
            if bundle_mode
            else _model_bundle_label(model_dir, model_asset_root)
        )
        entry["process_label"] = process_label
        entry["model_ir_label"] = (
            model_dir.name
            if model_instance_count == 1
            else f"{model_dir.name}-{model_instance_id}"
        )
        entry["bundle_root"] = str(model_bundle_root)
        entry["views_dir"] = str(
            (model_bundle_root / "views") if bundle_mode else (reports_dir / "views")
        )
        entry["report_paths"] = {}
        entry["dfx_status"] = "clean"
        entry["dfx_conclusion"] = ""
        entry.setdefault("view_states", {})
        rebuilt_entries.append(entry)
    return {
        "entries": rebuilt_entries,
        "model_instance_count": int(
            manifest.get("model_instance_count") or len(entries) or 1
        ),
    }


def _write_cached_model_entries(
    *,
    reports_dir: Path,
    model_dir: Path,
    entries: list[dict[str, Any]],
    model_instance_count: int,
) -> None:
    cache_dir = _model_cache_dir(reports_dir, model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_entries = []
    for entry in entries:
        cached_entry = {}
        excluded_keys = {
            "views_dir",
            "bundle_root",
            "report_paths",
            "dfx_status",
            "dfx_conclusion",
            "view_states",
        }
        for key, value in entry.items():
            if key in excluded_keys:
                continue
            cached_entry[key] = str(value) if isinstance(value, Path) else value
        cached_entries.append(cached_entry)
    manifest = {
        "fingerprint": _model_parse_fingerprint(model_dir),
        "model_instance_count": model_instance_count,
        "entries": cached_entries,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _allocate_instance_workers(
    *,
    model_dir_count: int,
    model_instance_count: int | None,
    requested_jobs: int | None,
    use_parallel: bool,
) -> tuple[int, int]:
    if not use_parallel:
        return 1, 1
    if model_instance_count is None:
        total_workers = max(1, requested_jobs or (os.cpu_count() or 1))
    else:
        total_workers = _resolve_parallel_workers(
            max(model_instance_count, model_dir_count), requested_jobs
        )
    model_workers = max(1, min(model_dir_count, total_workers))
    instance_workers = max(1, total_workers // model_workers)
    return model_workers, instance_workers


def _collect_model_report_entries(
    *,
    model_dirs: list[Path],
    model_asset_root: Path | None,
    reports_dir: Path,
    multi_model_mode: bool,
    use_parallel: bool,
    parallel_workers: int,
    instance_workers: int = 1,
    use_cache: bool = True,
    total_stages: int = 3,
) -> list[dict[str, Any]]:
    if not model_dirs:
        return []

    if use_parallel:
        stage1 = ProgressTracker(
            stage_index=1,
            total_stages=total_stages,
            label="收集 model 信息库",
            total_items=len(model_dirs),
            worker_count=parallel_workers,
        )
        stage1.start()
        collected_entries: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(
                    _collect_model_bundle_worker,
                    index=index,
                    model_dir_str=str(current_model_dir),
                    model_asset_root_str=str(model_asset_root)
                    if model_asset_root
                    else None,
                    reports_dir_str=str(reports_dir),
                    multi_model_mode=True,
                    instance_workers=instance_workers,
                    use_cache=use_cache,
                )
                for index, current_model_dir in enumerate(model_dirs)
            ]
            for future in as_completed(futures):
                result = future.result()
                collected_entries.extend(result.get("entries", []))
                stage1.step()
        return _normalize_model_report_entries(collected_entries)

    stage1 = ProgressTracker(
        stage_index=1,
        total_stages=total_stages,
        label="收集 model 信息库",
        total_items=len(model_dirs),
        worker_count=1 if multi_model_mode else None,
    )
    stage1.start()
    collected_entries = []
    for index, current_model_dir in enumerate(model_dirs):
        collected_entries.extend(
            _collect_model_bundle_worker(
                index=index,
                model_dir_str=str(current_model_dir),
                model_asset_root_str=str(model_asset_root)
                if model_asset_root
                else None,
                reports_dir_str=str(reports_dir),
                multi_model_mode=multi_model_mode,
                instance_workers=instance_workers,
                use_cache=use_cache,
            ).get("entries", [])
        )
        stage1.step()
    return _normalize_model_report_entries(collected_entries)


def _reports_relative_path(target_path: Path, reports_dir: Path) -> str:
    return os.path.relpath(target_path, reports_dir)


def _run_relative_path_from_reports(
    relative_path: str, reports_dir: Path, run_dir: Path
) -> str:
    return os.path.relpath(reports_dir / relative_path, run_dir)


def _load_model_update_report_from_libraries(
    *,
    model_dir: Path,
    model_ri: Any,
    model_instance_id: Any,
    model_instance_index: Any,
    model_instance_count: Any,
    scope_library_json: Path | None,
    graph_library_json: Path | None,
    dfx_library_json: Path | None,
) -> dict[str, Any]:
    def _read_json_object(path: Path | None) -> dict[str, Any]:
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    scope_library = _read_json_object(scope_library_json)
    graph_library = _read_json_object(graph_library_json)
    dfx_library = _read_json_object(dfx_library_json)
    return {
        "model_dir": str(model_dir),
        "model_ri": model_ri,
        "model_instance_id": _entry_model_instance_id(
            {"model_instance_id": model_instance_id}
        ),
        "model_instance_index": _entry_model_instance_index(
            {"model_instance_index": model_instance_index}
        ),
        "model_instance_count": _entry_model_instance_count(
            {"model_instance_count": model_instance_count}
        ),
        "scope_library": scope_library,
        "graph_library": graph_library,
        "dfx_library": dfx_library,
    }


def _serialize_model_report_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_dir": str(entry["model_dir"]),
        "model_label": entry["model_label"],
        "model_ri": entry.get("model_ri"),
        "model_instance_id": entry.get("model_instance_id"),
        "model_instance_index": entry.get("model_instance_index"),
        "model_instance_count": entry.get("model_instance_count"),
        "scope_library_json": str(entry["scope_library_json"])
        if entry.get("scope_library_json")
        else "",
        "graph_library_json": str(entry["graph_library_json"])
        if entry.get("graph_library_json")
        else "",
        "dfx_library_json": str(entry["dfx_library_json"])
        if entry.get("dfx_library_json")
        else "",
        "views_dir": str(entry["views_dir"]),
        "report_paths": dict(entry.get("report_paths", {})),
        "collection_error": str(entry.get("collection_error") or ""),
        "processing_error": str(entry.get("processing_error") or ""),
    }


def _render_model_report_entries(
    *,
    model_report_entries: list[dict[str, Any]],
    args_mode: str,
    expected_reports: list[str],
    blocked_reports: dict[str, str],
    run_dir: Path,
    reports_dir: Path,
    model_asset_root: Path | None,
    use_parallel: bool,
    parallel_workers: int,
    event_file_paths_by_process: dict[str, list[str]],
    event_stats_by_process: dict[str, dict[str, Any]] | None = None,
    stage_index: int = 2,
    total_stages: int = 3,
) -> list[dict[str, Any]]:
    if not model_report_entries:
        return []

    render_results: list[dict[str, Any]] = []
    event_stats_by_process = event_stats_by_process or {}
    performance_enabled = (
        "performance_report" in expected_reports
        and "performance_report" not in blocked_reports
    )
    render_workers = parallel_workers if use_parallel else 1
    stage2 = ProgressTracker(
        stage_index=stage_index,
        total_stages=total_stages,
        label="生成 model 视图",
        total_items=len(model_report_entries),
        worker_count=render_workers if use_parallel else None,
    )
    stage2.start()

    if use_parallel and len(model_report_entries) > 1:
        with ProcessPoolExecutor(max_workers=render_workers) as executor:
            futures = [
                executor.submit(
                    _render_model_bundle_worker,
                    index=index,
                    args_mode=args_mode,
                    expected_reports=expected_reports,
                    blocked_reports=blocked_reports,
                    run_dir_str=str(run_dir),
                    reports_dir_str=str(reports_dir),
                    model_asset_root_str=str(model_asset_root)
                    if model_asset_root
                    else None,
                    process_label=str(entry["process_label"]),
                    event_file_paths=event_file_paths_by_process.get(
                        str(entry["process_label"]), []
                    ),
                    event_stats=event_stats_by_process.get(
                        str(entry["process_label"]), {}
                    )
                    if performance_enabled
                    else {},
                    entry=_serialize_model_report_entry(entry),
                )
                for index, entry in enumerate(model_report_entries)
            ]
            for future in as_completed(futures):
                render_results.append(future.result())
                stage2.step()
    else:
        for index, entry in enumerate(model_report_entries):
            render_results.append(
                _render_model_bundle_worker(
                    index=index,
                    args_mode=args_mode,
                    expected_reports=expected_reports,
                    blocked_reports=blocked_reports,
                    run_dir_str=str(run_dir),
                    reports_dir_str=str(reports_dir),
                    model_asset_root_str=str(model_asset_root)
                    if model_asset_root
                    else None,
                    process_label=str(entry["process_label"]),
                    event_file_paths=event_file_paths_by_process.get(
                        str(entry["process_label"]), []
                    ),
                    event_stats=event_stats_by_process.get(
                        str(entry["process_label"]), {}
                    )
                    if performance_enabled
                    else {},
                    entry=_serialize_model_report_entry(entry),
                )
            )
            stage2.step()

    render_results.sort(key=lambda item: int(item.get("index", 0)))
    for entry, render_result in zip(model_report_entries, render_results):
        entry["report_paths"] = dict(render_result.get("report_paths", {}))
        entry["dfx_status"] = str(render_result.get("dfx_status") or "clean")
        entry["dfx_conclusion"] = str(render_result.get("dfx_conclusion") or "")
        entry["view_states"] = dict(render_result.get("view_states", {}))
        collection_error = str(render_result.get("collection_error") or "")
        if collection_error:
            entry["collection_error"] = collection_error
        processing_error = str(render_result.get("processing_error") or "")
        if processing_error:
            entry["processing_error"] = processing_error
    return model_report_entries


def _render_model_bundle_worker(
    *,
    index: int,
    args_mode: str,
    expected_reports: list[str],
    blocked_reports: dict[str, str],
    run_dir_str: str,
    reports_dir_str: str,
    model_asset_root_str: str | None,
    process_label: str,
    event_file_paths: list[str],
    event_stats: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    reports_dir = Path(reports_dir_str)
    model_dir = Path(str(entry["model_dir"]))
    views_dir = Path(str(entry["views_dir"]))
    scope_library_json = (
        Path(str(entry["scope_library_json"]))
        if entry.get("scope_library_json")
        else None
    )
    graph_library_json = (
        Path(str(entry["graph_library_json"]))
        if entry.get("graph_library_json")
        else None
    )
    dfx_library_json = (
        Path(str(entry["dfx_library_json"])) if entry.get("dfx_library_json") else None
    )
    local_update_report = _load_model_update_report_from_libraries(
        model_dir=model_dir,
        model_ri=entry.get("model_ri"),
        model_instance_id=entry.get("model_instance_id"),
        model_instance_index=entry.get("model_instance_index"),
        model_instance_count=entry.get("model_instance_count"),
        scope_library_json=scope_library_json,
        graph_library_json=graph_library_json,
        dfx_library_json=dfx_library_json,
    )
    local_portal_href = _portal_href_for_model_views(views_dir, reports_dir)
    report_paths = dict(entry.get("report_paths", {}))

    result = {
        "index": index,
        "report_paths": report_paths,
        "dfx_status": "clean",
        "dfx_conclusion": "",
        "collection_error": str(entry.get("collection_error") or ""),
        "processing_error": str(entry.get("processing_error") or ""),
        "view_states": {},
    }

    has_required_libraries = bool(
        scope_library_json and graph_library_json and dfx_library_json
    )
    if result["processing_error"] and not has_required_libraries:
        return result

    try:
        local_node_trace_report_links: dict[str, str] = {}
        local_tool_runs: dict[str, dict[str, Any]] = {}
        local_asset_inventory = {
            "files": {
                "super_kernel.log": (model_dir / "super_kernel.log").is_file(),
                "sk_scope_split.log": (model_dir / "sk_scope_split.log").is_file(),
                "sk_fused_nodes.log": (model_dir / "sk_fused_nodes.log").is_file(),
                "sk_node_detail.log": (model_dir / "sk_node_detail.log").is_file(),
                "sk_device_args.log": (model_dir / "sk_device_args.log").is_file(),
            },
            "event_file_count": len(
                [path for path in event_file_paths if Path(path).is_file()]
            ),
        }
        local_node_trace_summary = _collect_node_trace_summary(
            run_dir, local_node_trace_report_links, local_tool_runs
        )
        local_scope_view_state = _build_scope_view_state(
            local_asset_inventory, local_update_report
        )
        local_task_queue_view_state = _build_task_queue_view_state(
            local_asset_inventory, local_update_report
        )

        if args_mode == "full":
            child_scope_path = views_dir / "scope-graph.html"
            if (
                _scope_graph_has_content(local_update_report)
                and scope_library_json
                and graph_library_json
            ):
                with _suppress_stdout():
                    _render_scope_graph_html(
                        scope_library_json, graph_library_json, child_scope_path
                    )
                _decorate_graph_view(
                    child_scope_path,
                    "Scope View",
                    local_update_report.get("model_ri"),
                    [],
                    {},
                    portal_href=local_portal_href,
                    view_state=local_scope_view_state,
                )
            else:
                _render_update_missing_view_html(
                    title="Scope View",
                    output_path=child_scope_path,
                    portal_href=local_portal_href,
                    current_view_key="scope_graph",
                    model_label=str(entry["model_label"]),
                    update_report=local_update_report,
                    view_state=local_scope_view_state,
                )
            result["report_paths"]["scope_graph"] = os.path.relpath(
                child_scope_path, reports_dir
            )

            child_task_path = views_dir / "task-queue-graph.html"
            if (
                _task_queue_graph_has_content(local_update_report)
                and scope_library_json
                and graph_library_json
            ):
                with _suppress_stdout():
                    _render_task_queue_graph_html(
                        scope_library_json, graph_library_json, child_task_path
                    )
                _decorate_graph_view(
                    child_task_path,
                    "TaskQue View",
                    local_update_report.get("model_ri"),
                    [],
                    {},
                    portal_href=local_portal_href,
                    view_state=local_task_queue_view_state,
                )
            else:
                _render_update_missing_view_html(
                    title="TaskQue View",
                    output_path=child_task_path,
                    portal_href=local_portal_href,
                    current_view_key="task_queue_graph",
                    model_label=str(entry["model_label"]),
                    update_report=local_update_report,
                    view_state=local_task_queue_view_state,
                )
            result["report_paths"]["task_queue_graph"] = os.path.relpath(
                child_task_path, reports_dir
            )

        local_hang_summary = _build_dfx_hang_summary(local_update_report)
        local_dfx_view_state = _build_dfx_view_state(
            local_asset_inventory, local_update_report, local_hang_summary
        )
        result["dfx_status"] = str(local_hang_summary.get("analysis_status") or "clean")
        result["dfx_conclusion"] = str(local_hang_summary.get("conclusion") or "")
        local_phase_correlated_signals = local_hang_summary.get(
            "phase_correlated_signals", []
        )
        local_hang_presentation_state = _hang_presentation_state(
            local_hang_summary, local_phase_correlated_signals
        )

        if (
            "hang_crash_report" in expected_reports
            and "hang_crash_report" not in blocked_reports
        ):
            child_hang_path = views_dir / "hang-crash-report.html"
            _render_hang_report_html(
                HangReportHtmlInput(
                    run_dir,
                    local_update_report,
                    local_phase_correlated_signals,
                    local_hang_summary,
                    local_dfx_view_state,
                    local_hang_presentation_state,
                    child_hang_path,
                    local_portal_href,
                )
            )
            result["report_paths"]["hang_crash_report"] = os.path.relpath(
                child_hang_path, reports_dir
            )

        if (
            "performance_report" in expected_reports
            and "performance_report" not in blocked_reports
        ):
            local_event_stats = (
                event_stats
                if event_stats
                else _collect_event_stats(run_dir, event_file_paths)
            )
            local_performance_correlations = _correlate_performance(
                local_update_report, local_event_stats
            )
            local_performance_summary = _summarize_performance_diagnosis(
                local_update_report,
                local_event_stats,
                local_performance_correlations,
            )
            local_analysis_view_state, local_analysis_resource_state = (
                _build_analysis_view_state(
                    local_asset_inventory,
                    local_event_stats,
                    local_node_trace_summary,
                )
            )
            local_performance_presentation_state = _performance_presentation_state(
                local_performance_correlations
            )
            child_performance_path = views_dir / "performance-report.html"
            _render_performance_report_html(
                PerformanceReportHtmlInput(
                    run_dir,
                    local_update_report,
                    local_asset_inventory,
                    local_event_stats,
                    local_performance_correlations,
                    local_performance_summary,
                    local_node_trace_summary,
                    local_analysis_view_state,
                    local_analysis_resource_state,
                    local_performance_presentation_state,
                    child_performance_path,
                    local_portal_href,
                )
            )
            result["report_paths"]["performance_report"] = os.path.relpath(
                child_performance_path, reports_dir
            )
        else:
            local_analysis_view_state = _view_state(
                "missing", "当前没有生成 Analysis View。", "analysis_not_generated"
            )

        result["view_states"] = {
            "scope_graph": local_scope_view_state,
            "task_queue_graph": local_task_queue_view_state,
            "hang_crash_report": local_dfx_view_state,
            "performance_report": local_analysis_view_state,
        }
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        result["processing_error"] = str(exc)

    return result


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_report_layout(reports_dir: Path, multi_model_mode: bool) -> None:
    if not multi_model_mode:
        return
    for stale_dir in (
        reports_dir / "views",
        reports_dir / "data",
    ):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    for stale_dir in (reports_dir / "models",):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    for stale_event_dir in reports_dir.glob("*/event-data"):
        if stale_event_dir.is_dir():
            shutil.rmtree(stale_event_dir)


def _model_bundle_relative(model_dir: Path, model_asset_root: Path | None) -> Path:
    if model_asset_root is not None:
        try:
            return model_dir.relative_to(model_asset_root)
        except ValueError:
            pass
    return Path(model_dir.name)


def _model_bundle_key(model_dir: Path, model_asset_root: Path | None) -> str:
    relative = _model_bundle_relative(model_dir, model_asset_root)
    return "__".join(part.replace("/", "__") for part in relative.parts)


def _model_bundle_label(model_dir: Path, model_asset_root: Path | None) -> str:
    return str(_model_bundle_relative(model_dir, model_asset_root))


def _model_process_label(model_dir: Path, model_asset_root: Path | None) -> str:
    relative = _model_bundle_relative(model_dir, model_asset_root)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return "(direct)"


def _model_ir_label(model_dir: Path, model_asset_root: Path | None) -> str:
    relative = _model_bundle_relative(model_dir, model_asset_root)
    if relative.parts:
        return relative.parts[-1]
    return model_dir.name


def _model_instance_bundle_dirname(
    model_ri: Any, model_instance_id: str, model_dir: Path
) -> str:
    return f"{str(model_ri or model_dir.name)}-{model_instance_id}"


def _model_has_update(update_report: dict[str, Any]) -> bool:
    graph_library = _update_graph_library(update_report)
    rows = graph_library.get("node_update_registry", {}).get("rows", [])
    if rows:
        return True
    for scope in _update_scope_library(update_report).get("scopes", []):
        if isinstance(scope.get("update"), dict):
            return True
    return False


def _scope_graph_has_content(update_report: dict[str, Any]) -> bool:
    if _update_scope_count(update_report) > 0:
        return True
    node_rows = (
        _update_graph_library(update_report).get("node_library", {}).get("nodes", [])
    )
    return bool(node_rows)


def _task_queue_graph_has_content(update_report: dict[str, Any]) -> bool:
    sections = (
        _update_scope_library(update_report)
        .get("device_task_library", {})
        .get("sections", [])
    )
    if sections:
        return True
    return _update_function_count(update_report) > 0


def _model_is_unfused(update_report: dict[str, Any]) -> bool:
    scope_library = _update_scope_library(update_report)
    fused_functions = scope_library.get("fused_library", {}).get("functions", [])
    task_sections = scope_library.get("device_task_library", {}).get("sections", [])
    return (
        (not _model_has_update(update_report))
        and (not fused_functions)
        and (not task_sections)
    )


def _missing_view_messages(
    current_view_key: str, update_report: dict[str, Any]
) -> tuple[str, str, list[str]]:
    if current_view_key == "task_queue_graph":
        if _model_is_unfused(update_report):
            return (
                "当前未融合",
                "当前未融合，因此没有 update 相关内容，也没有 TaskQue 内容。",
                [
                    "这通常表示当前 modelRI 没有形成融合后的任务排布，因此这里不会展示 TaskQue 结构。",
                    "如果这是预期结果，可以回到导航页继续查看 Scope、Analysis 或 DFX；如果不是预期结果，请先检查对应源 log 是否缺失，或确认该 modelRI 是否应当产生融合产物。",
                ],
            )
        return (
            "当前没有 TaskQue 内容",
            "当前没有可展示的 TaskQue 事实源。",
            [
                "这表示当前 modelRI 没有足够的 device task / fused function 事实来构建任务排布视图。",
                "如果不是预期结果，请先检查 `sk_device_args.log`、`sk_fused_nodes.log` 等源日志是否缺失。",
            ],
        )
    return (
        "当前没有 Scope 内容",
        "当前没有可展示的 Scope 结构事实。",
        [
            "这表示当前 modelRI 没有足够的 scope / graph 事实来构建 Scope 关系图。",
            "如果不是预期结果，请先检查 `sk_scope_split.log`、`sk_node_detail.log` 等源日志是否缺失。",
        ],
    )


def _portal_href_for_model_views(views_dir: Path, reports_dir: Path) -> str:
    return os.path.relpath(reports_dir / "run-portal.html", views_dir)


def _build_multi_model_update_report(
    model_reports: list[dict[str, Any]], model_asset_root: Path | None
) -> dict[str, Any]:
    if not model_reports:
        return {
            "model_ri": None,
            "scope_ids": [],
            "node_ids": [],
            "stream_ids": [],
            "task_indices": [],
            "phase_summary": {"phase_keys": []},
            "functions": [],
            "scopes": [],
            "multi_model_index": {"model_count": 0, "models": []},
        }

    scope_ids: set[int] = set()
    node_ids: set[int] = set()
    stream_ids: set[int] = set()
    task_indices: set[int] = set()
    phase_keys: set[str] = set()
    models: list[dict[str, Any]] = []

    for item in model_reports:
        scope_ids.update(
            scope_id
            for scope_id in item.get("scope_ids", [])
            if isinstance(scope_id, int)
        )
        for node_id in item.get("node_ids", []):
            if isinstance(node_id, int):
                node_ids.add(node_id)
        stream_ids.update(
            stream_id
            for stream_id in item.get("stream_ids", [])
            if isinstance(stream_id, int)
        )
        task_indices.update(
            task_index
            for task_index in item.get("task_indices", [])
            if isinstance(task_index, int)
        )
        phase_keys.update(str(key) for key in item.get("phase_keys", []) if key)
        models.append(
            {
                "model_key": item["model_key"],
                "model_label": item["model_label"],
                "model_dir": str(item["model_dir"]),
                "model_ri": item.get("model_ri"),
                "model_instance_id": item.get("model_instance_id"),
                "model_instance_index": item.get("model_instance_index"),
                "model_instance_count": item.get("model_instance_count"),
                "scope_count": item.get("scope_count", 0),
                "function_count": item.get("function_count", 0),
            }
        )

    return {
        "model_ri": "multiple",
        "scope_ids": sorted(scope_ids),
        "node_ids": sorted(node_ids),
        "stream_ids": sorted(stream_ids),
        "task_indices": sorted(task_indices),
        "phase_summary": {"phase_keys": sorted(phase_keys)},
        "functions": [],
        "scopes": [],
        "multi_model_index": {
            "model_count": len(model_reports),
            "model_asset_root": str(model_asset_root) if model_asset_root else None,
            "sk_meta_dir": str(model_asset_root) if model_asset_root else None,
            "models": models,
        },
    }


def _render_model_selector_page(
    *,
    title: str,
    description: str,
    output_path: Path,
    entries: list[dict[str, Any]],
    current_view_key: str,
) -> None:
    nav_markup = render_view_nav(
        [
            ("导航页面", "../run-portal.html", False),
            ("Scope View", "scope-graph.html", current_view_key == "scope_graph"),
            (
                "TaskQue View",
                "task-queue-graph.html",
                current_view_key == "task_queue_graph",
            ),
            (
                "Analysis View",
                "performance-report.html",
                current_view_key == "performance_report",
            ),
            (
                "DFX View",
                "hang-crash-report.html",
                current_view_key == "hang_crash_report",
            ),
        ],
        kicker="",
    )
    header_markup = render_report_header(
        icon="M",
        kicker="Multi Model",
        title=title,
        note_html=html.escape(description),
        nav_html=nav_markup,
    )
    summary_markup = render_report_summary(
        [
            ("Model Count", str(len(entries))),
            ("View", title),
        ]
    )
    intro_markup = render_report_intro_section(
        title="当前摘要",
        note_html="这里只保留本页的最小导航上下文，具体的模型选择在下方表格中完成。",
        summary_html=summary_markup,
    )
    selection_markup = render_report_section(
        title="",
        note_html="",
        body_html="{table}",
        tone="primary",
    )
    rows = []
    for entry in entries:
        report_path = entry.get("report_paths", {}).get(current_view_key)
        action = (
            f"<a href='{html.escape('../' + report_path)}'>打开</a>"
            if report_path
            else "<span class='hint'>当前模型未生成该视图</span>"
        )
        rows.append(
            (
                "<tr><td>{process}</td><td>{model}</td><td>{ri}</td>"
                "<td>{update}</td><td>{scope_count}</td>"
                "<td>{function_count}</td><td>{action}</td></tr>"
            ).format(
                process=html.escape(str(entry.get("process_label", "-"))),
                model=html.escape(
                    str(entry.get("model_ir_label", entry.get("model_label", "-")))
                ),
                ri=html.escape(str(entry.get("model_ri") or "unknown")),
                update=html.escape("yes" if entry.get("has_update") else "no"),
                scope_count=html.escape(str(entry.get("scope_count", 0))),
                function_count=html.escape(str(entry.get("function_count", 0))),
                action=action,
            )
        )
    html_text = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{styles}</style>
</head>
<body>
  <div class="page-shell">
    {header_markup}
    {intro_markup}
    {selection_markup}
  </div>
</body>
</html>
""".format(
        title=html.escape(title),
        styles=_shared_report_styles(),
        header_markup=header_markup,
        intro_markup=intro_markup,
        selection_markup=selection_markup.format(
            table=_wrap_table_markup(
                (
                    "<table><thead><tr><th>Process</th><th>模型目录</th>"
                    "<th>modelRI</th><th>Update</th><th>Scope Count</th>"
                    "<th>Function Count</th><th>Action</th></tr></thead>"
                    "<tbody>{}</tbody></table>"
                ).format(
                    "".join(rows)
                    or "<tr><td colspan='7'>当前没有可展示的模型入口。</td></tr>"
                ),
                panel_id="model-selection",
                title="模型入口",
                subtitle="先定位 process、模型目录和 modelRI，再进入对应的分析视图。",
                min_width=960,
                collapsed=False,
                prominent=True,
                search_placeholder="查找 process、模型目录、modelRI",
            )
        ),
    )
    output_path.write_text(html_text, encoding="utf-8")


def _shared_report_styles() -> str:
    return build_report_styles()


def _join_html_fragments(*parts: str) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _render_summary_grid(items: list[tuple[str, str] | tuple[str, str, str]]) -> str:
    blocks: list[str] = []
    for item in items:
        label, value = item[0], item[1]
        note = item[2] if len(item) > 2 else ""
        note_block = (
            f"<div class='summary-metric-note'>{html.escape(str(note))}</div>"
            if str(note).strip()
            else ""
        )
        blocks.append(
            "<article class='summary-metric-card'>"
            f"<div class='summary-metric-label'>{html.escape(str(label))}</div>"
            f"<div class='summary-metric-value'>{html.escape(str(value))}</div>"
            f"{note_block}"
            "</article>"
        )
    return "<div class='summary-grid'>{}</div>".format("".join(blocks))


def _render_overview_section(
    *,
    title: str,
    note_html: str,
    items: list[tuple[str, str] | tuple[str, str, str]],
    section_id: str = "",
) -> str:
    return render_report_section(
        title=title,
        note_html=note_html,
        body_html=_render_summary_grid(items),
        tone="primary",
        section_id=section_id,
    )


def _presentation_state(mode: str, reason: str) -> dict[str, str]:
    return {"presentation_mode": mode, "presentation_reason": reason}


def _split_visible_hidden(items: list[Any], limit: int) -> tuple[list[Any], list[Any]]:
    if limit <= 0 or len(items) <= limit:
        return items, []
    return items[:limit], items[limit:]


def _collapse_by_importance(
    *,
    row_count: int,
    importance: str,
    has_empty_message: bool = True,
) -> bool:
    if importance == "primary":
        return row_count <= 0 and has_empty_message
    if importance == "secondary":
        return row_count <= 0
    return True


def _normalize_report_table_page_size(page_size: int) -> int:
    normalized = int(page_size)
    if normalized in (10, 20, 50, 100):
        return normalized
    return 20


def _wrap_table_markup(
    table_html: str,
    *,
    panel_id: str,
    title: str = "",
    subtitle: str = "",
    subtitle_html: str = "",
    min_width: int = 760,
    page_size: int = 20,
    collapsed: bool = True,
    prominent: bool = False,
    search_placeholder: str = "搜索当前表格",
) -> str:
    normalized_page_size = _normalize_report_table_page_size(page_size)
    return render_standard_table_panel(
        panel_id=panel_id,
        title=title,
        subtitle=subtitle,
        subtitle_html=subtitle_html,
        table_html=table_html,
        min_width=min_width,
        page_size=normalized_page_size,
        prominent=prominent,
        initially_collapsed=collapsed,
        search_placeholder=search_placeholder,
    )


def _wrap_paginated_table_markup(
    table_id: str,
    headers: list[str],
    rows: list[str],
    *,
    min_width: int = 760,
    page_size: int = 20,
    prominent: bool = False,
    empty_message: str = "当前没有可展示的数据。",
    collapsed: bool = False,
    title: str = "",
    subtitle: str = "",
    subtitle_html: str = "",
    search_placeholder: str = "搜索当前表格",
) -> str:
    normalized_page_size = _normalize_report_table_page_size(page_size)
    prepared_rows: list[str] = []
    for row in rows:
        stripped = row.lstrip()
        if stripped.startswith("<tr"):
            prepared_rows.append(row)
        else:
            prepared_rows.append(row)
    if not prepared_rows:
        prepared_rows = [
            "<tr data-table-empty-row='true'><td colspan='{count}'>{msg}</td></tr>".format(
                count=len(headers),
                msg=html.escape(empty_message),
            )
        ]
    table_html = render_paginated_table_shell(
        table_id=table_id,
        headers=headers,
        rows_html="".join(prepared_rows),
        title=title or table_id.replace("-", " ").title(),
        min_width=min_width,
        page_size=normalized_page_size,
        prominent=prominent,
    )
    return render_standard_table_panel(
        panel_id=table_id,
        title=title,
        subtitle=subtitle,
        subtitle_html=subtitle_html,
        table_html=table_html,
        min_width=min_width,
        page_size=normalized_page_size,
        prominent=prominent,
        initially_collapsed=collapsed,
        search_placeholder=search_placeholder,
    )


def _render_scope_graph_html(
    scope_library_path: Path, graph_library_path: Path, output_path: Path
) -> None:
    source = ScopeLibrarySource(
        str(scope_library_path), str(graph_library_path), mode="scope"
    )
    model = ScopeGraphModel.from_libraries(source)
    ScopeGraphRenderer(str(output_path)).render_html(model)


def _render_task_queue_graph_html(
    scope_library_path: Path, graph_library_path: Path, output_path: Path
) -> None:
    source = TaskQueueLibrarySource(str(scope_library_path), str(graph_library_path))
    model = TaskQueueModel.from_libraries(source)
    TaskQueueRenderer(str(output_path)).render_html(model)


def _render_update_missing_view_html(
    *,
    title: str,
    output_path: Path,
    portal_href: str,
    current_view_key: str,
    model_label: str,
    update_report: dict[str, Any],
    view_state: dict[str, Any] | None = None,
) -> None:
    nav_links = [
        ("导航页面", portal_href, False),
        ("Scope View", "scope-graph.html", current_view_key == "scope_graph"),
        (
            "TaskQue View",
            "task-queue-graph.html",
            current_view_key == "task_queue_graph",
        ),
        (
            "Analysis View",
            "performance-report.html",
            current_view_key == "performance_report",
        ),
        ("DFX View", "hang-crash-report.html", current_view_key == "hang_crash_report"),
    ]
    nav_markup = render_view_nav(nav_links, kicker="")
    section_title, lead_text, detail_lines = _missing_view_messages(
        current_view_key, update_report
    )
    if view_state:
        lead_text = str(view_state.get("summary") or lead_text)
        detail_lines = (
            _view_state_text_lines(
                view_state,
                include_hints=True,
                include_fallbacks=True,
                include_summary=False,
            )
            or detail_lines
        )
        section_title = {
            "missing": "当前缺少资源",
            "empty": "当前没有内容",
            "error": "当前存在异常",
            "blocked": "当前受阻",
            "partial": "当前内容不完整",
        }.get(str(view_state.get("level") or ""), section_title)
    details_markup = "".join(
        f'<p class="hint">{_analysis_format_resource_text(line)}</p>'
        for line in detail_lines
    )
    title_label = (
        "TaskQue View" if current_view_key == "task_queue_graph" else "Scope View"
    )
    header_note = (
        "当前页面没有可供渲染的任务排布；这里直接说明原因，避免误判成工具异常。"
        if current_view_key == "task_queue_graph"
        else "当前页面没有可供渲染的 Scope 结构；这里直接说明原因，避免误判成工具异常。"
    )
    header_markup = render_page_header(
        title=title_label,
        icon="Q" if current_view_key == "task_queue_graph" else "S",
        kicker="SK Meta View",
        note_html=html.escape(header_note),
        stat_badges_html="",
    )
    toolbar_markup = render_graph_toolbar(
        label="当前模型",
        nav_html="",
        trailing_html=f"<span class='graph-meta-note'>{html.escape(model_label)}</span>",
        controls_html="",
    )
    explainer_markup = _view_state_graph_markup(
        view_state
        or _view_state(
            "empty", lead_text, "graph_view_empty", detail_lines=detail_lines
        )
    )
    empty_markup = render_empty_note(lead_text)
    html_text = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{styles}</style>
</head>
<body>
  <div class="page-shell">
    {header_markup}
    {nav_markup}
    {toolbar_markup}
    {explainer_markup}
    <section class="graph-frame">
      <div class="graph-header">
        <div class="graph-heading">
          <h2>{section_title}</h2>
          <p class="hint">{lead_text}</p>
        </div>
      </div>
      <div class="svg-container">
        {empty_markup}
      </div>
    </section>
    <section class="card tertiary">
      {details_markup}
    </section>
  </div>
</body>
</html>
""".format(
        title=html.escape(title),
        styles=COMMON_VISUALIZER_CSS
        + COMMON_TOOLBAR_CSS
        + COMMON_GRAPH_TOOLBAR_CSS
        + COMMON_SCOPE_SECTION_CSS
        + COMMON_GRAPH_FRAME_CSS
        + COMMON_EMPTY_STATE_CSS,
        lead_text=html.escape(lead_text),
        nav_markup=nav_markup,
        section_title=html.escape(section_title),
        header_markup=header_markup,
        toolbar_markup=toolbar_markup,
        explainer_markup=explainer_markup,
        empty_markup=empty_markup,
        details_markup=details_markup,
    )
    output_path.write_text(html_text, encoding="utf-8")


def _collect_node_trace_summary(
    run_dir: Path, report_links: dict[str, str], tool_runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    relative = report_links.get("node_trace")
    if not relative:
        return {
            "status": tool_runs.get("node_trace", {}).get("status", "missing"),
            "trace_compatible": False,
            "trace_file": None,
            "meta_file": None,
            "trace_event_count": 0,
            "total_nodes": 0,
            "total_streams": 0,
            "event_edges": 0,
            "memory_edges": 0,
            "viewer_targets": ["edge://tracing", "chrome://tracing"],
            "covered_node_ids": [],
            "covered_stream_ids": [],
            "cross_report_keys": [],
        }

    trace_path = run_dir / relative
    meta_path = trace_path.with_name(f"{trace_path.stem}_meta.json")
    summary = {
        "status": tool_runs.get("node_trace", {}).get("status", "unknown"),
        "trace_compatible": False,
        "trace_file": relative,
        "meta_file": os.path.relpath(meta_path, run_dir)
        if meta_path.exists()
        else None,
        "trace_event_count": 0,
        "total_nodes": 0,
        "total_streams": 0,
        "event_edges": 0,
        "memory_edges": 0,
        "viewer_targets": ["edge://tracing", "chrome://tracing"],
        "covered_node_ids": [],
        "covered_stream_ids": [],
        "cross_report_keys": [],
    }
    if not trace_path.exists():
        return summary

    try:
        trace_payload = json.loads(_read_text(trace_path))
    except json.JSONDecodeError:
        return summary

    events = (
        trace_payload.get("traceEvents", []) if isinstance(trace_payload, dict) else []
    )
    summary["trace_event_count"] = len(events)
    summary["trace_compatible"] = (
        isinstance(trace_payload, dict)
        and isinstance(events, list)
        and trace_payload.get("displayTimeUnit") == "ns"
        and all(isinstance(event.get("pid"), int) for event in events if "pid" in event)
        and all(isinstance(event.get("tid"), int) for event in events if "tid" in event)
    )

    if meta_path.exists():
        try:
            meta_payload = json.loads(_read_text(meta_path))
        except json.JSONDecodeError:
            return summary
        metadata = meta_payload.get("metadata", {})
        summary["total_nodes"] = metadata.get("total_nodes", 0)
        summary["total_streams"] = metadata.get("total_streams", 0)
        summary["event_edges"] = metadata.get("event_edges", 0)
        summary["memory_edges"] = metadata.get("memory_edges", 0)
        summary["covered_stream_ids"] = metadata.get("stream_ids", [])
        compatibility = metadata.get("trace_compatibility", {})
        if isinstance(compatibility, dict):
            summary["viewer_targets"] = compatibility.get(
                "viewer_targets", summary["viewer_targets"]
            )
            summary["trace_compatible"] = (
                summary["trace_compatible"]
                and compatibility.get("pid_is_int", False)
                and compatibility.get("tid_is_int", False)
            )
        summary["cross_report_keys"] = metadata.get("cross_report_keys", [])
        cross_index = meta_payload.get("cross_report_index", {})
        if isinstance(cross_index, dict):
            covered_node_ids = []
            for item in cross_index.get("node_index", {}).keys():
                covered_node_ids.append(int(item))
            summary["covered_node_ids"] = covered_node_ids
    return summary


class ArtifactDiagnosticStateInput(NamedTuple):
    key: str
    generation_status: str
    asset_inventory: dict[str, Any]
    update_report: dict[str, Any]
    performance_summary: dict[str, Any]
    hang_crash_summary: dict[str, Any]
    node_trace_summary: dict[str, Any]


def _artifact_diagnostic_state(
    request: ArtifactDiagnosticStateInput,
) -> tuple[str, str | None]:
    key = request.key
    generation_status = request.generation_status
    asset_inventory = request.asset_inventory
    performance_summary = request.performance_summary
    hang_crash_summary = request.hang_crash_summary
    node_trace_summary = request.node_trace_summary
    if generation_status != "ok":
        return (
            "insufficient",
            "This report could not be generated from the current evidence set.",
        )
    if key == "run_portal":
        return "complete", None
    if key == "scope_graph":
        return "complete", None
    if key == "task_queue_graph":
        return "complete", None
    if key == "performance_report":
        if asset_inventory.get("event_file_count", 0) == 0:
            return (
                "limited",
                f"No `{EVENT_ASSET_NAME}` was found under the resolved model directory.",
            )
        if performance_summary.get("focus") != "structure_queue_time_correlation":
            return "limited", performance_summary.get("reason")
        return "complete", None
    if key == "hang_crash_report":
        if hang_crash_summary.get("signal_confidence") in {"none", "low"}:
            return (
                "limited",
                "Current crash evidence is weak and still dominated by warnings/noise.",
            )
        return "complete", None
    if key == "node_trace":
        if not node_trace_summary.get("trace_compatible", False):
            return (
                "limited",
                "Tracing output exists, but compatibility or coverage is still incomplete.",
            )
        return "complete", None
    return "complete", None


def _hang_presentation_state(
    hang_crash_summary: dict[str, Any], phase_correlated_signals: list[dict[str, Any]]
) -> dict[str, str]:
    total = (
        hang_crash_summary.get("hard_failure_count", 0)
        + hang_crash_summary.get("runtime_warning_count", 0)
        + hang_crash_summary.get("environment_noise_count", 0)
        + len(phase_correlated_signals)
    )
    if total > PRESENTATION_LIMITS["hang_secondary"]:
        return _presentation_state(
            "focused", "Hang/crash 证据较噪，默认只展开最高优先级子集。"
        )
    return _presentation_state("expanded", "Hang/crash 证据较紧凑，可以保持展开视图。")


def _performance_presentation_state(
    performance_correlations: list[dict[str, Any]],
) -> dict[str, str]:
    if len(performance_correlations) > PRESENTATION_LIMITS["performance_matrix"]:
        return _presentation_state(
            "focused", "性能矩阵较大，默认只展开最高相关的 function。"
        )
    return _presentation_state("expanded", "性能关联数量较少，可以保持展开矩阵视图。")


def _portal_presentation_state(
    capability_mode: str,
    missing_assets: list[dict[str, Any]],
    report_artifacts: dict[str, dict[str, Any]],
) -> dict[str, str]:
    if capability_mode == "summary-only":
        return _presentation_state(
            "focused",
            "summary-only 模式应优先突出少量引导块，而不是铺开展示大量报告卡片。",
        )
    visible_views = sum(
        1
        for artifact in report_artifacts.values()
        if artifact.get("generation_status") == "ok"
    )
    if len(missing_assets) > 2 or visible_views > 8:
        return _presentation_state(
            "focused", "当前 run 的视图和缺口都较多，运行总览应优先突出主路径。"
        )
    return _presentation_state(
        "expanded", "当前 run 规模较小，可以采用更展开的运行总览布局。"
    )


def _summarize_performance_diagnosis(
    update_report: dict[str, Any],
    event_stats: dict[str, Any],
    performance_correlations: list[dict[str, Any]],
) -> dict[str, Any]:
    event_group_count = len(performance_correlations)
    mapped_group_count = 0
    queue_backed_group_count = 0
    task_backed_group_count = 0
    for item in performance_correlations:
        if item.get("has_structure_match"):
            mapped_group_count += 1
        if item.get("has_queue_link"):
            queue_backed_group_count += 1
        if item.get("has_task_link"):
            task_backed_group_count += 1
    if event_stats["event_file_count"] == 0:
        focus = "structure_first"
        reason = "当前没有找到 SK time-event 文件，应优先查看 scope、fused node 和 queue 结构。"
    elif event_group_count == 0:
        focus = "timing_coverage_gap"
        reason = "虽然存在 time-event 文件，但当前还没有解析出可稳定组织的事件主视角。"
    elif mapped_group_count == 0:
        focus = "timing_coverage_gap"
        reason = (
            "虽然存在 time-event 文件，但它们还不能稳定映射到当前 SK function 或 node。"
        )
    elif queue_backed_group_count == 0 or task_backed_group_count == 0:
        focus = "queue_mapping_gap"
        reason = "虽然已有事件主视角和结构锚点，但推断出的 task 或 queue 仍不足以支撑稳定的队列级归因。"
    else:
        focus = "structure_queue_time_correlation"
        reason = "事件主视角、结构锚点和 task/queue 证据都已存在，可以做第一轮结构/队列/时间关联。"
    diagnostic_completeness = (
        "complete" if focus == "structure_queue_time_correlation" else "limited"
    )
    recommended_next_step = {
        "structure_first": "先看 scope 和 fused-node 结构，再尝试做时间归因。",
        "timing_coverage_gap": "先补齐或重新核对 time-event 资产，再尝试把事件与 function/node 稳定对齐。",
        "queue_mapping_gap": "先补强 task/queue 证据，再基于事件主视角做队列级归因。",
        "structure_queue_time_correlation": "先从事件主视角定位热点，再跳到具体 update/task/scope 对象做交叉确认。",
    }[focus]
    recommended_entry = (
        "performance-report.html#event-primary-matrix"
        if event_stats["event_file_count"] > 0
        else "performance-report.html#current-focus"
    )
    linked_update_scope = None
    linked_task_indices: list[int] = []
    trace_covered_node_ids: list[int] = []
    prioritized = next(
        (
            item
            for item in performance_correlations
            if item.get("has_structure_match") or item.get("has_task_link")
        ),
        performance_correlations[0] if performance_correlations else None,
    )
    if prioritized is not None:
        linked_update_scope = prioritized.get("linked_update_scope")
        linked_task_indices = prioritized.get("task_indices", [])[:8]
        trace_covered_node_ids = prioritized.get("node_ids", [])[:8]
    return {
        "focus": focus,
        "reason": reason,
        "diagnostic_completeness": diagnostic_completeness,
        "recommended_next_step": recommended_next_step,
        "recommended_entry": recommended_entry,
        "linked_update_scope": linked_update_scope,
        "linked_task_indices": linked_task_indices,
        "trace_covered_node_ids": trace_covered_node_ids,
        "event_group_count": event_group_count,
        "mapped_event_group_count": mapped_group_count,
        "queue_backed_group_count": queue_backed_group_count,
        "task_backed_group_count": task_backed_group_count,
        "scope_count": _update_scope_count(update_report),
        "function_count": _update_function_count(update_report),
        "top_event_groups": performance_correlations[:4],
        "top_events": event_stats.get("top_events", [])[:6],
    }


def _analysis_trace_status_label(node_trace_summary: dict[str, Any]) -> str:
    if node_trace_summary.get("trace_file"):
        return "可用"
    return "未采集"


def _analysis_resource_status_label(state: dict[str, Any]) -> str:
    level = str(state.get("level") or "unknown")
    return {
        "available": "可用",
        "partial": "部分受限",
        "missing": "未采集",
        "empty": "空资源",
        "error": "异常",
        "blocked": "受阻",
    }.get(level, "未知")


def _analysis_format_resource_text(text: str) -> str:
    escaped = html.escape(text)
    for token in (
        "export ASCEND_PROF_SK_ON=1",
        "export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1",
    ):
        escaped = escaped.replace(
            html.escape(token),
            f"<span class='mono'>{html.escape(token)}</span>",
        )
    return escaped


def _view_state(
    level: str,
    summary: str,
    reason_code: str,
    *,
    detail_lines: list[str] | None = None,
    acquisition_hints: list[str] | None = None,
    fallback_guidance: list[str] | None = None,
    diagnostic_refs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "level": level,
        "summary": summary,
        "reason_code": reason_code,
        "detail_lines": list(detail_lines or []),
        "acquisition_hints": list(acquisition_hints or []),
        "fallback_guidance": list(fallback_guidance or []),
        "diagnostic_refs": diagnostic_refs or {},
    }


def _view_state_label(state: dict[str, Any]) -> str:
    return _analysis_resource_status_label(state)


def _view_state_text_lines(
    state: dict[str, Any],
    *,
    include_hints: bool = True,
    include_fallbacks: bool = True,
    include_summary: bool = False,
) -> list[str]:
    lines: list[str] = []
    if include_summary and state.get("summary"):
        lines.append(str(state.get("summary")))
    for line in state.get("detail_lines", []):
        if str(line).strip():
            lines.append(str(line))
    if include_hints:
        lines.extend(
            str(line)
            for line in state.get("acquisition_hints", [])
            if str(line).strip()
        )
    if include_fallbacks:
        lines.extend(
            str(line)
            for line in state.get("fallback_guidance", [])
            if str(line).strip()
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)
    return deduped


def _view_state_note_html(
    state: dict[str, Any],
    *,
    include_hints: bool = True,
    include_fallbacks: bool = False,
    include_summary: bool = False,
) -> str:
    lines = _view_state_text_lines(
        state,
        include_hints=include_hints,
        include_fallbacks=include_fallbacks,
        include_summary=include_summary,
    )
    return " ".join(_analysis_format_resource_text(line) for line in lines if line)


def _view_state_graph_markup(state: dict[str, Any]) -> str:
    if str(state.get("level") or "available") == "available":
        return ""
    current = [str(state.get("summary") or "当前视图存在受限状态。")]
    explain = _view_state_text_lines(
        state, include_hints=False, include_fallbacks=False, include_summary=False
    )[:3]
    next_steps = _view_state_text_lines(
        state, include_hints=True, include_fallbacks=True, include_summary=False
    )[:3]
    sections: list[tuple[str, list[str]]] = [("当前状态", current)]
    if explain:
        sections.append(("该怎么理解", explain))
    if next_steps:
        sections.append(("下一步建议", next_steps))
    return render_scope_section_block(sections)


def _graph_parser_diagnostics(update_report: dict[str, Any]) -> dict[str, Any]:
    graph_library = _update_graph_library(update_report)
    diagnostics = graph_library.get("parser_diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}


def _analysis_resource_context_html(
    analysis_resource_state: dict[str, Any],
    *,
    include_tracing: bool = True,
    include_timing: bool = True,
    include_kernel_meta: bool = True,
    include_path_hint: bool = True,
) -> str:
    parts: list[str] = []
    if include_tracing:
        tracing = analysis_resource_state.get("tracing", {})
        if tracing.get("level") != "available":
            parts.append(
                _analysis_format_resource_text(str(tracing.get("summary") or ""))
            )
            if tracing.get("hint"):
                parts.append(
                    _analysis_format_resource_text(str(tracing.get("hint") or ""))
                )
    if include_timing:
        timing = analysis_resource_state.get("timing", {})
        if timing.get("level") != "available":
            parts.append(
                _analysis_format_resource_text(str(timing.get("summary") or ""))
            )
            if timing.get("hint"):
                parts.append(
                    _analysis_format_resource_text(str(timing.get("hint") or ""))
                )
    if include_kernel_meta:
        kernel_meta = analysis_resource_state.get("kernel_meta", {})
        if kernel_meta.get("level") != "available":
            parts.append(
                _analysis_format_resource_text(str(kernel_meta.get("summary") or ""))
            )
            if kernel_meta.get("hint"):
                parts.append(
                    _analysis_format_resource_text(str(kernel_meta.get("hint") or ""))
                )
    if include_path_hint and parts:
        parts.append(
            "如果你确认已经打开了相关环境变量，请再检查输入路径是否真的指向本轮结果目录。"
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        deduped.append(part)
        seen.add(part)
    return " ".join(deduped)


def _build_analysis_resource_state(
    asset_inventory: dict[str, Any],
    event_stats: dict[str, Any],
    node_trace_summary: dict[str, Any],
) -> dict[str, Any]:
    files = (
        asset_inventory.get("files", {}) if isinstance(asset_inventory, dict) else {}
    )
    trace_file = node_trace_summary.get("trace_file")
    trace_status = str(node_trace_summary.get("status") or "unknown")

    if trace_file:
        tracing = {
            "level": "available",
            "summary": "已找到原始 tracing 文件，可直接跳转查看。",
            "hint": "",
        }
    elif trace_status == "failed":
        tracing = {
            "level": "error",
            "summary": "当前没有原始 tracing 文件。节点追踪生成失败，当前属于解析异常，不是空资源。",
            "hint": (
                "当前未采集原始 tracing；请先检查 node tracing 工具输出；"
                "如果只是想补 tracing 文件，请使用 export ASCEND_PROF_SK_ON=1 "
                "重新采集；如果连 super_kernel.log、scope 或 queue 日志也缺失，"
                "请再使用 export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1。"
            ),
        }
    else:
        tracing = {
            "level": "missing",
            "summary": "当前没有原始 tracing 文件。",
            "hint": (
                "当前未采集原始 tracing；如需 tracing 文件和事件矩阵中的 "
                "tracing 跳转，请先 export ASCEND_PROF_SK_ON=1 后重新采集；"
                "如果基础的 super_kernel / scope / queue 日志也缺失，请再使用 "
                "export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1。"
            ),
        }

    event_file_count = int(event_stats.get("event_file_count") or 0)
    event_count = int(event_stats.get("event_count") or 0)
    parse_error_count = int(event_stats.get("parse_error_count") or 0)
    if event_file_count <= 0:
        timing = {
            "level": "missing",
            "summary": f"未找到 {EVENT_ASSET_NAME}。",
            "hint": "如需事件矩阵和热点事件，请先 export ASCEND_PROF_SK_ON=1 后重新采集。",
        }
    elif parse_error_count > 0 and event_count <= 0:
        timing = {
            "level": "error",
            "summary": f"已找到 {event_file_count} 个 time-event 文件，但解析失败。",
            "hint": "当前属于解析异常，不是空资源；请先检查事件文件格式和输入路径，再决定是否重新采集。",
        }
    elif event_count <= 0:
        timing = {
            "level": "empty",
            "summary": "已找到 time-event 文件，但解析结果为空。",
            "hint": "当前属于空资源，不是未采集；请先确认输入路径和采集结果，再决定是否重新采集。",
        }
    else:
        timing = {
            "level": "available",
            "summary": f"已找到 {event_file_count} 个 time-event 文件，共解析 {event_count} 条事件。",
            "hint": "",
        }

    missing_kernel_meta_assets = []
    for name in (
        "super_kernel.log",
        "sk_scope_split.log",
        "sk_fused_nodes.log",
        "sk_device_args.log",
    ):
        if not files.get(name):
            missing_kernel_meta_assets.append(name)
    if missing_kernel_meta_assets:
        kernel_meta = {
            "level": "missing",
            "summary": "缺少 " + "、".join(missing_kernel_meta_assets) + "。",
            "hint": "如需完整的 super-kernel、scope、fused 和 queue 证据，请先 export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1 后重新采集。",
        }
    else:
        kernel_meta = {
            "level": "available",
            "summary": "已找到 super-kernel、scope、fused 和 queue 所需的基础日志。",
            "hint": "",
        }

    states = [("tracing", tracing), ("timing", timing), ("kernel_meta", kernel_meta)]
    primary = next(
        ({"key": key, **state} for key, state in states if state["level"] == "error"),
        None,
    )
    if primary is None:
        primary = next(
            (
                {"key": key, **state}
                for key, state in states
                if state["level"] in {"missing", "empty"}
            ),
            None,
        )
    if primary is None:
        primary = {
            "key": "all",
            "level": "available",
            "summary": "当前 Analysis 所需资源完整，可直接围绕事件矩阵和 tracing 深挖。",
            "hint": "",
        }

    return {
        "tracing": tracing,
        "timing": timing,
        "kernel_meta": kernel_meta,
        "primary": primary,
    }


def _build_scope_view_state(
    asset_inventory: dict[str, Any],
    update_report: dict[str, Any],
) -> dict[str, Any]:
    files = (
        asset_inventory.get("files", {}) if isinstance(asset_inventory, dict) else {}
    )
    parser = _graph_parser_diagnostics(update_report)
    scope_count = _update_scope_count(update_report)
    node_library = (
        _update_graph_library(update_report).get("node_library", {}).get("nodes", [])
    )
    node_library_count = int(parser.get("node_library_count") or len(node_library))
    scope_node_count = int(parser.get("scope_node_count") or 0)
    missing_node_id_count = int(parser.get("missing_node_id_count") or 0)
    empty_scope_ids = []
    for item in parser.get("empty_scope_ids", []):
        if isinstance(item, int):
            empty_scope_ids.append(int(item))
    partial_scope_ids = [
        int(item)
        for item in parser.get("partial_scope_ids", [])
        if isinstance(item, int)
    ]
    diagnostic_refs = {
        "scope_count": scope_count,
        "node_library_count": node_library_count,
        "scope_node_count": scope_node_count,
        "missing_node_id_count": missing_node_id_count,
        "empty_scope_ids": empty_scope_ids[:10],
        "partial_scope_ids": partial_scope_ids[:10],
    }
    if not files.get("sk_scope_split.log"):
        return _view_state(
            "missing",
            "当前缺少 `sk_scope_split.log`，Scope 结构无法建立。",
            "missing_scope_split_log",
            acquisition_hints=[
                "如需完整的 Scope 结构，请先 export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1 后重新采集。"
            ],
            fallback_guidance=[
                "你仍可以先看 TaskQue、Analysis 或 DFX，但 Scope 图当前不会有稳定结构。"
            ],
            diagnostic_refs=diagnostic_refs,
        )
    if scope_node_count > 0 and node_library_count <= 0:
        return _view_state(
            "error",
            "已识别到 Scope 节点引用，但 graph node library 为空，当前无法构建 Scope 节点图。",
            "scope_node_library_empty",
            detail_lines=[
                "这通常表示节点明细没有解析出来，或当前日志格式与 parser 预期不一致。",
                "如果这是新版本日志，请先检查 parser diagnostics 和原始节点打印格式。",
            ],
            acquisition_hints=[
                "如需补齐 node 级结构日志，请先 export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1 后重新采集。"
            ],
            fallback_guidance=[
                "当前可先回到 DFX 或 TaskQue 继续定位，但不要把这个结果当作“当前没有 Scope”。"
            ],
            diagnostic_refs=diagnostic_refs,
        )
    if empty_scope_ids and scope_count > 0 and len(empty_scope_ids) >= scope_count:
        return _view_state(
            "error",
            "当前所有 Scope 都没有匹配到任何节点，Scope 图不能视为有效结果。",
            "all_scope_nodes_missed",
            detail_lines=[
                "这更像是 scope 与 node library 的对齐异常，而不是当前真的没有 Scope。",
                "请优先检查 parser diagnostics 中的 missing/empty scope 统计。",
            ],
            acquisition_hints=[
                "如需重新采集结构日志，请先 export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1 后重新采集。"
            ],
            fallback_guidance=[
                "如果同一 modelRI 在同一文件里包含多个 model instance，请优先检查 model-instance 切分是否正确。"
            ],
            diagnostic_refs=diagnostic_refs,
        )
    if missing_node_id_count > 0 or empty_scope_ids or partial_scope_ids:
        return _view_state(
            "partial",
            "当前 Scope 图只能部分成立；有些 Scope 或节点对齐正常，有些不正常。",
            "scope_graph_partial_alignment",
            detail_lines=[
                f"当前 missing node id 数量为 {missing_node_id_count}。",
                "这通常表示部分 scope 与 node library 对齐成功，部分仍有失配。",
            ],
            acquisition_hints=[
                "如果这是非预期结果，请先对照 parser diagnostics 检查 empty_scope_ids / partial_scope_ids。"
            ],
            fallback_guidance=[
                "你仍可优先查看那些已经正常显示的 Scope，再结合 DFX/TaskQue 做交叉确认。"
            ],
            diagnostic_refs=diagnostic_refs,
        )
    if scope_count <= 0 and node_library_count <= 0:
        return _view_state(
            "empty",
            "当前没有可展示的 Scope 结构。",
            "scope_graph_empty",
            detail_lines=[
                "当前结果里既没有 Scope，也没有可用于重建节点图的 node library。"
            ],
            fallback_guidance=[
                "如果这不是预期结果，请先检查输入路径或确认当前 modelRI 是否真的产生了 Scope。"
            ],
            diagnostic_refs=diagnostic_refs,
        )
    return _view_state(
        "available",
        "当前 Scope 结构完整，可直接围绕关系图和详情继续定位。",
        "scope_graph_available",
        diagnostic_refs=diagnostic_refs,
    )


def _build_task_queue_view_state(
    asset_inventory: dict[str, Any],
    update_report: dict[str, Any],
) -> dict[str, Any]:
    files = (
        asset_inventory.get("files", {}) if isinstance(asset_inventory, dict) else {}
    )
    scope_library = _update_scope_library(update_report)
    sections = scope_library.get("device_task_library", {}).get("sections", [])
    functions = scope_library.get("fused_library", {}).get("functions", [])
    if _model_is_unfused(update_report):
        return _view_state(
            "blocked",
            "当前未融合，因此没有 update 相关内容，也没有 TaskQue 内容。",
            "task_queue_unfused",
            detail_lines=[
                "这不是解析异常，而是当前 modelRI 没有形成融合后的任务排布。"
            ],
            fallback_guidance=["如果当前仍保留 Scope 结构，请先回到 Scope View。"],
        )
    if not files.get("sk_device_args.log") or not files.get("sk_fused_nodes.log"):
        missing_names = [
            name
            for name in ("sk_device_args.log", "sk_fused_nodes.log")
            if not files.get(name)
        ]
        return _view_state(
            "missing",
            "缺少 " + "、".join(missing_names) + "，TaskQue 视图无法完整建立。",
            "task_queue_missing_assets",
            acquisition_hints=[
                "如需完整的 fused function 和 queue 证据，请先 export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1 后重新采集。"
            ],
            fallback_guidance=[
                "你仍可先看 Scope、Analysis 或 DFX，但当前 queue 解释会明显退化。"
            ],
        )
    if sections or functions:
        return _view_state(
            "available",
            "当前 TaskQue 事实源完整，可直接围绕队列和同步关系继续排查。",
            "task_queue_available",
        )
    return _view_state(
        "empty",
        "当前没有可展示的 TaskQue 事实源。",
        "task_queue_empty",
        detail_lines=[
            "当前既没有可用的 device task section，也没有足够的 fused function 结构来重建任务排布。"
        ],
        fallback_guidance=[
            "如果这不是预期结果，请先检查 `sk_device_args.log` 和 `sk_fused_nodes.log` 的原始内容。"
        ],
    )


def _build_analysis_view_state(
    asset_inventory: dict[str, Any],
    event_stats: dict[str, Any],
    node_trace_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resource_state = _build_analysis_resource_state(
        asset_inventory, event_stats, node_trace_summary
    )
    tracing = resource_state["tracing"]
    timing = resource_state["timing"]
    kernel_meta = resource_state["kernel_meta"]
    detail_lines: list[str] = []
    acquisition_hints: list[str] = []
    for item in (tracing, timing, kernel_meta):
        if item.get("level") != "available":
            if item.get("summary"):
                detail_lines.append(str(item["summary"]))
            if item.get("hint"):
                acquisition_hints.append(str(item["hint"]))
    if timing.get("level") == "error":
        state = _view_state(
            "error",
            "当前事件矩阵依赖的 time-event 解析异常，Analysis 不能按正常时间视角理解。",
            "analysis_timing_error",
            detail_lines=detail_lines,
            acquisition_hints=acquisition_hints,
            fallback_guidance=[
                "当前可以先利用 Scope/TaskQue/DFX 做结构和运行期排查，暂时不要把事件矩阵当作稳定依据。"
            ],
        )
    elif timing.get("level") in {"missing", "empty"}:
        state = _view_state(
            str(timing.get("level")),
            str(timing.get("summary") or "当前事件矩阵所需资源不可用。"),
            f"analysis_timing_{timing.get('level')}",
            detail_lines=detail_lines,
            acquisition_hints=acquisition_hints,
            fallback_guidance=[
                "当前 Analysis 仍可作为结构索引页，但时间结论会明显减弱。"
            ],
        )
    elif (
        tracing.get("level") in {"missing", "error"}
        or kernel_meta.get("level") != "available"
    ):
        state = _view_state(
            "partial",
            "当前 Analysis 可渲染，但 tracing 或结构资源不完整，部分跳转和解释会受限。",
            "analysis_partial_resources",
            detail_lines=detail_lines,
            acquisition_hints=acquisition_hints,
            fallback_guidance=["事件矩阵仍可先用于找热点，再结合 Scope/TaskQue 深挖。"],
        )
    else:
        state = _view_state(
            "available",
            "当前 Analysis 所需资源完整，可直接围绕事件矩阵和 tracing 深挖。",
            "analysis_available",
        )
    return state, resource_state


def _build_dfx_view_state(
    asset_inventory: dict[str, Any],
    update_report: dict[str, Any],
    hang_crash_summary: dict[str, Any],
) -> dict[str, Any]:
    files = (
        asset_inventory.get("files", {}) if isinstance(asset_inventory, dict) else {}
    )
    dfx_library = _update_dfx_library(update_report)
    payload_summary = (
        dfx_library.get("payload_registry", {}).get("summary", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    counter_summary = (
        dfx_library.get("counter_registry", {}).get("summary", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    runtime_status = str(payload_summary.get("runtime_status") or "unknown")
    counter_problem_kind = str(counter_summary.get("counter_problem_kind") or "unknown")
    if not files.get("super_kernel.log"):
        return _view_state(
            "missing",
            "当前缺少 `super_kernel.log`，DFX 证据链无法完整建立。",
            "dfx_missing_super_kernel",
            acquisition_hints=[
                "如需完整的 DFX 证据，请先 export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1 后重新采集。"
            ],
            fallback_guidance=["当前页面里的结论不能被视为完整 DFX 结果。"],
        )
    if counter_problem_kind == "op_trace_disabled":
        return _view_state(
            "blocked",
            "当前 `op_trace=false`，counter 侧诊断被阻断；DFX 仍可渲染，但不能把 counter 结果当作可用依据。",
            "dfx_counter_blocked",
            acquisition_hints=[
                "如需子核级 counter 诊断，请先 export ASCEND_SK_OP_TRACE_ON=1 后重新复现。"
            ],
            fallback_guidance=[
                "当前请优先结合 Exception Registry、Source Status 和已有异常信号继续判断。"
            ],
        )
    if runtime_status in {
        "missing_exception_handler_dump",
        "missing_exception_handler_rows",
        "ambiguous_device_args_section",
        "unresolved_device_args_section",
    }:
        return _view_state(
            "partial",
            "当前 DFX 可渲染，但 runtime/payload 证据不完整，结论可信度会下降。",
            "dfx_partial_payload",
            detail_lines=[f"当前 Source Status 为 `{runtime_status}`。"],
            fallback_guidance=[
                "当前请优先看 Actionable Signals 和 Source Status，不要只依赖单一 target 结论。"
            ],
        )
    return _view_state(
        "available",
        "当前 DFX 证据链完整，可直接围绕主诊断表继续定位。",
        "dfx_available",
        detail_lines=[str(hang_crash_summary.get("conclusion") or "")]
        if hang_crash_summary.get("conclusion")
        else [],
    )


def _analysis_trace_notice_html(
    node_trace_summary: dict[str, Any],
    run_dir: Path,
    output_path: Path,
    analysis_resource_state: dict[str, Any],
) -> str:
    trace_relative = node_trace_summary.get("trace_file")
    if trace_relative:
        trace_path = run_dir / str(trace_relative)
        trace_link = ""
        if trace_path.exists():
            trace_href = html.escape(
                os.path.relpath(trace_path, output_path.parent), quote=True
            )
            trace_link = f"<a href='{trace_href}' target='_blank' rel='noreferrer'>{html.escape(trace_path.name)}</a>"
        viewer_links = " / ".join(
            (
                f"<a href='{html.escape(str(target), quote=True)}' "
                f"target='_blank' rel='noreferrer'>{html.escape(str(target))}</a>"
            )
            for target in node_trace_summary.get("viewer_targets", [])[:2]
        )
        if trace_link and viewer_links:
            return (
                "当前原始 tracing 文件为 {trace_link}；事件矩阵里的 tracing 跳转会直接指向它。"
                "如需进一步查看，请先打开 {viewer_links}，再加载这个 tracing 文件。"
            ).format(trace_link=trace_link, viewer_links=viewer_links)
        if trace_link:
            return "当前原始 tracing 文件为 {trace_link}；事件矩阵里的 tracing 跳转会直接指向它。".format(
                trace_link=trace_link
            )
        if viewer_links:
            return "事件矩阵里的 tracing 跳转现在直接指向原始 tracing 文件；如需进一步查看，请先打开 {viewer_links}。".format(
                viewer_links=viewer_links
            )
        return "事件矩阵里的 tracing 跳转现在直接指向原始 tracing 文件。"
    trace_state = analysis_resource_state.get("tracing", {})
    if trace_state.get("level") == "error":
        return (
            _analysis_format_resource_text(str(trace_state.get("summary") or ""))
            + " "
            + _analysis_format_resource_text(str(trace_state.get("hint") or ""))
        )
    return (
        "当前没有原始 tracing 文件，未采集原始 tracing；如需在这里继续查看 tracing，请先使用 "
        "<span class='mono'>export ASCEND_PROF_SK_ON=1</span> 重新采集。"
    )


def _analysis_trace_action_markup(
    item: dict[str, Any],
    run_dir: Path,
    output_path: Path,
    node_trace_summary: dict[str, Any],
    analysis_resource_state: dict[str, Any],
) -> str:
    task_indices = item.get("task_indices", [])[:4]
    task_label = (
        " / ".join(f"task {task_index}" for task_index in task_indices) or "无 task"
    )
    trace_relative = node_trace_summary.get("trace_file")
    if trace_relative:
        trace_path = run_dir / str(trace_relative)
        if trace_path.exists():
            trace_href = html.escape(
                os.path.relpath(trace_path, output_path.parent), quote=True
            )
            viewer_targets = [
                (
                    f"<a href='{html.escape(str(target), quote=True)}' "
                    f"target='_blank' rel='noreferrer'>{html.escape(str(target))}</a>"
                )
                for target in node_trace_summary.get("viewer_targets", [])[:2]
            ]
            link_parts = [
                f"<a href='{trace_href}' target='_blank' rel='noreferrer'>tracing 文件</a>"
            ]
            link_parts.extend(viewer_targets)
            return f"<div><div>{html.escape(task_label)}</div><div>{' · '.join(link_parts)}</div></div>"
    trace_state = analysis_resource_state.get("tracing", {})
    if trace_state.get("level") == "error":
        hint = _analysis_resource_context_html(
            analysis_resource_state,
            include_tracing=True,
            include_timing=False,
            include_kernel_meta=True,
            include_path_hint=False,
        )
        return (
            "<div>"
            f"<div>{html.escape(task_label)}</div>"
            f"<div>{hint or 'Tracing 生成异常；请先检查 node tracing 工具输出。'}</div>"
            "</div>"
        )
    hint = _analysis_resource_context_html(
        analysis_resource_state,
        include_tracing=True,
        include_timing=False,
        include_kernel_meta=True,
        include_path_hint=False,
    )
    fallback_hint = (
        '未采集原始 tracing；请先 <span class="mono">export ASCEND_PROF_SK_ON=1</span>'
    )
    return f"<div><div>{html.escape(task_label)}</div><div>{hint or fallback_hint}</div></div>"


def _build_next_information_needed(
    missing_assets: list[dict[str, Any]],
    update_report: dict[str, Any],
    performance_summary: dict[str, Any],
    hang_crash_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen = set()

    for item in missing_assets:
        key = ("missing_asset", item["name"])
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "area": "missing_asset",
                "name": item["name"],
                "reason": item["reason"],
                "diagnosis_value": item["diagnosis_value"],
                "acquisition_hint": item["acquisition_hint"],
                "fallback_if_missing": item["fallback_if_missing"],
                "related_reports": item.get("related_reports", []),
            }
        )

    if performance_summary["focus"] != "structure_queue_time_correlation":
        items.append(
            {
                "area": "performance",
                "name": "performance_correlation_gap",
                "reason": performance_summary["reason"],
                "diagnosis_value": "用于提升结构、队列和时间证据之间的关联置信度。",
                "acquisition_hint": ASSET_GUIDANCE[EVENT_ASSET_KEY]["acquisition_hint"],
                "fallback_if_missing": "在尝试更深的时间归因前，先使用 scope、fused-node 和 queue 结构做基础判断。",
                "related_reports": ["performance_report"],
            }
        )

    if hang_crash_summary["signal_confidence"] in {"none", "low"}:
        items.append(
            {
                "area": "hang_crash",
                "name": "higher_confidence_failure_signal",
                "reason": "当前异常证据仍被 warning/noise 主导，或不足以支撑根因级别的判断。",
                "diagnosis_value": "用于在 hang/coredump 排查时把真实失败信号和运行期噪声分开。",
                "acquisition_hint": "复现问题时，请保留更丰富的 runtime/device 日志和同一轮运行中的异常相关 SK 证据。",
                "fallback_if_missing": "当前可先围绕 phase 标记、queue/DFX 证据和已观察到的 warning 做锚定排查。",
                "related_reports": ["hang_crash_report"],
            }
        )
    return items


class ReportArtifactsInput(NamedTuple):
    expected_reports: list[str]
    validation: dict[str, Any]
    report_links: dict[str, str]
    missing_assets: list[dict[str, Any]]
    asset_inventory: dict[str, Any]
    update_report: dict[str, Any]
    performance_summary: dict[str, Any]
    hang_crash_summary: dict[str, Any]
    node_trace_summary: dict[str, Any]
    presentation_states: dict[str, dict[str, str]] | None = None


def _build_report_artifacts(
    request: ReportArtifactsInput,
) -> dict[str, dict[str, Any]]:
    expected_reports = request.expected_reports
    validation = request.validation
    report_links = request.report_links
    missing_assets = request.missing_assets
    asset_inventory = request.asset_inventory
    update_report = request.update_report
    performance_summary = request.performance_summary
    hang_crash_summary = request.hang_crash_summary
    node_trace_summary = request.node_trace_summary
    presentation_states = request.presentation_states
    missing_by_name: dict[str, dict[str, Any]] = {}
    for item in missing_assets:
        missing_by_name[item["name"]] = item
        canonical_name = item.get("canonical_name")
        if canonical_name:
            missing_by_name[str(canonical_name)] = item
    artifacts = {}
    presentation_states = presentation_states or {}

    def navigation_hints(report_key: str) -> tuple[list[str], dict[str, Any]]:
        scope_ids = _update_scope_ids(update_report)
        node_ids = _update_node_ids(update_report)
        task_indices = _update_task_indices(update_report)
        entry_points: list[str] = []
        linked_objects: dict[str, Any] = {}
        if report_key == "scope_graph":
            scope_graph_href = "scope-graph.html"
            query_parts = []
            if scope_ids:
                query_parts.append(f"scopeId={scope_ids[0]}")
            if node_ids:
                query_parts.append(f"nodeId={node_ids[0]}")
            if query_parts:
                scope_graph_href = f"{scope_graph_href}?{'&'.join(query_parts)}"
            entry_points.append(scope_graph_href)
            linked_objects = {"scope_ids": scope_ids[:3], "node_ids": node_ids[:6]}
        elif report_key == "task_queue_graph":
            task_graph_href = "task-queue-graph.html"
            query_parts = []
            if task_indices:
                query_parts.append(f"taskIndex={task_indices[0]}")
            if node_ids:
                query_parts.append(f"nodeId={node_ids[0]}")
            if query_parts:
                task_graph_href = f"{task_graph_href}?{'&'.join(query_parts)}"
            entry_points.append(task_graph_href)
            linked_objects = {
                "task_indices": task_indices[:8],
                "node_ids": node_ids[:6],
            }
        elif report_key == "hang_crash_report":
            entry_points.append("hang-crash-report.html")
            linked_objects = {
                "scope_ids": scope_ids[:3],
                "task_indices": task_indices[:8],
                "likely_failure_stage": hang_crash_summary.get("likely_failure_stage"),
            }
        elif report_key == "performance_report":
            if performance_summary.get("recommended_entry"):
                entry_points.append(performance_summary["recommended_entry"])
            elif scope_ids:
                entry_points.append(f"scope-graph.html?scopeId={scope_ids[0]}")
            trace_entry = report_links.get("node_trace") or node_trace_summary.get(
                "trace_file"
            )
            if trace_entry:
                entry_points.append(str(trace_entry))
            linked_objects = {
                "scope_ids": scope_ids[:3],
                "task_indices": task_indices[:8],
                "node_ids": node_ids[:6],
                "linked_update_scope": performance_summary.get("linked_update_scope"),
                "trace_covered_node_ids": performance_summary.get(
                    "trace_covered_node_ids", []
                )[:8],
            }
        elif report_key == "node_trace":
            linked_objects = {
                "covered_node_ids": node_trace_summary.get("covered_node_ids", [])[:8],
                "covered_stream_ids": node_trace_summary.get("covered_stream_ids", [])[
                    :8
                ],
            }
        return entry_points, linked_objects

    artifact_keys = expected_reports + ["scope_graph", "task_queue_graph", "run_portal"]
    for key in list(dict.fromkeys(artifact_keys)):
        guidance = REPORT_GUIDANCE.get(key)
        if not guidance:
            continue
        report_state = validation["reports"].get(key, {})
        generation_status = report_state.get(
            "status", "ok" if key in report_links else "failed"
        )
        tool_message = report_state.get("message", "") or None
        blocking_reason = None
        acquisition_hint = None
        next_information_needed = []
        for asset_name in guidance["required_assets"]:
            if asset_name in missing_by_name:
                acquisition_hint = missing_by_name[asset_name]["acquisition_hint"]
                next_information_needed.append(asset_name)
        if generation_status != "ok" and report_state.get("message"):
            blocking_reason = report_state["message"]
        if (
            key == "performance_report"
            and performance_summary["focus"] != "structure_queue_time_correlation"
        ):
            blocking_reason = blocking_reason or performance_summary["reason"]
            next_information_needed.extend(["performance_correlation_gap"])
        if key == "hang_crash_report" and hang_crash_summary["signal_confidence"] in {
            "none",
            "low",
        }:
            blocking_reason = (
                blocking_reason or "Current crash evidence is still low-confidence."
            )
            next_information_needed.extend(["higher_confidence_failure_signal"])
        diagnostic_completeness, completeness_reason = _artifact_diagnostic_state(
            ArtifactDiagnosticStateInput(
                key,
                generation_status,
                asset_inventory,
                update_report,
                performance_summary,
                hang_crash_summary,
                node_trace_summary,
            )
        )
        if (
            completeness_reason
            and not blocking_reason
            and diagnostic_completeness != "complete"
        ):
            blocking_reason = completeness_reason
        if (
            generation_status != "ok"
            and not blocking_reason
            and next_information_needed
        ):
            blocking_reason = "Required evidence for this view is currently missing."
        recommended_entry_points, linked_objects = navigation_hints(key)
        artifacts[key] = {
            "title": guidance["title"],
            "status": generation_status,
            "generation_status": generation_status,
            "diagnostic_completeness": diagnostic_completeness,
            "path": report_links.get(key),
            "diagnosis_value": guidance["diagnosis_value"],
            "viewer_hint": guidance["viewer_hint"],
            "view_kind": guidance.get("view_kind", "structured_report"),
            "recommended_for": guidance.get("recommended_for", "general_debug"),
            "acquisition_hint": acquisition_hint,
            "blocking_reason": blocking_reason or None,
            "tool_message": tool_message,
            "next_information_needed": next_information_needed,
            "structured_output": guidance.get("structured_output"),
            "structured_path": report_links.get(f"{key}_data"),
            "recommended_next_step": next_information_needed[0]
            if next_information_needed
            else None,
            "recommended_entry_points": recommended_entry_points,
            "linked_objects": linked_objects,
            "presentation_mode": presentation_states.get(key, {}).get(
                "presentation_mode", "expanded"
            ),
            "presentation_reason": presentation_states.get(key, {}).get(
                "presentation_reason", "默认完整展示"
            ),
        }
    return artifacts


def _overall_diagnostic_completeness(
    report_artifacts: dict[str, dict[str, Any]],
) -> str:
    relevant = [
        artifact["diagnostic_completeness"]
        for key, artifact in report_artifacts.items()
        if key in {"node_trace", "hang_crash_report", "performance_report"}
    ]
    if not relevant:
        return "insufficient"
    if any(item == "insufficient" for item in relevant):
        if any(item in {"complete", "limited"} for item in relevant):
            return "limited"
        return "insufficient"
    if any(item == "limited" for item in relevant):
        return "limited"
    return "complete"


def _build_capability_summary(
    report_artifacts: dict[str, dict[str, Any]],
    asset_inventory: dict[str, Any],
    update_report: dict[str, Any],
    hang_crash_summary: dict[str, Any],
    performance_summary: dict[str, Any],
) -> tuple[list[str], list[str]]:
    current: list[str] = ["diagnostic_overview_ready", "summary_guidance_ready"]
    missing: list[str] = []
    graph_ready = False

    if report_artifacts.get("scope_graph", {}).get("generation_status") == "ok":
        current.append("scope_structure_ready")
        graph_ready = True
    else:
        missing.append("scope_graph_unavailable")

    if report_artifacts.get("task_queue_graph", {}).get("generation_status") == "ok":
        current.append("queue_structure_ready")
        graph_ready = True
    else:
        missing.append("queue_graph_unavailable")

    if graph_ready:
        current.append("graph_capable_mode")
    else:
        current.append("summary_only_mode")
        missing.append("interactive_graph_navigation")

    if report_artifacts.get("node_trace", {}).get("generation_status") == "ok":
        current.append("node_trace_ready")
    else:
        missing.append("node_trace_unavailable")

    if report_artifacts.get("hang_crash_report", {}).get("generation_status") == "ok":
        if hang_crash_summary.get("signal_confidence") in {"high", "medium"}:
            current.append("hang_triage_actionable")
        else:
            current.append("hang_triage_limited")
            missing.append("high_confidence_failure_signal")

    if report_artifacts.get("performance_report", {}).get("generation_status") == "ok":
        if asset_inventory.get("event_file_count", 0) > 0:
            current.append("performance_time_covered")
        else:
            current.append("performance_structure_only")
            missing.append("timing_coverage_gap")
        if performance_summary.get("focus"):
            current.append(f"performance_focus:{performance_summary['focus']}")

    return current, missing


def _decorate_graph_view(
    html_path: Path,
    title: str,
    model_ri: str | None,
    recommended_entry_points: list[str],
    linked_objects: dict[str, Any],
    *,
    portal_href: str = "../run-portal.html",
    view_state: dict[str, Any] | None = None,
) -> None:
    if not html_path.exists() or not view_state:
        return
    if str(view_state.get("level") or "available") == "available":
        return
    html_text = html_path.read_text(encoding="utf-8")
    state_markup = _view_state_graph_markup(view_state)
    if not state_markup:
        return
    if "</nav>" in html_text:
        html_text = html_text.replace("</nav>", f"</nav>\n    {state_markup}", 1)
    elif '<div class="page-shell">' in html_text:
        html_text = html_text.replace(
            '<div class="page-shell">',
            f'<div class="page-shell">\n    {state_markup}',
            1,
        )
    html_path.write_text(html_text, encoding="utf-8")


class HangReportHtmlInput(NamedTuple):
    run_dir: Path
    update_report: dict[str, Any]
    phase_correlated_signals: list[dict[str, Any]]
    hang_crash_summary: dict[str, Any]
    dfx_view_state: dict[str, Any]
    presentation_state: dict[str, str]
    output_path: Path
    portal_href: str = "../run-portal.html"


def _render_hang_report_html(
    request: HangReportHtmlInput,
) -> None:
    update_report = request.update_report
    phase_correlated_signals = request.phase_correlated_signals
    hang_crash_summary = request.hang_crash_summary
    dfx_view_state = request.dfx_view_state
    presentation_state = request.presentation_state
    output_path = request.output_path
    portal_href = request.portal_href
    dfx_library = _update_dfx_library(update_report)
    phase_registry = (
        dfx_library.get("phase_registry", {}) if isinstance(dfx_library, dict) else {}
    )
    payload_registry = (
        dfx_library.get("payload_registry", {}) if isinstance(dfx_library, dict) else {}
    )
    exception_registry = (
        dfx_library.get("exception_registry", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    counter_registry = (
        dfx_library.get("counter_registry", {}) if isinstance(dfx_library, dict) else {}
    )
    pc_localization_registry = (
        dfx_library.get("pc_localization_registry", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    diagnostic_pc_registry = (
        dfx_library.get("diagnostic_pc_registry", {})
        if isinstance(dfx_library, dict)
        else {}
    )
    presentation_mode = presentation_state["presentation_mode"]
    visible_actionable, hidden_actionable = _split_visible_hidden(
        hang_crash_summary.get("top_actionable_signals", []),
        PRESENTATION_LIMITS["hang_primary"] if presentation_mode == "focused" else 9999,
    )
    actionable_rows = []
    for item in visible_actionable:
        actionable_rows.append(
            "<tr><td>{priority}</td><td>{kind}</td><td>{file}:{line}</td><td>{text}</td></tr>".format(
                priority=html.escape(_priority_label(item.get("priority", "unknown"))),
                kind=html.escape(_signal_kind_label(item.get("kind", "signal"))),
                file=html.escape(Path(item.get("file", "")).name),
                line=html.escape(str(item.get("line", "?"))),
                text=html.escape(item.get("text", "")),
            )
        )
    visible_phase_rows, hidden_phase_rows = _split_visible_hidden(
        phase_correlated_signals,
        PRESENTATION_LIMITS["hang_secondary"]
        if presentation_mode == "focused"
        else 9999,
    )
    phase_rows = []
    for item in visible_phase_rows:
        phase_rows.append(
            "<tr><td>{phase}</td><td>{bucket}</td><td>{kind}</td><td>{file}:{line}</td><td>{text}</td></tr>".format(
                phase=html.escape(_phase_key_label(item.get("phase_key", "unknown"))),
                bucket=html.escape(_signal_bucket_label(item.get("bucket", "unknown"))),
                kind=html.escape(_signal_kind_label(item.get("kind", "signal"))),
                file=html.escape(Path(item.get("file", "")).name),
                line=html.escape(str(item.get("line", "?"))),
                text=html.escape(item.get("text", "")),
            )
        )

    payload_summary = (
        payload_registry.get("summary", {})
        if isinstance(payload_registry, dict)
        else {}
    )
    counter_summary = (
        counter_registry.get("summary", {})
        if isinstance(counter_registry, dict)
        else {}
    )
    pc_localization_summary = (
        pc_localization_registry.get("summary", {})
        if isinstance(pc_localization_registry, dict)
        else {}
    )
    payload_functions = (
        payload_registry.get("functions", [])
        if isinstance(payload_registry, dict)
        else []
    )
    payload_functions = payload_functions if isinstance(payload_functions, list) else []
    counter_cores = (
        counter_registry.get("cores", [])
        if isinstance(counter_registry.get("cores"), list)
        else []
    )
    pc_localization_events = (
        pc_localization_registry.get("events", [])
        if isinstance(pc_localization_registry.get("events"), list)
        else []
    )
    diagnostic_pc_events = (
        diagnostic_pc_registry.get("events", [])
        if isinstance(diagnostic_pc_registry.get("events"), list)
        else []
    )
    exception_events = (
        exception_registry.get("events", [])
        if isinstance(exception_registry, dict)
        else []
    )
    exception_events = exception_events if isinstance(exception_events, list) else []
    core_symbol_events = (
        exception_registry.get("core_symbol_events", [])
        if isinstance(exception_registry, dict)
        else []
    )
    core_symbol_events = (
        core_symbol_events if isinstance(core_symbol_events, list) else []
    )
    warning_signals = hang_crash_summary.get("warning_signals", [])
    warning_signals = warning_signals if isinstance(warning_signals, list) else []
    phase_stats = (
        phase_registry.get("phase_stats", [])
        if isinstance(phase_registry, dict)
        else []
    )
    phase_stats = phase_stats if isinstance(phase_stats, list) else []

    def _chips(values: list[str]) -> str:
        if not values:
            return "<span class='capability-chip'>无</span>"
        return "".join(
            "<span class='capability-chip'>{}</span>".format(html.escape(str(value)))
            for value in values
        )

    def _candidate_text(candidates: list[dict[str, Any]]) -> str:
        parts = []
        for candidate in candidates:
            parts.append(
                "scope={scope}, sk={sk}, nodes={nodes}".format(
                    scope=candidate.get("scope_id"),
                    sk=candidate.get("sk_id"),
                    nodes=",".join(str(item) for item in candidate.get("node_ids", []))
                    or "无",
                )
            )
        return "；".join(parts) if parts else "无"

    analysis_status = str(hang_crash_summary.get("analysis_status", "clean"))
    counter_problem_kind = str(counter_summary.get("counter_problem_kind", "unknown"))
    if analysis_status == "clean":
        current_decision = (
            "当前模型资产侧维测分析未见异常，不需要再围绕 payload 缺失做过度判断。"
        )
    elif analysis_status == "warning":
        current_decision = "当前未发现明确异常，但存在 warning；请先看 Warning 明细，再决定是否需要继续深挖。"
    elif counter_problem_kind == "op_trace_disabled":
        current_decision = "当前 `op_trace=false`，counter 侧诊断已禁用；请先打开 `ASCEND_SK_OP_TRACE_ON=1` 后重新复现。"
    elif counter_problem_kind == "sk_level_issue":
        current_decision = "当前没有 launch=2，且已使用 core 都停在 launch=3；这更像是 SK 级逻辑/同步问题，应优先检查 SK 自身状态，而不是继续按子算子或异常 core 去定位。"
    elif pc_localization_summary.get("matched_count", 0) > 0:
        current_decision = "当前已经命中子算子 PC 区间；对子算子问题，应优先看每个 core 对应的 entry_start_pc，而不是 exception_start_pc。"
    elif counter_summary.get("dominant_active_function_name"):
        current_decision = "当前已经可以借助 SkCounterInfo 收敛到正在运行的子核函数，可优先围绕 opId 和 function 排查。"
    elif hang_crash_summary.get("needs_op_trace_rerun"):
        current_decision = "当前异常已成立，但对子核的定位证据仍不足；建议打开 ASCEND_SK_OP_TRACE_ON=1 重新复现。"
    elif payload_summary.get("runtime_status") == "missing_exception_handler_rows":
        current_decision = "当前拿到了 runtime dump 头，但没有 node rows；需要继续确认异常处理日志是否完整。"
    elif payload_functions:
        current_decision = "当前已具备 runtime payload 事实，但还没有 superkernel exception；先确认样例是否真的进入异常链路。"
    elif exception_events:
        current_decision = "先用异常函数和 runtime payload 建立落点，再结合 core symbol 线索缩小到具体 SK 或 node。"
    else:
        current_decision = "当前以故障信号为主，优先围绕异常函数、core symbol 和 payload 对照继续定位。"

    payload_summary_rows = [
        "<tr><td>runtime_status</td><td>{}</td></tr>".format(
            html.escape(str(payload_summary.get("runtime_status", "unknown")))
        ),
        "<tr><td>matched_function_count</td><td>{}</td></tr>".format(
            html.escape(str(payload_summary.get("matched_function_count", 0)))
        ),
        "<tr><td>runtime_entry_count</td><td>{}</td></tr>".format(
            html.escape(str(payload_summary.get("runtime_entry_count", 0)))
        ),
    ]

    pc_localization_summary_rows = [
        "<tr><td>event_count</td><td>{}</td></tr>".format(
            html.escape(str(pc_localization_summary.get("event_count", 0)))
        ),
        "<tr><td>matched_count</td><td>{}</td></tr>".format(
            html.escape(str(pc_localization_summary.get("matched_count", 0)))
        ),
        "<tr><td>unresolved_count</td><td>{}</td></tr>".format(
            html.escape(str(pc_localization_summary.get("unresolved_count", 0)))
        ),
    ]

    phase_registry_rows = []
    for item in phase_stats:
        phase_registry_rows.append(
            (
                "<tr><td>{key}</td><td>{count}</td><td>{first}</td><td>{last}</td><td>{sample}</td></tr>"
            ).format(
                key=html.escape(_phase_key_label(str(item.get("key", "unknown")))),
                count=html.escape(str(item.get("count", 0))),
                first=html.escape(str(item.get("first_line", "?"))),
                last=html.escape(str(item.get("last_line", "?"))),
                sample=html.escape(str(item.get("sample_message", ""))),
            )
        )

    payload_function_rows = []
    for item in payload_functions:
        payload_function_rows.append(
            (
                "<tr><td>{scope}</td><td>{sk}</td><td>{count}</td><td>{status}</td><td>{level}</td><td>{func}</td></tr>"
            ).format(
                scope=html.escape(str(item.get("scope_id", "unknown"))),
                sk=html.escape(str(item.get("sk_id", "unknown"))),
                count=html.escape(str(item.get("dfx_entry_count", 0))),
                status=html.escape(str(item.get("dfx_count_status", "unknown"))),
                level=html.escape(
                    _evidence_level_label(str(item.get("evidence_level", "unknown")))
                ),
                func=html.escape(str(item.get("function_text", "unknown"))),
            )
        )

    payload_detail_blocks = []
    for idx, item in enumerate(payload_functions, start=1):
        row_markup = []
        for row in item.get("rows", []):
            compare = row.get("device_args_compare", {})
            row_markup.append(
                (
                    "<tr><td>{node}</td><td>{eh_bin}</td><td>{eh_ori}</td>"
                    "<td>{eh_aic}</td><td>{eh_aiv}</td><td>{cmp_bin}</td>"
                    "<td>{cmp_ori}</td></tr>"
                ).format(
                    node=html.escape(str(row.get("node_index", "?"))),
                    eh_bin=html.escape(
                        str(
                            row.get("exception_handler", {}).get(
                                "bin_handle", "unknown"
                            )
                        )
                    ),
                    eh_ori=html.escape(
                        str(
                            row.get("exception_handler", {}).get(
                                "original_handle", "unknown"
                            )
                        )
                    ),
                    eh_aic=html.escape(
                        str(row.get("exception_handler", {}).get("aic_size", "unknown"))
                    ),
                    eh_aiv=html.escape(
                        str(row.get("exception_handler", {}).get("aiv_size", "unknown"))
                    ),
                    cmp_bin=html.escape(str(compare.get("bin_handle", "-"))),
                    cmp_ori=html.escape(str(compare.get("original_handle", "-"))),
                )
            )
        payload_detail_blocks.append(
            (
                "<details class='fold subblock'><summary>Payload Function {idx} · "
                "{func}</summary><div><p class='hint'>Reasons: {reasons}</p>"
                "{table}</div></details>"
            ).format(
                idx=idx,
                func=html.escape(str(item.get("function_text", "unknown"))),
                reasons=html.escape(
                    ", ".join(map(str, item.get("evidence_reasons", []))) or "无"
                ),
                table=_wrap_table_markup(
                    (
                        "<table><thead><tr><th>nodeIndex</th><th>EH bin</th>"
                        "<th>EH ori</th><th>EH aic</th><th>EH aiv</th>"
                        "<th>Compare bin</th><th>Compare ori</th></tr></thead>"
                        "<tbody>{}</tbody></table>"
                    ).format(
                        "".join(row_markup)
                        or "<tr><td colspan='7'>当前没有 payload rows。</td></tr>"
                    ),
                    panel_id=f"payload-function-{idx}",
                    title="Payload Function Detail",
                    subtitle="这里保留当前 payload function 的原始对照行，按需展开查看。",
                    min_width=980,
                ),
            )
        )

    exception_rows = []
    for item in exception_events:
        exception_rows.append(
            "<tr><td>{func}</td><td>{trace}</td><td>{cand}</td></tr>".format(
                func=html.escape(str(item.get("function", ""))),
                trace=html.escape(str(item.get("op_trace", ""))),
                cand=html.escape(_candidate_text(item.get("candidates", []))),
            )
        )

    core_symbol_rows = []
    for item in core_symbol_events:
        core_symbol_rows.append(
            (
                "<tr><td>{kind}</td><td>{line}</td><td>{core}</td><td>{node}</td><td>{entry}</td><td>{func}</td></tr>"
            ).format(
                kind=html.escape(str(item.get("kind", "unknown"))),
                line=html.escape(str(item.get("line", "?"))),
                core=html.escape(str(item.get("core_id", "?"))),
                node=html.escape(str(item.get("node_index", "-"))),
                entry=html.escape(str(item.get("entry_index", "-"))),
                func=html.escape(str(item.get("function_name", "-"))),
            )
        )

    warning_rows = []
    for item in warning_signals:
        warning_rows.append(
            "<tr><td>{line}</td><td>{file}</td><td>{text}</td></tr>".format(
                line=html.escape(str(item.get("line", "?"))),
                file=html.escape(Path(item.get("file", "")).name),
                text=html.escape(str(item.get("text", ""))),
            )
        )

    counter_rows = []
    for item in counter_cores:
        counter_rows.append(
            "<tr><td>{core}</td><td>{status}</td><td>{index}</td><td>{func}</td><td>{start_pc}</td></tr>".format(
                core=html.escape(str(item.get("core_id", "?"))),
                status=html.escape(
                    "Unused"
                    if item.get("is_unused_core")
                    else _counter_launch_state_label(
                        str(item.get("launch_state", "UNKNOWN"))
                    )
                ),
                index=html.escape(str(item.get("index", "?"))),
                func=html.escape(str(item.get("function_name", "-"))),
                start_pc=html.escape(str(item.get("derived_entry_start_pc", "-"))),
            )
        )

    affected_target_rows = []
    for item in diagnostic_pc_events:
        rule_text = (
            str(item.get("entry_slot_rule"))
            if item.get("entry_slot_rule") is not None
            else ""
        )
        affected_target_rows.append(
            (
                "<tr title='{title}'><td>{core}</td><td>{type}</td>"
                "<td>{func}</td><td>{start}</td><td>{current}</td>"
                "<td>{summary}</td></tr>"
            ).format(
                title=html.escape(rule_text or "final diagnostic target"),
                core=html.escape(str(item.get("core_id", "?"))),
                type=html.escape(str(item.get("core_type", "-"))),
                func=html.escape(
                    str(
                        item.get("function_name")
                        if item.get("function_name") is not None
                        else "-"
                    )
                ),
                start=html.escape(
                    str(
                        item.get("reported_start_pc")
                        if item.get("reported_start_pc") is not None
                        else "-"
                    )
                ),
                current=html.escape(
                    str(
                        item.get("reported_current_pc")
                        if item.get("reported_current_pc") is not None
                        else "-"
                    )
                ),
                summary=html.escape(
                    "SK issue"
                    if str(item.get("issue_kind", "-")) == "sk"
                    else "Subkernel issue"
                ),
            )
        )

    diagnostic_pc_rows = []
    for item in diagnostic_pc_events:
        rule_text = (
            str(item.get("entry_slot_rule"))
            if item.get("entry_slot_rule") is not None
            else ""
        )
        diagnostic_pc_rows.append(
            (
                "<tr title='{title}'><td>{core}</td><td>{type}</td>"
                "<td>{op}</td><td>{func}</td><td>{start}</td>"
                "<td>{current}</td><td>{basis}</td></tr>"
            ).format(
                title=html.escape(rule_text or "diagnostic pc target"),
                core=html.escape(str(item.get("core_id", "?"))),
                type=html.escape(str(item.get("core_type", "-"))),
                op=html.escape(
                    str(item.get("op_id") if item.get("op_id") is not None else "-")
                ),
                func=html.escape(
                    str(
                        item.get("function_name")
                        if item.get("function_name") is not None
                        else "-"
                    )
                ),
                start=html.escape(
                    str(
                        item.get("reported_start_pc")
                        if item.get("reported_start_pc") is not None
                        else "-"
                    )
                ),
                current=html.escape(
                    str(
                        item.get("reported_current_pc")
                        if item.get("reported_current_pc") is not None
                        else "-"
                    )
                ),
                basis=html.escape(str(item.get("reported_pc_basis", "-"))),
            )
        )

    pc_localization_rows = []
    for item in pc_localization_events:
        pc_localization_rows.append(
            (
                "<tr><td>{core}</td><td>{type}</td><td>{status}</td>"
                "<td>{esp}</td><td>{cp}</td><td>{entry_start}</td>"
                "<td>{entry_end}</td><td>{node}</td><td>{entry}</td>"
                "<td>{func}</td></tr>"
            ).format(
                core=html.escape(str(item.get("core_id", "?"))),
                type=html.escape(str(item.get("core_type", "-"))),
                status=html.escape(str(item.get("pc_match_status", "unknown"))),
                esp=html.escape(str(item.get("exception_start_pc", "-"))),
                cp=html.escape(str(item.get("current_pc", "-"))),
                entry_start=html.escape(str(item.get("entry_start_pc", "-"))),
                entry_end=html.escape(str(item.get("entry_end_pc", "-"))),
                node=html.escape(str(item.get("node_index", "-"))),
                entry=html.escape(str(item.get("entry_index", "-"))),
                func=html.escape(str(item.get("function_name", "-"))),
            )
        )

    nav_markup = render_view_nav(
        [
            ("导航页面", portal_href, False),
            ("Scope View", "scope-graph.html", False),
            ("TaskQue View", "task-queue-graph.html", False),
            ("Analysis View", "performance-report.html", False),
            ("DFX View", "hang-crash-report.html", True),
        ],
        kicker="",
    )
    header_markup = render_report_header(
        icon="D",
        kicker="",
        title="DFX View",
        note_html=html.escape("先看顶部结论，再进入主表格。"),
        nav_html=nav_markup,
    )
    overview_markup = _render_overview_section(
        title="当前摘要",
        note_html=_join_html_fragments(
            "先看当前结论、信号强度和阶段，再往下读主表格。",
            current_decision,
            _view_state_note_html(
                dfx_view_state,
                include_hints=True,
                include_fallbacks=True,
                include_summary=False,
            ),
        ),
        items=[
            ("Conclusion", _analysis_status_label(analysis_status)),
            (
                "Signal",
                _signal_confidence_label(
                    str(hang_crash_summary.get("signal_confidence", "unknown"))
                ),
            ),
            (
                "Failure Stage",
                _failure_stage_label(
                    str(hang_crash_summary.get("likely_failure_stage", "unknown"))
                ),
            ),
            ("State", _view_state_label(dfx_view_state)),
        ],
    )
    intro_markup = render_report_intro_section(
        title="当前摘要",
        note_html="这里先固定 DFX 入口的最小上下文，再进入后续主信号和诊断表格。",
        summary_html=render_report_summary(
            [
                ("Conclusion", _analysis_status_label(analysis_status)),
                (
                    "Signal",
                    _signal_confidence_label(
                        str(hang_crash_summary.get("signal_confidence", "unknown"))
                    ),
                ),
                (
                    "Stage",
                    _failure_stage_label(
                        str(hang_crash_summary.get("likely_failure_stage", "unknown"))
                    ),
                ),
            ],
            compact=True,
        ),
    )
    actionable_table_markup = _wrap_table_markup(
        (
            "<table><thead><tr><th>优先级</th><th>信号类型</th>"
            "<th>来源</th><th>内容</th></tr></thead><tbody>{}</tbody></table>"
        ).format(
            "".join(actionable_rows)
            or "<tr><td colspan='4'>当前没有可直接执行的主信号。</td></tr>"
        ),
        panel_id="actionable-signals",
        title="Actionable Signals",
        subtitle=(
            "这里只保留本轮最值得看的主信号，不再堆叠次级解释。"
            + (
                " "
                + _view_state_note_html(
                    dfx_view_state,
                    include_hints=False,
                    include_fallbacks=False,
                    include_summary=False,
                )
                if str(dfx_view_state.get("level"))
                in {"blocked", "partial", "error", "missing"}
                else ""
            )
        ),
        min_width=960,
        collapsed=_collapse_by_importance(
            row_count=len(actionable_rows), importance="primary"
        ),
        prominent=True,
        search_placeholder="查找优先级、信号类型、来源、内容",
    )
    actionable_more_markup = (
        "<details class='fold subblock'><summary>更多主信号 ({})</summary><div>{}</div></details>".format(
            len(hidden_actionable),
            _wrap_table_markup(
                (
                    "<table><thead><tr><th>优先级</th><th>信号类型</th>"
                    "<th>来源</th><th>内容</th></tr></thead>"
                    "<tbody>{}</tbody></table>"
                ).format(
                    "".join(
                        "<tr><td>{}</td><td>{}</td><td>{}:{}</td><td>{}</td></tr>".format(
                            html.escape(
                                _priority_label(item.get("priority", "unknown"))
                            ),
                            html.escape(_signal_kind_label(item.get("kind", "signal"))),
                            html.escape(Path(item.get("file", "")).name),
                            html.escape(str(item.get("line", "?"))),
                            html.escape(item.get("text", "")),
                        )
                        for item in hidden_actionable
                    )
                ),
                panel_id="actionable-signals-more",
                title="更多 Actionable Signals",
                subtitle="这里保留超出首屏的次级主信号，按需展开查看。",
                min_width=960,
                collapsed=True,
                search_placeholder="查找更多主信号",
            ),
        )
        if hidden_actionable
        else ""
    )
    html_text = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DFX View</title>
  <style>{styles}</style>
</head>
<body>
  <div class="page-shell">
    {header_markup}
    {intro_markup}
    {overview_markup}
    {actionable_table}
    {actionable_more}
    <div class="dfx-stack">
      {affected_targets_table}
      {counter_detail_table}
      {diagnostic_pc_table}
      {exception_table}
      {warning_table}
      {payload_summary_table}
    </div>
    <details class="fold card">
      <summary>Raw Evidence</summary>
      <div class="grid">
        {phase_registry_table}
        <div class="raw-evidence-stack">
          {pc_localization_summary_table}
          {pc_localization_table}
        </div>
      </div>
      <div class="grid">
        <div class="raw-evidence-stack">
          {payload_function_table}
          {payload_function_details}
        </div>
        {core_symbol_table}
      </div>
      <div class="grid">
        <div class="raw-evidence-stack">
          {phase_table}
          {phase_more}
        </div>
      </div>
    </details>
  </div>
</body>
<script>
{table_js}
document.querySelectorAll('[data-standard-table]').forEach(initStandardTablePanel);
</script>
</html>
""".format(
        styles=_shared_report_styles(),
        header_markup=header_markup,
        intro_markup=intro_markup,
        overview_markup=overview_markup,
        actionable_table=actionable_table_markup,
        actionable_more=actionable_more_markup,
        phase_registry_table=_wrap_table_markup(
            (
                "<table><thead><tr><th>Phase</th><th>Count</th>"
                "<th>First Line</th><th>Last Line</th><th>Sample</th>"
                "</tr></thead><tbody>{}</tbody></table>"
            ).format(
                "".join(phase_registry_rows)
                or "<tr><td colspan='5'>当前没有可展示的 phase registry。</td></tr>"
            ),
            panel_id="phase-registry",
            title="Phase Registry",
            subtitle="这里汇总优化阶段的关键里程碑，按需展开查看。",
            min_width=1040,
            collapsed=True,
            search_placeholder="查找 phase、sample",
        ),
        payload_summary_table=_wrap_table_markup(
            "<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>{}</tbody></table>".format(
                "".join(payload_summary_rows)
            ),
            panel_id="payload-summary",
            title="Source Status",
            subtitle=(
                "这里只看 DFX 事实源是否完整，例如 runtime dump 是否有 rows、是否成功对上 device args。"
                + (
                    " "
                    + _view_state_note_html(
                        dfx_view_state,
                        include_hints=True,
                        include_fallbacks=False,
                        include_summary=False,
                    )
                    if _view_state_note_html(
                        dfx_view_state,
                        include_hints=True,
                        include_fallbacks=False,
                        include_summary=False,
                    )
                    else ""
                )
            ),
            min_width=640,
            collapsed=False,
            search_placeholder="查找 source status 字段",
        ),
        phase_table=_wrap_table_markup(
            (
                "<table><thead><tr><th>阶段</th><th>信号桶</th>"
                "<th>信号类型</th><th>来源</th><th>内容</th></tr></thead>"
                "<tbody>{}</tbody></table>"
            ).format(
                "".join(phase_rows)
                or "<tr><td colspan='5'>当前没有可稳定提取的阶段关联信号。</td></tr>"
            ),
            panel_id="phase-correlations",
            title="Phase Correlations",
            subtitle="这里保留阶段与信号的关联，适合在主结论之外继续深挖。",
            min_width=1040,
            collapsed=True,
            search_placeholder="查找阶段关联信号",
        ),
        payload_function_table=_wrap_table_markup(
            (
                "<table><thead><tr><th>Scope</th><th>skId</th><th>Rows</th>"
                "<th>Status</th><th>Evidence</th><th>Function</th></tr></thead>"
                "<tbody>{}</tbody></table>"
            ).format(
                "".join(payload_function_rows)
                or "<tr><td colspan='6'>当前没有可展示的 runtime payload function。</td></tr>"
            ),
            panel_id="payload-functions",
            title="Payload Functions",
            subtitle="这里保留 runtime payload function 的映射结果，按需展开查看。",
            min_width=960,
            collapsed=True,
            search_placeholder="查找 payload function",
        ),
        payload_function_details="".join(payload_detail_blocks),
        affected_targets_table=_wrap_paginated_table_markup(
            "affected-targets",
            ["Core", "CoreType", "Function", "startPC", "currentPC", "Summary"],
            affected_target_rows,
            title="Affected Targets",
            subtitle="这里只保留用户最需要的 target 信息：哪个 core、哪个函数、startPC 和 currentPC。",
            min_width=980,
            page_size=6,
            prominent=True,
            collapsed=_collapse_by_importance(
                row_count=len(affected_target_rows), importance="primary"
            ),
            search_placeholder="查找 core、函数、startPC、currentPC",
            empty_message="当前还没有足够证据直接收敛到具体 target。",
        ),
        counter_detail_table=_wrap_paginated_table_markup(
            "counter-registry",
            ["Core", "Used / Status", "opId", "Function", "entry_start_pc"],
            counter_rows if counter_problem_kind != "op_trace_disabled" else [],
            title="Counter Registry",
            subtitle=(
                "当前 `op_trace=false`，counter 侧诊断已禁用；这里只保留原始计数器状态，请先打开 ASCEND_SK_OP_TRACE_ON=1 后重新复现。"
                if counter_problem_kind == "op_trace_disabled"
                else "这里先回答“是 SK 问题，还是子算子问题”。"
            ),
            min_width=920,
            page_size=8,
            collapsed=_collapse_by_importance(
                row_count=len(
                    counter_rows if counter_problem_kind != "op_trace_disabled" else []
                ),
                importance="primary",
                has_empty_message=False,
            ),
            search_placeholder="查找 core、状态、opId、函数",
            empty_message="op_trace=false，禁止使用 counter 结果做诊断；请重新开启开关后复现。",
        ),
        diagnostic_pc_table=_wrap_paginated_table_markup(
            "diagnostic-pc",
            [
                "Core",
                "CoreType",
                "opId",
                "Function",
                "reported_start_pc",
                "reported_current_pc",
                "Basis",
            ],
            diagnostic_pc_rows,
            title="Diagnostic PC",
            subtitle="这里只保留最终诊断需要的 PC 字段：core、函数、startPC、currentPC 和判定来源。",
            min_width=1040,
            page_size=6,
            collapsed=_collapse_by_importance(
                row_count=len(diagnostic_pc_rows), importance="primary"
            ),
            search_placeholder="查找 core、函数、reported pc、basis",
            empty_message="当前没有可展示的 diagnostic pc targets。",
        ),
        pc_localization_summary_table=_wrap_table_markup(
            "<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>{}</tbody></table>".format(
                "".join(pc_localization_summary_rows)
            ),
            panel_id="pc-localization-summary",
            title="PC Localization Summary",
            subtitle="这里汇总 PC localization 的可用度，便于判断后续明细是否可信。",
            min_width=640,
            collapsed=True,
            search_placeholder="查找 PC localization summary",
        ),
        pc_localization_table=_wrap_table_markup(
            (
                "<table><thead><tr><th>Core</th><th>CoreType</th>"
                "<th>Status</th><th>exception_start_pc</th><th>current_pc</th>"
                "<th>entry_start_pc</th><th>entry_end_pc</th><th>nodeIndex</th>"
                "<th>entryIndex</th><th>Function</th></tr></thead>"
                "<tbody>{}</tbody></table>"
            ).format(
                "".join(pc_localization_rows)
                or "<tr><td colspan='10'>当前没有可展示的 PC localization 事件。</td></tr>"
            ),
            panel_id="pc-localization",
            title="PC Localization",
            subtitle="这里保留定位到 core 和函数的原始 PC localization 事件，按需展开查看。",
            min_width=1500,
            collapsed=True,
            search_placeholder="查找 PC localization 事件",
        ),
        exception_table=_wrap_paginated_table_markup(
            "exception-registry",
            ["Function", "op_trace", "Resolved Objects"],
            exception_rows,
            title="Exception Registry",
            subtitle="这里只保留异常函数和映射对象，足够作为跳转入口。",
            min_width=1120,
            page_size=6,
            collapsed=_collapse_by_importance(
                row_count=len(exception_rows), importance="secondary"
            ),
            search_placeholder="查找异常函数、op_trace、映射对象",
            empty_message="当前没有可展示的 exception event。",
        ),
        warning_table=_wrap_paginated_table_markup(
            "warning-registry",
            ["Line", "File", "Warning"],
            warning_rows,
            title="Warnings",
            subtitle="这里保留当前 run 中最值得继续追踪的 warning 明细。",
            min_width=1120,
            page_size=8,
            collapsed=_collapse_by_importance(
                row_count=len(warning_rows), importance="secondary"
            ),
            search_placeholder="查找 warning 内容",
            empty_message="当前没有需要额外关注的 warning。",
        ),
        core_symbol_table=_wrap_table_markup(
            (
                "<table><thead><tr><th>Kind</th><th>Line</th><th>Core</th>"
                "<th>nodeIndex</th><th>entryIndex</th><th>Function</th></tr></thead>"
                "<tbody>{}</tbody></table>"
            ).format(
                "".join(core_symbol_rows)
                or "<tr><td colspan='6'>当前没有可展示的 core symbol event。</td></tr>"
            ),
            panel_id="core-symbol-events",
            title="Core Symbol Events",
            subtitle="这里保留 core symbol 侧的原始事件，按需展开查看。",
            min_width=900,
            collapsed=True,
            search_placeholder="查找 core symbol 事件",
        ),
        phase_more=(
            "<details class='fold subblock'><summary>更多阶段关联信号 ({})</summary><div>{}</div></details>".format(
                len(hidden_phase_rows),
                _wrap_table_markup(
                    (
                        "<table><thead><tr><th>阶段</th><th>信号桶</th>"
                        "<th>信号类型</th><th>来源</th><th>内容</th></tr></thead>"
                        "<tbody>{}</tbody></table>"
                    ).format(
                        "".join(
                            (
                                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}:{}</td><td>{}</td></tr>"
                            ).format(
                                html.escape(
                                    _phase_key_label(item.get("phase_key", "unknown"))
                                ),
                                html.escape(
                                    _signal_bucket_label(item.get("bucket", "unknown"))
                                ),
                                html.escape(
                                    _signal_kind_label(item.get("kind", "signal"))
                                ),
                                html.escape(Path(item.get("file", "")).name),
                                html.escape(str(item.get("line", "?"))),
                                html.escape(item.get("text", "")),
                            )
                            for item in hidden_phase_rows
                        )
                    ),
                    panel_id="phase-correlations-more",
                    title="更多阶段关联信号",
                    subtitle="这里保留超出首屏的阶段关联信号，按需展开查看。",
                    min_width=1040,
                    collapsed=True,
                    search_placeholder="查找更多阶段关联信号",
                ),
            )
            if hidden_phase_rows
            else ""
        ),
        table_js=COMMON_STANDARD_TABLE_JS,
    )
    output_path.write_text(html_text, encoding="utf-8")


def _comma_join_ids(values: Iterable[Any]) -> str:
    parts = []
    for value in values:
        parts.append(str(value))
    return ",".join(parts)


class PerformanceReportHtmlInput(NamedTuple):
    run_dir: Path
    update_report: dict[str, Any]
    asset_inventory: dict[str, Any]
    event_stats: dict[str, Any]
    performance_correlations: list[dict[str, Any]]
    performance_summary: dict[str, Any]
    node_trace_summary: dict[str, Any]
    analysis_view_state: dict[str, Any]
    analysis_resource_state: dict[str, Any]
    presentation_state: dict[str, str]
    output_path: Path
    portal_href: str = "../run-portal.html"


def _render_performance_report_html(
    request: PerformanceReportHtmlInput,
) -> None:
    run_dir = request.run_dir
    event_stats = request.event_stats
    performance_correlations = request.performance_correlations
    performance_summary = request.performance_summary
    node_trace_summary = request.node_trace_summary
    analysis_view_state = request.analysis_view_state
    analysis_resource_state = request.analysis_resource_state
    presentation_state = request.presentation_state
    output_path = request.output_path
    portal_href = request.portal_href
    presentation_mode = presentation_state["presentation_mode"]
    visible_rows, hidden_rows = _split_visible_hidden(
        performance_correlations,
        PRESENTATION_LIMITS["performance_matrix"]
        if presentation_mode == "focused"
        else 9999,
    )
    visible_top_events, hidden_top_events = _split_visible_hidden(
        performance_summary.get("top_events", []),
        6 if presentation_mode == "focused" else 9999,
    )
    event_matrix_row_template = (
        "<tr><td><a href='{update_link}'>{group}</a></td><td>{sk}</td>"
        "<td>{devices}</td><td>{event_count}</td><td>{total_dur}</td>"
        "<td>{matched_scopes}</td><td>{matched_nodes}</td><td>{judgment}</td>"
        "<td>{trace_links}</td></tr>"
    )
    event_matrix_table_template = (
        "<table><thead><tr><th>事件分组</th><th>skId</th><th>设备</th>"
        "<th>事件数</th><th>总时长</th><th>关联 Scope</th><th>关联 Node</th>"
        "<th>判断</th><th>Tracing 跳转</th></tr></thead><tbody>{}</tbody></table>"
    )
    top_event_row_template = (
        "<tr><td>{name}</td><td>{device}</td><td>{pid}</td><td>{tid}</td>"
        "<td>{duration}</td><td>{sk}</td><td>{node}</td></tr>"
    )
    top_event_table_template = (
        "<table><thead><tr><th>事件摘要</th><th>设备</th><th>pid</th>"
        "<th>tid</th><th>时长</th><th>skId</th><th>nodeId</th></tr></thead>"
        "<tbody>{}</tbody></table>"
    )
    top_event_more_table_template = (
        "<table><thead><tr><th>事件摘要</th><th>原始事件名</th><th>设备</th>"
        "<th>pid</th><th>tid</th><th>时长</th><th>skId</th><th>nodeId</th>"
        "</tr></thead><tbody>{}</tbody></table>"
    )
    top_event_more_row_template = (
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
        "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>"
    )
    matrix_rows = []
    for item in visible_rows:
        judgment = _performance_judgment_label(_performance_judgment(item, event_stats))
        update_link = (
            f"scope-graph.html?scopeId={item['linked_update_scope']}"
            if item.get("linked_update_scope") is not None
            else "scope-graph.html"
        )
        trace_links = _analysis_trace_action_markup(
            item, run_dir, output_path, node_trace_summary, analysis_resource_state
        )
        matrix_rows.append(
            event_matrix_row_template.format(
                update_link=html.escape(update_link),
                group=html.escape(str(item.get("group_label"))),
                sk=html.escape(str(item.get("sk_id"))),
                devices=html.escape(",".join(item.get("devices", [])) or "无"),
                event_count=html.escape(str(item.get("event_count", 0))),
                total_dur=html.escape(str(item.get("total_duration", 0))),
                matched_scopes=html.escape(
                    _comma_join_ids(item.get("matched_scope_ids", [])) or "无"
                ),
                matched_nodes=html.escape(
                    _comma_join_ids(item.get("matched_event_node_ids", [])) or "无"
                ),
                judgment=html.escape(judgment),
                trace_links=trace_links,
            )
        )
    top_event_rows = []
    for item in visible_top_events:
        top_event_rows.append(
            top_event_row_template.format(
                name=html.escape(
                    str(
                        item.get("display_name")
                        or _event_display_name(
                            str(item.get("name") or ""),
                            item.get("sk_id"),
                            item.get("node_id"),
                            str(item.get("device") or ""),
                        )
                    )
                ),
                device=html.escape(_display_optional(item.get("device"), "未知")),
                pid=html.escape(_display_optional(item.get("pid"), "无")),
                tid=html.escape(_display_optional(item.get("tid"), "无")),
                duration=html.escape(str(item.get("duration", 0))),
                sk=html.escape(_display_optional(item.get("sk_id"), "无")),
                node=html.escape(_display_optional(item.get("node_id"), "无")),
            )
        )
    nav_markup = render_view_nav(
        [
            ("导航页面", portal_href, False),
            ("Scope View", "scope-graph.html", False),
            ("TaskQue View", "task-queue-graph.html", False),
            ("Analysis View", "performance-report.html", True),
            ("DFX View", "hang-crash-report.html", False),
        ],
        kicker="",
    )
    header_markup = render_report_header(
        icon="A",
        kicker="",
        title="Analysis View",
        note_html=html.escape("先看顶部焦点，再进入事件矩阵。"),
        nav_html=nav_markup,
    )
    overview_markup = _render_overview_section(
        title="当前摘要",
        note_html=_join_html_fragments(
            "这里先回答当前时间分析关注什么、完整度如何，以及下一步应该先看哪里。",
            _view_state_note_html(
                analysis_view_state,
                include_hints=True,
                include_fallbacks=True,
                include_summary=True,
            ),
            _analysis_trace_notice_html(
                node_trace_summary, run_dir, output_path, analysis_resource_state
            ),
        ),
        items=[
            (
                "Focus",
                _performance_focus_label(
                    str(performance_summary.get("focus", "unknown"))
                ),
            ),
            (
                "Completeness",
                _diagnostic_completeness_label(
                    str(performance_summary.get("diagnostic_completeness", "unknown"))
                ),
            ),
            ("Tracing", _analysis_trace_status_label(node_trace_summary)),
            ("State", _view_state_label(analysis_view_state)),
            (
                "Next Step",
                str(performance_summary.get("recommended_next_step", ""))
                or "先从事件矩阵进入。",
            ),
        ],
    )
    intro_markup = render_report_intro_section(
        title="当前摘要",
        note_html="这里先固定 Analysis 入口的最小上下文，再进入事件矩阵和热点事件。",
        summary_html=render_report_summary(
            [
                (
                    "Focus",
                    _performance_focus_label(
                        str(performance_summary.get("focus", "unknown"))
                    ),
                ),
                (
                    "Completeness",
                    _diagnostic_completeness_label(
                        str(
                            performance_summary.get(
                                "diagnostic_completeness", "unknown"
                            )
                        )
                    ),
                ),
                ("Tracing", _analysis_trace_status_label(node_trace_summary)),
            ],
            compact=True,
        ),
    )
    top_event_table_markup = _wrap_table_markup(
        top_event_table_template.format(
            "".join(top_event_rows)
            or "<tr><td colspan='7'>当前没有可展示的时间热点事件。</td></tr>"
        ),
        panel_id="top-events",
        title="热点事件",
        subtitle_html=(
            "这里保留当前最值得先看的热点事件，作为进一步跳转和交叉确认入口。"
            if analysis_resource_state.get("timing", {}).get("level") == "available"
            else (
                _analysis_resource_context_html(
                    analysis_resource_state,
                    include_tracing=False,
                    include_timing=True,
                    include_kernel_meta=False,
                    include_path_hint=True,
                )
                or "当前热点事件会受 time-event 资源状态直接影响。"
            )
        ),
        min_width=920,
        collapsed=False,
        search_placeholder="查找热点事件、设备、skId、nodeId",
    )
    top_event_more_markup = (
        "<details class='fold subblock'><summary>更多热点事件 ({})</summary><div>{}</div></details>".format(
            len(hidden_top_events),
            _wrap_table_markup(
                top_event_more_table_template.format(
                    "".join(
                        top_event_more_row_template.format(
                            html.escape(
                                str(
                                    item.get("display_name")
                                    or _event_display_name(
                                        str(item.get("name") or ""),
                                        item.get("sk_id"),
                                        item.get("node_id"),
                                        str(item.get("device") or ""),
                                    )
                                )
                            ),
                            html.escape(
                                str(
                                    item.get("raw_name")
                                    or item.get("name")
                                    or "未命名事件"
                                )
                            ),
                            html.escape(_display_optional(item.get("device"), "未知")),
                            html.escape(_display_optional(item.get("pid"), "无")),
                            html.escape(_display_optional(item.get("tid"), "无")),
                            html.escape(str(item.get("duration", 0))),
                            html.escape(_display_optional(item.get("sk_id"), "无")),
                            html.escape(_display_optional(item.get("node_id"), "无")),
                        )
                        for item in hidden_top_events
                    )
                ),
                panel_id="top-events-more",
                title="更多热点事件",
                subtitle="这里保留超出首屏的热点事件，按需展开查看。",
                min_width=1120,
                collapsed=True,
                search_placeholder="查找更多热点事件",
            ),
        )
        if hidden_top_events
        else ""
    )
    matrix_table_markup = _wrap_table_markup(
        event_matrix_table_template.format(
            "".join(matrix_rows)
            or "<tr><td colspan='9'>当前没有可稳定组织的事件主视角。</td></tr>"
        ),
        panel_id="event-matrix",
        title="事件矩阵",
        subtitle_html=(
            "先从事件分组定位最值得先看的时间证据，再按关联 Scope、Node 和 tracing 往下深挖。"
            if analysis_resource_state.get("timing", {}).get("level") == "available"
            else (
                _analysis_resource_context_html(
                    analysis_resource_state,
                    include_tracing=False,
                    include_timing=True,
                    include_kernel_meta=True,
                    include_path_hint=True,
                )
                or "当前事件矩阵会受 time-event 和结构资源状态直接影响。"
            )
        ),
        min_width=1180,
        collapsed=False,
        prominent=True,
        search_placeholder="查找事件分组、设备、skId、Scope、Node",
    )
    matrix_more_markup = (
        "<details class='fold subblock'><summary>更多矩阵行 ({})</summary><div>{}</div></details>".format(
            len(hidden_rows),
            _wrap_table_markup(
                event_matrix_table_template.format(
                    "".join(
                        event_matrix_row_template.format(
                            update_link=html.escape(
                                f"scope-graph.html?scopeId={item['linked_update_scope']}"
                                if item.get("linked_update_scope") is not None
                                else "scope-graph.html"
                            ),
                            group=html.escape(str(item.get("group_label"))),
                            sk=html.escape(str(item.get("sk_id"))),
                            devices=html.escape(
                                ",".join(item.get("devices", [])) or "none"
                            ),
                            event_count=html.escape(str(item.get("event_count", 0))),
                            total_dur=html.escape(str(item.get("total_duration", 0))),
                            matched_scopes=html.escape(
                                _comma_join_ids(item.get("matched_scope_ids", []))
                                or "none"
                            ),
                            matched_nodes=html.escape(
                                _comma_join_ids(item.get("matched_event_node_ids", []))
                                or "none"
                            ),
                            judgment=html.escape(
                                _performance_judgment_label(
                                    _performance_judgment(item, event_stats)
                                )
                            ),
                            trace_links=_analysis_trace_action_markup(
                                item,
                                run_dir,
                                output_path,
                                node_trace_summary,
                                analysis_resource_state,
                            ),
                        )
                        for item in hidden_rows
                    )
                ),
                panel_id="event-matrix-more",
                title="更多事件矩阵条目",
                subtitle="这里保留超出首屏的矩阵行，按需展开查看。",
                min_width=1180,
                collapsed=True,
                search_placeholder="查找更多矩阵行",
            ),
        )
        if hidden_rows
        else ""
    )
    html_text = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Analysis View</title>
  <style>{styles}</style>
</head>
<body>
  <div class="page-shell">
    {header_markup}
    {intro_markup}
    {overview_markup}
    {matrix_table}
    {matrix_more}
    {top_event_table}
    {top_event_more}
  </div>
<script>
{table_js}
document.querySelectorAll('[data-standard-table]').forEach(initStandardTablePanel);
</script>
</body>
</html>
""".format(
        styles=_shared_report_styles(),
        header_markup=header_markup,
        intro_markup=intro_markup,
        overview_markup=overview_markup,
        matrix_table=matrix_table_markup,
        matrix_more=matrix_more_markup,
        top_event_table=top_event_table_markup,
        top_event_more=top_event_more_markup,
        table_js=COMMON_STANDARD_TABLE_JS,
    )
    output_path.write_text(html_text, encoding="utf-8")


def _render_run_portal(
    run_summary: dict[str, Any],
    validation: dict[str, Any],
    report_links: dict[str, str],
) -> str:
    hang_crash_summary = run_summary.get("hang_crash_summary", {})
    capability_mode = str(run_summary.get("capability_mode") or "summary-only")
    multi_model_index = (
        run_summary.get("multi_model_index", {})
        if isinstance(run_summary.get("multi_model_index"), dict)
        else {}
    )

    def _build_multi_model_rows(models: list[dict[str, Any]]) -> str:
        rows = []
        row_template = (
            "<tr><td>{process}</td><td>{model}</td><td>{ri}</td>"
            "<td>{scope}</td><td>{func}</td><td>{scope_view}</td>"
            "<td>{task_view}</td><td>{analysis_view}</td><td>{dfx_view}</td></tr>"
        )
        for item in sorted(
            models,
            key=lambda current: (
                str(current.get("process_label", "")),
                str(current.get("model_ir_label", "")),
            ),
        ):
            paths = item.get("report_paths", {})
            processing_error = str(item.get("processing_error") or "").strip()
            view_states = (
                item.get("view_states", {})
                if isinstance(item.get("view_states"), dict)
                else {}
            )

            def _view_link(
                key: str,
                label: str,
                current_paths: dict[str, str] = paths,
                current_processing_error: str = processing_error,
                current_view_states: dict[str, Any] = view_states,
            ) -> str:
                href = current_paths.get(key)
                if current_processing_error and not href:
                    return "<span class='hint' title='{reason}'>生成失败</span>".format(
                        reason=html.escape(current_processing_error),
                    )
                view_state = (
                    current_view_states.get(key, {})
                    if isinstance(current_view_states.get(key), dict)
                    else {}
                )
                state_level = str(view_state.get("level") or "available")
                state_title = (
                    f"{label} 当前{_view_state_label(view_state)}"
                    if state_level != "available"
                    else ""
                )
                if href:
                    link = f"<a href='{html.escape(href)}'>{html.escape(label)}</a>"
                    if current_processing_error:
                        return (
                            f"{link} <span class='hint' title='{html.escape(current_processing_error)}'>"
                            f"（partial）</span>"
                        )
                    if state_level != "available":
                        return (
                            f"{link} <span class='hint' title='{html.escape(state_title)}'>"
                            f"（{html.escape(_view_state_label(view_state))}）</span>"
                        )
                    return link
                if key in {"performance_report", "hang_crash_report"}:
                    return "<span class='hint' title='{summary}'>{label}</span>".format(
                        summary=html.escape(state_title or "当前模型未生成该视图"),
                        label=html.escape(
                            _view_state_label(view_state)
                            if state_level != "available"
                            else "当前模型未生成该视图"
                        ),
                    )
                if state_level != "available":
                    return "<span class='hint' title='{summary}'>{label}</span>".format(
                        summary=html.escape(state_title),
                        label=html.escape(_view_state_label(view_state)),
                    )
                return "-"

            rows.append(
                row_template.format(
                    process=html.escape(str(item.get("process_label", "-"))),
                    model=html.escape(str(item.get("model_ir_label", "-"))),
                    ri=html.escape(str(item.get("model_ri") or "unknown")),
                    scope=html.escape(str(item.get("scope_count", 0))),
                    func=html.escape(str(item.get("function_count", 0))),
                    scope_view=_view_link("scope_graph", "Scope"),
                    task_view=_view_link("task_queue_graph", "TaskQue"),
                    analysis_view=_view_link("performance_report", "Analysis"),
                    dfx_view=_view_link("hang_crash_report", "DFX"),
                )
            )
        return (
            "".join(rows) or "<tr><td colspan='9'>当前没有可展示的 modelRI。</td></tr>"
        )

    def _dfx_rank(value: str) -> int:
        return {"abnormal": 3, "warning": 2, "clean": 1}.get(str(value), 0)

    def _model_fusion_state(item: dict[str, Any]) -> str:
        has_task_or_update = bool(
            item.get("task_queue_available") or item.get("has_update")
        )
        has_fused_function = int(item.get("function_count", 0) or 0) > 0
        if has_task_or_update or has_fused_function:
            return "fused"
        if item.get("scope_graph_available"):
            return "scope_only"
        return "unfused"

    def _model_target_label(item: dict[str, Any]) -> str:
        process_label = str(item.get("process_label") or "-")
        model_ri = str(item.get("model_ri") or "unknown")
        return f"{process_label} / {model_ri}"

    multi_models = (
        multi_model_index.get("models", [])
        if isinstance(multi_model_index.get("models"), list)
        else []
    )
    multi_model_count = int(multi_model_index.get("model_count", 0) or 0) or max(
        len(multi_models), 1
    )
    has_multi_model = multi_model_count > 1

    dfx_abnormal_models = [
        item
        for item in multi_models
        if str(item.get("dfx_status", "clean")) == "abnormal"
    ]
    dfx_non_abnormal_models = [
        item
        for item in multi_models
        if str(item.get("dfx_status", "clean")) != "abnormal"
    ]
    updated_models = [item for item in multi_models if item.get("has_update")]
    no_update_models = [item for item in multi_models if not item.get("has_update")]
    scope_only_count = 0
    for item in multi_models:
        if _model_fusion_state(item) == "scope_only":
            scope_only_count += 1

    header_nav_items = [("运行总览", "#run-summary", False)]
    if has_multi_model and dfx_abnormal_models:
        header_nav_items.append(
            ("存在 DFX 问题的 modelRI", "#dfx-problem-models", False)
        )
        header_nav_items.append(("其余 modelRI", "#non-dfx-problem-models", False))
    elif has_multi_model:
        header_nav_items.append(("有更新的 modelRI", "#updated-models", False))
        header_nav_items.append(("无更新的 modelRI", "#no-update-models", False))

    header_markup = render_report_header(
        icon="R",
        kicker="",
        title="Run Portal",
        note_html="当前 run 总入口。先看顶部结论，再进入具体 modelRI。",
        nav_html=render_view_nav(header_nav_items, kicker=""),
    )

    summary_title = "运行总览"
    summary_note = "当前 run 先展示整体状态，再按下方分组进入各个 modelRI 的具体视图。"
    summary_items: list[tuple[str, str]]
    if has_multi_model and dfx_abnormal_models:
        highest_status = max(
            (str(item.get("dfx_status", "clean")) for item in dfx_abnormal_models),
            key=_dfx_rank,
            default="abnormal",
        )
        highest_status_models = [
            item
            for item in dfx_abnormal_models
            if str(item.get("dfx_status", "clean")) == highest_status
        ]
        preferred_item = sorted(
            highest_status_models,
            key=lambda current: (
                str(current.get("process_label", "")),
                str(current.get("model_ri", "")),
            ),
        )[0]
        summary_title = "DFX 总览"
        summary_note = "当前 run 存在 DFX 问题，应优先进入对应 DFX View；只有当 DFX 没问题时，再回头看融合情况。"
        summary_items = [
            ("最高优先级", _analysis_status_label(highest_status)),
            ("受影响的 modelRI", str(len(dfx_abnormal_models))),
            ("优先查看", _model_target_label(preferred_item)),
            ("modelRI 总数", str(multi_model_count)),
        ]
    elif has_multi_model:
        summary_title = "融合总览"
        summary_note = (
            "当前 run 没有 abnormal 级别的 DFX 问题，因此导航页优先汇报 "
            "update/融合情况；若某个 modelRI 没有 update，但仍保留 Scope，"
            "就继续优先查看 Scope。"
        )
        summary_items = [
            ("modelRI 总数", str(multi_model_count)),
            ("有更新", str(len(updated_models))),
            ("仅 Scope", str(scope_only_count)),
            ("无更新", str(len(no_update_models))),
        ]
    else:
        single_status = str(hang_crash_summary.get("analysis_status", "clean"))
        if single_status == "abnormal":
            summary_title = "DFX 总览"
            summary_note = "当前 run 存在 DFX 问题，应优先进入 DFX View 查看具体诊断。"
            summary_items = [
                ("当前结论", _analysis_status_label(single_status)),
                ("modelRI", str(run_summary.get("model_ri") or "unknown")),
                (
                    "诊断完整度",
                    _diagnostic_completeness_label(
                        str(
                            run_summary.get(
                                "overall_diagnostic_completeness", "unknown"
                            )
                        )
                    ),
                ),
                ("能力模式", _capability_mode_label(capability_mode)),
            ]
        else:
            has_task_content = bool(run_summary.get("task_indices"))
            has_scope_content = bool(run_summary.get("scope_ids"))
            fusion_label = (
                "已融合"
                if has_task_content
                else ("仅 Scope" if has_scope_content else "未融合")
            )
            summary_title = "融合总览"
            summary_note = (
                "当前 run 没有明显 DFX 问题，因此这里优先汇报当前 modelRI 的融合情况。"
            )
            summary_items = [
                ("modelRI", str(run_summary.get("model_ri") or "unknown")),
                ("融合状态", fusion_label),
                (
                    "诊断完整度",
                    _diagnostic_completeness_label(
                        str(
                            run_summary.get(
                                "overall_diagnostic_completeness", "unknown"
                            )
                        )
                    ),
                ),
                ("能力模式", _capability_mode_label(capability_mode)),
            ]

    overview_markup = _render_overview_section(
        title=summary_title,
        note_html=_join_html_fragments(
            summary_note,
            "这里承接当前 run 的摘要结果，再按下方分组进入具体 modelRI。",
        ),
        items=summary_items,
        section_id="run-summary",
    )
    intro_markup = render_report_intro_section(
        title="当前摘要",
        note_html="统一入口先给出当前 run 的最小导航上下文，再进入下方各个 modelRI 分组。",
        summary_html=render_report_summary(
            [
                ("Run", str(run_summary.get("run_id") or "unknown")),
                ("Model Count", str(multi_model_count)),
                ("入口", summary_title),
            ],
            compact=True,
        ),
    )

    multi_model_sections = ""
    table_template = (
        "<table><thead><tr><th>Process</th><th>模型目录</th><th>modelRI</th>"
        "<th>Scope Count</th><th>Function Count</th><th>Scope View</th>"
        "<th>TaskQue View</th><th>Analysis View</th><th>DFX View</th></tr></thead>"
        "<tbody>{rows}</tbody></table>"
    )
    if has_multi_model:
        if dfx_abnormal_models:
            multi_model_sections = (
                _wrap_table_markup(
                    table_template.format(
                        rows=_build_multi_model_rows(dfx_abnormal_models)
                    ),
                    panel_id="portal-dfx-problems",
                    title="存在 DFX 问题的 modelRI",
                    subtitle="这些 modelRI 的 DFX 结论达到 abnormal，应优先进入对应 DFX View；处理完 DFX 之后，再回头检查 update 和融合情况。",
                    min_width=1180,
                    collapsed=False,
                    prominent=True,
                    search_placeholder="查找 process、模型目录、modelRI",
                )
                + "\n"
                + _wrap_table_markup(
                    table_template.format(
                        rows=_build_multi_model_rows(dfx_non_abnormal_models)
                    ),
                    panel_id="portal-dfx-clean",
                    title="其余 modelRI",
                    subtitle="这些 modelRI 当前没有 abnormal 级别的 DFX 问题；若还需要继续确认，请再看对应的 Scope、TaskQue 和 Analysis。",
                    min_width=1180,
                    collapsed=True,
                    search_placeholder="查找其余 modelRI",
                )
            )
        else:
            multi_model_sections = (
                _wrap_table_markup(
                    table_template.format(rows=_build_multi_model_rows(updated_models)),
                    panel_id="portal-updated",
                    title="有更新的 modelRI",
                    subtitle="这些 modelRI 已经形成 update 相关事实，可以直接继续进入 Scope、TaskQue、Analysis 和 DFX 视图。",
                    min_width=1180,
                    collapsed=False,
                    prominent=True,
                    search_placeholder="查找有更新的 modelRI",
                )
                + "\n"
                + _wrap_table_markup(
                    table_template.format(
                        rows=_build_multi_model_rows(no_update_models)
                    ),
                    panel_id="portal-no-update",
                    title="无更新的 modelRI",
                    subtitle="这些 modelRI 当前还没有形成 update 相关事实；若仍保留 Scope 结构，就依然可以继续查看 Scope。",
                    min_width=1180,
                    collapsed=True,
                    search_placeholder="查找无更新的 modelRI",
                )
            )
    elif multi_models:
        multi_model_sections = _wrap_table_markup(
            table_template.format(rows=_build_multi_model_rows(multi_models)),
            panel_id="portal-single-model",
            title="当前 modelRI",
            subtitle="这里展示当前 run 中唯一的 modelRI 入口，直接从这里进入对应视图。",
            min_width=1180,
            collapsed=False,
            prominent=True,
            search_placeholder="查找当前 modelRI",
        )

    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Run Portal</title>
  <style>{styles}</style>
</head>
<body>
  <div class="page-shell">
    {header_markup}
    {intro_markup}
    {overview_markup}
    {multi_model_sections}
  </div>
<script>
{table_js}
document.querySelectorAll('[data-standard-table]').forEach(initStandardTablePanel);
</script>
</body>
</html>
""".format(
        styles=_shared_report_styles(),
        header_markup=header_markup,
        intro_markup=intro_markup,
        overview_markup=overview_markup,
        multi_model_sections=multi_model_sections,
        table_js=COMMON_STANDARD_TABLE_JS,
    )


def _render_ai_hints(
    run_dir: Path,
    phase_assertions: dict[str, bool],
    validation: dict[str, Any],
    event_stats: dict[str, Any],
) -> str:
    focus = []
    if not phase_assertions["built_tasks_finish"]:
        focus.append("重点先排查 build tasks 前后的日志区段。")
    if event_stats["event_file_count"] == 0:
        focus.append("当前没有 time event 文件，性能判断先停留在结构与队列层。")
    if not focus:
        focus.append("当前基础证据链完整度较高，可优先做跨报告联动和更细的结构归因。")

    lines = [
        "# AI-Ready Hints",
        "",
        "- status: `not-configured`",
        "- 当前仓已固定 AI 增强层位置，但尚未接入真实大模型后端。",
        f"- run: `{run_dir.name}`",
        "",
        "## Base Result Snapshot",
        "",
        f"- overall validation: `{validation['overall_status']}`",
        f"- entered aclskOptimize: `{phase_assertions['entered_aclsk_optimize']}`",
        f"- built tasks finish: `{phase_assertions['built_tasks_finish']}`",
        f"- has time event: `{phase_assertions['has_time_event']}`",
        "",
        "## Next Focus",
        "",
    ]
    for item in focus:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Future AI Input Contract",
            "",
            "- 优先消费 `run-portal.html`、`scope-library.json` 和 `graph-library.json`。",
            "- 再结合 Analysis / DFX / Scope / TaskQue 四个 View 做更细提示。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Compatibility CLI; all report generation now uses the unified pipeline."""
    from model_analysis import build_parser

    arguments = sys.argv[1:]
    mode = "full"
    if "--mode" in arguments:
        position = arguments.index("--mode")
        if position + 1 >= len(arguments):
            raise SystemExit("--mode requires a value")
        mode = arguments[position + 1]
        arguments = arguments[:position] + arguments[position + 2 :]
    command = {
        "full": "analyze",
        "hang": "diagnose-hang-crash",
    }.get(mode)
    if command is None:
        raise SystemExit("Unknown mode: " + mode)
    args = build_parser().parse_args([command, *arguments])
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
