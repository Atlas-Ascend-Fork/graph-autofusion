"""Build audience-specific DFX data without changing graph topology.

The parser result contains useful facts and low-level diagnostics in the same
model.  This module creates a small public projection for the normal report and
keeps the raw evidence behind an explicit debug-report option.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from typing import Any

from .fusion_annotations import build_fusion_annotations

_FUSION_SUMMARY_FIELDS = (
    "final_pass",
    "final_region_count",
    "successful_sk_count",
    "boundary_count",
    "operator_triggered_boundary_count",
    "control_boundary_count",
    "unclassified_boundary_count",
    "failure_record_count",
    "unique_failure_node_count",
)
_FUSION_COVERAGE_FIELDS = (
    "final_boundary_records",
    "boundary_trigger_node_matched_count",
    "boundary_trigger_node_unmatched_count",
    "historical_pass_count",
)
_PUBLIC_DIAGNOSTIC_CODES = {
    "log_event_association_skipped",
    "ambiguous_json_instance",
    "archive_failed",
    "ai_not_configured",
    "capability_unavailable",
    "count_mismatch",
    "graph_option_mismatch",
    "graph_options_fallback",
    "invalid_graph_options",
    "json_parser_failed",
    "log_after_partial",
    "log_before_coverage",
    "log_fusion_analysis_coverage",
    "log_instance_partition",
    "log_runtime_coverage",
    "log_scope_members_missing",
    "model_identity_mismatch",
    "no_model_assets",
    "parser_mismatch",
    "partial_coverage",
    "single_side_objects",
    "source_fallback",
    "log_parser_failed",
    "scope_alignment_conflict",
}
# These are already represented by the DFX status cards.  Repeating each
# per-model coverage record as a separate user finding makes a large report
# look noisy without adding a new action.
_STATUS_DIAGNOSTIC_CODES = {"log_runtime_coverage", "log_fusion_analysis_coverage"}
_NON_ACTIONABLE_PUBLIC_CODES = {
    "partial_coverage",
    "graph_options_fallback",
    "log_after_partial",
    "log_before_coverage",
    "log_scope_members_missing",
    "single_side_objects",
}
_REASON_LABELS = {
    "DEADLOCK_DETECTED": "为避免死锁而拆分融合",
    "UNFUSIBLE_NODE": "算子导致融合断开",
    "SYNCALL_OP_DROP": "同步信号导致融合断开",
    "OP_UNSUPPORT": "算子不支持当前融合条件",
    "BINDMAP_INIT_EMPTY": "算子不支持静态编译，或静态编译失败",
    "RESOURCE_INSUFFICIENT": "资源不足，未能完成融合",
    "SCOPE_FUSE_PART": "SK 仅完成部分融合",
    "MISSING_FUSION_RECORD": "融合记录缺失",
    "UNKNOWN": "其他已记录原因",
}

_MAX_PUBLIC_TARGETS = 32
_MAX_PUBLIC_IDS = 16


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_status(value: Any, default: str = "unavailable") -> str:
    text = str(value or default).strip().lower()
    return text if text else default


def _safe_source(source: Any, *, include_debug: bool = False) -> dict[str, Any]:
    source = source if isinstance(source, dict) else {}
    result = {
        "status": _safe_status(source.get("status")),
        "record_count": _number(source.get("record_count")) or 0,
        "line_count": _number(source.get("line_count")) or 0,
        "path": None,
        "line_range": None,
    }
    if include_debug:
        samples = source.get("sample_lines")
        result.update(
            {
                "error_count": _number(source.get("error_count")) or 0,
                "format_error_count": _number(source.get("format_error_count")) or 0,
                "unknown_line_count": _number(source.get("unknown_line_count")) or 0,
                "sample_count": len(samples) if isinstance(samples, list) else 0,
            }
        )
    return result


def _safe_target(target: Any) -> dict[str, Any] | None:
    if not isinstance(target, dict):
        return None
    result = {
        key: target[key]
        for key in ("kind", "stage", "relation", "scope_id", "node_id")
        if target.get(key) is not None
    }
    return result or None


def _safe_targets(targets: Any) -> list[dict[str, Any]]:
    if not isinstance(targets, list):
        return []
    result = []
    seen = set()
    for target in targets:
        safe = _safe_target(target)
        if not safe:
            continue
        key = tuple(sorted(safe.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(safe)
        if len(result) >= _MAX_PUBLIC_TARGETS:
            break
    return result


def _safe_ids(values: Any, *, limit: int = _MAX_PUBLIC_IDS) -> list[Any]:
    if not isinstance(values, (list, tuple, set)):
        return []
    result = []
    for value in values:
        if value is None or value in result:
            continue
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _safe_implementation(implementation: Any) -> dict[str, Any]:
    implementation = implementation if isinstance(implementation, dict) else {}
    return {
        key: implementation.get(key)
        for key in ("backend", "mode", "confidence")
        if implementation.get(key) is not None
    }


def _safe_operator(operator: Any, *, include_raw: bool = False) -> dict[str, Any]:
    operator = operator if isinstance(operator, dict) else {}
    return {
        "normalized_name": operator.get("normalized_name") or None,
        "raw_name": operator.get("raw_name") if include_raw else None,
        "implementation": _safe_implementation(operator.get("implementation")),
        "implementation_variants": copy.deepcopy(
            operator.get("implementation_variants", [])
        )
        if include_raw
        else [],
    }


def _reason_summary(code: Any, reason: Any, detail: Any = None) -> str:
    code_text = str(code or "").strip()
    if code_text in _REASON_LABELS:
        return _REASON_LABELS[code_text]
    reason_text = str(reason or "").strip()
    if reason_text in _REASON_LABELS:
        return _REASON_LABELS[reason_text]
    if (
        reason_text
        and reason_text.replace("_", "").isalnum()
        and reason_text.upper() == reason_text
    ):
        return _REASON_LABELS["UNKNOWN"]
    detail_text = str(detail or "").strip()
    return reason_text or detail_text or _REASON_LABELS["UNKNOWN"]


def _safe_operator_impact(item: Any, *, include_raw: bool = False) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    direct = (
        item.get("direct_breaks") if isinstance(item.get("direct_breaks"), dict) else {}
    )
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    observed = (
        item.get("observed_effect")
        if isinstance(item.get("observed_effect"), dict)
        else {}
    )
    reasons = {
        str(code): _number(count) or 0
        for code, count in (direct.get("by_reason") or {}).items()
    }
    return {
        "operator": _safe_operator(item.get("operator"), include_raw=include_raw),
        "impact_status": str(item.get("impact_status") or "unknown"),
        "source_kind": str(item.get("source_kind") or "unknown"),
        "direct_breaks": {
            "total": _number(direct.get("total")) or 0,
            "by_reason": reasons,
            "boundary_ids": _safe_ids(direct.get("boundary_ids"))
            if include_raw
            else [],
            "trigger_node_ids": _safe_ids(direct.get("trigger_node_ids")),
        },
        "evidence": {
            "failure_count": _number(evidence.get("failure_count")) or 0,
        },
        "observed_effect": {
            "affected_region_count": _number(observed.get("affected_region_count"))
            or 0,
            "affected_scope_count": _number(observed.get("affected_region_count")) or 0,
            "affected_scope_ids": _safe_ids(observed.get("affected_scope_ids")),
            "adjacent_scope_count": _number(observed.get("adjacent_scope_count")) or 0,
            "observed_unfused_kernel_count": _number(
                observed.get("observed_unfused_kernel_count")
            )
            or 0,
        },
        "targets": _safe_targets(item.get("targets")),
    }


def _safe_boundary(item: Any, *, include_raw: bool = False) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    trigger = item.get("trigger") if isinstance(item.get("trigger"), dict) else {}
    operator = (
        trigger.get("operator") if isinstance(trigger.get("operator"), dict) else {}
    )
    result = {
        "id": item.get("id") if include_raw else None,
        "scope_id": item.get("scope_id"),
        "parent_scope_id": item.get("parent_scope_id"),
        "break_reason": _reason_summary(
            item.get("break_reason"), item.get("break_reason")
        ),
        "break_reason_code": item.get("break_reason") if include_raw else None,
        "scope_fusion_status": str(item.get("scope_fusion_status") or "UNKNOWN"),
        "previous_scope_id": item.get("previous_scope_id"),
        "following_scope_id": item.get("following_scope_id"),
        "adjacent_scope_ids": list(item.get("adjacent_scope_ids") or []),
        "trigger": {
            "node_id": trigger.get("node_id"),
            "stream_idx_in_graph": trigger.get("stream_idx_in_graph"),
            "operator": _safe_operator(operator, include_raw=include_raw),
            "source_kind": str(trigger.get("source_kind") or "unknown"),
        },
        "sync_all_node_ids": list(item.get("sync_all_node_ids") or []),
        "observed_unfused_kernel_node_ids": list(
            item.get("observed_unfused_kernel_node_ids") or []
        ),
        "targets": _safe_targets(item.get("targets")),
    }
    if include_raw:
        result.update(
            {
                "detail": item.get("detail"),
                "break_reason_detail": item.get("break_reason_detail"),
                "same_region_failure_node_ids": list(
                    item.get("same_region_failure_node_ids") or []
                ),
                "successful_adjacent_scope_ids": list(
                    item.get("successful_adjacent_scope_ids") or []
                ),
                "source": copy.deepcopy(item.get("source") or {}),
            }
        )
    return result


def _safe_failure(item: Any, *, include_raw: bool = False) -> dict[str, Any]:
    item = item if isinstance(item, dict) else {}
    operator = item.get("operator") if isinstance(item.get("operator"), dict) else {}
    result = {
        "node_id": item.get("node_id"),
        "stream_id": item.get("stream_id"),
        "stream_idx_in_graph": item.get("stream_idx_in_graph"),
        "node_idx_in_stream": item.get("node_idx_in_stream"),
        "operator": _safe_operator(operator, include_raw=include_raw),
        "reason": _reason_summary(
            item.get("reason_code"), item.get("reason"), item.get("detail")
        ),
        "reason_code": item.get("reason_code") if include_raw else None,
        "detail": item.get("detail") if include_raw else None,
        "targets": _safe_targets(item.get("targets")),
    }
    if include_raw:
        result.update(
            {
                "raw_name": item.get("raw_name"),
                "source": copy.deepcopy(item.get("source") or {}),
                "line": item.get("line"),
            }
        )
    return result


def _safe_operator_outcomes(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result = {
        "status": _safe_status(value.get("status")),
        "expected_node_count": _number(value.get("expected_node_count")),
        "actual_node_count": _number(value.get("actual_node_count")),
        "unfused_node_count": _number(value.get("unfused_node_count")),
    }
    excluded = value.get("excluded_node_counts")
    if isinstance(excluded, dict):
        result["excluded_node_counts"] = {
            str(key): _number(count) or 0
            for key, count in excluded.items()
            if count is not None
        }
    excluded_failures = value.get("excluded_failure_counts")
    if isinstance(excluded_failures, dict):
        result["excluded_failure_counts"] = {
            str(key): _number(count) or 0
            for key, count in excluded_failures.items()
            if count is not None
        }
    return result


def _deadlock_break_detail(boundary: dict, fused_scopes: dict[str, list[str]]) -> dict:
    """Explain a recorded fusion-time split, without inferring runtime failure."""
    trigger = (
        boundary.get("trigger") if isinstance(boundary.get("trigger"), dict) else {}
    )
    node_id = str(trigger["node_id"]) if trigger.get("node_id") is not None else None
    match = re.fullmatch(
        r"Deadlock at node (\d+), split at Wait node (\d+)",
        str(boundary.get("detail") or "").strip(),
    )
    split_id = match[2] if match and match[1] == node_id else None
    scopes = fused_scopes.get(node_id, [])
    explanation = "融合时的死锁检查触发拆分"
    if node_id is not None:
        explanation += f"，触发节点为 #{node_id}"
    if split_id is not None:
        explanation += f"，实际在 Wait #{split_id} 前拆分"
    explanation += "。"
    if scopes:
        explanation += f"触发节点随后已融合到 SK {'、'.join(scopes)}。"
    targets = []
    if split_id is not None:
        targets.append({"kind": "node", "stage": "before", "node_id": split_id})
    if node_id is not None:
        targets.append({"kind": "node", "stage": "before", "node_id": node_id})
    targets.extend(
        {"kind": "scope", "stage": "after", "scope_id": sid} for sid in scopes
    )
    return {
        "trigger_node_id": node_id,
        "split_node_id": split_id,
        "fused_scope_ids": list(scopes),
        "explanation": explanation,
        "targets": targets,
    }


def _safe_fusion(
    fusion: Any, *, include_raw: bool = False, model: dict | None = None
) -> dict[str, Any]:
    fusion = fusion if isinstance(fusion, dict) else {}
    summary = fusion.get("summary") if isinstance(fusion.get("summary"), dict) else {}
    coverage = (
        fusion.get("coverage") if isinstance(fusion.get("coverage"), dict) else {}
    )
    sources = fusion.get("sources") if isinstance(fusion.get("sources"), dict) else {}
    filtered_failures = fusion.get("unfused_node_failures")
    if not isinstance(filtered_failures, list):
        filtered_failures = fusion.get("node_failures") or []
    result = {
        "schema_version": fusion.get("schema_version", "1.0"),
        "status": _safe_status(fusion.get("status")),
        "summary": {
            key: summary.get(key) for key in _FUSION_SUMMARY_FIELDS if key in summary
        },
        "coverage": {
            key: coverage.get(key) for key in _FUSION_COVERAGE_FIELDS if key in coverage
        },
        "sources": {
            str(name): _safe_source(source, include_debug=include_raw)
            for name, source in sources.items()
        },
        "operator_impacts": [
            _safe_operator_impact(item, include_raw=include_raw)
            for item in (fusion.get("operator_impacts") or [])
            if isinstance(item, dict)
        ],
        "boundaries": [
            _safe_boundary(item, include_raw=include_raw)
            for item in (fusion.get("boundaries") or [])
            if isinstance(item, dict)
        ],
        "node_failures": [
            _safe_failure(item, include_raw=include_raw)
            for item in filtered_failures
            if isinstance(item, dict)
        ],
    }
    # Only explicit fusion transformations establish where a trigger ended up.
    model = model or {}
    after_nodes = {
        str(n.get("id")): n for n in (model.get("after") or {}).get("nodes", [])
    }
    fusion_targets: dict[str, set[str]] = {}
    for row in model.get("transformations") or []:
        if row.get("kind") != "fusion":
            continue
        for node_id in row.get("before_ids") or []:
            fusion_targets.setdefault(str(node_id), set()).update(
                str(target)
                for target in row.get("after_ids") or []
                if str(target) in after_nodes
            )
    fused_scopes = {}
    for node_id, targets in fusion_targets.items():
        if len(targets) != 1:
            continue
        target = after_nodes[next(iter(targets))]
        if target.get("is_sk") and target.get("scope_id") is not None:
            fused_scopes[node_id] = [str(target["scope_id"])]
    boundaries = {
        str(b.get("id")): b
        for b in fusion.get("boundaries") or []
        if isinstance(b, dict) and b.get("id") is not None
    }
    original_impacts = [
        item for item in fusion.get("operator_impacts") or [] if isinstance(item, dict)
    ]
    public_impacts = result["operator_impacts"]
    if len(original_impacts) != len(public_impacts):
        raise ValueError("Fusion operator impact projection lost row alignment")
    for original, public in zip(original_impacts, public_impacts):
        details = [
            _deadlock_break_detail(boundaries[str(bid)], fused_scopes)
            for bid in (original.get("direct_breaks") or {}).get("boundary_ids") or []
            if str(bid) in boundaries
            and boundaries[str(bid)].get("break_reason") == "DEADLOCK_DETECTED"
        ]
        public["break_details"] = details[:8]
        public["break_detail_count"] = len(details)
    outcomes = _safe_operator_outcomes(fusion.get("operator_outcomes"))
    if outcomes is not None:
        result["operator_outcomes"] = outcomes
    return result


def _public_source(source: Any) -> dict[str, Any]:
    if isinstance(source, str):
        token = source.strip()
        normalized = _safe_status(token)
        if normalized in {
            "unavailable",
            "partial",
            "available",
            "ready",
            "complete",
            "ok",
        }:
            status = (
                "available"
                if normalized in {"available", "ready", "complete", "ok"}
                else normalized
            )
            return {
                "status": status,
                "parser": None,
                "available": status != "unavailable",
                "path": None,
            }
        parser = token or None
        return {
            "status": "available" if parser else "unavailable",
            "parser": parser,
            "available": bool(parser),
            "path": None,
        }
    source = source if isinstance(source, dict) else {}
    status = _safe_status(source.get("status"))
    if status in {"ready", "complete", "ok"}:
        status = "available"
    return {
        "status": status,
        "parser": source.get("parser"),
        "available": status != "unavailable",
        "path": None,
    }


def _public_sources(
    model: dict[str, Any], fusion: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    result = {
        str(name): _public_source(source)
        for name, source in (model.get("sources") or {}).items()
    }
    # The model-level map names the selected parser, while the fusion map has
    # the actual log coverage.  Keep both so the UI can show partial files.
    for name, source in (fusion.get("sources") or {}).items():
        result.setdefault(str(name), _public_source(source))
    return result


def _public_issue(
    issue: Any, *, count: int = 1, objects: list[dict] | None = None
) -> dict[str, Any] | None:
    """Turn one parser/runtime record into a user-facing conclusion.

    Parser diagnostics are intentionally not copied verbatim.  The default
    report should answer three questions: is there an error or warning, what
    does it affect, and where can the user inspect or fix it.
    """
    if not isinstance(issue, dict):
        return None
    code = str(issue.get("code") or "diagnostic")
    domain = str(issue.get("domain") or "analysis")
    severity = str(issue.get("severity") or "info").strip().lower()
    runtime_signal = domain == "runtime" and severity in {"error", "abnormal"}
    runtime_warning = (
        domain == "runtime"
        and severity == "warning"
        and isinstance(issue.get("details"), dict)
        and issue["details"].get("kind") == "runtime_warning_or_error"
    )
    # Queue-to-SK matching is an optional enrichment.  It is useful in the
    # debug profile, but a mapping miss is not a user warning when the graph
    # and internal task data are still present.
    if code.startswith("log_queue_unbound_"):
        return None
    if code in _NON_ACTIONABLE_PUBLIC_CODES and not runtime_signal:
        return None
    if (
        code not in _PUBLIC_DIAGNOSTIC_CODES
        and not runtime_signal
        and not runtime_warning
    ):
        return None
    state = "attention" if severity in {"warning", "error", "abnormal"} else "confirmed"
    public_severity = (
        "error"
        if severity in {"error", "abnormal"}
        else "warning"
        if state == "attention"
        else "info"
    )
    object_refs = list(objects or [])
    obj = issue.get("object")
    if isinstance(obj, dict):
        safe = _safe_target(obj)
        if safe and safe not in object_refs:
            object_refs.append(safe)
    title = str(issue.get("title") or issue.get("message") or code)
    summary = str(issue.get("message") or code)
    impact = str(issue.get("impact") or "")
    next_step = str(issue.get("next_step") or "")
    location = ""
    count_label = ""
    details = issue.get("details") if isinstance(issue.get("details"), dict) else {}
    if code == "log_event_association_skipped":
        title = "部分 Event 的融合关联被跳过"
        summary = str(
            issue.get("message") or "日志未找到相邻 Kernel，跳过了对应的融合关联。"
        )
        if count > 1:
            summary = f"日志记录 {count} 处 Event 前后未找到相邻 Kernel，跳过了对应方向的融合关联。"
        impact = "对应方向的相邻 Kernel 未建立融合关联。"
        location = (
            "对应 Event 节点" if object_refs else "super_kernel.log · AddEventAssociate"
        )
        next_step = (
            "核对 Event 前后的任务是否存在相邻 Kernel；没有相邻 Kernel 时无需处理。"
        )
    elif runtime_warning:
        title = "日志 Warning"
        summary = str(issue.get("message") or "日志记录了 Warning。")
        source = issue.get("source") or {}
        location = f"{str(source.get('file') or '日志').rsplit('/', 1)[-1]}:{source.get('line', '?')}"
        impact = "影响范围以原始 Warning 的具体说明为准。"
        next_step = "根据上述 Warning 内容核对对应任务或配置。"
    elif runtime_signal:
        target = issue.get("object") if isinstance(issue.get("object"), dict) else {}
        target_label = ""
        if target.get("scope_id") is not None:
            target_label = f" SK {target['scope_id']}"
        elif target.get("node_id") is not None:
            target_label = f" 节点 #{target['node_id']}"
        kind = str(details.get("kind") or "").strip()
        if not kind and code.startswith("log_runtime_"):
            parts = code.split("_", 3)
            kind = parts[3] if len(parts) == 4 else ""
        kind_label = {
            "function_name": "异常函数定位",
            "exception_core_symbol": "异常核心符号",
        }.get(kind, "运行异常信号")
        title = f"{kind_label}{target_label}"
        summary = f"已记录{kind_label}{target_label}，可在图中定位（Graph）。"
        impact = f"影响{target_label.strip() or '当前 Model'}的运行状态判断。"
        location = (
            f"Graph：{target_label.strip()}" if target_label else "Graph：当前 Model"
        )
        next_step = "打开对应节点详情，核对异常函数或核心符号。"
    elif code in {"source_fallback", "model_identity_mismatch"}:
        title = "当前图按 Log 结果展示"
        summary = "JSON 未匹配当前 Model，主图和融合结果按 Log 生成。"
        impact = "无法用 JSON 对当前 Model 做交叉校验。"
        next_step = "补齐与当前 Model 对应的 JSON 后重新生成报告。"
        location = "当前 Model 的来源选择"
    elif code == "parser_mismatch":
        title = "JSON 与 Log 存在差异"
        summary = "共同字段存在差异，Graph 按 JSON 主结果展示，Log 只作补充。"
        impact = "差异字段的交叉校验结果需要复核。"
        next_step = "核对两侧文件是否来自同一次运行，再检查差异字段。"
        location = "DFX 的来源校验"
    elif code == "invalid_graph_options":
        title = "图配置存在异常"
        summary = "部分图配置未能按原始值解析，已保留可识别的配置。"
        impact = "只影响图配置说明，不改变已生成的节点和连线。"
        next_step = "打开图配置查看异常字段；确认采集文件格式后重新生成。"
        location = "图配置"
    elif code == "scope_alignment_conflict":
        title = "部分 SK 的布局未完全对齐"
        summary = "部分成员在 Stream 上交错，已保留真实顺序并分段显示。"
        impact = "只影响 SK 外框的连续性，不影响节点关系。"
        next_step = "打开对应 SK 查看分段成员和同步连线。"
        location = "Graph：对应 SK"
    elif code in {"json_parser_failed", "log_parser_failed"}:
        source_name = "JSON" if code == "json_parser_failed" else "Log"
        title = f"{source_name} 解析失败"
        summary = f"{source_name} 未能完整解析，Graph 继续使用可用来源生成。"
        impact = f"只影响 {source_name} 提供的节点或交叉校验结果。"
        next_step = f"检查输入 {source_name} 文件格式和版本后重新生成报告。"
        location = f"{source_name} 输入文件"
    elif code == "no_model_assets":
        title = "没有找到 Model 数据"
        summary = "输入目录中没有可用于生成 Graph 的 Model 资产。"
        impact = "当前 Model 无法展示。"
        next_step = "确认输入目录包含对应的 JSON 或 Log 文件。"
        location = "输入目录"
    return {
        "id": str(issue.get("id") or code),
        "code": code,
        "category": domain,
        "severity": public_severity,
        "state": state,
        "title": title,
        "summary": summary,
        "impact": impact,
        "location": location,
        "next_step": next_step,
        "count": count,
        "count_label": count_label,
        "show_count": True,
        "objects": object_refs,
    }


def _issue_inputs(
    model: dict[str, Any], global_diagnostics: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    values = []
    values.extend(
        item for item in model.get("diagnostics", []) if isinstance(item, dict)
    )
    runtime = model.get("runtime") if isinstance(model.get("runtime"), dict) else {}
    values.extend(item for item in runtime.get("issues", []) if isinstance(item, dict))
    model_key = model.get("key")
    values.extend(
        item
        for item in global_diagnostics
        if isinstance(item, dict)
        and (
            item.get("model_key") is None
            or str(item.get("model_key")) == str(model_key)
        )
    )
    unique = []
    seen = set()
    for item in values:
        key = str(item.get("id") or "")
        if not key:
            key = repr(
                sorted((str(k), repr(v)) for k, v in item.items() if k != "details")
            )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _public_findings(
    model: dict[str, Any], global_diagnostics: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    inputs = _issue_inputs(model, global_diagnostics)
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in inputs:
        code = str(item.get("code") or "diagnostic")
        if code in _STATUS_DIAGNOSTIC_CODES:
            continue
        # A fallback usually emits both an identity mismatch and a source
        # selection record.  Present one actionable warning instead of two
        # parser-internal cards.
        group_key = (
            "source_selection"
            if code in {"source_fallback", "model_identity_mismatch"}
            else code
        )
        groups.setdefault(group_key, []).append(item)
    findings = []
    for group_key, items in groups.items():
        representative = items[0]
        code = group_key
        if group_key == "source_selection":
            representative = {**representative, "code": "source_fallback"}
            code = "source_fallback"
        refs = []
        for item in items[:8]:
            obj = item.get("object")
            if isinstance(obj, dict):
                safe = _safe_target(obj)
                if safe and safe not in refs:
                    refs.append(safe)
            details = item.get("details")
            scopes = details.get("scopes") if isinstance(details, dict) else None
            if isinstance(scopes, dict):
                for scope_id in list(scopes)[:8]:
                    target = {
                        "kind": "scope",
                        "stage": "before",
                        "scope_id": str(scope_id),
                    }
                    if target not in refs:
                        refs.append(target)
        finding = _public_issue(representative, count=len(items), objects=refs)
        # Confirmed facts remain in the debug profile and structured output.
        if finding and finding.get("severity") in {"error", "warning"}:
            findings.append(finding)
    findings.sort(
        key=lambda item: (
            item["state"] != "attention",
            item.get("severity") != "error",
            -int(item.get("count") or 0),
            item["title"],
        )
    )
    return findings


def _fusion_reason_counts(fusion: Any) -> list[dict[str, Any]]:
    """Aggregate known fusion failure reasons without exposing raw enums."""
    fusion = fusion if isinstance(fusion, dict) else {}
    counts: dict[str, int] = {}
    failures = fusion.get("node_failures") or []
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        label = _reason_summary(
            failure.get("reason_code"), failure.get("reason"), failure.get("detail")
        )
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        for boundary in fusion.get("boundaries") or []:
            if not isinstance(boundary, dict):
                continue
            label = _reason_summary(
                boundary.get("break_reason"),
                boundary.get("break_reason"),
                boundary.get("detail"),
            )
            counts[label] = counts.get(label, 0) + 1
    return [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _fusion_findings(
    model: dict[str, Any], safe_fusion: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return only a blocking fusion-data error.

    Fusion progress and break causes belong to the fusion result card.  They do
    not create a second warning card because the card already exposes the
    affected operator counts and Graph locations.  A missing fusion envelope is
    different: without it, the report cannot state whether fusion happened.
    """
    if safe_fusion.get("status") == "unavailable":
        return [
            {
                "id": "fusion_unavailable",
                "code": "fusion_unavailable",
                "category": "fusion",
                "severity": "error",
                "state": "attention",
                "title": "融合结果不可用",
                "summary": "没有找到可解析的融合日志，无法判断哪些节点融合成功。",
                "impact": "DFX 只能展示执行图，不能给出融合结论。",
                "location": "输入目录中的融合日志",
                "next_step": "确认 sk_scope_split.log 和 sk_fusion_fail_reasons.log 存在且可读后重新生成。",
                "count": 1,
                "count_label": "项",
                "objects": [],
            }
        ]
    return []


def _graph_status(model: dict[str, Any]) -> str:
    states = []
    for key in ("before", "after"):
        value = model.get(key)
        if isinstance(value, dict):
            states.append(_safe_status(value.get("status")))
    if any(value in {"complete", "partial", "ready"} for value in states):
        return "available" if "partial" not in states else "partial"
    return "unavailable"


def _status_summary(
    status: dict[str, str], fusion: dict[str, Any], runtime: dict[str, Any]
) -> str:
    graph_text = {
        "available": "执行图已生成",
        "partial": "执行图已生成",
        "unavailable": "执行图不可用",
    }.get(status["graph"], "执行图状态未知")
    runtime_text = {
        "abnormal": "发现明确运行异常",
        "clean": "未发现明确运行异常",
    }.get(status["runtime"], str(runtime.get("conclusion") or "运行状态未知"))
    fusion_text = {
        "complete": "融合断图结果已解析",
        "partial": "已解析可确认的融合断图结果",
        "unavailable": "未采集融合断图记录",
    }.get(fusion.get("status"), "融合证据状态未知")
    clauses = [graph_text]
    if status["runtime"] in {"abnormal", "clean"}:
        clauses.append(runtime_text)
    if fusion.get("status") in {"complete", "partial", "unavailable"}:
        clauses.append(fusion_text)
    return "；".join(clauses) + "。"


def _status_findings(
    status: dict[str, str], fusion: dict[str, Any], runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    findings = []
    if status["graph"] == "unavailable":
        findings.append(
            {
                "id": "graph_unavailable",
                "code": "graph_unavailable",
                "category": "graph",
                "severity": "error",
                "state": "attention",
                "title": "执行图不可用",
                "summary": "没有找到可用于展示的执行图节点。",
                "impact": "Graph 视图无法还原当前 Model。",
                "location": "输入目录中的 Graph 资产",
                "next_step": "检查输入目录中的 JSON 或 Log 资产。",
                "count": 1,
                "count_label": "项",
                "objects": [],
            }
        )
    # A missing runtime dump is an internal coverage state, not a user action.
    # Concrete runtime signals are collected by _public_findings instead.
    return findings


def _build_conclusion(
    status: dict[str, str],
    fusion: dict[str, Any],
    findings: list[dict[str, Any]],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors = [item for item in findings if item.get("severity") == "error"]
    warnings = [item for item in findings if item.get("severity") == "warning"]
    if errors:
        runtime_errors = [item for item in errors if item.get("category") == "runtime"]
        lead = "发现明确运行异常" if runtime_errors else "发现错误"
        summary = f"{lead}：{errors[0].get('summary') or errors[0].get('title') or '请查看错误详情'}"
        return {
            "state": "error",
            "title": "发现错误",
            "summary": summary,
            "errors": len(errors),
            "warnings": len(warnings),
        }
    if warnings:
        lead = warnings[0]
        summary = (
            f"解析日志记录了 {len(warnings)} 项 Warning："
            f"{lead.get('headline') or lead.get('summary') or '请查看日志诊断'}"
        )
        return {
            "state": "warning",
            "title": "日志诊断",
            "summary": summary,
            "errors": 0,
            "warnings": len(warnings),
        }

    metrics = metrics if isinstance(metrics, dict) else {}
    actual = _number(metrics.get("fused_operator_count"))
    expected = _number(metrics.get("expected_operator_count"))
    sk_count = _number(metrics.get("fused_sk_count"))
    if actual is not None and expected is not None:
        remaining = max(int(expected) - int(actual), 0)
        sk_phrase = f"{int(sk_count):,} 个 SK" if sk_count is not None else "SK"
        if remaining:
            normal_summary = (
                f"未发现 ERROR 或 Warning；已将 {int(actual):,} / {int(expected):,} 个算子融合到 "
                f"{sk_phrase}；其余 {remaining:,} 个算子保留为原始节点。"
            )
            title = "解析情况"
        else:
            normal_summary = (
                f"未发现 ERROR 或 Warning；{int(actual):,} 个算子全部完成融合，"
                f"形成 {sk_phrase}，执行图可用。"
            )
            title = "解析情况"
    elif status.get("graph") != "unavailable":
        normal_summary = "未发现 ERROR 或 Warning；执行图和已解析结果可用。"
        title = "解析情况"
    else:
        normal_summary = "未发现 ERROR 或 Warning。"
        title = "解析情况"
    return {
        "state": "normal",
        "title": title,
        "summary": normal_summary,
        "errors": 0,
        "warnings": 0,
    }


def build_dfx(
    model: dict[str, Any],
    global_diagnostics: Iterable[dict[str, Any]] = (),
    *,
    include_debug: bool = False,
) -> dict[str, Any]:
    """Return a stable DFX view model for one report model."""
    model = model if isinstance(model, dict) else {}
    runtime = model.get("runtime") if isinstance(model.get("runtime"), dict) else {}
    fusion = _safe_fusion(
        model.get("fusion_analysis"), include_raw=include_debug, model=model
    )
    status = {
        "graph": _graph_status(model),
        "runtime": _safe_status(runtime.get("status"), "insufficient"),
        "fusion": fusion["status"],
        "analysis": "partial"
        if fusion["status"] in {"partial", "unavailable"}
        else "complete",
    }
    statistics = (
        model.get("statistics") if isinstance(model.get("statistics"), dict) else {}
    )
    summary = fusion["summary"]
    outcomes = (
        fusion.get("operator_outcomes")
        if isinstance(fusion.get("operator_outcomes"), dict)
        else {}
    )

    def outcome_or_stat(
        outcome_key: str, statistic_key: str, fallback: Any = None
    ) -> Any:
        value = outcomes.get(outcome_key)
        return value if value is not None else statistics.get(statistic_key, fallback)

    metrics = {
        "before_nodes": statistics.get("before_nodes"),
        "after_nodes": statistics.get("after_nodes"),
        "visible_after_nodes": statistics.get("visible_after_nodes"),
        "scope_count": statistics.get("scope_count"),
        "visible_scope_count": statistics.get("visible_scope_count"),
        "sk_nodes": statistics.get("sk_nodes"),
        "fused_nodes": statistics.get("fused_nodes"),
        "single_kernel_fusion_count": statistics.get("single_kernel_fusion_count"),
        "multi_kernel_fusion_count": statistics.get("multi_kernel_fusion_count"),
        "pure_scope_kernel_scope_count": statistics.get(
            "pure_scope_kernel_scope_count"
        ),
        # The annotated outcome is derived from the pre-update kernel universe
        # and therefore wins over legacy statistics built from final Scope
        # membership.  Keep the statistics fallback for older reports.
        "fused_operator_count": outcome_or_stat(
            "actual_node_count",
            "fused_operator_count",
            statistics.get("fused_kernel_nodes"),
        ),
        "expected_operator_count": outcome_or_stat(
            "expected_node_count", "expected_operator_count"
        ),
        "unfused_operator_count": outcome_or_stat(
            "unfused_node_count", "unfused_operator_count"
        ),
        "fused_sk_count": statistics.get("fused_sk_count", statistics.get("sk_nodes")),
        "final_region_count": summary.get("final_region_count"),
        "successful_sk_count": summary.get("successful_sk_count"),
        "boundary_count": summary.get("boundary_count"),
        "operator_triggered_boundary_count": summary.get(
            "operator_triggered_boundary_count"
        ),
        "control_boundary_count": summary.get("control_boundary_count"),
        "failure_record_count": summary.get("failure_record_count"),
    }
    findings = (
        _status_findings(status, fusion, runtime)
        + _fusion_findings(model, fusion)
        + _public_findings(model, global_diagnostics)
    )
    conclusion = _build_conclusion(status, fusion, findings, metrics)
    public = {
        "schema_version": "1.0",
        "profile": "debug" if include_debug else "user",
        "status": status,
        "conclusion": conclusion,
        "headline": f"{conclusion['title']}：{conclusion['summary']}",
        "metrics": metrics,
        "findings": findings,
        "fusion": fusion,
        "coverage": {
            "sources": _public_sources(model, fusion),
            "parser_comparison": {
                key: model.get("comparison", {}).get(key)
                for key in ("status", "difference_count", "single_side_count")
                if isinstance(model.get("comparison"), dict)
                and key in model["comparison"]
            },
        },
    }
    if include_debug:
        public["debug"] = {
            "diagnostics": copy.deepcopy(model.get("diagnostics") or []),
            "runtime": copy.deepcopy(runtime),
            "comparison": copy.deepcopy(model.get("comparison") or {}),
            "fusion_analysis": copy.deepcopy(model.get("fusion_analysis") or {}),
            "sources": copy.deepcopy(model.get("sources") or {}),
        }
    return public


def _public_top_level_diagnostic(item: Any) -> dict[str, Any] | None:
    result = _public_issue(item)
    if not result:
        return None
    return {
        "id": result["id"],
        "code": result["code"],
        "domain": result["category"],
        "severity": result.get(
            "severity", "warning" if result["state"] == "attention" else "info"
        ),
        "message": result["summary"],
        "impact": result["impact"],
        "next_step": result["next_step"],
        "object": result["objects"][0] if result["objects"] else {},
        "model_key": item.get("model_key") if isinstance(item, dict) else None,
    }


def project_report(
    report: dict[str, Any],
    *,
    include_debug: bool = False,
    preserve_raw: bool = False,
) -> dict[str, Any]:
    """Create the report projection used by the selected HTML profile.

    The input is never mutated.  Graph data is shared by shallow copy, while
    DFX evidence and diagnostics are copied or summarized as appropriate.
    ``preserve_raw`` is for the backward-compatible machine summary; it keeps
    the parser fields alongside the user DFX projection without putting them in
    the default HTML.
    """
    report = report if isinstance(report, dict) else {}
    projected = dict(report)
    global_diagnostics = [
        item for item in report.get("diagnostics", []) if isinstance(item, dict)
    ]
    projected["dfx_profile"] = "debug" if include_debug else "user"
    if not include_debug and not preserve_raw:
        public_diagnostics = []
        for item in global_diagnostics:
            safe = _public_top_level_diagnostic(item)
            if safe:
                public_diagnostics.append(safe)
        projected["diagnostics"] = public_diagnostics
    projected_models = []
    for original in report.get("models", []):
        if not isinstance(original, dict):
            continue
        model = dict(original)
        model["graph_fusion"] = build_fusion_annotations(original)
        model["dfx"] = build_dfx(
            original, global_diagnostics, include_debug=include_debug
        )
        if include_debug or preserve_raw:
            projected_models.append(model)
            continue
        for key in ("diagnostics", "comparison", "fusion_analysis"):
            model.pop(key, None)
        runtime = (
            original.get("runtime") if isinstance(original.get("runtime"), dict) else {}
        )
        model["runtime"] = {
            key: runtime.get(key)
            for key in ("status", "conclusion")
            if runtime.get(key) is not None
        }
        projected_models.append(model)
    projected["models"] = projected_models
    return projected
