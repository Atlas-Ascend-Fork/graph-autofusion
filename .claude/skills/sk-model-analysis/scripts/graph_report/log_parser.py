"""Independent LOG adapter. 更新后执行图 coverage is deliberately partial."""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from pathlib import Path

import sk_library_extractor as legacy


def _issue(
    code,
    message,
    *,
    domain="parsing",
    source=None,
    details=None,
    obj=None,
    severity="warning",
):
    return {
        "id": code,
        "code": code,
        "domain": domain,
        "severity": severity,
        "message": message,
        "source": source or {},
        "details": details or {},
        "object": obj or {},
        "impact": "现有日志证据限制了数据覆盖或问题定位。",
        "next_step": "检查引用的日志证据，必要时补采缺失数据。",
    }


def _source(path, line=None):
    result = {"file": str(path or "")}
    if isinstance(line, int):
        result["line"] = line
    return result


def _type(value):
    name = str(value or "UNKNOWN").upper().removeprefix("EVENT_")
    return {"VALUE_WRITE": "MEMORY_WRITE", "VALUE_WAIT": "MEMORY_WAIT"}.get(name, name)


def _updated_label(node, node_type):
    """Use the observed post-update type for non-kernel task labels.

    Log node records are collected before graph update, so their label can be
    the original KERNEL/NOTIFY/WAIT name even when the update record changes
    the runtime task into a VALUE_WRITE/VALUE_WAIT.  Keeping that label makes
    the Graph appear unchanged and can mislead the type filters.
    """
    if node_type in {"MEMORY_WRITE", "MEMORY_WAIT", "INVALID"}:
        return node_type
    return node.get("label")


def _node(row, path, scope_id=None):
    return {
        "id": str(row["node_id"]),
        "stream": str(row.get("stream_id", row.get("stream_idx_in_graph", "?"))),
        "stream_index": row.get("stream_idx_in_graph", -1),
        "order": row.get("node_idx_in_stream", -1),
        "type": _type(row.get("node_type")),
        "label": row.get("func_name") or _type(row.get("node_type")),
        "scope_id": scope_id,
        "is_sk": False,
        "params": {k: v for k, v in row.items() if v is not None},
        "source": _source(path, row.get("line")),
    }


def _precedes(left, right):
    return (
        left["stream"] == right["stream"]
        and isinstance(left.get("order"), int)
        and isinstance(right.get("order"), int)
        and 0 <= left["order"] < right["order"]
    )


def _not_backward(left, right):
    return left["stream"] != right["stream"] or not (
        isinstance(left.get("order"), int)
        and isinstance(right.get("order"), int)
        and left["order"] >= right["order"]
    )


def _edges(nodes, issues, stage, *, stream_order=True):
    streams, events = defaultdict(list), defaultdict(list)
    memory = defaultdict(list)
    edges = []
    for node in nodes:
        streams[node["stream"]].append(node)
        event = node["params"].get("event_id")
        if event is not None:
            events[str(event)].append(node)
        address = node["params"].get("addr")
        if address is not None and node["type"] in {"MEMORY_WRITE", "MEMORY_WAIT"}:
            memory[str(address)].append(node)
    if stream_order:
        for group in streams.values():
            group.sort(key=lambda n: (n["order"], n["id"]))
            for left, right in itertools.pairwise(group):
                if left["order"] >= 0 and right["order"] > left["order"]:
                    edges.append(
                        {
                            "source": left["id"],
                            "target": right["id"],
                            "kind": "stream",
                            "evidence": "inferred",
                        }
                    )
    for event, group in events.items():
        notify = [n for n in group if n["type"] == "NOTIFY"]
        wait = [n for n in group if n["type"] == "WAIT"]
        reset = [n for n in group if n["type"] == "RESET"]
        if (
            len(notify) == len(wait) == 1
            and all(
                _precedes(wait[0], reset_node) or _precedes(reset_node, notify[0])
                for reset_node in reset
            )
            and _not_backward(notify[0], wait[0])
        ):
            edges.append(
                {
                    "source": notify[0]["id"],
                    "target": wait[0]["id"],
                    "kind": "event",
                    "evidence": "inferred",
                    "event": event,
                }
            )
        elif wait:
            issues.append(
                _issue(
                    f"log_event_ambiguous_{stage}_{event}",
                    f"Event {event} 的通知与重置轮次不明确；未添加确定依赖。",
                    obj={"stage": stage},
                    details={"nodes": [n["id"] for n in group]},
                )
            )
    for address, group in memory.items():
        writes = [n for n in group if n["type"] == "MEMORY_WRITE"]
        waits = [n for n in group if n["type"] == "MEMORY_WAIT"]
        if (
            len(writes) == len(waits) == 1
            and waits[0]["params"].get("flag") in (1, "1", "0x1")
            and _not_backward(writes[0], waits[0])
            and writes[0]["params"].get("value") is not None
            and str(writes[0]["params"]["value"])
            == str(waits[0]["params"].get("value"))
        ):
            edges.append(
                {
                    "source": writes[0]["id"],
                    "target": waits[0]["id"],
                    "kind": "memory",
                    "evidence": "inferred",
                    "address": address,
                    "value": writes[0]["params"]["value"],
                }
            )
        elif waits:
            issues.append(
                _issue(
                    f"log_memory_ambiguous_{stage}_{address}",
                    f"内存信号 {address} 的生产者或数值轮次不明确；未添加确定依赖。",
                    obj={"stage": stage},
                    details={"nodes": [n["id"] for n in group]},
                )
            )
    from .synchronization import recover_pairs

    recovered = recover_pairs(nodes, edges)
    edges.extend(recovered)
    accepted = {(e["kind"], str(e["target"])) for e in edges}
    by_id = {str(n["id"]): n for n in nodes}

    def resolved(diagnostic):
        code = diagnostic["code"]
        if diagnostic.get("object", {}).get("stage") != stage or not code.startswith(
            ("log_event_ambiguous_", "log_memory_ambiguous_")
        ):
            return False
        kind = "event" if code.startswith("log_event_") else "memory"
        waits = [
            str(n)
            for n in diagnostic.get("details", {}).get("nodes", [])
            if by_id.get(str(n), {}).get("type") in {"WAIT", "MEMORY_WAIT"}
        ]
        return bool(waits) and all((kind, n) in accepted for n in waits)

    issues[:] = [i for i in issues if not resolved(i)]
    return edges


def _function(detail, marker):
    return str(detail or "").split(marker, 1)[-1].split(", scope id:", 1)[0].strip()


def _runtime_target(report, signal):
    """Bind real producer evidence; never treat an op index as a graph node ID."""
    explicit = {
        key: str(signal[key])
        for key in ("node_id", "scope_id")
        if signal.get(key) is not None
    }
    if explicit:
        return {"stage": "before", **explicit}
    library = report.get("scope_library", {})
    dfx = report.get("dfx_library", {})
    events = dfx.get("exception_registry", {}).get("events", [])
    kind = signal.get("kind")
    if kind == "function_name":
        events = [event for event in events if event.get("line") == signal.get("line")]
    elif kind not in {"exception_core_symbol", "counter_localization", "op_trace_hint"}:
        return {}
    # Multiple matching headers or multiple exceptional scopes are ambiguous even
    # if only one of the candidate scopes happens to be present in the node dump.
    if not events or any(len(event.get("candidates", [])) != 1 for event in events):
        return {}
    candidates = {str(event["candidates"][0].get("scope_id")) for event in events}
    known_scopes = {
        str(scope["scope_id"]): scope for scope in library.get("scopes", [])
    }
    if len(candidates) != 1:
        return {}
    scope_id = next(iter(candidates))
    if scope_id not in known_scopes:
        return {}
    target = {"stage": "before", "scope_id": scope_id}
    if kind == "counter_localization":
        counter = dfx.get("counter_registry", {}).get("summary", {})
        function_name = counter.get("dominant_active_function_name")
        if counter.get("op_trace_enabled") and function_name:
            members = {str(n) for n in known_scopes[scope_id].get("node_ids", [])}
            nodes = (
                report.get("graph_library", {}).get("node_library", {}).get("nodes", [])
            )
            matches = [
                n
                for n in nodes
                if str(n["node_id"]) in members and n.get("func_name") == function_name
            ]
            if len(matches) == 1:
                target["node_id"] = str(matches[0]["node_id"])
    return target


def _runtime(report):
    from diagnose_run import _build_dfx_hang_summary

    summary = _build_dfx_hang_summary(report)
    status = summary["analysis_status"]
    # Absence of exceptions in build logs does not prove a successful runtime.
    status = status if status in {"abnormal", "warning"} else "insufficient"
    conclusion = (
        summary["conclusion"]
        if status in {"abnormal", "warning"}
        else "未发现明确的运行故障；现有日志不足以证明执行过程完整。"
    )
    issues = []
    signals = list(summary.get("top_actionable_signals", []))
    seen = {(s.get("file"), s.get("line"), s.get("text")) for s in signals}
    warnings = summary.get("warning_signals", [])
    warning_keys = {(s.get("file"), s.get("line"), s.get("text")) for s in warnings}
    for signal in warnings:
        key = (signal.get("file"), signal.get("line"), signal.get("text"))
        if key not in seen:
            signals.append(signal)
            seen.add(key)
    for index, signal in enumerate(signals):
        text = signal.get("text", "")
        is_warning = (
            bool(re.search(r"\[(?:WARN|WARNING)\]", text, re.IGNORECASE))
            or (signal.get("file"), signal.get("line"), text) in warning_keys
        )
        code = f"log_runtime_{index}_{signal.get('kind', 'signal')}"
        target = _runtime_target(report, signal)
        details = dict(signal)
        if (
            is_warning
            and "AddEventAssociate" in text
            and "cannot find KERNEL node" in text
        ):
            code = "log_event_association_skipped"
            position = "wait 后" if "after wait" in text else "notify 前"
            address = re.search(r"\bEvent\s+(0x[\da-fA-F]+)", text)
            event = address[1] if address else ""
            text = f"Event {event} 的 {position}未找到 Kernel，日志跳过了该方向的融合关联。"
            details["event_id"] = event
            details["position"] = position
            endpoint = "wait" if position == "wait 后" else "notify"
            matches = re.findall(
                rf"{endpoint} node\s*\[nodeId:(\d+)", signal.get("text", "")
            )
            if len(matches) == 1:
                target = {"stage": "before", "node_id": matches[0]}
            elif event:
                node_rows = (
                    report.get("graph_library", {})
                    .get("node_library", {})
                    .get("nodes", [])
                )
                endpoints = [
                    n
                    for n in node_rows
                    if str(n.get("event_id", "")).lower() == event.lower()
                    and _type(n.get("node_type"))
                    == ("WAIT" if endpoint == "wait" else "NOTIFY")
                ]
                if len(endpoints) == 1:
                    target = {
                        "stage": "before",
                        "node_id": str(endpoints[0]["node_id"]),
                    }
        item = _issue(
            code,
            text,
            domain="runtime",
            source=_source(signal.get("file"), signal.get("line")),
            details=details,
            obj=target,
            severity="warning" if is_warning or status != "abnormal" else "error",
        )
        item["id"] = f"log_runtime_{index}_{code}"
        issues.append(item)
    if status == "insufficient":
        issues.append(_issue("log_runtime_coverage", conclusion, domain="runtime"))
    return {
        "status": status,
        "conclusion": conclusion,
        "issues": issues,
        "evidence": {"summary": summary, **report.get("dfx_library", {})},
    }


def _adapt(report):
    from .options import read_log_options

    graph_options = read_log_options(
        Path(report.get("model_dir", "")) / "super_kernel.log",
        report.get("model_instance_range"),
    )
    library = report["scope_library"]
    graph = report["graph_library"]
    node_library = graph["node_library"]
    source = node_library.get("path")
    issues = []
    scopes = [
        {
            "id": str(s["scope_id"]),
            "node_ids": [str(n) for n in s.get("node_ids", [])],
            "label": str(s.get("scope_names") or s["scope_id"]),
            "source": _source(library.get("path")),
        }
        for s in library.get("scopes", [])
    ]
    membership = {n: s["id"] for s in scopes for n in s["node_ids"]}
    nodes = [
        _node(n, source, membership.get(str(n["node_id"])))
        for n in node_library.get("nodes", [])
    ]
    from .log_functions import enrich_log_functions

    enrich_log_functions(
        nodes,
        Path(report.get("model_dir", "")) / "super_kernel.log",
        report.get("model_instance_range"),
    )
    by_id = {n["id"]: n for n in nodes}
    fused_library = library.get("fused_library", {})
    fusions = [
        {
            "scope_id": str(f["scope_id"]),
            "node_ids": [str(n) for n in f.get("node_ids", [])],
            "label": _function(f.get("detail"), "SK Function:"),
            "source": _source(fused_library.get("path")),
        }
        for f in fused_library.get("functions", [])
        if f.get("scope_id") is not None
    ]
    functions = defaultdict(set)
    for fusion in fusions:
        functions[fusion["label"]].add(fusion["scope_id"])
    queues = []
    device = library.get("device_task_library", {})
    for index, section in enumerate(device.get("sections", [])):
        name = _function(section.get("detail"), "Dumping device args for function:")
        candidates = functions.get(name, set())
        # An exact function-to-fused-header match is stronger than positional fallback.
        scope_id = next(iter(candidates)) if len(candidates) == 1 else None
        if scope_id is None:
            issues.append(
                _issue(
                    f"log_queue_unbound_{index}",
                    "设备队列无法唯一关联到有明确记录的融合 Scope。",
                    source=_source(device.get("path")),
                    details={"function": name},
                )
            )
        for kind, tasks in section.get("queues", {}).items():
            normalized = []
            for task in tasks:
                debug = task.get("debug_options")
                normalized.append(
                    {
                        "index": task.get("task_id", len(normalized)),
                        "node_index": task.get("task_index"),
                        "type": str(task.get("task_type") or "UNKNOWN"),
                        "num_blocks": task.get("block_count"),
                        "entries": task.get("entries", []),
                        "args": task.get("args"),
                        "debug_options": int(str(debug), 0)
                        if debug is not None
                        else None,
                        "source": _source(device.get("path")),
                        "params": task,
                    }
                )
            if tasks or kind in section.get("queue_details", {}):
                queues.append(
                    {
                        "scope_id": scope_id,
                        "id": f"{index}:{kind}",
                        "kind": kind,
                        "tasks": normalized,
                        "source": _source(device.get("path")),
                    }
                )
    updates = {}
    update_source = graph["node_update_registry"].get("path")
    for row in graph["node_update_registry"].get("rows", []):
        updates[str(row["node_id"])] = row
    after = []
    sk_hosts = {}
    function_fusions = defaultdict(list)
    for fusion in fusions:
        if fusion["label"]:
            function_fusions[fusion["label"]].append(fusion)
    function_hosts = defaultdict(list)
    for node_id, row in updates.items():
        if row.get("type") == "KERNEL" and row.get("sk_function"):
            function_hosts[row["sk_function"]].append(node_id)
    explicit_scopes = set()
    for function, host_ids in function_hosts.items():
        matches = function_fusions.get(function, [])
        if len(host_ids) == 1 and len(matches) == 1:
            sk_hosts[host_ids[0]] = matches[0]
            explicit_scopes.add(matches[0]["scope_id"])
    scope_members = {s["id"]: s["node_ids"] for s in scopes}
    for fusion in fusions:
        if fusion["scope_id"] in explicit_scopes:
            continue
        candidates = [
            n
            for n in scope_members.get(fusion["scope_id"], fusion["node_ids"])
            if updates.get(n, {}).get("type") == "KERNEL"
            and not updates[n].get("sk_function")
        ]
        if len(candidates) == 1 and candidates[0] in fusion["node_ids"]:
            sk_hosts[candidates[0]] = fusion
    for node_id, row in updates.items():
        node = dict(by_id.get(node_id) or _node({"node_id": node_id}, update_source))
        node.update(
            type=_type(row.get("type")),
            params={**row, "original_order": node.get("order")},
            order=node.get("order") if row.get("type") == "INVALID" else None,
            is_sk=False if row.get("type") == "INVALID" else None,
            source=_source(update_source, row.get("line")),
        )
        node["label"] = _updated_label(node, node["type"])
        if node_id in sk_hosts:
            node.update(
                is_sk=True,
                scope_id=sk_hosts[node_id]["scope_id"],
                label=sk_hosts[node_id]["label"],
            )
        after.append(node)
    issues.append(
        _issue(
            "log_after_partial",
            "Log 更新记录只描述已观测的变化；未变化节点和最终连边未知。",
            source=_source(update_source),
            obj={"stage": "after"},
        )
    )
    if report.get("collection_error"):
        issues.append(
            _issue(
                "log_instance_partition", report["collection_error"], severity="error"
            )
        )
    missing = sorted(set(membership) - set(by_id))
    if missing:
        issues.append(
            _issue(
                "log_scope_members_missing",
                "Scope 成员中存在节点日志未记录的节点。",
                details={"node_ids": missing},
            )
        )
    runtime = _runtime(report)
    from .user_scopes import log_user_scopes

    user_scopes = log_user_scopes(
        nodes,
        Path(report.get("model_dir", "")) / "super_kernel.log",
        library,
        report.get("model_instance_range"),
    )
    fusion_analysis = report.get("fusion_analysis")
    fusion_status = (
        str(fusion_analysis.get("status") or "unavailable")
        if isinstance(fusion_analysis, dict)
        else "unavailable"
    )
    if isinstance(fusion_analysis, dict) and fusion_status in {
        "partial",
        "unavailable",
    }:
        issues.append(
            _issue(
                "log_fusion_analysis_coverage",
                "融合断图分析证据不完整；主图和已有节点数据仍可用。",
                source=_source(
                    fusion_analysis.get("sources", {})
                    .get("sk_scope_split.log", {})
                    .get("path")
                ),
                details={
                    "status": fusion_status,
                    "summary": fusion_analysis.get("summary", {}),
                    "sources": fusion_analysis.get("sources", {}),
                },
                obj={"stage": "fusion_analysis"},
            )
        )
    files = sorted(
        {
            str(p)
            for p in [
                str(Path(report.get("model_dir", "")) / "super_kernel.log"),
                str(Path(report.get("model_dir", "")) / "sk_fusion_fail_reasons.log"),
                source,
                library.get("path"),
                fused_library.get("path"),
                device.get("path"),
                update_source,
            ]
            if p and Path(p).is_file() and Path(p).suffix == ".log"
        }
    )
    before_edges = _edges(nodes, issues, "before")
    stats = node_library.get("stats", {})
    membership_verified = (
        stats.get("model_instance_partition_verified") is True
        and stats.get("membership_partition_basis") is not None
        and not stats.get("scope_node_missing_count")
    )
    before_partial = (
        not membership_verified
        or missing
        or report.get("collection_error")
        or any(i.get("object", {}).get("stage") == "before" for i in issues)
    )
    if nodes and not membership_verified:
        issues.append(
            _issue(
                "log_before_coverage",
                "原始节点的成员覆盖情况尚未得到独立验证。",
                obj={"stage": "before"},
            )
        )
    return {
        "parser": "log",
        "options": graph_options,
        "instance_id": report["model_instance_id"],
        "before": {
            "status": ("partial" if before_partial else "complete")
            if nodes
            else "unavailable",
            "nodes": nodes,
            "edges": before_edges,
            "source": str(source or ""),
        },
        "after": {
            "status": "partial" if updates else "unavailable",
            "nodes": after,
            "edges": _edges(after, issues, "after", stream_order=False),
            "source": str(update_source or ""),
        },
        "scopes": scopes,
        "user_scopes": user_scopes,
        "fusions": fusions,
        # Fusion-boundary evidence is kept separate from graph topology so
        # future DFX/Graph consumers can opt into it without changing layout.
        "fusion_analysis": fusion_analysis,
        "queues": queues,
        "runtime": runtime,
        "diagnostics": issues,
        "capabilities": {
            "scopes": "partial" if missing else "complete" if scopes else "unavailable",
            "fusions": "partial"
            if report.get("collection_error")
            else "complete"
            if fusions
            else "unavailable",
            "queues": "partial"
            if any(q["scope_id"] is None for q in queues)
            else "complete"
            if queues
            else "unavailable",
            "fusion_analysis": fusion_status,
            "runtime": "partial",
        },
        "source_files": files,
    }


def parse_models(model_dir: Path) -> list[dict]:
    """Parse log evidence only, including each independently partitioned instance."""
    with legacy.use_parse_context(
        legacy.ParseContext(Path(model_dir), validate_json_shadow=False)
    ):
        return [
            _adapt(report)
            for report in legacy.collect_update_model_instance_reports(model_dir)
        ]
