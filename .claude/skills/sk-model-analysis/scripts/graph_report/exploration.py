"""Conservative, reversible exploration metadata over normalized report models.

Nothing here changes source nodes, edges, queues, or parser conclusions. Repeated
structures are ordered graph fingerprints, not an assertion of equal numerics.
"""

import re
from collections import Counter, defaultdict

from sk_library_extractor import SYNC_TYPE_MAP


def _operator(node):
    if node.get("type") not in ("KERNEL", "FUNC"):
        return ""
    label = str(node.get("label", ""))
    return re.sub(r"_[0-9a-f]{24,}(?:_.*)?$", "", label)


def _projection(model, sids, ids):
    selected = set(ids)
    mapped = {
        str(n)
        for row in model.get("transformations", [])
        if row.get("before_ids") and set(map(str, row["before_ids"])).issubset(selected)
        for n in row.get("after_ids", [])
    }
    return [
        str(n["id"])
        for n in model.get("after", {}).get("nodes", [])
        if str(n.get("scope_id")) in sids or str(n["id"]) in mapped
    ]


def _iteration_details(model, selected, period, incident):
    iterations, fingerprints = [], []
    after = {str(n["id"]): n for n in model.get("after", {}).get("nodes", [])}
    for offset in range(0, len(selected), period):
        scopes = selected[offset : offset + period]
        sids = [str(s["id"]) for s in scopes]
        ids = [str(n) for s in scopes for n in s["node_ids"]]
        positions = {nid: i for i, nid in enumerate(ids)}
        topology = Counter()
        for nid in ids:
            for edge in incident[nid]:
                a, b = str(edge["source"]), str(edge["target"])
                if a == nid and b in positions:
                    topology[(positions[a], positions[b], edge.get("kind", ""))] += 1
        fingerprints.append(topology)
        mapped = _projection(model, sids, ids)
        iterations.append(
            {
                "index": len(iterations),
                "scope_ids": sids,
                "node_ids": ids,
                "after_ids": mapped,
                "after_statistics": {
                    "node_count": len(mapped),
                    "type_counts": dict(
                        sorted(
                            Counter(
                                after[n].get("type", "UNKNOWN") for n in mapped
                            ).items()
                        )
                    ),
                },
            }
        )
    differences = []
    reference = fingerprints[0]
    for iteration, fingerprint in zip(iterations, fingerprints):
        index = iteration["index"]
        if len(iteration["node_ids"]) > 1 and not fingerprint:
            differences.append(
                {
                    "iteration_index": index,
                    "kind": "missing_topology",
                    "message": "本轮没有块内依赖边，无法验证完整拓扑。",
                }
            )
        if fingerprint != reference:
            differences.append(
                {
                    "iteration_index": index,
                    "kind": "before_topology",
                    "message": "本轮原图块内（含跨 Scope）连接与首轮不一致。",
                    "missing_edges": [
                        list(edge) for edge in (reference - fingerprint).elements()
                    ],
                    "extra_edges": [
                        list(edge) for edge in (fingerprint - reference).elements()
                    ],
                }
            )
        if iteration["after_statistics"] != iterations[0]["after_statistics"]:
            differences.append(
                {
                    "iteration_index": index,
                    "kind": "after_statistics",
                    "message": "本轮更新后节点数量或类型与首轮不同；重复标识仅依据原图。",
                    "expected": iterations[0]["after_statistics"],
                    "actual": iteration["after_statistics"],
                }
            )
    if model.get("before", {}).get("status") in ("partial", "unavailable"):
        differences.append(
            {
                "iteration_index": 0,
                "kind": "incomplete_source",
                "message": "原图数据覆盖不完整，拓扑一致仅为候选。",
            }
        )
    return iterations, differences


def _groups(model):
    nodes = {str(n["id"]): n for n in model.get("before", {}).get("nodes", [])}
    edges = model.get("before", {}).get("edges", [])
    incident = defaultdict(list)
    for e in edges:
        incident[str(e["source"])].append(e)
        if e["target"] != e["source"]:
            incident[str(e["target"])].append(e)
    signatures = []

    def scope_order(scope):
        positions = [
            (nodes[str(n)].get("stream_index", 0), nodes[str(n)].get("order", -1))
            for n in scope.get("node_ids", [])
            if str(n) in nodes
        ]
        valid = [p for p in positions if all(isinstance(v, int) and v >= 0 for v in p)]
        # Stream/order provides a stable source order; no lexical Scope-ID sort.
        return min(valid) if valid else (float("inf"), float("inf"))

    scopes = model.get("scopes", [])
    if scopes and all(str(scope["id"]).isdigit() for scope in scopes):
        scopes = sorted(scopes, key=lambda scope: int(scope["id"]))
    elif all(scope_order(scope)[0] != float("inf") for scope in scopes):
        scopes = sorted(scopes, key=scope_order)
    for scope in scopes:
        ids = [str(n) for n in scope.get("node_ids", [])]
        if not ids or len(set(ids)) != len(ids) or any(n not in nodes for n in ids):
            signatures.append(None)
            continue
        # Order in a scope is source evidence; preserve relative lane placement.
        lanes = {}
        typed = []
        for nid in ids:
            n = nodes[nid]
            lane = n.get("stream", n.get("stream_index"))
            lanes.setdefault(lane, len(lanes))
            typed.append((n.get("type"), _operator(n), lanes[lane]))
        signatures.append(tuple(typed))
    groups = []
    start = 0
    while start < len(scopes):
        best = None
        for period in range(1, min(64, (len(scopes) - start) // 2) + 1):
            pattern = signatures[start : start + period]
            if any(signature is None for signature in pattern):
                continue
            end = start + period
            while (
                end + period <= len(scopes)
                and signatures[end : end + period] == pattern
            ):
                end += period
            count = (end - start) // period
            if count >= 2 and (best is None or end > best[0]):
                best = (end, period, count)
        if best is None:
            start += 1
            continue
        end, period, count = best
        selected = scopes[start:end]
        sids = [str(s["id"]) for s in selected]
        ids = [str(n) for s in selected for n in s["node_ids"]]
        # Overlapping membership cannot represent independent repeated blocks.
        if len(ids) == len(set(ids)):
            after_ids = _projection(model, sids, ids)
            iterations, differences = _iteration_details(
                model, selected, period, incident
            )
            candidate = any(d["kind"] != "after_statistics" for d in differences)
            groups.append(
                {
                    "id": "repeat:" + sids[0],
                    "label": f"原图重复结构 ×{count}",
                    "scope_ids": sids,
                    "node_ids": ids,
                    "after_ids": after_ids,
                    "repeat_count": count,
                    "period": period,
                    "confidence": "candidate" if candidate else "structural",
                    "basis": "before",
                    "iterations": iterations,
                    "topology_status": "candidate" if candidate else "verified",
                    "differences": differences,
                    "reason": f"每 {period} 个连续 Scope 为一组，原图有序算子类型及相对队列相同；完整块内拓扑校验与更新后差异另列，不代表数值语义等价。",
                }
            )
        start = end
    return groups


def _unresolved(view, task, code, message):
    task["status"] = "unresolved"
    view["unresolved"].append({"task_id": task["id"], "code": code, "message": message})


def _sync(model):
    result = {}

    def view(sid):
        return result.setdefault(
            sid, {"tasks": [], "links": [], "unresolved": [], "boundary": []}
        )

    for scope in model.get("scopes", []):
        view(str(scope["id"]))
    for queue in model.get("queues", []):
        sid = queue.get("scope_id")
        if sid is None:
            continue
        sid = str(sid)
        current = view(sid)
        previous = None
        for raw in queue.get("tasks", []):
            task = {
                **raw,
                "id": f"{queue['id']}:{raw['index']}",
                "queue_id": queue["id"],
                "queue_kind": queue["kind"],
                "scope_id": sid,
                "status": "observed",
            }
            current["tasks"].append(task)
            if previous is not None:
                current["links"].append(
                    {
                        "source": previous,
                        "target": task["id"],
                        "kind": "queue",
                        "evidence": "explicit",
                        "reason": "源队列中的任务顺序。",
                    }
                )
            previous = task["id"]
            if task.get("type") == "SYNC":
                try:
                    text = str(task.get("args", "")).strip()
                    value = int(text, 16 if text.lower().startswith("0x") else 10)
                except (ValueError, TypeError):
                    value = None
                name, kind, direction = SYNC_TYPE_MAP.get(
                    value, ("UNKNOWN", "UNKNOWN", "UNKNOWN")
                )
                task.update(
                    sync_type=name,
                    sync_kind=kind,
                    sync_direction=direction,
                    sync_value=value,
                )
                if kind == "UNKNOWN":
                    _unresolved(
                        current,
                        task,
                        "unknown_sync",
                        "未知 SYNC 编码，未建立跨队列依赖。",
                    )
            elif task.get("type") in (
                "EVENT_NOTIFY",
                "EVENT_WAIT",
                "EVENT_RESET",
                "NOTIFY",
                "WAIT",
                "RESET",
                "MEMORY_WRITE",
                "MEMORY_WAIT",
            ):
                # Event args can identify a value address, but do not encode its value or round.
                _unresolved(
                    current,
                    task,
                    "opaque_signal_payload",
                    "同步值或轮次未解码；地址关联单独核对，不把地址相同当作确定执行依赖。",
                )
    from .sync_pairing import pair_sync

    for current in result.values():
        pair_sync(current)
    # Graph boundaries are distinct from unproven queue-task bindings. Preserve
    # their source stage and evidence rather than inventing task endpoints.
    members = defaultdict(set)
    for scope in model.get("scopes", []):
        for nid in scope.get("node_ids", []):
            members[str(nid)].add(str(scope["id"]))
    for stage in ("before", "after"):
        owners = members if stage == "before" else defaultdict(set)
        if stage == "after":
            for node in model.get(stage, {}).get("nodes", []):
                if node.get("scope_id") is not None:
                    owners[str(node["id"])].add(str(node["scope_id"]))
        for edge in model.get(stage, {}).get("edges", []):
            left, right = (
                owners.get(str(edge["source"]), set()),
                owners.get(str(edge["target"]), set()),
            )
            for sid in left ^ right:
                view(sid)["boundary"].append(
                    {
                        **edge,
                        "stage": stage,
                        "reason": "源执行图中的 Scope 边界依赖；未绑定内部任务。",
                    }
                )
    return result


def enrich_model(model):
    """Add idempotent exploration metadata and aggregate Chinese diagnostics."""
    from .relations import enrich_relations
    from .signals import enrich_signals

    sync = _sync(model)
    enrich_relations(model, sync)
    signal_groups = enrich_signals(model, sync)
    model["exploration"] = {
        "groups": _groups(model),
        "sync_by_scope": sync,
        "signal_groups": signal_groups,
    }
    diagnostics = model.setdefault("diagnostics", [])
    diagnostics[:] = [
        d
        for d in diagnostics
        if d.get("code")
        not in {"exploration_sync_unresolved", "exploration_repeat_difference"}
    ]
    affected = {sid: len(v["unresolved"]) for sid, v in sync.items() if v["unresolved"]}
    if affected:
        diagnostics.append(
            {
                "id": "exploration_sync_unresolved",
                "code": "exploration_sync_unresolved",
                "domain": "exploration",
                "severity": "warning",
                "message": f"{len(affected)} 个 Scope 有同步校验或布局提示，具体原因见详情。",
                "impact": "同步数据校验和显示布局需分别核对，不把布局提示当作配对失败。",
                "next_step": "按具体诊断核对源队列或对齐约束。",
                "source": {},
                "object": {},
                "details": {"scope_counts": affected},
            }
        )
    for sid, count in affected.items():
        current = sync[sid]
        has_error = any(
            u["code"]
            in {
                "sync_endpoint_missing",
                "sync_queue_invalid",
                "sync_count_mismatch",
                "sync_dependency_cycle",
                "sync_barrier_mismatch",
            }
            for u in current["unresolved"]
        )
        layout_only = all(
            u["code"] == "sync_alignment_conflict" for u in current["unresolved"]
        )
        diagnostics.append(
            {
                "id": "exploration_sync_unresolved:" + sid,
                "code": "exploration_sync_unresolved",
                "domain": "exploration",
                "severity": "error" if has_error else "warning",
                "message": f"Scope {sid}：同步结构异常，请核对端点、数量和队列。"
                if has_error
                else f"Scope {sid}：同步配对已恢复，部分端点无法同时对齐。"
                if layout_only
                else f"Scope {sid}：{count} 项同步信息需核对，详见原因。",
                "impact": "配对不受影响，仅端点未全部位于同一列。"
                if layout_only
                else "请核对诊断所列的同步端点。",
                "next_step": "沿同步连线查看配对；布局优先保留队列顺序。"
                if layout_only
                else "核对源队列及具体诊断字段。",
                "source": next(
                    (
                        t.get("source", {})
                        for t in current["tasks"]
                        if t.get("status") == "unresolved"
                    ),
                    {},
                ),
                "object": {"scope_id": sid, "stage": "before"},
                "details": {"unresolved": current["unresolved"]},
            }
        )
    for group in model["exploration"]["groups"]:
        if group["differences"]:
            diagnostics.append(
                {
                    "id": "exploration_repeat_difference:" + group["id"],
                    "code": "exploration_repeat_difference",
                    "domain": "exploration",
                    "severity": "warning",
                    "message": f"{group['label']}：存在拓扑证据缺失或迭代差异，请检查候选。",
                    "impact": "折叠保留全部原始节点；重复标识只依据原图，不能推断更新后等价。",
                    "next_step": "展开对应迭代并核对差异边和更新后统计。",
                    "source": {},
                    "object": {"scope_id": group["scope_ids"][0], "stage": "before"},
                    "details": {
                        "group_id": group["id"],
                        "topology_status": group["topology_status"],
                        "differences": group["differences"],
                    },
                }
            )
    return model
