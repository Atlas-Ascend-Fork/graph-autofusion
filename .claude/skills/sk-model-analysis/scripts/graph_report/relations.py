"""Model-local, evidence-backed associations, separate from dependency edges."""

import itertools
from collections import Counter, defaultdict, deque


def address(value):
    try:
        number = int(str(value), 16 if str(value).lower().startswith("0x") else 10)
        return hex(number) if number > 0 else None
    except (ValueError, TypeError):
        return None


def _columns(tasks, links, origins, alignment_pairs=()):
    queues = defaultdict(list)
    for t in tasks:
        queues[t["queue_id"]].append(t)
    for q in queues.values():
        q.sort(key=lambda t: t["index"])
    raw = {t["id"]: i for q in queues.values() for i, t in enumerate(q)}
    edges = [
        (a["id"], b["id"]) for q in queues.values() for a, b in itertools.pairwise(q)
    ]
    # Only explicit sync edges can constrain scheduling; inferred pairs stay visual.
    edges += [
        (e["source"], e["target"])
        for e in links
        if e.get("kind") == "sync" and e.get("evidence") == "explicit"
    ]
    weighted_edges = [(a, b, 1) for a, b in edges] + [
        (e["source"], e["target"], 0)
        for e in links
        if e.get("kind") == "sync" and e.get("basis") == "flag_counter_fifo"
    ]
    parent = {t["id"]: t["id"] for t in tasks}

    def ranks(mapping):
        graph = {v: {} for v in mapping.values()}
        degree = dict.fromkeys(graph, 0)
        for a, b, distance in weighted_edges:
            if a not in mapping or b not in mapping:
                continue
            a, b = mapping[a], mapping[b]
            if a == b:
                if distance:
                    return None
                continue
            if b not in graph[a]:
                degree[b] += 1
            graph[a][b] = max(graph[a].get(b, 0), distance)
        ready = deque(k for k, v in degree.items() if not v)
        result = dict.fromkeys(graph, 0)
        count = 0
        while ready:
            a = ready.popleft()
            count += 1
            for b, distance in graph[a].items():
                result[b] = max(result[b], result[a] + distance)
                degree[b] -= 1
                if degree[b] == 0:
                    ready.append(b)
        return (
            {k: result[v] for k, v in mapping.items()} if count == len(graph) else None
        )

    by_origin = defaultdict(list)
    for t in tasks:
        if t["id"] in origins:
            by_origin[origins[t["id"]]].append(t)
    by_id = {t["id"]: t for t in tasks}
    anchor_groups = list(by_origin.values())
    anchor_groups.extend(
        [by_id[a], by_id[b]] for a, b in alignment_pairs if a in by_id and b in by_id
    )
    conflicts = 0
    for same in anchor_groups:
        if len(same) < 2:
            continue
        if len({t["queue_id"] for t in same}) != len(same):
            conflicts += 1
            continue
        roots = {parent[t["id"]] for t in same}
        root = parent[same[0]["id"]]
        trial = {k: root if v in roots else v for k, v in parent.items()}
        if ranks(trial) is None:
            conflicts += 1
        else:
            parent = trial
    aligned = ranks(parent)
    if aligned is not None:
        group_sizes = Counter(parent.values())
        successors = defaultdict(list)
        for a, b, distance in weighted_edges:
            successors[a].append((b, distance))
        # Keep a source-marked PRELOAD/WAIT beside its FUNC when alignment inserts gaps.
        # Moving right cannot violate incoming constraints; all outgoing constraints bound it.
        for q in queues.values():
            for t, following in reversed(list(itertools.pairwise(q))):
                origin = t.get("companion_node_id")
                next_origin = following.get(
                    "origin_node_id", following.get("companion_node_id")
                )
                if (
                    origin is not None
                    and origin == next_origin
                    and t.get("type") != "FUNC"
                    and group_sizes[parent[t["id"]]] == 1
                ):
                    bound = min(
                        (
                            aligned[b] - d
                            for b, d in successors[t["id"]]
                            if b in aligned
                        ),
                        default=aligned[t["id"]],
                    )
                    aligned[t["id"]] = max(aligned[t["id"]], bound)
    return {
        "raw": raw,
        "aligned": aligned or raw,
        "conflicts": conflicts + int(aligned is None),
    }


def enrich_relations(model, views):
    before = {str(n["id"]): n for n in model.get("before", {}).get("nodes", [])}
    fused = defaultdict(list)
    for f in model.get("fusions", []):
        for nid in f.get("node_ids", []):
            if str(nid) not in fused[str(f["scope_id"])]:
                fused[str(f["scope_id"])].append(str(nid))
    values = defaultdict(list)
    for n in model.get("after", {}).get("nodes", []):
        for field, role in [("valueWaitParams", "wait"), ("valueWriteParams", "write")]:
            p = n.get("params", {}).get(field)
            if p and address(p.get("devAddr")):
                values[address(p["devAddr"])].append((n, p, role))
    # Log parser exposes the same value tasks as normalized addr/value fields.
    for n in model.get("after", {}).get("nodes", []):
        p = n.get("params", {})
        role = {"MEMORY_WAIT": "wait", "MEMORY_WRITE": "write"}.get(n.get("type"))
        addr = address(p.get("addr"))
        if (
            role
            and addr
            and not any(str(other[0]["id"]) == str(n["id"]) for other in values[addr])
        ):
            values[addr].append((n, p, role))
    for sid, view in views.items():
        relations = []
        issues = []
        origins = {}
        tasks = view["tasks"]
        members = fused[sid]
        for t in tasks:
            if t.get("type") != "FUNC":
                continue
            entries = {a for e in t.get("entries", []) if (a := address(e))}
            candidates = []
            for nid in members:
                k = before.get(nid, {}).get("params", {}).get("kernelParams", {})
                funcs = {
                    a
                    for f in k.get("resolvedFuncs", [])
                    for e in f.get("funcAddr", [])
                    if (a := address(e))
                }
                if entries and entries <= funcs:
                    candidates.append(nid)
            index = t.get("node_index")
            indexed = (
                members[index]
                if isinstance(index, int) and 0 <= index < len(members)
                else None
            )
            # Index disambiguates reused functions only when addresses corroborate it.
            nid = (
                indexed
                if indexed in candidates
                else candidates[0]
                if len(candidates) == 1 and indexed is None
                else None
            )
            if nid is None:
                issues.append(
                    {
                        "code": "origin_unconfirmed",
                        "task_id": t["id"],
                        "candidates": candidates,
                        "message": "FUNC 来源索引与函数地址未形成唯一对应。",
                    }
                )
                continue
            origins[t["id"]] = nid
            t["origin_node_id"] = nid
            t["origin_label"] = before[nid].get("label", nid)
            relations.append(
                {
                    "kind": "origin",
                    "task_id": t["id"],
                    "node_id": nid,
                    "stage": "before",
                    "evidence": "corroborated",
                    "reason": "融合成员索引与函数地址交叉核对。",
                    "source": t.get("source", {}),
                }
            )
        funcs = defaultdict(list)
        for t in tasks:
            if t["id"] in origins:
                funcs[(t["queue_id"], t.get("node_index"))].append(t)
        for t in tasks:
            if t.get("type") not in ("PRELOAD", "SYNC"):
                continue
            candidates = funcs[(t["queue_id"], t.get("node_index"))]
            if len(candidates) != 1:
                continue
            f = candidates[0]
            if t["type"] == "PRELOAD" and {
                address(e) for e in t.get("entries", [])
            } != {address(e) for e in f.get("entries", [])}:
                continue
            relations.append(
                {
                    "kind": "companion",
                    "source": t["id"],
                    "target": f["id"],
                    "evidence": "source_marker",
                    "reason": "同队列、同 nodeIndex 的配套标记；不表示额外执行依赖。",
                }
            )
            t["companion_node_id"] = origins[f["id"]]
        for t in tasks:
            if t.get("type") not in ("EVENT_NOTIFY", "EVENT_WAIT", "EVENT_RESET"):
                continue
            origin = t.get("event_origin") or {}
            nid = str(origin.get("node_id", ""))
            if nid in members and before.get(nid, {}).get("type") == t[
                "type"
            ].removeprefix("EVENT_"):
                t["origin_node_id"] = nid
                t["origin_label"] = before[nid].get("label", nid)
                relations.append(
                    {
                        "kind": "event_origin",
                        "task_id": t["id"],
                        "node_id": nid,
                        "stage": "before",
                        "evidence": "explicit_node_id",
                        "reason": "设备任务日志明确记录融合前事件的 nodeId。",
                        "source": origin.get("source", {}),
                    }
                )
            addr = address(t.get("args"))
            matches = values.get(addr, [])
            for n, p, role in matches:
                relations.append(
                    {
                        "kind": "address",
                        "task_id": t["id"],
                        "node_id": str(n["id"]),
                        "stage": "after",
                        "address": addr,
                        "role": role,
                        "value": p.get("value"),
                        "round_status": "unknown",
                        "evidence": "address_equal",
                        "reason": "内部 args 与更新后 value 事件 devAddr 相同；角色独立展示，未推定通知值或轮次。",
                        "source": {
                            "internal": t.get("source", {}),
                            "external": n.get("source", {}),
                        },
                    }
                )
            if matches:
                view["unresolved"] = [
                    u
                    for u in view.get("unresolved", [])
                    if not (
                        u.get("task_id") == t["id"]
                        and u.get("code") == "opaque_signal_payload"
                    )
                ]
                t["status"] = "address_linked"
                issues.append(
                    {
                        "code": "address_round_unknown",
                        "task_id": t["id"],
                        "address": addr,
                        "message": "地址关联已确认；通知值及具体轮次未提供，不推定生产消费配对。",
                    }
                )
        view["relations"] = relations
        view["layout"] = _columns(tasks, view.get("links", []), origins)
        if view["layout"]["conflicts"]:
            issues.append(
                {
                    "code": "alignment_conflict",
                    "message": "部分来源对齐与队列顺序冲突，已保留原顺序。",
                }
            )
        view["relation_issues"] = issues
    diagnostics = model.setdefault("diagnostics", [])
    diagnostics[:] = [
        d for d in diagnostics if d.get("code") != "graph_relations_incomplete"
    ]
    affected = {
        sid: v["relation_issues"] for sid, v in views.items() if v["relation_issues"]
    }
    if affected:
        diagnostics.append(
            {
                "id": "graph_relations_incomplete",
                "code": "graph_relations_incomplete",
                "domain": "exploration",
                "severity": "warning",
                "message": f"{len(affected)} 个 Scope 存在来源映射、地址轮次或对齐待确认项。",
                "impact": "保留已确认关联，不建立未知轮次依赖。",
                "next_step": "核对来源索引、函数地址与信号值/轮次。",
                "details": {"scopes": affected},
            }
        )
