"""Recover fixed-flag, one-token SET/WAIT consumption in SK queue order.

Source: super_kernel/kernel/sk_entry.asc, AutoCoreSyncImpl / FullCoreSyncImpl.
C->V uses SYNC_AIC_AIV_FLAG, V->C uses SYNC_AIV_FLAG. CrossCoreWaitFlag
consumes one counter increment (asc-devkit CrossCoreWaitFlag_ISASI.md).
ALL uses both flags; validate balanced intervals between shared barriers.
No nodeIndex, address grouping, or Cartesian product defines these pairs.
"""

from collections import defaultdict, deque

ERROR_CODES = {
    "sync_endpoint_missing",
    "sync_queue_invalid",
    "sync_count_mismatch",
    "sync_dependency_cycle",
    "sync_barrier_mismatch",
}


def pair_sync(view):
    def issue(tasks, code, message):
        for t in tasks:
            t["status"] = "unresolved"
            view["unresolved"].append(
                {"task_id": t["id"], "code": code, "message": message}
            )

    queues = defaultdict(list)
    for t in view["tasks"]:
        queues[t["queue_id"]].append(t)
    for q in queues.values():
        q.sort(key=lambda t: t["index"])
    lanes = defaultdict(list)
    for q in queues.values():
        lanes[q[0]["queue_kind"]].append(q)
    view["barrier_pairs"] = []
    if any(len(qs) > 1 for qs in lanes.values()):
        issue(
            [t for t in view["tasks"] if t.get("type") == "SYNC"],
            "sync_queue_invalid",
            "同一 Scope 存在多个同类型队列，无法确定共享 flag 的队列身份。",
        )
        return
    barriers = {
        k: [t for q in qs for t in q if t.get("sync_kind") == "ALL"]
        for k, qs in lanes.items()
    }
    a, b = barriers.get("AIC", []), barriers.get("AIV", [])
    if len(lanes) > 1 and len(a) != len(b):
        issue(
            a + b,
            "sync_barrier_mismatch",
            "AIC/AIV 的 ALL 屏障数量不一致，无法恢复共同屏障。",
        )
        return
    for i, (left, right) in enumerate(zip(a, b)):
        view["barrier_pairs"].append(
            {
                "source": left["id"],
                "target": right["id"],
                "kind": "all_sync",
                "evidence": "inferred",
                "basis": "flag_counter_fifo",
                "round": i + 1,
                "reason": "两队列按执行顺序到达的共同 ALL 屏障；内部同时发送并等待两个方向的 flag。",
            }
        )
        left["status"] = right["status"] = "paired"
    groups = defaultdict(lambda: [[], []])
    for q in queues.values():
        phase = 0
        for t in q:
            kind = t.get("sync_kind")
            if kind == "ALL":
                phase += 1
            elif kind in ("SET", "WAIT"):
                groups[(phase, t["sync_direction"])][kind == "WAIT"].append(t)
    pairs = []
    for (phase, direction), (sets, waits) in sorted(groups.items()):
        sender, receiver = direction.split("_TO_")
        if any(t["queue_kind"] != sender for t in sets) or any(
            t["queue_kind"] != receiver for t in waits
        ):
            issue(
                sets + waits, "sync_queue_invalid", "SET/WAIT 所在队列与同步方向不符。"
            )
        elif not sets or not waits:
            issue(
                sets + waits,
                "sync_endpoint_missing",
                "该屏障区间内缺少对应方向的 SET 或 WAIT，未跨 ALL 错配。",
            )
        elif len(sets) != len(waits):
            issue(
                sets + waits,
                "sync_count_mismatch",
                f"该屏障区间内 SET={len(sets)}、WAIT={len(waits)}，端点数量不一致。",
            )
        else:
            for i, (s, w) in enumerate(zip(sets, waits)):
                pairs.append(
                    {
                        "source": s["id"],
                        "target": w["id"],
                        "kind": "sync",
                        "direction": direction,
                        "evidence": "inferred",
                        "basis": "flag_counter_fifo",
                        "barrier_interval": phase,
                        "round": i + 1,
                        "reason": f"固定方向 flag 每次 WAIT 消耗一次 SET；按源队列顺序配对，ALL 区间 {phase + 1}，第 {i + 1} 对。",
                    }
                )
    # Merge the two halves of a shared barrier, then reject dependency cycles.
    parent = {t["id"]: t["id"] for t in view["tasks"]}
    for e in view["barrier_pairs"]:
        parent[e["target"]] = e["source"]
    graph = {n: set() for n in parent.values()}
    indegree = dict.fromkeys(graph, 0)
    for e in view["links"] + pairs:
        a, b = parent[e["source"]], parent[e["target"]]
        if b not in graph[a]:
            graph[a].add(b)
            indegree[b] += 1
    ready = deque(n for n, d in indegree.items() if not d)
    visited = 0
    while ready:
        n = ready.popleft()
        visited += 1
        for peer in graph[n]:
            indegree[peer] -= 1
            if not indegree[peer]:
                ready.append(peer)
    if visited != len(graph):
        ids = {e[k] for e in pairs for k in ("source", "target")}
        issue(
            [t for t in view["tasks"] if t["id"] in ids],
            "sync_dependency_cycle",
            "同步配对与队列顺序构成循环等待，请核对源任务队列；未绘制矛盾依赖。",
        )
        return
    by_id = {t["id"]: t for t in view["tasks"]}
    for e in pairs:
        for own, peer in [(e["source"], e["target"]), (e["target"], e["source"])]:
            by_id[own].update(status="paired", sync_peer_id=peer, sync_round=e["round"])
    view["links"].extend(pairs)
