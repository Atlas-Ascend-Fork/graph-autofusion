"""Signal groups are model-local identities, not scheduling edges or rounds."""

from .relations import _columns, address


def _align_sync_pairs(view):
    view["unresolved"] = [
        u
        for u in view.get("unresolved", [])
        if u.get("code") != "sync_alignment_conflict"
    ]
    view["sync_candidates"] = []
    barriers = view.get("barrier_pairs", [])
    anchors = [
        (e["source"], e["target"])
        for e in view.get("links", []) + barriers
        if e.get("kind") in ("sync", "all_sync")
    ]
    if anchors:
        origins = {
            t["id"]: t["origin_node_id"]
            for t in view["tasks"]
            if t.get("origin_node_id") is not None
        }
        view["layout"] = _columns(
            view["tasks"], view.get("links", []), origins, anchors
        )

        if view["layout"]["conflicts"]:
            view.setdefault("unresolved", []).append(
                {
                    "code": "sync_alignment_conflict",
                    "task_id": anchors[0][0],
                    "message": "部分同步端点不能同时与来源任务对齐；配对已恢复，布局保留队列顺序和依赖方向。",
                }
            )


def enrich_signals(model, views):
    groups = {}

    def group(addr):
        return groups.setdefault(
            addr,
            {
                "id": addr,
                "address": addr,
                "internal": [],
                "after": [],
                "before": [],
                "bridges": [],
                "round_status": "unknown",
            },
        )

    for sid, v in views.items():
        _align_sync_pairs(v)
        v["relations"] = [
            r for r in v.get("relations", []) if r.get("kind") != "signal_bridge"
        ]
        for t in v["tasks"]:
            if t.get("type") not in ("EVENT_NOTIFY", "EVENT_WAIT", "EVENT_RESET"):
                continue
            addr = address(t.get("args"))
            if addr:
                group(addr)["internal"].append(
                    {
                        "scope_id": sid,
                        "task_id": t["id"],
                        "role": t["type"],
                        "source": t.get("source", {}),
                    }
                )
    for n in model.get("after", {}).get("nodes", []):
        p = n.get("params", {})
        value = p.get("valueWaitParams") or p.get("valueWriteParams")
        if value:
            addr = address(value.get("devAddr"))
            role = "wait" if "valueWaitParams" in p else "write"
        elif n.get("type") in ("MEMORY_WAIT", "MEMORY_WRITE"):
            value = p
            addr = address(p.get("addr"))
            role = "wait" if n["type"] == "MEMORY_WAIT" else "write"
        else:
            continue
        if not addr:
            continue
        g = group(addr)
        g["after"].append(
            {
                "node_id": str(n["id"]),
                "role": role,
                "value": value.get("value"),
                "source": n.get("source", {}),
            }
        )
    # An updated graph node can reuse an original Event's ID for another signal.
    # Matching IDs (even with the same stream or role) do not prove signal identity.
    # Refine extraction diagnostics only after model-local cross-layer evidence exists.
    for d in model.get("diagnostics", []):
        details = d.get("details", {})
        if d.get("object", {}).get("stage") != "after":
            continue
        is_json = (
            d.get("code") == "unresolved_synchronization"
            and details.get("kind") == "memory"
            and not details.get("producer_ids")
        )
        is_log = d.get("code", "").startswith("log_memory_ambiguous_after_")
        if not (is_json or is_log):
            continue
        candidates = [
            g
            for g in groups.values()
            if (is_json and address(details.get("signal")) == g["address"])
            or (
                is_log
                and any(
                    n["role"] == "wait" and n["node_id"] in details.get("nodes", [])
                    for n in g["after"]
                )
            )
        ]
        if len(candidates) != 1:
            continue
        g = candidates[0]
        if is_log:
            from .synchronization import normalized, signal_fields

            wait_ids = {n["node_id"] for n in g["after"] if n["role"] == "wait"}
            waits = [
                n
                for n in model.get("after", {}).get("nodes", [])
                if str(n["id"]) in wait_ids
            ]
            if (
                len(waits) != 1
                or normalized(signal_fields(waits[0])["wait_flag"]) != "1"
            ):
                continue
        producers = [t for t in g["internal"] if t["role"] == "EVENT_NOTIFY"]
        if len(producers) != 1 or any(
            n["role"] == "write" and str(n["value"]) not in ("0", "0x0")
            for n in g["after"]
        ):
            continue
        d["details"] = {
            **details,
            "original_code": d["code"],
            "original_message": d.get("message"),
            "internal_producers": producers,
            "address": g["address"],
            "round_status": "unknown",
        }
        d.update(
            code="cross_layer_signal_association",
            severity="info",
            message="已找到同地址的 SK 内部 EVENT_NOTIFY 与外部 Value 关联。",
            impact="跨层地址关联已确认；未据此推定通知值、轮次或执行依赖。",
            next_step="查看关联的内部任务及外部 Value 参数。",
        )
    return list(groups.values())
