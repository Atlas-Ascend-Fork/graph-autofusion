"""Evidence-backed graph annotations, separate from DFX presentation."""

from __future__ import annotations

from collections import defaultdict


_BREAK_REASONS = {
    "UNFUSIBLE_NODE": "算子导致融合断开",
    "SYNCALL_OP_DROP": "同步信号导致融合断开",
}


def build_fusion_annotations(model):
    fusion = model.get("fusion_analysis")
    if not isinstance(fusion, dict):
        fusion = {}
    before = {
        str(n["id"]): n
        for n in (model.get("before") or {}).get("nodes", [])
        if n.get("id") is not None
    }
    after = {
        str(n["id"]): n
        for n in (model.get("after") or {}).get("nodes", [])
        if n.get("id") is not None
    }
    scopes = {
        str(s["id"]): s for s in model.get("scopes", []) if s.get("id") is not None
    }
    fused = {
        str(f.get("scope_id"))
        for f in model.get("fusions", [])
        if f.get("scope_id") is not None
    }
    fused_members = {
        str(i) for f in model.get("fusions", []) for i in f.get("node_ids") or []
    }
    fused.update(
        str(n["scope_id"])
        for n in after.values()
        if n.get("is_sk") and n.get("scope_id") is not None
    )
    failures = defaultdict(list)
    for failure in fusion.get("node_failures") or []:
        if isinstance(failure, dict) and failure.get("node_id") is not None:
            failures[str(failure["node_id"])].append(failure)
    breaks = []
    seen = set()
    for boundary in fusion.get("boundaries") or []:
        if not isinstance(boundary, dict) or not isinstance(
            boundary.get("trigger"), dict
        ):
            continue
        sid = str(boundary.get("scope_id"))
        node_id = str((boundary.get("trigger") or {}).get("node_id"))
        reason = _BREAK_REASONS.get(boundary.get("break_reason"))
        key = (sid, node_id, reason)
        if (
            reason
            and sid in scopes
            and sid in fused
            and node_id in before
            and key not in seen
        ):
            item = {"scope_id": sid, "node_id": node_id, "reason": reason}
            for failure in failures[node_id]:
                if failure.get("reason_code") == "BINDMAP_INIT_EMPTY":
                    name = str(failure.get("raw_name") or "")
                    item["detail"] = (
                        "算子不支持静态编译，或静态编译失败"
                        if name and not name.lower().startswith("static_kernel_")
                        else "融合所需的 BindMap 为空"
                    )
                    break
                if failure.get("reason") == "OP_UNSUPPORT":
                    item["detail"] = "算子不满足当前融合条件"
            breaks.append(item)
            seen.add(key)

    failed_scopes = []
    for sid, scope in scopes.items():
        ids = [str(i) for i in scope.get("node_ids") or []]
        if sid in fused or not ids or any(i not in before for i in ids):
            continue
        kernels = [
            i
            for i in ids
            if str(before[i].get("type", "")).upper() == "KERNEL"
            and not str(before[i].get("label", ""))
            .lower()
            .startswith(("sk_scope_kernel_", "sk_placeholder_kernel"))
        ]
        if not kernels or any(i in fused_members for i in kernels):
            continue
        if not all(
            i in after
            and not after[i].get("is_sk")
            and str(after[i].get("type", "")).upper() == "KERNEL"
            for i in kernels
        ):
            continue
        if not all(
            any(
                f.get("reason") == "SCOPE_FUSE_PART"
                and f.get("reason_code") == "RESOURCE_INSUFFICIENT"
                for f in failures[i]
            )
            for i in kernels
        ):
            continue
        failed_scopes.append(
            {"scope_id": sid, "reason": "资源不足，未能完成融合", "node_ids": kernels}
        )
    return {"breaks": breaks, "failed_scopes": failed_scopes}
