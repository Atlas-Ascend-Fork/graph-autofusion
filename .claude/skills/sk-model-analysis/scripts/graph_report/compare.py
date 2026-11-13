"""Compare independent facts, then resolve display sources without erasing either side."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

from .common import empty_result, issue


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return int(value, 16 if value.lower().startswith("0x") else 10)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_norm(item) for item in value]
    return value


def _normalize_updated_labels(snapshot: dict) -> None:
    """Keep visible labels aligned with explicit post-update task types."""
    for node in snapshot.get("nodes", []):
        if str(node.get("type", "")).upper() in {
            "MEMORY_WRITE",
            "MEMORY_WAIT",
            "INVALID",
        }:
            node["label"] = str(node["type"]).upper()


def _attach_event_origins(model: dict, log_result: dict) -> None:
    """Transfer explicit log identities only between identical, unique queues."""
    log_queues = defaultdict(list)
    for queue in log_result.get("queues", []):
        if queue.get("scope_id") is not None:
            log_queues[(str(queue["scope_id"]), queue.get("kind"))].append(queue)
    selected = Counter(
        (str(q.get("scope_id")), q.get("kind")) for q in model.get("queues", [])
    )
    for queue in model.get("queues", []):
        key = (str(queue.get("scope_id")), queue.get("kind"))
        matches = log_queues.get(key, [])
        if queue.get("scope_id") is None or len(matches) != 1 or selected[key] != 1:
            continue
        target, source = queue.get("tasks", []), matches[0].get("tasks", [])
        if len(target) != len(source) or any(
            any(
                _norm(a.get(field)) != _norm(b.get(field))
                for field in ("index", "type", "args", "entries")
            )
            for a, b in zip(target, source)
        ):
            continue
        for task, original in zip(target, source):
            evidence = original.get("params", {})
            if (
                task.get("type") in {"EVENT_NOTIFY", "EVENT_WAIT", "EVENT_RESET"}
                and evidence.get("graph_identity_valid") is True
                and evidence.get("graph_identity_source") == "event_explicit_node_id"
                and evidence.get("node_id") is not None
            ):
                task["event_origin"] = {
                    "node_id": str(evidence["node_id"]),
                    "source": original.get("source", {}),
                }


def compare_results(json_result: dict, log_result: dict) -> dict:
    differences, single_sides, incomparable = [], [], []
    compared = 0

    def fields(j: dict, log_data: dict, names: tuple, obj: dict, capability: str):
        nonlocal compared
        for name in names:
            if (
                name not in j
                or name not in log_data
                or j[name] is None
                or log_data[name] is None
            ):
                continue
            compared += 1
            if _norm(j[name]) != _norm(log_data[name]):
                differences.append(
                    {
                        "capability": capability,
                        "object": obj,
                        "field": name,
                        "json": j[name],
                        "log": log_data[name],
                        "json_source": j.get("source", {}),
                        "log_source": log_data.get("source", {}),
                    }
                )

    def membership(j: dict, log_data: dict, capability: str):
        for source, keys in (
            ("json", j.keys() - log_data.keys()),
            ("log", log_data.keys() - j.keys()),
        ):
            for key in sorted(keys, key=str):
                single_sides.append(
                    {"capability": capability, "key": str(key), "source": source}
                )

    for stage in ("before", "after"):
        j, log_data = json_result[stage], log_result[stage]
        if "unavailable" in (j["status"], log_data["status"]):
            incomparable.append(
                {"capability": stage, "json": j["status"], "log": log_data["status"]}
            )
            continue
        if "partial" in (j["status"], log_data["status"]):
            incomparable.append(
                {
                    "capability": stage,
                    "reason": "partial_coverage",
                    "json": j["status"],
                    "log": log_data["status"],
                }
            )
        jnodes = {n["id"]: n for n in j["nodes"]}
        lnodes = {n["id"]: n for n in log_data["nodes"]}
        membership(jnodes, lnodes, stage)
        for key in sorted(jnodes.keys() & lnodes.keys()):
            fields(
                jnodes[key],
                lnodes[key],
                ("type", "stream_index", "order", "is_sk", "scope_id"),
                {"stage": stage, "node_id": key},
                stage,
            )
            from .synchronization import signal_fields

            fields(
                signal_fields(jnodes[key]),
                signal_fields(lnodes[key]),
                (
                    "event_id",
                    "event_flag",
                    "event_task_flag",
                    "address",
                    "value",
                    "wait_flag",
                ),
                {"stage": stage, "node_id": key},
                stage + ".signal",
            )
        # Compare edge multiplicities, without equating parser confidence with topology.
        je = Counter((e["source"], e["target"], e["kind"]) for e in j["edges"])
        le = Counter((e["source"], e["target"], e["kind"]) for e in log_data["edges"])
        for key in sorted(je.keys() | le.keys()):
            if not je[key] or not le[key]:
                single_sides.append(
                    {
                        "capability": stage + ".edges",
                        "key": str(key),
                        "source": "json" if je[key] else "log",
                    }
                )
            else:
                compared += 1
                if je[key] != le[key]:
                    differences.append(
                        {
                            "capability": stage + ".edges",
                            "object": {"stage": stage, "node_id": key[0]},
                            "field": "edge_count",
                            "json": je[key],
                            "log": le[key],
                        }
                    )

    for capability, key_name in (("scopes", "id"), ("fusions", "scope_id")):
        jc = json_result["capabilities"].get(capability, "unavailable")
        lc = log_result["capabilities"].get(capability, "unavailable")
        if "unavailable" in (jc, lc):
            incomparable.append({"capability": capability, "json": jc, "log": lc})
            continue
        js = {r[key_name]: r for r in json_result[capability]}
        ls = {r[key_name]: r for r in log_result[capability]}
        membership(js, ls, capability)
        for key in sorted(js.keys() & ls.keys()):
            j, log_data = dict(js[key]), dict(ls[key])
            j["node_ids"] = sorted(set(j.get("node_ids", [])))
            log_data["node_ids"] = sorted(set(log_data.get("node_ids", [])))
            fields(j, log_data, ("node_ids",), {"scope_id": key}, capability)

    # Manual boundaries are compared independently of optimizer Scope IDs.
    ja = {
        tuple(sorted(s.get("marker_node_ids", []))): s
        for s in json_result.get("user_scopes", [])
    }
    la = {
        tuple(sorted(s.get("marker_node_ids", []))): s
        for s in log_result.get("user_scopes", [])
    }
    membership(ja, la, "user_scopes")
    for key in sorted(ja.keys() & la.keys()):
        fields(
            ja[key],
            la[key],
            ("name",),
            {"stage": "before", "node_id": key[0]},
            "user_scopes",
        )

    jc = json_result["capabilities"].get("queues", "unavailable")
    lc = log_result["capabilities"].get("queues", "unavailable")
    if "unavailable" in (jc, lc):
        incomparable.append({"capability": "queues", "json": jc, "log": lc})
    else:

        def queue_map(result):
            out = {}
            for q in result["queues"]:
                if q.get("scope_id") is None:
                    incomparable.append(
                        {
                            "capability": "queues",
                            "reason": "unbound",
                            "parser": result["parser"],
                            "queue": q["id"],
                        }
                    )
                    continue
                key = (q["scope_id"], q["kind"])
                if key in out:
                    incomparable.append(
                        {
                            "capability": "queues",
                            "reason": "duplicate_binding",
                            "parser": result["parser"],
                            "queue": str(key),
                        }
                    )
                    out[key] = None
                else:
                    out[key] = q
            return out

        js, ls = queue_map(json_result), queue_map(log_result)
        membership(js, ls, "queues")
        for key in sorted(js.keys() & ls.keys()):
            j, log_data = js[key], ls[key]
            if j is None or log_data is None:
                continue
            obj = {"scope_id": key[0]}
            compared += 1
            if len(j["tasks"]) != len(log_data["tasks"]):
                differences.append(
                    {
                        "capability": "queues",
                        "object": obj,
                        "field": "task_count",
                        "json": len(j["tasks"]),
                        "log": len(log_data["tasks"]),
                    }
                )
            jtasks = {t["index"]: t for t in j["tasks"]}
            ltasks = {t["index"]: t for t in log_data["tasks"]}
            if len(jtasks) != len(j["tasks"]) or len(ltasks) != len(log_data["tasks"]):
                incomparable.append(
                    {
                        "capability": "queues",
                        "reason": "duplicate_task_index",
                        "scope_id": key[0],
                        "queue": key[1],
                    }
                )
                continue
            membership(jtasks, ltasks, f"queues:{key[0]}:{key[1]}")
            for task_index in sorted(jtasks.keys() & ltasks.keys()):
                jt, lt = jtasks[task_index], ltasks[task_index]
                fields(
                    jt,
                    lt,
                    (
                        "index",
                        "node_index",
                        "type",
                        "num_blocks",
                        "entries",
                        "args",
                        "debug_options",
                    ),
                    {**obj, "queue": key[1], "task_index": jt["index"]},
                    "queues",
                )
    status = (
        "differences"
        if differences
        else "limited"
        if single_sides or incomparable
        else "consistent"
    )
    if compared == 0:
        status = "not_comparable"
    return {
        "status": status,
        "compared_fields": compared,
        "difference_count": len(differences),
        "single_side_count": len(single_sides),
        "differences": differences,
        "single_sides": single_sides,
        "not_comparable": incomparable,
    }


def _transformations(before: dict, after: dict, fusions: list) -> list:
    b = {n["id"]: n for n in before["nodes"]}
    a = {n["id"]: n for n in after["nodes"]}
    out = []
    for key in sorted(b.keys() & a.keys()):
        if a[key].get("update_status") == "not_recorded":
            continue
        out.append(
            {
                "before_ids": [key],
                "after_ids": [key],
                "kind": "modified"
                if (b[key]["type"], b[key].get("is_sk"))
                != (a[key]["type"], a[key].get("is_sk"))
                else "retained",
            }
        )
    # Absence in a partial snapshot cannot prove removal/addition.
    if after["status"] == "complete":
        out.extend(
            {"before_ids": [key], "after_ids": [], "kind": "removed"}
            for key in sorted(b.keys() - a.keys())
        )
    if before["status"] == "complete":
        out.extend(
            {"before_ids": [], "after_ids": [key], "kind": "added"}
            for key in sorted(a.keys() - b.keys())
        )
    by_scope = {}
    for n in after["nodes"]:
        if n.get("is_sk") and n.get("scope_id") is not None:
            by_scope.setdefault(n["scope_id"], []).append(n["id"])
    for f in fusions:
        if f["scope_id"] in by_scope:
            out.append(
                {
                    "kind": "fusion",
                    "scope_id": f["scope_id"],
                    "before_ids": f["node_ids"],
                    "after_ids": by_scope[f["scope_id"]],
                }
            )
    return out


_SCOPE_KERNEL_PREFIXES = (
    "sk_scope_kernel_begin",
    "sk_scope_kernel_end",
    "sk_placeholder_kernel",
)


def _is_scope_kernel_node(node: dict | None) -> bool:
    """Recognize framework boundary kernels without classifying real operators."""
    if not isinstance(node, dict):
        return False
    params = node.get("params") if isinstance(node.get("params"), dict) else {}
    kernel_params = (
        params.get("kernelParams")
        if isinstance(params.get("kernelParams"), dict)
        else {}
    )
    names = (
        node.get("label"),
        node.get("name"),
        params.get("funcName"),
        params.get("func_name"),
        kernel_params.get("funcName"),
        kernel_params.get("func_name"),
    )
    return any(
        str(name or "").strip().lower().startswith(_SCOPE_KERNEL_PREFIXES)
        for name in names
    )


def _sort_ids(values: set[str]) -> list[str]:
    return sorted(
        values, key=lambda value: (0, int(value)) if value.isdigit() else (1, value)
    )


def _node_label(node: dict | None) -> str | None:
    if not isinstance(node, dict):
        return None
    params = node.get("params") if isinstance(node.get("params"), dict) else {}
    kernel_params = (
        params.get("kernelParams")
        if isinstance(params.get("kernelParams"), dict)
        else {}
    )
    for value in (
        node.get("label"),
        node.get("name"),
        params.get("funcName"),
        params.get("func_name"),
        kernel_params.get("funcName"),
        kernel_params.get("func_name"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _kernel_node_ids(nodes: dict[str, dict]) -> set[str]:
    """Return every pre-update kernel node, including kernels that failed fusion.

    Final Scope membership is a result of fusion, so using it as the expected
    set would silently remove OP_UNSUPPORT and resource-failure records.  The
    pre-update graph is the stable universe; framework boundary kernels are
    removed separately because they are graph scaffolding rather than user
    operators.
    """
    return {
        str(node_id)
        for node_id, node in nodes.items()
        if isinstance(node, dict) and str(node.get("type", "")).upper() == "KERNEL"
    }


def _select_fusion_analysis(
    json_result: dict | None, log_result: dict | None, mode: str
) -> tuple[dict | None, str]:
    """Select fusion evidence using the same JSON-first rule as other facts."""
    candidates = (
        (json_result, log_result)
        if mode == "both"
        else (json_result,)
        if mode == "json"
        else (log_result,)
    )
    fallback: tuple[dict, str] | None = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        block = candidate.get("fusion_analysis")
        if not isinstance(block, dict):
            continue
        parser = str(candidate.get("parser") or "unavailable")
        if fallback is None:
            fallback = (block, parser)
        if str(block.get("status") or "unavailable") != "unavailable":
            return block, parser
    return fallback or (None, "unavailable")


def _explicit_not_in_scope_ids(
    fusion: dict | None, operator_kernel_ids: set[str]
) -> set[str]:
    """Return kernels explicitly marked outside the fusion scope.

    A kernel with an OP_UNSUPPORT or resource failure is still an expected
    operator.  Only a parser record that explicitly says NOT_IN_SCOPE may
    remove a kernel from the expected set.
    """
    if not isinstance(fusion, dict):
        return set()
    result: set[str] = set()
    for failure in fusion.get("node_failures") or []:
        if not isinstance(failure, dict) or failure.get("node_id") is None:
            continue
        node_id = str(failure.get("node_id"))
        if node_id not in operator_kernel_ids:
            continue
        fields = (failure.get("reason_code"), failure.get("reason"))
        if any(
            "NOT_IN_SCOPE"
            in str(value or "").upper().replace("-", "_").replace(" ", "_")
            or "OUT_OF_SCOPE"
            in str(value or "").upper().replace("-", "_").replace(" ", "_")
            for value in fields
        ):
            result.add(node_id)
    return result


def _annotate_fusion_operator_outcomes(
    fusion: dict,
    *,
    expected_ids: set[str] | None,
    actual_ids: set[str] | None,
    kernel_ids: set[str],
    pure_node_ids: set[str],
    structural_node_ids: set[str] | None = None,
    not_in_scope_ids: set[str] | None = None,
    before_by_id: dict[str, dict],
) -> None:
    """Attach the user-facing operator outcome without discarding raw failures.

    ``node_failures`` is a diagnostic stream and can contain kernels that were
    outside every named Scope.  The user result must instead be derived from
    the eligible set and the successful fusion membership.
    """
    if expected_ids is None or actual_ids is None:
        return

    expected = {str(node_id) for node_id in expected_ids}
    actual = {str(node_id) for node_id in actual_ids} & expected
    unfused = expected - actual
    structural_ids = {
        str(node_id) for node_id in (structural_node_ids or pure_node_ids)
    }
    explicit_not_in_scope = {str(node_id) for node_id in (not_in_scope_ids or set())}
    excluded_not_in_scope = explicit_not_in_scope & kernel_ids
    excluded_unknown = kernel_ids - expected - structural_ids - excluded_not_in_scope

    raw_failures = fusion.get("node_failures")
    raw_failures = raw_failures if isinstance(raw_failures, list) else []
    failures_by_node: dict[str, dict] = {}
    excluded_failure_counts: Counter[str] = Counter()
    for failure in raw_failures:
        if not isinstance(failure, dict) or failure.get("node_id") is None:
            continue
        node_id = str(failure.get("node_id"))
        failures_by_node.setdefault(node_id, failure)
        if node_id in expected:
            continue
        if node_id in structural_ids:
            excluded_failure_counts["structural"] += 1
        elif node_id in excluded_not_in_scope:
            excluded_failure_counts["not_in_scope"] += 1
        elif node_id in excluded_unknown:
            excluded_failure_counts["unknown"] += 1
        else:
            excluded_failure_counts["unknown"] += 1

    unfused_failures = []
    for node_id in _sort_ids(unfused):
        failure = failures_by_node.get(node_id)
        if failure is not None:
            unfused_failures.append(copy.deepcopy(failure))
            continue
        node = before_by_id.get(node_id, {})
        label = _node_label(node)
        unfused_failures.append(
            {
                "node_id": node_id,
                "reason": "融合记录缺失",
                "reason_code": "MISSING_FUSION_RECORD",
                "detail": "该算子处于有效融合范围内，但没有找到成功融合记录或失败原因。",
                "operator": {"normalized_name": label or "未命名算子"},
            }
        )

    fusion["operator_outcomes"] = {
        "status": "complete",
        "expected_node_count": len(expected),
        "actual_node_count": len(actual),
        "unfused_node_count": len(unfused),
        "unfused_node_ids": _sort_ids(unfused),
        "excluded_node_counts": {
            "out_of_scope": len(excluded_not_in_scope),
            "not_in_scope": len(excluded_not_in_scope),
            "structural": len(structural_ids),
        },
        "excluded_failure_counts": dict(excluded_failure_counts),
    }
    fusion["unfused_node_failures"] = unfused_failures


def _graph_semantics(
    before: dict, after: dict, scopes: list[dict], fusions: list[dict]
) -> dict:
    """Separate structural-only Scope ranges from real fusion outcomes.

    A single-member fusion remains a real updated SK.  Only a Scope made up
    entirely of framework boundary kernels is structural-only and eligible for
    hiding in the user graph.
    """
    before_by_id = {str(node.get("id")): node for node in before.get("nodes", [])}
    after_by_id = {str(node.get("id")): node for node in after.get("nodes", [])}
    pure_scope_ids: set[str] = set()
    pure_node_ids: set[str] = set()
    for scope in scopes or []:
        scope_id = str(scope.get("id"))
        ids = [str(value) for value in scope.get("node_ids", [])]
        if not ids:
            continue
        members = [
            before_by_id.get(node_id) or after_by_id.get(node_id) for node_id in ids
        ]
        if len(members) == len(ids) and all(
            _is_scope_kernel_node(node) for node in members
        ):
            pure_scope_ids.add(scope_id)
            pure_node_ids.update(ids)

    single_scope_ids: set[str] = set()
    single_node_ids: set[str] = set()
    multi_scope_ids: set[str] = set()
    for fusion in fusions or []:
        scope_id = str(fusion.get("scope_id"))
        ids = [str(value) for value in fusion.get("node_ids", [])]
        if len(ids) == 1:
            member = before_by_id.get(ids[0]) or after_by_id.get(ids[0])
            if (
                scope_id not in pure_scope_ids
                and isinstance(member, dict)
                and str(member.get("type", "")).upper() == "KERNEL"
            ):
                single_scope_ids.add(scope_id)
                single_node_ids.add(ids[0])
        elif len(ids) > 1 and scope_id not in pure_scope_ids:
            multi_scope_ids.add(scope_id)

    return {
        "pure_scope_kernel_scope_ids": _sort_ids(pure_scope_ids),
        "pure_scope_kernel_node_ids": _sort_ids(pure_node_ids),
        "single_kernel_fusion_scope_ids": _sort_ids(single_scope_ids),
        "single_kernel_fusion_node_ids": _sort_ids(single_node_ids),
        "multi_kernel_fusion_scope_ids": _sort_ids(multi_scope_ids),
    }


def resolve_model(
    key: str, label: str, json_result: dict | None, log_result: dict | None, mode: str
) -> dict:
    j = json_result or empty_result("json")
    log_data = log_result or empty_result("log")
    active = [r for r in (json_result, log_result) if r is not None]
    diagnostics = [copy.deepcopy(i) for r in active for i in r.get("diagnostics", [])]
    sources = {}
    coverage = {}
    out = {
        "key": key,
        "label": label,
        "instance_id": (log_result or json_result or {}).get("instance_id", "mi01"),
    }
    for capability in ("before", "after", "scopes", "fusions", "queues", "runtime"):

        def state(result, capability=capability):
            return (
                result[capability]["status"]
                if capability in ("before", "after")
                else result["capabilities"].get(capability, "unavailable")
            )

        js = state(j)
        chosen = (
            j
            if mode == "json"
            else log_data
            if mode == "log"
            else j
            if js != "unavailable"
            else log_data
        )
        source = chosen["parser"] if state(chosen) != "unavailable" else "unavailable"
        sources[capability] = source
        coverage[capability] = state(chosen)
        out[capability] = copy.deepcopy(chosen[capability])
        if capability == "after":
            _normalize_updated_labels(out[capability])
        if (
            mode == "both"
            and source == "log"
            and capability in ("before", "after", "queues")
        ):
            diagnostics.append(
                issue(
                    "source_fallback",
                    f"{capability} 的 JSON 不可用，采用 Log 结果",
                    impact="该能力不能完成双侧校验",
                    next_step="检查 JSON 资产及解析诊断",
                    details={"capability": capability, "selected": "log"},
                )
            )
        if source == "unavailable":
            diagnostics.append(
                issue(
                    "capability_unavailable",
                    f"{capability} 缺少可用证据",
                    severity="info",
                    impact="相关能力不可用，不代表数量为零或运行正常",
                    details={"capability": capability},
                )
            )
        elif state(chosen) == "partial":
            diagnostics.append(
                issue(
                    "partial_coverage",
                    f"{capability} 仅部分还原",
                    impact="展示已知事实，统计不代表完整执行图",
                    details={"capability": capability, "selected": source},
                )
            )
    if mode == "both":
        comparison = compare_results(j, log_data)
        if comparison["differences"]:
            diagnostics.append(
                issue(
                    "parser_mismatch",
                    f"JSON / Log 有 {comparison['difference_count']} 项差异，主结果采用有效 JSON",
                    impact="两侧提取结果不一致，运行异常状态不受此问题直接影响",
                    next_step="展开双方值和源位置，核查日志版本与输入是否属于同一次运行",
                    details=comparison["differences"],
                )
            )
        if comparison["single_sides"]:
            diagnostics.append(
                issue(
                    "single_side_objects",
                    f"{comparison['single_side_count']} 项信息仅存在于一侧",
                    severity="info",
                    impact="不作为校验一致；可能为格式覆盖差异或缺失证据",
                    details=comparison["single_sides"],
                )
            )
    else:
        comparison = {
            "status": "not_requested",
            "compared_fields": 0,
            "difference_count": 0,
            "single_side_count": 0,
            "differences": [],
            "single_sides": [],
            "not_comparable": [],
        }
    if mode in ("both", "log") and log_result is not None:
        _attach_event_origins(out, log_data)
    # Preserve explicit invalidation evidence omitted from the active JSON snapshot.
    # Never override a JSON node or infer INVALID from an absent node.
    if mode == "both" and sources["after"] == "json":
        present = {str(n["id"]) for n in out["after"]["nodes"]}
        invalidated = [
            copy.deepcopy(n)
            for n in log_data["after"]["nodes"]
            if n.get("type") == "INVALID" and str(n["id"]) not in present
        ]
        if invalidated:
            out["after"]["nodes"].extend(invalidated)
            diagnostics.append(
                issue(
                    "invalid_nodes_retained",
                    "保留 Log 明确标为 INVALID、JSON 活动图未列出的节点",
                    severity="info",
                    details={"count": len(invalidated), "source": "log"},
                )
            )
    b, a = out["before"], out["after"]
    if sources["after"] == "log" and a["status"] == "partial":
        # Log after is a change list, not a replacement snapshot. Keep baseline
        # nodes without an update record visible, without claiming their final
        # state was observed or copying old edges into the updated graph.
        present = {str(n["id"]) for n in a["nodes"]}
        baseline = [
            {**copy.deepcopy(n), "update_status": "not_recorded"}
            for n in b["nodes"]
            if str(n["id"]) not in present
        ]
        a["nodes"].extend(baseline)
        a["baseline_node_ids"] = [str(n["id"]) for n in baseline]
    members = {nid for f in out["fusions"] for nid in f["node_ids"]}
    bnodes = {str(n["id"]): n for n in b["nodes"]}
    fusion_result, fusion_parser = _select_fusion_analysis(
        json_result, log_result, mode
    )
    semantics = _graph_semantics(b, a, out["scopes"], out["fusions"])
    pure_scope_count = len(semantics["pure_scope_kernel_scope_ids"])
    single_fusion_count = len(semantics["single_kernel_fusion_scope_ids"])
    multi_fusion_count = len(semantics["multi_kernel_fusion_scope_ids"])
    pure_node_ids = set(semantics["pure_scope_kernel_node_ids"])
    kernel_ids = _kernel_node_ids(bnodes)
    structural_kernel_ids = {
        node_id for node_id in kernel_ids if _is_scope_kernel_node(bnodes.get(node_id))
    }
    operator_kernel_ids = kernel_ids - structural_kernel_ids
    not_in_scope_ids = _explicit_not_in_scope_ids(fusion_result, operator_kernel_ids)
    expected_operator_ids = operator_kernel_ids - not_in_scope_ids
    fused_kernel_ids = {
        str(node_id) for node_id in members if str(node_id) in operator_kernel_ids
    }
    actual_operator_ids = fused_kernel_ids & expected_operator_ids
    expected_operator_count = (
        len(expected_operator_ids)
        if coverage["scopes"] != "unavailable"
        and coverage["fusions"] != "unavailable"
        and b["status"] != "unavailable"
        else None
    )
    fused_operator_count = (
        len(actual_operator_ids)
        if coverage["scopes"] != "unavailable"
        and coverage["fusions"] != "unavailable"
        and b["status"] != "unavailable"
        else None
    )
    unfused_operator_count = (
        max(expected_operator_count - fused_operator_count, 0)
        if expected_operator_count is not None and fused_operator_count is not None
        else None
    )
    visible_after_node_count = (
        sum(
            1
            for node in a["nodes"]
            if str(node.get("type", "")).upper() != "INVALID"
            and str(node.get("id")) not in pure_node_ids
        )
        if a["status"] != "unavailable"
        else None
    )
    scope_count = len(out["scopes"]) if coverage["scopes"] != "unavailable" else None
    out["statistics"] = {
        "before_nodes": len(b["nodes"]) if b["status"] != "unavailable" else None,
        "after_nodes": len(a["nodes"]) if a["status"] != "unavailable" else None,
        # User Graph excludes explicit INVALID records and structural-only
        # Scope boundary kernels; the raw count above remains available for
        # machine-side evidence and debug reports.
        "visible_after_nodes": visible_after_node_count,
        "sk_nodes": sum(bool(n.get("is_sk")) for n in a["nodes"])
        if a["status"] != "unavailable"
        else None,
        "scope_count": scope_count,
        "visible_scope_count": scope_count - pure_scope_count
        if scope_count is not None
        else None,
        "fused_nodes": len(members) if coverage["fusions"] != "unavailable" else None,
        "fused_kernel_nodes": sum(
            bnodes.get(str(i), {}).get("type") == "KERNEL" for i in members
        )
        if coverage["fusions"] != "unavailable" and b["status"] != "unavailable"
        else None,
        "single_kernel_fusion_count": single_fusion_count
        if coverage["fusions"] != "unavailable"
        else None,
        "multi_kernel_fusion_count": multi_fusion_count
        if coverage["fusions"] != "unavailable"
        else None,
        "pure_scope_kernel_scope_count": pure_scope_count
        if coverage["scopes"] != "unavailable"
        else None,
        # User-facing fusion completion is based on real operator kernels.
        # Framework-only Scope kernels are deliberately excluded from the
        # expected total because they describe graph boundaries, not work.
        "fused_operator_count": fused_operator_count,
        "expected_operator_count": expected_operator_count,
        "unfused_operator_count": unfused_operator_count,
        "fused_sk_count": sum(bool(n.get("is_sk")) for n in a["nodes"])
        if a["status"] != "unavailable"
        else None,
    }
    from .options import resolve_options

    out["graph_options"] = resolve_options(j, log_data, mode, diagnostics)
    if fusion_result is not None:
        out["fusion_analysis"] = copy.deepcopy(fusion_result)
        _annotate_fusion_operator_outcomes(
            out["fusion_analysis"],
            expected_ids=expected_operator_ids
            if expected_operator_count is not None
            else None,
            actual_ids=actual_operator_ids
            if fused_operator_count is not None
            else None,
            kernel_ids=kernel_ids,
            pure_node_ids=pure_node_ids,
            structural_node_ids=structural_kernel_ids,
            not_in_scope_ids=not_in_scope_ids,
            before_by_id=bnodes,
        )
        sources["fusion_analysis"] = fusion_parser
        coverage["fusion_analysis"] = str(fusion_result.get("status") or "unavailable")
    out.update(
        sources=sources,
        coverage=coverage,
        comparison=comparison,
        diagnostics=diagnostics,
        graph_semantics=semantics,
        transformations=_transformations(b, a, out["fusions"]),
    )
    from .user_scopes import resolve_user_scopes

    out["user_scopes"] = resolve_user_scopes(json_result, log_result, mode, out)
    return out
