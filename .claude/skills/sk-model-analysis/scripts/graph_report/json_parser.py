"""Strict, independent JSON graph reader; never consults runtime logs.

Synchronization edges are inferred only for a unique producer and consumer for
an event, or an exact memory address/value pair. Reused signals are evidence of
ambiguity, not permission to draw every possible dependency.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

_NULL_POINTERS = (None, "none", "null", "")


def _identity(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    return str(value) if str(value).strip() else None


def _number(value):
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _signal(value):
    value = _identity(value)
    if value is None:
        return None
    try:
        return str(int(value, 16 if value.lower().startswith("0x") else 10))
    except ValueError:
        return value


class _Reader:
    def __init__(self, model_dir):
        self.path = Path(model_dir)
        self.issues = []
        self.sources = []
        self.model_id = (
            self.path.name.removeprefix("model_")
            if self.path.name.startswith("model_")
            else None
        )

    def issue(self, code, message, file, path="$", stage=None, node=None, details=None):
        source = {"file": str(file), "path": path}
        target = {}
        if stage:
            target["stage"] = stage
        if node is not None:
            target["node_id"] = str(node)
        identity = json.dumps([code, source, target], sort_keys=True)
        self.issues.append(
            {
                "id": "json-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                "domain": "parsing",
                "severity": "warning",
                "code": code,
                "message": message,
                "impact": "该来源的数据覆盖或依赖关系确定性受限。",
                "next_step": "检查引用的 JSON 字段，必要时重新采集对应模型的数据。",
                "source": source,
                "object": target,
                "details": details or {},
            }
        )

    def read(self, name):
        file = self.path / name
        try:
            content = file.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            self.issue("invalid_encoding", str(exc), file)
            return None, file
        except OSError as exc:
            self.issue(
                "missing_source"
                if isinstance(exc, FileNotFoundError)
                else "unreadable_source",
                str(exc),
                file,
            )
            return None, file
        self.sources.append(str(file))
        try:
            data = json.loads(
                content,
                parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError("Nonfinite JSON number: " + x)
                ),
            )
        except (ValueError, UnicodeError) as exc:
            self.issue("invalid_json", str(exc), file)
            return None, file
        if not isinstance(data, dict):
            self.issue("invalid_schema", "预期为 JSON 对象。", file)
            return None, file
        if data.get("version") != "1.0":
            self.issue(
                "unsupported_version",
                "当前仅支持 1.0 版本的 JSON 数据结构。",
                file,
                "$.version",
                details={"actual": data.get("version")},
            )
            return None, file
        mid = _identity(data.get("modelId"))
        if mid is None:
            self.issue(
                "invalid_model_identity",
                "modelId 缺失或无效。",
                file,
                "$.modelId",
            )
            return None, file
        if self.model_id is None:
            self.model_id = mid
        if mid != self.model_id:
            self.issue(
                "model_identity_mismatch",
                "JSON 属于其他模型。",
                file,
                "$.modelId",
                details={"expected": self.model_id, "actual": mid},
            )
            return None, file
        return data, file

    def count(self, data, key, actual, file, path, stage=None):
        if key in data and (_number(data[key]) is None or data[key] != actual):
            self.issue(
                "count_mismatch",
                f"{key} 与实际提取数量不一致。",
                file,
                path + "." + key,
                stage,
                details={"declared": data[key], "actual": actual},
            )

    def snapshot(self, name, stage):
        data, file = self.read(name)
        snapshot = {
            "status": "unavailable",
            "nodes": [],
            "edges": [],
            "source": str(file),
        }
        if data is None:
            return snapshot
        start = len(self.issues)
        snapshot["options"] = {}

        def collect_options(rows, prefix=""):
            if not isinstance(rows, dict):
                self.issue(
                    "invalid_graph_options",
                    "options 必须为对象。",
                    file,
                    "$.options",
                    stage,
                )
                return
            for key, row in rows.items():
                name = prefix + key
                if key == "inner_options":
                    collect_options(row, name + ".")
                elif isinstance(row, dict) and "value" in row:
                    snapshot["options"][name] = {
                        **row,
                        "source": {"file": str(file), "path": "$.options." + name},
                    }
                else:
                    self.issue(
                        "invalid_graph_options",
                        "配置项缺少 value 字段。",
                        file,
                        "$.options." + name,
                        stage,
                    )

        if "options" in data:
            collect_options(data["options"])
        streams = data.get("streams")
        if not isinstance(streams, list):
            self.issue(
                "invalid_schema", "streams 必须为数组。", file, "$.streams", stage
            )
            return snapshot
        nodes, raw_nodes = {}, {}
        stream_indices = set()
        for si, stream in enumerate(streams):
            sp = f"$.streams[{si}]"
            if not isinstance(stream, dict) or not isinstance(
                stream.get("nodes"), list
            ):
                self.issue(
                    "invalid_stream",
                    "Stream 必须包含 nodes 数组。",
                    file,
                    sp,
                    stage,
                )
                continue
            stream_index = _number(stream.get("streamIdxInGraph"))
            if stream_index is None or stream_index in stream_indices:
                self.issue(
                    "invalid_stream_index",
                    "Stream 索引缺失、无效或重复。",
                    file,
                    sp,
                    stage,
                )
                stream_index = si
            stream_indices.add(stream_index)
            for ni, raw in enumerate(stream["nodes"]):
                np = sp + f".nodes[{ni}]"
                if (
                    not isinstance(raw, dict)
                    or _identity(raw.get("nodeId")) is None
                    or not isinstance(raw.get("nodeType"), str)
                ):
                    self.issue(
                        "invalid_node",
                        "节点必须包含标量 nodeId 和字符串 nodeType。",
                        file,
                        np,
                        stage,
                    )
                    continue
                nid = _identity(raw["nodeId"])
                if nid in nodes:
                    self.issue(
                        "duplicate_node_id",
                        "节点 ID 重复；保留首次出现的数据。",
                        file,
                        np,
                        stage,
                        nid,
                    )
                    continue
                tasks = raw.get("taskInfo", [])
                if not isinstance(tasks, list) or any(
                    not isinstance(t, dict) for t in tasks
                ):
                    self.issue(
                        "invalid_task_info",
                        "taskInfo 必须为对象数组。",
                        file,
                        np + ".taskInfo",
                        stage,
                        nid,
                    )
                    tasks = []
                task = tasks[0] if tasks else {}
                for key in (
                    "kernelParams",
                    "eventParams",
                    "valueWriteParams",
                    "valueWaitParams",
                ):
                    if key in task and not isinstance(task[key], dict):
                        self.issue(
                            "invalid_task_params",
                            key + " 必须为对象。",
                            file,
                            np + ".taskInfo",
                            stage,
                            nid,
                        )
                        task = {**task, key: {}}
                skinfos = raw.get("skinfos")
                scope_id = (
                    _identity(skinfos.get("scopeId"))
                    if isinstance(skinfos, dict)
                    else None
                )
                if skinfos is not None and scope_id is None:
                    self.issue(
                        "invalid_sk_info",
                        "skinfos 必须包含标量 scopeId。",
                        file,
                        np + ".skinfos",
                        stage,
                        nid,
                    )
                order = _number(raw.get("nodeIdxInStream"))
                if (
                    order is None
                    or raw.get("streamIdxInGraph", stream_index) != stream_index
                ):
                    self.issue(
                        "invalid_node_position",
                        "节点顺序无效或 Stream 索引不匹配。",
                        file,
                        np,
                        stage,
                        nid,
                    )
                typ = raw["nodeType"]
                label = task.get("kernelParams", {}).get("funcName") or typ + " " + nid
                node = {
                    "id": nid,
                    "stream": _identity(task.get("streamId")) or str(stream_index),
                    "stream_index": stream_index,
                    "order": ni if order is None else order,
                    "type": typ,
                    "label": str(label),
                    "scope_id": scope_id,
                    "is_sk": scope_id is not None,
                    "params": {
                        **task,
                        **({"skinfos": skinfos} if skinfos is not None else {}),
                    },
                    "source": {"file": str(file), "path": np},
                }
                nodes[nid], raw_nodes[nid] = node, raw
            self.count(stream, "nodeCount", len(stream["nodes"]), file, sp, stage)
        edges = snapshot["edges"]
        for nid, raw in raw_nodes.items():
            node = nodes[nid]
            for field in ("preNodeId", "nextNodeId"):
                pointer = raw.get(field)
                if pointer in _NULL_POINTERS:
                    continue
                target = _identity(pointer)
                if target not in nodes:
                    self.issue(
                        "dangling_pointer",
                        f"{field} 无法关联到节点。",
                        file,
                        node["source"]["path"] + "." + field,
                        stage,
                        nid,
                        {"target": pointer},
                    )
                    continue
                other = nodes[target]
                if (
                    target == nid
                    or other["stream_index"] != node["stream_index"]
                    or (field == "nextNodeId" and other["order"] <= node["order"])
                    or (field == "preNodeId" and other["order"] >= node["order"])
                ):
                    self.issue(
                        "invalid_pointer",
                        "节点指针形成自环、指向前序位置或跨越 Stream。",
                        file,
                        node["source"]["path"] + "." + field,
                        stage,
                        nid,
                    )
                    continue
                if (
                    field == "preNodeId"
                    and "nextNodeId" in raw_nodes[target]
                    and _identity(raw_nodes[target]["nextNodeId"]) != nid
                ):
                    self.issue(
                        "inconsistent_pointer",
                        "preNodeId 与 nextNodeId 不一致。",
                        file,
                        node["source"]["path"],
                        stage,
                        nid,
                    )
                if field == "nextNodeId":
                    reverse = raw_nodes[target].get("preNodeId")
                    if "preNodeId" in raw_nodes[target] and _identity(reverse) != nid:
                        self.issue(
                            "inconsistent_pointer",
                            "nextNodeId 与 preNodeId 不一致。",
                            file,
                            node["source"]["path"],
                            stage,
                            nid,
                        )
                        continue
                    edges.append(
                        {
                            "source": nid,
                            "target": target,
                            "kind": "stream",
                            "evidence": "explicit",
                        }
                    )
        self.synchronization(nodes.values(), edges, file, stage)
        self.count(data, "totalNodes", len(nodes), file, "$", stage)
        self.count(data, "totalStreams", len(streams), file, "$", stage)
        from graph_report.user_scopes import explicit_user_scopes

        snapshot.update(
            nodes=list(nodes.values()),
            user_scopes=explicit_user_scopes(raw_nodes, nodes),
            status="partial" if len(self.issues) > start else "complete",
        )
        return snapshot

    def synchronization(self, nodes, edges, file, stage):
        signals = defaultdict(lambda: [[], []])
        resets = defaultdict(list)
        memory_writes = defaultdict(list)
        for node in nodes:
            typ, params = node["type"], node["params"]
            if typ == "RESET":
                event = _signal(params.get("eventParams", {}).get("eventId"))
                if event is not None:
                    resets[event].append(node)
            elif typ in ("NOTIFY", "WAIT"):
                event = _signal(params.get("eventParams", {}).get("eventId"))
                if event is not None:
                    signals[("event", event, None)][typ == "WAIT"].append(node)
            elif typ in ("MEMORY_WRITE", "MEMORY_WAIT"):
                p = params.get(
                    "valueWriteParams" if typ == "MEMORY_WRITE" else "valueWaitParams",
                    {},
                )
                if typ == "MEMORY_WAIT" and p.get("flag") != 1:
                    self.issue(
                        "unsupported_memory_predicate",
                        "该内存等待标志不支持精确值配对。",
                        file,
                        node["source"]["path"],
                        stage,
                        node["id"],
                        {"flag": p.get("flag")},
                    )
                    continue
                address, value = _signal(p.get("devAddr")), _signal(p.get("value"))
                if address is not None and typ == "MEMORY_WRITE":
                    memory_writes[address].append(node)
                if address is not None and value is not None:
                    signals[("memory", address, value)][typ == "MEMORY_WAIT"].append(
                        node
                    )
        for (kind, signal, value), (writers, readers) in signals.items():
            if not readers:
                continue
            if len(writers) == len(readers) == 1:
                a, b = writers[0], readers[0]
                if a["stream_index"] == b["stream_index"] and a["order"] >= b["order"]:
                    self.issue(
                        "ambiguous_synchronization",
                        "同一 Stream 中信号生产者位于消费者之后。",
                        file,
                        b["source"]["path"],
                        stage,
                        b["id"],
                    )
                    continue
                competitors = (
                    resets[signal]
                    if kind == "event"
                    else [n for n in memory_writes[signal] if n["id"] != a["id"]]
                )
                unsafe = [
                    n
                    for n in competitors
                    if not (
                        (
                            n["stream_index"] == a["stream_index"]
                            and 0 <= n["order"] < a["order"]
                        )
                        or (
                            n["stream_index"] == b["stream_index"]
                            and n["order"] > b["order"] >= 0
                        )
                    )
                ]
                if unsafe:
                    self.issue(
                        "ambiguous_synchronization",
                        "无法证明 Reset 或竞争写入位于生产者与消费者区间之外。",
                        file,
                        b["source"]["path"],
                        stage,
                        b["id"],
                        {
                            "kind": kind,
                            "signal": signal,
                            "competing_node_ids": [n["id"] for n in unsafe],
                        },
                    )
                    continue
                edge = {
                    "source": a["id"],
                    "target": b["id"],
                    "kind": kind,
                    "evidence": "inferred",
                }
                edge.update(
                    {"event": signal}
                    if kind == "event"
                    else {"address": signal, "value": value}
                )
                edges.append(edge)
            else:
                self.issue(
                    "ambiguous_synchronization"
                    if writers
                    else "unresolved_synchronization",
                    "信号无法唯一匹配生产者与消费者。",
                    file,
                    readers[0]["source"]["path"],
                    stage,
                    readers[0]["id"],
                    {
                        "kind": kind,
                        "signal": signal,
                        "value": value,
                        "producer_ids": [n["id"] for n in writers],
                        "consumer_ids": [n["id"] for n in readers],
                    },
                )

        from graph_report.synchronization import recover_pairs

        recovered = recover_pairs(nodes, edges)
        edges.extend(recovered)
        accepted = {e["target"] for e in recovered}
        self.issues[:] = [
            i
            for i in self.issues
            if not (
                i.get("object", {}).get("stage") == stage
                and i.get("code")
                in {"ambiguous_synchronization", "unresolved_synchronization"}
                and str(i.get("object", {}).get("node_id")) in accepted
            )
        ]

    def queues(self, known_scopes):
        data, file = self.read("sk_task_queue.json")
        if data is None:
            return [], "unavailable"
        scopes = data.get("scopes")
        if not isinstance(scopes, list):
            self.issue("invalid_schema", "scopes 必须为数组。", file, "$.scopes")
            return [], "unavailable"
        start, queues, seen = len(self.issues), [], set()
        self.count(data, "scopeCount", len(scopes), file, "$")
        for si, scope in enumerate(scopes):
            sp = f"$.scopes[{si}]"
            if (
                not isinstance(scope, dict)
                or _identity(scope.get("scopeId")) is None
                or not isinstance(scope.get("taskQueues"), dict)
            ):
                self.issue(
                    "invalid_queue_scope",
                    "Scope 必须包含 scopeId 和 taskQueues 对象。",
                    file,
                    sp,
                )
                continue
            sid = _identity(scope["scopeId"])
            if sid in seen:
                self.issue(
                    "duplicate_queue_scope",
                    "Scope 队列重复；保留首次出现的数据。",
                    file,
                    sp,
                )
                continue
            seen.add(sid)
            bound = sid if sid in known_scopes else None
            if bound is None:
                self.issue(
                    "unbound_queue",
                    "队列所属 Scope 没有匹配的更新后 SK 节点。",
                    file,
                    sp,
                    details={"scope_id": sid},
                )
            for kind, raw in scope["taskQueues"].items():
                qp = sp + ".taskQueues." + kind
                if (
                    kind not in ("aic", "aiv")
                    or not isinstance(raw, dict)
                    or not isinstance(raw.get("taskQue"), dict)
                    or not isinstance(raw["taskQue"].get("taskInfos"), list)
                ):
                    self.issue(
                        "invalid_queue",
                        "队列必须包含已知类型和 taskQue.taskInfos 数组。",
                        file,
                        qp,
                    )
                    continue
                tasks = []
                for ti, task in enumerate(raw["taskQue"]["taskInfos"]):
                    tp = qp + f".taskQue.taskInfos[{ti}]"
                    if (
                        not isinstance(task, dict)
                        or not isinstance(task.get("type"), str)
                        or not isinstance(task.get("entries", []), list)
                        or any(not isinstance(e, str) for e in task.get("entries", []))
                    ):
                        self.issue(
                            "invalid_queue_task",
                            "Task 必须包含 type 和由字符串组成的 entries 数组。",
                            file,
                            tp,
                        )
                        continue
                    self.count(task, "entryCnt", len(task.get("entries", [])), file, tp)
                    for key in ("nodeIndex", "numBlocks", "debugOptions"):
                        if key in task and _number(task[key]) is None:
                            self.issue(
                                "invalid_queue_task",
                                key + " 必须为非负整数。",
                                file,
                                tp + "." + key,
                            )
                    tasks.append(
                        {
                            "index": ti,
                            "node_index": _number(task.get("nodeIndex")),
                            "type": task["type"],
                            "num_blocks": _number(task.get("numBlocks")),
                            "entries": task.get("entries", []),
                            "args": str(task["args"])
                            if task.get("args") is not None
                            else None,
                            "debug_options": _number(task.get("debugOptions")),
                            "source": {"file": str(file), "path": tp},
                        }
                    )
                self.count(raw["taskQue"], "taskCnt", len(tasks), file, qp + ".taskQue")
                queues.append(
                    {
                        "scope_id": bound,
                        "declared_scope_id": sid,
                        "id": sid + ":" + kind.upper(),
                        "kind": kind.upper(),
                        "tasks": tasks,
                        "source": {"file": str(file), "path": qp},
                    }
                )
        return queues, "partial" if len(self.issues) > start else "complete"


def parse_model(model_dir: Path) -> dict:
    """Read the three supported JSON files, retaining usable partial evidence."""
    reader = _Reader(model_dir)
    before = reader.snapshot("sk_graph_origin.json", "before")
    after = reader.snapshot("sk_graph_updated.json", "after")
    queues, queue_status = reader.queues(
        {n["scope_id"] for n in after["nodes"] if n["is_sk"]}
    )
    return {
        "parser": "json",
        "instance_id": "mi01",
        "model_id": reader.model_id,
        "before": before,
        "after": after,
        "user_scopes": list(before.get("user_scopes", [])),
        "scopes": [],
        "fusions": [],
        "queues": queues,
        "runtime": {
            "status": "insufficient",
            "conclusion": "JSON 执行图文件不包含运行诊断证据。",
            "issues": [],
            "evidence": {},
        },
        "diagnostics": reader.issues,
        "capabilities": {
            "scopes": "unavailable",
            "fusions": "unavailable",
            "queues": queue_status,
            "runtime": "unavailable",
        },
        "source_files": reader.sources,
    }
