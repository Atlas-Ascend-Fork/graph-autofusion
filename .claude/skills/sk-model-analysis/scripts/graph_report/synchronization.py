"""Signal facts and conservative recovery using already established graph paths."""

from collections import defaultdict


def signal_fields(node):
    p = node.get("params", {})
    event = p.get("eventParams", {})
    value = p.get("valueWaitParams") or p.get("valueWriteParams") or {}
    return {
        "event_id": event.get("eventId", p.get("event_id")),
        "event_flag": event.get("eventFlag", p.get("event_flag")),
        "event_task_flag": event.get("eventTaskFlag", p.get("event_task_flag")),
        "address": value.get("devAddr", p.get("addr")),
        "value": value.get("value", p.get("value")),
        "wait_flag": value.get("flag", p.get("flag")),
        "source": node.get("source", {}),
    }


def normalized(value):
    if value is None:
        return None
    try:
        return str(int(str(value), 16 if str(value).lower().startswith("0x") else 10))
    except ValueError:
        return str(value)


def recover_pairs(nodes, edges):
    """Never use a pending edge to establish its own Reset safety proof."""
    groups = defaultdict(lambda: [[], [], []])
    for n in nodes:
        f = signal_fields(n)
        typ = n["type"]
        if typ in ("NOTIFY", "WAIT", "RESET") and f["event_id"] is not None:
            groups[("event", normalized(f["event_id"]))][
                ("NOTIFY", "WAIT", "RESET").index(typ)
            ].append(n)
        elif typ in ("MEMORY_WRITE", "MEMORY_WAIT") and f["address"] is not None:
            groups[("memory", normalized(f["address"]))][typ == "MEMORY_WAIT"].append(n)
    graph = defaultdict(set)
    for e in edges:
        graph[str(e["source"])].add(str(e["target"]))

    def reaches(a, b):
        seen, todo = set(), [str(a)]
        while todo:
            n = todo.pop()
            if n == str(b):
                return True
            if n not in seen:
                seen.add(n)
                todo.extend(graph[n] - seen)
        return False

    pending = []
    existing = {(str(e["source"]), str(e["target"]), e["kind"]) for e in edges}
    for (kind, signal), (writers, readers, resets) in groups.items():
        for reader in readers:
            f = signal_fields(reader)
            if kind == "memory" and normalized(f["wait_flag"]) != "1":
                continue
            value = normalized(f["value"])
            matches = (
                writers
                if kind == "event"
                else [
                    w
                    for w in writers
                    if value is not None
                    and normalized(signal_fields(w)["value"]) == value
                ]
            )
            peers = (
                readers
                if kind == "event"
                else [
                    r for r in readers if normalized(signal_fields(r)["value"]) == value
                ]
            )
            if len(matches) != 1 or len(peers) != 1:
                continue
            writer = matches[0]
            same_stream = writer.get(
                "stream", writer.get("stream_index")
            ) == reader.get("stream", reader.get("stream_index"))
            if (
                same_stream
                and isinstance(writer.get("order"), int)
                and isinstance(reader.get("order"), int)
                and writer["order"] >= reader["order"]
            ):
                continue
            a, b = matches[0]["id"], reader["id"]
            if (str(a), str(b), kind) in existing or reaches(b, a):
                continue
            competitors = (
                resets if kind == "event" else [w for w in writers if w["id"] != a]
            )
            if not all(reaches(n["id"], a) or reaches(b, n["id"]) for n in competitors):
                continue
            pending.append(
                dict(
                    source=str(a),
                    target=str(b),
                    kind=kind,
                    evidence="inferred",
                    basis="existing_dependency_path",
                    **(
                        {"event": signal}
                        if kind == "event"
                        else {"address": signal, "value": value}
                    ),
                )
            )
    accepted = []
    for e in pending:
        if not reaches(e["target"], e["source"]):
            accepted.append(e)
            graph[e["source"]].add(e["target"])
    return accepted
