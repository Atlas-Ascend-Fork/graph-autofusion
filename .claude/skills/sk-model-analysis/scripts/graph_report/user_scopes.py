"""User-authored graph annotations, independent of optimizer SK boundaries."""

import copy
import re
from collections import Counter


def explicit_user_scopes(
    raw_nodes: dict[str, dict], nodes: dict[str, dict]
) -> list[dict]:
    """Extract scopes explicitly encoded by the graph producer.

    A scope is a user annotation only when the JSON marks its boundary or
    carries a non-empty scope name.  Optimizer ``skinfos`` and inferred
    fusion membership are deliberately ignored here.  Membership is scoped
    per stream and ordered by the producer's node index; this avoids using
    global node IDs as a proxy for execution order.
    """
    markers = []
    for node_id, raw in raw_nodes.items():
        name = str(raw.get("scopeName") or "").strip()
        begin = raw.get("isScopeBegin") is True
        end = raw.get("isScopeEnd") is True
        placeholder = raw.get("isScopePlaceholder") is True
        marked = (
            raw.get("isScopeNode") is True or begin or end or placeholder or bool(name)
        )
        if not marked:
            continue
        node = nodes.get(node_id, {})
        markers.append(
            {
                "id": str(node_id),
                "name": name or "未命名 Scope",
                "stream": str(
                    node.get("stream_index", raw.get("streamIdxInGraph", "?"))
                ),
                "order": node.get("order", raw.get("nodeIdxInStream", 0)),
                "begin": begin,
                "end": end,
                "placeholder": placeholder,
            }
        )
    if not markers:
        return []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for marker in markers:
        grouped.setdefault((marker["stream"], marker["name"]), []).append(marker)
    result = []
    ordinal = 0
    for (stream, name), rows in grouped.items():
        rows.sort(key=lambda row: (row["order"], row["id"]))
        stack = []
        for marker in rows:
            if marker["begin"]:
                stack.append(marker)
                continue
            if not marker["end"] or not stack:
                continue
            begin = stack.pop()
            member_ids = [
                str(node["id"])
                for node in nodes.values()
                if str(node.get("stream_index", node.get("stream"))) == stream
                and begin["order"] <= node.get("order", -1) <= marker["order"]
            ]
            result.append(
                {
                    "id": f"user:{name}:{begin['id']}:{marker['id']}:{ordinal}",
                    "name": name,
                    "node_ids": member_ids,
                    "marker_node_ids": [begin["id"], marker["id"]],
                    "streams": [stream],
                    "source": "json",
                    "status": "explicit",
                }
            )
            ordinal += 1
    return result


def log_user_scopes(nodes, path, library, span=None):
    """Read explicit IsScopeKernel markers within the current Model's span.

    Final SK scopes supply only the documented name/flag association. They
    never create user annotations: a matching begin/end pair is required.
    """
    if not path.is_file():
        return []
    by_id = {str(n["id"]): n for n in nodes}
    raw, pending = {}, {}
    marker_pattern = re.compile(
        r"scopeName:\s*([^,]+), isBegin:\s*([01]), isEnd:\s*([01]), isPlaceholder:\s*([01])"
    )
    with path.open(errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            if span and not span["begin_line"] <= line_no <= span["end_line"]:
                continue
            thread = re.search(r"\[(\d+:\d+)\]", line)
            if not thread:
                continue
            if "[IsScopeKernel]" in line:
                match = marker_pattern.search(line)
                pending.pop(thread[1], None)
                if match:
                    pending[thread[1]] = {
                        "scopeName": match[1].strip(),
                        "isScopeBegin": match[2] == "1",
                        "isScopeEnd": match[3] == "1",
                        "isScopePlaceholder": match[4] == "1",
                        "isScopeNode": True,
                    }
            elif "[InitNode]" in line:
                match = re.search(r"Kernel node (\d+) is a scope kernel node", line)
                annotation = pending.pop(thread[1], None)
                if match and annotation and match[1] in by_id:
                    raw[match[1]] = annotation
    scopes = explicit_user_scopes(raw, by_id)
    flags_by_name = {}
    for scope in library.get("scopes", []):
        name = str(scope.get("scope_names") or "").strip().strip("[]")
        flag = str(scope.get("scope_bit_flags") or "")
        if not name or name.lower() == "none" or "," in name or "|" in name:
            continue
        try:
            value = int(flag, 16)
        except ValueError:
            continue
        if value and value & (value - 1) == 0:
            flags_by_name.setdefault(name, set()).add(value)
    names = Counter(s["name"] for s in scopes)
    for scope in scopes:
        scope["source"] = "log"
        flags = flags_by_name.get(scope["name"], set())
        # Repeated same-name ranges may reuse flags; do not merge them.
        if len(flags) != 1 or names[scope["name"]] != 1:
            continue
        bit = next(iter(flags))
        members = set(scope["node_ids"])
        for node in nodes:
            match = re.search(
                r"\bscopeFlags=(0x[\da-fA-F]+|[\da-fA-F]+)\b",
                str(node.get("params", {}).get("detail", "")),
            )
            if match and int(match[1], 16) & bit:
                members.add(str(node["id"]))
        scope["node_ids"] = sorted(members)
        scope["streams"] = sorted(
            {str(by_id[n].get("stream_index")) for n in members if n in by_id}
        )
    return scopes


def resolve_user_scopes(json_result, log_result, mode, model):
    json_result, log_result = json_result or {}, log_result or {}
    json_available = json_result.get("before", {}).get(
        "status", "unavailable"
    ) != "unavailable" or bool(json_result.get("user_scopes"))
    source = (
        json_result
        if mode == "json" or mode == "both" and json_available
        else log_result
    )
    scopes = copy.deepcopy(source.get("user_scopes", []))
    logged = log_result.get("user_scopes", []) if mode == "both" else []
    before = {str(n["id"]) for n in model.get("before", {}).get("nodes", [])}
    after = {str(n["id"]) for n in model.get("after", {}).get("nodes", [])}
    for scope in scopes:
        matches = [
            s
            for s in logged
            if s["name"] == scope["name"]
            and set(s.get("marker_node_ids", []))
            == set(scope.get("marker_node_ids", []))
        ]
        if len(matches) == 1:
            scope["node_ids"] = sorted(
                set(scope["node_ids"]) | set(matches[0]["node_ids"])
            )
            scope["source"] = "both"
        ids = set(scope["node_ids"]) & before
        scope["node_ids"] = sorted(ids)
        targets = ids & after
        for change in model.get("transformations", []):
            origins = set(map(str, change.get("before_ids", [])))
            if origins and origins <= ids:
                targets.update(set(map(str, change.get("after_ids", []))) & after)
        scope["after_node_ids"] = sorted(targets)
    return scopes
