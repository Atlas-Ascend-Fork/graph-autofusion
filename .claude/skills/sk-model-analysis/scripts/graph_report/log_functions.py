"""Resolved FUNC evidence bound by instance, thread, explicit node ID and name."""

import re
from collections import defaultdict


def enrich_log_functions(nodes, path, span=None):
    if not path.is_file():
        return
    by_id = {str(n["id"]): n for n in nodes}
    contexts, records = {}, defaultdict(list)
    with path.open(errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if span and not span["begin_line"] <= line_no <= span["end_line"]:
                continue
            thread = re.search(r"\[(\d+:\d+)\]", line)
            if not thread:
                continue
            key = thread[1]
            node = re.search(
                r"\[InitNode\] Kernel node (\d+).*?funcName=([^,\s]+)", line
            )
            if node:
                contexts[key] = {
                    "id": node[1],
                    "name": node[2],
                    "verified": False,
                    "rows": [],
                }
                records[node[1]].append(contexts[key])
            elif "[InitNode]" in line:
                contexts.pop(key, None)
            context = contexts.get(key)
            if not context or "[InitKernelResolvedFuncs]" not in line:
                continue
            name = re.search(r"funcName=([^,\s]+)", line)
            if name:
                context["verified"] = name[1] == context["name"]
            split = re.search(
                r"split\[(\d+)\] funcAddr\[0\]=(0x[\da-fA-F]+), funcAddr\[1\]=(0x[\da-fA-F]+)",
                line,
            )
            if split and context["verified"]:
                context["rows"].append(
                    {
                        "funcAddr": [split[2], split[3]],
                        "source": {"file": str(path), "line": line_no},
                    }
                )
    for nid, batches in records.items():
        n = by_id.get(nid)
        if not n:
            continue
        batches = [b for b in batches if b["rows"] and b["name"] == n.get("label")]
        signatures = {
            tuple(tuple(row["funcAddr"]) for row in b["rows"]) for b in batches
        }
        if len(signatures) != 1:
            continue
        n["params"]["kernelParams"] = {
            "funcName": n["label"],
            "resolvedFuncs": batches[0]["rows"],
        }
