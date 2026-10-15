#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the stable model-level fork, event, and join graph for multistream tuning."""

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "superkernel-multistream-logical-graph-v1"
HARD_DEPENDENCY_KINDS = {
    "DATA",
    "STREAM_ORDER",
    "EVENT",
    "WAIT",
    "BARRIER",
    "COMMUNICATION",
    "CACHE_MUTATION",
    "SIDE_EFFECT",
    "CONTROL_FLOW",
}
STAGE_BLOCKERS = {
    "COMMUNICATION",
    "BARRIER",
    "CACHE_MUTATION",
    "SIDE_EFFECT",
    "RANDOM_STATE",
}


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _identifier_list(value, label, *, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label} must be a{' non-empty' if nonempty else ''} list")
    result = [_text(item, f"{label}[]") for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicates")
    return result


def _topological_check(stage_ids, edges):
    outgoing = {stage_id: [] for stage_id in stage_ids}
    incoming = {stage_id: 0 for stage_id in stage_ids}
    for before, after in edges:
        outgoing[before].append(after)
        incoming[after] += 1
    ready = sorted(stage_id for stage_id, count in incoming.items() if count == 0)
    visited = []
    while ready:
        current = ready.pop(0)
        visited.append(current)
        for successor in sorted(outgoing[current]):
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
        ready.sort()
    if len(visited) != len(stage_ids):
        raise ValueError("logical graph hard_dependencies contain a cycle")


def validate(value, *, require_fingerprint=True):
    required = {
        "schema_version",
        "graph_id",
        "request_fingerprint",
        "stages",
        "event_edges",
        "forks",
        "joins",
        "hard_dependencies",
        "dependency_evidence_fingerprint",
    }
    if require_fingerprint:
        required.add("graph_fingerprint")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"logical graph must contain exactly {sorted(required)}")
    if value["schema_version"] != SCHEMA:
        raise ValueError(f"logical graph must use {SCHEMA}")

    stages = value["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("logical graph stages must be a non-empty list")
    normalized_stages = []
    stage_ids = set()
    for index, item in enumerate(stages):
        fields = {
            "stage_id",
            "stream_role",
            "stream_reliable",
            "source_statement_ids",
            "movable",
            "blocked_effects",
        }
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"stages[{index}] fields are invalid")
        stage_id = _text(item["stage_id"], f"stages[{index}].stage_id")
        if stage_id in stage_ids:
            raise ValueError(f"duplicate stage_id: {stage_id}")
        stage_ids.add(stage_id)
        if not isinstance(item["movable"], bool):
            raise ValueError(f"stages[{index}].movable must be boolean")
        if not isinstance(item["stream_reliable"], bool):
            raise ValueError(f"stages[{index}].stream_reliable must be boolean")
        blocked_effects = _identifier_list(
            item["blocked_effects"], f"stages[{index}].blocked_effects", nonempty=False
        )
        if set(blocked_effects) - STAGE_BLOCKERS:
            raise ValueError(
                f"stages[{index}].blocked_effects contains an unknown blocker"
            )
        if item["movable"] and blocked_effects:
            raise ValueError(f"stages[{index}] cannot be movable with blocked effects")
        normalized_stages.append(
            {
                "stage_id": stage_id,
                "stream_role": _text(
                    item["stream_role"], f"stages[{index}].stream_role"
                ),
                "stream_reliable": item["stream_reliable"],
                "source_statement_ids": _identifier_list(
                    item["source_statement_ids"],
                    f"stages[{index}].source_statement_ids",
                ),
                "movable": item["movable"],
                "blocked_effects": sorted(blocked_effects),
            }
        )

    event_edges = value["event_edges"]
    if not isinstance(event_edges, list):
        raise ValueError("logical graph event_edges must be a list")
    event_ids = set()
    normalized_events = []
    for index, item in enumerate(event_edges):
        fields = {
            "event_edge_id",
            "producer_stage_id",
            "consumer_stage_id",
            "reuse_scope",
            "reuse_proven_safe",
        }
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"event_edges[{index}] fields are invalid")
        event_id = _text(item["event_edge_id"], f"event_edges[{index}].event_edge_id")
        if event_id in event_ids:
            raise ValueError(f"duplicate event_edge_id: {event_id}")
        producer = _text(
            item["producer_stage_id"], f"event_edges[{index}].producer_stage_id"
        )
        consumer = _text(
            item["consumer_stage_id"], f"event_edges[{index}].consumer_stage_id"
        )
        if producer not in stage_ids or consumer not in stage_ids:
            raise ValueError(f"event_edges[{index}] references an unknown stage")
        reuse_scope = _text(item["reuse_scope"], f"event_edges[{index}].reuse_scope")
        if reuse_scope not in {"single_use", "per_iteration", "ring"}:
            raise ValueError(f"event_edges[{index}].reuse_scope is invalid")
        if item["reuse_proven_safe"] is not True:
            raise ValueError(f"event_edges[{index}] has ambiguous event reuse")
        event_ids.add(event_id)
        normalized_events.append(
            {
                "event_edge_id": event_id,
                "producer_stage_id": producer,
                "consumer_stage_id": consumer,
                "reuse_scope": reuse_scope,
                "reuse_proven_safe": True,
            }
        )

    forks = value["forks"]
    if not isinstance(forks, list) or not forks:
        raise ValueError("logical graph forks must be a non-empty list")
    fork_ids = set()
    normalized_forks = []
    for index, item in enumerate(forks):
        fields = {"fork_id", "source_stage_id", "branch_stage_ids"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"forks[{index}] fields are invalid")
        fork_id = _text(item["fork_id"], f"forks[{index}].fork_id")
        if fork_id in fork_ids:
            raise ValueError(f"duplicate fork_id: {fork_id}")
        source = _text(item["source_stage_id"], f"forks[{index}].source_stage_id")
        branches = _identifier_list(
            item["branch_stage_ids"], f"forks[{index}].branch_stage_ids"
        )
        if source not in stage_ids or set(branches) - stage_ids or len(branches) < 2:
            raise ValueError(
                f"forks[{index}] must bind at least two known branch stages"
            )
        fork_ids.add(fork_id)
        normalized_forks.append(
            {
                "fork_id": fork_id,
                "source_stage_id": source,
                "branch_stage_ids": branches,
            }
        )

    joins = value["joins"]
    if not isinstance(joins, list) or not joins:
        raise ValueError("logical graph joins must be a non-empty list")
    join_ids = set()
    normalized_joins = []
    fork_by_id = {item["fork_id"]: item for item in normalized_forks}
    event_by_id = {item["event_edge_id"]: item for item in normalized_events}
    for index, item in enumerate(joins):
        fields = {
            "join_id",
            "fork_id",
            "branch_stage_ids",
            "downstream_stage_id",
            "required_event_edge_ids",
        }
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"joins[{index}] fields are invalid")
        join_id = _text(item["join_id"], f"joins[{index}].join_id")
        if join_id in join_ids:
            raise ValueError(f"duplicate join_id: {join_id}")
        fork_id = _text(item["fork_id"], f"joins[{index}].fork_id")
        branches = _identifier_list(
            item["branch_stage_ids"], f"joins[{index}].branch_stage_ids"
        )
        downstream = _text(
            item["downstream_stage_id"], f"joins[{index}].downstream_stage_id"
        )
        event_refs = _identifier_list(
            item["required_event_edge_ids"],
            f"joins[{index}].required_event_edge_ids",
            nonempty=False,
        )
        if (
            fork_id not in fork_by_id
            or set(branches) - stage_ids
            or downstream not in stage_ids
        ):
            raise ValueError(f"joins[{index}] references an unknown fork or stage")
        if set(branches) != set(fork_by_id[fork_id]["branch_stage_ids"]):
            raise ValueError(
                f"joins[{index}].branch_stage_ids must equal its fork branches"
            )
        for event_id in event_refs:
            event = event_by_id.get(event_id)
            if event is None:
                raise ValueError(
                    f"joins[{index}] references an unknown event: {event_id}"
                )
            if (
                event["consumer_stage_id"] != downstream
                or event["producer_stage_id"] not in branches
            ):
                raise ValueError(
                    f"joins[{index}] event {event_id} does not bind a branch to downstream"
                )
        join_ids.add(join_id)
        normalized_joins.append(
            {
                "join_id": join_id,
                "fork_id": fork_id,
                "branch_stage_ids": branches,
                "downstream_stage_id": downstream,
                "required_event_edge_ids": event_refs,
            }
        )

    dependencies = value["hard_dependencies"]
    if not isinstance(dependencies, list):
        raise ValueError("logical graph hard_dependencies must be a list")
    normalized_dependencies = []
    seen_dependencies = set()
    for index, item in enumerate(dependencies):
        fields = {"before_stage_id", "after_stage_id", "kind"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"hard_dependencies[{index}] fields are invalid")
        before = _text(
            item["before_stage_id"], f"hard_dependencies[{index}].before_stage_id"
        )
        after = _text(
            item["after_stage_id"], f"hard_dependencies[{index}].after_stage_id"
        )
        if before not in stage_ids or after not in stage_ids or before == after:
            raise ValueError(f"hard_dependencies[{index}] references invalid stages")
        kind = _text(item["kind"], f"hard_dependencies[{index}].kind").upper()
        if kind not in HARD_DEPENDENCY_KINDS:
            raise ValueError(f"hard_dependencies[{index}].kind is invalid")
        if (before, after, kind) in seen_dependencies:
            raise ValueError("logical graph has duplicate hard dependency")
        seen_dependencies.add((before, after, kind))
        normalized_dependencies.append(
            {"before_stage_id": before, "after_stage_id": after, "kind": kind}
        )
    _topological_check(
        stage_ids, {(before, after) for before, after, _ in seen_dependencies}
    )

    normalized = {
        "schema_version": SCHEMA,
        "graph_id": _text(value["graph_id"], "graph_id"),
        "request_fingerprint": _text(
            value["request_fingerprint"], "request_fingerprint"
        ),
        "stages": normalized_stages,
        "event_edges": normalized_events,
        "forks": normalized_forks,
        "joins": normalized_joins,
        "hard_dependencies": normalized_dependencies,
        "dependency_evidence_fingerprint": _text(
            value["dependency_evidence_fingerprint"], "dependency_evidence_fingerprint"
        ),
    }
    normalized["graph_fingerprint"] = fingerprint(normalized)
    if (
        require_fingerprint
        and value["graph_fingerprint"] != normalized["graph_fingerprint"]
    ):
        raise ValueError("logical graph fingerprint mismatch")
    return normalized


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument("--graph", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(json.loads(args.graph.read_text()))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
