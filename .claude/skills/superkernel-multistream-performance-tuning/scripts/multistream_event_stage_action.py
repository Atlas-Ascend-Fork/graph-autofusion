#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Authorize event-first and stage-split actions from critical-path evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import multistream_critical_path
import multistream_logical_graph


SCHEMA = "superkernel-multistream-event-stage-action-v1"
CHANGE_KINDS = {"event_edge_refinement", "stage_split"}
EVENT_REQUIREMENTS = (
    "preserve_data_producer_before_record",
    "preserve_first_consumer_before_wait",
    "preserve_event_reuse_order",
    "multistream_path_only",
)
STAGE_REQUIREMENTS = (
    "preserve_data_event_wait_alias_and_side_effect_edges",
    "preserve_single_stream_projection",
    "preserve_stage_state_lifetime",
    "multistream_path_only",
)


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


def _validate_analysis(value):
    fields = {
        "schema_version",
        "capture_fingerprint",
        "request_fingerprint",
        "decision",
        "targets",
        "analysis_fingerprint",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("critical path analysis fields are invalid")
    if value["schema_version"] != multistream_critical_path.ANALYSIS_SCHEMA:
        raise ValueError("critical path analysis schema is invalid")
    unsigned = {
        key: item for key, item in value.items() if key != "analysis_fingerprint"
    }
    if value["analysis_fingerprint"] != multistream_critical_path.fingerprint(unsigned):
        raise ValueError("critical path analysis fingerprint mismatch")
    if value["decision"] not in {"opportunity", "no_event_or_stage_candidate"}:
        raise ValueError("critical path analysis decision is invalid")
    return value


def build(analysis, graph):
    analysis = _validate_analysis(analysis)
    graph = multistream_logical_graph.validate(graph)
    if analysis["request_fingerprint"] != graph["request_fingerprint"]:
        raise ValueError(
            "critical path analysis and logical graph request fingerprints differ"
        )
    stage_by_id = {item["stage_id"]: item for item in graph["stages"]}
    event_by_id = {}
    for event in graph["event_edges"]:
        event_by_id[event["event_edge_id"]] = event
    actions = []
    for target in sorted(analysis["targets"], key=lambda item: item["join_id"]):
        join_id = _text(target.get("join_id"), "analysis target.join_id")
        if target.get("reason") is not None:
            continue
        stage_id = target.get("critical_branch_stage_id")
        stage = stage_by_id.get(stage_id)
        if stage is None:
            raise ValueError("analysis target references unknown critical stage")
        event_action_ids = []
        for event_id in sorted(target.get("actionable_event_edge_ids", [])):
            event = event_by_id.get(event_id)
            if event is None or event["producer_stage_id"] != stage_id:
                raise ValueError(
                    "analysis event action does not bind the critical producer stage"
                )
            event_action_id = f"event:{join_id}:{stage_id}:{event_id}"
            event_action_ids.append(event_action_id)
            actions.append(
                {
                    "action_id": event_action_id,
                    "change_kind": "event_edge_refinement",
                    "join_id": join_id,
                    "stage_id": stage_id,
                    "event_edge_id": event_id,
                    "risk": "low",
                    "activation_condition": "immediate",
                    "parent_action_id": None,
                    "expected_dispatch_change": {
                        "kind": "event_notify_earlier",
                        "logical_id": event_id,
                    },
                    "safety_requirements": list(EVENT_REQUIREMENTS),
                }
            )
        for stage_id in sorted(target.get("actionable_stage_ids", [])):
            stage = stage_by_id.get(stage_id)
            if stage is None or stage["movable"] is not True:
                raise ValueError(
                    "analysis action references non-movable or unknown stage"
                )
            event_action_id = event_action_ids[0] if event_action_ids else None
            actions.append(
                {
                    "action_id": f"stage:{join_id}:{stage_id}",
                    "change_kind": "stage_split",
                    "join_id": join_id,
                    "stage_id": stage_id,
                    "event_edge_id": (
                        target.get("actionable_event_edge_ids", [None])[0]
                        if target.get("actionable_event_edge_ids")
                        else None
                    ),
                    "risk": "medium",
                    "activation_condition": (
                        "event_clean3_nonregressing_without_incremental_gain"
                        if event_action_id
                        else "immediate"
                    ),
                    "parent_action_id": event_action_id,
                    "expected_dispatch_change": {
                        "kind": "stage_dispatch_earlier",
                        "logical_id": stage_id,
                    },
                    "safety_requirements": list(STAGE_REQUIREMENTS),
                }
            )
    actions.sort(
        key=lambda item: (
            item["activation_condition"] != "immediate",
            item["risk"],
            item["action_id"],
        )
    )
    result = {
        "schema_version": SCHEMA,
        "request_fingerprint": graph["request_fingerprint"],
        "critical_path_analysis_fingerprint": analysis["analysis_fingerprint"],
        "actions": actions,
    }
    result["action_catalog_fingerprint"] = fingerprint(result)
    return result


def validate(value):
    fields = {
        "schema_version",
        "request_fingerprint",
        "critical_path_analysis_fingerprint",
        "actions",
        "action_catalog_fingerprint",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema_version") != SCHEMA
    ):
        raise ValueError(f"event/stage action catalog must use {SCHEMA}")
    unsigned = {
        key: item for key, item in value.items() if key != "action_catalog_fingerprint"
    }
    if value["action_catalog_fingerprint"] != fingerprint(unsigned):
        raise ValueError("event/stage action catalog fingerprint mismatch")
    seen = set()
    parents = set()
    for index, action in enumerate(value["actions"]):
        required = {
            "action_id",
            "change_kind",
            "join_id",
            "stage_id",
            "event_edge_id",
            "risk",
            "activation_condition",
            "parent_action_id",
            "expected_dispatch_change",
            "safety_requirements",
        }
        if not isinstance(action, dict) or set(action) != required:
            raise ValueError(f"actions[{index}] fields are invalid")
        action_id = _text(action["action_id"], f"actions[{index}].action_id")
        if action_id in seen or action["change_kind"] not in CHANGE_KINDS:
            raise ValueError("event/stage action ids or change kind are invalid")
        seen.add(action_id)
        if action["risk"] not in {"low", "medium"}:
            raise ValueError("event/stage action risk is invalid")
        requirements = (
            EVENT_REQUIREMENTS
            if action["change_kind"] == "event_edge_refinement"
            else STAGE_REQUIREMENTS
        )
        if action["safety_requirements"] != list(requirements):
            raise ValueError("event/stage action safety requirements are invalid")
        expected = action["expected_dispatch_change"]
        if not isinstance(expected, dict) or set(expected) != {"kind", "logical_id"}:
            raise ValueError("event/stage expected dispatch change fields are invalid")
        expected_kind = (
            "event_notify_earlier"
            if action["change_kind"] == "event_edge_refinement"
            else "stage_dispatch_earlier"
        )
        expected_id = (
            action["event_edge_id"]
            if action["change_kind"] == "event_edge_refinement"
            else action["stage_id"]
        )
        if expected != {"kind": expected_kind, "logical_id": expected_id}:
            raise ValueError("event/stage expected dispatch change is inconsistent")
        parent = action["parent_action_id"]
        if parent is not None:
            parents.add(
                (action_id, _text(parent, f"actions[{index}].parent_action_id"))
            )
    for _, parent in parents:
        if parent not in seen:
            raise ValueError("event/stage action parent is missing")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            if not args.analysis or not args.graph or not args.out:
                raise ValueError("build requires --analysis, --graph, and --out")
            if args.out.exists():
                raise ValueError(f"output already exists: {args.out}")
            result = build(
                json.loads(args.analysis.read_text()),
                json.loads(args.graph.read_text()),
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        else:
            if not args.catalog:
                raise ValueError("validate requires --catalog")
            result = validate(json.loads(args.catalog.read_text()))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
