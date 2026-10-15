#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Freeze event-first/stage-second candidate lineage from critical-path evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import multistream_critical_path
import multistream_event_stage_action


SCHEMA = "superkernel-multistream-event-stage-candidate-matrix-v1"
HISTORY_SCHEMA = "superkernel-multistream-event-stage-history-v1"


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


def _validate_analysis(value):
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != multistream_critical_path.ANALYSIS_SCHEMA
    ):
        raise ValueError("critical path analysis schema is invalid")
    unsigned = {
        key: item for key, item in value.items() if key != "analysis_fingerprint"
    }
    if value.get("analysis_fingerprint") != multistream_critical_path.fingerprint(
        unsigned
    ):
        raise ValueError("critical path analysis fingerprint mismatch")
    return value


def _history(value):
    if not isinstance(value, dict) or set(value) != {"settled_actions"}:
        raise ValueError("event/stage history must contain exactly settled_actions")
    records = {}
    for index, item in enumerate(value["settled_actions"]):
        fields = {
            "action_id",
            "status",
            "correctness_passed",
            "mechanism_validated",
            "clean3_nonregressing",
        }
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"settled_actions[{index}] fields are invalid")
        action_id = item["action_id"]
        if not isinstance(action_id, str) or not action_id or action_id in records:
            raise ValueError("settled action ids must be unique non-empty strings")
        if item["status"] not in {"accepted", "rejected", "blocked", "failed"}:
            raise ValueError("settled action status is invalid")
        if any(
            not isinstance(item[field], bool)
            for field in (
                "correctness_passed",
                "mechanism_validated",
                "clean3_nonregressing",
            )
        ):
            raise ValueError("settled action gate values must be boolean")
        records[action_id] = dict(item)
    return records


def _eligible_parent(record):
    return record is not None and (
        record["status"] == "rejected"
        and record["correctness_passed"]
        and record["mechanism_validated"]
        and record["clean3_nonregressing"]
    )


def plan(analysis, catalog, history, max_trials, scope_derivatives=None):
    analysis = _validate_analysis(analysis)
    catalog = multistream_event_stage_action.validate(catalog)
    if (
        catalog["request_fingerprint"] != analysis["request_fingerprint"]
        or catalog["critical_path_analysis_fingerprint"]
        != analysis["analysis_fingerprint"]
    ):
        raise ValueError("event/stage catalog does not bind the critical path analysis")
    if (
        isinstance(max_trials, bool)
        or not isinstance(max_trials, int)
        or max_trials < 1
    ):
        raise ValueError("max_trials must be a positive integer")
    settled = _history(history)
    target_by_join = {item["join_id"]: item for item in analysis["targets"]}
    candidates, blocked = [], []
    immediate_event_joins = set()
    for action in catalog["actions"]:
        target = target_by_join.get(action["join_id"])
        if target is None:
            raise ValueError(
                "event/stage action no longer binds an actionable critical target"
            )
        if action["change_kind"] == "event_edge_refinement":
            bound = action["event_edge_id"] in target.get(
                "actionable_event_edge_ids", []
            )
        else:
            bound = action["stage_id"] in target.get("actionable_stage_ids", [])
        if not bound:
            raise ValueError(
                "event/stage action no longer binds an actionable critical target"
            )
        parent = action["parent_action_id"]
        if action["activation_condition"] == "immediate":
            if action["action_id"] in settled:
                continue
            if action["change_kind"] == "event_edge_refinement":
                if action["join_id"] in immediate_event_joins:
                    blocked.append(
                        {
                            "action_id": action["action_id"],
                            "code": "JOIN_EVENT_BUDGET_EXHAUSTED",
                            "detail": "one join may plan only one minimal immediate event candidate",
                        }
                    )
                    continue
                immediate_event_joins.add(action["join_id"])
        elif not _eligible_parent(settled.get(parent)):
            blocked.append(
                {
                    "action_id": action["action_id"],
                    "code": "PARENT_MECHANISM_NOT_VALIDATED",
                    "detail": "parent action lacks rejected, correct, mechanism-valid, non-regressing clean3 evidence",
                }
            )
            continue
        score = target["predicted_e2e_upper_bound"] * 100.0 - (
            0 if action["risk"] == "low" else 10
        )
        candidates.append(
            {
                "change_kind": action["change_kind"],
                "action_id": action["action_id"],
                "parent_action_id": parent,
                "join_id": action["join_id"],
                "stage_id": action["stage_id"],
                "event_edge_id": action["event_edge_id"],
                "risk": action["risk"],
                "score": score,
                "activation_condition": action["activation_condition"],
                "lineage_depth": 1,
                "comparison_baselines": ["incumbent"],
                "acceptance_gate": "incremental_clean_end_to_end_only",
            }
        )
    for item in scope_derivatives or []:
        required = {"derivative_id", "parent_action_id", "source_action"}
        if (
            not isinstance(item, dict)
            or set(item) != required
            or not isinstance(item["source_action"], dict)
        ):
            raise ValueError("scope derivative fields are invalid")
        parent = item["parent_action_id"]
        source_action = item["source_action"]
        if set(source_action) != {"change_kind", "range_id", "factor_id"}:
            raise ValueError(
                "scope derivative source_action must declare one exact factor"
            )
        if source_action["change_kind"] not in {"scope_split", "range_exclusion"}:
            raise ValueError("scope derivative source action kind is invalid")
        for field in ("range_id", "factor_id"):
            if (
                not isinstance(source_action[field], str)
                or not source_action[field].strip()
            ):
                raise ValueError("scope derivative range/factor identity is invalid")
        if not _eligible_parent(settled.get(parent)):
            blocked.append(
                {
                    "action_id": item["derivative_id"],
                    "code": "PARENT_MECHANISM_NOT_VALIDATED",
                    "detail": "scope derivative requires a validated non-regressing parent",
                }
            )
            continue
        candidates.append(
            {
                "change_kind": "scope_event_derivative",
                "action_id": item["derivative_id"],
                "parent_action_id": parent,
                "join_id": None,
                "stage_id": None,
                "event_edge_id": None,
                "risk": "high",
                "score": -25.0,
                "activation_condition": "parent_clean3_nonregressing_without_incremental_gain",
                "source_action": source_action,
                "lineage_depth": 2,
                "comparison_baselines": ["incumbent", "parent_candidate"],
                "acceptance_gate": "incremental_clean_end_to_end_only",
            }
        )
    candidates.sort(
        key=lambda item: (
            item["activation_condition"] != "immediate",
            -item["score"],
            item["action_id"],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"MS-CP-C{index:03d}"
        candidate["rank"] = index
        candidate["selected_for_execution"] = (
            index <= max_trials and candidate["activation_condition"] == "immediate"
        )
    result = {
        "schema_version": SCHEMA,
        "request_fingerprint": analysis["request_fingerprint"],
        "critical_path_analysis_fingerprint": analysis["analysis_fingerprint"],
        "action_catalog_fingerprint": catalog["action_catalog_fingerprint"],
        "history_fingerprint": fingerprint(history),
        "budget": {
            "max_trials": max_trials,
            "candidate_count": len(candidates),
            "planned_trials": sum(
                item["selected_for_execution"] for item in candidates
            ),
        },
        "candidates": candidates,
        "blocked_actions": sorted(blocked, key=lambda item: item["action_id"]),
    }
    result["matrix_fingerprint"] = fingerprint(result)
    return result


def validate(value, analysis, catalog, history, max_trials, scope_derivatives=None):
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError(f"event/stage candidate matrix must use {SCHEMA}")
    expected = plan(analysis, catalog, history, max_trials, scope_derivatives)
    if value != expected:
        raise ValueError(
            "event/stage candidate matrix differs from deterministic replay"
        )
    return {
        "valid": True,
        "matrix_fingerprint": value["matrix_fingerprint"],
        "candidate_count": len(value["candidates"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan",))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--scope-derivatives", type=Path)
    parser.add_argument("--max-trials", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        derivatives = (
            json.loads(args.scope_derivatives.read_text())
            if args.scope_derivatives
            else None
        )
        result = plan(
            json.loads(args.analysis.read_text()),
            json.loads(args.catalog.read_text()),
            json.loads(args.history.read_text()),
            args.max_trials,
            derivatives,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
