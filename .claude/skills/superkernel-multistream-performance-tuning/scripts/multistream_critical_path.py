#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Analyze stable multistream joins before authorizing event or stage actions."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import multistream_core_family
import multistream_logical_graph


CAPTURE_SCHEMA = "superkernel-multistream-critical-path-capture-v1"
ANALYSIS_SCHEMA = "superkernel-multistream-critical-path-analysis-v1"
MIN_OCCURRENCES = 3
PREDICTED_E2E_GATE = 0.03


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


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(
            f"{label} must be a finite {'positive' if positive else 'non-negative'} number"
        )
    return value


def _summary(values):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("cannot summarize empty values")
    p90_index = max(0, math.ceil(len(values) * 0.9) - 1)
    return {
        "p50": statistics.median(values),
        "p90": values[p90_index],
        "min": values[0],
        "max": values[-1],
    }


def _merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def _intersection(left, right):
    intervals = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start, end = max(left_start, right_start), min(left_end, right_end)
            if end > start:
                intervals.append((start, end))
    return _merge(intervals)


def _duration(intervals):
    return sum(end - start for start, end in _merge(intervals))


def _validate_stage(value, label, stage_ids):
    fields = {
        "stage_id",
        "stream_id",
        "accelerator_core",
        "block_num",
        "mix_block_num",
        "ready_time_us",
        "start_us",
        "duration_us",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    stage_id = _text(value["stage_id"], f"{label}.stage_id")
    if stage_id not in stage_ids:
        raise ValueError(f"{label}.stage_id is not in logical graph")
    stream_id = value["stream_id"]
    if isinstance(stream_id, bool) or not isinstance(stream_id, int) or stream_id < 0:
        raise ValueError(f"{label}.stream_id must be a non-negative integer")
    resource = multistream_core_family.classify_profile_identity(
        value["accelerator_core"], value["block_num"], value["mix_block_num"]
    )
    ready = _number(value["ready_time_us"], f"{label}.ready_time_us")
    start = _number(value["start_us"], f"{label}.start_us")
    duration = _number(value["duration_us"], f"{label}.duration_us", positive=True)
    if ready > start:
        raise ValueError(f"{label}.ready_time_us must not exceed start_us")
    return {
        "stage_id": stage_id,
        "stream_id": stream_id,
        "accelerator_core": resource["accelerator_core"],
        "block_num": resource["block_num"],
        "mix_block_num": resource["mix_block_num"],
        "ready_time_us": ready,
        "start_us": start,
        "duration_us": duration,
    }


def _validate_event(value, label, event_ids):
    fields = {"event_edge_id", "event_kind", "stream_id", "time_us"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    event_id = _text(value["event_edge_id"], f"{label}.event_edge_id")
    if event_id not in event_ids:
        raise ValueError(f"{label}.event_edge_id is not in logical graph")
    if value["event_kind"] != "notify":
        raise ValueError(f"{label}.event_kind must be notify")
    stream_id = value["stream_id"]
    if isinstance(stream_id, bool) or not isinstance(stream_id, int) or stream_id < 0:
        raise ValueError(f"{label}.stream_id must be a non-negative integer")
    return {
        "event_edge_id": event_id,
        "event_kind": "notify",
        "stream_id": stream_id,
        "time_us": _number(value["time_us"], f"{label}.time_us"),
    }


def validate_capture(value, *, require_fingerprint=True):
    fields = {
        "schema_version",
        "capture_id",
        "request_fingerprint",
        "logical_graph",
        "clock_domain",
        "timestamp_resolution_us",
        "trace_overflow_detected",
        "occurrences",
    }
    if require_fingerprint:
        fields.add("capture_fingerprint")
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema_version") != CAPTURE_SCHEMA
    ):
        raise ValueError(
            f"critical path capture must use {CAPTURE_SCHEMA} with exactly {sorted(fields)}"
        )
    graph = multistream_logical_graph.validate(value["logical_graph"])
    if value["trace_overflow_detected"] is not False:
        raise ValueError("critical path capture trace overflow is a blocker")
    if value["request_fingerprint"] != graph["request_fingerprint"]:
        raise ValueError(
            "critical path capture request_fingerprint differs from logical graph"
        )
    stage_ids = {item["stage_id"] for item in graph["stages"]}
    event_ids = {item["event_edge_id"] for item in graph["event_edges"]}
    join_by_id = {item["join_id"]: item for item in graph["joins"]}
    occurrences = value["occurrences"]
    if not isinstance(occurrences, list) or len(occurrences) < MIN_OCCURRENCES:
        raise ValueError("critical path capture requires at least three occurrences")
    normalized_occurrences = []
    alignment_ids = set()
    seen_join_alignment = set()
    for index, item in enumerate(occurrences):
        required = {
            "alignment_id",
            "join_id",
            "fork_time_us",
            "step_latency_us",
            "timeline_complete",
            "stages",
            "events",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"occurrences[{index}] fields are invalid")
        alignment_id = _text(item["alignment_id"], f"occurrences[{index}].alignment_id")
        join_id = _text(item["join_id"], f"occurrences[{index}].join_id")
        if join_id not in join_by_id:
            raise ValueError(f"occurrences[{index}].join_id is not in logical graph")
        if item["timeline_complete"] is not True:
            raise ValueError(f"occurrences[{index}] timeline is incomplete")
        key = (alignment_id, join_id)
        if key in seen_join_alignment:
            raise ValueError(
                "critical path capture has duplicate alignment/join occurrence"
            )
        alignment_ids.add(alignment_id)
        seen_join_alignment.add(key)
        stages = [
            _validate_stage(
                entry, f"occurrences[{index}].stages[{position}]", stage_ids
            )
            for position, entry in enumerate(item["stages"])
        ]
        stage_map = {entry["stage_id"]: entry for entry in stages}
        if len(stage_map) != len(stages):
            raise ValueError(f"occurrences[{index}] has duplicate stage observation")
        required_stages = set(join_by_id[join_id]["branch_stage_ids"]) | {
            join_by_id[join_id]["downstream_stage_id"]
        }
        if required_stages - set(stage_map):
            raise ValueError(
                f"occurrences[{index}] lacks required branch or downstream stage"
            )
        events = [
            _validate_event(
                entry, f"occurrences[{index}].events[{position}]", event_ids
            )
            for position, entry in enumerate(item["events"])
        ]
        event_map = {entry["event_edge_id"]: entry for entry in events}
        if len(event_map) != len(events):
            raise ValueError(f"occurrences[{index}] has duplicate event observation")
        required_events = set(join_by_id[join_id]["required_event_edge_ids"])
        if required_events - set(event_map):
            raise ValueError(f"occurrences[{index}] lacks required event notify")
        event_by_id = {entry["event_edge_id"]: entry for entry in graph["event_edges"]}
        for event_id in required_events:
            producer_id = event_by_id[event_id]["producer_stage_id"]
            if event_map[event_id]["time_us"] < _end(stage_map[producer_id]):
                raise ValueError(
                    f"occurrences[{index}] event notify precedes its producer end"
                )
        normalized_occurrences.append(
            {
                "alignment_id": alignment_id,
                "join_id": join_id,
                "fork_time_us": _number(
                    item["fork_time_us"], f"occurrences[{index}].fork_time_us"
                ),
                "step_latency_us": _number(
                    item["step_latency_us"],
                    f"occurrences[{index}].step_latency_us",
                    positive=True,
                ),
                "timeline_complete": True,
                "stages": stages,
                "events": events,
            }
        )
        downstream = stage_map[join_by_id[join_id]["downstream_stage_id"]]
        if (
            normalized_occurrences[-1]["step_latency_us"]
            < _end(downstream) - normalized_occurrences[-1]["fork_time_us"]
        ):
            raise ValueError(
                f"occurrences[{index}].step_latency_us ends before downstream stage"
            )
    normalized = {
        "schema_version": CAPTURE_SCHEMA,
        "capture_id": _text(value["capture_id"], "capture_id"),
        "request_fingerprint": _text(
            value["request_fingerprint"], "request_fingerprint"
        ),
        "logical_graph": graph,
        "clock_domain": _text(value["clock_domain"], "clock_domain"),
        "timestamp_resolution_us": _number(
            value["timestamp_resolution_us"], "timestamp_resolution_us", positive=True
        ),
        "trace_overflow_detected": False,
        "occurrences": normalized_occurrences,
    }
    normalized["capture_fingerprint"] = fingerprint(normalized)
    if (
        require_fingerprint
        and value["capture_fingerprint"] != normalized["capture_fingerprint"]
    ):
        raise ValueError("critical path capture fingerprint mismatch")
    return normalized


def _analyze_occurrence(occurrence, join, graph, tolerance_us):
    stages = {item["stage_id"]: item for item in occurrence["stages"]}
    events = {item["event_edge_id"]: item for item in occurrence["events"]}
    event_by_id = {item["event_edge_id"]: item for item in graph["event_edges"]}
    stage_by_id = {item["stage_id"]: item for item in graph["stages"]}
    ready = {}
    for stage_id in join["branch_stage_ids"]:
        values = [_end(stages[stage_id])]
        for event_id in join["required_event_edge_ids"]:
            if event_by_id[event_id]["producer_stage_id"] == stage_id:
                values.append(events[event_id]["time_us"])
        ready[stage_id] = max(values)
    join_ready = max(ready.values())
    downstream_start = stages[join["downstream_stage_id"]]["start_us"]
    if downstream_start < join_ready:
        raise ValueError("downstream stage starts before its logical join is ready")
    critical = sorted(
        stage_id
        for stage_id, value in ready.items()
        if join_ready - value <= tolerance_us
    )
    # A tied critical path has no stable single branch and is deliberately not actionable.
    if len(critical) != 1:
        return {
            "alignment_id": occurrence["alignment_id"],
            "critical_branch_stage_id": None,
            "step_latency_us": occurrence["step_latency_us"],
            "join_ready_us": join_ready,
            "join_stall_us": downstream_start - join_ready,
            "branch_slack_us": {
                stage_id: join_ready - value for stage_id, value in ready.items()
            },
            "recoverable_by_stage_us": {},
            "recoverable_by_event_us": {},
            "stage_reason": "tied_critical_branch",
            "event_reason": "tied_critical_branch",
        }
    critical_id = critical[0]
    critical_stage = stages[critical_id]
    if not stage_by_id[critical_id]["stream_reliable"]:
        return {
            "alignment_id": occurrence["alignment_id"],
            "critical_branch_stage_id": critical_id,
            "step_latency_us": occurrence["step_latency_us"],
            "fork_time_us": occurrence["fork_time_us"],
            "join_ready_us": join_ready,
            "join_stall_us": downstream_start - join_ready,
            "branch_slack_us": {
                stage_id: join_ready - value for stage_id, value in ready.items()
            },
            "recoverable_by_stage_us": {},
            "recoverable_by_event_us": {},
            "stage_reason": "critical_stream_unreliable",
            "event_reason": "critical_stream_unreliable",
        }
    critical_family = multistream_core_family.classify_profile_identity(
        critical_stage["accelerator_core"],
        critical_stage["block_num"],
        critical_stage["mix_block_num"],
    )["core_family"]
    if critical_family not in {"CUBE", "VECTOR"}:
        stage_reason = "critical_stage_not_cube_vector"
        recovered = {}
    else:
        static = stage_by_id[critical_id]
        deferred = [
            (
                max(critical_stage["ready_time_us"], occurrence["fork_time_us"]),
                min(critical_stage["start_us"], join_ready),
            )
        ]
        if not static["movable"] or _duration(deferred) == 0:
            stage_reason = "no_deferred_critical_stage"
            recovered = {}
        else:
            complementary = "VECTOR" if critical_family == "CUBE" else "CUBE"
            other_intervals = [
                (stages[stage_id]["start_us"], _end(stages[stage_id]))
                for stage_id in join["branch_stage_ids"]
                if stage_id != critical_id
                and stage_by_id[stage_id]["stream_reliable"]
                and multistream_core_family.classify_profile_identity(
                    stages[stage_id]["accelerator_core"],
                    stages[stage_id]["block_num"],
                    stages[stage_id]["mix_block_num"],
                )["core_family"]
                == complementary
            ]
            recovered_value = _duration(_intersection(deferred, other_intervals))
            if recovered_value <= 0:
                stage_reason = "no_complementary_critical_gap"
                recovered = {}
            else:
                stage_reason = None
                recovered = {critical_id: recovered_value}
    delayed_events = {}
    for event_id in join["required_event_edge_ids"]:
        event = event_by_id[event_id]
        if event["producer_stage_id"] != critical_id:
            continue
        delay = events[event_id]["time_us"] - _end(critical_stage)
        if delay > 0:
            delayed_events[event_id] = delay
    event_reason = None if delayed_events else "no_delayed_critical_event"
    return {
        "alignment_id": occurrence["alignment_id"],
        "critical_branch_stage_id": critical_id,
        "step_latency_us": occurrence["step_latency_us"],
        "fork_time_us": occurrence["fork_time_us"],
        "join_ready_us": join_ready,
        "join_stall_us": downstream_start - join_ready,
        "branch_slack_us": {
            stage_id: join_ready - value for stage_id, value in ready.items()
        },
        "recoverable_by_stage_us": recovered,
        "recoverable_by_event_us": delayed_events,
        "stage_reason": stage_reason,
        "event_reason": event_reason,
    }


def _end(stage):
    return stage["start_us"] + stage["duration_us"]


def analyze(capture):
    capture = validate_capture(capture)
    graph = capture["logical_graph"]
    joins = {item["join_id"]: item for item in graph["joins"]}
    by_join = defaultdict(list)
    tolerance_by_join = {}
    for join_id, join in joins.items():
        samples = defaultdict(list)
        for occurrence in capture["occurrences"]:
            if occurrence["join_id"] != join_id:
                continue
            stages = {item["stage_id"]: item for item in occurrence["stages"]}
            events = {item["event_edge_id"]: item for item in occurrence["events"]}
            event_by_id = {item["event_edge_id"]: item for item in graph["event_edges"]}
            for stage_id in join["branch_stage_ids"]:
                ready = [_end(stages[stage_id])]
                ready.extend(
                    events[event_id]["time_us"]
                    for event_id in join["required_event_edge_ids"]
                    if event_by_id[event_id]["producer_stage_id"] == stage_id
                )
                samples[stage_id].append(max(ready) - occurrence["fork_time_us"])
        deviations = []
        for values in samples.values():
            median = statistics.median(values)
            deviations.extend(abs(value - median) for value in values)
        tolerance_by_join[join_id] = max(
            capture["timestamp_resolution_us"],
            statistics.median(deviations) if deviations else 0.0,
        )
    for occurrence in capture["occurrences"]:
        join_id = occurrence["join_id"]
        by_join[join_id].append(
            _analyze_occurrence(
                occurrence, joins[join_id], graph, tolerance_by_join[join_id]
            )
        )
    targets = []
    for join_id in sorted(by_join):
        occurrences = sorted(by_join[join_id], key=lambda item: item["alignment_id"])
        if len(occurrences) < MIN_OCCURRENCES:
            raise ValueError(f"join {join_id} has fewer than three aligned occurrences")
        critical_ids = {item["critical_branch_stage_id"] for item in occurrences}
        common_actions = (
            set.intersection(
                *(set(item["recoverable_by_stage_us"]) for item in occurrences)
            )
            if occurrences
            else set()
        )
        stable_action_ids = (
            sorted(common_actions)
            if len(critical_ids) == 1 and None not in critical_ids
            else []
        )
        common_events = (
            set.intersection(
                *(set(item["recoverable_by_event_us"]) for item in occurrences)
            )
            if occurrences
            else set()
        )
        stable_event_ids = (
            sorted(common_events)
            if len(critical_ids) == 1 and None not in critical_ids
            else []
        )
        recoverable = {
            stage_id: _summary(
                [item["recoverable_by_stage_us"][stage_id] for item in occurrences]
            )
            for stage_id in stable_action_ids
        }
        event_recoverable = {
            event_id: _summary(
                [item["recoverable_by_event_us"][event_id] for item in occurrences]
            )
            for event_id in stable_event_ids
        }
        critical_id = next(iter(critical_ids)) if len(critical_ids) == 1 else None
        step_latency = [item["step_latency_us"] for item in occurrences]
        predicted = max(
            [recoverable[stage_id]["p50"] for stage_id in stable_action_ids]
            + [event_recoverable[event_id]["p50"] for event_id in stable_event_ids],
            default=0.0,
        )
        denominator = max(_summary(step_latency)["p50"], 1.0)
        predicted_ratio = predicted / denominator
        if not stable_action_ids and not stable_event_ids:
            stage_reasons = {
                item["stage_reason"] for item in occurrences if item["stage_reason"]
            }
            event_reasons = {
                item["event_reason"] for item in occurrences if item["event_reason"]
            }
            if len(stage_reasons) == 1:
                reason = next(iter(stage_reasons))
            elif len(event_reasons) == 1:
                reason = next(iter(event_reasons))
            else:
                reason = "unstable_critical_path"
        elif predicted_ratio < PREDICTED_E2E_GATE:
            reason = "predicted_e2e_bound_below_gate"
        else:
            reason = None
        targets.append(
            {
                "join_id": join_id,
                "occurrence_count": len(occurrences),
                "timing_tolerance_us": tolerance_by_join[join_id],
                "critical_branch_stage_id": critical_id,
                "actionable_stage_ids": stable_action_ids if reason is None else [],
                "actionable_event_edge_ids": stable_event_ids if reason is None else [],
                "stage_recoverable_us": recoverable,
                "event_recoverable_us": event_recoverable,
                "recoverable_us": _summary([predicted])
                if reason is None
                else _summary([0.0]),
                "predicted_e2e_upper_bound": predicted_ratio,
                "join_ready_us": _summary(
                    [item["join_ready_us"] for item in occurrences]
                ),
                "join_stall_us": _summary(
                    [item["join_stall_us"] for item in occurrences]
                ),
                "step_latency_us": _summary(step_latency),
                "reason": reason,
                "occurrences": occurrences,
            }
        )
    decision = (
        "opportunity"
        if any(
            item["actionable_stage_ids"] or item["actionable_event_edge_ids"]
            for item in targets
        )
        else "no_event_or_stage_candidate"
    )
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "capture_fingerprint": capture["capture_fingerprint"],
        "request_fingerprint": capture["request_fingerprint"],
        "decision": decision,
        "targets": targets,
    }
    result["analysis_fingerprint"] = fingerprint(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze",))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        result = analyze(json.loads(args.capture.read_text()))
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
