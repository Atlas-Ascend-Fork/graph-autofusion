#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate bound short traces and classify parent-SK internal parallelism."""

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import multistream_contract
import multistream_core_family


CAPTURE_SCHEMA = "superkernel-multistream-short-trace-capture-v2"
ANALYSIS_SCHEMA = "superkernel-multistream-trace-analysis-v2"
CORE_FAMILIES = multistream_core_family.CORE_FAMILIES
PARALLELISM_EFFECTS = {"improved", "preserved", "degraded", "unknown"}


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


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _rooted(root, value, label, *, directory=False):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(
            f"{label} must be a finite {'positive' if positive else 'non-negative'} number"
        )
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _unsigned(value, field):
    return {key: item for key, item in value.items() if key != field}


def _validate_source_files(root, records):
    if not isinstance(records, list) or not records:
        raise ValueError("short trace source_files must be a non-empty list")
    normalized = []
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "file_fingerprint",
        }:
            raise ValueError(f"source_files[{index}] is invalid")
        path = _rooted(root, record["path"], f"source_files[{index}]")
        relative = _relative(root, path)
        if relative in seen:
            raise ValueError(f"duplicate short trace source file: {relative}")
        seen.add(relative)
        actual = file_fingerprint(path)
        if (
            path.stat().st_size != record["size_bytes"]
            or actual != record["file_fingerprint"]
        ):
            raise ValueError(f"short trace source file changed: {relative}")
        normalized.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "file_fingerprint": actual,
            }
        )
    return sorted(normalized, key=lambda item: item["path"])


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _union_duration(intervals):
    return sum(end - start for start, end in _merge_intervals(intervals))


def _pair_overlap(left, right):
    intersections = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if end > start:
                intersections.append((start, end))
    return _union_duration(intersections)


def _max_active_streams(events):
    points = []
    for event in events:
        points.append((event["start_us"], 1, event["stream_id"]))
        points.append((event["end_us"], -1, event["stream_id"]))
    active = defaultdict(int)
    maximum = 0
    for _, delta, stream in sorted(points, key=lambda item: (item[0], item[1])):
        active[stream] += delta
        if active[stream] <= 0:
            active.pop(stream, None)
        maximum = max(maximum, len(active))
    return maximum


def _validate_event(value, label):
    required = {
        "child_origin_identity",
        "lane_id",
        "stream_id",
        "accelerator_core",
        "block_num",
        "mix_block_num",
        "start_us",
        "duration_us",
        "event_kind",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    origin = _text(value["child_origin_identity"], f"{label}.child_origin_identity")
    lane = value["lane_id"]
    if isinstance(lane, bool) or not isinstance(lane, (str, int)) or str(lane) == "":
        raise ValueError(f"{label}.lane_id must be a string or integer")
    stream = value["stream_id"]
    if isinstance(stream, bool) or not isinstance(stream, int) or stream < 0:
        raise ValueError(f"{label}.stream_id must be a non-negative integer")
    resource = multistream_core_family.classify_profile_identity(
        value["accelerator_core"], value["block_num"], value["mix_block_num"]
    )
    start = _number(value["start_us"], f"{label}.start_us")
    duration = _number(value["duration_us"], f"{label}.duration_us", positive=True)
    return {
        "child_origin_identity": origin,
        "lane_id": lane,
        "stream_id": stream,
        **resource,
        "start_us": start,
        "duration_us": duration,
        "end_us": start + duration,
        "event_kind": _text(value["event_kind"], f"{label}.event_kind"),
    }


def _occurrence_metrics(value, label):
    required = {"alignment_id", "step_id", "events"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    alignment_id = _text(value["alignment_id"], f"{label}.alignment_id")
    step_id = value["step_id"]
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
        raise ValueError(f"{label}.step_id must be a non-negative integer")
    raw_events = value["events"]
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"{label}.events must be non-empty")
    events = [
        _validate_event(item, f"{label}.events[{index}]")
        for index, item in enumerate(raw_events)
    ]
    intervals = [(item["start_us"], item["end_us"]) for item in events]
    by_stream = defaultdict(list)
    by_stream_family = defaultdict(list)
    for event in events:
        interval = (event["start_us"], event["end_us"])
        by_stream[event["stream_id"]].append(interval)
        by_stream_family[(event["stream_id"], event["core_family"])].append(interval)
    streams = sorted(by_stream)
    any_overlap = []
    cv_overlap = []
    same_overlap = []
    mix_overlap = []
    for left_index, left_stream in enumerate(streams):
        for right_stream in streams[left_index + 1 :]:
            overlap = _pair_overlap(by_stream[left_stream], by_stream[right_stream])
            if overlap:
                any_overlap.append(overlap)
            for left_family in CORE_FAMILIES:
                left = by_stream_family.get((left_stream, left_family), [])
                if not left:
                    continue
                for right_family in CORE_FAMILIES:
                    right = by_stream_family.get((right_stream, right_family), [])
                    if not right:
                        continue
                    value_us = _pair_overlap(left, right)
                    if not value_us:
                        continue
                    families = {left_family, right_family}
                    if families == {"CUBE", "VECTOR"}:
                        cv_overlap.append(value_us)
                    if left_family == right_family and left_family not in {
                        "WAIT",
                        "OTHER",
                    }:
                        same_overlap.append(value_us)
                    if "MIX" in families and families & {"CUBE", "VECTOR"}:
                        mix_overlap.append(value_us)
    start = min(item[0] for item in intervals)
    end = max(item[1] for item in intervals)
    interval_us = end - start
    cube_count = sum(item["core_family"] == "CUBE" for item in events)
    vector_count = sum(item["core_family"] == "VECTOR" for item in events)
    return {
        "alignment_id": alignment_id,
        "step_id": step_id,
        "event_count": len(events),
        "child_identity_count": len({item["child_origin_identity"] for item in events}),
        "child_origin_identities": sorted(
            {item["child_origin_identity"] for item in events}
        ),
        "lane_count": len({str(item["lane_id"]) for item in events}),
        "stream_count": len(streams),
        "core_family_counts": {
            family: sum(item["core_family"] == family for item in events)
            for family in sorted(CORE_FAMILIES)
            if any(item["core_family"] == family for item in events)
        },
        "cube_vector_identity_complete": cube_count > 0 and vector_count > 0,
        "interval_us": interval_us,
        "duration_sum_us": sum(item["duration_us"] for item in events),
        "union_duration_us": _union_duration(intervals),
        "overlap_work_us": sum(item["duration_us"] for item in events)
        - _union_duration(intervals),
        "max_active_streams": _max_active_streams(events),
        "stream_pair_overlap_us": sum(any_overlap),
        "cube_vector_overlap_us": sum(cv_overlap),
        "cube_vector_overlap_ratio": sum(cv_overlap) / interval_us
        if interval_us
        else 0.0,
        "same_resource_overlap_us": sum(same_overlap),
        "mix_competition_overlap_us": sum(mix_overlap),
        "wait_sync_duration_us": sum(
            item["duration_us"]
            for item in events
            if item["core_family"] == "WAIT"
            or item["event_kind"].lower() in {"wait", "sync", "barrier"}
        ),
    }


def _stats(values):
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("cannot summarize an empty metric")
    middle = statistics.median(ordered)
    position = (len(ordered) - 1) * 0.9
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(ordered),
        "p50": middle,
        "p90": p90,
        "mad": statistics.median(abs(item - middle) for item in ordered),
        "min": min(ordered),
        "max": max(ordered),
    }


def _summarize_occurrences(values, label):
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError(f"{label} requires at least three occurrences")
    occurrences = [
        _occurrence_metrics(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    ]
    ids = [item["alignment_id"] for item in occurrences]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} has duplicate alignment_id")
    child_sets = [item["child_origin_identities"] for item in occurrences]
    if any(item != child_sets[0] for item in child_sets[1:]):
        raise ValueError(f"{label} child origin identity set is unstable")
    metric_names = (
        "interval_us",
        "duration_sum_us",
        "union_duration_us",
        "overlap_work_us",
        "max_active_streams",
        "stream_pair_overlap_us",
        "cube_vector_overlap_us",
        "cube_vector_overlap_ratio",
        "same_resource_overlap_us",
        "mix_competition_overlap_us",
        "wait_sync_duration_us",
    )
    return {
        "occurrence_count": len(occurrences),
        "alignment_ids": ids,
        "child_origin_identities": child_sets[0],
        "cube_vector_identity_complete": all(
            item["cube_vector_identity_complete"] for item in occurrences
        ),
        "metrics": {
            name: _stats([item[name] for item in occurrences]) for name in metric_names
        },
        "occurrences": occurrences,
    }


def _classify_parallelism(
    baseline,
    candidate,
    degraded_ratio,
    improved_ratio,
    minimum_overlap_us,
):
    before = baseline["metrics"]["stream_pair_overlap_us"]["p50"]
    after = candidate["metrics"]["stream_pair_overlap_us"]["p50"]
    before_cv = baseline["metrics"]["cube_vector_overlap_us"]["p50"]
    after_cv = candidate["metrics"]["cube_vector_overlap_us"]["p50"]
    if before < minimum_overlap_us:
        return "preserved", "baseline_cross_stream_overlap_absent", None
    ratio = after / before
    cv_degraded = (
        before_cv >= minimum_overlap_us
        and before_cv - after_cv >= minimum_overlap_us
        and after_cv / before_cv <= degraded_ratio
    )
    if cv_degraded:
        return "degraded", "cube_vector_overlap_loss", None
    if ratio <= degraded_ratio and before - after >= minimum_overlap_us:
        return "degraded", "cross_stream_overlap_loss", None
    if ratio >= improved_ratio and after - before >= minimum_overlap_us:
        return "improved", "cross_stream_overlap_gain", None
    return "preserved", "cross_stream_overlap_preserved", None


def _actionability(baseline, candidate, effect, degraded_ratio, minimum_overlap_us):
    if effect == "unknown":
        return "blocked", "parallelism_evidence_incomplete"
    if effect != "degraded":
        return "no_action", None
    if (
        not baseline["cube_vector_identity_complete"]
        or not candidate["cube_vector_identity_complete"]
    ):
        return "blocked", "degraded_overlap_not_resource_complementary"
    before = baseline["metrics"]["cube_vector_overlap_us"]["p50"]
    after = candidate["metrics"]["cube_vector_overlap_us"]["p50"]
    if (
        before < minimum_overlap_us
        or before - after < minimum_overlap_us
        or after / before > degraded_ratio
    ):
        return "blocked", "degraded_overlap_not_resource_complementary"
    return "opportunity", None


def _diagnostic_decomposition(baseline, candidate):
    before = baseline["metrics"]
    after = candidate["metrics"]

    def delta(name):
        return after[name]["p50"] - before[name]["p50"]

    before_cv = before["cube_vector_overlap_us"]["p50"]
    after_cv = after["cube_vector_overlap_us"]["p50"]
    before_all = before["stream_pair_overlap_us"]["p50"]
    after_all = after["stream_pair_overlap_us"]["p50"]
    return {
        "interval_delta_us": delta("interval_us"),
        "child_work_inflation_us": delta("duration_sum_us"),
        "wait_sync_delta_us": delta("wait_sync_duration_us"),
        "lost_stream_pair_overlap_us": max(0.0, before_all - after_all),
        "lost_cube_vector_overlap_us": max(0.0, before_cv - after_cv),
        "cube_vector_overlap_retention_ratio": (
            after_cv / before_cv if before_cv > 0 else None
        ),
        "max_active_streams_delta": delta("max_active_streams"),
    }


def analyze(
    request_path,
    capture_path,
    artifact_root=None,
    *,
    degraded_ratio=0.5,
    improved_ratio=1.2,
    minimum_overlap_us=1.0,
):
    if (
        isinstance(degraded_ratio, bool)
        or not isinstance(degraded_ratio, (int, float))
        or not 0 <= degraded_ratio < 1
    ):
        raise ValueError("degraded_ratio must be in [0, 1)")
    if (
        isinstance(improved_ratio, bool)
        or not isinstance(improved_ratio, (int, float))
        or improved_ratio <= 1
    ):
        raise ValueError("improved_ratio must be greater than 1")
    if (
        isinstance(minimum_overlap_us, bool)
        or not isinstance(minimum_overlap_us, (int, float))
        or minimum_overlap_us <= 0
    ):
        raise ValueError("minimum_overlap_us must be greater than 0")
    request_path = Path(request_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else request_path.parent
    request = json.loads(request_path.read_text())
    request_summary = multistream_contract.validate_request(request, root)
    capture_path = _rooted(root, capture_path, "short trace capture")
    capture = json.loads(capture_path.read_text())
    required = {
        "schema_version",
        "capture_id",
        "trial_id",
        "request_fingerprint",
        "overflow_detected",
        "source_files",
        "targets",
        "capture_fingerprint",
    }
    if not isinstance(capture, dict) or set(capture) != required:
        raise ValueError(f"short trace capture must contain exactly {sorted(required)}")
    if capture["schema_version"] != CAPTURE_SCHEMA:
        raise ValueError(f"short trace capture must use {CAPTURE_SCHEMA}")
    if capture["capture_fingerprint"] != fingerprint(
        _unsigned(capture, "capture_fingerprint")
    ):
        raise ValueError("short trace capture fingerprint mismatch")
    if capture["request_fingerprint"] != request_summary["request_fingerprint"]:
        raise ValueError("short trace request fingerprint mismatch")
    _text(capture["capture_id"], "capture_id")
    trial_id = _text(capture["trial_id"], "trial_id")
    if not isinstance(capture["overflow_detected"], bool):
        raise ValueError("overflow_detected must be boolean")
    source_files = _validate_source_files(root, capture["source_files"])
    targets_by_id = {item["range_id"]: item for item in request["targets"]}
    raw_targets = capture["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("short trace targets must be non-empty")
    if len(raw_targets) != len(targets_by_id) or {
        item.get("range_id") for item in raw_targets if isinstance(item, dict)
    } != set(targets_by_id):
        raise ValueError("short trace target set differs from request")
    results = []
    blockers = []
    for index, target in enumerate(raw_targets):
        expected_fields = {
            "range_id",
            "graph_occurrence_fingerprint",
            "parent_identity",
            "baseline_occurrences",
            "candidate_occurrences",
        }
        if not isinstance(target, dict) or set(target) != expected_fields:
            raise ValueError(
                f"targets[{index}] must contain exactly {sorted(expected_fields)}"
            )
        range_id = _text(target["range_id"], f"targets[{index}].range_id")
        request_target = targets_by_id[range_id]
        if (
            target["graph_occurrence_fingerprint"]
            != request_target["graph_occurrence_fingerprint"]
        ):
            raise ValueError(f"target {range_id} graph occurrence fingerprint mismatch")
        parent = target["parent_identity"]
        if not isinstance(parent, dict) or set(parent) != {
            "device_id",
            "model_id",
            "parent_sk_id",
        }:
            raise ValueError(f"target {range_id} parent_identity is incomplete")
        for field in ("device_id", "model_id", "parent_sk_id"):
            if (
                isinstance(parent[field], bool)
                or not isinstance(parent[field], int)
                or parent[field] < 0
            ):
                raise ValueError(
                    f"target {range_id} parent_identity.{field} is invalid"
                )
        baseline = _summarize_occurrences(
            target["baseline_occurrences"], f"target {range_id} baseline"
        )
        candidate = _summarize_occurrences(
            target["candidate_occurrences"], f"target {range_id} candidate"
        )
        if baseline["alignment_ids"] != candidate["alignment_ids"]:
            raise ValueError(f"target {range_id} occurrence alignment differs")
        if baseline["child_origin_identities"] != candidate["child_origin_identities"]:
            raise ValueError(f"target {range_id} child origin identity set differs")
        effect, parallelism_basis, blocker = _classify_parallelism(
            baseline,
            candidate,
            degraded_ratio,
            improved_ratio,
            minimum_overlap_us,
        )
        if capture["overflow_detected"]:
            effect, parallelism_basis, blocker = (
                "unknown",
                "trace_overflow",
                "trace_overflow",
            )
        if blocker:
            blockers.append(f"{range_id}:{blocker}")
        optimization, actionability_blocker = _actionability(
            baseline,
            candidate,
            effect,
            degraded_ratio,
            minimum_overlap_us,
        )
        net_effect = request_target["net_effect"]
        latent_opportunity = net_effect == "beneficial" and effect == "degraded"
        results.append(
            {
                "range_id": range_id,
                "graph_occurrence_fingerprint": target["graph_occurrence_fingerprint"],
                "parent_identity": parent,
                "aligned_occurrence_count": baseline["occurrence_count"],
                "net_effect": net_effect,
                "parallelism_effect": effect,
                "parallelism_basis": parallelism_basis,
                "optimization_status": optimization,
                "latent_opportunity": latent_opportunity,
                "actionable_opportunity": optimization == "opportunity",
                "opportunity_kind": (
                    "beneficial_with_degraded_parallelism"
                    if latent_opportunity
                    else "degraded_parallelism"
                    if effect == "degraded"
                    else None
                ),
                "blocker": blocker,
                "actionability_blocker": actionability_blocker,
                "diagnostic_decomposition": _diagnostic_decomposition(
                    baseline, candidate
                ),
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "request_id": request["request_id"],
        "request_fingerprint": request_summary["request_fingerprint"],
        "trial_id": trial_id,
        "capture": {
            "path": _relative(root, capture_path),
            "file_fingerprint": file_fingerprint(capture_path),
            "capture_id": capture["capture_id"],
            "capture_fingerprint": capture["capture_fingerprint"],
            "overflow_detected": capture["overflow_detected"],
            "source_files": source_files,
        },
        "thresholds": {
            "degraded_ratio": degraded_ratio,
            "improved_ratio": improved_ratio,
            "minimum_overlap_us": minimum_overlap_us,
            "minimum_aligned_occurrences": 3,
        },
        "targets": sorted(results, key=lambda item: item["range_id"]),
        "blockers": sorted(blockers),
    }
    analysis["analysis_fingerprint"] = fingerprint(analysis)
    return analysis


def validate_analysis(
    path, request_path, artifact_root=None, *, expected_trial_id=None
):
    path = Path(path).resolve()
    root = (
        Path(artifact_root).resolve()
        if artifact_root
        else Path(request_path).resolve().parent
    )
    value = json.loads(path.read_text())
    validate_bound_analysis(
        path,
        root,
        expected_request_fingerprint=value.get("request_fingerprint"),
        expected_trial_id=expected_trial_id,
    )
    if not isinstance(value, dict) or value.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError(f"trace analysis must use {ANALYSIS_SCHEMA}")
    if value.get("analysis_fingerprint") != fingerprint(
        _unsigned(value, "analysis_fingerprint")
    ):
        raise ValueError("trace analysis fingerprint mismatch")
    if expected_trial_id is not None and value.get("trial_id") != expected_trial_id:
        raise ValueError("trace analysis trial_id mismatch")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("trace analysis thresholds are missing")
    rebuilt = analyze(
        request_path,
        value.get("capture", {}).get("path"),
        root,
        degraded_ratio=thresholds.get("degraded_ratio"),
        improved_ratio=thresholds.get("improved_ratio"),
        minimum_overlap_us=thresholds.get("minimum_overlap_us"),
    )
    if _canonical(rebuilt) != _canonical(value):
        raise ValueError("trace analysis differs from deterministic replay")
    return {
        "valid": True,
        "trial_id": value["trial_id"],
        "overflow_detected": value["capture"]["overflow_detected"],
        "minimum_aligned_occurrence_count": min(
            item["aligned_occurrence_count"] for item in value["targets"]
        ),
        "blockers": value["blockers"],
        "analysis_fingerprint": value["analysis_fingerprint"],
    }


def validate_bound_analysis(
    path, artifact_root, *, expected_request_fingerprint, expected_trial_id=None
):
    root = Path(artifact_root).resolve()
    path = _rooted(root, path, "trace analysis")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError(f"trace analysis must use {ANALYSIS_SCHEMA}")
    if value.get("analysis_fingerprint") != fingerprint(
        _unsigned(value, "analysis_fingerprint")
    ):
        raise ValueError("trace analysis fingerprint mismatch")
    if value.get("request_fingerprint") != expected_request_fingerprint:
        raise ValueError("trace analysis request fingerprint mismatch")
    if expected_trial_id is not None and value.get("trial_id") != expected_trial_id:
        raise ValueError("trace analysis trial_id mismatch")
    capture = value.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("trace analysis capture binding is missing")
    capture_path = _rooted(root, capture.get("path"), "trace analysis capture")
    if file_fingerprint(capture_path) != capture.get("file_fingerprint"):
        raise ValueError("trace analysis capture file changed")
    _validate_source_files(root, capture.get("source_files"))
    targets = value.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("trace analysis targets must be non-empty")
    if any(
        item.get("parallelism_effect") not in PARALLELISM_EFFECTS
        or not isinstance(item.get("aligned_occurrence_count"), int)
        or item["aligned_occurrence_count"] < 3
        for item in targets
        if isinstance(item, dict)
    ) or any(not isinstance(item, dict) for item in targets):
        raise ValueError("trace analysis target evidence is invalid")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--request", type=Path, required=True)
    analyze_parser.add_argument("--capture", type=Path, required=True)
    analyze_parser.add_argument("--artifact-root", type=Path)
    analyze_parser.add_argument("--degraded-ratio", type=float, default=0.5)
    analyze_parser.add_argument("--improved-ratio", type=float, default=1.2)
    analyze_parser.add_argument("--minimum-overlap-us", type=float, default=1.0)
    analyze_parser.add_argument("--out", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--request", type=Path, required=True)
    validate_parser.add_argument("--analysis", type=Path, required=True)
    validate_parser.add_argument("--artifact-root", type=Path)
    validate_parser.add_argument("--trial-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            if not 0 <= args.degraded_ratio < 1:
                raise ValueError("degraded_ratio must be in [0, 1)")
            if args.improved_ratio <= 1:
                raise ValueError("improved_ratio must be greater than 1")
            if args.minimum_overlap_us <= 0:
                raise ValueError("minimum_overlap_us must be greater than 0")
            result = analyze(
                args.request,
                args.capture,
                args.artifact_root,
                degraded_ratio=args.degraded_ratio,
                improved_ratio=args.improved_ratio,
                minimum_overlap_us=args.minimum_overlap_us,
            )
            if args.out.exists():
                raise ValueError(f"trace analysis output already exists: {args.out}")
            _atomic_json(args.out, result)
        else:
            result = validate_analysis(
                args.analysis,
                args.request,
                args.artifact_root,
                expected_trial_id=args.trial_id,
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
