#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Render a deterministic Chinese SuperKernel fusion performance report."""

import argparse
import hashlib
import html
import json
import math
import os
import re
import tempfile
from pathlib import Path


MISSING = "N/A"
EMPTY = "无"
SUPPORTED_SCHEMA_VERSIONS = {"1.2"}
MAPPING_METHODS = {
    "sk_meta_node_ids",
    "source_scope_map",
    "kernel_projection_structural",
}
MAPPING_CONFIDENCES = {
    "exact",
    "exact_projected_trace",
    "diagnostic_only",
    "ambiguous",
    "unmapped",
}
MARKDOWN_SPECIAL_CHARACTERS = frozenset(r"\`*_{}[]()#+-.!|")
LIST_OF_OBJECT_FIELDS = (
    "per_sk_decisions",
    "scope_actions",
    "diagnostic_hypotheses",
    "recommended_experiments",
)
CONDITIONAL_BINDING_REPORT_FIELDS = (
    "source_revision",
    "baseline_config_fingerprint",
    "candidate_config_fingerprint",
    "control_fingerprint",
    "workload_fingerprint",
)
OPTIONAL_TEXT_FIELDS = ("analysis_agent_id",) + CONDITIONAL_BINDING_REPORT_FIELDS
TOP_LEVEL_TEXT_FIELDS = (
    "experiment_id",
    "round_id",
    "candidate_name",
    "next_agent_guidance_zh",
) + OPTIONAL_TEXT_FIELDS
REQUIRED_IDENTITY_FIELDS = (
    "analysis_id",
    "experiment_id",
    "round_id",
    "analysis_agent_id",
    "candidate_name",
    "source_revision",
)
ANALYSIS_IDENTITY_FIELDS = REQUIRED_IDENTITY_FIELDS[1:]
REQUIRED_FINGERPRINT_FIELDS = (
    "baseline_profile_fingerprint",
    "candidate_profile_fingerprint",
    "baseline_config_fingerprint",
    "candidate_config_fingerprint",
    "baseline_workload_fingerprint",
    "candidate_workload_fingerprint",
    "workload_fingerprint",
    "baseline_control_fingerprint",
    "candidate_control_fingerprint",
    "control_fingerprint",
)
PROFILE_COMPARISON_INPUT_FIELDS = (
    "baseline_profile",
    "candidate_profile",
    "sk_meta",
    "baseline_config",
    "candidate_config",
    "baseline_workload",
    "candidate_workload",
    "declared_change_set",
    "sk_prof",
    "environment_evidence",
    "source_scope_map",
)
PROFILE_STRUCTURAL_INPUT_FIELDS = (
    "baseline_collection_manifest",
    "profile_collection_manifest",
)
STRUCTURAL_REPORT_FIELDS = (
    "association_protocol",
    "candidate_binding_evidence",
    "canonical_graph_fingerprints",
    "stream_role_mapping",
    "graph_alignment_proof",
    "mapping_coverage",
)
STRUCTURAL_HARD_CONSTRAINTS = {
    "one_to_one",
    "ordered",
    "op",
    "core_family",
    "shape_dtype",
    "dependency_edges",
}
CLASSIFICATION_ACTIONS = {
    "beneficial": frozenset(("keep",)),
    "neutral": frozenset(("prune",)),
    "regressed": frozenset(("prune",)),
    "insufficient_evidence": frozenset(("reprofile", "block")),
}
DECISION_TEXT_FIELDS = (
    "range_id",
    "sk_id",
    "classification",
    "classification_zh",
    "action",
    "action_blocker",
    "mapping_method",
    "mapping_confidence",
)
DECISION_NUMBER_FIELDS = (
    "child_count",
    "sk_duration_us",
    "candidate_occurrence_count",
    "improvement_pct",
    "noise_threshold_pct",
)
STATISTIC_FIELDS = ("count", "p50", "p90", "mad", "min", "max", "mean")
BOUNDARY_FIELDS = ("scope", "start_function", "end_function", "start_op", "end_op")
OVERLAP_NUMBER_FIELDS = (
    "cube_vector_overlap_us",
    "cube_vector_overlap_ratio",
    "mix_competition_overlap_us",
    "same_resource_overlap_us",
)


def _display(value):
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return MISSING
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return str(value)


def _validate_json_value(value, path, active_containers=None):
    """Reject values that cannot be rendered as finite, deterministic JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must contain only standard JSON values")

    active_containers = active_containers or set()
    identity = id(value)
    if identity in active_containers:
        raise ValueError(f"{path} must not contain cyclic values")
    active_containers.add(identity)
    try:
        if isinstance(value, list):
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", active_containers)
            return
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            _validate_json_value(item, f"{path}.{key}", active_containers)
    finally:
        active_containers.remove(identity)


def _validate_text_field(container, field, path):
    value = container.get(field)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{path}.{field} must be a string or null")


def _validate_number_field(container, field, path):
    value = container.get(field)
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError(f"{path}.{field} must be a number or null")


def _validate_bool_field(container, field, path):
    value = container.get(field)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{path}.{field} must be a boolean or null")


def _optional_object(container, field, path):
    value = container.get(field)
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"{path}.{field} must be an object or null")
    return value


def _optional_list(container, field, path):
    value = container.get(field)
    if value is not None and not isinstance(value, list):
        raise ValueError(f"{path}.{field} must be a list or null")
    return value


def _validate_text_list(value, path):
    if value is None:
        return
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{path}[{index}] must be a string")


def _validate_scalar_list(value, path):
    if value is None:
        return
    for index, item in enumerate(value):
        if isinstance(item, (dict, list)):
            raise ValueError(f"{path}[{index}] must be a scalar")


def _validate_statistics(statistics, path):
    if statistics is None:
        return
    for field in STATISTIC_FIELDS:
        _validate_number_field(statistics, field, path)


def _validate_boundary(boundary, path):
    if boundary is None:
        return
    for field in BOUNDARY_FIELDS:
        _validate_text_field(boundary, field, path)


def _source_interval_proven(boundary, path):
    fields = ("source_file", "start_offset", "end_offset")
    present = [field for field in fields if field in boundary]
    if not present:
        return False
    if len(present) != len(fields):
        raise ValueError(
            f"{path} source interval must contain source_file, start_offset, "
            "and end_offset together"
        )
    source_file = boundary["source_file"]
    if (
        not isinstance(source_file, str)
        or not source_file.strip()
        or Path(source_file).is_absolute()
        or ".." in Path(source_file).parts
    ):
        raise ValueError(f"{path}.source_file must be a non-empty relative source path")
    start_offset = boundary["start_offset"]
    end_offset = boundary["end_offset"]
    if (
        isinstance(start_offset, bool)
        or not isinstance(start_offset, int)
        or start_offset < 0
    ):
        raise ValueError(f"{path}.start_offset must be a non-negative integer")
    if (
        isinstance(end_offset, bool)
        or not isinstance(end_offset, int)
        or end_offset <= start_offset
    ):
        raise ValueError(
            f"{path}.end_offset must be an integer greater than start_offset"
        )
    return True


def _validate_scope_action(action, path):
    required_fields = (
        "range_id",
        "sk_id",
        "classification",
        "action",
        "source_scope",
        "boundary",
        "ordered_child_op_sequence",
        "interval_unproven",
    )
    for field in required_fields:
        if field not in action:
            raise ValueError(f"{path}.{field} is required")
    for field in (
        "range_id",
        "sk_id",
        "classification",
        "action",
        "source_scope",
    ):
        value = action[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}.{field} must be a non-empty string")

    boundary = action["boundary"]
    if not isinstance(boundary, dict):
        raise ValueError(f"{path}.boundary must be an object")
    _validate_boundary(boundary, f"{path}.boundary")
    for field in ("start_op", "end_op"):
        value = boundary.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}.boundary.{field} must be a non-empty string")

    sequence = action["ordered_child_op_sequence"]
    if (
        not isinstance(sequence, list)
        or not sequence
        or any(not isinstance(item, str) or not item.strip() for item in sequence)
    ):
        raise ValueError(
            f"{path}.ordered_child_op_sequence must be a non-empty string list"
        )
    interval_unproven = action["interval_unproven"]
    if not isinstance(interval_unproven, bool):
        raise ValueError(f"{path}.interval_unproven must be a boolean")
    interval_proven = _source_interval_proven(boundary, f"{path}.boundary")
    if interval_unproven == interval_proven:
        raise ValueError(
            f"{path}.interval_unproven does not match boundary source interval proof"
        )


def _required_text(container, field, path):
    value = container.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{field} must be a non-empty string")
    return value


def _decision_scope_contract(decision, path):
    sk_id = _required_text(decision, "sk_id", path)
    range_id = _required_text(decision, "range_id", path)
    classification = _required_text(decision, "classification", path)
    action = _required_text(decision, "action", path)
    allowed_actions = CLASSIFICATION_ACTIONS.get(classification)
    if classification in {"neutral", "regressed"}:
        allowed_actions = {"block", "prune"}
    if allowed_actions is None:
        raise ValueError(f"{path}.classification is not a schema 1.2 enum")
    if action not in allowed_actions:
        expected = "/".join(sorted(allowed_actions))
        raise ValueError(
            f"{path}.action must be {expected} for classification {classification}"
        )

    boundary = decision.get("boundary")
    if not isinstance(boundary, dict):
        raise ValueError(f"{path}.boundary must be an object")
    for field in ("start_op", "end_op"):
        _required_text(boundary, field, f"{path}.boundary")
    interval_unproven = not _source_interval_proven(boundary, f"{path}.boundary")
    if classification in {"neutral", "regressed"}:
        source_actionable = (
            decision.get("mapping_method") == "source_scope_map"
            and decision.get("mapping_confidence") == "exact"
            and decision.get("mapping_blockers") == []
            and not interval_unproven
        )
        expected_action = "prune" if source_actionable else "block"
        if action != expected_action:
            raise ValueError(
                f"{path}.action must be {expected_action}; schema 1.2 prune "
                "requires exact source interval mapping"
            )
        if action == "block" and decision.get("action_blocker") != (
            "prune_requires_exact_source_scope_boundary_mapping"
        ):
            raise ValueError(
                f"{path}.action_blocker must explain the missing exact source "
                "interval mapping"
            )

    identity = decision.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{path}.identity must be an object")
    source_scope = _required_text(identity, "source_scope", f"{path}.identity")
    sequence = identity.get("ordered_child_op_sequence")
    if (
        not isinstance(sequence, list)
        or not sequence
        or any(not isinstance(item, str) or not item.strip() for item in sequence)
    ):
        raise ValueError(
            f"{path}.identity.ordered_child_op_sequence must be a non-empty string list"
        )
    return {
        "sk_id": sk_id,
        "range_id": range_id,
        "classification": classification,
        "action": action,
        "source_scope": source_scope,
        "boundary": boundary,
        "ordered_child_op_sequence": sequence,
        "interval_unproven": interval_unproven,
    }


def _scope_action_contract(action, _path):
    return {
        field: action[field]
        for field in (
            "sk_id",
            "range_id",
            "classification",
            "action",
            "source_scope",
            "boundary",
            "ordered_child_op_sequence",
            "interval_unproven",
        )
    }


def _unique_scope_contracts(items, path, contract):
    by_key = {}
    seen_sk_ids = set()
    seen_range_ids = set()
    for index, item in enumerate(items):
        location = f"{path}[{index}]"
        projected = contract(item, location)
        key = (projected["sk_id"], projected["range_id"])
        if (
            key in by_key
            or projected["sk_id"] in seen_sk_ids
            or projected["range_id"] in seen_range_ids
        ):
            raise ValueError(f"{location} has duplicate sk_id/range_id identity")
        by_key[key] = projected
        seen_sk_ids.add(projected["sk_id"])
        seen_range_ids.add(projected["range_id"])
    return by_key


def _validate_decision(decision, path):
    for field in DECISION_TEXT_FIELDS:
        _validate_text_field(decision, field, path)
    for field in DECISION_NUMBER_FIELDS:
        _validate_number_field(decision, field, path)

    evidence_errors = _optional_list(decision, "evidence_errors", path)
    _validate_text_list(evidence_errors, f"{path}.evidence_errors")

    boundary = _optional_object(decision, "boundary", path)
    _validate_boundary(boundary, f"{path}.boundary")

    identity = _optional_object(decision, "identity", path)
    if identity is not None:
        identity_boundary = _optional_object(identity, "boundary", f"{path}.identity")
        _validate_boundary(identity_boundary, f"{path}.identity.boundary")
        sequence = _optional_list(
            identity, "ordered_child_op_sequence", f"{path}.identity"
        )
        _validate_text_list(sequence, f"{path}.identity.ordered_child_op_sequence")

    sk_duration = _optional_object(decision, "sk_duration", path)
    _validate_statistics(sk_duration, f"{path}.sk_duration")

    original = _optional_object(decision, "original", path)
    if original is not None:
        _validate_number_field(original, "occurrence_count", f"{path}.original")
        for field in ("interval_us", "duration_sum_us"):
            statistics = _optional_object(original, field, f"{path}.original")
            _validate_statistics(statistics, f"{path}.original.{field}")
        occurrence = _optional_object(
            original, "example_occurrence", f"{path}.original"
        )
        if occurrence is not None:
            _validate_number_field(
                occurrence, "stream_count", f"{path}.original.example_occurrence"
            )
            stream_ids = _optional_list(
                occurrence, "stream_ids", f"{path}.original.example_occurrence"
            )
            _validate_scalar_list(
                stream_ids, f"{path}.original.example_occurrence.stream_ids"
            )
            overlap = _optional_object(
                occurrence,
                "multi_stream_analysis",
                f"{path}.original.example_occurrence",
            )
            if overlap is not None:
                for field in (
                    "multi_stream_detected",
                    "cube_vector_parallel_detected",
                ):
                    _validate_bool_field(
                        overlap,
                        field,
                        f"{path}.original.example_occurrence.multi_stream_analysis",
                    )
                for field in OVERLAP_NUMBER_FIELDS:
                    _validate_number_field(
                        overlap,
                        field,
                        f"{path}.original.example_occurrence.multi_stream_analysis",
                    )

    missing_fields = _missing_decision_fields(decision)
    if missing_fields and not (evidence_errors or decision.get("action_blocker")):
        raise ValueError(
            f"{path} has missing required report fields and requires evidence_errors "
            "or action_blocker"
        )


def _validate_list_item_fields(items, path, text_fields=(), bool_fields=()):
    for index, item in enumerate(items or []):
        item_path = f"{path}[{index}]"
        for field in text_fields:
            _validate_text_field(item, field, item_path)
        for field in bool_fields:
            _validate_bool_field(item, field, item_path)


def _value_or_na(value):
    return MISSING if value is None else value


def _nested_value(container, *fields):
    value = container
    for field in fields:
        if not isinstance(value, dict):
            return None
        value = value.get(field)
    return value


def _missing_decision_fields(decision):
    required = {
        "range_id": decision.get("range_id"),
        "sk_id": decision.get("sk_id"),
        "child_count": decision.get("child_count"),
        "original.interval_us.p50": _nested_value(
            decision, "original", "interval_us", "p50"
        ),
        "original.interval_us.p90": _nested_value(
            decision, "original", "interval_us", "p90"
        ),
        "original.interval_us.mad": _nested_value(
            decision, "original", "interval_us", "mad"
        ),
        "original.duration_sum_us.p50": _nested_value(
            decision, "original", "duration_sum_us", "p50"
        ),
        "original.duration_sum_us.p90": _nested_value(
            decision, "original", "duration_sum_us", "p90"
        ),
        "original.duration_sum_us.mad": _nested_value(
            decision, "original", "duration_sum_us", "mad"
        ),
        "sk_duration.p50": _nested_value(decision, "sk_duration", "p50"),
        "sk_duration.p90": _nested_value(decision, "sk_duration", "p90"),
        "sk_duration.mad": _nested_value(decision, "sk_duration", "mad"),
        "original.occurrence_count": _nested_value(
            decision, "original", "occurrence_count"
        ),
        "candidate_occurrence_count": decision.get("candidate_occurrence_count"),
        "classification": decision.get("classification"),
        "classification_zh": decision.get("classification_zh"),
        "action": decision.get("action"),
        "mapping_method": decision.get("mapping_method"),
        "mapping_confidence": decision.get("mapping_confidence"),
        "boundary": decision.get("boundary"),
        "identity.ordered_child_op_sequence": _nested_value(
            decision, "identity", "ordered_child_op_sequence"
        ),
        "original.example_occurrence.stream_ids": _nested_value(
            decision, "original", "example_occurrence", "stream_ids"
        ),
        "original.example_occurrence.stream_count": _nested_value(
            decision, "original", "example_occurrence", "stream_count"
        ),
        "original.example_occurrence.multi_stream_analysis": _nested_value(
            decision, "original", "example_occurrence", "multi_stream_analysis"
        ),
    }
    return sorted(
        field
        for field, value in required.items()
        if value is None
        or value == ""
        or (isinstance(value, (dict, list)) and not value)
    )


def _mapping_reliable(decision):
    confidence = decision.get("mapping_confidence")
    if confidence in {"exact", "exact_projected_trace"}:
        return True
    if confidence in {"diagnostic_only", "ambiguous", "unmapped"}:
        return False
    return MISSING


def _occurrence_counts(decision):
    return {
        "baseline": _value_or_na(
            _nested_value(decision, "original", "occurrence_count")
        ),
        "SK": _value_or_na(decision.get("candidate_occurrence_count")),
    }


def _stream_overlap(decision):
    occurrence = _nested_value(decision, "original", "example_occurrence") or {}
    overlap = occurrence.get("multi_stream_analysis") or {}
    return {
        "stream_ids": _value_or_na(occurrence.get("stream_ids")),
        "stream_count": _value_or_na(occurrence.get("stream_count")),
        "multi_stream_detected": _value_or_na(overlap.get("multi_stream_detected")),
        "cube_vector_parallel_detected": _value_or_na(
            overlap.get("cube_vector_parallel_detected")
        ),
        "cube_vector_overlap_us": _value_or_na(overlap.get("cube_vector_overlap_us")),
        "cube_vector_overlap_ratio": _value_or_na(
            overlap.get("cube_vector_overlap_ratio")
        ),
    }


def _decision_evidence(decision):
    evidence = {}
    missing_fields = _missing_decision_fields(decision)
    if missing_fields:
        evidence["missing_fields"] = missing_fields
    if decision.get("evidence_errors"):
        evidence["evidence_errors"] = decision["evidence_errors"]
    if decision.get("action_blocker"):
        evidence["action_blocker"] = decision["action_blocker"]
    return evidence or EMPTY


def _markdown(value):
    text = html.escape(_display(value), quote=False)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    escaped = "".join(
        f"\\{character}" if character in MARKDOWN_SPECIAL_CHARACTERS else character
        for character in text
    )
    return escaped.replace("\n", "<br>")


def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown(value) for value in row) + " |" for row in rows
    )
    return lines


def _sorted_by_range(items):
    return sorted(
        items or [],
        key=lambda item: (
            str(item.get("range_id") or ""),
            str(item.get("sk_id") or ""),
            str(item.get("kind") or item.get("option") or ""),
        ),
    )


def _section_or_empty(lines, content):
    lines.extend(content if content else [f"- {EMPTY}"])


def _conditional_evidence_binding(report, range_id):
    fields = [(field, report.get(field)) for field in CONDITIONAL_BINDING_REPORT_FIELDS]
    fields.append(("range_id", range_id))
    return "; ".join(f"{field}={_display(value)}" for field, value in fields)


def _structural_identity_key(decision):
    return (
        f"device:{decision.get('device_id')}/model:{decision.get('model_id')}"
        f"/sk:{decision.get('raw_sk_id')}"
    )


def _structural_source_action(decision):
    if (
        decision.get("mapping_method") == "source_scope_map"
        and decision.get("mapping_confidence") == "exact"
    ):
        return decision.get("action")
    if decision.get("mapping_confidence") == "exact_projected_trace":
        return "仅性能分类；不可直接 prune"
    return "无自动源码动作"


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _structural_proof_fingerprint(proof):
    payload = {
        field: proof.get(field)
        for field in (
            "mapping_method",
            "mapping_confidence",
            "baseline_graph_fingerprint",
            "binding_evidence_fingerprints",
            "graph_occurrence_fingerprint",
            "canonical_baseline_child_keys",
            "alternative_solution_count",
            "three_graph_consistent",
            "roots",
            "mapping_blockers",
        )
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _structural_object(report, field):
    value = report[field]
    if not isinstance(value, dict):
        raise ValueError(f"report.{field} must be an object for schema 1.2")
    return value


def _validate_candidate_binding_occurrence(item, path):
    occurrence_fields = {
        "status",
        "process_identity",
        "step_id",
        "parent_interval_ns",
        "ordered_child_node_keys",
        "ordered_child_ops",
        "child_count",
        "evidence_fingerprints",
        "blockers",
    }
    if not isinstance(item, dict) or set(item) != occurrence_fields:
        raise ValueError(f"{path} projected-trace binding contract is invalid")
    status = item.get("status")
    if not isinstance(status, str) or status not in {
        "bound",
        "ambiguous",
        "blocked",
    }:
        raise ValueError(f"{path} projected-trace binding status is invalid")
    process = item.get("process_identity")
    process_fields = {"native_pid", "device_id", "model_id", "sk_id"}
    if (
        not isinstance(process, dict)
        or set(process) != process_fields
        or any(
            isinstance(process[field], bool)
            or not isinstance(process[field], int)
            or process[field] < 0
            for field in process_fields
        )
    ):
        raise ValueError(f"{path} projected-trace process identity is invalid")
    child_count = item.get("child_count")
    child_keys = item.get("ordered_child_node_keys")
    child_ops = item.get("ordered_child_ops")
    if (
        isinstance(child_count, bool)
        or not isinstance(child_count, int)
        or child_count < 0
        or not isinstance(child_keys, list)
        or not isinstance(child_ops, list)
        or any(
            not isinstance(value, str) or not value for value in child_keys + child_ops
        )
        or len(child_keys) != child_count
        or len(child_ops) != child_count
    ):
        raise ValueError(f"{path} projected-trace child binding is invalid")
    blockers = item.get("blockers")
    if (
        not isinstance(blockers, list)
        or any(not isinstance(blocker, str) or not blocker for blocker in blockers)
        or len(blockers) != len(set(blockers))
        or (status == "bound" and blockers)
        or (status != "bound" and not blockers)
    ):
        raise ValueError(f"{path} projected-trace binding blockers are invalid")
    step_id = item.get("step_id")
    interval = item.get("parent_interval_ns")
    if (
        isinstance(step_id, bool)
        or not isinstance(step_id, int)
        or step_id < 0
        or not isinstance(interval, list)
        or len(interval) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in interval
        )
        or interval[0] < 0
        or interval[0] >= interval[1]
    ):
        raise ValueError(f"{path} projected-trace occurrence timing is invalid")
    fingerprints = item.get("evidence_fingerprints")
    fingerprint_fields = {
        "kernel_details",
        "projection_mapping",
        "profile_sk_graph_origin",
    }
    if not isinstance(fingerprints, dict) or set(fingerprints) != fingerprint_fields:
        raise ValueError(f"{path} projected-trace evidence fingerprints are invalid")
    for value in fingerprints.values():
        if value is None and status != "bound":
            continue
        if not _is_sha256(value):
            raise ValueError(
                f"{path} projected-trace evidence fingerprints are invalid"
            )


def _validate_mapping_coverage(report):
    coverage = _structural_object(report, "mapping_coverage")
    count_fields = {
        "total_sk_ids",
        "bound_sk_ids",
        "exact_projected_trace_sk_ids",
        "ambiguous_sk_ids",
        "unmapped_sk_ids",
        "filtered_by_child_count",
    }
    if set(coverage) != count_fields | {
        "child_count_distribution",
        "blocker_counts",
    }:
        raise ValueError("report.mapping_coverage must contain schema 1.2 totals")
    for field in count_fields:
        if type(coverage[field]) is not int or coverage[field] < 0:
            raise ValueError(
                f"report.mapping_coverage.{field} must be a non-negative integer"
            )
    if coverage["filtered_by_child_count"] != 0:
        raise ValueError("report.mapping_coverage.filtered_by_child_count must equal 0")
    if coverage["total_sk_ids"] != (
        coverage["exact_projected_trace_sk_ids"]
        + coverage["ambiguous_sk_ids"]
        + coverage["unmapped_sk_ids"]
    ):
        raise ValueError("report.mapping_coverage confidence totals do not add up")
    if coverage["bound_sk_ids"] > coverage["total_sk_ids"]:
        raise ValueError("report.mapping_coverage.bound_sk_ids exceeds total_sk_ids")

    distribution = coverage["child_count_distribution"]
    if not isinstance(distribution, dict) or any(
        not isinstance(bucket, str) or not bucket or type(count) is not int or count < 0
        for bucket, count in distribution.items()
    ):
        raise ValueError(
            "report.mapping_coverage.child_count_distribution must contain counts"
        )
    if sum(distribution.values()) != coverage["total_sk_ids"]:
        raise ValueError(
            "report.mapping_coverage.child_count_distribution must cover every SK"
        )
    blocker_counts = coverage["blocker_counts"]
    if not isinstance(blocker_counts, dict) or any(
        not isinstance(blocker, str)
        or not blocker
        or type(count) is not int
        or count < 0
        for blocker, count in blocker_counts.items()
    ):
        raise ValueError("report.mapping_coverage.blocker_counts must contain counts")

    decisions = report["per_sk_decisions"]
    inventoried = {}
    for decision in decisions:
        inventory_key = (
            decision.get("device_id"),
            str(decision.get("model_id")),
            decision.get("raw_sk_id"),
        )
        if (
            isinstance(inventory_key[0], bool)
            or not isinstance(inventory_key[0], int)
            or inventory_key[1] in {"", "None"}
            or isinstance(inventory_key[2], bool)
            or not isinstance(inventory_key[2], int)
        ):
            continue
        inventoried.setdefault(inventory_key, decision)
    inventory = list(inventoried.values())
    exact_count = sum(
        item.get("mapping_confidence") == "exact_projected_trace" for item in inventory
    )
    ambiguous_count = sum(
        item.get("mapping_confidence") == "ambiguous" for item in inventory
    )
    bound_count = sum(
        item.get("candidate_binding_status") == "bound" for item in inventory
    )
    expected_distribution = {}
    expected_blockers = {}
    for decision in inventory:
        child_count = decision.get("child_count")
        if type(child_count) is int:
            bucket = "5+" if child_count >= 5 else str(child_count)
            expected_distribution[bucket] = expected_distribution.get(bucket, 0) + 1
        for blocker in set(decision.get("mapping_blockers") or ()):
            expected_blockers[blocker] = expected_blockers.get(blocker, 0) + 1
    if (
        coverage["total_sk_ids"] != len(inventory)
        or coverage["bound_sk_ids"] != bound_count
        or coverage["exact_projected_trace_sk_ids"] != exact_count
        or coverage["ambiguous_sk_ids"] != ambiguous_count
        or coverage["unmapped_sk_ids"] != len(inventory) - exact_count - ambiguous_count
        or coverage["child_count_distribution"] != expected_distribution
        or coverage["blocker_counts"] != expected_blockers
    ):
        raise ValueError("report.mapping_coverage does not match per_sk_decisions")


def _validate_exact_projected_trace_decision(report, decision, path):
    key = _structural_identity_key(decision)
    if (
        type(decision.get("device_id")) is not int
        or type(decision.get("raw_sk_id")) is not int
        or not str(decision.get("model_id") or "").isdecimal()
        or decision.get("candidate_binding_status") != "bound"
        or decision.get("mapping_blockers") != []
    ):
        raise ValueError(f"{path} projected-trace native identity is invalid")

    binding_evidence = report["candidate_binding_evidence"]
    summary = binding_evidence.get("summary")
    occurrences = binding_evidence.get("occurrences")
    if (
        not isinstance(summary, dict)
        or set(summary) != {"protocol", "parent_occurrences", "status_counts"}
        or summary.get("protocol") != "kernel_projection_trace_v2"
        or type(summary.get("parent_occurrences")) is not int
        or not isinstance(summary.get("status_counts"), dict)
        or not isinstance(occurrences, list)
        or summary["parent_occurrences"] != len(occurrences)
    ):
        raise ValueError(f"{path} projected-trace binding summary is invalid")

    matching = []
    observed_status_counts = {}
    occurrence_keys = set()
    for index, occurrence in enumerate(occurrences):
        occurrence_path = f"report.candidate_binding_evidence.occurrences[{index}]"
        _validate_candidate_binding_occurrence(occurrence, occurrence_path)
        status = occurrence["status"]
        observed_status_counts[status] = observed_status_counts.get(status, 0) + 1
        process = occurrence["process_identity"]
        occurrence_key = (
            process["native_pid"],
            process["device_id"],
            process["model_id"],
            process["sk_id"],
            occurrence["step_id"],
        )
        if occurrence_key in occurrence_keys:
            raise ValueError(f"{path} projected-trace duplicate occurrence")
        occurrence_keys.add(occurrence_key)
        if (
            process["device_id"] == decision["device_id"]
            and str(process["model_id"]) == str(decision["model_id"])
            and process["sk_id"] == decision["raw_sk_id"]
        ):
            matching.append(occurrence)
    if summary["status_counts"] != dict(sorted(observed_status_counts.items())):
        raise ValueError(f"{path} projected-trace status counts do not match")
    if (
        len(matching) < 3
        or len({item["step_id"] for item in matching}) != len(matching)
        or len({item["process_identity"]["native_pid"] for item in matching}) != 1
        or any(
            item["status"] != "bound"
            or item["blockers"] != []
            or item["child_count"] != decision.get("child_count")
            or item["ordered_child_ops"]
            != (decision.get("identity") or {}).get("ordered_child_op_sequence")
            for item in matching
        )
    ):
        raise ValueError(f"{path} projected-trace occurrence evidence is incomplete")

    fingerprints = report["canonical_graph_fingerprints"].get(key)
    fingerprint_fields = {
        "baseline_kernel_details_sha256",
        "profile_manifest_fingerprint",
        "profile_origin_graph_fingerprint",
    }
    if (
        not isinstance(fingerprints, dict)
        or set(fingerprints) != fingerprint_fields
        or any(not _is_sha256(value) for value in fingerprints.values())
    ):
        raise ValueError(f"{path} projected-trace fingerprints are invalid")

    proof = report["graph_alignment_proof"].get(key)
    proof_fields = {
        "protocol",
        "mapping_method",
        "mapping_confidence",
        "mapping_fingerprint",
        "graph_occurrence_fingerprint",
        "alternative_solution_count_by_step",
    }
    alternatives = (
        proof.get("alternative_solution_count_by_step")
        if isinstance(proof, dict)
        else None
    )
    matching_steps = {str(item["step_id"]) for item in matching}
    if (
        not isinstance(proof, dict)
        or set(proof) != proof_fields
        or proof.get("protocol") != "kernel_projection_trace_v2"
        or proof.get("mapping_method") != "kernel_projection_structural"
        or proof.get("mapping_confidence") != "exact_projected_trace"
        or not _is_sha256(proof.get("mapping_fingerprint"))
        or not _is_sha256(proof.get("graph_occurrence_fingerprint"))
        or not isinstance(alternatives, dict)
        or set(alternatives) != matching_steps
        or any(type(value) is not int or value != 0 for value in alternatives.values())
    ):
        raise ValueError(f"{path} projected-trace proof is invalid")

    stream_roles = report["stream_role_mapping"].get(key)
    if not isinstance(stream_roles, dict) or set(stream_roles) != matching_steps:
        raise ValueError(f"{path} projected-trace stream roles are incomplete")
    for step, mapping in stream_roles.items():
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(source, str)
                or not source
                or type(target) is not int
                or target < 0
                for source, target in mapping.items()
            )
            or len(set(mapping.values())) != len(mapping)
        ):
            raise ValueError(
                f"{path} projected-trace stream roles for step {step} are invalid"
            )

    for occurrence in matching:
        evidence = occurrence["evidence_fingerprints"]
        if (
            evidence["kernel_details"] != fingerprints["baseline_kernel_details_sha256"]
            or evidence["projection_mapping"] != proof["mapping_fingerprint"]
            or evidence["profile_sk_graph_origin"]
            != fingerprints["profile_origin_graph_fingerprint"]
        ):
            raise ValueError(f"{path} projected-trace evidence is inconsistent")


def _validate_structural_schema(report):
    protocol = _structural_object(report, "association_protocol")
    if not isinstance(protocol.get("blockers"), list):
        raise ValueError("report.association_protocol.blockers must be a list")
    binding = _structural_object(report, "candidate_binding_evidence")
    if set(binding) != {"summary", "occurrences"}:
        raise ValueError("report.candidate_binding_evidence has an invalid shape")
    for field in (
        "canonical_graph_fingerprints",
        "stream_role_mapping",
        "graph_alignment_proof",
    ):
        _structural_object(report, field)

    has_exact_projected_trace = any(
        decision.get("mapping_confidence") == "exact_projected_trace"
        for decision in report["per_sk_decisions"]
        if isinstance(decision, dict)
    )
    if has_exact_projected_trace:
        structural_input_fields = PROFILE_STRUCTURAL_INPUT_FIELDS
        manifest_paths = []
        for field in structural_input_fields:
            value = report["inputs"].get(field)
            if (
                not isinstance(value, str)
                or not value.strip()
                or Path(value).is_absolute()
            ):
                raise ValueError(
                    f"report.inputs.{field} must bind a relative manifest path"
                )
            manifest_paths.append(os.path.normcase(os.path.normpath(value)))
        if len(set(manifest_paths)) != len(manifest_paths):
            raise ValueError(
                "report structural manifest inputs must identify distinct roots"
            )
        content_fingerprints = protocol.get("manifest_content_fingerprints")
        if (
            protocol.get("protocol") != "kernel_projection_trace_v2"
            or protocol.get("status") != "associated"
            or protocol.get("blockers") != []
            or not _is_sha256(protocol.get("manifest_set_fingerprint"))
            or not isinstance(content_fingerprints, dict)
            or set(content_fingerprints) != set(structural_input_fields)
            or any(not _is_sha256(value) for value in content_fingerprints.values())
            or len(set(content_fingerprints.values())) != len(content_fingerprints)
        ):
            raise ValueError(
                "report.association_protocol must bind the complete manifest set"
            )

    for index, decision in enumerate(report["per_sk_decisions"]):
        path = f"per_sk_decisions[{index}]"
        method = decision.get("mapping_method")
        confidence = decision.get("mapping_confidence")
        mapping_blockers = decision.get("mapping_blockers")
        if not isinstance(mapping_blockers, list) or any(
            not isinstance(item, str) or not item for item in mapping_blockers
        ):
            raise ValueError(f"{path}.mapping_blockers is required for schema 1.2")
        if method is not None and (
            not isinstance(method, str) or method not in MAPPING_METHODS
        ):
            raise ValueError(f"{path}.mapping_method is not a schema 1.2 enum")
        if confidence is not None and (
            not isinstance(confidence, str) or confidence not in MAPPING_CONFIDENCES
        ):
            raise ValueError(f"{path}.mapping_confidence is not a schema 1.2 enum")
        if confidence in {"exact", "exact_projected_trace"} and mapping_blockers:
            raise ValueError(f"{path} exact mapping requires empty mapping blockers")
        if confidence == "exact" and method != "source_scope_map":
            raise ValueError(f"{path} schema 1.2 exact requires source_scope_map")
        if confidence == "exact_projected_trace":
            if method != "kernel_projection_structural":
                raise ValueError(
                    f"{path} exact_projected_trace requires kernel_projection_structural"
                )
            _validate_exact_projected_trace_decision(report, decision, path)
    if has_exact_projected_trace and (
        protocol.get("status") != "associated" or protocol["blockers"]
    ):
        raise ValueError(
            "report.association_protocol must be associated without blockers for "
            "exact_projected_trace"
        )
    _validate_mapping_coverage(report)


def _validate_report_schema(report):
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    _validate_json_value(report, "report")
    schema_version = report.get("schema_version")
    if not isinstance(schema_version, str) or schema_version != "1.2":
        raise ValueError("report.schema_version must equal 1.2")
    required_fields = (
        *REQUIRED_IDENTITY_FIELDS,
        *REQUIRED_FINGERPRINT_FIELDS,
        "declared_change_set",
        "inputs",
        "thresholds",
        *LIST_OF_OBJECT_FIELDS,
        "blockers",
        "next_agent_guidance_zh",
        "analysis_content_fingerprint",
    )
    for field in required_fields:
        if field not in report:
            raise ValueError(f"report.{field} is required")
    for field in STRUCTURAL_REPORT_FIELDS:
        if field not in report:
            raise ValueError(f"report.{field} is required for schema 1.2")
    for field in REQUIRED_IDENTITY_FIELDS:
        value = report[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"report.{field} must be a non-empty string")
    identity = {field: report[field] for field in ANALYSIS_IDENTITY_FIELDS}
    canonical_identity = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    expected_analysis_id = "analysis-" + hashlib.sha256(canonical_identity).hexdigest()
    if report["analysis_id"] != expected_analysis_id:
        raise ValueError("report.analysis_id does not match analyzer identity fields")
    candidate_name = report["candidate_name"]
    if (
        re.fullmatch(
            rf"{re.escape(candidate_name)}-(?:AUTO|BASE|P[1-9][0-9]*|FINAL)",
            report["round_id"],
        )
        is None
    ):
        raise ValueError("report.round_id must belong to candidate_name")
    for field in REQUIRED_FINGERPRINT_FIELDS:
        value = report[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"report.{field} must be a non-empty string or null")
    for field in (
        *REQUIRED_FINGERPRINT_FIELDS[:6],
        "baseline_control_fingerprint",
        "candidate_control_fingerprint",
    ):
        if report[field] is None:
            raise ValueError(f"report.{field} must be a non-empty string")
    for field in TOP_LEVEL_TEXT_FIELDS:
        _validate_text_field(report, field, "report")

    thresholds = _optional_object(report, "thresholds", "report")
    threshold_fields = (
        "min_relative_change_pct",
        "min_absolute_change_us",
        "min_occurrences",
    )
    if set(thresholds or {}) != set(threshold_fields):
        raise ValueError(
            "thresholds must contain min_relative_change_pct, "
            "min_absolute_change_us, and min_occurrences"
        )
    for field in threshold_fields:
        _validate_number_field(thresholds, field, "thresholds")
        if thresholds[field] is None:
            raise ValueError(f"thresholds.{field} is required")
    min_occurrences = thresholds["min_occurrences"]
    if (
        isinstance(min_occurrences, bool)
        or not isinstance(min_occurrences, int)
        or min_occurrences < 3
    ):
        raise ValueError("thresholds.min_occurrences must be an integer >= 3")

    inputs = _optional_object(report, "inputs", "report")
    required_input_fields = (
        PROFILE_COMPARISON_INPUT_FIELDS + PROFILE_STRUCTURAL_INPUT_FIELDS
    )
    if set(inputs or {}) != set(required_input_fields):
        raise ValueError(
            f"inputs must contain the complete schema {schema_version} input set"
        )
    for field in required_input_fields:
        value = inputs[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"inputs.{field} must be a string or null")
    declared_change_set = _optional_object(report, "declared_change_set", "report")
    pointers = (declared_change_set or {}).get("allowed_json_pointers")
    if not isinstance(pointers, list) or not all(
        isinstance(pointer, str) for pointer in pointers
    ):
        raise ValueError(
            "declared_change_set.allowed_json_pointers must be a list of strings"
        )

    for field in LIST_OF_OBJECT_FIELDS:
        if field not in report:
            continue
        items = report[field]
        if not isinstance(items, list):
            raise ValueError(f"{field} must be a list")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"{field}[{index}] must be an object")
            if field == "per_sk_decisions":
                _validate_decision(item, f"per_sk_decisions[{index}]")

    decisions = report["per_sk_decisions"]
    if not decisions:
        raise ValueError("per_sk_decisions must contain at least one decision")
    _validate_structural_schema(report)

    _validate_list_item_fields(
        report.get("scope_actions"),
        "scope_actions",
        text_fields=("range_id", "sk_id", "classification", "action"),
    )
    for index, item in enumerate(report.get("scope_actions") or []):
        _validate_scope_action(item, f"scope_actions[{index}]")
    decision_contracts = _unique_scope_contracts(
        decisions,
        "per_sk_decisions",
        _decision_scope_contract,
    )
    action_contracts = _unique_scope_contracts(
        report["scope_actions"], "scope_actions", _scope_action_contract
    )
    if decision_contracts != action_contracts:
        raise ValueError(
            "scope_actions must match per_sk_decisions one-to-one for sk_id, "
            "range_id, classification, action, source_scope, boundary, "
            "ordered_child_op_sequence, and interval_unproven"
        )
    _validate_list_item_fields(
        report.get("diagnostic_hypotheses"),
        "diagnostic_hypotheses",
        text_fields=("range_id", "kind", "confidence", "explanation_zh"),
        bool_fields=("requires_ab_test",),
    )
    for index, item in enumerate(report.get("diagnostic_hypotheses") or []):
        evidence = _optional_list(item, "evidence", f"diagnostic_hypotheses[{index}]")
        _validate_text_list(evidence, f"diagnostic_hypotheses[{index}].evidence")

    _validate_list_item_fields(
        report.get("recommended_experiments"),
        "recommended_experiments",
        text_fields=(
            "experiment_id",
            "range_id",
            "option",
            "expected_signal_zh",
        ),
    )
    for index, item in enumerate(report.get("recommended_experiments") or []):
        item_path = f"recommended_experiments[{index}]"
        _optional_object(item, "only_change", item_path)
        _optional_object(item, "accepted_evidence", item_path)
        gates = _optional_list(item, "lifecycle_gates", item_path)
        _validate_text_list(gates, f"{item_path}.lifecycle_gates")

    if "blockers" in report:
        blockers = report["blockers"]
        if not isinstance(blockers, list):
            raise ValueError("blockers must be a list")
        for index, blocker in enumerate(blockers):
            if not isinstance(blocker, (str, dict)):
                raise ValueError(f"blockers[{index}] must be a string or object")

    guidance = report["next_agent_guidance_zh"]
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError("report.next_agent_guidance_zh must be a non-empty string")
    content_fingerprint = report["analysis_content_fingerprint"]
    if (
        not isinstance(content_fingerprint, str)
        or len(content_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in content_fingerprint)
    ):
        raise ValueError(
            "report.analysis_content_fingerprint must be a lowercase SHA-256"
        )
    unsigned_report = dict(report)
    unsigned_report.pop("analysis_content_fingerprint")
    canonical = json.dumps(
        unsigned_report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    expected_fingerprint = hashlib.sha256(canonical).hexdigest()
    if content_fingerprint != expected_fingerprint:
        raise ValueError("report.analysis_content_fingerprint does not match content")


def render_report(report):
    _validate_report_schema(report)

    lines = ["# SuperKernel 融合性能分析", "", "## 输入与可信度", ""]
    lines.extend(
        _table(
            ["字段", "值"],
            [
                ("experiment_id", report.get("experiment_id")),
                ("round_id", report.get("round_id")),
                ("analysis_agent_id", report.get("analysis_agent_id")),
                ("candidate_name", report.get("candidate_name")),
                ("source_revision", report.get("source_revision")),
                ("workload_fingerprint", report.get("workload_fingerprint")),
                ("control_fingerprint", report.get("control_fingerprint")),
                ("declared_change_set", report.get("declared_change_set")),
                ("inputs", report.get("inputs")),
            ],
        )
    )

    lines.extend(["", "## 判定阈值", ""])
    thresholds = report.get("thresholds") or {}
    lines.extend(
        _table(
            ["阈值", "值"],
            [
                (
                    "min_relative_change_pct",
                    thresholds.get("min_relative_change_pct"),
                ),
                (
                    "min_absolute_change_us",
                    thresholds.get("min_absolute_change_us"),
                ),
                ("min_occurrences", thresholds.get("min_occurrences")),
            ],
        )
    )

    if report["schema_version"] == "1.2":
        coverage = report["mapping_coverage"]
        lines.extend(["", "## 结构关联覆盖率", ""])
        lines.extend(
            _table(
                ["指标", "数量"],
                [
                    ("total_sk_ids", coverage.get("total_sk_ids")),
                    ("bound_sk_ids", coverage.get("bound_sk_ids")),
                    (
                        "exact_projected_trace_sk_ids",
                        coverage.get("exact_projected_trace_sk_ids"),
                    ),
                    ("ambiguous_sk_ids", coverage.get("ambiguous_sk_ids")),
                    ("unmapped_sk_ids", coverage.get("unmapped_sk_ids")),
                    (
                        "filtered_by_child_count",
                        coverage.get("filtered_by_child_count"),
                    ),
                ],
            )
        )

        lines.extend(["", "## Child Count 分布", ""])
        distribution = coverage.get("child_count_distribution") or {}
        lines.extend(
            _table(
                ["Child 数", "SK 数"],
                sorted(distribution.items(), key=lambda item: str(item[0])),
            )
        )
        lines.extend(
            [
                "",
                "raw Task ID/名称提示仅用于诊断；child_count<5 没有被过滤。",
                "",
                "## 逐 SK 结构与性能证据",
                "",
            ]
        )
        structural_rows = []
        proofs = report["graph_alignment_proof"]
        for decision in _sorted_by_range(report["per_sk_decisions"]):
            key = _structural_identity_key(decision)
            proof = proofs.get(key) if isinstance(proofs, dict) else None
            proof = proof if isinstance(proof, dict) else {}
            structural_rows.append(
                (
                    decision.get("sk_id"),
                    decision.get("child_count"),
                    decision.get("candidate_binding_status"),
                    (
                        f"{_display(decision.get('mapping_method'))} / "
                        f"{_display(decision.get('mapping_confidence'))}"
                    ),
                    proof.get("alternative_solution_count_by_step"),
                    decision.get("classification"),
                    _structural_source_action(decision),
                )
            )
        lines.extend(
            _table(
                [
                    "SK",
                    "Child 数",
                    "同进程绑定",
                    "Kernel Projection",
                    "各 Step 替代解",
                    "性能分类",
                    "源码动作",
                ],
                structural_rows,
            )
        )

        lines.extend(["", "## 关联阻塞项", ""])
        blocker_rows = [
            (decision.get("sk_id"), decision.get("mapping_blockers") or EMPTY)
            for decision in _sorted_by_range(report["per_sk_decisions"])
        ]
        lines.extend(_table(["SK", "mapping blockers"], blocker_rows))
        protocol_blockers = report["association_protocol"].get("blockers") or []
        lines.extend(
            [f"- protocol: {_markdown(item)}" for item in protocol_blockers]
            or [f"- protocol: {EMPTY}"]
        )
        exclusions = report["association_protocol"].get("model_projection_exclusions")
        if isinstance(exclusions, dict) and exclusions:
            lines.extend(["", "## Projection 排除门禁", ""])
            exclusion_rows = []
            for identity, evidence in sorted(exclusions.items()):
                exclusion_rows.append(
                    (
                        identity,
                        evidence.get("protocol"),
                        evidence.get("status"),
                        evidence.get("occurrence_count", 0),
                        len(evidence.get("candidate_node_keys") or []),
                        len(evidence.get("excluded_node_keys") or []),
                        evidence.get("blockers") or EMPTY,
                    )
                )
            lines.extend(
                _table(
                    [
                        "模型",
                        "协议",
                        "状态",
                        "occurrence",
                        "候选节点",
                        "排除节点",
                        "blockers",
                    ],
                    exclusion_rows,
                )
            )

        lines.extend(["", "## Automatic AOT 与源码动作边界", ""])
        lines.append(
            "automatic AOT kernel projection 可用于性能分类，但不能证明 Python "
            "源码 offset，不能直接驱动 prune；应先完成 winner-only SMAP，"
            "由 stable_source + marker_only_calibration 独立证明原始源码区间。"
        )

    lines.extend(["", "## 逐 SK 性能对比", ""])
    decision_rows = []
    for decision in _sorted_by_range(report.get("per_sk_decisions")):
        decision_rows.append(
            (
                decision.get("range_id"),
                decision.get("sk_id"),
                decision.get("child_count"),
                _nested_value(decision, "original", "interval_us", "p50"),
                _nested_value(decision, "original", "interval_us", "p90"),
                _nested_value(decision, "original", "interval_us", "mad"),
                _nested_value(decision, "original", "duration_sum_us", "p50"),
                _nested_value(decision, "original", "duration_sum_us", "p90"),
                _nested_value(decision, "original", "duration_sum_us", "mad"),
                _nested_value(decision, "sk_duration", "p50"),
                _nested_value(decision, "sk_duration", "p90"),
                _nested_value(decision, "sk_duration", "mad"),
                _occurrence_counts(decision),
                decision.get("improvement_pct"),
                decision.get("noise_threshold_pct"),
                (
                    f"{_display(decision.get('classification'))} / "
                    f"{_display(decision.get('classification_zh'))}"
                ),
                decision.get("action"),
                decision.get("mapping_method"),
                decision.get("mapping_confidence"),
                _mapping_reliable(decision),
                decision.get("boundary"),
                _nested_value(decision, "identity", "ordered_child_op_sequence"),
                _stream_overlap(decision),
                report.get("analysis_agent_id"),
                _conditional_evidence_binding(report, decision.get("range_id")),
                _decision_evidence(decision),
            )
        )
    _section_or_empty(
        lines,
        _table(
            [
                "range_id",
                "sk_id",
                "child count",
                "baseline interval P50 (us)",
                "baseline interval P90 (us)",
                "baseline interval MAD (us)",
                "baseline duration sum P50 (us)",
                "baseline duration sum P90 (us)",
                "baseline duration sum MAD (us)",
                "SK P50 (us)",
                "SK P90 (us)",
                "SK MAD (us)",
                "occurrence count",
                "change (%)",
                "MAD/dynamic threshold (%)",
                "classification（英文 + 中文）",
                "action",
                "mapping method",
                "mapping confidence",
                "mapping reliable",
                "boundary",
                "ordered child sequence",
                "stream/overlap",
                "analysis Agent",
                "条件证据绑定",
                "blocker/evidence",
            ],
            decision_rows,
        )
        if decision_rows
        else [],
    )

    lines.extend(["", "## Scope 保留与裁剪", ""])
    action_rows = [
        (
            item.get("range_id"),
            item.get("sk_id"),
            item.get("classification"),
            item.get("action"),
        )
        for item in _sorted_by_range(report.get("scope_actions"))
    ]
    _section_or_empty(
        lines,
        _table(["range_id", "sk_id", "classification", "action"], action_rows)
        if action_rows
        else [],
    )

    lines.extend(["", "## 劣化原因假设", ""])
    hypothesis_rows = [
        (
            item.get("range_id"),
            item.get("kind"),
            item.get("confidence"),
            item.get("evidence") or EMPTY,
            item.get("explanation_zh"),
            item.get("requires_ab_test"),
        )
        for item in _sorted_by_range(report.get("diagnostic_hypotheses"))
    ]
    _section_or_empty(
        lines,
        _table(
            [
                "range_id",
                "kind",
                "confidence",
                "evidence",
                "说明",
                "requires A/B test",
            ],
            hypothesis_rows,
        )
        if hypothesis_rows
        else [],
    )

    lines.extend(["", "## 建议验证实验", ""])
    experiment_rows = [
        (
            item.get("experiment_id"),
            item.get("range_id"),
            item.get("option"),
            item.get("value"),
            item.get("only_change"),
            item.get("accepted_evidence"),
            item.get("expected_signal_zh"),
            item.get("lifecycle_gates"),
        )
        for item in _sorted_by_range(report.get("recommended_experiments"))
    ]
    _section_or_empty(
        lines,
        _table(
            [
                "experiment_id",
                "range_id",
                "option",
                "value",
                "only change",
                "accepted evidence",
                "预期信号",
                "lifecycle gates",
            ],
            experiment_rows,
        )
        if experiment_rows
        else [],
    )

    lines.extend(["", "## 阻塞项", ""])
    blockers = sorted(_markdown(item) for item in report.get("blockers") or [])
    lines.extend([f"- {item}" for item in blockers] or [f"- {EMPTY}"])

    lines.extend(["", "## 给主实验 Agent 的后续指导", ""])
    guidance = report.get("next_agent_guidance_zh")
    lines.append(_markdown(guidance) if guidance else MISSING)
    return "\n".join(lines) + "\n"


def _atomic_write(path, content):
    if path.exists() and path.is_dir():
        raise ValueError(f"{path}: Markdown output must not be a directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(content)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _resolve_path(path):
    try:
        return path.resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{path}: cannot resolve path: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-in", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if _resolve_path(args.json_in) == _resolve_path(args.markdown_out):
            raise ValueError("JSON input and Markdown output paths must be distinct")
        report = json.loads(args.json_in.read_text(encoding="utf-8"))
        _atomic_write(args.markdown_out, render_report(report))
    except (OSError, ValueError, json.JSONDecodeError, RecursionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
