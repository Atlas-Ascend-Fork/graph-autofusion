#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate and merge portable SuperKernel experiment handoffs."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import recommend_sk_strategy


SCHEMA_VERSION = 2
ROUND_KINDS = {"base", "performance_prune", "final", "automatic_aot"}
PERFORMANCE_CLASSES = {
    "beneficial",
    "regressed",
    "neutral",
    "insufficient_evidence",
}
PERFORMANCE_ACTIONS = {"keep", "prune", "reprofile", "block"}
DECISION_STATUSES = {"proposed", "applied", "verified", "blocked"}
SCOPE_KINDS = {"manual", "automatic_aot"}
PERFORMANCE_EXACT_MAPPINGS = {
    ("source_scope_map", "exact"),
    ("kernel_projection_structural", "exact_projected_trace"),
}
SOURCE_ACTIONABLE_MAPPINGS = {("source_scope_map", "exact")}
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
SUPPORTED_MAPPING_PAIRS = PERFORMANCE_EXACT_MAPPINGS | {
    ("source_scope_map", "diagnostic_only"),
    ("kernel_projection_structural", "diagnostic_only"),
    ("kernel_projection_structural", "ambiguous"),
    ("kernel_projection_structural", "unmapped"),
    ("sk_meta_node_ids", "diagnostic_only"),
}
FINGERPRINT_FIELDS = (
    "source_revision",
    "baseline_config_fingerprint",
    "candidate_config_fingerprint",
    "control_fingerprint",
    "workload_fingerprint",
)
CONDITIONAL_KEY_FIELDS = (*FINGERPRINT_FIELDS, "range_id")
PROFILING_BINDING_FIELDS = (
    "baseline_profile",
    "baseline_profile_fingerprint",
    "candidate_profile",
    "candidate_profile_fingerprint",
    "declared_change_set",
)
BASELINE_REUSE_FIELDS = (
    "baseline_revision",
    "baseline_config_fingerprint",
    "control_fingerprint",
    "workload_fingerprint",
)
PATH_KEYS = {
    "artifact",
    "artifacts",
    "baseline_profile",
    "candidate_profile",
    "clean",
    "config",
    "metadata",
    "path",
    "profiler",
    "profiling_analysis_result",
    "replay",
    "round_evidence",
    "round_report",
    "source_evidence",
    "source_path",
    "source_profiling_analysis_result",
}
NON_ARTIFACT_CONTAINER_KEYS = {"lifecycle"}
ARTIFACT_ROLE_BY_KEY = {
    "baseline_profile": "baseline_profile",
    "candidate_profile": "candidate_profile",
    "clean": "clean",
    "config": "config",
    "metadata": "metadata",
    "profiler": "profiler",
    "profiling_analysis_result": "profiling_analysis_result",
    "replay": "replay",
    "round_evidence": "round_evidence",
    "round_report": "round_report",
    "source_evidence": "source_evidence",
    "source_path": "source_evidence",
    "source_profiling_analysis_result": "profiling_analysis_result",
}
REQUIRED_RESULT_FIELDS = {
    "experiment_id",
    "parent_experiment_id",
    "child_agent_id",
    "baseline_revision",
    "source_revision",
    "baseline_config_fingerprint",
    "control_fingerprint",
    "workload_fingerprint",
    "rounds",
    "profiling_analysis_agent_ids",
    "profiling_analysis_result",
    "performance_scope_decisions",
    "unresolved_performance_ranges",
    "inherited_exclusions_for_next_agents",
    "lifecycle",
    "blockers",
    "next_agent_guidance_zh",
}
PROMOTION_MODES = {"range_optimized", "whole_scope"}


def _artifact_leaf_path_errors(value, location):
    errors = []
    if isinstance(value, str):
        if not _is_relative_artifact(value):
            errors.append(f"{location} must be a relative artifact path")
        return errors
    if isinstance(value, dict):
        for key, item in value.items():
            child_location = f"{location}.{key}" if location else key
            errors.extend(_artifact_leaf_path_errors(item, child_location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_artifact_leaf_path_errors(item, f"{location}[{index}]"))
    return errors


def _relative_artifact_errors(value, location=""):
    errors = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_location = f"{location}.{key}" if location else key
            if key in NON_ARTIFACT_CONTAINER_KEYS:
                continue
            if key in PATH_KEYS:
                errors.extend(_artifact_leaf_path_errors(item, child_location))
            else:
                errors.extend(_relative_artifact_errors(item, child_location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_relative_artifact_errors(item, f"{location}[{index}]"))
    return errors


def _is_non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _has_surrounding_whitespace(value):
    return isinstance(value, str) and value != value.strip()


def _normalized_id(value):
    return value.strip() if isinstance(value, str) else value


def _is_canonical_string(value):
    return _is_non_empty_string(value) and not _has_surrounding_whitespace(value)


def _safe_mapping_get(mapping, key, default=None):
    if not isinstance(mapping, dict) or not _is_canonical_string(key):
        return default
    return mapping.get(key, default)


def _enum_member(value, allowed):
    return isinstance(value, str) and value in allowed


def _is_relative_artifact(value):
    if not _is_non_empty_string(value) or _has_surrounding_whitespace(value):
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _contains_chinese(value):
    return isinstance(value, str) and any(
        "\u4e00" <= char <= "\u9fff" for char in value
    )


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value_errors(value, location="experiment result"):
    errors = []
    active = set()
    stack = [("visit", value, location)]
    while stack:
        operation, current, current_location = stack.pop()
        if operation == "leave":
            active.remove(id(current))
            continue
        if current is None or isinstance(current, (bool, str, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                errors.append(f"{current_location} must contain finite JSON numbers")
            continue
        if not isinstance(current, (dict, list)):
            errors.append(
                f"{current_location} contains non-JSON value {type(current).__name__}"
            )
            continue
        identity = id(current)
        if identity in active:
            errors.append(f"{current_location} contains a cyclic JSON structure")
            continue
        active.add(identity)
        stack.append(("leave", current, current_location))
        if isinstance(current, dict):
            children = []
            for key, child in current.items():
                if not isinstance(key, str):
                    errors.append(
                        f"{current_location} contains non-string JSON object key"
                    )
                    continue
                children.append(("visit", child, f"{current_location}.{key}"))
        else:
            children = [
                ("visit", child, f"{current_location}[{index}]")
                for index, child in enumerate(current)
            ]
        stack.extend(reversed(children))
    return errors


def _decode_absolute_rfc6901_pointer(value):
    if not isinstance(value, str) or not value.startswith("/"):
        return None
    if _has_surrounding_whitespace(value):
        return None
    decoded_tokens = []
    for token in value[1:].split("/"):
        decoded = []
        index = 0
        while index < len(token):
            if token[index] != "~":
                decoded.append(token[index])
                index += 1
                continue
            if index + 1 >= len(token) or token[index + 1] not in "01":
                return None
            decoded.append("~" if token[index + 1] == "0" else "/")
            index += 2
        decoded_tokens.append("".join(decoded))
    return decoded_tokens


def _is_absolute_rfc6901_pointer(value):
    return _decode_absolute_rfc6901_pointer(value) is not None


def _validate_declared_change_set(value, location):
    if not isinstance(value, dict) or not value:
        return [f"{location} must be a non-empty object"]
    errors = []
    pointers = value.get("allowed_json_pointers")
    if not isinstance(pointers, list):
        errors.append(f"{location}.allowed_json_pointers must be a list")
    elif any(not _is_absolute_rfc6901_pointer(pointer) for pointer in pointers):
        errors.append(
            f"{location}.allowed_json_pointers must contain absolute RFC6901 pointers"
        )
    elif len(pointers) != len(set(pointers)):
        errors.append(f"{location}.allowed_json_pointers must not contain duplicates")
    only_change_zh = value.get("only_change_zh")
    if not _is_non_empty_string(only_change_zh) or not _contains_chinese(
        only_change_zh
    ):
        errors.append(f"{location}.only_change_zh must contain non-empty Chinese text")
    return errors


def _validate_profiling_binding(record, location, *, expected=None):
    errors = []
    for field in ("baseline_profile", "candidate_profile"):
        if not _is_relative_artifact(record.get(field)):
            errors.append(f"{location}.{field} must be a relative artifact path")
    for field in ("baseline_profile_fingerprint", "candidate_profile_fingerprint"):
        value = record.get(field)
        if not _is_non_empty_string(value) or _has_surrounding_whitespace(value):
            errors.append(f"{location}.{field} must be a non-empty canonical string")
    errors.extend(
        _validate_declared_change_set(
            record.get("declared_change_set"), f"{location}.declared_change_set"
        )
    )
    if expected is not None:
        for field in PROFILING_BINDING_FIELDS:
            if record.get(field) != expected.get(field):
                errors.append(f"{location}.{field} must match its source round")
    return errors


def _validate_string_list(value, location, *, allow_empty=True):
    errors = []
    if not isinstance(value, list):
        return [f"{location} must be a list"]
    if not allow_empty and not value:
        errors.append(f"{location} must not be empty")
    if any(not _is_non_empty_string(item) for item in value):
        errors.append(f"{location} must contain non-empty strings")
    elif len(value) != len(set(value)):
        errors.append(f"{location} must not contain duplicate range IDs")
    return errors


def _string_set(value):
    if not isinstance(value, list):
        return set()
    return {item for item in value if _is_non_empty_string(item)}


def _validate_fingerprints(record, location, round_data=None):
    errors = []
    for field in FINGERPRINT_FIELDS:
        if not _is_non_empty_string(record.get(field)):
            errors.append(f"{location}.{field} must be a non-empty string")
        elif round_data is not None and record[field] != round_data.get(field):
            errors.append(f"{location}.{field} must match its declared round")
    return errors


def _mapping_is_performance_exact(record):
    pair = (
        record.get("mapping_method"),
        record.get("mapping_confidence"),
    )
    if pair not in PERFORMANCE_EXACT_MAPPINGS:
        return False
    if pair != ("kernel_projection_structural", "exact_projected_trace"):
        return True
    fingerprint = record.get("graph_occurrence_fingerprint")
    return (
        record.get("candidate_binding_status") == "bound"
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
        and record.get("mapping_blockers") == []
    )


def _mapping_is_source_actionable(record):
    pair = (
        record.get("mapping_method"),
        record.get("mapping_confidence"),
    )
    boundary = record.get("boundary")
    if pair not in SOURCE_ACTIONABLE_MAPPINGS or not isinstance(boundary, dict):
        return False
    source_file = boundary.get("source_file")
    start = boundary.get("start_offset")
    end = boundary.get("end_offset")
    return (
        _is_non_empty_string(source_file)
        and not Path(source_file).is_absolute()
        and ".." not in Path(source_file).parts
        and isinstance(start, int)
        and not isinstance(start, bool)
        and start >= 0
        and isinstance(end, int)
        and not isinstance(end, bool)
        and end > start
    )


def _validate_decision(
    decision,
    location,
    *,
    collection,
    rounds_by_id,
    round_positions,
    profiling_round_ids,
    analysis_agent_ids,
    analysis_results,
):
    errors = []
    if not isinstance(decision, dict):
        return [f"{location} must be an object"]

    for field in ("round_id", "range_id", "sk_id", "source_scope"):
        if not _is_non_empty_string(decision.get(field)):
            errors.append(f"{location}.{field} must be a non-empty string")
        elif _has_surrounding_whitespace(decision.get(field)):
            errors.append(f"{location}.{field} must not contain surrounding whitespace")
    mapping_method = decision.get("mapping_method")
    mapping_confidence = decision.get("mapping_confidence")
    if mapping_method not in MAPPING_METHODS:
        errors.append(
            f"{location}.mapping_method must preserve a producer mapping method"
        )
    if mapping_confidence not in MAPPING_CONFIDENCES:
        errors.append(f"{location}.mapping_confidence is invalid")
    if (mapping_method, mapping_confidence) not in SUPPORTED_MAPPING_PAIRS:
        errors.append(f"{location} mapping method/confidence pair is unsupported")
    if "mapping_reliable" in decision:
        errors.append(
            f"{location}.mapping_reliable is consumer-reported and must be omitted"
        )
    boundary = decision.get("boundary")
    if not isinstance(boundary, dict):
        errors.append(f"{location}.boundary must be an object")
    else:
        for field in ("start_op", "end_op"):
            if not _is_non_empty_string(boundary.get(field)):
                errors.append(f"{location}.boundary.{field} must be a non-empty string")
    classification = decision.get("classification")
    action = decision.get("action")
    status = decision.get("status")
    if not _enum_member(classification, PERFORMANCE_CLASSES):
        errors.append(f"{location}.classification is invalid")
    if not _enum_member(action, PERFORMANCE_ACTIONS):
        errors.append(f"{location}.action is invalid")
    if not _enum_member(status, DECISION_STATUSES):
        errors.append(f"{location}.status is invalid")
    decision_tuple = (classification, action, status)
    if collection == "performance_scope_decisions":
        allowed_tuples = {
            ("beneficial", "keep", allowed_status)
            for allowed_status in ("proposed", "applied", "verified")
        } | {
            (allowed_class, "prune", allowed_status)
            for allowed_class in ("neutral", "regressed")
            for allowed_status in ("proposed", "applied", "verified")
        }
    else:
        allowed_tuples = {
            (allowed_class, "reprofile", "proposed")
            for allowed_class in (
                "neutral",
                "regressed",
                "insufficient_evidence",
            )
        } | {
            (allowed_class, "block", "blocked")
            for allowed_class in (
                "neutral",
                "regressed",
                "insufficient_evidence",
            )
        }
    if (
        not (
            _enum_member(classification, PERFORMANCE_CLASSES)
            and _enum_member(action, PERFORMANCE_ACTIONS)
            and _enum_member(status, DECISION_STATUSES)
        )
        or decision_tuple not in allowed_tuples
    ):
        errors.append(
            f"{location} classification/action/status tuple {decision_tuple!r} "
            f"is invalid for {collection}"
        )
    if (
        collection == "performance_scope_decisions"
        and status == "proposed"
        and not _mapping_is_performance_exact(decision)
    ):
        errors.append(
            f"{location} proposed measured classification requires "
            "performance-exact mapping evidence"
        )
    if (
        not _mapping_is_performance_exact(decision)
        and classification != "insufficient_evidence"
    ):
        errors.append(f"{location} non-exact mapping must remain insufficient_evidence")

    round_id = decision.get("round_id")
    round_data = rounds_by_id.get(round_id) if _is_non_empty_string(round_id) else None
    if round_data is None:
        errors.append(f"{location}.round_id must reference a declared round")
    errors.extend(_validate_fingerprints(decision, location, round_data))
    errors.extend(_validate_profiling_binding(decision, location, expected=round_data))
    if not _is_non_empty_string(round_id) or round_id not in profiling_round_ids:
        errors.append(f"{location} source round must have profiling passed")
    source_agent_id = (
        analysis_agent_ids.get(round_id) if _is_non_empty_string(round_id) else None
    )
    if not _is_non_empty_string(source_agent_id):
        errors.append(f"{location} source round must have a profiling analysis Agent")
    expected_source_path = (
        analysis_results.get(round_id) if _is_non_empty_string(round_id) else None
    )
    if not _is_relative_artifact(expected_source_path):
        errors.append(f"{location} source round must have a profiling analysis result")
    if decision.get("source_profiling_analysis_result") != expected_source_path:
        errors.append(
            f"{location}.source_profiling_analysis_result must match its source round"
        )

    effective_prune = action == "prune" and status in ("applied", "verified")
    if effective_prune and not (
        classification in ("neutral", "regressed")
        and _mapping_is_source_actionable(decision)
    ):
        errors.append(
            f"{location} effective prune requires source-actionable neutral/regressed mapping"
        )
    effective_keep = action == "keep" and status in ("applied", "verified")
    if effective_keep and not (
        classification == "beneficial" and _mapping_is_source_actionable(decision)
    ):
        errors.append(
            f"{location} effective keep requires source-actionable beneficial mapping"
        )

    if round_data is not None and round_data.get("round_kind") == "automatic_aot":
        automatic_scope_tuples = {
            ("beneficial", "keep", "proposed"),
            ("neutral", "prune", "proposed"),
            ("regressed", "prune", "proposed"),
        }
        if _mapping_is_source_actionable(decision):
            automatic_scope_tuples.update(
                {
                    ("beneficial", "keep", "applied"),
                    ("neutral", "prune", "verified"),
                    ("regressed", "prune", "verified"),
                }
            )
        automatic_unresolved_tuples = {
            (allowed_class, "reprofile", "proposed")
            for allowed_class in (
                "neutral",
                "regressed",
                "insufficient_evidence",
            )
        } | {
            (allowed_class, "block", "blocked")
            for allowed_class in (
                "neutral",
                "regressed",
                "insufficient_evidence",
            )
        }
        allowed_automatic = (
            collection == "performance_scope_decisions"
            and decision_tuple in automatic_scope_tuples
        ) or (
            collection == "unresolved_performance_ranges"
            and decision_tuple in automatic_unresolved_tuples
        )
        if not allowed_automatic:
            errors.append(
                f"{location} automatic_aot may only retain proposed carry-forward, "
                "source-actionable later-verified, or blocked unresolved evidence"
            )

    if status == "verified":
        verified_round_id = decision.get("verified_in_round_id")
        if not _is_non_empty_string(verified_round_id) or (
            verified_round_id not in profiling_round_ids
        ):
            errors.append(
                f"{location}.verified_in_round_id must reference a profiling-passed round"
            )
        expected_path = (
            analysis_results.get(verified_round_id)
            if _is_non_empty_string(verified_round_id)
            else None
        )
        if decision.get("profiling_analysis_result") != expected_path:
            errors.append(
                f"{location}.profiling_analysis_result must reference fresh verification profiling"
            )
        source_position = (
            round_positions.get(round_id) if _is_non_empty_string(round_id) else None
        )
        verified_position = (
            round_positions.get(verified_round_id)
            if _is_non_empty_string(verified_round_id)
            else None
        )
        if status == "verified" and (
            source_position is not None
            and verified_position is not None
            and verified_position <= source_position
        ):
            errors.append(
                f"{location}.verified_in_round_id must be strictly later than its source round"
            )
        if status == "verified" and action == "prune":
            verification_round = (
                rounds_by_id.get(verified_round_id)
                if _is_non_empty_string(verified_round_id)
                else None
            )
            if (
                verification_round is None
                or verification_round.get("round_kind") != "performance_prune"
            ):
                errors.append(
                    f"{location}.verified_in_round_id must reference a later "
                    "performance_prune round"
                )
            elif not _is_canonical_string(decision.get("range_id")) or decision.get(
                "range_id"
            ) not in _string_set(
                verification_round.get("performance_decision_range_ids")
            ):
                errors.append(
                    f"{location} verified prune range must be referenced by its "
                    "performance_prune verification round"
                )
    return errors


def _conditional_key(decision):
    return json.dumps(
        [decision[field] for field in CONDITIONAL_KEY_FIELDS],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _profiling_rounds(result):
    rounds = result.get("rounds") if isinstance(result, dict) else None
    if not isinstance(rounds, list):
        return []
    return [
        round_data
        for round_data in rounds
        if isinstance(round_data, dict)
        and isinstance(round_data.get("lifecycle"), dict)
        and round_data["lifecycle"].get("profiling") == "passed"
    ]


def _baseline_reuse_key(result):
    return tuple(result.get(field) for field in BASELINE_REUSE_FIELDS)


def _all_profile_paths(result):
    paths = set()
    rounds = result.get("rounds") if isinstance(result, dict) else None
    if not isinstance(rounds, list):
        return paths
    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        for field in ("baseline_profile", "candidate_profile"):
            value = round_data.get(field)
            if _is_relative_artifact(value):
                paths.add(Path(value).as_posix())
    return paths


def _all_baseline_profile_pairs(result):
    pairs = []
    rounds = result.get("rounds") if isinstance(result, dict) else None
    if not isinstance(rounds, list):
        return pairs
    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        path = round_data.get("baseline_profile")
        fingerprint = round_data.get("baseline_profile_fingerprint")
        if _is_relative_artifact(path) and _is_non_empty_string(fingerprint):
            pairs.append((Path(path).as_posix(), fingerprint.strip()))
    return pairs


def _generic_artifact_role(path):
    normalized = Path(path).as_posix().lower().replace("_", "-")
    if "profiling-analysis" in normalized:
        return "profiling_analysis_result"
    if "round-report" in normalized:
        return "round_report"
    if "round-evidence" in normalized:
        return "round_evidence"
    if "source-map" in normalized or "source-evidence" in normalized:
        return "source_evidence"
    if "sk-meta" in normalized or "metadata" in normalized:
        return "metadata"
    if "replay" in normalized:
        return "replay"
    if "profiler" in normalized:
        return "profiler"
    if "config" in normalized:
        return "config"
    if "clean" in normalized:
        return "clean"
    return "artifact"


def _artifact_role_for_key(key, inherited_role=None):
    if key in ARTIFACT_ROLE_BY_KEY:
        return ARTIFACT_ROLE_BY_KEY[key]
    if key in {"artifact", "artifacts", "path"}:
        return inherited_role or "artifact"
    return None


def _collect_declared_artifacts(value, role, paths_by_role, *, role_locked):
    if isinstance(value, str):
        if not _is_relative_artifact(value):
            return
        normalized_path = Path(value).as_posix()
        normalized_role = (
            _generic_artifact_role(normalized_path) if role == "artifact" else role
        )
        paths_by_role.setdefault(normalized_role, set()).add(normalized_path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_role = role
            child_role_locked = role_locked
            if not role_locked and key in PATH_KEYS:
                child_role = _artifact_role_for_key(key, role)
                child_role_locked = child_role != "artifact"
            _collect_declared_artifacts(
                item,
                child_role,
                paths_by_role,
                role_locked=child_role_locked,
            )
        return
    if isinstance(value, list):
        for item in value:
            _collect_declared_artifacts(
                item, role, paths_by_role, role_locked=role_locked
            )


def _scan_artifact_fields(value, paths_by_role):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in NON_ARTIFACT_CONTAINER_KEYS:
                continue
            if key in PATH_KEYS:
                role = _artifact_role_for_key(key)
                _collect_declared_artifacts(
                    item,
                    role,
                    paths_by_role,
                    role_locked=role != "artifact",
                )
            else:
                _scan_artifact_fields(item, paths_by_role)
    elif isinstance(value, list):
        for item in value:
            _scan_artifact_fields(item, paths_by_role)


def _artifact_paths_by_role(result):
    """Return normalized paths from declared artifact fields, grouped by role."""
    paths_by_role = {}
    _scan_artifact_fields(result, paths_by_role)
    return paths_by_role


def _artifact_role_errors(result):
    paths_by_role = _artifact_paths_by_role(result)
    roles_by_path = {}
    for role, paths in paths_by_role.items():
        for path in paths:
            roles_by_path.setdefault(path, set()).add(role)
    errors = []
    for path, roles in sorted(roles_by_path.items()):
        if len(roles) > 1:
            errors.append(
                f"artifact path {path} has conflicting roles: "
                + ", ".join(sorted(roles))
            )

    round_artifacts = []
    rounds = result.get("rounds") if isinstance(result, dict) else None
    if isinstance(rounds, list):
        for round_data in rounds:
            if not isinstance(round_data, dict):
                continue
            artifacts = round_data.get("artifacts")
            if not isinstance(artifacts, dict):
                continue
            for role in ("round_evidence", "round_report"):
                path = artifacts.get(role)
                if _is_relative_artifact(path):
                    round_artifacts.append(Path(path).as_posix())
    if len(round_artifacts) != len(set(round_artifacts)):
        errors.append(
            "round report/evidence artifact paths must be unique across rounds and roles"
        )
    return errors


def _validate_experiment_result(result):
    errors = []
    if not isinstance(result, dict):
        return {"valid": False, "errors": ["experiment result must be an object"]}
    json_errors = _json_value_errors(result)
    if json_errors:
        return {"valid": False, "errors": json_errors}

    missing = sorted(REQUIRED_RESULT_FIELDS - result.keys())
    if missing:
        errors.append("missing fields: " + ", ".join(missing))

    experiment_id = result.get("experiment_id")
    child_agent_id = result.get("child_agent_id")
    if not _is_non_empty_string(experiment_id):
        errors.append("experiment_id must be a non-empty string")
    elif _has_surrounding_whitespace(experiment_id):
        errors.append("experiment_id must not contain surrounding whitespace")
    if not _is_non_empty_string(child_agent_id):
        errors.append("child_agent_id must be a non-empty string")
    elif _has_surrounding_whitespace(child_agent_id):
        errors.append("child_agent_id must not contain surrounding whitespace")
    if _has_surrounding_whitespace(result.get("parent_experiment_id")):
        errors.append("parent_experiment_id must not contain surrounding whitespace")
    if result.get("parent_experiment_id") != experiment_id:
        errors.append("parent_experiment_id must equal experiment_id")
    if not isinstance(result.get("next_agent_guidance_zh"), str) or not any(
        "\u4e00" <= char <= "\u9fff"
        for char in result.get("next_agent_guidance_zh", "")
    ):
        errors.append("next_agent_guidance_zh must contain Chinese guidance")
    for field in (
        "baseline_revision",
        "source_revision",
        "baseline_config_fingerprint",
        "control_fingerprint",
        "workload_fingerprint",
    ):
        if not _is_non_empty_string(result.get(field)):
            errors.append(f"{field} must be a non-empty string")
    for field in ("inherited_exclusions_for_next_agents", "blockers"):
        if field in result and not isinstance(result[field], list):
            errors.append(f"{field} must be a list")
    if "single_child_exclusions" in result and not isinstance(
        result["single_child_exclusions"], list
    ):
        errors.append("single_child_exclusions legacy evidence must be a list")
    if not isinstance(result.get("lifecycle"), dict):
        errors.append("lifecycle must be an object")
    promotion_mode = result.get("promotion_mode", "range_optimized")
    if not _enum_member(promotion_mode, PROMOTION_MODES):
        errors.append("promotion_mode is invalid")

    rounds = result.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        errors.append("rounds must be a non-empty list")
        rounds = []
    rounds_by_id = {}
    round_positions = {}
    profiling_round_ids = set()
    optimization_round_kinds = []
    optimization_rounds = []
    for index, round_data in enumerate(rounds):
        location = f"rounds[{index}]"
        if not isinstance(round_data, dict):
            errors.append(f"{location} must be an object")
            continue
        round_id = round_data.get("round_id")
        if not _is_non_empty_string(round_id):
            errors.append(f"{location}.round_id must be a non-empty string")
        elif _has_surrounding_whitespace(round_id):
            errors.append(
                f"{location}.round_id must not contain surrounding whitespace"
            )
        elif round_id in rounds_by_id:
            errors.append(f"duplicate round_id: {round_id}")
        else:
            rounds_by_id[round_id] = round_data
            round_positions[round_id] = index
        if round_data.get("child_agent_id") != child_agent_id:
            errors.append(f"{location}.child_agent_id must match parent child_agent_id")
        if _has_surrounding_whitespace(round_data.get("child_agent_id")):
            errors.append(
                f"{location}.child_agent_id must not contain surrounding whitespace"
            )

        round_kind = round_data.get("round_kind")
        scope_kind = round_data.get("scope_kind")
        if not _enum_member(round_kind, ROUND_KINDS):
            errors.append(f"{location}.round_kind is invalid")
        if not _enum_member(scope_kind, SCOPE_KINDS):
            errors.append(f"{location}.scope_kind is invalid")
        if round_kind == "automatic_aot" and scope_kind != "automatic_aot":
            errors.append(
                f"{location} automatic_aot round requires automatic_aot scope_kind"
            )
        if round_kind == "base" and scope_kind != "manual":
            errors.append(f"{location} base round requires manual scope_kind")
        if _enum_member(round_kind, ROUND_KINDS):
            optimization_round_kinds.append(round_kind)
            optimization_rounds.append((location, round_data))

        lifecycle = round_data.get("lifecycle")
        if not isinstance(lifecycle, dict):
            errors.append(f"{location}.lifecycle must be an object")
        elif lifecycle.get("profiling") == "passed" and _is_canonical_string(round_id):
            profiling_round_ids.add(round_id)
            if lifecycle.get("correctness") != "passed":
                errors.append(
                    f"{location}.lifecycle.correctness must be passed before profiling"
                )
            errors.extend(_validate_profiling_binding(round_data, location))

        errors.extend(_validate_fingerprints(round_data, location))
        for field in (
            "source_revision",
            "baseline_config_fingerprint",
            "control_fingerprint",
            "workload_fingerprint",
        ):
            if _is_non_empty_string(round_data.get(field)) and round_data.get(
                field
            ) != result.get(field):
                errors.append(f"{location}.{field} must match the experiment")

        if (
            round_kind != "automatic_aot"
            or "performance_decision_range_ids" in round_data
        ):
            errors.extend(
                _validate_string_list(
                    round_data.get("performance_decision_range_ids"),
                    f"{location}.performance_decision_range_ids",
                )
            )
        artifacts = round_data.get("artifacts")
        if not isinstance(artifacts, dict):
            errors.append(f"{location}.artifacts must be an object")
        else:
            for artifact_name in ("round_evidence", "round_report"):
                artifact_path = artifacts.get(artifact_name)
                if not _is_relative_artifact(artifact_path):
                    errors.append(
                        f"{location}.artifacts.{artifact_name} must be a relative path"
                    )

    all_baseline_profile_paths = []
    all_baseline_profile_fingerprints = []
    all_candidate_profile_paths = []
    all_candidate_profile_fingerprints = []
    candidate_profile_paths = []
    candidate_profile_fingerprints = []
    baseline_path_fingerprints = {}
    baseline_fingerprint_paths = {}
    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        baseline_path = round_data.get("baseline_profile")
        if _is_relative_artifact(baseline_path):
            all_baseline_profile_paths.append(Path(baseline_path).as_posix())
        baseline_fingerprint = round_data.get("baseline_profile_fingerprint")
        if _is_non_empty_string(baseline_fingerprint):
            all_baseline_profile_fingerprints.append(baseline_fingerprint.strip())
        if _is_relative_artifact(baseline_path) and _is_non_empty_string(
            baseline_fingerprint
        ):
            normalized_baseline_path = Path(baseline_path).as_posix()
            normalized_baseline_fingerprint = baseline_fingerprint.strip()
            baseline_path_fingerprints.setdefault(normalized_baseline_path, set()).add(
                normalized_baseline_fingerprint
            )
            baseline_fingerprint_paths.setdefault(
                normalized_baseline_fingerprint, set()
            ).add(normalized_baseline_path)
        candidate_path = round_data.get("candidate_profile")
        if _is_relative_artifact(candidate_path):
            all_candidate_profile_paths.append(Path(candidate_path).as_posix())
        candidate_fingerprint = round_data.get("candidate_profile_fingerprint")
        if _is_non_empty_string(candidate_fingerprint):
            all_candidate_profile_fingerprints.append(candidate_fingerprint.strip())
        candidate_round_id = round_data.get("round_id")
        if not _is_canonical_string(candidate_round_id) or (
            candidate_round_id not in profiling_round_ids
        ):
            continue
        if _is_relative_artifact(candidate_path):
            candidate_profile_paths.append(Path(candidate_path).as_posix())
        if _is_non_empty_string(candidate_fingerprint):
            candidate_profile_fingerprints.append(candidate_fingerprint.strip())
    if len(candidate_profile_paths) != len(set(candidate_profile_paths)):
        errors.append("candidate_profile must be unique across profiling rounds")
    if len(candidate_profile_fingerprints) != len(set(candidate_profile_fingerprints)):
        errors.append(
            "candidate_profile_fingerprint must be unique across profiling rounds"
        )
    if any(len(values) != 1 for values in baseline_path_fingerprints.values()):
        errors.append(
            "baseline profile path must map to exactly one content fingerprint"
        )
    if any(len(values) != 1 for values in baseline_fingerprint_paths.values()):
        errors.append(
            "baseline profile fingerprint must map to exactly one artifact path"
        )
    conflicting_profile_paths = set(all_baseline_profile_paths) & set(
        all_candidate_profile_paths
    )
    if conflicting_profile_paths:
        errors.append(
            "baseline and candidate profile paths must be disjoint: "
            + ", ".join(sorted(conflicting_profile_paths))
        )
    conflicting_profile_fingerprints = set(all_baseline_profile_fingerprints) & set(
        all_candidate_profile_fingerprints
    )
    if conflicting_profile_fingerprints:
        errors.append(
            "baseline and candidate profile fingerprints must be disjoint: "
            + ", ".join(sorted(conflicting_profile_fingerprints))
        )

    if optimization_round_kinds:
        phase = {
            "base": 0,
            "automatic_aot": 0,
            "performance_prune": 1,
            "final": 2,
        }
        starters = [
            kind
            for kind in optimization_round_kinds
            if kind in {"base", "automatic_aot"}
        ]
        if (
            optimization_round_kinds[0] not in {"base", "automatic_aot"}
            or len(starters) != 1
        ):
            errors.append(
                "optimization rounds must start with exactly one base or automatic_aot round"
            )
        expected_scope_kind = (
            "automatic_aot"
            if optimization_round_kinds[0] == "automatic_aot"
            else "manual"
        )
        if any(
            round_data.get("scope_kind") != expected_scope_kind
            for _, round_data in optimization_rounds
        ):
            errors.append(
                "optimization descendants must preserve the starter scope_kind"
            )
        if any(
            phase[current] < phase[previous]
            for previous, current in zip(
                optimization_round_kinds, optimization_round_kinds[1:]
            )
        ):
            errors.append(
                "optimization round transition order must be base/automatic_aot -> prune -> final"
            )
        if optimization_round_kinds.count("final") > 1:
            errors.append("optimization rounds may contain at most one final round")
        if (
            "final" in optimization_round_kinds
            and optimization_round_kinds[-1] != "final"
        ):
            errors.append("final must be the last optimization round")
        requires_final = optimization_round_kinds[0] == "base" or any(
            kind in {"performance_prune", "final"}
            for kind in optimization_round_kinds[1:]
        )
        if (
            requires_final
            and "final" not in optimization_round_kinds
            and promotion_mode != "whole_scope"
        ):
            blockers = result.get("blockers")
            last_lifecycle = optimization_rounds[-1][1].get("lifecycle")
            explicit_blockers = (
                isinstance(blockers, list)
                and bool(blockers)
                and all(_is_non_empty_string(blocker) for blocker in blockers)
            )
            terminal_incomplete = isinstance(last_lifecycle, dict) and any(
                last_lifecycle.get(gate) in ("failed", "blocked", "not_run")
                for gate in ("correctness", "profiling")
            )
            if not (explicit_blockers and terminal_incomplete):
                errors.append(
                    "successful manual experiment requires a fresh final round; "
                    "automatic-optimized experiment has the same requirement; "
                    "missing final is only allowed for a terminal incomplete round "
                    "with explicit non-empty blockers"
                )

    agent_ids = result.get("profiling_analysis_agent_ids")
    analysis_results = result.get("profiling_analysis_result")
    if not isinstance(agent_ids, dict):
        errors.append("profiling_analysis_agent_ids must be an object")
        agent_ids = {}
    if not isinstance(analysis_results, dict):
        errors.append("profiling_analysis_result must be an object")
        analysis_results = {}
    for mapping_name, mapping in (
        ("profiling_analysis_agent_ids", agent_ids),
        ("profiling_analysis_result", analysis_results),
    ):
        missing_rounds = sorted(profiling_round_ids - set(mapping), key=str)
        unknown_rounds = sorted(set(mapping) - profiling_round_ids, key=str)
        if missing_rounds:
            errors.append(
                f"{mapping_name} missing profiling rounds: "
                + ", ".join(map(str, missing_rounds))
            )
        if unknown_rounds:
            errors.append(
                f"{mapping_name} has unknown rounds: "
                + ", ".join(map(str, unknown_rounds))
            )
    valid_agent_ids = []
    for round_id, agent_id in agent_ids.items():
        if not _is_non_empty_string(agent_id):
            errors.append(
                f"profiling_analysis_agent_ids.{round_id} must be a non-empty string"
            )
        else:
            normalized_agent_id = _normalized_id(agent_id)
            valid_agent_ids.append(normalized_agent_id)
            if _has_surrounding_whitespace(agent_id):
                errors.append(
                    f"profiling_analysis_agent_ids.{round_id} must not contain "
                    "surrounding whitespace"
                )
            if normalized_agent_id == _normalized_id(child_agent_id):
                errors.append(
                    f"profiling_analysis_agent_ids.{round_id} must differ from child_agent_id"
                )
    if len(valid_agent_ids) != len(set(valid_agent_ids)):
        errors.append("profiling analysis Agent IDs must be unique across rounds")
    valid_analysis_paths = []
    for round_id, artifact_path in analysis_results.items():
        if not _is_relative_artifact(artifact_path):
            errors.append(
                f"profiling_analysis_result.{round_id} must be a relative artifact path"
            )
        else:
            valid_analysis_paths.append(Path(artifact_path).as_posix())
    if len(valid_analysis_paths) != len(set(valid_analysis_paths)):
        errors.append("profiling analysis result paths must be unique across rounds")
    conflicting_analysis_profile_paths = set(valid_analysis_paths) & (
        set(all_baseline_profile_paths) | set(all_candidate_profile_paths)
    )
    if conflicting_analysis_profile_paths:
        errors.append(
            "profiling analysis result and profile paths must be disjoint: "
            + ", ".join(sorted(conflicting_analysis_profile_paths))
        )

    for (location, round_data), (_, next_round) in zip(
        optimization_rounds, optimization_rounds[1:]
    ):
        lifecycle = round_data.get("lifecycle")
        if not isinstance(lifecycle, dict) or any(
            lifecycle.get(gate) != "passed" for gate in ("correctness", "profiling")
        ):
            errors.append(
                f"{location} optimization round cannot be followed by "
                f"{next_round.get('round_id')} unless correctness and profiling passed"
            )

    collections = {}
    decision_identities = {}
    for field in (
        "performance_scope_decisions",
        "unresolved_performance_ranges",
    ):
        items = result.get(field)
        if not isinstance(items, list):
            errors.append(f"{field} must be a list")
            items = []
        collections[field] = items
        for index, decision in enumerate(items):
            location = f"{field}[{index}]"
            errors.extend(
                _validate_decision(
                    decision,
                    location,
                    collection=field,
                    rounds_by_id=rounds_by_id,
                    round_positions=round_positions,
                    profiling_round_ids=profiling_round_ids,
                    analysis_agent_ids=agent_ids,
                    analysis_results=analysis_results,
                )
            )
            if isinstance(decision, dict):
                decision_round_id = decision.get("round_id")
                decision_range_id = decision.get("range_id")
                if _is_non_empty_string(decision_round_id) and _is_non_empty_string(
                    decision_range_id
                ):
                    identity = (decision_round_id, decision_range_id)
                    previous_collection = decision_identities.get(identity)
                    if previous_collection == field:
                        errors.append(
                            f"{field} contains duplicate round/range identity"
                        )
                    elif previous_collection is not None:
                        errors.append(
                            f"{field} contains a conflicting round/range identity "
                            f"already recorded in {previous_collection}"
                        )
                    else:
                        decision_identities[identity] = field

    decisions = [
        item
        for item in collections["performance_scope_decisions"]
        if isinstance(item, dict)
    ]
    unresolved = [
        item
        for item in collections["unresolved_performance_ranges"]
        if isinstance(item, dict)
    ]
    if promotion_mode == "whole_scope":
        if (
            len(rounds) != 1
            or not isinstance(rounds[0], dict)
            or rounds[0].get("round_kind") not in {"base", "automatic_aot"}
        ):
            errors.append(
                "whole_scope promotion requires exactly one unchanged base or automatic_aot round"
            )
        elif rounds[0].get("round_id") not in profiling_round_ids:
            errors.append(
                "whole_scope promotion requires correctness-passed fresh profiling"
            )
        whole_scope_records = decisions + unresolved
        if not whole_scope_records:
            errors.append("whole_scope promotion requires complete per-SK decisions")
        if any(
            item.get("classification") == "insufficient_evidence"
            for item in whole_scope_records
        ):
            errors.append(
                "whole_scope promotion must not contain insufficient_evidence"
            )
        if any(not _mapping_is_performance_exact(item) for item in whole_scope_records):
            errors.append(
                "whole_scope promotion requires performance-exact mapping for every SK"
            )
        if rounds and isinstance(rounds[0], dict):
            expected_whole_scope_ranges = _string_set(
                rounds[0].get("performance_decision_range_ids")
            )
            recorded_whole_scope_ranges = {
                item.get("range_id")
                for item in whole_scope_records
                if _is_non_empty_string(item.get("range_id"))
            }
            if recorded_whole_scope_ranges != expected_whole_scope_ranges:
                errors.append(
                    "whole_scope promotion decisions must cover the full profiled inventory"
                )
    effective_prunes = [
        item
        for item in decisions
        if item.get("classification") in ("neutral", "regressed")
        and item.get("action") == "prune"
        and item.get("status") in ("applied", "verified")
        and _mapping_is_source_actionable(item)
    ]

    def has_prior_effective_prune(range_id, round_index):
        return any(
            item.get("range_id") == range_id
            and _is_non_empty_string(item.get("round_id"))
            and round_positions.get(item.get("round_id"), len(rounds)) < round_index
            for item in effective_prunes
        )

    def previous_candidate_fingerprint(round_index, *, required_kind=None):
        for previous in reversed(rounds[:round_index]):
            if not isinstance(previous, dict):
                continue
            if (
                required_kind is not None
                and previous.get("round_kind") != required_kind
            ):
                continue
            candidate = previous.get("candidate_config_fingerprint")
            if _is_non_empty_string(candidate):
                return candidate
        return None

    retained_range_ids = {
        item.get("range_id")
        for item in decisions
        if item.get("classification") == "beneficial"
        and item.get("action") == "keep"
        and item.get("status") in ("applied", "verified")
        and _is_non_empty_string(item.get("range_id"))
        and _safe_mapping_get(rounds_by_id, item.get("round_id"), {}).get("round_kind")
        != "final"
    }

    for index, round_data in enumerate(rounds):
        if not isinstance(round_data, dict):
            continue
        location = f"rounds[{index}]"
        round_id = round_data.get("round_id")
        round_kind = round_data.get("round_kind")
        references = round_data.get("performance_decision_range_ids")
        reference_set = _string_set(references)
        if round_kind in {"performance_prune", "final"}:
            option_changes = round_data.get("declared_option_changes")
            if option_changes != []:
                errors.append(
                    f"{location} {round_kind} forbids option changes after Sbest-BASE; "
                    "declared_option_changes must be empty"
                )
            declared_change_set = round_data.get("declared_change_set")
            declared_pointers = (
                declared_change_set.get("allowed_json_pointers")
                if isinstance(declared_change_set, dict)
                else None
            )
            if isinstance(declared_pointers, list):
                for pointer in declared_pointers:
                    if not _is_absolute_rfc6901_pointer(pointer):
                        continue
                    segments = [
                        segment.lower()
                        for segment in _decode_absolute_rfc6901_pointer(pointer)
                    ]
                    if any("option" in segment for segment in segments):
                        errors.append(
                            f"{location}.declared_change_set.allowed_json_pointers "
                            "must not target options after Sbest-BASE"
                        )
                        break
        if round_kind == "base":
            expected = {
                item.get("range_id")
                for item in decisions + unresolved
                if item.get("round_id") == round_id
                and _is_non_empty_string(item.get("range_id"))
            }
            if reference_set != expected:
                errors.append(f"{location} base references must match its decisions")
        elif round_kind == "performance_prune":
            previous_candidate = previous_candidate_fingerprint(index)
            if (
                previous_candidate is not None
                and round_data.get("candidate_config_fingerprint") == previous_candidate
            ):
                errors.append(
                    f"{location} performance_prune candidate fingerprint must change"
                )
            if not reference_set or any(
                not has_prior_effective_prune(range_id, index)
                for range_id in reference_set
            ):
                errors.append(
                    f"{location} performance_prune must reference exact reliable neutral/regressed decisions"
                )
            same_round_evidence_range_ids = {
                item.get("range_id")
                for item in decisions + unresolved
                if item.get("round_id") == round_id
                and _is_non_empty_string(item.get("range_id"))
            }
            conflicting_removed_ranges = reference_set & same_round_evidence_range_ids
            if conflicting_removed_ranges:
                errors.append(
                    f"{location} performance_prune removed range must not have "
                    "same-round performance evidence: "
                    + ", ".join(sorted(conflicting_removed_ranges))
                )
        elif round_kind == "final":
            final_retained = round_data.get("retained_range_ids")
            errors.extend(
                _validate_string_list(final_retained, f"{location}.retained_range_ids")
            )
            retained_set = _string_set(final_retained)
            if retained_set != retained_range_ids:
                errors.append(f"{location} final must reference all retained ranges")
            if reference_set != retained_set:
                errors.append(
                    f"{location} final decision references must match final ranges"
                )
            if round_id not in profiling_round_ids:
                errors.append(f"{location} final requires fresh profiling analysis")
            final_range_ids = retained_set
            final_records = [
                (collection_name, item)
                for collection_name, items in (
                    ("performance_scope_decisions", decisions),
                    ("unresolved_performance_ranges", unresolved),
                )
                for item in items
                if item.get("round_id") == round_id
            ]
            recorded_final_range_ids = {
                item.get("range_id")
                for _, item in final_records
                if _is_non_empty_string(item.get("range_id"))
            }
            if recorded_final_range_ids != final_range_ids:
                errors.append(f"{location} final decisions must match final ranges")
            for range_id in sorted(final_range_ids):
                range_records = [
                    (collection_name, item)
                    for collection_name, item in final_records
                    if item.get("range_id") == range_id
                ]
                valid_keep_records = [
                    item
                    for collection_name, item in range_records
                    if collection_name == "performance_scope_decisions"
                    and item.get("classification") == "beneficial"
                    and item.get("action") == "keep"
                    and item.get("status") == "applied"
                    and _mapping_is_source_actionable(item)
                ]
                if len(valid_keep_records) != 1:
                    errors.append(
                        f"{location} final range {range_id} requires exactly one fresh "
                        "applied keep decision"
                    )
                if len(range_records) != len(valid_keep_records):
                    errors.append(
                        f"{location} final range {range_id} must not contain unresolved, "
                        "regressed, or blocked outcomes"
                    )

    errors.extend(_artifact_role_errors(result))
    errors.extend(_relative_artifact_errors(result))
    return {"valid": not errors, "errors": errors}


def _artifact_path(root, relative_path, location):
    if not _is_relative_artifact(relative_path):
        raise ValueError(f"{location} must be a relative artifact path")
    try:
        resolved = (root / relative_path).resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{location} escapes artifact root") from error
    if not resolved.exists():
        raise ValueError(f"{location} artifact is missing: {relative_path}")
    return resolved


def _validate_analysis_binding(result, round_data, analysis, analysis_path, root):
    round_id = round_data["round_id"]
    location = f"profiling_analysis_result.{round_id}"
    expected_identity = {
        "candidate_name": result["experiment_id"],
        "experiment_id": result["experiment_id"],
        "round_id": round_id,
        "analysis_agent_id": result["profiling_analysis_agent_ids"][round_id],
        "source_revision": result["source_revision"],
    }
    for field, expected in expected_identity.items():
        if analysis.get(field) != expected:
            raise ValueError(f"{location} {field} does not match experiment result")

    field_pairs = {
        "baseline_config_fingerprint": "baseline_config_fingerprint",
        "candidate_config_fingerprint": "candidate_config_fingerprint",
        "control_fingerprint": "control_fingerprint",
        "workload_fingerprint": "workload_fingerprint",
        "baseline_profile_fingerprint": "baseline_profile_fingerprint",
        "candidate_profile_fingerprint": "candidate_profile_fingerprint",
        "declared_change_set": "declared_change_set",
    }
    for analysis_field, round_field in field_pairs.items():
        if analysis.get(analysis_field) != round_data.get(round_field):
            raise ValueError(
                f"{location} {analysis_field} does not match declared round"
            )
    for field in ("baseline", "candidate"):
        if analysis.get(f"{field}_workload_fingerprint") != round_data.get(
            "workload_fingerprint"
        ):
            raise ValueError(
                f"{location} {field}_workload_fingerprint does not match declared round"
            )
        if analysis.get(f"{field}_control_fingerprint") != round_data.get(
            "control_fingerprint"
        ):
            raise ValueError(
                f"{location} {field}_control_fingerprint does not match declared round"
            )

    required_inputs, optional_inputs = recommend_sk_strategy._analysis_input_fields(
        analysis["schema_version"]
    )
    for field in (*required_inputs, *optional_inputs):
        relative_path = analysis["inputs"].get(field)
        if relative_path is None:
            continue
        input_path = recommend_sk_strategy._analysis_input_path(
            result["experiment_id"], analysis, analysis_path, field
        )
        try:
            input_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"{location} inputs.{field} escapes artifact root"
            ) from error
        if not input_path.exists():
            raise ValueError(f"{location} inputs.{field} artifact is missing")

    for field in ("baseline_profile", "candidate_profile"):
        declared_path = _artifact_path(
            root, round_data[field], f"rounds.{round_id}.{field}"
        )
        input_path = recommend_sk_strategy._analysis_input_path(
            result["experiment_id"], analysis, analysis_path, field
        )
        if input_path != declared_path:
            raise ValueError(f"{location} inputs.{field} does not match declared round")


def _validate_analysis_decisions(result, analyses):
    producer_by_round = {}
    for round_id, analysis in analyses.items():
        per_sk = {}
        for item in analysis["per_sk_decisions"]:
            range_id = item["range_id"]
            if range_id in per_sk:
                raise ValueError(
                    f"profiling_analysis_result.{round_id} duplicates range_id={range_id}"
                )
            per_sk[range_id] = item
        scope = {}
        for item in analysis["scope_actions"]:
            range_id = item["range_id"]
            if range_id in scope:
                raise ValueError(
                    f"profiling_analysis_result.{round_id} scope_actions duplicates "
                    f"range_id={range_id}"
                )
            scope[range_id] = item
        if set(scope) != set(per_sk):
            missing = sorted(set(per_sk) - set(scope))
            extra = sorted(set(scope) - set(per_sk))
            raise ValueError(
                f"profiling_analysis_result.{round_id} scope_actions must provide "
                f"complete analyzer coverage; missing={missing}, extra={extra}"
            )
        producer_by_round[round_id] = (per_sk, scope)

    submitted_by_round = {round_id: {} for round_id in producer_by_round}
    for collection in (
        "performance_scope_decisions",
        "unresolved_performance_ranges",
    ):
        for index, decision in enumerate(result[collection]):
            round_id = decision["round_id"]
            range_id = decision["range_id"]
            location = f"{collection}[{index}]"
            submitted_by_round.setdefault(round_id, {}).setdefault(range_id, []).append(
                location
            )
            if round_id not in producer_by_round:
                raise ValueError(
                    f"{location} has no profiling analyzer result for its round"
                )
            per_sk, scope = producer_by_round[round_id]
            producer = per_sk.get(range_id)
            producer_action = scope.get(range_id)
            if producer is None or producer_action is None:
                raise ValueError(
                    f"{location} has no exact analyzer decision for round/range"
                )
            for field in (
                "sk_id",
                "classification",
                "action",
                "mapping_method",
                "mapping_confidence",
                "boundary",
            ):
                if decision.get(field) != producer.get(field):
                    raise ValueError(
                        f"{location}.{field} does not match analyzer per_sk_decisions"
                    )
            for field in (
                "candidate_binding_status",
                "graph_occurrence_fingerprint",
                "mapping_blockers",
            ):
                if field in producer and decision.get(field) != producer.get(field):
                    raise ValueError(
                        f"{location}.{field} does not match analyzer structural proof"
                    )
            for field in (
                "sk_id",
                "classification",
                "action",
                "source_scope",
                "boundary",
            ):
                if decision.get(field) != producer_action.get(field):
                    raise ValueError(
                        f"{location}.{field} does not match analyzer scope_actions"
                    )

    for round_id, (per_sk, _) in producer_by_round.items():
        submitted = submitted_by_round[round_id]
        producer_ids = set(per_sk)
        submitted_ids = set(submitted)
        duplicate_ids = sorted(
            range_id for range_id, locations in submitted.items() if len(locations) != 1
        )
        if submitted_ids != producer_ids or duplicate_ids:
            missing = sorted(producer_ids - submitted_ids)
            extra = sorted(submitted_ids - producer_ids)
            raise ValueError(
                f"profiling_analysis_result.{round_id} handoff collections must "
                "provide complete analyzer coverage exactly once; "
                f"missing={missing}, extra={extra}, duplicates={duplicate_ids}"
            )


def _validate_analysis_evidence(result, artifact_root):
    try:
        root = Path(artifact_root).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"artifact root cannot be resolved: {error}") from error
    if not root.is_dir():
        raise ValueError("artifact root must be an existing directory")
    rounds_by_id = {item["round_id"]: item for item in result["rounds"]}
    analyses = {}
    for round_id, relative_path in sorted(result["profiling_analysis_result"].items()):
        location = f"profiling_analysis_result.{round_id}"
        analysis_path = _artifact_path(root, relative_path, location)
        if not analysis_path.is_file():
            raise ValueError(f"{location} must be a JSON file")
        analysis = recommend_sk_strategy._load_json(analysis_path)
        analysis_validation = recommend_sk_strategy._validate_analysis(
            result["experiment_id"], analysis
        )
        _validate_analysis_binding(
            result, rounds_by_id[round_id], analysis, analysis_path, root
        )
        regenerated = recommend_sk_strategy._run_read_only_reanalysis(
            result["experiment_id"], analysis, analysis_path
        )
        recommend_sk_strategy._validate_analysis(result["experiment_id"], regenerated)
        recommend_sk_strategy._validate_reanalysis_semantics(
            result["experiment_id"], analysis, regenerated
        )
        analyses[round_id] = {
            "per_sk_decisions": analysis_validation["decisions"],
            "scope_actions": analysis_validation["actions"],
        }
    _validate_analysis_decisions(result, analyses)


def validate_experiment_result(result, artifact_root=None):
    try:
        validation = _validate_experiment_result(result)
        validation["artifact_evidence_validated"] = False
        validation["conditional_evidence_eligible"] = False
        if validation["valid"] and artifact_root is not None:
            _validate_analysis_evidence(result, artifact_root)
            validation["artifact_evidence_validated"] = True
            validation["conditional_evidence_eligible"] = True
        return validation
    except Exception as error:
        return {
            "valid": False,
            "artifact_evidence_validated": False,
            "conditional_evidence_eligible": False,
            "errors": [
                "experiment result contains invalid JSON value types: "
                f"{type(error).__name__}: {error}"
            ],
        }


def _promoted_conditional_records(result):
    promoted = []
    for decision in result["performance_scope_decisions"]:
        if (
            decision.get("status") == "verified"
            and decision.get("action") == "prune"
            and decision.get("classification") in ("neutral", "regressed")
            and _mapping_is_source_actionable(decision)
        ):
            promoted.append(("prune", decision))
    return promoted


def _rebuild_conditional_evidence(experiments, artifact_validated_ids=()):
    eligible_ids = set(artifact_validated_ids)
    conditional = {}
    for experiment_id in sorted(experiments):
        if experiment_id not in eligible_ids:
            continue
        result = experiments[experiment_id]
        for evidence_kind, decision in _promoted_conditional_records(result):
            key = _conditional_key(decision)
            evidence = {
                **copy.deepcopy(decision),
                "evidence_kind": evidence_kind,
                "experiment_id": result["experiment_id"],
                "child_agent_id": result["child_agent_id"],
            }
            existing = conditional.get(key)
            if existing is not None and _canonical_json(existing) != _canonical_json(
                evidence
            ):
                raise ValueError(
                    "conflicting conditional performance evidence for composite key "
                    + key
                )
            conditional[key] = evidence
    return dict(sorted(conditional.items()))


def _is_task6_experiment(result):
    return isinstance(result, dict) and any(
        field in result
        for field in (
            "profiling_analysis_agent_ids",
            "profiling_analysis_result",
            "performance_scope_decisions",
            "unresolved_performance_ranges",
        )
    )


def _migrate_v1_ledger(ledger):
    source_experiments = ledger.get("experiments", {})
    if not isinstance(source_experiments, dict):
        raise ValueError("schema 1 ledger experiments must be an object")
    active_experiments = {}
    legacy_experiments = copy.deepcopy(ledger.get("legacy_experiments", {}))
    if not isinstance(legacy_experiments, dict):
        raise ValueError("schema 1 legacy_experiments must be an object")

    for experiment_id, experiment in source_experiments.items():
        if not _is_canonical_string(experiment_id):
            raise ValueError("schema 1 ledger experiment IDs must be canonical strings")
        if experiment_id in legacy_experiments:
            raise ValueError(
                f"schema 1 experiment {experiment_id} has duplicate legacy identity"
            )
        if _is_task6_experiment(experiment):
            validation = validate_experiment_result(experiment)
            if not validation["valid"]:
                raise ValueError(
                    f"schema 1 performance experiment {experiment_id} is invalid: "
                    + "; ".join(validation["errors"])
                )
            if experiment.get("experiment_id") != experiment_id:
                raise ValueError(
                    f"schema 1 performance experiment {experiment_id} identity mismatch"
                )
            active_experiments[experiment_id] = copy.deepcopy(experiment)
        else:
            legacy_experiments[experiment_id] = copy.deepcopy(experiment)

    verified = ledger.get(
        "legacy_verified_exclusions", ledger.get("verified_exclusions", [])
    )
    if not isinstance(verified, list):
        raise ValueError("schema 1 verified_exclusions must be a list")
    _validate_legacy_experiments(legacy_experiments)
    _validate_legacy_verified_exclusions(verified)
    migrated = {
        "schema_version": SCHEMA_VERSION,
        "migrated_from_schema_version": 1,
        "experiments": active_experiments,
        "legacy_experiments": legacy_experiments,
        "legacy_verified_exclusions": copy.deepcopy(verified),
        "artifact_validated_experiment_ids": [],
    }
    migrated["conditional_performance_evidence"] = {}
    return migrated


def _prepare_ledger(ledger):
    if not isinstance(ledger, dict):
        raise ValueError("ledger must be an object")
    json_errors = _json_value_errors(ledger, "ledger")
    if json_errors:
        raise ValueError("; ".join(json_errors))
    if not ledger:
        return {
            "schema_version": SCHEMA_VERSION,
            "experiments": {},
            "legacy_experiments": {},
            "legacy_verified_exclusions": [],
            "artifact_validated_experiment_ids": [],
            "conditional_performance_evidence": {},
        }, False

    version = ledger.get("schema_version", SCHEMA_VERSION)
    if type(version) is not int:
        raise ValueError("unsupported ledger schema version")
    if version == 1:
        return _migrate_v1_ledger(copy.deepcopy(ledger)), True
    if version != SCHEMA_VERSION:
        raise ValueError("unsupported ledger schema version")
    prepared = copy.deepcopy(ledger)
    if "artifact_validated_experiment_ids" not in prepared:
        prepared["artifact_validated_experiment_ids"] = []
        return prepared, True
    return prepared, False


def _validated_experiment_ids(ledger, experiments):
    values = ledger.get("artifact_validated_experiment_ids")
    if not isinstance(values, list):
        raise ValueError("ledger artifact_validated_experiment_ids must be a list")
    if any(not _is_canonical_string(item) for item in values):
        raise ValueError(
            "ledger artifact_validated_experiment_ids must contain canonical strings"
        )
    if len(values) != len(set(values)):
        raise ValueError("ledger artifact_validated_experiment_ids must be unique")
    unknown = set(values) - set(experiments)
    if unknown:
        raise ValueError(
            "ledger artifact_validated_experiment_ids references unknown experiments: "
            + ", ".join(sorted(unknown))
        )
    return set(values)


def _validate_persisted_experiments(experiments):
    if not isinstance(experiments, dict):
        raise ValueError("ledger experiments must be an object")
    for experiment_id, experiment in experiments.items():
        if not _is_canonical_string(experiment_id):
            raise ValueError("ledger experiment IDs must be canonical strings")
        if not isinstance(experiment, dict):
            raise ValueError(f"ledger experiment {experiment_id} must be an object")
        if experiment.get("experiment_id") != experiment_id:
            raise ValueError(f"ledger experiment {experiment_id} identity mismatch")
        validation = validate_experiment_result(experiment)
        if not validation["valid"]:
            raise ValueError(
                f"ledger experiment {experiment_id} is invalid: "
                + "; ".join(validation["errors"])
            )


def _validate_legacy_experiments(legacy_experiments):
    if not isinstance(legacy_experiments, dict):
        raise ValueError("ledger legacy_experiments must be an object")
    for experiment_id, experiment in legacy_experiments.items():
        if not _is_canonical_string(experiment_id):
            raise ValueError("ledger legacy experiment IDs must be canonical strings")
        if not isinstance(experiment, dict):
            raise ValueError(
                f"ledger legacy experiment {experiment_id} must be an object"
            )
        if experiment.get("experiment_id") != experiment_id:
            raise ValueError(
                f"ledger legacy experiment {experiment_id} identity mismatch"
            )
        path_errors = _relative_artifact_errors(
            experiment, f"legacy_experiments.{experiment_id}"
        )
        if path_errors:
            raise ValueError("; ".join(path_errors))


def _validate_legacy_verified_exclusions(legacy_verified_exclusions):
    if not isinstance(legacy_verified_exclusions, list):
        raise ValueError("ledger legacy_verified_exclusions must be a list")
    for index, evidence in enumerate(legacy_verified_exclusions):
        path_errors = _relative_artifact_errors(
            evidence, f"legacy_verified_exclusions[{index}]"
        )
        if path_errors:
            raise ValueError("; ".join(path_errors))


def _validate_historical_artifact_registry(
    experiments, legacy_verified_exclusions, result=None
):
    historical_roles = {}
    role_owners = {}
    historical_sources = list(experiments.items())
    for index, evidence in enumerate(legacy_verified_exclusions):
        owner = (
            evidence.get("experiment_id")
            if isinstance(evidence, dict)
            and _is_canonical_string(evidence.get("experiment_id"))
            else f"legacy_verified_exclusions[{index}]"
        )
        historical_sources.append((owner, evidence))
    for owner, source in historical_sources:
        for role, paths in _artifact_paths_by_role(source).items():
            for path in paths:
                historical_roles.setdefault(path, set()).add(role)
                role_owners.setdefault((role, path), set()).add(owner)
    for path, roles in historical_roles.items():
        if len(roles) > 1:
            raise ValueError(
                f"ledger artifact path {path} has conflicting roles: "
                + ", ".join(sorted(roles))
            )
    for (role, path), owners in role_owners.items():
        if role != "baseline_profile" and len(owners) > 1:
            raise ValueError(
                f"ledger {role} artifact path is reused across experiments: {path}"
            )

    if result is None:
        return

    for role, paths in _artifact_paths_by_role(result).items():
        for path in paths:
            existing_roles = historical_roles.get(path, set())
            if existing_roles and existing_roles != {role}:
                raise ValueError(
                    f"artifact role conflict for {path}: historical "
                    + ", ".join(sorted(existing_roles))
                    + f", new {role}"
                )
            if existing_roles and role != "baseline_profile":
                raise ValueError(
                    f"new experiment requires fresh {role} artifact path: {path}"
                )


def merge_experiment_result(ledger, result, artifact_root=None):
    validation = validate_experiment_result(result, artifact_root=artifact_root)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    merged, migrated = _prepare_ledger(ledger)
    experiments = merged.setdefault("experiments", {})
    _validate_persisted_experiments(experiments)
    validated_ids = _validated_experiment_ids(merged, experiments)
    legacy_experiments = merged.setdefault("legacy_experiments", {})
    _validate_legacy_experiments(legacy_experiments)
    legacy_verified = merged.setdefault("legacy_verified_exclusions", [])
    _validate_legacy_verified_exclusions(legacy_verified)
    duplicate_active_legacy_ids = set(experiments) & set(legacy_experiments)
    if duplicate_active_legacy_ids:
        raise ValueError(
            "ledger active and legacy experiment IDs must be disjoint: "
            + ", ".join(sorted(map(str, duplicate_active_legacy_ids)))
        )
    conditional = merged.setdefault("conditional_performance_evidence", {})
    if not isinstance(conditional, dict):
        raise ValueError("ledger conditional_performance_evidence must be an object")
    rebuilt_conditional = _rebuild_conditional_evidence(experiments, validated_ids)
    if not migrated and _canonical_json(conditional) != _canonical_json(
        rebuilt_conditional
    ):
        raise ValueError(
            "ledger conditional_performance_evidence does not match its derived index"
        )
    merged["conditional_performance_evidence"] = rebuilt_conditional
    historical_experiments = {**legacy_experiments, **experiments}
    _validate_historical_artifact_registry(historical_experiments, legacy_verified)
    experiment_id = result["experiment_id"]
    if experiment_id in legacy_experiments:
        raise ValueError(
            f"legacy experiment ID {experiment_id} cannot be reused by an active experiment"
        )
    existing_result = experiments.get(experiment_id)
    if existing_result is not None:
        if _canonical_json(existing_result) != _canonical_json(result):
            raise ValueError(
                f"experiment {experiment_id} is immutable and cannot be overwritten"
            )
        if validation["artifact_evidence_validated"]:
            validated_ids.add(experiment_id)
            merged["artifact_validated_experiment_ids"] = sorted(validated_ids)
            merged["conditional_performance_evidence"] = _rebuild_conditional_evidence(
                experiments, validated_ids
            )
        return merged

    used_agent_ids = set()
    used_analysis_paths = set()
    for existing_experiment_id, existing_experiment in experiments.items():
        if not isinstance(existing_experiment, dict):
            raise ValueError(
                f"ledger experiment {existing_experiment_id} must be an object"
            )
        existing_agent_ids = existing_experiment.get("profiling_analysis_agent_ids")
        existing_analysis_results = existing_experiment.get("profiling_analysis_result")
        if not isinstance(existing_agent_ids, dict) or not isinstance(
            existing_analysis_results, dict
        ):
            raise ValueError(
                f"ledger experiment {existing_experiment_id} has invalid analysis evidence"
            )
        if any(
            not _is_non_empty_string(value) for value in existing_agent_ids.values()
        ):
            raise ValueError(
                f"ledger experiment {existing_experiment_id} has invalid analysis Agent IDs"
            )
        if any(
            not _is_relative_artifact(path)
            for path in existing_analysis_results.values()
        ):
            raise ValueError(
                f"ledger experiment {existing_experiment_id} has invalid analysis paths"
            )
        used_agent_ids.update(
            _normalized_id(value) for value in existing_agent_ids.values()
        )
        used_analysis_paths.update(
            Path(path).as_posix() for path in existing_analysis_results.values()
        )

    reused_agent_ids = used_agent_ids & set(
        _normalized_id(value)
        for value in result["profiling_analysis_agent_ids"].values()
    )
    if reused_agent_ids:
        raise ValueError(
            "new experiment requires fresh profiling analysis Agent IDs: "
            + ", ".join(sorted(reused_agent_ids))
        )
    normalized_result_paths = {
        Path(path).as_posix() for path in result["profiling_analysis_result"].values()
    }
    reused_analysis_paths = used_analysis_paths & normalized_result_paths
    if reused_analysis_paths:
        raise ValueError(
            "new experiment requires fresh profiling analysis result paths: "
            + ", ".join(sorted(reused_analysis_paths))
        )

    used_candidate_paths = set()
    used_candidate_fingerprints = set()
    used_baseline_paths = set()
    used_baseline_fingerprints = set()
    baseline_path_keys = {}
    baseline_fingerprint_keys = {}
    baseline_path_fingerprints = {}
    baseline_fingerprint_paths = {}
    for existing_experiment_id, existing_experiment in experiments.items():
        reuse_key = _baseline_reuse_key(existing_experiment)
        for round_data in _profiling_rounds(existing_experiment):
            candidate_path = round_data.get("candidate_profile")
            candidate_fingerprint = round_data.get("candidate_profile_fingerprint")
            baseline_path = round_data.get("baseline_profile")
            baseline_fingerprint = round_data.get("baseline_profile_fingerprint")
            if not (
                _is_relative_artifact(candidate_path)
                and _is_non_empty_string(candidate_fingerprint)
                and _is_relative_artifact(baseline_path)
                and _is_non_empty_string(baseline_fingerprint)
            ):
                raise ValueError(
                    f"ledger experiment {existing_experiment_id} has invalid profile evidence"
                )
            used_candidate_paths.add(Path(candidate_path).as_posix())
            used_candidate_fingerprints.add(candidate_fingerprint.strip())
        for baseline_path, baseline_fingerprint in _all_baseline_profile_pairs(
            existing_experiment
        ):
            used_baseline_paths.add(baseline_path)
            used_baseline_fingerprints.add(baseline_fingerprint)
            baseline_path_keys.setdefault(baseline_path, set()).add(reuse_key)
            baseline_fingerprint_keys.setdefault(baseline_fingerprint, set()).add(
                reuse_key
            )
            baseline_path_fingerprints.setdefault(baseline_path, set()).add(
                baseline_fingerprint
            )
            baseline_fingerprint_paths.setdefault(baseline_fingerprint, set()).add(
                baseline_path
            )

    if any(len(values) != 1 for values in baseline_path_fingerprints.values()) or any(
        len(values) != 1 for values in baseline_fingerprint_paths.values()
    ):
        raise ValueError("ledger contains a non-immutable baseline profile registry")

    conflicting_existing_paths = used_baseline_paths & used_candidate_paths
    conflicting_existing_fingerprints = (
        used_baseline_fingerprints & used_candidate_fingerprints
    )
    if conflicting_existing_paths or conflicting_existing_fingerprints:
        raise ValueError("ledger baseline and candidate profile roles must be disjoint")

    result_rounds = _profiling_rounds(result)
    historical_profile_paths = set()
    for existing_experiment in experiments.values():
        historical_profile_paths.update(_all_profile_paths(existing_experiment))
    result_profile_paths = _all_profile_paths(result)
    conflicting_historical_artifact_roles = (
        used_analysis_paths & historical_profile_paths
    )
    if conflicting_historical_artifact_roles:
        raise ValueError(
            "ledger analysis result and profile path roles must be disjoint"
        )
    analysis_reused_profile_paths = normalized_result_paths & historical_profile_paths
    if analysis_reused_profile_paths:
        raise ValueError(
            "profiling analysis result path cannot reuse a historical profile path: "
            + ", ".join(sorted(analysis_reused_profile_paths))
        )
    profile_reused_analysis_paths = result_profile_paths & used_analysis_paths
    if profile_reused_analysis_paths:
        raise ValueError(
            "profile path cannot reuse a historical profiling analysis result path: "
            + ", ".join(sorted(profile_reused_analysis_paths))
        )
    result_candidate_paths = {
        Path(round_data["candidate_profile"]).as_posix() for round_data in result_rounds
    }
    result_candidate_fingerprints = {
        round_data["candidate_profile_fingerprint"].strip()
        for round_data in result_rounds
    }
    result_baseline_pairs = _all_baseline_profile_pairs(result)
    result_baseline_paths = {path for path, _ in result_baseline_pairs}
    result_baseline_fingerprints = {
        fingerprint for _, fingerprint in result_baseline_pairs
    }
    candidate_reused_baseline_paths = used_baseline_paths & result_candidate_paths
    if candidate_reused_baseline_paths:
        raise ValueError(
            "candidate profile path cannot reuse a historical baseline: "
            + ", ".join(sorted(candidate_reused_baseline_paths))
        )
    candidate_reused_baseline_fingerprints = (
        used_baseline_fingerprints & result_candidate_fingerprints
    )
    if candidate_reused_baseline_fingerprints:
        raise ValueError(
            "candidate profile fingerprint cannot reuse a historical baseline: "
            + ", ".join(sorted(candidate_reused_baseline_fingerprints))
        )
    baseline_reused_candidate_paths = used_candidate_paths & result_baseline_paths
    if baseline_reused_candidate_paths:
        raise ValueError(
            "baseline profile path cannot reuse a historical candidate: "
            + ", ".join(sorted(baseline_reused_candidate_paths))
        )
    baseline_reused_candidate_fingerprints = (
        used_candidate_fingerprints & result_baseline_fingerprints
    )
    if baseline_reused_candidate_fingerprints:
        raise ValueError(
            "baseline profile fingerprint cannot reuse a historical candidate: "
            + ", ".join(sorted(baseline_reused_candidate_fingerprints))
        )
    reused_candidate_paths = used_candidate_paths & result_candidate_paths
    if reused_candidate_paths:
        raise ValueError(
            "new experiment requires fresh candidate profile paths: "
            + ", ".join(sorted(reused_candidate_paths))
        )
    reused_candidate_fingerprints = (
        used_candidate_fingerprints & result_candidate_fingerprints
    )
    if reused_candidate_fingerprints:
        raise ValueError(
            "new experiment requires fresh candidate profile fingerprints: "
            + ", ".join(sorted(reused_candidate_fingerprints))
        )

    result_reuse_key = _baseline_reuse_key(result)
    for baseline_path, baseline_fingerprint in result_baseline_pairs:
        historical_fingerprints = baseline_path_fingerprints.get(baseline_path, set())
        if historical_fingerprints and historical_fingerprints != {
            baseline_fingerprint
        }:
            raise ValueError(
                "baseline profile path must retain its content fingerprint: "
                + baseline_path
            )
        historical_paths = baseline_fingerprint_paths.get(baseline_fingerprint, set())
        if historical_paths and historical_paths != {baseline_path}:
            raise ValueError(
                "baseline profile fingerprint must retain its artifact path: "
                + baseline_fingerprint
            )
        if any(
            key != result_reuse_key
            for key in baseline_path_keys.get(baseline_path, set())
        ):
            raise ValueError(
                "baseline profile path may only be reused with the same frozen baseline: "
                + baseline_path
            )
        if any(
            key != result_reuse_key
            for key in baseline_fingerprint_keys.get(baseline_fingerprint, set())
        ):
            raise ValueError(
                "baseline profile fingerprint may only be reused with the same frozen "
                "baseline: " + baseline_fingerprint
            )

    _validate_historical_artifact_registry(
        historical_experiments, legacy_verified, result
    )
    experiments[experiment_id] = copy.deepcopy(result)
    if validation["artifact_evidence_validated"]:
        validated_ids.add(experiment_id)
    merged["artifact_validated_experiment_ids"] = sorted(validated_ids)
    merged["conditional_performance_evidence"] = _rebuild_conditional_evidence(
        experiments, validated_ids
    )
    merged["schema_version"] = SCHEMA_VERSION
    return merged


def _read_json(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {constant}")
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("result", type=Path)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--ledger", type=Path, required=True)
    merge_parser.add_argument("--result", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            result = _read_json(args.result)
        except Exception as error:
            validation = {
                "valid": False,
                "errors": [f"invalid JSON input: {type(error).__name__}: {error}"],
            }
        else:
            validation = validate_experiment_result(
                result, artifact_root=args.result.resolve().parent
            )
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if validation["valid"] else 1

    try:
        ledger = _read_json(args.ledger) if args.ledger.exists() else {}
        result = _read_json(args.result)
        merged = merge_experiment_result(
            ledger, result, artifact_root=args.result.resolve().parent
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    except Exception as error:
        failure = {
            "valid": False,
            "errors": [f"merge failed: {type(error).__name__}: {error}"],
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    print(f"已合并实验交接: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
