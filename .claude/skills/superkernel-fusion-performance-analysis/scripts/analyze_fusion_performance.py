#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Summarize Ascend profiler kernel_details.csv and compare SK intervals."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_contract import (  # noqa: E402 - load sibling scripts after sys.path setup
    _load_json as _load_contract_json,
    _read_stable_input_bytes,
    canonical_sha256,
    normalize_workload,
    round_belongs_to_candidate,
    validate_manifest_set,
)
from render_fusion_performance_report import render_report  # noqa: E402 - load sibling scripts after sys.path setup
from projected_trace_mapping import (  # noqa: E402 - load sibling scripts after sys.path setup
    build_mapping as build_projected_trace_mapping,
    graph_occurrence_fingerprint,
)
from structural_association import (  # noqa: E402 - load sibling scripts after sys.path setup
    load_candidate_kernel_rows,
    load_candidate_profile_manifest,
)
from source_calibration_common import file_sha256  # noqa: E402 - load sibling scripts after sys.path setup
from source_scope_map_v2 import load_source_scope_map_v2  # noqa: E402 - load sibling scripts after sys.path setup


KERNEL_DETAILS = "kernel_details.csv"
SK_START_END = re.compile(
    r"^sk_\d+_(?P<scope>.+?)_start_(?P<start>.+)_end_(?P<end>.+)$"
)
STATIC_KERNEL_OP = re.compile(r"(?:^|_)static_kernel_([A-Za-z0-9]+)_")
MODEL_DIRECTORY = re.compile(r"^model_(?P<model_id>\d+)(?:_|$)")
FUSED_LOG_HEADER = re.compile(
    r"SK Function:\s*(?P<name>.*?),\s*scope id:\s*(?P<scope_id>\d+),\s*Node Count:\s*(?P<count>\d+)"
)
FUSED_LOG_NODE = re.compile(
    r"nodeId:(?P<node_id>\d+),\s*streamId:(?P<stream_id>\d+).*?"
    r"KernelInfos\{funcName:(?P<func_name>[^,}]+),\s*kernelType:(?P<kernel_type>[^,}]+)"
)
LAYER_PATTERNS = (
    re.compile(r"(?:decoder\.layer|layers?|blocks?)\.?[_-]?(?P<layer>\d+)"),
    re.compile(r"layer[_-](?P<layer>\d+)"),
)
COMMUNICATION_OP = re.compile(
    r"(?:allreduce|reducescatter|allgather|alltoall|broadcast|send|recv|hccl)",
    re.IGNORECASE,
)
CORE_FAMILIES = ("CUBE", "VECTOR", "MIX", "COMMUNICATION", "OTHER")
SCALAR_RATIO_FIELDS = ("aic_scalar_ratio", "aiv_scalar_ratio")
HARD_MIN_OCCURRENCES = 3
OCCURRENCE_DISCRIMINATOR_FIELDS = (
    "occurrence_id",
    "iteration_id",
    "request_id",
    "batch_id",
)
CLASSIFICATION_ZH = {
    "beneficial": "明确有性能收益",
    "regressed": "明确性能劣化",
    "neutral": "无明确性能收益",
    "insufficient_evidence": "证据不足",
}
ACTION_BY_CLASSIFICATION = {
    "beneficial": "keep",
    "regressed": "prune",
    "neutral": "prune",
    "insufficient_evidence": "reprofile",
}
_JSON_POINTER_REMOVED = object()
_JSON_ARRAY_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)
COLLECTION_MANIFEST_ARGUMENTS = {
    "baseline_collection_manifest": "baseline_profile",
    "profile_collection_manifest": "candidate_profile",
}
PERFORMANCE_COLLECTION_ROLES = tuple(COLLECTION_MANIFEST_ARGUMENTS.values())


def performance_mapping_exact(method, confidence):
    return (method, confidence) in {
        ("source_scope_map", "exact"),
        ("kernel_projection_structural", "exact_projected_trace"),
    }


def source_mapping_actionable(method, confidence, boundary):
    return (method, confidence) == (
        "source_scope_map",
        "exact",
    ) and _source_interval_proven(boundary)


def scope_action(classification, mapping_method, mapping_confidence, boundary):
    action = ACTION_BY_CLASSIFICATION[classification]
    if action != "prune":
        return action
    return (
        "prune"
        if source_mapping_actionable(mapping_method, mapping_confidence, boundary)
        else "block"
    )


def _source_interval_proven(boundary):
    if not isinstance(boundary, dict):
        return False
    source_file = boundary.get("source_file")
    start_offset = boundary.get("start_offset")
    end_offset = boundary.get("end_offset")
    safe_source_file = (
        isinstance(source_file, str)
        and bool(source_file.strip())
        and not Path(source_file).is_absolute()
        and ".." not in Path(source_file).parts
    )
    return (
        safe_source_file
        and isinstance(start_offset, int)
        and not isinstance(start_offset, bool)
        and start_offset >= 0
        and isinstance(end_offset, int)
        and not isinstance(end_offset, bool)
        and end_offset > start_offset
    )


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


_canonical_sha256 = canonical_sha256


def _stable_identity_ids(identity):
    digest = _canonical_sha256(identity)
    return f"sk-{digest}", f"range-{digest}"


def _disambiguate_report_identity_ids(comparisons):
    by_ids = defaultdict(list)
    for comparison in comparisons:
        by_ids[(comparison["sk_id"], comparison["range_id"])].append(comparison)

    for colliding in by_ids.values():
        if len(colliding) < 2:
            continue
        occurrence_fingerprints = Counter(
            item.get("graph_occurrence_fingerprint")
            for item in colliding
            if isinstance(item.get("graph_occurrence_fingerprint"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["graph_occurrence_fingerprint"])
        )
        generated = set()
        for item in colliding:
            occurrence_fingerprint = item.get("graph_occurrence_fingerprint")
            if occurrence_fingerprints[occurrence_fingerprint] == 1:
                discriminator = {
                    "kind": "graph_occurrence",
                    "graph_occurrence_fingerprint": occurrence_fingerprint,
                }
            else:
                if item.get("action") not in {"reprofile", "block"}:
                    raise ValueError(
                        "actionable duplicate fusion identity lacks a unique "
                        "graph occurrence fingerprint"
                    )
                discriminator = {
                    "kind": "native_inventory_diagnostic",
                    "device_id": item.get("device_id"),
                    "model_id": item.get("model_id"),
                    "raw_sk_id": item.get("raw_sk_id"),
                }
            digest = _canonical_sha256(
                {"identity": item["identity"], "discriminator": discriminator}
            )
            ids = (f"sk-{digest}", f"range-{digest}")
            if ids in generated:
                raise ValueError("duplicate fusion identity cannot be disambiguated")
            generated.add(ids)
            item["sk_id"], item["range_id"] = ids
            item["identity_disambiguation"] = discriminator


def _metadata_child_sequence(group):
    return [node.get("op_type") for node in group.get("nodes", [])]


def _identity_evidence_errors(identity):
    errors = []
    if not identity.get("model_id"):
        errors.append("candidate_identity_model_id_missing")
    if not identity.get("source_scope"):
        errors.append("candidate_identity_source_scope_missing")
    boundary = identity.get("boundary") or {}
    if not boundary.get("start_op") or not boundary.get("end_op"):
        errors.append("candidate_identity_boundary_missing")
    sequence = identity.get("ordered_child_op_sequence")
    if not sequence or any(not op_type for op_type in sequence):
        errors.append("candidate_identity_ordered_child_sequence_missing")
    if errors:
        errors.insert(0, "candidate_identity_incomplete")
    return errors


def _rejected_row_matches_identity(rejected_row, identity):
    boundary = identity.get("boundary") or {}
    signature = (
        identity.get("source_scope"),
        boundary.get("start_op"),
        boundary.get("end_op"),
    )
    return (
        rejected_row.get("model_id") == identity.get("model_id")
        and rejected_row.get("signature") == signature
    )


def _decode_json_pointer_token(token):
    decoded = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            decoded.append(token[index])
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in "01":
            raise ValueError(f"invalid RFC6901 escape in JSON pointer token: {token}")
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _parse_json_array_index(token):
    if not _JSON_ARRAY_INDEX.fullmatch(token):
        raise ValueError(f"invalid RFC6901 array index: {token!r}")
    return int(token)


def _validate_json_pointer(value, pointer):
    if pointer == "":
        raise ValueError("root JSON pointer is not allowed for declared changes")
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"invalid RFC6901 JSON pointer: {pointer!r}")
    tokens = [_decode_json_pointer_token(token) for token in pointer[1:].split("/")]
    current = value
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                return
            current = current[token]
        elif isinstance(current, list):
            index = _parse_json_array_index(token)
            if index >= len(current):
                return
            current = current[index]
        else:
            return


def _mark_json_pointer_removed(value, pointer):
    if pointer == "":
        return _JSON_POINTER_REMOVED
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError(f"invalid RFC6901 JSON pointer: {pointer!r}")
    tokens = [_decode_json_pointer_token(token) for token in pointer[1:].split("/")]
    parent = value
    for token in tokens[:-1]:
        if isinstance(parent, dict):
            if token not in parent:
                return value
            parent = parent[token]
        elif isinstance(parent, list):
            index = _parse_json_array_index(token)
            if index >= len(parent):
                return value
            parent = parent[index]
        else:
            return value
    final = tokens[-1]
    if isinstance(parent, dict):
        if final in parent:
            parent[final] = _JSON_POINTER_REMOVED
    elif isinstance(parent, list):
        index = _parse_json_array_index(final)
        if index < len(parent):
            parent[index] = _JSON_POINTER_REMOVED
    return value


def _compact_json_pointer_removals(value):
    if isinstance(value, dict):
        return {
            key: _compact_json_pointer_removals(child)
            for key, child in value.items()
            if child is not _JSON_POINTER_REMOVED
        }
    if isinstance(value, list):
        return [
            _compact_json_pointer_removals(child)
            for child in value
            if child is not _JSON_POINTER_REMOVED
        ]
    return value


def _remove_json_pointer(value, pointer):
    return _remove_json_pointers(value, [pointer])


def _remove_json_pointers(value, pointers):
    for pointer in pointers:
        _validate_json_pointer(value, pointer)
    normalized = copy.deepcopy(value)
    for pointer in pointers:
        normalized = _mark_json_pointer_removed(normalized, pointer)
    if normalized is _JSON_POINTER_REMOVED:
        return None
    return _compact_json_pointer_removals(normalized)


def _valid_positive_count(value, *, allow_zero=False):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (value >= 0 if allow_zero else value > 0)
    )


def _valid_normalized_workload_value(concept, value):
    if concept in {"model", "mode"}:
        return isinstance(value, str) and bool(value.strip())
    if concept in {"batch", "rank", "iterations"}:
        return _valid_positive_count(value)
    if concept == "warmup":
        return _valid_positive_count(value, allow_zero=True)
    if concept == "runtime":
        return isinstance(value, dict) and bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _valid_positive_count(value)
    if isinstance(value, list):
        return bool(value) and all(
            _valid_positive_count(dimension) for dimension in value
        )
    return isinstance(value, (str, list, dict)) and bool(value)


def _validate_workload_manifest(workload):
    try:
        normalized = normalize_workload(workload)
    except ValueError as error:
        message = str(error)
        match = re.fullmatch(r"workload\.([a-z]+) is required", message)
        if match:
            errors = [f"workload_manifest_missing_{match.group(1)}"]
        else:
            match = re.fullmatch(r"workload\.([a-z]+) aliases conflict", message)
            if match:
                concept = match.group(1)
                errors = [
                    f"workload_manifest_conflicting_{concept}",
                    f"workload_manifest_invalid_{concept}",
                ]
            else:
                errors = ["workload_manifest_invalid"]
        return ["workload_manifest_incomplete", *errors]

    errors = [
        f"workload_manifest_invalid_{concept}"
        for concept, value in normalized.items()
        if not _valid_normalized_workload_value(concept, value)
    ]
    errors = list(dict.fromkeys(errors))
    return ["workload_manifest_incomplete", *errors] if errors else []


def validate_fingerprints(
    baseline_config,
    candidate_config,
    baseline_workload,
    candidate_workload,
    declared_change_set,
):
    if not isinstance(declared_change_set, dict):
        raise ValueError("declared change set must be a JSON object")
    allowed_pointers = declared_change_set.get("allowed_json_pointers")
    if not isinstance(allowed_pointers, list) or not all(
        isinstance(pointer, str) for pointer in allowed_pointers
    ):
        raise ValueError("declared change set allowed_json_pointers must be a list")

    baseline_control = _remove_json_pointers(baseline_config, allowed_pointers)
    candidate_control = _remove_json_pointers(candidate_config, allowed_pointers)
    try:
        baseline_normalized_workload = normalize_workload(baseline_workload)
    except ValueError:
        baseline_normalized_workload = baseline_workload
    try:
        candidate_normalized_workload = normalize_workload(candidate_workload)
    except ValueError:
        candidate_normalized_workload = candidate_workload
    baseline_workload_canonical = _canonical_json(baseline_normalized_workload)
    candidate_workload_canonical = _canonical_json(candidate_normalized_workload)
    baseline_control_canonical = _canonical_json(baseline_control)
    candidate_control_canonical = _canonical_json(candidate_control)
    workload_match = baseline_workload_canonical == candidate_workload_canonical
    control_match = baseline_control_canonical == candidate_control_canonical
    baseline_workload_fingerprint = canonical_sha256(baseline_normalized_workload)
    candidate_workload_fingerprint = canonical_sha256(candidate_normalized_workload)
    baseline_control_fingerprint = _canonical_sha256(baseline_control)
    candidate_control_fingerprint = _canonical_sha256(candidate_control)
    evidence_errors = []
    evidence_errors.extend(_validate_workload_manifest(baseline_workload))
    evidence_errors.extend(_validate_workload_manifest(candidate_workload))
    if not workload_match:
        evidence_errors.append("workload_fingerprint_mismatch")
    if not control_match:
        evidence_errors.append("undeclared_config_differences")
    return {
        "baseline_config_fingerprint": canonical_sha256(baseline_config),
        "candidate_config_fingerprint": canonical_sha256(candidate_config),
        "baseline_workload_fingerprint": baseline_workload_fingerprint,
        "candidate_workload_fingerprint": candidate_workload_fingerprint,
        "workload_fingerprint": (
            baseline_workload_fingerprint if workload_match else None
        ),
        "baseline_control_fingerprint": baseline_control_fingerprint,
        "candidate_control_fingerprint": candidate_control_fingerprint,
        "control_fingerprint": baseline_control_fingerprint if control_match else None,
        "workload_fingerprint_match": workload_match,
        "control_fingerprint_match": control_match,
        "declared_change_set": copy.deepcopy(declared_change_set),
        "evidence_errors": list(dict.fromkeys(evidence_errors)),
    }


def _mapping_coverage(decisions):
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
    child_counts = Counter()
    blocker_counts = Counter()
    bound = exact = ambiguous = unmapped = 0
    for decision in inventoried.values():
        child_count = decision.get("child_count")
        if isinstance(child_count, int) and not isinstance(child_count, bool):
            child_counts["5+" if child_count >= 5 else str(child_count)] += 1
        if decision.get("candidate_binding_status") == "bound":
            bound += 1
        confidence = decision.get("mapping_confidence")
        if confidence == "exact_projected_trace":
            exact += 1
        elif confidence == "ambiguous":
            ambiguous += 1
        else:
            unmapped += 1
        blocker_counts.update(set(decision.get("mapping_blockers") or ()))

    def child_bucket(item):
        return (5, item[0]) if item[0] == "5+" else (int(item[0]), item[0])

    return {
        "total_sk_ids": len(inventoried),
        "bound_sk_ids": bound,
        "exact_projected_trace_sk_ids": exact,
        "ambiguous_sk_ids": ambiguous,
        "unmapped_sk_ids": unmapped,
        "filtered_by_child_count": 0,
        "child_count_distribution": dict(
            sorted(child_counts.items(), key=child_bucket)
        ),
        "blocker_counts": dict(sorted(blocker_counts.items())),
    }


def _structural_original(baseline_occurrences):
    occurrences = (baseline_occurrences or {}).get("occurrences") or []
    if not occurrences:
        return None
    interval = _stats([item["interval_us"] for item in occurrences])
    duration_sum = _stats([item["duration_sum_us"] for item in occurrences])
    union_duration = _stats([item["union_duration_us"] for item in occurrences])
    example_children = occurrences[0].get("children") or []
    stream_ids = sorted(
        {
            child.get("baseline_stream_id")
            for child in example_children
            if isinstance(child.get("baseline_stream_id"), int)
            and not isinstance(child.get("baseline_stream_id"), bool)
        }
    )
    return {
        "occurrence_count": len(occurrences),
        "interval_us": interval,
        "duration_sum_us": duration_sum,
        "union_duration_us": union_duration,
        "max_scalar_ratio": 0.0,
        "example_occurrence": {
            "stream_ids": stream_ids or ["projected_stream_role"],
            "stream_count": len(stream_ids) if stream_ids else 1,
            "multi_stream_analysis": {
                "multi_stream_detected": len(stream_ids) > 1,
                "cube_vector_parallel_detected": None,
                "cube_vector_overlap_us": None,
                "cube_vector_overlap_ratio": None,
                "mix_competition_overlap_us": None,
                "same_resource_overlap_us": None,
            },
        },
    }


def _source_scope_entry_matches(
    association,
    source_entry,
    *,
    device_id,
    model_id,
    source_scope,
    ordered_child_ops,
    fusion_boundary,
):
    if association is None or source_entry is None:
        return False
    if not performance_mapping_exact(
        association.get("mapping_method"), association.get("mapping_confidence")
    ):
        return False
    proof = association.get("graph_alignment_proof") or {}
    return (
        str(source_entry.get("device_id")) == str(device_id)
        and str(source_entry.get("model_id")) == str(model_id)
        and source_entry.get("candidate_source_scope") == source_scope
        and source_entry.get("ordered_child_op_sequence") == ordered_child_ops
        and {
            key: (source_entry.get("boundary") or {}).get(key)
            for key in ("start_op", "end_op")
        }
        == fusion_boundary
        and source_entry.get("sk_occurrence_fingerprint")
        == proof.get("graph_occurrence_fingerprint")
        and source_entry.get("baseline_projection_fingerprint")
        == proof.get("mapping_fingerprint")
    )


def _manifest_content_fingerprint(path):
    if path is None:
        return None
    try:
        data, _ = _read_stable_input_bytes(path, "collection manifest")
        return hashlib.sha256(data).hexdigest()
    except (OSError, ValueError):
        return None


def _collection_manifest_protocol(paths):
    supplied = {field: paths.get(field) for field in COLLECTION_MANIFEST_ARGUMENTS}
    content_fingerprints = {
        field: _manifest_content_fingerprint(path) for field, path in supplied.items()
    }
    present = {field for field, path in supplied.items() if path is not None}
    protocol = {
        "protocol": "kernel_projection_trace_v2",
        "status": "not_requested",
        "manifest_set_fingerprint": None,
        "association_config_fingerprint": None,
        "association_control_fingerprint": None,
        "manifest_content_fingerprints": content_fingerprints,
        "blockers": [],
    }
    if not present:
        protocol["blockers"] = ["structural_collection_manifests_missing"]
        return protocol, None
    if present != set(COLLECTION_MANIFEST_ARGUMENTS):
        protocol["status"] = "blocked"
        protocol["blockers"] = ["structural_collection_manifest_set_incomplete"]
        return protocol, None
    manifests = {
        role: supplied[field] for field, role in COLLECTION_MANIFEST_ARGUMENTS.items()
    }
    try:
        summary = validate_manifest_set(
            manifests, expected_roles=PERFORMANCE_COLLECTION_ROLES
        )
    except (OSError, TypeError, ValueError):
        protocol["status"] = "blocked"
        protocol["blockers"] = ["structural_collection_manifest_set_invalid"]
        return protocol, None
    protocol["status"] = "validated"
    protocol["manifest_set_fingerprint"] = summary["set_fingerprint"]
    protocol["association_config_fingerprint"] = summary["config_fingerprint"]
    protocol["association_control_fingerprint"] = summary["control_fingerprint"]
    return protocol, summary


def _manifest_value(path):
    value, _ = _load_contract_json(path, "collection manifest")
    return value


def _manifest_record_fingerprints(path, file_role):
    manifest = _manifest_value(path)
    return [record["sha256"] for record in manifest["files"][file_role]]


def _raw_sk_id(name):
    match = re.match(r"^sk_(\d+)_", str(name or ""))
    return int(match.group(1)) if match else None


def _strict_structural_binding(
    structural_associations, *, device_id, model_id, raw_sk_id
):
    normalized_model_id = int(model_id) if str(model_id).isdecimal() else None
    association = structural_associations.get(
        (device_id, normalized_model_id, raw_sk_id)
    )
    if association is None:
        return None
    binding = association.get("candidate_binding") or {}
    process = binding.get("process_identity") or {}
    ordered_child_ops = binding.get("ordered_child_ops") or []
    child_count = binding.get("child_count")
    valid = (
        binding.get("status") == "bound"
        and process.get("device_id") == device_id
        and process.get("model_id") == normalized_model_id
        and process.get("sk_id") == raw_sk_id
        and isinstance(ordered_child_ops, list)
        and bool(ordered_child_ops)
        and all(isinstance(op, str) and op for op in ordered_child_ops)
        and isinstance(child_count, int)
        and not isinstance(child_count, bool)
        and child_count == len(ordered_child_ops)
    )
    return {
        "association": association,
        "ordered_child_ops": list(ordered_child_ops) if valid else [],
        "child_count": child_count if valid else 0,
        "valid": valid,
    }


def _metadata_groups_by_sequence(matching_metadata_groups, strict_binding):
    grouped = defaultdict(list)
    for group in matching_metadata_groups:
        grouped[_canonical_json(_metadata_child_sequence(group))].append(group)
    if strict_binding is not None and strict_binding["valid"]:
        return {_canonical_json(strict_binding["ordered_child_ops"]): []}
    if not grouped:
        grouped[_canonical_json([])] = []
    return grouped


def _structural_identity_key(device_id, model_id, sk_id):
    return f"device:{device_id}/model:{model_id}/sk:{sk_id}"


def _structural_collection_identity_matches(summary, args, analysis_fingerprints):
    # Association config/control are collection-domain inputs validated across the
    # two profile manifests. Actual execution configs are validated independently through
    # the declared-change contract and are intentionally different JSON domains.
    expected = {
        "workload": analysis_fingerprints.get("workload_fingerprint"),
        "source": args.source_revision,
    }
    observed = {
        "workload": summary["workload_fingerprint"],
        "source": summary["source_revision"],
    }
    return expected == observed


def _projected_trace_structural_context(
    args, protocol, summary, manifest_set, profile_manifest, candidate_rows
):
    """Build the v2 structural context without crossing profiler clock domains."""
    identities = sorted({(row.device_id, row.model_id) for row in candidate_rows.rows})
    associations = {}
    binding_occurrences = []
    graph_fingerprints = {}
    stream_roles = {}
    proofs = {}
    model_results = []
    for device_id, model_id in identities:
        result = build_projected_trace_mapping(
            args.baseline_profile,
            args.profile_collection_manifest,
            device_id=device_id,
            model_id=model_id,
        )
        model_results.append(result)
        for item in result["mappings"]:
            identity = (device_id, model_id, item["sk_id"])
            process = {
                "native_pid": profile_manifest.producer.native_pid,
                "device_id": device_id,
                "model_id": model_id,
                "sk_id": item["sk_id"],
            }
            occurrence_bindings = []
            for occurrence in item["candidate_occurrences"]:
                start_us = Decimal(str(occurrence["start_us"]))
                duration_us = Decimal(str(occurrence["duration_us"]))
                start_ns = int(start_us * 1000)
                end_ns = int((start_us + duration_us) * 1000)
                binding = {
                    "status": "bound" if item["status"] == "exact" else "blocked",
                    "process_identity": copy.deepcopy(process),
                    "step_id": occurrence["step_id"],
                    "parent_interval_ns": [start_ns, end_ns],
                    "ordered_child_node_keys": copy.deepcopy(
                        item["ordered_child_node_keys"]
                    ),
                    "ordered_child_ops": copy.deepcopy(item["ordered_child_ops"]),
                    "child_count": item["child_count"],
                    "evidence_fingerprints": {
                        "kernel_details": result["evidence"][
                            "baseline_kernel_details_sha256"
                        ],
                        "projection_mapping": result["mapping_fingerprint"],
                        "profile_sk_graph_origin": result["evidence"][
                            "profile_origin_graph_fingerprint"
                        ],
                    },
                    "blockers": copy.deepcopy(item["mapping_blockers"]),
                }
                occurrence_bindings.append(binding)
                binding_occurrences.append(binding)
            representative = (
                occurrence_bindings[0]
                if occurrence_bindings
                else {
                    "status": "blocked",
                    "process_identity": process,
                    "step_id": 0,
                    "parent_interval_ns": [0, 1],
                    "ordered_child_node_keys": [],
                    "ordered_child_ops": [],
                    "child_count": 0,
                    "evidence_fingerprints": {},
                    "blockers": item["mapping_blockers"],
                }
            )
            occurrence_fingerprint = graph_occurrence_fingerprint(
                result["mapping_fingerprint"],
                device_id=device_id,
                model_id=model_id,
                sk_id=item["sk_id"],
                ordered_child_node_keys=item["ordered_child_node_keys"],
            )
            association = {
                "mapping_method": item["mapping_method"],
                "mapping_confidence": item["mapping_confidence"],
                "mapping_blockers": copy.deepcopy(item["mapping_blockers"]),
                "candidate_binding_status": representative["status"],
                "candidate_binding": representative,
                "baseline_occurrences": {
                    "status": item["status"],
                    "occurrence_count": len(item["baseline_occurrences"]),
                    "occurrences": copy.deepcopy(item["baseline_occurrences"]),
                    "blockers": copy.deepcopy(item["mapping_blockers"]),
                },
                "canonical_graph_fingerprints": copy.deepcopy(result["evidence"]),
                "stream_role_mapping": copy.deepcopy(result["stream_role_mapping"]),
                "graph_alignment_proof": {
                    "protocol": result["protocol"],
                    "mapping_method": item["mapping_method"],
                    "mapping_confidence": item["mapping_confidence"],
                    "mapping_fingerprint": result["mapping_fingerprint"],
                    "graph_occurrence_fingerprint": occurrence_fingerprint,
                    "alternative_solution_count_by_step": copy.deepcopy(
                        result["alternative_solution_count_by_step"]
                    ),
                },
            }
            associations[identity] = association
            key = _structural_identity_key(*identity)
            graph_fingerprints[key] = association["canonical_graph_fingerprints"]
            stream_roles[key] = association["stream_role_mapping"]
            proofs[key] = association["graph_alignment_proof"]

    revalidated_summary = validate_manifest_set(
        manifest_set, expected_roles=PERFORMANCE_COLLECTION_ROLES
    )
    if revalidated_summary != summary:
        raise ValueError("structural collection changed during projected mapping")
    status_counts = Counter(item["status"] for item in binding_occurrences)
    protocol = copy.deepcopy(protocol)
    protocol["protocol"] = "kernel_projection_trace_v2"
    protocol["status"] = "associated"
    protocol["blockers"] = []
    protocol["model_mapping_fingerprints"] = {
        f"device:{result['identity']['device_id']}/model:{result['identity']['model_id']}": result[
            "mapping_fingerprint"
        ]
        for result in model_results
    }
    protocol["model_projection_exclusions"] = {
        f"device:{result['identity']['device_id']}/model:{result['identity']['model_id']}": copy.deepcopy(
            result["projection_exclusions"]
        )
        for result in model_results
    }
    return {
        "protocol": protocol,
        "associations": associations,
        "candidate_binding_evidence": {
            "summary": {
                "protocol": "kernel_projection_trace_v2",
                "parent_occurrences": len(binding_occurrences),
                "status_counts": dict(sorted(status_counts.items())),
            },
            "occurrences": binding_occurrences,
        },
        "canonical_graph_fingerprints": graph_fingerprints,
        "stream_role_mapping": stream_roles,
        "graph_alignment_proof": proofs,
    }


def _load_structural_context(args, baseline_rows, analysis_fingerprints):
    manifest_paths = {
        field: getattr(args, field) for field in COLLECTION_MANIFEST_ARGUMENTS
    }
    manifest_set = {
        role: manifest_paths[field]
        for field, role in COLLECTION_MANIFEST_ARGUMENTS.items()
        if manifest_paths[field] is not None
    }
    protocol, summary = _collection_manifest_protocol(manifest_paths)
    empty = {
        "protocol": protocol,
        "associations": {},
        "candidate_binding_evidence": {"summary": None, "occurrences": []},
        "canonical_graph_fingerprints": {},
        "stream_role_mapping": {},
        "graph_alignment_proof": {},
    }
    if summary is None:
        return empty

    baseline_profile_sha = _profile_fingerprint(args.baseline_profile)
    candidate_profile_sha = _profile_fingerprint(args.candidate_profile)
    baseline_records = _manifest_record_fingerprints(
        args.baseline_collection_manifest, "kernel_details"
    )
    candidate_records = _manifest_record_fingerprints(
        args.profile_collection_manifest, "kernel_details"
    )
    if (
        not _structural_collection_identity_matches(
            summary, args, analysis_fingerprints
        )
        or baseline_profile_sha not in baseline_records
        or candidate_profile_sha not in candidate_records
    ):
        protocol["status"] = "blocked"
        protocol["blockers"] = ["structural_fingerprint_mismatch"]
        return empty

    try:
        profile_manifest = load_candidate_profile_manifest(
            args.profile_collection_manifest
        )
        candidate_rows = load_candidate_kernel_rows(profile_manifest)
        return _projected_trace_structural_context(
            args,
            protocol,
            summary,
            manifest_set,
            profile_manifest,
            candidate_rows,
        )
    except (OSError, TypeError, ValueError):
        protocol["status"] = "blocked"
        protocol["blockers"] = ["kernel_projection_artifact_association_failed"]
        return empty


def _sorted_stream_ids(rows):
    return sorted({row["stream_id"] for row in rows if row["stream_id"] is not None})


def _to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "N/A":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _to_int(value):
    number = _to_float(value)
    return int(number) if number is not None else None


def _finite_or_none(value):
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _finite_sum(values):
    values = list(values)
    if any(value is None for value in values):
        return None
    return _finite_or_none(sum(values))


def _finite_difference(left, right):
    if left is None or right is None:
        return None
    return _finite_or_none(left - right)


def _finite_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return _finite_or_none(numerator / denominator)


def _finite_percentage(numerator, denominator):
    ratio = _finite_ratio(numerator, denominator)
    return _finite_or_none(ratio * 100) if ratio is not None else None


def _find_kernel_details(path):
    path = Path(path)
    if path.is_file():
        return path
    matches = sorted(path.rglob(KERNEL_DETAILS))
    if not matches:
        raise FileNotFoundError(f"{path}: no {KERNEL_DETAILS} found")
    if len(matches) > 1:
        raise ValueError(
            f"{path}: found multiple {KERNEL_DETAILS} files; pass one exact file"
        )
    return matches[0]


def _profile_fingerprint(path):
    return hashlib.sha256(_find_kernel_details(path).read_bytes()).hexdigest()


def _op_from_function(name):
    if not name:
        return None
    match = STATIC_KERNEL_OP.search(name)
    if match:
        return match.group(1)
    for marker in ("aclnn", "aclnnInplace"):
        if name.startswith(marker):
            tail = name[len(marker) :]
            return tail.split("_", 1)[0] or None
    return name.split("_", 1)[0]


def _normalize_op_type(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower()) or None


def _baseline_row_operator(row):
    row_type = _normalize_op_type(row.get("type"))
    function_type = _normalize_op_type(_op_from_function(row.get("name")))
    if row_type and function_type and row_type != function_type:
        return None
    return row_type or function_type


def _baseline_node_operators_match(rows, expected_op_type):
    expected = _normalize_op_type(expected_op_type)
    if not rows or expected is None:
        return False
    return all(_baseline_row_operator(row) == expected for row in rows)


def _parse_sk_name(name):
    match = SK_START_END.match(name or "")
    if not match:
        return None
    start = match.group("start")
    end = match.group("end")
    return {
        "scope": match.group("scope").strip(),
        "start_function": start,
        "end_function": end,
        "start_op": _op_from_function(start),
        "end_op": _op_from_function(end),
    }


def _signature_from_name(name):
    parsed = _parse_sk_name(name)
    if not parsed:
        return None
    return (parsed["scope"], parsed["start_op"], parsed["end_op"])


def _layer_from_name(name):
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name or "")
        if match:
            return int(match.group("layer"))
    return None


def _scope_segments(source_scope):
    if not isinstance(source_scope, str) or not source_scope.strip():
        return []
    source_scope = source_scope.strip()
    segments = []
    for pattern in LAYER_PATTERNS:
        match = pattern.search(source_scope)
        if not match:
            continue
        suffix = source_scope[match.end() :].strip().strip("._-/").strip()
        if suffix:
            segment = re.split(r"[._/\-]", suffix, maxsplit=1)[0].strip()
            if segment:
                segments.append(segment)
        break
    return sorted(set(segments))


def _model_id_from_artifact(path, root):
    root = Path(root).resolve()
    for parent in Path(path).resolve().parents:
        match = MODEL_DIRECTORY.match(parent.name)
        if match:
            return match.group("model_id")
        if parent == root:
            break
    return None


def _core_family(accelerator_core, op_type="", name="", kernel_type=""):
    text = " ".join(
        str(value or "") for value in (accelerator_core, kernel_type)
    ).upper()
    op_text = f"{op_type} {name}"
    if COMMUNICATION_OP.search(op_text):
        return "COMMUNICATION"
    if "MIX" in text:
        return "MIX"
    if any(marker in text for marker in ("AI_VECTOR", "AIV", "VECTOR")):
        return "VECTOR"
    if any(marker in text for marker in ("AI_CORE", "AIC", "CUBE")):
        return "CUBE"
    return "OTHER"


def _row_layer_from_map(row, layer_map):
    if not layer_map or row["task_id"] is None:
        return None
    for item in layer_map:
        model_id = item.get("model_id")
        if model_id is not None and str(model_id) != row["model_id"]:
            continue
        if item["start_task_id"] <= row["task_id"] <= item["end_task_id"]:
            return item["layer"]
    return None


def _load_layer_map(path):
    if not path:
        return []
    data = json.loads(Path(path).read_text(errors="replace"))
    if isinstance(data, dict) and data.get("protocol") == "source_scope_map_v2":
        return []
    ranges = data.get("task_ranges", data) if isinstance(data, dict) else data
    if not isinstance(ranges, list):
        raise ValueError(f"{path}: layer map must be a list or contain task_ranges")
    normalized = []
    for index, item in enumerate(ranges):
        try:
            layer = int(item["layer"])
            start = int(item["start_task_id"])
            end = int(item["end_task_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: invalid layer range at index {index}: {exc}"
            ) from exc
        if start > end:
            raise ValueError(
                f"{path}: start_task_id exceeds end_task_id at index {index}"
            )
        source_scope = item.get("source_scope", item.get("scope"))
        if (
            "source_scope" in item
            and "scope" in item
            and item["source_scope"] != item["scope"]
        ):
            raise ValueError(
                f"{path}: invalid layer range at index {index}: scope aliases differ"
            )
        if source_scope is not None and (
            not isinstance(source_scope, str) or not source_scope
        ):
            raise ValueError(
                f"{path}: invalid layer range at index {index}: source_scope"
            )
        boundary = item.get("boundary")
        allowed_boundary_fields = {
            "start_op",
            "end_op",
            "source_file",
            "start_offset",
            "end_offset",
        }
        if boundary is not None and (
            not isinstance(boundary, dict)
            or not {"start_op", "end_op"}.issubset(boundary)
            or not set(boundary).issubset(allowed_boundary_fields)
            or not all(
                isinstance(boundary[key], str) and boundary[key]
                for key in ("start_op", "end_op")
            )
            or (
                "source_file" in boundary
                and (
                    not isinstance(boundary["source_file"], str)
                    or not boundary["source_file"].strip()
                )
            )
            or any(
                field in boundary
                and (
                    isinstance(boundary[field], bool)
                    or not isinstance(boundary[field], int)
                    or boundary[field] < 0
                )
                for field in ("start_offset", "end_offset")
            )
        ):
            raise ValueError(f"{path}: invalid layer range at index {index}: boundary")
        sequence = item.get("ordered_child_op_sequence")
        if sequence is not None and (
            not isinstance(sequence, list)
            or not sequence
            or not all(isinstance(op_type, str) and op_type for op_type in sequence)
        ):
            raise ValueError(
                f"{path}: invalid layer range at index {index}: "
                "ordered_child_op_sequence"
            )
        normalized.append(
            {
                "layer": layer,
                "start_task_id": start,
                "end_task_id": end,
                "model_id": item.get("model_id"),
                "source_scope": source_scope,
                "boundary": copy.deepcopy(boundary),
                "ordered_child_op_sequence": copy.deepcopy(sequence),
            }
        )
    return normalized


def load_kernel_rows(path, layer_map=None, include_rejected=False):
    csv_path = _find_kernel_details(path)
    rows = []
    rejected_rows = []
    with csv_path.open(newline="", errors="replace") as fp:
        reader = csv.DictReader(fp)
        for index, row in enumerate(reader):
            name = (row.get("Name") or "").strip()
            kind = (row.get("Type") or "").strip()
            model_id = str(row.get("Model ID") or "").strip()
            task_id = _to_int(row.get("Task ID"))
            boundary = _parse_sk_name(name)
            start = _to_float(row.get("Start Time(us)"))
            duration = _to_float(row.get("Duration(us)"))
            rejection_reason = None
            if start is None:
                rejection_reason = "invalid_or_missing_start"
            elif duration is None:
                rejection_reason = "invalid_or_missing_duration"
            elif duration <= 0:
                rejection_reason = "non_positive_duration"
            if rejection_reason is not None:
                rejected_rows.append(
                    {
                        "index": index,
                        "model_id": model_id,
                        "task_id": task_id,
                        "name": name,
                        "type": kind,
                        "boundary": boundary,
                        "signature": _signature_from_name(name),
                        "reason": rejection_reason,
                    }
                )
                continue
            end = _finite_sum((start, duration))
            item = {
                "index": index,
                "step_id": str(row.get("Step Id") or "").strip(),
                "occurrence_id": str(
                    row.get("Occurrence Id") or row.get("Occurrence ID") or ""
                ).strip(),
                "iteration_id": str(
                    row.get("Iteration Id") or row.get("Iteration ID") or ""
                ).strip(),
                "request_id": str(
                    row.get("Request Id") or row.get("Request ID") or ""
                ).strip(),
                "batch_id": str(
                    row.get("Batch Id") or row.get("Batch ID") or ""
                ).strip(),
                "device_id": _to_int(row.get("Device_id")),
                "model_id": model_id,
                "task_id": task_id,
                "stream_id": _to_int(row.get("Stream ID")),
                "name": name,
                "type": kind,
                "op_state": (row.get("OP State") or "").strip(),
                "accelerator_core": (row.get("Accelerator Core") or "").strip(),
                "start_us": start,
                "duration_us": duration,
                "end_us": end,
                "evidence_errors": (
                    [] if end is not None else ["non_finite_derived_statistics"]
                ),
                "block_num": _to_int(row.get("Block Num")),
                "mix_block_num": _to_int(row.get("Mix Block Num")),
                "wait_us": _to_float(row.get("Wait Time(us)")),
                "aic_scalar_time_us": _to_float(row.get("aic_scalar_time(us)")),
                "aiv_scalar_time_us": _to_float(row.get("aiv_scalar_time(us)")),
                "aic_scalar_ratio": _to_float(row.get("aic_scalar_ratio")),
                "aiv_scalar_ratio": _to_float(row.get("aiv_scalar_ratio")),
                "aic_icache_miss_rate": _to_float(row.get("aic_icache_miss_rate")),
                "aiv_icache_miss_rate": _to_float(row.get("aiv_icache_miss_rate")),
                "layer": _layer_from_name(name),
                "layer_source": "name" if _layer_from_name(name) is not None else None,
                "sk_boundary": boundary,
            }
            item["core_family"] = _core_family(
                item["accelerator_core"], item["type"], item["name"]
            )
            mapped_layer = _row_layer_from_map(item, layer_map)
            if item["layer"] is None and mapped_layer is not None:
                item["layer"] = mapped_layer
                item["layer_source"] = "task_range_map"
            rows.append(item)
    _infer_layers_from_scope_anchors(rows)
    if include_rejected:
        return csv_path, rows, rejected_rows
    return csv_path, rows


def _infer_layers_from_scope_anchors(rows):
    """Fill unlabeled graph rows from layer-bearing SK scope anchors.

    This is intentionally medium-confidence evidence. It is useful for an SK-on
    profile whose generated SK names carry layer scopes, while explicit task ranges
    remain the authority when exact model boundaries matter.
    """
    by_model = defaultdict(list)
    for row in rows:
        if row["task_id"] is not None:
            by_model[row["model_id"]].append(row)
    for model_rows in by_model.values():
        anchors = defaultdict(list)
        for row in model_rows:
            if row["layer"] is not None and row["layer_source"] == "name":
                anchors[row["layer"]].append(row["task_id"])
        ordered = sorted(
            (layer, statistics.median(task_ids))
            for layer, task_ids in anchors.items()
            if task_ids
        )
        if len(ordered) < 2:
            continue
        if any(
            right[0] <= left[0] or right[1] <= left[1]
            for left, right in zip(ordered, ordered[1:])
        ):
            continue
        boundaries = [
            (left[1] + right[1]) / 2 for left, right in zip(ordered, ordered[1:])
        ]
        first_width = ordered[1][1] - ordered[0][1]
        last_width = ordered[-1][1] - ordered[-2][1]
        lower_bound = ordered[0][1] - first_width / 2
        upper_bound = ordered[-1][1] + last_width / 2
        for row in model_rows:
            if row["layer"] is not None:
                continue
            task_id = row["task_id"]
            if task_id < lower_bound or task_id > upper_bound:
                continue
            layer_index = 0
            while layer_index < len(boundaries) and task_id >= boundaries[layer_index]:
                layer_index += 1
            row["layer"] = ordered[layer_index][0]
            row["layer_source"] = "scope_anchor_inference"


def _interval_union(intervals):
    intervals = list(intervals)
    if any(
        start is None
        or end is None
        or not math.isfinite(start)
        or not math.isfinite(end)
        for start, end in intervals
    ):
        return None
    return _finite_sum(
        _finite_difference(end, start) for start, end in _merge_intervals(intervals)
    )


def _merge_intervals(intervals):
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _intersection_duration(left_intervals, right_intervals):
    left_intervals = list(left_intervals)
    right_intervals = list(right_intervals)
    if any(
        start is None or end is None for start, end in left_intervals + right_intervals
    ):
        return None
    left = _merge_intervals(left_intervals)
    right = _merge_intervals(right_intervals)
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            total = _finite_sum((total, _finite_difference(end, start)))
            if total is None:
                return None
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def _max_active_streams(rows):
    events = []
    for row in rows:
        if row["stream_id"] is None or row["end_us"] is None:
            continue
        events.append((row["start_us"], 1, row["stream_id"]))
        events.append((row["end_us"], -1, row["stream_id"]))
    active = defaultdict(int)
    max_streams = 0
    for _, delta, stream_id in sorted(events, key=lambda item: (item[0], item[1])):
        active[stream_id] += delta
        if active[stream_id] <= 0:
            active.pop(stream_id, None)
        max_streams = max(max_streams, len(active))
    return max_streams


def _overlap_pairs(rows, limit=10):
    by_stream = defaultdict(list)
    for row in rows:
        if row["stream_id"] is None or row["end_us"] is None:
            continue
        by_stream[row["stream_id"]].append(row)
    overlaps = []
    streams = sorted(by_stream)
    for i, left_stream in enumerate(streams):
        for right_stream in streams[i + 1 :]:
            overlap_us = 0.0
            pair_count = 0
            for left in by_stream[left_stream]:
                for right in by_stream[right_stream]:
                    overlap = _finite_difference(
                        min(left["end_us"], right["end_us"]),
                        max(left["start_us"], right["start_us"]),
                    )
                    if overlap is not None and overlap > 0:
                        overlap_us = _finite_sum((overlap_us, overlap))
                        if overlap_us is None:
                            break
                        pair_count += 1
            if overlap_us is not None and overlap_us > 0:
                overlaps.append(
                    {
                        "streams": [left_stream, right_stream],
                        "overlap_us": overlap_us,
                        "overlap_pair_count": pair_count,
                    }
                )
    return sorted(overlaps, key=lambda item: item["overlap_us"], reverse=True)[:limit]


def _core_overlap_analysis(rows, limit=20):
    by_stream_family = defaultdict(list)
    for row in rows:
        if row["stream_id"] is None or row["end_us"] is None:
            continue
        by_stream_family[(row["stream_id"], row["core_family"])].append(
            (row["start_us"], row["end_us"])
        )
    stream_ids = sorted({stream for stream, _ in by_stream_family})
    records = []
    family_totals = Counter()
    for index, left_stream in enumerate(stream_ids):
        for right_stream in stream_ids[index + 1 :]:
            for left_family in CORE_FAMILIES:
                left_intervals = by_stream_family.get((left_stream, left_family), [])
                if not left_intervals:
                    continue
                for right_family in CORE_FAMILIES:
                    right_intervals = by_stream_family.get(
                        (right_stream, right_family), []
                    )
                    if not right_intervals:
                        continue
                    overlap_us = _intersection_duration(left_intervals, right_intervals)
                    if overlap_us is None or overlap_us <= 0:
                        continue
                    family_pair = "+".join(sorted((left_family, right_family)))
                    family_total = _finite_sum((family_totals[family_pair], overlap_us))
                    if family_total is None:
                        continue
                    family_totals[family_pair] = family_total
                    records.append(
                        {
                            "streams": [left_stream, right_stream],
                            "core_families": [left_family, right_family],
                            "overlap_us": overlap_us,
                        }
                    )
    cube_vector_us = family_totals.get("CUBE+VECTOR", 0.0)
    mix_competition_us = _finite_sum(
        value
        for pair, value in family_totals.items()
        if "MIX" in pair and pair != "MIX+MIX"
    )
    same_resource_us = _finite_sum(
        family_totals.get(f"{family}+{family}", 0.0)
        for family in ("CUBE", "VECTOR", "MIX")
    )
    device_union = _interval_union((row["start_us"], row["end_us"]) for row in rows)
    if cube_vector_us > 0:
        strategy = "preserve_cube_vector_parallelism"
    elif (mix_competition_us or 0) > 0 or (same_resource_us or 0) > 0:
        strategy = "split_competing_resource_regions"
    else:
        strategy = "single_stream_or_no_useful_overlap"
    return {
        "multi_stream_detected": len(stream_ids) > 1,
        "cube_vector_parallel_detected": cube_vector_us > 0,
        "cube_vector_overlap_us": cube_vector_us,
        "cube_vector_overlap_ratio": _finite_ratio(cube_vector_us, device_union),
        "mix_competition_overlap_us": mix_competition_us,
        "same_resource_overlap_us": same_resource_us,
        "family_overlap_us": [
            {"families": pair.split("+"), "overlap_us": overlap}
            for pair, overlap in family_totals.most_common()
        ],
        "top_stream_core_overlaps": sorted(
            records, key=lambda item: item["overlap_us"], reverse=True
        )[:limit],
        "fusion_strategy_hint": strategy,
    }


def _top_counter(counter, limit):
    return [
        {"name": name, "count": count} for name, count in counter.most_common(limit)
    ]


def _node_sequence(rows):
    by_node = defaultdict(list)
    for row in rows:
        key = (row["model_id"], row["task_id"], row["name"], row["type"])
        by_node[key].append(row)
    sequence = []
    for (model_id, task_id, name, op_type), node_rows in sorted(
        by_node.items(),
        key=lambda item: (
            item[0][0],
            item[0][1] is None,
            item[0][1] if item[0][1] is not None else item[1][0]["index"],
        ),
    ):
        durations = [row["duration_us"] for row in node_rows]
        scalar_ratios = [
            row[field]
            for row in node_rows
            for field in SCALAR_RATIO_FIELDS
            if row[field] is not None
        ]
        sequence.append(
            {
                "model_id": model_id,
                "task_id": task_id,
                "name": name,
                "type": op_type,
                "core_family": node_rows[0]["core_family"],
                "accelerator_core": node_rows[0]["accelerator_core"],
                "op_state": node_rows[0]["op_state"],
                "occurrence_count": len(node_rows),
                "duration_us": _stats(durations),
                "stream_ids": sorted(
                    {
                        row["stream_id"]
                        for row in node_rows
                        if row["stream_id"] is not None
                    }
                ),
                "max_scalar_ratio": max(scalar_ratios) if scalar_ratios else None,
            }
        )
    return sequence


def _summarize_layers(rows, top=20):
    model_rows = [row for row in rows if row["model_id"] not in ("", "4294967295")]
    denominator = len(model_rows) or len(rows)
    labeled_rows = [row for row in (model_rows or rows) if row["layer"] is not None]
    by_layer = defaultdict(list)
    for row in labeled_rows:
        by_layer[row["layer"]].append(row)
    layers = {}
    task_ranges = []
    for layer, layer_rows in sorted(by_layer.items()):
        task_ids = [row["task_id"] for row in layer_rows if row["task_id"] is not None]
        model_ids = sorted({row["model_id"] for row in layer_rows})
        interval = _interval_summary(layer_rows)
        layers[str(layer)] = {
            "row_count": len(layer_rows),
            "graph_node_count": len(_node_sequence(layer_rows)),
            "task_id_range": [min(task_ids), max(task_ids)] if task_ids else None,
            "model_ids": model_ids,
            "layer_sources": _top_counter(
                Counter(row["layer_source"] or "unknown" for row in layer_rows), top
            ),
            "interval": interval,
            "stream_count": len({row["stream_id"] for row in layer_rows}),
            "max_active_streams": _max_active_streams(layer_rows),
            "op_type_counts": _top_counter(
                Counter(row["type"] for row in layer_rows), top
            ),
            "core_family_counts": _top_counter(
                Counter(row["core_family"] for row in layer_rows), top
            ),
            "multi_stream_analysis": _core_overlap_analysis(layer_rows),
            "operator_sequence": _node_sequence(layer_rows),
        }
        if task_ids:
            task_ranges.append(
                {
                    "layer": layer,
                    "model_id": model_ids[0] if len(model_ids) == 1 else None,
                    "start_task_id": min(task_ids),
                    "end_task_id": max(task_ids),
                    "source": "profile_annotation",
                }
            )
    coverage = len(labeled_rows) / denominator if denominator else 0.0
    if coverage >= 0.9 and layers:
        status = "complete"
    elif layers:
        status = "partial"
    else:
        status = "unavailable"
    limitations = []
    if status != "complete":
        limitations.append(
            "Layer coverage is incomplete; use an explicit --layer-map or an SK-on "
            "profile with deterministic layer scope names before claiming every-layer coverage."
        )
    return {
        "status": status,
        "coverage_ratio": coverage,
        "labeled_row_count": len(labeled_rows),
        "eligible_row_count": denominator,
        "layer_count": len(layers),
        "task_ranges": task_ranges,
        "layers": layers,
        "unassigned_op_types": _top_counter(
            Counter(
                row["type"] for row in (model_rows or rows) if row["layer"] is None
            ),
            top,
        ),
        "limitations": limitations,
    }


def summarize_profile(path, top=20, layer_map=None):
    csv_path, rows, rejected_rows = load_kernel_rows(
        path, layer_map=layer_map, include_rejected=True
    )
    if not rows:
        raise ValueError(f"{csv_path}: no kernel rows with start/duration")
    ends = [row["end_us"] for row in rows]
    total_interval = (
        _finite_difference(max(ends), min(row["start_us"] for row in rows))
        if all(end is not None for end in ends)
        else None
    )
    duration_sum = _finite_sum(row["duration_us"] for row in rows)
    evidence_errors = {
        error for row in rows for error in row.get("evidence_errors", [])
    }
    if total_interval is None or duration_sum is None:
        evidence_errors.add("non_finite_derived_statistics")
    stream_ids = _sorted_stream_ids(rows)
    sk_rows = [
        row
        for row in rows
        if row["type"] == "SuperKernel" or row["name"].startswith("sk_")
    ]
    layers = sorted({row["layer"] for row in rows if row["layer"] is not None})
    task_ids = [row["task_id"] for row in rows if row["task_id"] is not None]
    by_layer = defaultdict(list)
    for row in rows:
        if row["layer"] is not None:
            by_layer[row["layer"]].append(row)

    top_duration = sorted(rows, key=lambda row: row["duration_us"], reverse=True)[:top]
    layer_analysis = _summarize_layers(rows, top=top)
    static_rows = [row for row in rows if row["op_state"].lower() == "static"]
    return {
        "path": str(csv_path),
        "rejected_rows": rejected_rows,
        "rejected_row_count": len(rejected_rows),
        "row_count": len(rows),
        "model_ids": sorted({row["model_id"] for row in rows}),
        "task_id_range": [min(task_ids), max(task_ids)] if task_ids else None,
        "device_interval_us": total_interval,
        "duration_sum_us": duration_sum,
        "parallelism_hint": _finite_ratio(duration_sum, total_interval),
        "evidence_errors": sorted(evidence_errors),
        "stream_count": len(stream_ids),
        "stream_ids": stream_ids,
        "max_active_streams": _max_active_streams(rows),
        "top_stream_overlaps": _overlap_pairs(rows),
        "multi_stream_analysis": _core_overlap_analysis(rows),
        "static_kernel": {
            "static_row_count": len(static_rows),
            "non_static_row_count": len(rows) - len(static_rows),
            "static_row_ratio": len(static_rows) / len(rows),
            "model_static_row_ratio": (
                sum(
                    row["op_state"].lower() == "static"
                    for row in rows
                    if row["model_id"] not in ("", "4294967295")
                )
                / max(
                    1,
                    sum(row["model_id"] not in ("", "4294967295") for row in rows),
                )
            ),
        },
        "layer_count": len(layers),
        "layers": layers,
        "layer_analysis": layer_analysis,
        "layers_with_ops": {
            str(layer): {
                "row_count": len(layer_rows),
                "op_types": _top_counter(
                    Counter(row["type"] for row in layer_rows), 10
                ),
                "stream_count": len({row["stream_id"] for row in layer_rows}),
                "core_families": _top_counter(
                    Counter(row["core_family"] for row in layer_rows), 10
                ),
            }
            for layer, layer_rows in sorted(by_layer.items())
        },
        "op_type_counts": _top_counter(Counter(row["type"] for row in rows), top),
        "accelerator_core_counts": _top_counter(
            Counter(row["accelerator_core"] for row in rows), top
        ),
        "superkernel": {
            "count": len(sk_rows),
            "duration_sum_us": _finite_sum(row["duration_us"] for row in sk_rows),
            "top_by_duration": [
                {
                    "task_id": row["task_id"],
                    "stream_id": row["stream_id"],
                    "name": row["name"],
                    "duration_us": row["duration_us"],
                    "boundary": row["sk_boundary"],
                }
                for row in sorted(
                    sk_rows, key=lambda row: row["duration_us"], reverse=True
                )[:top]
            ],
        },
        "top_rows_by_duration": [
            {
                "task_id": row["task_id"],
                "stream_id": row["stream_id"],
                "name": row["name"],
                "type": row["type"],
                "accelerator_core": row["accelerator_core"],
                "core_family": row["core_family"],
                "duration_us": row["duration_us"],
                "start_us": row["start_us"],
                "end_us": row["end_us"],
                "op_state": row["op_state"],
                "layer": row["layer"],
            }
            for row in top_duration
        ],
    }, rows


def _load_sk_meta_groups(root):
    if not root:
        return [], {}
    groups = []
    for log_path in sorted(Path(root).rglob("sk_fused_nodes.log")):
        current = None
        for line in log_path.read_text(errors="replace").splitlines():
            header = FUSED_LOG_HEADER.search(line)
            if header:
                current = {
                    "function": header.group("name"),
                    "model_id": _model_id_from_artifact(log_path, root),
                    "scope_id": int(header.group("scope_id")),
                    "node_count": int(header.group("count")),
                    "nodes": [],
                    "path": str(log_path),
                    "signature": _signature_from_name(header.group("name")),
                }
                groups.append(current)
                continue
            node = FUSED_LOG_NODE.search(line)
            if current and node:
                current["nodes"].append(
                    {
                        "node_id": int(node.group("node_id")),
                        "stream_id": int(node.group("stream_id")),
                        "func_name": node.group("func_name"),
                        "kernel_type": node.group("kernel_type"),
                        "op_type": _op_from_function(node.group("func_name")),
                        "core_family": _core_family(
                            "", "", node.group("func_name"), node.group("kernel_type")
                        ),
                    }
                )
    by_signature = {}
    for group in groups:
        if group["signature"]:
            by_signature.setdefault(group["signature"], []).append(group)
    return groups, by_signature


def _baseline_rows_by_task(rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["model_id"], row["task_id"])].append(row)
        by_key[(None, row["task_id"])].append(row)
    return by_key


def _interval_summary(rows):
    if not rows:
        return None
    rows = [
        (
            row
            if row.get("end_us") is not None
            else {
                **row,
                "end_us": _finite_sum((row["start_us"], row["duration_us"])),
            }
        )
        for row in rows
    ]
    start = min(row["start_us"] for row in rows)
    ends = [row["end_us"] for row in rows]
    end = max(ends) if all(value is not None for value in ends) else None
    interval = _finite_difference(end, start)
    duration_sum = _finite_sum(row["duration_us"] for row in rows)
    union_duration = _interval_union((row["start_us"], row["end_us"]) for row in rows)
    scalar_time = _finite_sum(
        _finite_sum((row["aic_scalar_time_us"] or 0, row["aiv_scalar_time_us"] or 0))
        for row in rows
    )
    evidence_errors = {
        error for row in rows for error in row.get("evidence_errors", [])
    }
    if any(
        value is None
        for value in (end, interval, duration_sum, union_duration, scalar_time)
    ):
        evidence_errors.add("non_finite_derived_statistics")
    return {
        "row_count": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "op_types": _top_counter(Counter(row["type"] for row in rows), 12),
        "core_families": _top_counter(Counter(row["core_family"] for row in rows), 12),
        "stream_ids": _sorted_stream_ids(rows),
        "stream_count": len(_sorted_stream_ids(rows)),
        "start_us": start,
        "end_us": end,
        "interval_us": interval,
        "duration_sum_us": duration_sum,
        "union_duration_us": union_duration,
        "multi_stream_analysis": _core_overlap_analysis(rows),
        "scalar_time_us": scalar_time,
        "max_scalar_ratio": max(
            (
                row[field]
                for row in rows
                for field in SCALAR_RATIO_FIELDS
                if row[field] is not None
            ),
            default=None,
        ),
        "evidence_errors": sorted(evidence_errors),
    }


def _quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return _finite_or_none(
        ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    )


def _robust_stats(values):
    values = sorted(
        value for value in values if value is not None and math.isfinite(value)
    )
    if not values:
        return None
    p50 = _finite_or_none(statistics.median(values))
    deviations = (
        [_finite_difference(value, p50) for value in values] if p50 is not None else []
    )
    absolute_deviations = [
        _finite_or_none(abs(value)) for value in deviations if value is not None
    ]
    return {
        "count": len(values),
        "p50": p50,
        "p90": _quantile(values, 0.9),
        "mad": (
            _finite_or_none(statistics.median(absolute_deviations))
            if len(absolute_deviations) == len(values)
            else None
        ),
        "min": min(values),
        "max": max(values),
    }


def classify_performance(
    baseline,
    candidate,
    *,
    min_occurrences=3,
    min_relative_change_pct=3.0,
    min_absolute_change_us=1.0,
    evidence_errors=(),
):
    errors = list(evidence_errors)
    required_occurrences = max(HARD_MIN_OCCURRENCES, min_occurrences)
    if baseline is None:
        errors.append("baseline_statistics_missing")
    if candidate is None:
        errors.append("candidate_statistics_missing")
    if (
        baseline is not None
        and candidate is not None
        and (
            baseline["count"] < required_occurrences
            or candidate["count"] < required_occurrences
        )
    ):
        errors.append("insufficient_occurrences")
    if (
        errors
        or baseline["count"] < required_occurrences
        or candidate["count"] < required_occurrences
    ):
        return {
            "classification": "insufficient_evidence",
            "classification_zh": CLASSIFICATION_ZH["insufficient_evidence"],
            "noise_threshold_pct": None,
            "improvement_us": None,
            "improvement_pct": None,
            "evidence_errors": errors,
        }
    baseline_p50 = baseline["p50"]
    candidate_p50 = candidate["p50"]
    statistic_values = (
        baseline_p50,
        baseline["mad"],
        candidate_p50,
        candidate["mad"],
        min_relative_change_pct,
        min_absolute_change_us,
    )
    if (
        not all(
            value is not None and math.isfinite(value) for value in statistic_values
        )
        or baseline_p50 == 0
        or candidate_p50 == 0
    ):
        errors.append("non_finite_performance_statistics")
        return {
            "classification": "insufficient_evidence",
            "classification_zh": CLASSIFICATION_ZH["insufficient_evidence"],
            "noise_threshold_pct": None,
            "improvement_us": None,
            "improvement_pct": None,
            "evidence_errors": errors,
        }
    for statistics_name, statistics_value in (
        ("baseline", baseline),
        ("candidate", candidate),
    ):
        if statistics_value["count"] != HARD_MIN_OCCURRENCES:
            continue
        tail_distance = max(
            statistics_value["p50"] - statistics_value["min"],
            statistics_value["max"] - statistics_value["p50"],
        )
        represented_distance = max(
            3 * statistics_value["mad"],
            abs(statistics_value["p50"]) * min_relative_change_pct / 100,
            min_absolute_change_us,
        )
        if tail_distance > represented_distance:
            errors.append("minimum_sample_unrepresented_tail")
            errors.append(f"{statistics_name}_minimum_sample_unrepresented_tail")
    if errors:
        return {
            "classification": "insufficient_evidence",
            "classification_zh": CLASSIFICATION_ZH["insufficient_evidence"],
            "noise_threshold_pct": None,
            "improvement_us": None,
            "improvement_pct": None,
            "evidence_errors": list(dict.fromkeys(errors)),
        }
    baseline_noise_pct = _finite_percentage(baseline["mad"], baseline_p50)
    candidate_noise_pct = _finite_percentage(candidate["mad"], candidate_p50)
    baseline_noise_pct = _finite_or_none(
        baseline_noise_pct * 2 if baseline_noise_pct is not None else None
    )
    candidate_noise_pct = _finite_or_none(
        candidate_noise_pct * 2 if candidate_noise_pct is not None else None
    )
    improvement_us = _finite_difference(baseline_p50, candidate_p50)
    improvement_pct = _finite_percentage(improvement_us, baseline_p50)
    if any(
        value is None
        for value in (
            baseline_noise_pct,
            candidate_noise_pct,
            improvement_us,
            improvement_pct,
        )
    ):
        errors.append("non_finite_derived_statistics")
        return {
            "classification": "insufficient_evidence",
            "classification_zh": CLASSIFICATION_ZH["insufficient_evidence"],
            "noise_threshold_pct": None,
            "improvement_us": None,
            "improvement_pct": None,
            "evidence_errors": errors,
        }
    noise_pct = max(
        min_relative_change_pct,
        baseline_noise_pct,
        candidate_noise_pct,
    )
    if (
        improvement_pct is not None
        and improvement_us >= min_absolute_change_us
        and improvement_pct >= noise_pct
    ):
        classification = "beneficial"
    elif (
        improvement_pct is not None
        and improvement_us <= -min_absolute_change_us
        and improvement_pct <= -noise_pct
    ):
        classification = "regressed"
    else:
        classification = "neutral"
    return {
        "classification": classification,
        "classification_zh": CLASSIFICATION_ZH[classification],
        "noise_threshold_pct": noise_pct,
        "improvement_us": improvement_us,
        "improvement_pct": improvement_pct,
        "evidence_errors": errors,
    }


def _stats(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    result = _robust_stats(values)
    if result is None:
        return None
    total = _finite_sum(values)
    result["mean"] = _finite_ratio(total, len(values))
    return result


def _occurrence_summaries(mapped_rows, node_ids=None):
    if not mapped_rows:
        return (
            [],
            ["baseline_child_occurrence_count_mismatch"] if node_ids else [],
        )
    if not node_ids:
        return [_interval_summary(mapped_rows)], []

    by_node = {}
    for node_id in node_ids:
        rows = [row for row in mapped_rows if row["task_id"] == node_id]
        if rows:
            by_node[node_id] = rows
    child_occurrence_counts = {node_id: len(rows) for node_id, rows in by_node.items()}
    if len(by_node) != len(node_ids) or len(set(child_occurrence_counts.values())) != 1:
        return [], ["baseline_child_occurrence_count_mismatch"]

    domains = {
        (str(row.get("model_id") or "").strip(), row.get("device_id"))
        for rows in by_node.values()
        for row in rows
    }
    if len(domains) != 1 or any(
        not model_id or isinstance(device_id, bool) or not isinstance(device_id, int)
        for model_id, device_id in domains
    ):
        return [], ["baseline_child_occurrence_domain_mismatch"]

    def occurrence_key(row):
        step_id = row.get("step_id")
        if step_id is not None and str(step_id).strip():
            return (
                str(row["model_id"]).strip(),
                row["device_id"],
                "step_id",
                str(step_id).strip(),
            )
        for field in OCCURRENCE_DISCRIMINATOR_FIELDS:
            value = row.get(field)
            if value is not None and str(value).strip():
                return (
                    str(row["model_id"]).strip(),
                    row["device_id"],
                    field,
                    str(value).strip(),
                )
        return None

    rows_by_node_and_key = {}
    for node_id, rows in by_node.items():
        keyed_rows = {}
        for row in rows:
            key = occurrence_key(row)
            if key is None:
                return [], ["baseline_child_occurrence_discriminator_missing"]
            if key in keyed_rows:
                return [], ["baseline_child_occurrence_key_duplicate"]
            keyed_rows[key] = row
        rows_by_node_and_key[node_id] = keyed_rows
    key_sets = [set(rows) for rows in rows_by_node_and_key.values()]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        return [], ["baseline_child_occurrence_key_mismatch"]

    summaries = []
    for key in sorted(key_sets[0], key=_canonical_json):
        occurrence_rows = [rows_by_node_and_key[node_id][key] for node_id in node_ids]
        summaries.append(_interval_summary(occurrence_rows))
    return summaries, []


def _aggregate_original(mapped_rows, node_ids=None):
    summaries, errors = _occurrence_summaries(mapped_rows, node_ids)
    occurrences = [item for item in summaries if item]
    if errors:
        return None, errors
    if not occurrences:
        return None, []
    combined = _interval_summary(mapped_rows)
    derived_errors = {
        error
        for summary in [*occurrences, combined]
        for error in summary.get("evidence_errors", [])
    }
    interval_stats = _stats(item["interval_us"] for item in occurrences)
    duration_sum_stats = _stats(item["duration_sum_us"] for item in occurrences)
    union_duration_stats = _stats(item["union_duration_us"] for item in occurrences)
    if any(
        value is None
        for value in (interval_stats, duration_sum_stats, union_duration_stats)
    ):
        derived_errors.add("non_finite_derived_statistics")
    return (
        {
            "occurrence_count": len(occurrences),
            "interval_us": interval_stats,
            "duration_sum_us": duration_sum_stats,
            "union_duration_us": union_duration_stats,
            "max_scalar_ratio": max(
                (
                    item["max_scalar_ratio"]
                    for item in occurrences
                    if item.get("max_scalar_ratio") is not None
                ),
                default=None,
            ),
            "stream_count": _stats(item["stream_count"] for item in occurrences),
            "combined_repeated_rows": combined,
            "example_occurrence": occurrences[0],
            "evidence_errors": sorted(derived_errors),
        },
        sorted(derived_errors),
    )


def _pct_change(new_value, old_value):
    return _finite_percentage(_finite_difference(new_value, old_value), old_value)


def _fusion_benefit(row, original, metadata_child_node_count):
    child_count = metadata_child_node_count
    if not child_count and original:
        example = original.get("example_occurrence") or {}
        child_count = example.get("row_count")
    launch_count_without_sk = child_count
    launch_count_with_sk = 1 if child_count else None
    original_interval = None
    original_duration_sum = None
    if original:
        interval_stats = original.get("interval_us") or {}
        duration_sum_stats = original.get("duration_sum_us") or {}
        original_interval = interval_stats.get("p50")
        original_duration_sum = duration_sum_stats.get("p50")
    interval_improvement = _finite_difference(original_interval, row["duration_us"])
    duration_sum_improvement = _finite_difference(
        original_duration_sum, row["duration_us"]
    )
    return {
        "child_count": child_count,
        "launch_count_without_sk": launch_count_without_sk,
        "launch_count_with_sk": launch_count_with_sk,
        "estimated_launch_reduction": (
            max(child_count - 1, 0) if child_count is not None else None
        ),
        "sk_duration_us": row["duration_us"],
        "original_interval_p50_us": original_interval,
        "original_duration_sum_p50_us": original_duration_sum,
        "interval_improvement_us": interval_improvement,
        "interval_improvement_pct": _finite_percentage(
            interval_improvement, original_interval
        ),
        "duration_sum_improvement_us": duration_sum_improvement,
        "duration_sum_improvement_pct": _finite_percentage(
            duration_sum_improvement, original_duration_sum
        ),
        "benefit_basis": (
            "profiler_original_child_interval"
            if original_interval is not None
            else "metadata_launch_reduction_only"
        ),
    }


def _stable_symbol_regex(function_name):
    op_type = _op_from_function(function_name)
    if not op_type:
        return None
    return f".*{re.escape(op_type)}.*"


def _fusion_performance_summary(comparisons):
    mapped = [item for item in comparisons if item.get("original")]
    regressions = [item for item in mapped if item.get("classification") == "regressed"]
    improvements = [
        item for item in mapped if item.get("classification") == "beneficial"
    ]
    neutral = [item for item in mapped if item.get("classification") == "neutral"]
    insufficient = [
        item
        for item in comparisons
        if item.get("classification") == "insufficient_evidence"
    ]
    changes = [
        item["sk_vs_original_interval_pct"]
        for item in mapped
        if item.get("sk_vs_original_interval_pct") is not None
    ]
    regression_us = []
    for item in regressions:
        interval_stats = item["original"].get("interval_us") or {}
        regression = _finite_difference(
            item["sk_duration_us"], interval_stats.get("p50")
        )
        if regression is not None:
            regression_us.append(regression)
    return {
        "mapped_sk_count": len(mapped),
        "regressed_sk_count": len(regressions),
        "improved_sk_count": len(improvements),
        "neutral_sk_count": len(neutral),
        "insufficient_evidence_sk_count": len(insufficient),
        "interval_change_pct": _stats(changes),
        "estimated_regression_us": _finite_sum(regression_us),
        "top_regressions": sorted(
            regressions,
            key=lambda item: item.get("sk_vs_original_interval_pct") or 0,
            reverse=True,
        )[:20],
        "dcci_diagnostic_candidates": [],
        "dcci_guardrail": (
            "DCCI hypotheses require explicit runtime state. Use only exact option values "
            "accepted by check_environment.py and re-run correctness plus clean performance."
        ),
    }


def _profile_delta(baseline_summary, candidate_summary):
    return {
        "row_count_delta": candidate_summary["row_count"]
        - baseline_summary["row_count"],
        "row_count_change_pct": _pct_change(
            candidate_summary["row_count"], baseline_summary["row_count"]
        ),
        "device_interval_change_pct": _pct_change(
            candidate_summary["device_interval_us"],
            baseline_summary["device_interval_us"],
        ),
        "duration_sum_change_pct": _pct_change(
            candidate_summary["duration_sum_us"], baseline_summary["duration_sum_us"]
        ),
        "superkernel_row_delta": (
            candidate_summary["superkernel"]["count"]
            - baseline_summary["superkernel"]["count"]
        ),
    }


def _schedule_assessment(baseline_summary, candidate_summary):
    baseline = baseline_summary["multi_stream_analysis"]
    candidate = candidate_summary["multi_stream_analysis"]
    before = baseline["cube_vector_overlap_ratio"] or 0
    after = candidate["cube_vector_overlap_ratio"] or 0
    if not baseline["cube_vector_parallel_detected"]:
        status = "no_baseline_cube_vector_parallelism"
        action = "prefer dependency-aligned deep fusion; auto_op_parallel lacks profiling evidence"
    elif after < before * 0.5:
        status = "possible_cube_vector_serialization"
        action = (
            "inspect sk_prof_device JSON; if child C/V overlap is lost, test accepted "
            "auto_op_parallel=1 before manual scheduling"
        )
    else:
        status = "cube_vector_parallelism_preserved_or_inconclusive"
        action = "confirm child ordering with sk_prof_device JSON for finalist scopes"
    return {
        "status": status,
        "baseline_cube_vector_overlap_ratio": before,
        "candidate_cube_vector_overlap_ratio": after,
        "overlap_ratio_change_pct": _pct_change(after, before),
        "action": action,
        "evidence_limit": (
            "Framework profile overlap is graph-level. Use ASCEND_PROF_SK_ON child trace "
            "to prove scheduling inside a SuperKernel."
        ),
    }


def compare_candidate(
    baseline_rows,
    candidate_name,
    candidate_path,
    sk_meta=None,
    top=20,
    baseline_summary=None,
    layer_map=None,
    min_occurrences=3,
    min_relative_change_pct=3.0,
    min_absolute_change_us=1.0,
    evidence_errors=(),
    structural_associations=None,
    structural_blockers=(),
    source_scope_lookup=None,
):
    structural_associations = structural_associations or {}
    source_scope_lookup = source_scope_lookup or {}
    candidate_summary, candidate_rows = summarize_profile(
        candidate_path, top=top, layer_map=layer_map
    )
    candidate_rejected_rows = candidate_summary.get("rejected_rows", [])
    baseline_rejected_rows = (
        baseline_summary.get("rejected_rows", []) if baseline_summary else []
    )
    _, groups_by_signature = _load_sk_meta_groups(sk_meta)
    baseline_by_task = _baseline_rows_by_task(baseline_rows)
    sk_rows = [
        row
        for row in candidate_rows
        if row["type"] == "SuperKernel" or row["name"].startswith("sk_")
    ]
    candidate_groups = {}
    for row in sk_rows:
        signature = _signature_from_name(row["name"])
        matching_metadata_groups = (
            [
                group
                for group in groups_by_signature.get(signature, [])
                if group.get("model_id") == row["model_id"]
            ]
            if signature
            else []
        )
        boundary = row["sk_boundary"] or {}
        raw_sk_id = _raw_sk_id(row["name"])
        strict_binding = _strict_structural_binding(
            structural_associations,
            device_id=row["device_id"],
            model_id=row["model_id"],
            raw_sk_id=raw_sk_id,
        )
        metadata_by_sequence = _metadata_groups_by_sequence(
            matching_metadata_groups,
            strict_binding,
        )
        for metadata_groups in metadata_by_sequence.values():
            sequence = (
                strict_binding["ordered_child_ops"]
                if strict_binding is not None and strict_binding["valid"]
                else (
                    _metadata_child_sequence(metadata_groups[0])
                    if metadata_groups
                    else []
                )
            )
            identity = {
                "model_id": row["model_id"],
                "source_scope": boundary.get("scope"),
                "ordered_child_op_sequence": sequence,
                "boundary": {
                    "start_op": boundary.get("start_op"),
                    "end_op": boundary.get("end_op"),
                },
            }
            identity_key = _canonical_json(
                {
                    "identity": identity,
                    "device_id": row["device_id"],
                    "raw_sk_id": raw_sk_id,
                }
            )
            candidate_groups.setdefault(
                identity_key,
                {
                    "identity": identity,
                    "device_id": row["device_id"],
                    "raw_sk_id": raw_sk_id,
                    "structural_association": (
                        strict_binding["association"]
                        if strict_binding is not None
                        else None
                    ),
                    "strict_binding": strict_binding,
                    "metadata_groups": metadata_groups,
                    "matching_metadata_group_count": len(
                        matching_metadata_groups
                        if strict_binding is None or not strict_binding["valid"]
                        else [strict_binding]
                    ),
                    "rows": [],
                },
            )["rows"].append(row)
    comparisons = []
    for identity_key in sorted(candidate_groups):
        candidate_group = candidate_groups[identity_key]
        rows = sorted(candidate_group["rows"], key=lambda item: item["start_us"])
        row = rows[0]
        metadata_groups = candidate_group["metadata_groups"]
        association = candidate_group["structural_association"]
        strict_binding = candidate_group["strict_binding"]
        sk_id, range_id = _stable_identity_ids(candidate_group["identity"])
        candidate_durations = [item["duration_us"] for item in rows]
        candidate_duration = _robust_stats(candidate_durations)
        candidate_derived_errors = {
            error for item in rows for error in item.get("evidence_errors", [])
        }
        if _finite_sum(candidate_durations) is None:
            candidate_derived_errors.add("non_finite_derived_statistics")
        if any(
            _rejected_row_matches_identity(item, candidate_group["identity"])
            for item in candidate_rejected_rows
        ):
            candidate_derived_errors.add("invalid_profile_occurrence")
        node_ids = []
        mapping_method = None
        mapping_confidence = "diagnostic_only"
        mapping_hints = []
        decision_errors = [*evidence_errors, *sorted(candidate_derived_errors)]
        mapping_errors = _identity_evidence_errors(candidate_group["identity"])
        mapping_blockers = []
        matching_metadata_group_count = candidate_group["matching_metadata_group_count"]
        if matching_metadata_group_count != 1:
            mapping_errors.extend(
                [
                    "candidate_metadata_identity_ambiguous",
                    "metadata_group_count_mismatch",
                ]
            )
        raw_mapped = []
        raw_node_ids = []
        raw_mapping_errors = []
        if association is not None:
            if strict_binding is None or not strict_binding["valid"]:
                mapping_errors.append("candidate_binding_not_exact")
        elif metadata_groups:
            if any(
                group["node_count"] != len(group["nodes"]) for group in metadata_groups
            ):
                mapping_errors.append("metadata_child_node_count_mismatch")
            metadata_nodes = [
                node for group in metadata_groups for node in group["nodes"]
            ]
            raw_node_ids = [node["node_id"] for node in metadata_nodes]
            if len(raw_node_ids) != len(set(raw_node_ids)):
                mapping_errors.append("metadata_child_node_id_duplicate")
            if not mapping_errors:
                for node in metadata_nodes:
                    child_rows = baseline_by_task.get(
                        (row["model_id"], node["node_id"]), []
                    )
                    if not child_rows:
                        continue
                    if not _baseline_node_operators_match(child_rows, node["op_type"]):
                        raw_mapping_errors.append("baseline_child_operator_mismatch")
                        break
                    raw_mapped.extend(child_rows)
        else:
            mapping_errors.append("exact_mapping_missing")

        if raw_node_ids:
            other_model_rows = sum(
                (baseline_by_task.get((None, node_id), []) for node_id in raw_node_ids),
                [],
            )
            if other_model_rows:
                mapping_hints.append(
                    {
                        "method": "task_id_any_model",
                        "confidence": "diagnostic_only",
                        "baseline_row_count": len(other_model_rows),
                    }
                )
        if row["task_id"] is not None:
            hinted_rows = baseline_by_task.get((row["model_id"], row["task_id"]), [])
            if hinted_rows:
                mapping_hints.append(
                    {
                        "method": "same_task_id",
                        "confidence": "diagnostic_only",
                        "baseline_row_count": len(hinted_rows),
                    }
                )
        boundary_ops = {
            op
            for op in (
                (row["sk_boundary"] or {}).get("start_op"),
                (row["sk_boundary"] or {}).get("end_op"),
            )
            if op
        }
        name_matches = [
            baseline_row
            for baseline_row in baseline_rows
            if _op_from_function(baseline_row["name"]) in boundary_ops
        ]
        if name_matches:
            mapping_hints.append(
                {
                    "method": "name_boundary",
                    "confidence": "diagnostic_only",
                    "baseline_row_count": len(name_matches),
                }
            )

        structural_key = (
            row["device_id"],
            int(row["model_id"]) if str(row["model_id"]).isdecimal() else None,
            candidate_group["raw_sk_id"],
        )
        if association is None:
            association = structural_associations.get(structural_key)
        graph_occurrence_fingerprint = (
            (association.get("graph_alignment_proof") or {}).get(
                "graph_occurrence_fingerprint"
            )
            if association is not None
            else None
        )
        source_entry = source_scope_lookup.get(graph_occurrence_fingerprint)
        candidate_binding_status = (
            association.get("candidate_binding_status") if association else None
        )
        source_identity_exact = _source_scope_entry_matches(
            association,
            source_entry,
            device_id=row["device_id"],
            model_id=row["model_id"],
            source_scope=candidate_group["identity"]["source_scope"],
            ordered_child_ops=candidate_group["identity"]["ordered_child_op_sequence"],
            fusion_boundary=candidate_group["identity"]["boundary"],
        )
        if source_identity_exact and not mapping_errors:
            mapping_method = "source_scope_map"
            mapping_confidence = "exact"
            mapping_blockers = []
            original = _structural_original(association.get("baseline_occurrences"))
            node_ids = []
        elif association is not None:
            mapping_method = association["mapping_method"]
            mapping_confidence = association["mapping_confidence"]
            mapping_blockers = list(association.get("mapping_blockers") or ())
            original = (
                _structural_original(association.get("baseline_occurrences"))
                if performance_mapping_exact(mapping_method, mapping_confidence)
                else None
            )
            node_ids = []
            if not performance_mapping_exact(mapping_method, mapping_confidence):
                mapping_errors.extend(mapping_blockers)
        else:
            mapping_method = "sk_meta_node_ids" if metadata_groups else None
            mapping_confidence = "diagnostic_only"
            mapping_blockers = list(
                dict.fromkeys(
                    [
                        *structural_blockers,
                        "cross_compile_runtime_id_not_identity",
                    ]
                )
            )
            mapping_errors.extend(raw_mapping_errors)
            mapping_errors.extend(mapping_blockers)
            raw_original, raw_occurrence_errors = _aggregate_original(
                raw_mapped, node_ids=raw_node_ids
            )
            mapping_errors.extend(raw_occurrence_errors)
            if raw_original is not None:
                mapping_hints.append(
                    {
                        "method": "sk_meta_node_ids",
                        "confidence": "diagnostic_only",
                        "baseline_occurrence_count": raw_original["occurrence_count"],
                    }
                )
            node_ids = []
            original = None

        decision_errors.extend(mapping_errors)
        if any(
            item.get("model_id") == row["model_id"] and item.get("task_id") in node_ids
            for item in baseline_rejected_rows
        ):
            decision_errors.append("invalid_profile_occurrence")
        if original is None:
            decision_errors.append("baseline_mapping_missing")
        decision_errors = list(dict.fromkeys(decision_errors))
        classification = classify_performance(
            original.get("interval_us") if original else None,
            candidate_duration,
            min_occurrences=min_occurrences,
            min_relative_change_pct=min_relative_change_pct,
            min_absolute_change_us=min_absolute_change_us,
            evidence_errors=decision_errors,
        )
        representative_row = {**row, "duration_us": candidate_duration["p50"]}
        decision_boundary = copy.deepcopy(row["sk_boundary"]) or {}
        if source_identity_exact:
            decision_boundary.update(source_entry["boundary"])
        comparison = {
            "candidate": candidate_name,
            "device_id": row["device_id"],
            "model_id": row["model_id"],
            "raw_sk_id": candidate_group["raw_sk_id"],
            "task_id": row["task_id"],
            "stream_id": row["stream_id"],
            "name": row["name"],
            "boundary": decision_boundary,
            "identity": candidate_group["identity"],
            "sk_id": sk_id,
            "range_id": range_id,
            "graph_occurrence_fingerprint": graph_occurrence_fingerprint,
            "sk_duration_us": candidate_duration["p50"],
            "sk_duration": candidate_duration,
            "mapping_method": mapping_method,
            "mapping_confidence": mapping_confidence,
            "mapping_blockers": list(dict.fromkeys(mapping_blockers)),
            "mapping_hints": mapping_hints,
            "candidate_binding_status": candidate_binding_status,
            "metadata_group_count": matching_metadata_group_count,
            "metadata_child_node_count": (
                strict_binding["child_count"]
                if strict_binding is not None and strict_binding["valid"]
                else sum(len(group["nodes"]) for group in metadata_groups)
            ),
            "metadata_child_functions": [
                node["func_name"]
                for group in metadata_groups
                for node in group["nodes"]
            ],
            "metadata_child_op_types": _top_counter(
                Counter(
                    strict_binding["ordered_child_ops"]
                    if strict_binding is not None and strict_binding["valid"]
                    else (
                        node["op_type"]
                        for group in metadata_groups
                        for node in group["nodes"]
                    )
                ),
                20,
            ),
            "metadata_child_core_families": _top_counter(
                Counter(
                    node["core_family"]
                    for group in metadata_groups
                    for node in group["nodes"]
                ),
                20,
            ),
            "original": original,
            "child_count": (
                association["candidate_binding"].get("child_count", 0)
                if association is not None
                else sum(len(group["nodes"]) for group in metadata_groups)
            ),
            "candidate_occurrence_count": candidate_duration["count"],
        }
        comparison["fusion_benefit"] = _fusion_benefit(
            representative_row,
            original,
            comparison["metadata_child_node_count"],
        )
        comparison.update(classification)
        comparison["action"] = scope_action(
            comparison["classification"],
            comparison["mapping_method"],
            comparison["mapping_confidence"],
            comparison["boundary"],
        )
        if comparison["action"] == "block":
            comparison["action_blocker"] = (
                "prune_requires_exact_source_scope_boundary_mapping"
            )
        if original:
            interval_stats = original.get("interval_us") or {}
            duration_sum_stats = original.get("duration_sum_us") or {}
            comparison["sk_vs_original_interval_pct"] = _pct_change(
                candidate_duration["p50"], interval_stats.get("p50")
            )
            comparison["sk_vs_original_duration_sum_pct"] = _pct_change(
                candidate_duration["p50"], duration_sum_stats.get("p50")
            )
        comparisons.append(comparison)
    _disambiguate_report_identity_ids(comparisons)
    comparable = [item for item in comparisons if item.get("original")]
    report = {
        "summary": candidate_summary,
        "superkernel_inventory": comparisons,
        "superkernel_comparisons": sorted(
            comparisons,
            key=lambda item: (
                item.get("original") is None,
                -((item.get("original") or {}).get("interval_us") or {}).get("p50", 0),
            ),
        )[:top],
        "comparison_coverage": {
            "superkernel_count": len(comparisons),
            "superkernel_occurrence_count": len(sk_rows),
            "mapped_count": len(comparable),
            "mapped_ratio": (
                len(comparable) / len(comparisons) if comparisons else None
            ),
        },
        "fusion_performance": _fusion_performance_summary(comparisons),
    }
    if baseline_summary:
        report["profile_delta"] = _profile_delta(baseline_summary, candidate_summary)
        report["schedule_assessment"] = _schedule_assessment(
            baseline_summary, candidate_summary
        )
    return report


def _find_sk_prof_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    matches = sorted(path.rglob("sk_prof_device_*.json"))
    if not matches:
        raise FileNotFoundError(f"{path}: no sk_prof_device_*.json found")
    return matches


def _walk_trace_events(value):
    if isinstance(value, dict):
        if "ts" in value and ("dur" in value or value.get("ph") in {"B", "E", "X"}):
            yield value
        for key, child in value.items():
            if key not in {"args"}:
                yield from _walk_trace_events(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_trace_events(child)


def summarize_sk_child_profile(path, top=20):
    rows = []
    range_ids = set()
    scoped_event_count = 0
    files = _find_sk_prof_files(path)
    for file_index, json_path in enumerate(files):
        data = json.loads(json_path.read_text(errors="replace"))
        file_range_id = data.get("range_id") if isinstance(data, dict) else None
        for event_index, event in enumerate(_walk_trace_events(data)):
            if event.get("ph") not in (None, "X"):
                continue
            start = _to_float(event.get("ts"))
            duration = _to_float(event.get("dur"))
            if start is None or duration is None or duration <= 0:
                continue
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            name = str(event.get("name") or args.get("name") or "").strip()
            kernel_type = str(
                args.get("kernelType")
                or args.get("kernel_type")
                or args.get("Accelerator Core")
                or ""
            )
            stream_id = (
                _to_int(args.get("streamId"))
                if args.get("streamId") is not None
                else _to_int(args.get("stream_id"))
            )
            if stream_id is None:
                stream_id = _to_int(event.get("tid"))
            range_id = args.get("range_id") or event.get("range_id") or file_range_id
            if isinstance(range_id, str) and range_id:
                range_ids.add(range_id)
                scoped_event_count += 1
            end = _finite_sum((start, duration))
            rows.append(
                {
                    "index": event_index,
                    "step_id": "",
                    "device_id": file_index,
                    "model_id": "",
                    "task_id": _to_int(args.get("nodeId") or args.get("task_id")),
                    "stream_id": stream_id,
                    "name": name,
                    "type": str(args.get("type") or "SK child"),
                    "op_state": "",
                    "accelerator_core": kernel_type,
                    "core_family": _core_family("", "", name, kernel_type),
                    "start_us": start,
                    "duration_us": duration,
                    "end_us": end,
                    "evidence_errors": (
                        [] if end is not None else ["non_finite_derived_statistics"]
                    ),
                    "block_num": _to_int(args.get("numBlocks")),
                    "mix_block_num": None,
                    "wait_us": None,
                    "aic_scalar_time_us": None,
                    "aiv_scalar_time_us": None,
                    "aic_scalar_ratio": None,
                    "aiv_scalar_ratio": None,
                    "aic_icache_miss_rate": None,
                    "aiv_icache_miss_rate": None,
                    "layer": _layer_from_name(name),
                    "layer_source": "name"
                    if _layer_from_name(name) is not None
                    else None,
                    "sk_boundary": _parse_sk_name(name),
                }
            )
    interval = _interval_summary(rows)
    known_stream_event_count = sum(row["stream_id"] is not None for row in rows)
    cube_vector_rows = [row for row in rows if row["core_family"] in {"CUBE", "VECTOR"}]
    return {
        "paths": [str(path) for path in files],
        "range_ids": sorted(range_ids),
        "range_association_complete": (
            scoped_event_count == len(rows) and len(range_ids) == 1
        ),
        "event_count": len(rows),
        "known_stream_event_count": known_stream_event_count,
        "stream_identity_complete": (
            bool(rows) and known_stream_event_count == len(rows)
        ),
        "cube_vector_stream_identity_complete": (
            bool(cube_vector_rows)
            and all(row["stream_id"] is not None for row in cube_vector_rows)
        ),
        "interval": interval,
        "core_family_counts": _top_counter(
            Counter(row["core_family"] for row in rows), top
        ),
        "stream_count": len(
            {row["stream_id"] for row in rows if row["stream_id"] is not None}
        ),
        "multi_stream_analysis": _core_overlap_analysis(rows, limit=top),
        "top_events_by_duration": [
            {
                "name": row["name"],
                "stream_id": row["stream_id"],
                "core_family": row["core_family"],
                "duration_us": row["duration_us"],
            }
            for row in sorted(rows, key=lambda item: item["duration_us"], reverse=True)[
                :top
            ]
        ],
        "diagnostic_only": True,
    }


def _attach_sk_child_schedule(comparison, baseline_summary, child_summary):
    baseline = baseline_summary["multi_stream_analysis"]
    child = child_summary["multi_stream_analysis"]
    if not _child_trace_has_cube_and_vector(
        child_summary
    ) or not _child_trace_has_reliable_cube_vector_streams(child_summary):
        verdict = "child_trace_missing_cube_vector_evidence"
        action = (
            "collect a complete child trace containing both CUBE and VECTOR events "
            "with reliable stream identities"
        )
    elif (
        baseline.get("cube_vector_parallel_detected") is True
        and child.get("cube_vector_parallel_detected") is False
    ):
        verdict = "cube_vector_serialized_inside_sk"
        action = "test accepted auto_op_parallel=1 or split the scope at the C/V dependency boundary"
    elif child.get("cube_vector_parallel_detected") is True:
        verdict = "cube_vector_parallel_inside_sk"
        action = (
            "keep the schedule only if clean worst-rank latency and correctness pass"
        )
    else:
        verdict = "no_cube_vector_parallel_evidence"
        action = "prefer deeper single-stream fusion unless source inspection proves another schedule"
    comparison["sk_child_schedule"] = {
        "verdict": verdict,
        "action": action,
        "profile": child_summary,
    }


def _optional_mapping_field(parent, key, path):
    if key not in parent:
        return None
    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _validated_environment_options(environment_evidence):
    if environment_evidence is None:
        return {}, {}
    if not isinstance(environment_evidence, dict):
        raise ValueError("environment_evidence must be an object")

    options = _optional_mapping_field(
        environment_evidence, "options", "environment_evidence.options"
    )
    optimize_options = {}
    if options is not None:
        optimize_options = (
            _optional_mapping_field(
                options,
                "optimize_options",
                "environment_evidence.options.optimize_options",
            )
            or {}
        )
    for option_name, option_evidence in optimize_options.items():
        option_path = f"environment_evidence.options.optimize_options.{option_name}"
        if not isinstance(option_evidence, dict):
            raise ValueError(f"{option_path} must be an object")
        if "accepted_values" in option_evidence and not isinstance(
            option_evidence["accepted_values"], list
        ):
            raise ValueError(f"{option_path}.accepted_values must be a list")

    accepted_options = (
        _optional_mapping_field(
            environment_evidence,
            "accepted_options",
            "environment_evidence.accepted_options",
        )
        or {}
    )
    for option_name, values in accepted_options.items():
        if not isinstance(values, list):
            raise ValueError(
                f"environment_evidence.accepted_options.{option_name} must be a list"
            )
    return optimize_options, accepted_options


def _accepted_values(environment_evidence, option):
    optimize_options, accepted_options = _validated_environment_options(
        environment_evidence
    )
    option_evidence = optimize_options.get(option)
    if option_evidence is not None and "accepted_values" in option_evidence:
        return (
            option_evidence["accepted_values"],
            f"environment_evidence.options.optimize_options.{option}.accepted_values",
        )
    if option in accepted_options:
        return (
            accepted_options[option],
            f"environment_evidence.accepted_options.{option}",
        )
    return [], None


def _accepted_value_evidence(environment_evidence, option, expected):
    values, source = _accepted_values(environment_evidence, option)
    expected_json = _canonical_json(expected)
    if any(_canonical_json(value) == expected_json for value in values):
        return {"source": source, "accepted_value": expected}
    return None


def _dcci_runtime_evidence(environment_evidence, candidate_config):
    if environment_evidence is None:
        environment_evidence = {}
    elif not isinstance(environment_evidence, dict):
        raise ValueError("environment_evidence must be an object")
    if candidate_config is None:
        candidate_config = {}
    elif not isinstance(candidate_config, dict):
        raise ValueError("candidate_config must be an object")

    environment_runtime_evidence = _optional_mapping_field(
        environment_evidence,
        "runtime_evidence",
        "environment_evidence.runtime_evidence",
    )
    candidate_runtime_evidence = _optional_mapping_field(
        candidate_config,
        "runtime_evidence",
        "candidate_config.runtime_evidence",
    )
    candidate_runtime = _optional_mapping_field(
        candidate_config,
        "runtime",
        "candidate_config.runtime",
    )
    nested_runtime_evidence = (
        _optional_mapping_field(
            candidate_runtime,
            "runtime_evidence",
            "candidate_config.runtime.runtime_evidence",
        )
        if candidate_runtime is not None
        else None
    )
    sources = (
        (
            "environment_evidence.runtime_evidence.dcci_state",
            environment_runtime_evidence,
        ),
        (
            "candidate_config.runtime_evidence.dcci_state",
            candidate_runtime_evidence,
        ),
        (
            "candidate_config.runtime.runtime_evidence.dcci_state",
            nested_runtime_evidence,
        ),
    )
    observed = []
    for source, runtime_evidence in sources:
        if not isinstance(runtime_evidence, dict):
            continue
        state = runtime_evidence.get("dcci_state")
        if state in {"enabled", "disabled", "unknown"}:
            observed.append({"source": source, "state": state})
    known_states = sorted(
        {item["state"] for item in observed if item["state"] != "unknown"}
    )
    if len(known_states) > 1:
        state = "conflict"
    elif known_states:
        state = known_states[0]
    else:
        state = "unknown"
    return {"state": state, "sources": observed}


def _regex_literal_fragments(pattern):
    fragments = []
    current = []
    in_character_class = False

    def flush():
        if current:
            fragments.append("".join(current))
            current.clear()

    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            if index + 1 >= len(pattern):
                flush()
                break
            escaped = pattern[index + 1]
            if in_character_class or escaped.isalnum():
                flush()
            else:
                current.append(escaped)
            index += 2
            continue
        if character == "[":
            flush()
            in_character_class = True
        elif character == "]" and in_character_class:
            in_character_class = False
        elif in_character_class or character in ".^$*+?{}()|":
            flush()
        else:
            current.append(character)
        index += 1
    flush()
    return fragments


def _regex_has_child_literal_evidence(regex, symbols):
    fragments = _regex_literal_fragments(regex)
    evidence_tokens = set(symbols)
    evidence_tokens.update(
        token for token in (_op_from_function(symbol) for symbol in symbols) if token
    )
    return any(token in fragment for token in evidence_tokens for fragment in fragments)


def _accepted_narrow_regex(environment_evidence, option, child_functions):
    values, source = _accepted_values(environment_evidence, option)
    symbols = sorted(
        {item for item in child_functions if isinstance(item, str) and item}
    )
    candidates = []
    for value in values:
        if (
            not isinstance(value, list)
            or len(value) != 1
            or not isinstance(value[0], str)
        ):
            continue
        regex = value[0]
        try:
            compiled = re.compile(regex)
        except re.error:
            continue
        if not any(compiled.fullmatch(symbol) for symbol in symbols):
            continue
        if not _regex_has_child_literal_evidence(regex, symbols):
            continue
        if any(
            compiled.fullmatch(sentinel) is not None
            for sentinel in (
                "",
                "CompletelyUnrelated",
                "static_kernel_Unrelated_hash",
            )
        ):
            continue
        literal_count = sum(
            1 for character in regex if character not in r".^$*+?{}[]()|\\"
        )
        anchor_count = int(regex.startswith("^")) + int(regex.endswith("$"))
        wildcard_count = sum(regex.count(character) for character in ".*+?{")
        candidates.append(
            (
                (-anchor_count, -literal_count, wildcard_count, regex),
                value,
            )
        )
    if candidates:
        _, value = min(candidates, key=lambda item: item[0])
        return value, {"source": source, "accepted_value": value}
    return None, None


def _child_profile_range_id(pruned_decisions, child_summary):
    explicit_range_ids = child_summary.get("range_ids") or []
    if len(explicit_range_ids) == 1:
        if not child_summary.get("range_association_complete"):
            return None
        range_id = explicit_range_ids[0]
        if any(item["range_id"] == range_id for item in pruned_decisions):
            return range_id
        return None
    if not explicit_range_ids and len(pruned_decisions) == 1:
        return pruned_decisions[0]["range_id"]
    return None


def _child_trace_has_cube_and_vector(child_summary):
    counts = {
        item.get("name"): item.get("count", 0)
        for item in child_summary.get("core_family_counts", [])
        if isinstance(item, dict)
    }
    return counts.get("CUBE", 0) > 0 and counts.get("VECTOR", 0) > 0


def _child_trace_has_reliable_cube_vector_streams(child_summary):
    completeness = child_summary.get("cube_vector_stream_identity_complete")
    if completeness is None:
        completeness = child_summary.get("stream_identity_complete")
    return completeness is True


def _append_missing_cube_vector_trace(
    hypotheses,
    blockers,
    range_id,
    *,
    detail_code=None,
    detail_zh=None,
):
    evidence = [
        "baseline_cube_vector_overlap",
        "child_trace_missing_cube_vector_evidence",
    ]
    if detail_code:
        evidence.append(detail_code)
    hypotheses.append(
        {
            "kind": "cube_vector_child_trace_missing",
            "confidence": "low",
            "range_id": range_id,
            "evidence": evidence,
            "explanation_zh": (
                "融合前存在 Cube/Vector 重叠，但缺少可归属该 range 且同时包含 "
                "CUBE、VECTOR 完整事件的 SK child trace，需补采后再判断是否串行化。"
            ),
            "requires_ab_test": True,
        }
    )
    blocker_detail = f"{detail_code}；{detail_zh}；" if detail_code else ""
    blockers.append(
        f"range_id={range_id}：child_trace_missing_cube_vector_evidence；"
        f"{blocker_detail}需补采可精确关联到该 range、同时包含 CUBE 与 VECTOR "
        "完整事件的 SK child trace。"
    )


def _build_regression_diagnostics(
    decisions,
    child_summary,
    environment_evidence,
    candidate_config,
):
    hypotheses = []
    blockers = []
    pruned = sorted(
        (
            item
            for item in decisions
            if item.get("action") == "prune"
            or (
                item.get("classification") == "regressed"
                and performance_mapping_exact(
                    item.get("mapping_method"), item.get("mapping_confidence")
                )
            )
        ),
        key=lambda item: item["range_id"],
    )
    child_range_id = (
        _child_profile_range_id(pruned, child_summary) if child_summary else None
    )
    dcci_runtime = _dcci_runtime_evidence(environment_evidence, candidate_config)
    for decision in pruned:
        original = decision.get("original") or {}
        interval_p50 = (original.get("interval_us") or {}).get("p50")
        duration_sum_p50 = (original.get("duration_sum_us") or {}).get("p50")
        sk_p50 = decision.get("sk_duration_us")
        if (
            interval_p50 is not None
            and duration_sum_p50 is not None
            and sk_p50 is not None
            and sk_p50 > interval_p50
            and sk_p50 < duration_sum_p50
        ):
            hypotheses.append(
                {
                    "kind": "likely_lost_parallelism",
                    "confidence": "medium",
                    "range_id": decision["range_id"],
                    "evidence": [
                        "sk_interval_regressed",
                        "sk_duration_better_than_baseline_duration_sum",
                    ],
                    "explanation_zh": (
                        "SK P50 劣于融合前 interval P50，但优于融合前 duration sum "
                        "P50，提示并行重叠可能丢失；该证据不改变性能分类或裁剪动作。"
                    ),
                    "requires_ab_test": True,
                }
            )
        scalar_ratio = original.get("max_scalar_ratio")
        if scalar_ratio is None:
            scalar_ratio = (original.get("example_occurrence") or {}).get(
                "max_scalar_ratio"
            )
        if scalar_ratio is not None and scalar_ratio >= 0.2:
            if dcci_runtime["state"] == "enabled":
                hypotheses.append(
                    {
                        "kind": "dcci_boundary_overhead",
                        "confidence": "medium",
                        "range_id": decision["range_id"],
                        "evidence": [
                            f"{item['source']}={item['state']}"
                            for item in dcci_runtime["sources"]
                        ]
                        + ["baseline_max_scalar_ratio>=0.2"],
                        "explanation_zh": (
                            "运行证据确认 DCCI 已开启，且目标范围 scalar ratio "
                            "不低于 0.2；这支持边界 DCCI 开销假设，但仍需单变量 A/B 验证。"
                        ),
                        "requires_ab_test": True,
                    }
                )
            elif dcci_runtime["state"] == "conflict":
                source_summary = "，".join(
                    f"{item['source']}={item['state']}"
                    for item in dcci_runtime["sources"]
                )
                blockers.append(
                    "DCCI 运行状态冲突（dcci_state_conflict）："
                    f"{source_summary}；冲突消除前不生成 DCCI 假设或实验。"
                )
            elif dcci_runtime["state"] == "unknown":
                source_summary = (
                    "，".join(
                        f"{item['source']}={item['state']}"
                        for item in dcci_runtime["sources"]
                    )
                    or "未提供显式 runtime_evidence.dcci_state 来源"
                )
                blockers.append(
                    "DCCI 运行状态未知（dcci_state_unknown）："
                    f"{source_summary}；scalar/cache 指标不能代替明确的状态证据。"
                )
        baseline_overlap = (
            (original.get("example_occurrence") or {})
            .get("multi_stream_analysis", {})
            .get("cube_vector_parallel_detected")
        )
        if not baseline_overlap:
            continue
        if not child_summary or child_range_id != decision["range_id"]:
            _append_missing_cube_vector_trace(
                hypotheses, blockers, decision["range_id"]
            )
            continue
        if not _child_trace_has_cube_and_vector(child_summary):
            _append_missing_cube_vector_trace(
                hypotheses, blockers, decision["range_id"]
            )
            continue
        if not _child_trace_has_reliable_cube_vector_streams(child_summary):
            _append_missing_cube_vector_trace(
                hypotheses,
                blockers,
                decision["range_id"],
                detail_code="child_trace_stream_identity_incomplete",
                detail_zh="用于判定的 CUBE/VECTOR 完整事件缺少可靠 stream identity",
            )
            continue
        child_overlap = child_summary["multi_stream_analysis"].get(
            "cube_vector_parallel_detected"
        )
        if child_overlap is True:
            continue
        if child_overlap is not False:
            _append_missing_cube_vector_trace(
                hypotheses, blockers, decision["range_id"]
            )
            continue
        hypotheses.append(
            {
                "kind": "cube_vector_serialization",
                "confidence": "high",
                "range_id": decision["range_id"],
                "evidence": [
                    "baseline_cube_vector_overlap",
                    "sk_child_trace_serialized",
                ],
                "explanation_zh": (
                    "融合前存在 Cube/Vector 重叠，SK 子算子 trace 显示融合后串行。"
                ),
                "requires_ab_test": True,
            }
        )
    return {
        "diagnostic_hypotheses": sorted(
            hypotheses, key=lambda item: (item["range_id"], item["kind"])
        ),
        "recommended_experiments": [],
        "blockers": list(dict.fromkeys(blockers)),
    }


def _load_json_artifact(path, label):
    artifact = Path(path)
    try:
        value = json.loads(artifact.read_text(errors="replace"))
    except OSError as error:
        raise ValueError(f"{artifact}: cannot read {label}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{artifact}: invalid {label} JSON: {error.msg} at line "
            f"{error.lineno} column {error.colno}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"{artifact}: {label} must be a JSON object")
    return value


def _resolve_path(path):
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{path}: cannot resolve path: {error}") from error


def _relative_artifact_path(path, output_path):
    if path is None:
        return None
    relative = os.path.relpath(
        _resolve_path(path), _resolve_path(Path(output_path).parent)
    )
    return Path(relative).as_posix()


def _validate_output_target(path, label):
    if path is not None and path.exists() and path.is_dir():
        raise ValueError(f"{path}: {label} must not be a directory")


def _validate_analysis_paths(args):
    outputs = [args.json_out]
    if args.markdown_out is not None:
        outputs.append(args.markdown_out)
    resolved_outputs = [_resolve_path(path) for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("JSON and Markdown output paths must be distinct")
    input_names = (
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
        "source_root",
        *COLLECTION_MANIFEST_ARGUMENTS,
    )
    resolved_inputs = {
        _resolve_path(getattr(args, name))
        for name in input_names
        if getattr(args, name) is not None
    }
    for output in resolved_outputs:
        for input_path in resolved_inputs:
            if output == input_path or (
                input_path.is_dir() and input_path in output.parents
            ):
                raise ValueError("output path conflicts with an input artifact")


def _write_output_temp(path, content):
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
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_analysis_outputs(
    json_path, json_content, markdown_path=None, markdown_content=None
):
    _validate_output_target(json_path, "JSON output")
    _validate_output_target(markdown_path, "Markdown output")
    temporary_paths = []
    markdown_backup = None
    markdown_committed = False
    try:
        json_temp = _write_output_temp(json_path, json_content)
        temporary_paths.append(json_temp)
        markdown_temp = None
        if markdown_path is not None:
            markdown_temp = _write_output_temp(
                markdown_path,
                markdown_content or "",
            )
            temporary_paths.append(markdown_temp)
        if markdown_temp is not None:
            if markdown_path.exists():
                markdown_backup = _write_output_temp(
                    markdown_path, markdown_path.read_text()
                )
                temporary_paths.append(markdown_backup)
            os.replace(markdown_temp, markdown_path)
            markdown_committed = True
        os.replace(json_temp, json_path)
    except BaseException:
        if markdown_committed:
            if markdown_backup is not None:
                os.replace(markdown_backup, markdown_path)
            else:
                markdown_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _analysis_main(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline-profile", required=True, type=Path)
    parser.add_argument("--candidate-profile", required=True, type=Path)
    parser.add_argument("--sk-meta", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--baseline-workload", required=True, type=Path)
    parser.add_argument("--candidate-workload", required=True, type=Path)
    parser.add_argument("--declared-change-set", required=True, type=Path)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--analysis-agent-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--sk-prof", type=Path)
    parser.add_argument("--environment-evidence", type=Path)
    parser.add_argument("--source-scope-map", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--baseline-collection-manifest", type=Path)
    parser.add_argument("--profile-collection-manifest", type=Path)
    parser.add_argument("--min-relative-change-pct", type=float, default=3.0)
    parser.add_argument("--min-absolute-change-us", type=float, default=1.0)
    parser.add_argument("--min-occurrences", type=int, default=3)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)

    for option, value in (
        ("--candidate-name", args.candidate_name),
        ("--experiment-id", args.experiment_id),
        ("--round-id", args.round_id),
        ("--analysis-agent-id", args.analysis_agent_id),
        ("--source-revision", args.source_revision),
    ):
        if not value.strip() or value != value.strip():
            parser.error(f"{option} must be a non-empty trimmed string")
    if not round_belongs_to_candidate(args.candidate_name, args.round_id):
        parser.error(
            "--round-id must belong to --candidate-name and use AUTO/BASE/Pn/Rn/FINAL"
        )

    if args.min_occurrences < 1:
        parser.error("--min-occurrences must be at least 1")
    if (
        not math.isfinite(args.min_relative_change_pct)
        or args.min_relative_change_pct < 0
    ):
        parser.error("--min-relative-change-pct must be finite and non-negative")
    if (
        not math.isfinite(args.min_absolute_change_us)
        or args.min_absolute_change_us < 0
    ):
        parser.error("--min-absolute-change-us must be finite and non-negative")

    try:
        _validate_analysis_paths(args)
        _validate_output_target(args.json_out, "JSON output")
        _validate_output_target(args.markdown_out, "Markdown output")
        if not args.sk_meta.is_dir():
            raise ValueError(f"{args.sk_meta}: sk_meta must be an existing directory")
        baseline_config = _load_json_artifact(args.baseline_config, "baseline config")
        candidate_config = _load_json_artifact(
            args.candidate_config, "candidate config"
        )
        baseline_workload = _load_json_artifact(
            args.baseline_workload, "baseline workload"
        )
        candidate_workload = _load_json_artifact(
            args.candidate_workload, "candidate workload"
        )
        declared_change_set = _load_json_artifact(
            args.declared_change_set, "declared change set"
        )
        environment_evidence = (
            _load_json_artifact(args.environment_evidence, "environment evidence")
            if args.environment_evidence
            else None
        )
        layer_map = _load_layer_map(args.source_scope_map)
        fingerprints = validate_fingerprints(
            baseline_config,
            candidate_config,
            baseline_workload,
            candidate_workload,
            declared_change_set,
        )
        baseline_summary, baseline_rows = summarize_profile(
            args.baseline_profile, layer_map=layer_map
        )
        baseline_profile_fingerprint = _profile_fingerprint(args.baseline_profile)
        candidate_profile_fingerprint = _profile_fingerprint(args.candidate_profile)
        structural_context = _load_structural_context(args, baseline_rows, fingerprints)
        source_scope_context = (
            load_source_scope_map_v2(
                args.source_scope_map,
                expected_source_revision=args.source_revision,
                expected_candidate_manifest_sha256=(
                    file_sha256(args.profile_collection_manifest)
                    if args.profile_collection_manifest
                    else None
                ),
                expected_baseline_manifest_sha256=(
                    file_sha256(args.baseline_collection_manifest)
                    if args.baseline_collection_manifest
                    else None
                ),
                expected_source_root=args.source_root,
            )
            if args.source_scope_map
            else {
                "protocol": "source_scope_map_v2",
                "status": "not_requested",
                "lookup": {},
                "blockers": ["source_scope_map_missing"],
            }
        )
        if source_scope_context.get("status") == "exact" and (
            not args.profile_collection_manifest
            or not args.baseline_collection_manifest
            or not args.source_root
        ):
            raise ValueError(
                "source_scope_map_v2 exact requires current baseline/candidate manifests and --source-root"
            )
        comparison = compare_candidate(
            baseline_rows,
            args.candidate_name,
            args.candidate_profile,
            sk_meta=args.sk_meta,
            baseline_summary=baseline_summary,
            layer_map=layer_map,
            min_occurrences=args.min_occurrences,
            min_relative_change_pct=args.min_relative_change_pct,
            min_absolute_change_us=args.min_absolute_change_us,
            evidence_errors=fingerprints["evidence_errors"],
            structural_associations=structural_context["associations"],
            structural_blockers=structural_context["protocol"]["blockers"],
            source_scope_lookup=source_scope_context["lookup"],
        )
        decisions = comparison["superkernel_inventory"]
        if not decisions:
            raise ValueError("candidate profile contains no SuperKernel occurrences")
        sk_child_profile = (
            summarize_sk_child_profile(args.sk_prof) if args.sk_prof else None
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    inputs = {
        name: _relative_artifact_path(getattr(args, name), args.json_out)
        for name in (
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
            *COLLECTION_MANIFEST_ARGUMENTS,
        )
    }
    scope_actions = []
    for decision in decisions:
        identity = decision["identity"]
        boundary = copy.deepcopy(decision["boundary"]) or {}
        interval_proven = _source_interval_proven(boundary)
        scope_actions.append(
            {
                "sk_id": decision["sk_id"],
                "range_id": decision["range_id"],
                "classification": decision["classification"],
                "action": decision["action"],
                "source_scope": identity["source_scope"],
                "boundary": boundary,
                "ordered_child_op_sequence": copy.deepcopy(
                    identity["ordered_child_op_sequence"]
                ),
                "interval_unproven": not interval_proven,
                "status": "proposed",
            }
        )
    try:
        diagnostics = _build_regression_diagnostics(
            decisions,
            sk_child_profile,
            environment_evidence,
            candidate_config,
        )
    except ValueError as error:
        parser.error(str(error))
    identity = {
        "experiment_id": args.experiment_id,
        "round_id": args.round_id,
        "analysis_agent_id": args.analysis_agent_id,
        "candidate_name": args.candidate_name,
        "source_revision": args.source_revision,
    }
    report = {
        "schema_version": "1.2",
        "analysis_id": f"analysis-{_canonical_sha256(identity)}",
        **identity,
        "baseline_profile_fingerprint": baseline_profile_fingerprint,
        "candidate_profile_fingerprint": candidate_profile_fingerprint,
        "baseline_config_fingerprint": fingerprints["baseline_config_fingerprint"],
        "candidate_config_fingerprint": fingerprints["candidate_config_fingerprint"],
        "baseline_workload_fingerprint": fingerprints["baseline_workload_fingerprint"],
        "candidate_workload_fingerprint": fingerprints[
            "candidate_workload_fingerprint"
        ],
        "workload_fingerprint": fingerprints["workload_fingerprint"],
        "baseline_control_fingerprint": fingerprints["baseline_control_fingerprint"],
        "candidate_control_fingerprint": fingerprints["candidate_control_fingerprint"],
        "control_fingerprint": fingerprints["control_fingerprint"],
        "declared_change_set": fingerprints["declared_change_set"],
        "source_scope_mapping": {
            key: copy.deepcopy(source_scope_context.get(key))
            for key in (
                "protocol",
                "status",
                "blockers",
                "source_revision",
                "source_revision_role",
                "stable_marker_revision",
            )
            if key in source_scope_context
        },
        "inputs": inputs,
        "outputs": {
            "markdown": _relative_artifact_path(args.markdown_out, args.json_out)
        },
        "thresholds": {
            "min_relative_change_pct": args.min_relative_change_pct,
            "min_absolute_change_us": args.min_absolute_change_us,
            "min_occurrences": max(HARD_MIN_OCCURRENCES, args.min_occurrences),
        },
        "coverage": comparison["comparison_coverage"],
        "association_protocol": structural_context["protocol"],
        "candidate_binding_evidence": structural_context["candidate_binding_evidence"],
        "canonical_graph_fingerprints": structural_context[
            "canonical_graph_fingerprints"
        ],
        "stream_role_mapping": structural_context["stream_role_mapping"],
        "graph_alignment_proof": structural_context["graph_alignment_proof"],
        "mapping_coverage": _mapping_coverage(decisions),
        "per_sk_decisions": decisions,
        "scope_actions": scope_actions,
        "diagnostic_hypotheses": diagnostics["diagnostic_hypotheses"],
        "recommended_experiments": diagnostics["recommended_experiments"],
        "blockers": list(fingerprints["evidence_errors"])
        + (
            structural_context["protocol"]["blockers"]
            if structural_context["protocol"]["status"] != "not_requested"
            else []
        )
        + diagnostics["blockers"],
        "environment_evidence_loaded": environment_evidence is not None,
        "next_agent_guidance_zh": "仅按证据充分的逐 SK 判定处理本轮 scope。",
    }
    if sk_child_profile is not None:
        sk_child_profile["paths"] = [
            _relative_artifact_path(path, args.json_out)
            for path in sk_child_profile["paths"]
        ]
        report["sk_child_profile"] = sk_child_profile
    report["analysis_content_fingerprint"] = _canonical_sha256(report)

    try:
        output = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        parser.error(f"analysis result is not standard JSON: {error}")
    try:
        _write_analysis_outputs(
            args.json_out,
            output + "\n",
            markdown_path=args.markdown_out,
            markdown_content=(render_report(report) if args.markdown_out else None),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


def main(argv=None):
    return _analysis_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
