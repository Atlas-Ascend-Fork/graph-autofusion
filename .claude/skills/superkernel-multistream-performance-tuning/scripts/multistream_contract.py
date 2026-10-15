#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate isolated SuperKernel multi-stream tuning requests and results."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import multistream_execution
import multistream_component_reorder
import multistream_runner
import multistream_plan_compiler
import multistream_evidence
import multistream_operator_order
import multistream_trace_analysis


REQUEST_SCHEMA = "superkernel-multistream-request-v1"
RESULT_SCHEMA = "superkernel-multistream-result-v2"
RESULT_STATUSES = {"accepted", "no_gain", "blocked", "failed"}
TRIAL_DECISIONS = {"accepted", "rejected", "blocked", "failed"}
CHANGE_KINDS = {
    "option",
    "scope_split",
    "range_exclusion",
    multistream_execution.REORDER_CHANGE_KIND,
    multistream_component_reorder.CHANGE_KIND,
}
NET_EFFECTS = {"beneficial", "neutral", "regressed", "insufficient_evidence"}
PARALLELISM_EFFECTS = {"improved", "preserved", "degraded", "unknown"}
OPTIMIZATION_STATUSES = {"no_action", "opportunity", "validated", "blocked"}
FINGERPRINT_FIELDS = (
    "source_fingerprint",
    "config_fingerprint",
    "control_fingerprint",
    "workload_fingerprint",
)
REQUIRED_ARTIFACTS = (
    "baseline_collection_manifest",
    "candidate_collection_manifest",
    "projected_trace_mapping",
    "environment_evidence",
    "option_history",
)
ISOLATION_ROOTS = (
    "source_worktree",
    "experiment_root",
    "config_root",
    "cache_root",
)


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _analysis_fingerprint(value):
    payload = {
        key: item
        for key, item in value.items()
        if key != "analysis_content_fingerprint"
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _source_map_fingerprint(value):
    payload = json.loads(_canonical_json(value))
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("source_scope_map_content_fingerprint", None)
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _load_json(path):
    try:
        return json.loads(
            Path(path).read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        suffix = " a non-empty list" if nonempty else " a list"
        raise ValueError(f"{label} must be{suffix}")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _chinese_text(value, label):
    value = _text(value, label)
    if not any("\u4e00" <= char <= "\u9fff" for char in value):
        raise ValueError(f"{label} must contain Chinese text")
    return value


def _boolean(value, label):
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _integer(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def _json_values(value, label="root"):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_values(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string key")
            _json_values(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} contains an unsupported JSON value")


def _relative_path(value, label):
    value = _text(value, label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {".", ""}:
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _existing_artifact(root, value, label):
    relative = _relative_path(value, label)
    root = Path(root).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the request root") from error
    if not target.is_file():
        raise ValueError(f"{label} does not exist as a file: {relative}")
    return relative


def _existing_directory(root, value, label):
    relative = _relative_path(value, label)
    root = Path(root).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the request root") from error
    if not target.is_dir():
        raise ValueError(f"{label} does not exist as a directory: {relative}")
    return relative


def _enum(value, choices, label):
    value = _text(value, label)
    if value not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return value


def _fingerprint_identity(incumbent):
    return {
        "candidate_name": incumbent["candidate_name"],
        "source_revision": incumbent["source_revision"],
        **{field: incumbent[field] for field in FINGERPRINT_FIELDS},
    }


def _validate_analysis_binding(request, artifact_root, targets):
    relative = request["incumbent"]["profiling_analysis_result"]
    analysis = _load_json(Path(artifact_root) / relative)
    analysis = _mapping(analysis, "incumbent profiling analysis")
    if analysis.get("schema_version") != "1.2":
        raise ValueError("incumbent profiling analysis must use schema_version 1.2")
    fingerprint = _text(
        analysis.get("analysis_content_fingerprint"),
        "incumbent profiling analysis.analysis_content_fingerprint",
    )
    if fingerprint != _analysis_fingerprint(analysis):
        raise ValueError("incumbent profiling analysis content fingerprint mismatch")

    incumbent = request["incumbent"]
    bindings = (
        ("candidate_name", "candidate_name"),
        ("source_revision", "source_revision"),
        ("candidate_config_fingerprint", "config_fingerprint"),
        ("control_fingerprint", "control_fingerprint"),
        ("workload_fingerprint", "workload_fingerprint"),
    )
    for analysis_field, incumbent_field in bindings:
        if analysis.get(analysis_field) != incumbent[incumbent_field]:
            raise ValueError(
                f"incumbent profiling analysis.{analysis_field} does not match incumbent"
            )

    decisions = _list(
        analysis.get("per_sk_decisions"),
        "incumbent profiling analysis.per_sk_decisions",
        nonempty=True,
    )
    source_context = _mapping(
        analysis.get("source_scope_mapping", {}),
        "incumbent profiling analysis.source_scope_mapping",
    )
    validated = {}
    for target in targets:
        matches = [
            decision
            for decision in decisions
            if isinstance(decision, dict)
            and decision.get("range_id") == target["range_id"]
            and decision.get("graph_occurrence_fingerprint")
            == target["graph_occurrence_fingerprint"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"target {target['range_id']} must match exactly one profiling decision"
            )
        decision = matches[0]
        if decision.get("classification") != target["net_effect"]:
            raise ValueError(
                f"target {target['range_id']} net_effect differs from profiling analysis"
            )
        method = decision.get("mapping_method")
        confidence = decision.get("mapping_confidence")
        if (method, confidence) not in {
            ("kernel_projection_structural", "exact_projected_trace"),
            ("source_scope_map", "exact"),
        }:
            raise ValueError(
                f"target {target['range_id']} lacks exact projected occurrence mapping"
            )
        candidate_count = _integer(
            decision.get("candidate_occurrence_count"),
            f"profiling decision {target['range_id']}.candidate_occurrence_count",
            minimum=3,
        )
        original = _mapping(
            decision.get("original"),
            f"profiling decision {target['range_id']}.original",
        )
        interval = _mapping(
            original.get("interval_us"),
            f"profiling decision {target['range_id']}.original.interval_us",
        )
        baseline_count = _integer(
            interval.get("count"),
            f"profiling decision {target['range_id']}.original.interval_us.count",
            minimum=3,
        )
        validated[target["range_id"]] = {
            "graph_occurrence_fingerprint": target["graph_occurrence_fingerprint"],
            "mapping_method": method,
            "mapping_confidence": confidence,
            "candidate_occurrence_count": candidate_count,
            "baseline_occurrence_count": baseline_count,
            "boundary": decision.get("boundary"),
            "source_scope_mapping": source_context,
        }
    return validated


def validate_request(request, artifact_root):
    _json_values(request)
    request = _mapping(request, "request")
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(f"request.schema_version must be {REQUEST_SCHEMA}")
    request_id = _text(request.get("request_id"), "request.request_id")
    parent_id = _text(
        request.get("parent_experiment_id"), "request.parent_experiment_id"
    )

    incumbent = _mapping(request.get("incumbent"), "request.incumbent")
    _text(incumbent.get("candidate_name"), "request.incumbent.candidate_name")
    _text(incumbent.get("source_revision"), "request.incumbent.source_revision")
    for field in FINGERPRINT_FIELDS:
        _text(incumbent.get(field), f"request.incumbent.{field}")
    _existing_artifact(
        artifact_root,
        incumbent.get("clean_performance_summary"),
        "request.incumbent.clean_performance_summary",
    )
    _existing_artifact(
        artifact_root,
        incumbent.get("profiling_analysis_result"),
        "request.incumbent.profiling_analysis_result",
    )
    if (
        _integer(
            incumbent.get("clean_run_count"),
            "request.incumbent.clean_run_count",
            minimum=5,
            maximum=5,
        )
        != 5
    ):
        raise ValueError("request.incumbent.clean_run_count must be exactly 5")
    if incumbent.get("stable") is not True:
        raise ValueError("request.incumbent.stable must be true")

    artifacts = _mapping(request.get("artifacts"), "request.artifacts")
    for field in REQUIRED_ARTIFACTS:
        _existing_artifact(
            artifact_root, artifacts.get(field), f"request.artifacts.{field}"
        )
    for field in (
        "sk_prof",
        "source_scope_map",
        "operator_order_capture",
        "component_source_map",
        "component_order_capture",
    ):
        if artifacts.get(field) is not None:
            _existing_artifact(
                artifact_root, artifacts[field], f"request.artifacts.{field}"
            )

    target_ids = set()
    targets = _list(request.get("targets"), "request.targets", nonempty=True)
    for index, target in enumerate(targets):
        target = _mapping(target, f"request.targets[{index}]")
        range_id = _text(target.get("range_id"), f"request.targets[{index}].range_id")
        if range_id in target_ids:
            raise ValueError(f"request.targets has duplicate range_id: {range_id}")
        target_ids.add(range_id)
        _text(
            target.get("graph_occurrence_fingerprint"),
            f"request.targets[{index}].graph_occurrence_fingerprint",
        )
        _enum(
            target.get("net_effect"),
            NET_EFFECTS,
            f"request.targets[{index}].net_effect",
        )
        _enum(
            target.get("parallelism_effect"),
            PARALLELISM_EFFECTS,
            f"request.targets[{index}].parallelism_effect",
        )
        _enum(
            target.get("optimization_status"),
            OPTIMIZATION_STATUSES,
            f"request.targets[{index}].optimization_status",
        )

    analysis_targets = _validate_analysis_binding(request, artifact_root, targets)

    kinds = _list(
        request.get("requested_change_kinds"),
        "request.requested_change_kinds",
        nonempty=True,
    )
    normalized_kinds = []
    for index, kind in enumerate(kinds):
        normalized_kinds.append(
            _enum(kind, CHANGE_KINDS, f"request.requested_change_kinds[{index}]")
        )
    if len(normalized_kinds) != len(set(normalized_kinds)):
        raise ValueError("request.requested_change_kinds must be unique")
    ordinary_source_kinds = set(normalized_kinds) - {
        "option",
        multistream_component_reorder.CHANGE_KIND,
    }
    if ordinary_source_kinds and artifacts.get("source_scope_map") is None:
        raise ValueError("source-changing request requires artifacts.source_scope_map")
    if multistream_execution.REORDER_CHANGE_KIND in normalized_kinds:
        capture = artifacts.get("operator_order_capture")
        if capture is None:
            raise ValueError(
                "dependency_safe_operator_reorder requires artifacts.operator_order_capture"
            )
        for index, target in enumerate(targets):
            if (
                target.get("parallelism_effect") != "degraded"
                or target.get("optimization_status") != "opportunity"
            ):
                raise ValueError(
                    "business operator reorder is allowed only for degraded multistream "
                    f"opportunity targets; request.targets[{index}] is not eligible"
                )
        order_analysis = multistream_operator_order.analyze(
            Path(artifact_root) / capture,
            artifact_root,
            expected_request_fingerprint=content_fingerprint(request),
        )
        order_targets = {item["range_id"]: item for item in order_analysis["targets"]}
        if set(order_targets) != target_ids:
            raise ValueError(
                "operator order capture targets must exactly match reorder request targets"
            )
        for range_id in target_ids:
            order_target = order_targets.get(range_id)
            if (
                order_target is None
                or order_target.get("multistream_reorder_authorized") is not True
            ):
                raise ValueError(
                    "business operator reorder requires direct multistream evidence and a "
                    f"stable dispatch inversion for every target: {range_id}"
                )
            binding = analysis_targets[range_id]
            boundary = binding.get("boundary") or {}
            if (
                order_target["graph_occurrence_fingerprint"]
                != binding["graph_occurrence_fingerprint"]
                or (binding["mapping_method"], binding["mapping_confidence"])
                != ("source_scope_map", "exact")
                or boundary.get("source_file") != order_target["source_file"]
                or boundary.get("start_offset") != order_target["range_start_offset"]
                or boundary.get("end_offset") != order_target["range_end_offset"]
            ):
                raise ValueError(
                    f"operator order capture target does not match exact source binding: {range_id}"
                )

    isolation = _mapping(request.get("isolation"), "request.isolation")
    roots = []
    for field in ISOLATION_ROOTS:
        roots.append(_relative_path(isolation.get(field), f"request.isolation.{field}"))
    if len(roots) != len(set(roots)):
        raise ValueError("request isolation roots must be distinct")
    for field in (
        "immutable_incumbent",
        "dedicated_source_worktree",
        "dedicated_experiment_root",
        "dedicated_config_root",
        "dedicated_cache_namespace",
    ):
        if isolation.get(field) is not True:
            raise ValueError(f"request.isolation.{field} must be true")

    if multistream_component_reorder.CHANGE_KIND in normalized_kinds:
        map_relative = artifacts.get("component_source_map")
        capture_relative = artifacts.get("component_order_capture")
        immutable_source_relative = artifacts.get("component_immutable_source_root")
        if (
            map_relative is None
            or capture_relative is None
            or immutable_source_relative is None
        ):
            raise ValueError(
                "component_overlap_reorder requires component_source_map, "
                "component_order_capture, and component_immutable_source_root"
            )
        immutable_source_relative = _existing_directory(
            artifact_root,
            immutable_source_relative,
            "request.artifacts.component_immutable_source_root",
        )
        source_root = (Path(artifact_root) / immutable_source_relative).resolve()
        source_map = multistream_component_reorder.validate_source_map(
            _load_json(Path(artifact_root) / map_relative), source_root
        )
        capture = _load_json(Path(artifact_root) / capture_relative)
        multistream_component_reorder.validate_capture(
            capture, artifact_root, source_map
        )
        if capture["request_fingerprint"] != content_fingerprint(request):
            raise ValueError("component order capture request fingerprint mismatch")
        map_bindings = {
            item["range_id"]: item for item in source_map["target_bindings"]
        }
        if set(map_bindings) != target_ids:
            raise ValueError(
                "component source map targets must exactly match request targets"
            )
        for range_id, binding in map_bindings.items():
            analysis = analysis_targets[range_id]
            if (
                binding["graph_occurrence_fingerprint"]
                != analysis["graph_occurrence_fingerprint"]
            ):
                raise ValueError(
                    f"component source map occurrence identity differs for {range_id}"
                )

    authorization = _mapping(request.get("authorization"), "request.authorization")
    if authorization.get("run_inference") is not True:
        raise ValueError("request.authorization.run_inference must be true")
    if authorization.get("edit_isolated_worktree") is not True:
        raise ValueError("request.authorization.edit_isolated_worktree must be true")

    execution = _mapping(request.get("execution"), "request.execution")
    command = _list(
        execution.get("command_argv"), "request.execution.command_argv", nonempty=True
    )
    for index, argument in enumerate(command):
        _text(argument, f"request.execution.command_argv[{index}]")
    _text(
        execution.get("correctness_method"),
        "request.execution.correctness_method",
    )
    _integer(
        execution.get("expected_ranks"),
        "request.execution.expected_ranks",
        minimum=1,
    )
    _integer(execution.get("warmup"), "request.execution.warmup", minimum=0)

    budget = _mapping(request.get("budget"), "request.budget")
    _integer(budget.get("max_trials"), "request.budget.max_trials", minimum=1)
    _integer(
        budget.get("max_recollections_per_trial"),
        "request.budget.max_recollections_per_trial",
        minimum=0,
        maximum=3,
    )
    _integer(
        budget.get("timeout_seconds"),
        "request.budget.timeout_seconds",
        minimum=1,
    )

    return {
        "valid": True,
        "schema_version": REQUEST_SCHEMA,
        "request_id": request_id,
        "parent_experiment_id": parent_id,
        "request_fingerprint": content_fingerprint(request),
        "incumbent": _fingerprint_identity(incumbent),
        "target_range_ids": sorted(target_ids),
        "requested_change_kinds": sorted(normalized_kinds),
        "source_scope_map_available": artifacts.get("source_scope_map") is not None,
        "sk_prof_available": artifacts.get("sk_prof") is not None,
        "operator_order_capture_available": artifacts.get("operator_order_capture")
        is not None,
        "component_source_map_available": artifacts.get("component_source_map")
        is not None,
        "component_order_capture_available": artifacts.get("component_order_capture")
        is not None,
        "analysis_targets": analysis_targets,
    }


def _validate_option_change(change, root, label):
    expected = {"json_pointer", "before", "after", "accepted_evidence"}
    if set(change) != expected:
        raise ValueError(
            f"{label} option fields must be: {', '.join(sorted(expected))}"
        )
    pointer = _text(change.get("json_pointer"), f"{label}.json_pointer")
    if not pointer.startswith("/") or pointer == "/":
        raise ValueError(f"{label}.json_pointer must be a non-root RFC6901 pointer")
    if _canonical_json(change.get("before")) == _canonical_json(change.get("after")):
        raise ValueError(f"{label} before and after must differ")
    _existing_artifact(
        root, change.get("accepted_evidence"), f"{label}.accepted_evidence"
    )


def _validate_source_change(change, request, analysis_target, result_root, label):
    expected = {
        "source_scope_map",
        "source_file",
        "start_offset",
        "end_offset",
        "boundary_change",
    }
    if set(change) != expected:
        raise ValueError(
            f"{label} source fields must be: {', '.join(sorted(expected))}"
        )
    source_map = request["artifacts"].get("source_scope_map")
    if source_map is None:
        raise ValueError(f"{label} requires request.artifacts.source_scope_map")
    if change.get("source_scope_map") != source_map:
        raise ValueError(f"{label}.source_scope_map must match the request")
    source_map_value = _mapping(
        _load_json(Path(result_root) / source_map), f"{label}.source_scope_map"
    )
    if source_map_value.get(
        "protocol"
    ) != "source_scope_map_v2" or source_map_value.get("schema_version") not in {
        "2.0",
        "2.1",
    }:
        raise ValueError(f"{label} requires a source_scope_map_v2 schema 2.0 or 2.1")
    provenance = _mapping(
        source_map_value.get("provenance"), f"{label}.source_scope_map.provenance"
    )
    fingerprint = _text(
        provenance.get("source_scope_map_content_fingerprint"),
        f"{label}.source_scope_map.provenance.source_scope_map_content_fingerprint",
    )
    if fingerprint != _source_map_fingerprint(source_map_value):
        raise ValueError(f"{label} source_scope_map_v2 content fingerprint mismatch")
    source_revision = provenance.get(
        "source_revision", provenance.get("stable_marker_revision")
    )
    if source_revision != request["incumbent"]["source_revision"]:
        raise ValueError(f"{label} source_scope_map_v2 source revision mismatch")
    source_file = _relative_path(change.get("source_file"), f"{label}.source_file")
    start = _integer(change.get("start_offset"), f"{label}.start_offset", minimum=0)
    end = _integer(change.get("end_offset"), f"{label}.end_offset", minimum=1)
    if end <= start:
        raise ValueError(f"{label}.end_offset must be greater than start_offset")
    _text(change.get("boundary_change"), f"{label}.boundary_change")
    source_context = analysis_target["source_scope_mapping"]
    if (
        source_context.get("protocol") != "source_scope_map_v2"
        or source_context.get("status") != "exact"
    ):
        raise ValueError(
            f"{label} requires analyzer-validated source_scope_map_v2 exact"
        )
    if (
        analysis_target["mapping_method"],
        analysis_target["mapping_confidence"],
    ) != ("source_scope_map", "exact"):
        raise ValueError(f"{label} target is not source_scope_map + exact")
    boundary = _mapping(analysis_target.get("boundary"), f"{label}.analysis_boundary")
    expected = {
        "source_file": source_file,
        "start_offset": start,
        "end_offset": end,
    }
    for field, value in expected.items():
        if boundary.get(field) != value:
            raise ValueError(
                f"{label}.{field} differs from analyzer-validated boundary"
            )
    source_ranges = _list(
        source_map_value.get("source_ranges"),
        f"{label}.source_scope_map.source_ranges",
        nonempty=True,
    )
    map_matches = [
        item
        for item in source_ranges
        if isinstance(item, dict)
        and item.get("sk_occurrence_fingerprint")
        == analysis_target["graph_occurrence_fingerprint"]
    ]
    if len(map_matches) != 1 or map_matches[0].get("relation") != "exact_cover":
        raise ValueError(f"{label} target lacks one source_scope_map_v2 exact_cover")
    map_boundary = _mapping(
        map_matches[0].get("boundary"), f"{label}.source_scope_map.boundary"
    )
    for field, value in expected.items():
        if map_boundary.get(field) != value:
            raise ValueError(
                f"{label}.{field} differs from source_scope_map_v2 boundary"
            )


def _validate_reorder_change(change, request, analysis_target, result_root, label):
    required = {
        "source_scope_map",
        "operator_order_analysis",
        "operator_order_analysis_fingerprint",
        "source_file",
        "range_id",
        "start_offset",
        "end_offset",
        "before_order",
        "after_order",
        "route",
        "hard_dependencies",
    }
    if set(change) != required:
        raise ValueError(
            f"{label} reorder fields must be: {', '.join(sorted(required))}"
        )
    _validate_source_change(
        {
            "source_scope_map": change["source_scope_map"],
            "source_file": change["source_file"],
            "start_offset": change["start_offset"],
            "end_offset": change["end_offset"],
            "boundary_change": "dependency-safe business operator reorder",
        },
        request,
        analysis_target,
        result_root,
        label,
    )
    if change["range_id"] not in {item["range_id"] for item in request["targets"]}:
        raise ValueError(f"{label}.range_id is not a request target")
    before = _list(change["before_order"], f"{label}.before_order", nonempty=True)
    after = _list(change["after_order"], f"{label}.after_order", nonempty=True)
    for index, item in enumerate(before):
        _text(item, f"{label}.before_order[{index}]")
    for index, item in enumerate(after):
        _text(item, f"{label}.after_order[{index}]")
    if len(before) != len(set(before)) or set(before) != set(after) or before == after:
        raise ValueError(f"{label} before/after order must be distinct permutations")
    _enum(change["route"], {"route2", "route3"}, f"{label}.route")
    dependencies = _list(change["hard_dependencies"], f"{label}.hard_dependencies")
    for index, edge in enumerate(dependencies):
        edge_label = f"{label}.hard_dependencies[{index}]"
        if not isinstance(edge, dict) or set(edge) != {"before", "after", "kind"}:
            raise ValueError(f"{edge_label} fields are invalid")
    analysis_path = _existing_artifact(
        result_root,
        change["operator_order_analysis"],
        f"{label}.operator_order_analysis",
    )
    validation = multistream_operator_order.validate_analysis(
        Path(result_root) / analysis_path,
        result_root,
        expected_request_fingerprint=content_fingerprint(request),
    )
    if (
        change["operator_order_analysis_fingerprint"]
        != validation["analysis_fingerprint"]
    ):
        raise ValueError(f"{label}.operator_order_analysis_fingerprint mismatch")
    analysis = _load_json(Path(result_root) / analysis_path)
    matches = [
        item for item in analysis["targets"] if item["range_id"] == change["range_id"]
    ]
    if (
        len(matches) != 1
        or matches[0].get("multistream_reorder_authorized") is not True
    ):
        raise ValueError(f"{label} lacks direct multistream reorder authorization")
    target = matches[0]
    legal = [target["route2_order"], *target["route3_orders"]]
    if before != target["current_source_order"] or after not in legal:
        raise ValueError(
            f"{label} order differs from analyzer-generated legal candidates"
        )
    if dependencies != target["hard_dependencies"]:
        raise ValueError(f"{label}.hard_dependencies differs from order analysis")


def _validate_component_change(change, request, result_root, label):
    required = {
        "source_file",
        "target_set_id",
        "target_range_ids",
        "span_ids",
        "before_order",
        "after_order",
        "hard_dependencies",
    }
    if not isinstance(change, dict) or set(change) != required:
        raise ValueError(
            f"{label} component reorder fields must be: {', '.join(sorted(required))}"
        )
    _relative_path(change["source_file"], f"{label}.source_file")
    _text(change["target_set_id"], f"{label}.target_set_id")
    if change["target_range_ids"] != sorted(
        item["range_id"] for item in request["targets"]
    ):
        raise ValueError(f"{label}.target_range_ids differ from request targets")
    span_ids = _list(change["span_ids"], f"{label}.span_ids", nonempty=True)
    if len(span_ids) < 2 or len(span_ids) != len(set(span_ids)):
        raise ValueError(f"{label}.span_ids require distinct multi-hunk spans")
    for index, span_id in enumerate(span_ids):
        _text(span_id, f"{label}.span_ids[{index}]")
    before = _list(change["before_order"], f"{label}.before_order", nonempty=True)
    after = _list(change["after_order"], f"{label}.after_order", nonempty=True)
    if len(before) != 2 or set(before) != set(after) or before == after:
        raise ValueError(f"{label} before/after must reverse the component pair")
    if not isinstance(change["hard_dependencies"], list):
        raise ValueError(f"{label}.hard_dependencies must be a list")
    artifacts = request["artifacts"]
    immutable_source_relative = _existing_directory(
        result_root,
        request["artifacts"].get("component_immutable_source_root"),
        "request.artifacts.component_immutable_source_root",
    )
    source_root = (Path(result_root) / immutable_source_relative).resolve()
    source_map = multistream_component_reorder.validate_source_map(
        _load_json(Path(result_root) / artifacts["component_source_map"]), source_root
    )
    capture = _load_json(Path(result_root) / artifacts["component_order_capture"])
    multistream_component_reorder.validate_capture(capture, result_root, source_map)
    policy = capture["component_policy"]
    if (
        change["target_set_id"] != source_map["target_set_id"]
        or change["before_order"] != policy["before_order"]
        or change["after_order"] != policy["after_order"]
        or change["hard_dependencies"] != source_map["hard_dependencies"]
    ):
        raise ValueError(f"{label} differs from the sealed component map/capture")


def _validate_trial(trial, request, result_root, target_ids, analysis_targets, index):
    label = f"result.trials[{index}]"
    trial = _mapping(trial, label)
    trial_id = _text(trial.get("trial_id"), f"{label}.trial_id")
    target = _text(trial.get("target_range_id"), f"{label}.target_range_id")
    if target not in target_ids:
        raise ValueError(f"{label}.target_range_id is not declared by the request")
    kind = _enum(trial.get("change_kind"), CHANGE_KINDS, f"{label}.change_kind")
    if kind not in request["requested_change_kinds"]:
        raise ValueError(f"{label}.change_kind was not requested")
    decision = _enum(trial.get("decision"), TRIAL_DECISIONS, f"{label}.decision")
    change = _mapping(trial.get("only_change"), f"{label}.only_change")
    if kind == "option":
        _validate_option_change(change, result_root, f"{label}.only_change")
    elif kind == multistream_execution.REORDER_CHANGE_KIND:
        _validate_reorder_change(
            change,
            request,
            analysis_targets[target],
            result_root,
            f"{label}.only_change",
        )
    elif kind == multistream_component_reorder.CHANGE_KIND:
        _validate_component_change(change, request, result_root, f"{label}.only_change")
    else:
        _validate_source_change(
            change,
            request,
            analysis_targets[target],
            result_root,
            f"{label}.only_change",
        )

    action_path = _existing_artifact(
        result_root, trial.get("action_manifest"), f"{label}.action_manifest"
    )
    action = _mapping(
        _load_json(Path(result_root) / action_path), f"{label}.action_manifest"
    )
    expected_action_schema = (
        multistream_execution.ACTION_SCHEMA_V2
        if kind == multistream_component_reorder.CHANGE_KIND
        else multistream_execution.ACTION_SCHEMA
    )
    if action.get("schema_version") != expected_action_schema:
        raise ValueError(f"{label}.action_manifest must use {expected_action_schema}")
    if action.get("trial_id") != trial_id or action.get("change_kind") != kind:
        raise ValueError(f"{label}.action_manifest identity mismatch")
    if action.get("single_change_verified") is not True:
        raise ValueError(f"{label}.action_manifest must prove one isolated change")
    action_change = _mapping(
        action.get("only_change"), f"{label}.action_manifest.only_change"
    )
    if kind == "option":
        expected_action = {
            field: change[field] for field in ("json_pointer", "before", "after")
        }
        if _canonical_json(action_change) != _canonical_json(expected_action):
            raise ValueError(f"{label}.action_manifest differs from only_change")
        if action.get("changed_pointers") != [change["json_pointer"]]:
            raise ValueError(f"{label}.action_manifest changed_pointers mismatch")
    elif kind == multistream_execution.REORDER_CHANGE_KIND:
        expected_action = {
            field: change[field]
            for field in (
                "source_file",
                "range_id",
                "start_offset",
                "end_offset",
                "before_order",
                "after_order",
                "route",
                "hard_dependencies",
            )
        }
        if _canonical_json(action_change) != _canonical_json(expected_action):
            raise ValueError(
                f"{label}.action_manifest differs from reorder only_change"
            )
        if action.get("multistream_only_verified") is not True:
            raise ValueError(
                f"{label}.action_manifest lacks multistream-only verification"
            )
        if (
            action.get("resource_pair_policy")
            != multistream_operator_order.RESOURCE_PAIR_POLICY
        ):
            raise ValueError(f"{label}.action_manifest resource pair policy mismatch")
        order_analysis = _load_json(
            Path(result_root) / change["operator_order_analysis"]
        )
        order_target = next(
            item
            for item in order_analysis["targets"]
            if item["range_id"] == change["range_id"]
        )
        if (
            not order_target["stable_resource_complementary_pairs"]
            or action.get("stable_resource_complementary_pairs")
            != order_target["stable_resource_complementary_pairs"]
        ):
            raise ValueError(
                f"{label}.action_manifest lacks bound Cube/Vector resource evidence"
            )
        if action.get("operator_order_analysis") != change["operator_order_analysis"]:
            raise ValueError(f"{label}.action_manifest order analysis path mismatch")
        if (
            action.get("operator_order_analysis_fingerprint")
            != change["operator_order_analysis_fingerprint"]
        ):
            raise ValueError(
                f"{label}.action_manifest order analysis fingerprint mismatch"
            )
        if action.get("source_adapter_validation") != "passed":
            raise ValueError(f"{label}.action_manifest source validation must pass")
        prerequisite = action.get("route3_prerequisite")
        if change["route"] == "route2":
            if prerequisite is not None:
                raise ValueError(
                    f"{label}.action_manifest route2 cannot cite route3 prerequisite"
                )
        else:
            required = {
                "route2_trial_id",
                "route2_dispatch_evidence",
                "route2_dispatch_evidence_fingerprint",
                "route2_dispatch_occurrence_count",
                "route2_clean3_evidence",
                "route2_clean3_evidence_fingerprint",
                "route2_clean3_decision",
            }
            if not isinstance(prerequisite, dict) or set(prerequisite) != required:
                raise ValueError(
                    f"{label}.action_manifest route3 prerequisite is invalid"
                )
            dispatch_path = _existing_artifact(
                result_root,
                prerequisite["route2_dispatch_evidence"],
                f"{label}.route3_prerequisite.route2_dispatch_evidence",
            )
            dispatch = multistream_operator_order.validate_dispatch_evidence(
                Path(result_root) / dispatch_path,
                result_root,
                trial_id=prerequisite["route2_trial_id"],
                request_fingerprint=content_fingerprint(request),
            )
            dispatch_value = _load_json(Path(result_root) / dispatch_path)
            route2_action = _load_json(
                Path(result_root) / dispatch_value["action_manifest"]
            )
            route2_change = route2_action.get("only_change", {})
            if (
                prerequisite["route2_dispatch_evidence_fingerprint"]
                != dispatch_value["evidence_fingerprint"]
                or prerequisite["route2_dispatch_occurrence_count"]
                != dispatch["aligned_occurrence_count"]
                or route2_action.get("operator_order_analysis_fingerprint")
                != change["operator_order_analysis_fingerprint"]
                or route2_change.get("range_id") != change["range_id"]
                or route2_change.get("route") != "route2"
                or route2_change.get("after_order") != order_target["route2_order"]
            ):
                raise ValueError(
                    f"{label}.action_manifest route2 dispatch binding mismatch"
                )
            clean_path = _existing_artifact(
                result_root,
                prerequisite["route2_clean3_evidence"],
                f"{label}.route3_prerequisite.route2_clean3_evidence",
            )
            clean = multistream_operator_order.validate_route2_no_gain_evidence(
                Path(result_root) / clean_path,
                result_root,
                trial_id=prerequisite["route2_trial_id"],
                request_fingerprint=content_fingerprint(request),
            )
            if (
                clean["decision"] != "reject"
                or prerequisite["route2_clean3_decision"] != "reject"
                or prerequisite["route2_clean3_evidence_fingerprint"]
                != clean["evidence_fingerprint"]
            ):
                raise ValueError(
                    f"{label}.action_manifest route3 lacks rejected route2 clean3"
                )
    elif kind == multistream_component_reorder.CHANGE_KIND:
        if _canonical_json(action_change) != _canonical_json(change):
            raise ValueError(
                f"{label}.action_manifest differs from component only_change"
            )
        if (
            action.get("component_aware_verified") is not True
            or action.get("multistream_only_verified") is not True
            or action.get("source_adapter_validation") != "passed"
        ):
            raise ValueError(
                f"{label}.action_manifest lacks component-aware verification"
            )
        artifacts = request["artifacts"]
        source_map = _load_json(Path(result_root) / artifacts["component_source_map"])
        capture = _load_json(Path(result_root) / artifacts["component_order_capture"])
        if action.get("component_source_map_fingerprint") != source_map.get(
            "mapping_fingerprint"
        ) or action.get("component_capture_fingerprint") != capture.get(
            "capture_fingerprint"
        ):
            raise ValueError(
                f"{label}.action_manifest component evidence binding mismatch"
            )
    else:
        expected_action = {
            field: change[field]
            for field in (
                "source_file",
                "start_offset",
                "end_offset",
                "boundary_change",
            )
        }
        if _canonical_json(action_change) != _canonical_json(expected_action):
            raise ValueError(f"{label}.action_manifest differs from only_change")
        if action.get("source_adapter_validation") != "passed":
            raise ValueError(
                f"{label}.action_manifest source_adapter_validation must be passed"
            )

    state_path = _existing_artifact(
        result_root, trial.get("execution_state"), f"{label}.execution_state"
    )
    state = _mapping(
        _load_json(Path(result_root) / state_path), f"{label}.execution_state"
    )
    expected_terminal = decision
    multistream_execution.validate_state(
        state,
        trial_id=trial_id,
        request_fingerprint=content_fingerprint(request),
        require_accepted=decision == "accepted",
        artifact_root=result_root,
    )
    if state.get("state") != expected_terminal:
        raise ValueError(
            f"{label}.execution_state must end in the trial decision {expected_terminal}"
        )

    plan_path = _existing_artifact(
        result_root, trial.get("execution_plan"), f"{label}.execution_plan"
    )
    plan = multistream_runner.validate_plan(
        _mapping(_load_json(Path(result_root) / plan_path), f"{label}.execution_plan")
    )
    if plan["trial_id"] != trial_id:
        raise ValueError(f"{label}.execution_plan trial_id mismatch")
    if plan["request_fingerprint"] != content_fingerprint(request):
        raise ValueError(f"{label}.execution_plan request_fingerprint mismatch")
    if plan["artifact_root"] != str(Path(result_root).resolve()):
        raise ValueError(f"{label}.execution_plan artifact_root mismatch")
    adapter_path = _existing_artifact(
        result_root, trial.get("model_adapter"), f"{label}.model_adapter"
    )
    adapter = _mapping(
        _load_json(Path(result_root) / adapter_path), f"{label}.model_adapter"
    )
    compilation_path = _existing_artifact(
        result_root, trial.get("plan_compilation"), f"{label}.plan_compilation"
    )
    compilation = _mapping(
        _load_json(Path(result_root) / compilation_path), f"{label}.plan_compilation"
    )
    multistream_plan_compiler.validate_compilation(
        compilation, plan, adapter, action, result_root, request=request
    )

    events = {event["state"]: event.get("evidence") for event in state["events"]}
    if events.get("diff_verified") != action_path:
        raise ValueError(
            f"{label}.execution_state diff_verified must cite action_manifest"
        )
    if decision == "accepted":
        for phase in plan["phases"]:
            evidence = events.get(phase["state_after"])
            if evidence != phase["manifest"]:
                raise ValueError(
                    f"{label}.execution_state {phase['state_after']} must cite its phase manifest"
                )
            phase_manifest = _mapping(
                _load_json(Path(result_root) / evidence),
                f"{label}.execution_plan.{phase['phase_id']}.manifest",
            )
            multistream_runner.validate_phase_manifest(
                phase_manifest, plan, phase, result_root, require_passed=True
            )
            validator_options = multistream_plan_compiler._validator_options(
                phase["validator_argv"], phase["state_after"]
            )
            semantic_evidence = multistream_evidence.validate_evidence(
                validator_options["--out"],
                result_root,
                trial_id=trial_id,
                request_fingerprint=content_fingerprint(request),
                state_after=phase["state_after"],
            )
            if semantic_evidence["decision"] != "pass":
                raise ValueError(
                    f"{label}.{phase['state_after']} semantic evidence must pass"
                )

    gates = _mapping(trial.get("gates"), f"{label}.gates")
    required_gates = {"correctness", "profiling", "clean_performance"}
    if set(gates) != required_gates:
        raise ValueError(
            f"{label}.gates fields must be: {', '.join(sorted(required_gates))}"
        )
    for gate in ("correctness", "profiling", "clean_performance"):
        _enum(
            gates.get(gate),
            {"passed", "failed", "blocked", "not_run"},
            f"{label}.gates.{gate}",
        )
    if kind in {
        multistream_execution.REORDER_CHANGE_KIND,
        multistream_component_reorder.CHANGE_KIND,
    } and (gates["profiling"] != "not_run" or decision == "accepted"):
        dispatch_path = _existing_artifact(
            result_root,
            trial.get("dispatch_order_evidence"),
            f"{label}.dispatch_order_evidence",
        )
        validator = (
            multistream_component_reorder.validate_dispatch_evidence
            if kind == multistream_component_reorder.CHANGE_KIND
            else multistream_operator_order.validate_dispatch_evidence
        )
        dispatch = validator(
            Path(result_root) / dispatch_path,
            result_root,
            trial_id=trial_id,
            request_fingerprint=content_fingerprint(request),
        )
        if dispatch["decision"] != "pass":
            raise ValueError(
                f"{label}.dispatch_order_evidence must pass before profiling"
            )
    metrics = _mapping(trial.get("clean_metrics"), f"{label}.clean_metrics")
    run_count = _integer(
        metrics.get("run_count"), f"{label}.clean_metrics.run_count", minimum=0
    )
    for field in (
        "stable",
        "incremental_mean_gain",
        "positive_median_direction",
        "p90_non_regression",
        "stddev_non_regression",
    ):
        _boolean(metrics.get(field), f"{label}.clean_metrics.{field}")
    mechanism_status = _enum(
        trial.get("mechanism_status"),
        {"validated", "unproven", "not_observed", "blocked"},
        f"{label}.mechanism_status",
    )
    scheduling = trial.get("scheduling_evidence")
    trace_finding = None
    if mechanism_status == "validated":
        scheduling = _mapping(scheduling, f"{label}.scheduling_evidence")
        required = {
            "trace_artifact",
            "binding_artifact",
            "overflow_detected",
            "aligned_occurrence_count",
        }
        if set(scheduling) != required:
            raise ValueError(
                f"{label}.scheduling_evidence fields must be: "
                f"{', '.join(sorted(required))}"
            )
        trace_path = _existing_artifact(
            result_root,
            scheduling.get("trace_artifact"),
            f"{label}.scheduling_evidence.trace_artifact",
        )
        binding_path = _existing_artifact(
            result_root,
            scheduling.get("binding_artifact"),
            f"{label}.scheduling_evidence.binding_artifact",
        )
        if scheduling.get("overflow_detected") is not False:
            raise ValueError(
                f"{label} validated mechanism requires overflow_detected=false"
            )
        declared_count = _integer(
            scheduling.get("aligned_occurrence_count"),
            f"{label}.scheduling_evidence.aligned_occurrence_count",
            minimum=3,
        )
        trace_analysis = multistream_trace_analysis.validate_bound_analysis(
            Path(result_root) / binding_path,
            result_root,
            expected_request_fingerprint=content_fingerprint(request),
            expected_trial_id=trial_id,
        )
        capture = trace_analysis["capture"]
        if capture.get("path") != trace_path:
            raise ValueError(
                f"{label} trace_artifact differs from trace analysis capture"
            )
        if capture.get("overflow_detected") is not False or trace_analysis.get(
            "blockers"
        ):
            raise ValueError(
                f"{label} validated mechanism requires complete unblocked trace"
            )
        matches = [
            item for item in trace_analysis["targets"] if item.get("range_id") == target
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{label} trace analysis must contain exactly one target binding"
            )
        trace_finding = matches[0]
        if trace_finding.get("parallelism_effect") != "improved":
            raise ValueError(
                f"{label} validated mechanism requires improved parallelism evidence"
            )
        if trace_finding.get("aligned_occurrence_count") != declared_count:
            raise ValueError(
                f"{label} aligned occurrence count differs from trace analysis"
            )
    elif scheduling is not None:
        raise ValueError(
            f"{label}.scheduling_evidence is allowed only for a validated mechanism"
        )
    if decision == "accepted":
        if any(gates[gate] != "passed" for gate in required_gates):
            raise ValueError(f"{label} accepted trial requires every gate passed")
        if run_count != 5:
            raise ValueError(f"{label} accepted trial requires exactly five clean runs")
        for field in (
            "stable",
            "incremental_mean_gain",
            "positive_median_direction",
            "p90_non_regression",
            "stddev_non_regression",
        ):
            if metrics[field] is not True:
                raise ValueError(f"{label} accepted trial requires {field}=true")
    return trial_id, decision, action, trace_finding


def validate_result(request, result, artifact_root):
    request_summary = validate_request(request, artifact_root)
    _json_values(result)
    result = _mapping(result, "result")
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError(f"result.schema_version must be {RESULT_SCHEMA}")
    if result.get("request_id") != request["request_id"]:
        raise ValueError("result.request_id must match the request")
    if result.get("parent_experiment_id") != request["parent_experiment_id"]:
        raise ValueError("result.parent_experiment_id must match the request")
    if result.get("request_fingerprint") != request_summary["request_fingerprint"]:
        raise ValueError("result.request_fingerprint does not match request content")
    status = _enum(result.get("status"), RESULT_STATUSES, "result.status")
    unchanged = _boolean(
        result.get("incumbent_unchanged"), "result.incumbent_unchanged"
    )

    blockers = _list(result.get("blockers"), "result.blockers")
    for index, blocker in enumerate(blockers):
        _text(blocker, f"result.blockers[{index}]")
    if status in {"blocked", "failed"} and not blockers:
        raise ValueError(f"result status {status} requires at least one blocker")

    findings = _list(result.get("multistream_findings"), "result.multistream_findings")
    for index, finding in enumerate(findings):
        finding = _mapping(finding, f"result.multistream_findings[{index}]")
        _text(finding.get("range_id"), f"result.multistream_findings[{index}].range_id")
        _enum(
            finding.get("net_effect"),
            NET_EFFECTS,
            f"result.multistream_findings[{index}].net_effect",
        )
        _enum(
            finding.get("parallelism_effect"),
            PARALLELISM_EFFECTS,
            f"result.multistream_findings[{index}].parallelism_effect",
        )
        _enum(
            finding.get("optimization_status"),
            OPTIMIZATION_STATUSES,
            f"result.multistream_findings[{index}].optimization_status",
        )

    target_ids = {target["range_id"] for target in request["targets"]}
    trial_ids = set()
    accepted_trial_ids = []
    actions_by_trial = {}
    trace_findings_by_trial = {}
    trials = _list(result.get("trials"), "result.trials")
    for index, trial in enumerate(trials):
        trial_id, decision, action, trace_finding = _validate_trial(
            trial,
            request,
            artifact_root,
            target_ids,
            request_summary["analysis_targets"],
            index,
        )
        if trial_id in trial_ids:
            raise ValueError(f"result.trials has duplicate trial_id: {trial_id}")
        trial_ids.add(trial_id)
        actions_by_trial[trial_id] = action
        trace_findings_by_trial[trial_id] = trace_finding
        if decision == "accepted":
            accepted_trial_ids.append(trial_id)
    if len(trials) > request["budget"]["max_trials"]:
        raise ValueError("result.trials exceeds request.budget.max_trials")

    fallback = _mapping(result.get("fallback"), "result.fallback")
    expected_fallback = _fingerprint_identity(request["incumbent"])
    for field, expected in expected_fallback.items():
        if fallback.get(field) != expected:
            raise ValueError(f"result.fallback.{field} must match the incumbent")
    _chinese_text(fallback.get("reason_zh"), "result.fallback.reason_zh")

    selected = result.get("selected_candidate")
    if status == "accepted":
        if unchanged is not False:
            raise ValueError("accepted result requires incumbent_unchanged=false")
        if len(accepted_trial_ids) != 1:
            raise ValueError("accepted result requires exactly one accepted trial")
        selected = _mapping(selected, "result.selected_candidate")
        if selected.get("trial_id") != accepted_trial_ids[0]:
            raise ValueError(
                "selected_candidate.trial_id must identify the accepted trial"
            )
        _text(
            selected.get("candidate_name"), "result.selected_candidate.candidate_name"
        )
        _text(
            selected.get("derived_experiment_id"),
            "result.selected_candidate.derived_experiment_id",
        )
        _text(
            selected.get("source_revision"), "result.selected_candidate.source_revision"
        )
        for field in FINGERPRINT_FIELDS:
            _text(selected.get(field), f"result.selected_candidate.{field}")
        for field in (
            "config_manifest",
            "config_snapshot",
            "source_manifest",
            "profiling_analysis_result",
            "clean_performance_summary",
        ):
            _existing_artifact(
                artifact_root,
                selected.get(field),
                f"result.selected_candidate.{field}",
            )
        if all(
            selected[field] == request["incumbent"][field]
            for field in ("source_fingerprint", "config_fingerprint")
        ):
            raise ValueError(
                "accepted selected candidate must change source or config fingerprint"
            )
        if (
            selected["control_fingerprint"]
            != request["incumbent"]["control_fingerprint"]
        ):
            raise ValueError(
                "accepted selected candidate must preserve control_fingerprint"
            )
        if (
            selected["workload_fingerprint"]
            != request["incumbent"]["workload_fingerprint"]
        ):
            raise ValueError(
                "accepted selected candidate must preserve workload_fingerprint"
            )
        accepted_trial = next(
            trial for trial in trials if trial["trial_id"] == accepted_trial_ids[0]
        )
        accepted_trace_finding = trace_findings_by_trial[accepted_trial_ids[0]]
        if accepted_trace_finding is not None:
            matching_findings = [
                item
                for item in findings
                if item.get("range_id") == accepted_trial["target_range_id"]
            ]
            if len(matching_findings) != 1:
                raise ValueError(
                    "validated mechanism requires one matching multistream finding"
                )
            finding = matching_findings[0]
            if (
                finding.get("parallelism_effect")
                != accepted_trace_finding["parallelism_effect"]
                or finding.get("optimization_status") != "validated"
            ):
                raise ValueError(
                    "validated mechanism finding differs from bound trace analysis"
                )
        accepted_action = actions_by_trial[accepted_trial_ids[0]]
        config_snapshot = Path(artifact_root) / selected["config_snapshot"]
        config_fingerprint = multistream_execution.content_fingerprint(
            multistream_execution._load_structured(config_snapshot)
        )
        if selected["config_fingerprint"] != config_fingerprint:
            raise ValueError(
                "selected_candidate.config_fingerprint differs from config_snapshot"
            )
        source_manifest = _mapping(
            _load_json(Path(artifact_root) / selected["source_manifest"]),
            "result.selected_candidate.source_manifest",
        )
        source_snapshot = multistream_execution.validate_source_snapshot(
            source_manifest, artifact_root
        )
        for field in ("source_revision", "source_fingerprint"):
            if source_snapshot[field] != selected[field]:
                raise ValueError(
                    f"selected_candidate.source_manifest.{field} identity mismatch"
                )
        if accepted_trial["change_kind"] == "option":
            if selected["config_manifest"] != accepted_trial["action_manifest"]:
                raise ValueError(
                    "accepted option selected config_manifest must be its action_manifest"
                )
            if (
                accepted_action.get("output_content_fingerprint")
                != selected["config_fingerprint"]
            ):
                raise ValueError(
                    "accepted option action manifest config fingerprint mismatch"
                )
            if (
                selected["source_revision"] != request["incumbent"]["source_revision"]
                or selected["source_fingerprint"]
                != request["incumbent"]["source_fingerprint"]
            ):
                raise ValueError("accepted option trial must preserve source identity")
            if (
                selected["config_fingerprint"]
                == request["incumbent"]["config_fingerprint"]
            ):
                raise ValueError("accepted option trial must change config_fingerprint")
        else:
            if (
                selected["source_revision"] == request["incumbent"]["source_revision"]
                or selected["source_fingerprint"]
                == request["incumbent"]["source_fingerprint"]
            ):
                raise ValueError(
                    "accepted source trial must change source revision and fingerprint"
                )
    else:
        if unchanged is not True:
            raise ValueError(f"{status} result requires incumbent_unchanged=true")
        if selected is not None:
            raise ValueError(f"{status} result requires selected_candidate=null")
        if accepted_trial_ids:
            raise ValueError(f"{status} result cannot contain an accepted trial")

    return {
        "valid": True,
        "schema_version": RESULT_SCHEMA,
        "request_id": request["request_id"],
        "request_fingerprint": request_summary["request_fingerprint"],
        "status": status,
        "incumbent_unchanged": unchanged,
        "accepted_trial_id": accepted_trial_ids[0] if accepted_trial_ids else None,
        "fallback": expected_fallback,
    }


def build_fallback(request, artifact_root, status, reason_zh):
    summary = validate_request(request, artifact_root)
    if status not in {"no_gain", "blocked", "failed"}:
        raise ValueError("fallback status must be no_gain, blocked, or failed")
    reason_zh = _chinese_text(reason_zh, "reason_zh")
    fallback = dict(summary["incumbent"])
    fallback["reason_zh"] = reason_zh
    return {
        "schema_version": RESULT_SCHEMA,
        "request_id": request["request_id"],
        "parent_experiment_id": request["parent_experiment_id"],
        "request_fingerprint": summary["request_fingerprint"],
        "status": status,
        "incumbent_unchanged": True,
        "selected_candidate": None,
        "trials": [],
        "multistream_findings": [],
        "blockers": [reason_zh] if status in {"blocked", "failed"} else [],
        "fallback": fallback,
    }


def _write_json(value, output):
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    else:
        print(rendered, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    request_parser = subparsers.add_parser("validate-request")
    request_parser.add_argument("request", type=Path)
    request_parser.add_argument("--json-out", type=Path)

    result_parser = subparsers.add_parser("validate-result")
    result_parser.add_argument("request", type=Path)
    result_parser.add_argument("result", type=Path)
    result_parser.add_argument("--json-out", type=Path)

    fallback_parser = subparsers.add_parser("fallback")
    fallback_parser.add_argument("request", type=Path)
    fallback_parser.add_argument(
        "--status", required=True, choices=("no_gain", "blocked", "failed")
    )
    fallback_parser.add_argument("--reason-zh", required=True)
    fallback_parser.add_argument("--json-out", type=Path)

    args = parser.parse_args(argv)
    try:
        request = _load_json(args.request)
        root = args.request.parent
        if args.command == "validate-request":
            output = validate_request(request, root)
        elif args.command == "validate-result":
            output = validate_result(request, _load_json(args.result), root)
        else:
            output = build_fallback(request, root, args.status, args.reason_zh)
    except ValueError as error:
        parser.error(str(error))
    _write_json(output, args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
