#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build a bounded, deduplicated multistream trial candidate matrix."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import multistream_contract
import multistream_component_reorder
import multistream_execution
import multistream_operator_order
import multistream_trace_analysis


CATALOG_SCHEMA = "superkernel-multistream-action-catalog-v1"
HISTORY_SCHEMA = "superkernel-multistream-option-history-v1"
MATRIX_SCHEMA = "superkernel-multistream-candidate-matrix-v1"
SETTLED_STATUSES = {"accepted", "rejected", "no_gain", "failed", "executed"}
RISK_PENALTY = {"low": 0, "medium": 10, "high": 25}


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


def _rooted(root, value, label):
    root = Path(root).resolve()
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _risk(value, label):
    value = _text(value, label)
    if value not in RISK_PENALTY:
        raise ValueError(f"{label} must be low, medium, or high")
    return value


def _load_history(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get("schema_version") != HISTORY_SCHEMA:
        raise ValueError(f"option history must use {HISTORY_SCHEMA}")
    trials = value.get("settled_trials")
    if not isinstance(trials, list):
        raise ValueError("option history settled_trials must be a list")
    settled = set()
    for index, trial in enumerate(trials):
        required = {"json_pointer", "after", "status"}
        if not isinstance(trial, dict) or set(trial) != required:
            raise ValueError(
                f"settled_trials[{index}] must contain exactly {sorted(required)}"
            )
        pointer = _text(trial["json_pointer"], f"settled_trials[{index}].json_pointer")
        status = _text(trial["status"], f"settled_trials[{index}].status")
        if status in SETTLED_STATUSES:
            settled.add((pointer, _canonical(trial["after"])))
    return value, settled


def _validate_catalog(catalog, root, target_ids):
    if not isinstance(catalog, dict) or catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError(f"action catalog must use {CATALOG_SCHEMA}")
    if set(catalog) != {
        "schema_version",
        "options",
        "source_actions",
        "catalog_fingerprint",
    }:
        raise ValueError("action catalog fields are invalid")
    if catalog["catalog_fingerprint"] != fingerprint(
        {key: value for key, value in catalog.items() if key != "catalog_fingerprint"}
    ):
        raise ValueError("action catalog fingerprint mismatch")
    options = catalog["options"]
    sources = catalog["source_actions"]
    if not isinstance(options, list) or not isinstance(sources, list):
        raise ValueError("action catalog options/source_actions must be lists")
    normalized_options = []
    for index, option in enumerate(options):
        required = {
            "option_name",
            "json_pointer",
            "before",
            "accepted_values",
            "accepted_evidence",
            "applies_to",
            "risk",
            "priority",
        }
        if not isinstance(option, dict) or set(option) != required:
            raise ValueError(
                f"options[{index}] must contain exactly {sorted(required)}"
            )
        values = option["accepted_values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"options[{index}].accepted_values must be non-empty")
        if len({_canonical(item) for item in values}) != len(values):
            raise ValueError(f"options[{index}].accepted_values contains duplicates")
        if any(_canonical(item) == _canonical(option["before"]) for item in values):
            raise ValueError(
                f"options[{index}].accepted_values must differ from before"
            )
        applies = option["applies_to"]
        if not isinstance(applies, list) or not applies:
            raise ValueError(f"options[{index}].applies_to must be non-empty")
        if any(item != "*" and item not in target_ids for item in applies):
            raise ValueError(f"options[{index}].applies_to references unknown target")
        evidence = _rooted(
            root, option["accepted_evidence"], f"options[{index}].accepted_evidence"
        )
        priority = option["priority"]
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 0 <= priority <= 100
        ):
            raise ValueError(
                f"options[{index}].priority must be an integer in [0, 100]"
            )
        normalized_options.append(
            {
                **option,
                "option_name": _text(
                    option["option_name"], f"options[{index}].option_name"
                ),
                "json_pointer": _text(
                    option["json_pointer"], f"options[{index}].json_pointer"
                ),
                "accepted_evidence": _relative(root, evidence),
                "risk": _risk(option["risk"], f"options[{index}].risk"),
            }
        )
    normalized_sources = []
    for index, action in enumerate(sources):
        required = {
            "range_id",
            "change_kind",
            "source_file",
            "start_offset",
            "end_offset",
            "boundary_change",
            "insertions",
            "risk",
            "priority",
        }
        if not isinstance(action, dict) or set(action) != required:
            raise ValueError(
                f"source_actions[{index}] must contain exactly {sorted(required)}"
            )
        if action["range_id"] not in target_ids:
            raise ValueError(f"source_actions[{index}] references unknown target")
        if action["change_kind"] not in {"scope_split", "range_exclusion"}:
            raise ValueError(f"source_actions[{index}].change_kind is invalid")
        start, end = action["start_offset"], action["end_offset"]
        if (
            any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in (start, end)
            )
            or not 0 <= start < end
        ):
            raise ValueError(f"source_actions[{index}] byte range is invalid")
        insertions = action["insertions"]
        if not isinstance(insertions, list) or not insertions:
            raise ValueError(f"source_actions[{index}].insertions must be non-empty")
        multistream_execution._validate_source_action(
            action["change_kind"], start, end, insertions
        )
        priority = action["priority"]
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 0 <= priority <= 100
        ):
            raise ValueError(f"source_actions[{index}].priority must be in [0, 100]")
        normalized_sources.append(
            {**action, "risk": _risk(action["risk"], f"source_actions[{index}].risk")}
        )
    return normalized_options, normalized_sources


def _target_score(target):
    parallelism = {"degraded": 50, "preserved": 10, "improved": 0, "unknown": -100}[
        target["parallelism_effect"]
    ]
    net = {
        "beneficial": 30,
        "regressed": 20,
        "neutral": 10,
        "insufficient_evidence": -50,
    }[target["net_effect"]]
    decomposition = target.get("diagnostic_decomposition", {})
    lost_cv = decomposition.get("lost_cube_vector_overlap_us", 0.0)
    if isinstance(lost_cv, bool) or not isinstance(lost_cv, (int, float)):
        lost_cv = 0.0
    latent_bonus = 20 if target.get("latent_opportunity") is True else 0
    return parallelism + net + latent_bonus + min(20.0, max(0.0, float(lost_cv)))


def _reorder_candidates(
    request,
    request_summary,
    trace_targets,
    analysis_path,
    root,
):
    validation = multistream_operator_order.validate_analysis(
        analysis_path,
        root,
        expected_request_fingerprint=request_summary["request_fingerprint"],
    )
    analysis = json.loads(Path(analysis_path).read_text())
    source_map = request["artifacts"].get("source_scope_map")
    analysis_targets = request_summary["analysis_targets"]
    candidates = []
    for target in analysis["targets"]:
        range_id = target["range_id"]
        if range_id not in trace_targets or range_id not in analysis_targets:
            raise ValueError(
                f"operator reorder target is not declared by request: {range_id}"
            )
        trace_target = trace_targets[range_id]
        if trace_target["parallelism_effect"] != "degraded":
            raise ValueError(
                f"operator reorder target {range_id} requires degraded parallelism"
            )
        if (
            target["graph_occurrence_fingerprint"]
            != analysis_targets[range_id]["graph_occurrence_fingerprint"]
        ):
            raise ValueError(f"operator reorder target {range_id} occurrence differs")
        binding = analysis_targets[range_id]
        boundary = binding.get("boundary") or {}
        expected_boundary = {
            "source_file": target["source_file"],
            "start_offset": target["range_start_offset"],
            "end_offset": target["range_end_offset"],
        }
        if (binding["mapping_method"], binding["mapping_confidence"]) != (
            "source_scope_map",
            "exact",
        ) or any(
            boundary.get(field) != value for field, value in expected_boundary.items()
        ):
            raise ValueError(
                f"operator reorder target {range_id} lacks matching source_scope_map + exact"
            )
        if target["multistream_reorder_authorized"] is not True:
            continue
        base_action = {
            "source_scope_map": source_map,
            "operator_order_analysis": _relative(root, analysis_path),
            "operator_order_analysis_fingerprint": validation["analysis_fingerprint"],
            "range_id": range_id,
            **expected_boundary,
            "before_order": target["current_source_order"],
            "hard_dependencies": target["hard_dependencies"],
            "statements": target["statements"],
            "stable_parallel_pairs": target["stable_parallel_pairs"],
            "stable_resource_complementary_pairs": target[
                "stable_resource_complementary_pairs"
            ],
            "stable_dispatch_inversions": target["stable_dispatch_inversions"],
        }
        score = _target_score(trace_target) + 100
        candidates.append(
            {
                "change_kind": multistream_execution.REORDER_CHANGE_KIND,
                "target_range_ids": [range_id],
                "global_effect": False,
                "risk": "medium",
                "score": score,
                "route": "route2",
                "activation_condition": "immediate",
                "action": {
                    **base_action,
                    "after_order": target["route2_order"],
                    "route": "route2",
                },
            }
        )
        for index, order in enumerate(target["route3_orders"], start=1):
            candidates.append(
                {
                    "change_kind": multistream_execution.REORDER_CHANGE_KIND,
                    "target_range_ids": [range_id],
                    "global_effect": False,
                    "risk": "medium",
                    "score": score - 20 - index,
                    "route": "route3",
                    "activation_condition": "route2_settled_without_incremental_gain",
                    "action": {**base_action, "after_order": order, "route": "route3"},
                }
            )
    return candidates, validation


def plan(
    request_path,
    trace_analysis_path,
    catalog_path,
    artifact_root=None,
    operator_order_analysis_path=None,
    component_transform_path=None,
):
    request_path = Path(request_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else request_path.parent
    request = json.loads(request_path.read_text())
    request_summary = multistream_contract.validate_request(request, root)
    trace_path = _rooted(root, trace_analysis_path, "trace analysis")
    trace_validation = multistream_trace_analysis.validate_analysis(
        trace_path, request_path, root
    )
    trace = json.loads(trace_path.read_text())
    catalog_path = _rooted(root, catalog_path, "action catalog")
    catalog = json.loads(catalog_path.read_text())
    target_ids = set(request_summary["target_range_ids"])
    options, sources = _validate_catalog(catalog, root, target_ids)
    history_path = _rooted(
        root, request["artifacts"]["option_history"], "option history"
    )
    history, settled = _load_history(history_path)
    trace_targets = {item["range_id"]: item for item in trace["targets"]}
    candidates = []
    deduplicated = []
    requested = set(request_summary["requested_change_kinds"])
    order_validation = None
    if multistream_execution.REORDER_CHANGE_KIND in requested:
        if operator_order_analysis_path is None:
            raise ValueError(
                "dependency_safe_operator_reorder requires --operator-order-analysis"
            )
        order_path = _rooted(
            root, operator_order_analysis_path, "operator order analysis"
        )
        reorder_candidates, order_validation = _reorder_candidates(
            request, request_summary, trace_targets, order_path, root
        )
        candidates.extend(reorder_candidates)
    component_transform_record = None
    if multistream_component_reorder.CHANGE_KIND in requested:
        if component_transform_path is None:
            raise ValueError("component_overlap_reorder requires --component-transform")
        transform_path = _rooted(root, component_transform_path, "component transform")
        source_map_path = _rooted(
            root, request["artifacts"]["component_source_map"], "component source map"
        )
        capture_path = _rooted(
            root, request["artifacts"]["component_order_capture"], "component capture"
        )
        source_root = (root / request["isolation"]["source_worktree"]).resolve()
        source_map = multistream_component_reorder.validate_source_map(
            json.loads(source_map_path.read_text()), source_root
        )
        capture = json.loads(capture_path.read_text())
        transform = json.loads(transform_path.read_text())
        normalized, component_validation = (
            multistream_component_reorder.validate_transform(
                transform, source_map, capture, source_root, root
            )
        )
        target_ranges = sorted(
            item["range_id"] for item in source_map["target_bindings"]
        )
        candidates.append(
            {
                "change_kind": multistream_component_reorder.CHANGE_KIND,
                "target_range_ids": target_ranges,
                "global_effect": False,
                "risk": "high",
                "score": max(
                    _target_score(trace_targets[item]) for item in target_ranges
                )
                + 100,
                "route": "component_exception",
                "activation_condition": "immediate",
                "action": {
                    "component_transform": _relative(root, transform_path),
                    "component_transform_fingerprint": normalized[
                        "transform_fingerprint"
                    ],
                    "component_source_map": _relative(root, source_map_path),
                    "component_source_map_fingerprint": source_map[
                        "mapping_fingerprint"
                    ],
                    "component_order_capture": _relative(root, capture_path),
                    "component_order_capture_fingerprint": capture[
                        "capture_fingerprint"
                    ],
                    "target_set_id": source_map["target_set_id"],
                    "before_order": component_validation["before_order"],
                    "after_order": component_validation["after_order"],
                },
            }
        )
        component_transform_record = {
            "path": _relative(root, transform_path),
            "transform_fingerprint": normalized["transform_fingerprint"],
            "component_aware": True,
        }
    if "option" in requested:
        for option in options:
            applicable = sorted(
                target_ids
                if option["applies_to"] == ["*"]
                else set(option["applies_to"])
            )
            opportunity_targets = [
                item
                for item in applicable
                if trace_targets[item]["optimization_status"] == "opportunity"
            ]
            if not opportunity_targets:
                continue
            trigger_score = max(
                _target_score(trace_targets[item]) for item in opportunity_targets
            )
            for value in option["accepted_values"]:
                key = (option["json_pointer"], _canonical(value))
                if key in settled:
                    deduplicated.append(
                        {
                            "change_kind": "option",
                            "json_pointer": option["json_pointer"],
                            "after": value,
                            "reason": "identical_global_option_value_settled_by_stage_o",
                        }
                    )
                    continue
                candidates.append(
                    {
                        "change_kind": "option",
                        "target_range_ids": applicable,
                        "global_effect": True,
                        "risk": option["risk"],
                        "score": trigger_score
                        + option["priority"]
                        - RISK_PENALTY[option["risk"]],
                        "action": {
                            "option_name": option["option_name"],
                            "json_pointer": option["json_pointer"],
                            "before": option["before"],
                            "after": value,
                            "accepted_evidence": option["accepted_evidence"],
                        },
                    }
                )
    analysis_targets = request_summary["analysis_targets"]
    source_map = request["artifacts"].get("source_scope_map")
    for action in sources:
        if action["change_kind"] not in requested:
            continue
        range_id = action["range_id"]
        if trace_targets[range_id]["optimization_status"] != "opportunity":
            continue
        binding = analysis_targets[range_id]
        boundary = binding.get("boundary") or {}
        expected = {
            "source_file": action["source_file"],
            "start_offset": action["start_offset"],
            "end_offset": action["end_offset"],
        }
        if (
            binding["mapping_method"] != "source_scope_map"
            or binding["mapping_confidence"] != "exact"
        ):
            raise ValueError(
                f"source action {range_id} lacks source_scope_map + exact binding"
            )
        if any(boundary.get(field) != value for field, value in expected.items()):
            raise ValueError(f"source action {range_id} differs from analyzer boundary")
        candidates.append(
            {
                "change_kind": action["change_kind"],
                "target_range_ids": [range_id],
                "global_effect": False,
                "risk": action["risk"],
                "score": _target_score(trace_targets[range_id])
                + action["priority"]
                - RISK_PENALTY[action["risk"]],
                "action": {
                    "source_scope_map": source_map,
                    **expected,
                    "boundary_change": action["boundary_change"],
                    "insertions": action["insertions"],
                },
            }
        )
    candidates.sort(
        key=lambda item: (
            0 if item.get("activation_condition", "immediate") == "immediate" else 1,
            -item["score"],
            0 if item.get("route") == "route2" else 1,
            RISK_PENALTY[item["risk"]],
            item["change_kind"],
            _canonical(item["action"]),
        )
    )
    maximum = request["budget"]["max_trials"]
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"MS-C{index:03d}"
        candidate["rank"] = index
        candidate["selected_for_execution"] = (
            index <= maximum
            and candidate.get("activation_condition", "immediate") == "immediate"
        )
        candidate["acceptance_gate"] = "incremental_clean_end_to_end_only"
    matrix = {
        "schema_version": MATRIX_SCHEMA,
        "request_id": request["request_id"],
        "request_fingerprint": request_summary["request_fingerprint"],
        "trace_analysis": {
            "path": _relative(root, trace_path),
            "analysis_fingerprint": trace_validation["analysis_fingerprint"],
        },
        "action_catalog": {
            "path": _relative(root, catalog_path),
            "catalog_fingerprint": catalog["catalog_fingerprint"],
        },
        "option_history": {
            "path": _relative(root, history_path),
            "history_fingerprint": fingerprint(history),
        },
        "budget": {
            "max_trials": maximum,
            "planned_trials": min(len(candidates), maximum),
            "candidate_count": len(candidates),
        },
        "candidates": candidates,
        "deduplicated_candidates": deduplicated,
        "stop_conditions": [
            "budget_exhausted",
            "all_legal_candidates_settled",
            "correctness_or_runtime_failure",
            "three_trace_recollections_exhausted",
            "no_incremental_clean_gain",
        ],
    }
    if order_validation is not None:
        matrix["operator_order_analysis"] = {
            "path": _relative(root, order_path),
            "analysis_fingerprint": order_validation["analysis_fingerprint"],
            "multistream_only": True,
        }
    if component_transform_record is not None:
        matrix["component_source_transform"] = component_transform_record
    matrix["matrix_fingerprint"] = fingerprint(matrix)
    return matrix


def validate_matrix(path, request_path, artifact_root=None):
    path = Path(path).resolve()
    root = (
        Path(artifact_root).resolve()
        if artifact_root
        else Path(request_path).resolve().parent
    )
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError(f"candidate matrix must use {MATRIX_SCHEMA}")
    if value.get("matrix_fingerprint") != fingerprint(
        {key: item for key, item in value.items() if key != "matrix_fingerprint"}
    ):
        raise ValueError("candidate matrix fingerprint mismatch")
    rebuilt = plan(
        request_path,
        value.get("trace_analysis", {}).get("path"),
        value.get("action_catalog", {}).get("path"),
        root,
        value.get("operator_order_analysis", {}).get("path"),
        value.get("component_source_transform", {}).get("path"),
    )
    if _canonical(rebuilt) != _canonical(value):
        raise ValueError("candidate matrix differs from deterministic replay")
    return {
        "valid": True,
        "candidate_count": len(value["candidates"]),
        "planned_trials": value["budget"]["planned_trials"],
        "matrix_fingerprint": value["matrix_fingerprint"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan")
    create.add_argument("--request", type=Path, required=True)
    create.add_argument("--trace-analysis", type=Path, required=True)
    create.add_argument("--action-catalog", type=Path, required=True)
    create.add_argument("--operator-order-analysis", type=Path)
    create.add_argument("--component-transform", type=Path)
    create.add_argument("--artifact-root", type=Path)
    create.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--matrix", type=Path, required=True)
    validate.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(
                args.request,
                args.trace_analysis,
                args.action_catalog,
                args.artifact_root,
                args.operator_order_analysis,
                args.component_transform,
            )
            if args.out.exists():
                raise ValueError(f"candidate matrix output already exists: {args.out}")
            _atomic_json(args.out, result)
        else:
            result = validate_matrix(args.matrix, args.request, args.artifact_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
