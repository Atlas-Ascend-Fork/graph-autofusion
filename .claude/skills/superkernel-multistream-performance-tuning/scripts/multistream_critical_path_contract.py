#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate graph-level multistream request-v2 and result-v3 artifacts."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import multistream_critical_path
import multistream_dependency_evidence
import multistream_event_stage_action
import multistream_event_stage_planner
import multistream_evidence
import multistream_join_validation
import multistream_logical_graph
import multistream_source_transform


REQUEST_SCHEMA = "superkernel-multistream-request-v2"
RESULT_SCHEMA = "superkernel-multistream-result-v3"
RESULT_STATUSES = {"accepted", "no_gain", "blocked", "failed"}
CHANGE_KINDS = {"event_edge_refinement", "stage_split", "scope_event_derivative"}
FINGERPRINT_FIELDS = (
    "source_fingerprint",
    "config_fingerprint",
    "control_fingerprint",
    "workload_fingerprint",
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


def graph_fingerprint(value):
    unsigned = {key: item for key, item in value.items() if key != "graph_fingerprint"}
    return multistream_logical_graph.fingerprint(unsigned)


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _relative(value, label):
    value = _text(value, label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _rooted(root, value, label, *, directory=False):
    root = Path(root).resolve()
    path = (root / _relative(value, label)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{label} does not exist: {value}")
    return path


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def _incumbent(value, root):
    required = {
        "candidate_name",
        "source_revision",
        *FINGERPRINT_FIELDS,
        "clean_performance_summary",
        "clean_run_count",
        "stable",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("request.incumbent fields are invalid")
    for field in ("candidate_name", "source_revision", *FINGERPRINT_FIELDS):
        _text(value[field], f"request.incumbent.{field}")
    _rooted(
        root,
        value["clean_performance_summary"],
        "request.incumbent.clean_performance_summary",
    )
    if value["clean_run_count"] != 5 or value["stable"] is not True:
        raise ValueError("critical path request requires a stable clean5 incumbent")
    return dict(value)


def validate_request(value, artifact_root):
    required = {
        "schema_version",
        "request_id",
        "parent_experiment_id",
        "incumbent",
        "artifacts",
        "isolation",
        "budget",
        "authorization",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != REQUEST_SCHEMA
    ):
        raise ValueError(f"critical path request must use {REQUEST_SCHEMA}")
    root = Path(artifact_root).resolve()
    request_fp = fingerprint(value)
    incumbent = _incumbent(value["incumbent"], root)
    artifacts = value["artifacts"]
    artifact_fields = {
        "logical_graph",
        "dependency_evidence",
        "critical_path_capture",
        "event_stage_action_catalog",
        "event_stage_history",
        "candidate_matrix",
        "scope_derivatives",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != artifact_fields:
        raise ValueError("critical path request artifacts fields are invalid")
    paths = {
        field: _rooted(root, artifacts[field], f"request.artifacts.{field}")
        for field in artifact_fields - {"scope_derivatives"}
    }
    derivatives = None
    if artifacts["scope_derivatives"] is not None:
        derivatives_path = _rooted(
            root, artifacts["scope_derivatives"], "request.artifacts.scope_derivatives"
        )
        derivatives = _load(derivatives_path)
    graph = multistream_logical_graph.validate(_load(paths["logical_graph"]))
    if graph["request_fingerprint"] != request_fp:
        raise ValueError("logical graph does not bind request fingerprint")
    dependency = multistream_dependency_evidence.validate(
        paths["dependency_evidence"],
        root,
        require_complete=True,
        request_fingerprint=request_fp,
    )
    if graph["dependency_evidence_fingerprint"] != dependency["evidence_fingerprint"]:
        raise ValueError("logical graph does not bind dependency evidence")
    graph_statements = {
        statement_id
        for stage in graph["stages"]
        for statement_id in stage["source_statement_ids"]
    }
    if not graph_statements.issubset(set(dependency["statement_ids"])):
        raise ValueError(
            "dependency evidence does not cover every logical graph statement"
        )
    capture = multistream_critical_path.validate_capture(
        _load(paths["critical_path_capture"])
    )
    if (
        capture["request_fingerprint"] != request_fp
        or capture["logical_graph"]["graph_fingerprint"] != graph["graph_fingerprint"]
    ):
        raise ValueError("critical path capture does not bind request or logical graph")
    analysis = multistream_critical_path.analyze(capture)
    catalog = multistream_event_stage_action.validate(
        _load(paths["event_stage_action_catalog"])
    )
    if (
        catalog["critical_path_analysis_fingerprint"]
        != analysis["analysis_fingerprint"]
    ):
        raise ValueError(
            "event/stage action catalog does not bind critical path analysis"
        )
    history = _load(paths["event_stage_history"])
    budget = value["budget"]
    if not isinstance(budget, dict) or set(budget) != {
        "max_trials",
        "min_predicted_e2e_upper_bound",
        "min_clean_gain_pct",
    }:
        raise ValueError("critical path request budget fields are invalid")
    if (
        isinstance(budget["max_trials"], bool)
        or not isinstance(budget["max_trials"], int)
        or budget["max_trials"] < 1
    ):
        raise ValueError("budget.max_trials must be positive")
    if (
        budget["min_predicted_e2e_upper_bound"]
        != multistream_critical_path.PREDICTED_E2E_GATE
        or budget["min_clean_gain_pct"] != 2.0
    ):
        raise ValueError(
            "critical path request must use frozen 3% screening and 2% clean gates"
        )
    matrix = _load(paths["candidate_matrix"])
    multistream_event_stage_planner.validate(
        matrix, analysis, catalog, history, budget["max_trials"], derivatives
    )
    isolation = value["isolation"]
    if not isinstance(isolation, dict) or set(isolation) != {
        "source_worktree",
        "experiment_root",
        "config_root",
        "cache_root",
        "immutable_incumbent",
        "dedicated_roots",
    }:
        raise ValueError("critical path request isolation fields are invalid")
    roots = [
        _rooted(root, isolation[field], f"request.isolation.{field}", directory=True)
        for field in ("source_worktree", "experiment_root", "config_root", "cache_root")
    ]
    if (
        len(set(roots)) != 4
        or isolation["immutable_incumbent"] is not True
        or isolation["dedicated_roots"] is not True
    ):
        raise ValueError(
            "critical path request requires four distinct dedicated roots and immutable incumbent"
        )
    authorization = value["authorization"]
    if not isinstance(authorization, dict) or set(authorization) != {
        "run_inference",
        "edit_isolated_worktree",
        "allowed_change_kinds",
    }:
        raise ValueError("critical path request authorization fields are invalid")
    if (
        authorization["run_inference"] is not True
        or authorization["edit_isolated_worktree"] is not True
    ):
        raise ValueError("critical path request lacks execution authorization")
    if authorization["allowed_change_kinds"] != sorted(CHANGE_KINDS):
        raise ValueError(
            "allowed_change_kinds must be the complete sorted critical path catalog"
        )
    return {
        "valid": True,
        "request_id": _text(value["request_id"], "request_id"),
        "parent_experiment_id": _text(
            value["parent_experiment_id"], "parent_experiment_id"
        ),
        "request_fingerprint": request_fp,
        "incumbent": incumbent,
        "analysis_decision": analysis["decision"],
        "candidate_count": len(matrix["candidates"]),
    }


def _fallback_identity(incumbent):
    return {
        field: incumbent[field]
        for field in ("candidate_name", "source_revision", *FINGERPRINT_FIELDS)
    }


def validate_result(request, value, artifact_root):
    summary = validate_request(request, artifact_root)
    required = {
        "schema_version",
        "request_id",
        "parent_experiment_id",
        "request_fingerprint",
        "status",
        "incumbent_unchanged",
        "selected_candidate",
        "trials",
        "blockers",
        "fallback",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != RESULT_SCHEMA
    ):
        raise ValueError(f"critical path result must use {RESULT_SCHEMA}")
    if (
        value["request_id"] != summary["request_id"]
        or value["parent_experiment_id"] != summary["parent_experiment_id"]
        or value["request_fingerprint"] != summary["request_fingerprint"]
    ):
        raise ValueError("critical path result identity differs from request")
    status = value["status"]
    if status not in RESULT_STATUSES:
        raise ValueError("critical path result status is invalid")
    if not isinstance(value["trials"], list) or not isinstance(value["blockers"], list):
        raise ValueError("critical path result trials/blockers must be lists")
    if status in {"blocked", "failed"} and not value["blockers"]:
        raise ValueError("blocked/failed critical path result requires blockers")
    expected_fallback = _fallback_identity(request["incumbent"])
    fallback = value["fallback"]
    if (
        not isinstance(fallback, dict)
        or {key: fallback.get(key) for key in expected_fallback} != expected_fallback
    ):
        raise ValueError("critical path result fallback differs from incumbent")
    root = Path(artifact_root).resolve()
    matrix = _load(
        _rooted(
            root,
            request["artifacts"]["candidate_matrix"],
            "request.artifacts.candidate_matrix",
        )
    )
    graph = multistream_logical_graph.validate(
        _load(
            _rooted(
                root,
                request["artifacts"]["logical_graph"],
                "request.artifacts.logical_graph",
            )
        )
    )
    candidates = {item["candidate_id"]: item for item in matrix["candidates"]}
    accepted = []
    seen_trials = set()
    for index, trial in enumerate(value["trials"]):
        fields = {
            "trial_id",
            "candidate_id",
            "action_id",
            "parent_action_id",
            "change_kind",
            "decision",
            "action_manifest",
            "join_validation",
            "parent_clean_evidence",
            "post_dependency_evidence",
            "clean_evidence",
            "clean_state",
        }
        if not isinstance(trial, dict) or set(trial) != fields:
            raise ValueError(f"result.trials[{index}] fields are invalid")
        trial_id = _text(trial["trial_id"], f"result.trials[{index}].trial_id")
        if trial_id in seen_trials:
            raise ValueError("critical path result has duplicate trial_id")
        seen_trials.add(trial_id)
        candidate = candidates.get(trial["candidate_id"])
        if (
            candidate is None
            or candidate["action_id"] != trial["action_id"]
            or candidate["change_kind"] != trial["change_kind"]
            or candidate.get("parent_action_id") != trial["parent_action_id"]
        ):
            raise ValueError("critical path trial differs from frozen candidate matrix")
        if trial["decision"] not in {"accepted", "rejected", "blocked", "failed"}:
            raise ValueError("critical path trial decision is invalid")
        if trial["decision"] in {"blocked", "failed"}:
            if any(
                trial[field] is not None
                for field in (
                    "action_manifest",
                    "join_validation",
                    "parent_clean_evidence",
                    "post_dependency_evidence",
                    "clean_evidence",
                    "clean_state",
                )
            ):
                raise ValueError(
                    "blocked/failed trial cannot cite unvalidated phase evidence"
                )
            continue
        action_path = _rooted(
            root, trial["action_manifest"], f"result.trials[{index}].action_manifest"
        )
        action = _load(action_path)
        if (
            action.get("schema_version")
            != multistream_source_transform.ACTION_MANIFEST_SCHEMA
            or action.get("trial_id") != trial_id
            or action.get("action_id") != trial["action_id"]
            or action.get("change_kind") != trial["change_kind"]
            or action.get("single_change_verified") is not True
            or action.get("multistream_only_verified") is not True
        ):
            raise ValueError(
                "critical path trial action manifest identity or gates are invalid"
            )
        if (
            action.get("dependency_evidence_fingerprint_before")
            != graph["dependency_evidence_fingerprint"]
        ):
            raise ValueError(
                "critical path action does not bind initial dependency evidence"
            )
        post_dependency_path = _rooted(
            root,
            trial["post_dependency_evidence"],
            f"result.trials[{index}].post_dependency_evidence",
        )
        post_dependency = multistream_dependency_evidence.validate(
            post_dependency_path,
            root,
            require_complete=True,
            request_fingerprint=summary["request_fingerprint"],
        )
        if (
            action.get("dependency_evidence_fingerprint_after")
            != post_dependency["evidence_fingerprint"]
        ):
            raise ValueError(
                "critical path action does not bind post-transform dependency evidence"
            )
        if (
            post_dependency["evidence_fingerprint"]
            == graph["dependency_evidence_fingerprint"]
        ):
            raise ValueError(
                "post-transform dependency evidence must be independently sealed"
            )
        action_fp = multistream_source_transform.fingerprint(action)
        join_path = _rooted(
            root, trial["join_validation"], f"result.trials[{index}].join_validation"
        )
        join = multistream_join_validation.validate(_load(join_path))
        if (
            join["trial_id"] != trial_id
            or join["request_fingerprint"] != summary["request_fingerprint"]
            or join["action_manifest_fingerprint"] != action_fp
        ):
            raise ValueError("critical path trial join validation identity mismatch")
        if join["decision"] == "not_effective":
            if trial["decision"] != "rejected" or any(
                trial[field] is not None
                for field in ("parent_clean_evidence", "clean_evidence", "clean_state")
            ):
                raise ValueError("not-effective join must reject before clean")
            continue
        if trial["change_kind"] == "scope_event_derivative":
            if trial["parent_clean_evidence"] is None:
                raise ValueError(
                    "scope derivative requires clean comparison against parent candidate"
                )
            parent_clean_path = _rooted(
                root,
                trial["parent_clean_evidence"],
                f"result.trials[{index}].parent_clean_evidence",
            )
            parent_clean = multistream_evidence.validate_evidence(
                parent_clean_path,
                root,
                trial_id=trial_id,
                request_fingerprint=summary["request_fingerprint"],
                state_after="clean3_passed",
            )
            if parent_clean["decision"] != "pass":
                raise ValueError(
                    "scope derivative must improve against its parent candidate"
                )
        elif trial["parent_clean_evidence"] is not None:
            raise ValueError("non-scope trial cannot cite parent_clean_evidence")
        if (
            trial["clean_state"] not in {"clean3_passed", "clean5_passed"}
            or trial["clean_evidence"] is None
        ):
            raise ValueError("mechanism-validated trial requires clean evidence")
        clean_path = _rooted(
            root, trial["clean_evidence"], f"result.trials[{index}].clean_evidence"
        )
        clean = multistream_evidence.validate_evidence(
            clean_path,
            root,
            trial_id=trial_id,
            request_fingerprint=summary["request_fingerprint"],
            state_after=trial["clean_state"],
        )
        if trial["decision"] == "accepted":
            if trial["clean_state"] != "clean5_passed" or clean["decision"] != "pass":
                raise ValueError("accepted critical path trial requires passing clean5")
            accepted.append(trial)
        elif clean["decision"] != "reject":
            raise ValueError(
                "rejected critical path trial requires rejected clean evidence"
            )
    if len(value["trials"]) > request["budget"]["max_trials"]:
        raise ValueError("critical path result exceeds trial budget")
    if status == "accepted":
        if (
            value["incumbent_unchanged"] is not False
            or len(accepted) != 1
            or not isinstance(value["selected_candidate"], dict)
        ):
            raise ValueError(
                "accepted critical path result requires one accepted trial and selected candidate"
            )
        selected = value["selected_candidate"]
        if selected.get("trial_id") != accepted[0].get("trial_id"):
            raise ValueError("selected_candidate trial_id differs from accepted trial")
        for field in ("candidate_name", "source_revision", *FINGERPRINT_FIELDS):
            _text(selected.get(field), f"selected_candidate.{field}")
        if (
            selected["source_fingerprint"] == request["incumbent"]["source_fingerprint"]
            or selected["source_revision"] == request["incumbent"]["source_revision"]
        ):
            raise ValueError(
                "accepted critical path source action must change source identity"
            )
        if (
            selected["control_fingerprint"]
            != request["incumbent"]["control_fingerprint"]
            or selected["workload_fingerprint"]
            != request["incumbent"]["workload_fingerprint"]
        ):
            raise ValueError(
                "accepted critical path candidate must preserve control/workload identity"
            )
    else:
        if (
            value["incumbent_unchanged"] is not True
            or value["selected_candidate"] is not None
            or accepted
        ):
            raise ValueError(
                f"{status} result requires unchanged incumbent and selected_candidate=null"
            )
    return {
        "valid": True,
        "status": status,
        "incumbent_unchanged": value["incumbent_unchanged"],
        "request_fingerprint": summary["request_fingerprint"],
    }


def build_fallback(request, artifact_root, status, reason_zh):
    summary = validate_request(request, artifact_root)
    if status not in {"no_gain", "blocked", "failed"}:
        raise ValueError("critical path fallback status is invalid")
    reason_zh = _text(reason_zh, "reason_zh")
    if not any("\u4e00" <= char <= "\u9fff" for char in reason_zh):
        raise ValueError("reason_zh must contain Chinese text")
    fallback = _fallback_identity(request["incumbent"])
    fallback["reason_zh"] = reason_zh
    return {
        "schema_version": RESULT_SCHEMA,
        "request_id": summary["request_id"],
        "parent_experiment_id": summary["parent_experiment_id"],
        "request_fingerprint": summary["request_fingerprint"],
        "status": status,
        "incumbent_unchanged": True,
        "selected_candidate": None,
        "trials": [],
        "blockers": [reason_zh] if status in {"blocked", "failed"} else [],
        "fallback": fallback,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate_request_parser = commands.add_parser("validate-request")
    validate_request_parser.add_argument("--request", type=Path, required=True)
    validate_request_parser.add_argument("--artifact-root", type=Path, required=True)
    validate_result_parser = commands.add_parser("validate-result")
    validate_result_parser.add_argument("--request", type=Path, required=True)
    validate_result_parser.add_argument("--result", type=Path, required=True)
    validate_result_parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = _load(args.request)
        result = (
            validate_request(request, args.artifact_root)
            if args.command == "validate-request"
            else validate_result(request, _load(args.result), args.artifact_root)
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
