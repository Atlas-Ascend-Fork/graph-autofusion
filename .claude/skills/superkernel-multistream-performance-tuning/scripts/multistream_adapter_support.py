#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build a replayable capability and action support matrix for network adapters."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath


ADAPTER_SCHEMA = "superkernel-multistream-adapter-capability-v1"
ADAPTER_SCHEMA_V2 = "superkernel-multistream-adapter-capability-v2"
ADAPTER_SCHEMA_V3 = "superkernel-multistream-adapter-capability-v3"
MATRIX_SCHEMA = "superkernel-multistream-adapter-support-matrix-v1"
MATRIX_SCHEMA_V2 = "superkernel-multistream-adapter-support-matrix-v2"
MATRIX_SCHEMA_V3 = "superkernel-multistream-adapter-support-matrix-v3"
COLLECTION_PLAN_SCHEMA = "superkernel-adapter-collection-plan-v1"
PREFLIGHT_SCHEMA = "superkernel-adapter-capability-preflight-v1"
MODEL_RUN_SPEC_SCHEMA = "superkernel-multistream-model-run-spec-v1"
ROUND_ACTION_MANIFEST_SCHEMA = "superkernel-round-action-manifest-v1"
ROUND_KINDS = ("FINAL", "P", "multistream")
EXPERIMENT_IDENTITY_ROLES = (
    "config_identity",
    "control_identity",
    "source_identity",
    "workload_identity",
)
CAPABILITIES = (
    "clean_performance_validation",
    "core_family_classification",
    "dependency_evidence",
    "four_profile_orchestration",
    "identity_binding",
    "isolated_cleanup",
    "model_execution",
    "operator_reorder_materialization",
    "raw_capture_ingestion",
    "shared_npu_lease",
    "source_exact_mapping",
    "scope_api_materialization",
)
CAPABILITIES_V2 = CAPABILITIES + (
    "critical_path_analysis",
    "event_stage_dispatch_validation",
    "event_transform_materialization",
    "join_mechanism_validation",
    "logical_graph_capture",
    "stage_transform_materialization",
)
CAPABILITIES_V3 = CAPABILITIES_V2 + (
    "component_reorder_materialization",
    "multi_source_exact_mapping",
)
ARTIFACT_SCHEMAS = {
    "cleanup_evaluation": "superkernel-multistream-cleanup-evaluation-v1",
    "critical_path_npu_closure_receipt": "superkernel-multistream-critical-path-npu-closure-v1",
    "network_conformance_report": "superkernel-multistream-network-conformance-report-v1",
    "real_npu_closure_receipt": "superkernel-multistream-real-npu-receipt-v1",
    "resource_screening_receipt": "superkernel-multistream-resource-screening-receipt-v1",
    "shared_npu_lease_inventory": "superkernel-shared-npu-lease-inventory-v1",
}
ACTION_REQUIREMENTS = {
    "option": {
        "capabilities": (
            "clean_performance_validation",
            "isolated_cleanup",
            "model_execution",
            "shared_npu_lease",
        ),
        "artifacts": ("network_conformance_report", "shared_npu_lease_inventory"),
    },
    "scope_split": {
        "capabilities": (
            "clean_performance_validation",
            "identity_binding",
            "isolated_cleanup",
            "model_execution",
            "raw_capture_ingestion",
            "scope_api_materialization",
            "shared_npu_lease",
            "source_exact_mapping",
        ),
        "artifacts": ("network_conformance_report", "shared_npu_lease_inventory"),
    },
    "range_exclusion": {
        "capabilities": (
            "clean_performance_validation",
            "identity_binding",
            "isolated_cleanup",
            "model_execution",
            "raw_capture_ingestion",
            "scope_api_materialization",
            "shared_npu_lease",
            "source_exact_mapping",
        ),
        "artifacts": ("network_conformance_report", "shared_npu_lease_inventory"),
    },
    "dependency_safe_operator_reorder": {
        "capabilities": (
            "clean_performance_validation",
            "core_family_classification",
            "dependency_evidence",
            "four_profile_orchestration",
            "identity_binding",
            "isolated_cleanup",
            "model_execution",
            "operator_reorder_materialization",
            "raw_capture_ingestion",
            "shared_npu_lease",
            "source_exact_mapping",
        ),
        "artifacts": ("network_conformance_report", "shared_npu_lease_inventory"),
    },
}
ACTION_REQUIREMENTS_V2 = {
    **ACTION_REQUIREMENTS,
    "event_edge_refinement": {
        "capabilities": (
            "clean_performance_validation",
            "critical_path_analysis",
            "dependency_evidence",
            "event_stage_dispatch_validation",
            "event_transform_materialization",
            "four_profile_orchestration",
            "identity_binding",
            "isolated_cleanup",
            "join_mechanism_validation",
            "logical_graph_capture",
            "model_execution",
            "raw_capture_ingestion",
            "shared_npu_lease",
            "source_exact_mapping",
        ),
        "artifacts": (
            "critical_path_npu_closure_receipt",
            "network_conformance_report",
            "shared_npu_lease_inventory",
        ),
    },
    "stage_split": {
        "capabilities": (
            "clean_performance_validation",
            "critical_path_analysis",
            "dependency_evidence",
            "event_stage_dispatch_validation",
            "four_profile_orchestration",
            "identity_binding",
            "isolated_cleanup",
            "join_mechanism_validation",
            "logical_graph_capture",
            "model_execution",
            "raw_capture_ingestion",
            "shared_npu_lease",
            "source_exact_mapping",
            "stage_transform_materialization",
        ),
        "artifacts": (
            "critical_path_npu_closure_receipt",
            "network_conformance_report",
            "shared_npu_lease_inventory",
        ),
    },
    "scope_event_derivative": {
        "capabilities": (
            "clean_performance_validation",
            "critical_path_analysis",
            "four_profile_orchestration",
            "identity_binding",
            "isolated_cleanup",
            "join_mechanism_validation",
            "model_execution",
            "scope_api_materialization",
            "shared_npu_lease",
            "source_exact_mapping",
        ),
        "artifacts": (
            "critical_path_npu_closure_receipt",
            "network_conformance_report",
            "shared_npu_lease_inventory",
        ),
    },
}
ACTION_REQUIREMENTS_V3 = {
    **ACTION_REQUIREMENTS_V2,
    "component_overlap_reorder": {
        "capabilities": (
            "clean_performance_validation",
            "component_reorder_materialization",
            "core_family_classification",
            "dependency_evidence",
            "four_profile_orchestration",
            "identity_binding",
            "isolated_cleanup",
            "model_execution",
            "multi_source_exact_mapping",
            "raw_capture_ingestion",
            "shared_npu_lease",
        ),
        "artifacts": ("network_conformance_report", "shared_npu_lease_inventory"),
    },
}

COMMON_COLLECTION_BINDINGS = (
    "clean_validator",
    "correctness_validator",
    "device_lease_runner",
    "model_run_spec",
)
SOURCE_COLLECTION_BINDINGS = COMMON_COLLECTION_BINDINGS + (
    "identity_provider",
    "raw_capture_plugin",
    "source_mapping_provider",
    "source_materializer",
)
ROUND_COLLECTION_BINDINGS = (
    "profile_launcher",
    "round_action_manifest",
)
REORDER_COLLECTION_BINDINGS = SOURCE_COLLECTION_BINDINGS + (
    "dependency_provider",
    "four_profile_launcher",
    "reorder_materializer",
)
COMPONENT_REORDER_COLLECTION_BINDINGS = COMMON_COLLECTION_BINDINGS + (
    "component_capture_provider",
    "component_reorder_materializer",
    "component_source_mapping_provider",
    "dependency_provider",
    "four_profile_launcher",
    "identity_provider",
    "raw_capture_plugin",
)
EVENT_COLLECTION_BINDINGS = SOURCE_COLLECTION_BINDINGS + (
    "critical_path_provider",
    "dependency_provider",
    "event_stage_dispatch_provider",
    "four_profile_launcher",
    "logical_graph_provider",
)
COLLECTION_BINDING_REQUIREMENTS = {
    "option": COMMON_COLLECTION_BINDINGS,
    "scope_split": SOURCE_COLLECTION_BINDINGS,
    "range_exclusion": SOURCE_COLLECTION_BINDINGS,
    "dependency_safe_operator_reorder": REORDER_COLLECTION_BINDINGS,
    "component_overlap_reorder": COMPONENT_REORDER_COLLECTION_BINDINGS,
    "event_edge_refinement": EVENT_COLLECTION_BINDINGS + ("event_materializer",),
    "stage_split": EVENT_COLLECTION_BINDINGS + ("stage_materializer",),
    "scope_event_derivative": SOURCE_COLLECTION_BINDINGS
    + (
        "critical_path_provider",
        "derivative_materializer",
        "event_stage_dispatch_provider",
        "four_profile_launcher",
        "join_validation_provider",
        "logical_graph_provider",
        "parent_trial_evidence",
    ),
}
COMMON_EXPECTED_OUTPUTS = (
    "clean_performance_summary",
    "correctness_summary",
)
SOURCE_EXPECTED_OUTPUTS = COMMON_EXPECTED_OUTPUTS + (
    "baseline_profile_manifest",
    "candidate_profile_manifest",
    "identity_registry",
    "profile_owned_sk_meta",
    "profiling_analysis_result",
    "source_scope_map",
)
REORDER_EXPECTED_OUTPUTS = SOURCE_EXPECTED_OUTPUTS + (
    "bound_short_trace_analysis",
    "dependency_evidence",
    "four_profile_summary",
    "operator_order_capture",
    "post_reorder_dispatch_evidence",
)
COMPONENT_REORDER_EXPECTED_OUTPUTS = SOURCE_EXPECTED_OUTPUTS + (
    "bound_component_capture",
    "component_source_map",
    "dependency_evidence",
    "four_profile_summary",
    "post_component_dispatch_evidence",
)
EVENT_EXPECTED_OUTPUTS = SOURCE_EXPECTED_OUTPUTS + (
    "bound_short_trace_analysis",
    "critical_path_analysis",
    "dependency_evidence",
    "event_stage_dispatch_evidence",
    "four_profile_summary",
    "join_mechanism_validation",
    "logical_graph_capture",
)
COLLECTION_OUTPUT_REQUIREMENTS = {
    "option": COMMON_EXPECTED_OUTPUTS,
    "scope_split": SOURCE_EXPECTED_OUTPUTS,
    "range_exclusion": SOURCE_EXPECTED_OUTPUTS,
    "dependency_safe_operator_reorder": REORDER_EXPECTED_OUTPUTS,
    "component_overlap_reorder": COMPONENT_REORDER_EXPECTED_OUTPUTS,
    "event_edge_refinement": EVENT_EXPECTED_OUTPUTS,
    "stage_split": EVENT_EXPECTED_OUTPUTS,
    "scope_event_derivative": SOURCE_EXPECTED_OUTPUTS
    + (
        "critical_path_analysis",
        "event_stage_dispatch_evidence",
        "four_profile_summary",
        "join_mechanism_validation",
        "logical_graph_capture",
        "parent_trial_clean_evidence",
    ),
}
ROUND_EXPECTED_OUTPUTS = (
    "baseline_profile_manifest",
    "candidate_profile_manifest",
    "profile_owned_sk_meta",
    "profiling_analysis_result",
)
COLLECTION_BINDING_ROLES = tuple(
    sorted(
        {role for roles in COLLECTION_BINDING_REQUIREMENTS.values() for role in roles}
        | set(ROUND_COLLECTION_BINDINGS)
    )
)
COLLECTION_OUTPUTS = tuple(
    sorted(
        {
            output
            for outputs in COLLECTION_OUTPUT_REQUIREMENTS.values()
            for output in outputs
        }
        | set(ROUND_EXPECTED_OUTPUTS)
    )
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


def file_fingerprint(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _sha256_text(value, label):
    value = _text(value, label)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a sha256 fingerprint")
    return value


def _relative(value, label):
    value = _text(value, label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _rooted(root, relative, label):
    root = Path(root).resolve()
    path = (root / _relative(relative, label)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_artifact_semantics(kind, value):
    if kind == "network_conformance_report" and (
        value.get("decision") != "pass"
        or value.get("core_adapter_identity_leaks") != []
    ):
        raise ValueError("network conformance artifact is not passed")
    if kind == "critical_path_npu_closure_receipt":
        outcome = value.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("result_status") not in {
            "accepted",
            "no_gain",
        }:
            raise ValueError("critical path NPU closure has no valid executed outcome")
        if (
            value.get("action_closed") is not True
            or value.get("four_profile_closed") is not True
        ):
            raise ValueError(
                "critical path NPU closure lacks action/four-profile closure"
            )
    if kind == "real_npu_closure_receipt":
        outcome = value.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("result_status") not in {
            "accepted",
            "no_gain",
        }:
            raise ValueError("real NPU closure receipt has no valid executed outcome")
    if kind == "resource_screening_receipt" and (
        value.get("decision") != "no_reorder_candidate"
        or value.get("stable_parent_match_count") != 0
    ):
        raise ValueError("resource screening artifact does not prove its blocker")
    if kind == "cleanup_evaluation":
        acceptance = value.get("acceptance")
        if not isinstance(acceptance, dict) or any(
            acceptance.get(field) != "passed"
            for field in (
                "failure_injection_recovery",
                "idempotent_replay",
                "incumbent_unchanged",
            )
        ):
            raise ValueError("cleanup evaluation is not passed")
    if kind == "shared_npu_lease_inventory" and (
        value.get("status") != "passed" or value.get("blockers") != []
    ):
        raise ValueError("shared NPU lease inventory is not passed")


def validate_adapter(value, root, *, require_fingerprint=True):
    required = {
        "schema_version",
        "adapter_id",
        "network_class",
        "artifacts",
        "capabilities",
    }
    if require_fingerprint:
        required.add("adapter_fingerprint")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"capability adapter must contain exactly {sorted(required)}")
    schema = value["schema_version"]
    if schema not in {ADAPTER_SCHEMA, ADAPTER_SCHEMA_V2, ADAPTER_SCHEMA_V3}:
        raise ValueError(
            f"capability adapter must use {ADAPTER_SCHEMA}, {ADAPTER_SCHEMA_V2}, or {ADAPTER_SCHEMA_V3}"
        )
    capability_catalog = (
        CAPABILITIES_V3
        if schema == ADAPTER_SCHEMA_V3
        else CAPABILITIES_V2
        if schema == ADAPTER_SCHEMA_V2
        else CAPABILITIES
    )
    adapter_id = _text(value["adapter_id"], "adapter_id")
    network_class = _text(value["network_class"], "network_class")
    artifacts = []
    artifact_kinds = set()
    for index, item in enumerate(value["artifacts"]):
        label = f"artifacts[{index}]"
        fields = {"kind", "path", "schema_version", "file_fingerprint"}
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f"{label} fields are invalid")
        kind = item["kind"]
        if kind not in ARTIFACT_SCHEMAS or kind in artifact_kinds:
            raise ValueError(f"{label}.kind is unknown or duplicated")
        path = _rooted(root, item["path"], f"{label}.path")
        artifact = _load(path)
        expected_schema = ARTIFACT_SCHEMAS[kind]
        actual = file_fingerprint(path)
        if artifact.get("schema_version") != expected_schema:
            raise ValueError(f"{label} content must use {expected_schema}")
        if (
            item["schema_version"] != expected_schema
            or item["file_fingerprint"] != actual
        ):
            raise ValueError(f"{label} sealed identity mismatch")
        _validate_artifact_semantics(kind, artifact)
        artifact_kinds.add(kind)
        artifacts.append(
            {
                "kind": kind,
                "path": item["path"],
                "schema_version": expected_schema,
                "file_fingerprint": actual,
            }
        )
    if not artifacts:
        raise ValueError("capability adapter requires artifact evidence")
    if [item["kind"] for item in artifacts] != sorted(artifact_kinds):
        raise ValueError("artifacts must be sorted by kind")

    capabilities = value["capabilities"]
    if not isinstance(capabilities, list) or [
        item.get("name") for item in capabilities if isinstance(item, dict)
    ] != list(capability_catalog):
        raise ValueError(
            "capabilities must declare the complete ordered capability catalog"
        )
    normalized_capabilities = []
    for index, item in enumerate(capabilities):
        label = f"capabilities[{index}]"
        fields = {"name", "status", "evidence_artifacts", "blocker"}
        if set(item) != fields or item["status"] not in {"supported", "unsupported"}:
            raise ValueError(f"{label} fields or status are invalid")
        evidence = item["evidence_artifacts"]
        if (
            not isinstance(evidence, list)
            or evidence != sorted(set(evidence))
            or set(evidence) - artifact_kinds
        ):
            raise ValueError(f"{label}.evidence_artifacts are invalid")
        if item["status"] == "supported":
            if not evidence or item["blocker"] is not None:
                raise ValueError(
                    f"{label} supported capability requires evidence and no blocker"
                )
        else:
            blocker = item["blocker"]
            if not isinstance(blocker, dict) or set(blocker) != {"code", "detail"}:
                raise ValueError(
                    f"{label} unsupported capability requires a structured blocker"
                )
            _text(blocker["code"], f"{label}.blocker.code")
            _text(blocker["detail"], f"{label}.blocker.detail")
        normalized_capabilities.append(dict(item))
    normalized = {
        "schema_version": schema,
        "adapter_id": adapter_id,
        "network_class": network_class,
        "artifacts": artifacts,
        "capabilities": normalized_capabilities,
    }
    normalized["adapter_fingerprint"] = fingerprint(normalized)
    if (
        require_fingerprint
        and value["adapter_fingerprint"] != normalized["adapter_fingerprint"]
    ):
        raise ValueError("adapter_fingerprint mismatch")
    return normalized


def freeze_adapter(draft, root):
    if "adapter_fingerprint" in draft:
        raise ValueError("adapter draft must not contain adapter_fingerprint")
    prepared = dict(draft)
    artifacts = []
    for index, item in enumerate(draft.get("artifacts", [])):
        if not isinstance(item, dict) or set(item) != {"kind", "path"}:
            raise ValueError(
                f"adapter draft artifacts[{index}] must contain kind and path"
            )
        kind = item["kind"]
        if kind not in ARTIFACT_SCHEMAS:
            raise ValueError(f"adapter draft artifacts[{index}].kind is unknown")
        path = _rooted(root, item["path"], f"artifacts[{index}].path")
        artifacts.append(
            {
                **item,
                "schema_version": ARTIFACT_SCHEMAS[kind],
                "file_fingerprint": file_fingerprint(path),
            }
        )
    prepared["artifacts"] = artifacts
    return validate_adapter(prepared, root, require_fingerprint=False)


def _action_support(adapter):
    capabilities = {item["name"]: item for item in adapter["capabilities"]}
    artifacts = {item["kind"] for item in adapter["artifacts"]}
    resource_screened = "resource_screening_receipt" in artifacts
    actions = []
    requirements_catalog = (
        ACTION_REQUIREMENTS_V3
        if adapter["schema_version"] == ADAPTER_SCHEMA_V3
        else ACTION_REQUIREMENTS_V2
        if adapter["schema_version"] == ADAPTER_SCHEMA_V2
        else ACTION_REQUIREMENTS
    )
    for action, requirements in requirements_catalog.items():
        blockers = []
        for name in requirements["capabilities"]:
            capability = capabilities[name]
            if capability["status"] != "supported":
                blockers.append(
                    {
                        "type": "capability",
                        "name": name,
                        "code": capability["blocker"]["code"],
                        "detail": capability["blocker"]["detail"],
                    }
                )
        for kind in requirements["artifacts"]:
            if kind not in artifacts:
                blockers.append(
                    {
                        "type": "artifact",
                        "name": kind,
                        "code": "MISSING_REQUIRED_ARTIFACT",
                        "detail": f"adapter has no sealed {kind}",
                    }
                )
        if action == "dependency_safe_operator_reorder" and resource_screened:
            blockers.append(
                {
                    "type": "current_evidence",
                    "name": "stable_same_parent_pair",
                    "code": "NO_STABLE_SAME_PARENT_PAIR",
                    "detail": "resource screening proves zero stable same-parent Cube/Vector matches",
                }
            )
        actions.append(
            {
                "action": action,
                "status": "available" if not blockers else "blocked",
                "required_capabilities": list(requirements["capabilities"]),
                "required_artifacts": list(requirements["artifacts"]),
                "blockers": blockers,
            }
        )
    return actions


def build_matrix(matrix_id, adapter_paths, root):
    matrix_id = _text(matrix_id, "matrix_id")
    if not isinstance(adapter_paths, list) or not adapter_paths:
        raise ValueError("support matrix requires at least one adapter")
    records = []
    schemas = set()
    identities = set()
    for index, relative in enumerate(adapter_paths):
        path = _rooted(root, relative, f"adapters[{index}]")
        adapter = validate_adapter(_load(path), root)
        if adapter["adapter_id"] in identities:
            raise ValueError("support matrix contains duplicate adapter_id")
        identities.add(adapter["adapter_id"])
        schemas.add(adapter["schema_version"])
        records.append(
            {
                "path": relative,
                "file_fingerprint": file_fingerprint(path),
                "adapter_id": adapter["adapter_id"],
                "network_class": adapter["network_class"],
                "adapter_fingerprint": adapter["adapter_fingerprint"],
                "capabilities": adapter["capabilities"],
                "actions": _action_support(adapter),
            }
        )
    matrix_schema = (
        MATRIX_SCHEMA_V3
        if ADAPTER_SCHEMA_V3 in schemas
        else MATRIX_SCHEMA_V2
        if ADAPTER_SCHEMA_V2 in schemas
        else MATRIX_SCHEMA
    )
    requirements = (
        ACTION_REQUIREMENTS_V3
        if matrix_schema == MATRIX_SCHEMA_V3
        else ACTION_REQUIREMENTS_V2
        if matrix_schema == MATRIX_SCHEMA_V2
        else ACTION_REQUIREMENTS
    )
    matrix = {
        "schema_version": matrix_schema,
        "matrix_id": matrix_id,
        "availability_scope": "adapter_capability_not_candidate_recommendation",
        "action_requirements_fingerprint": fingerprint(requirements),
        "adapters": records,
    }
    matrix["matrix_fingerprint"] = fingerprint(matrix)
    return matrix


def validate_matrix(matrix, root):
    required = {
        "schema_version",
        "matrix_id",
        "availability_scope",
        "action_requirements_fingerprint",
        "adapters",
        "matrix_fingerprint",
    }
    if (
        not isinstance(matrix, dict)
        or set(matrix) != required
        or matrix.get("schema_version")
        not in {MATRIX_SCHEMA, MATRIX_SCHEMA_V2, MATRIX_SCHEMA_V3}
    ):
        raise ValueError(
            f"support matrix must use {MATRIX_SCHEMA}, {MATRIX_SCHEMA_V2}, or {MATRIX_SCHEMA_V3} with exact fields"
        )
    paths = [
        item.get("path")
        for item in matrix.get("adapters", [])
        if isinstance(item, dict)
    ]
    actual = build_matrix(matrix["matrix_id"], paths, root)
    if matrix != actual:
        raise ValueError("support matrix differs from deterministic adapter replay")
    return {
        "valid": True,
        "adapter_count": len(matrix["adapters"]),
        "matrix_fingerprint": matrix["matrix_fingerprint"],
    }


def _validate_round_actions(round_kind, requested_actions):
    round_kind = _text(round_kind, "collection plan round_kind")
    if round_kind not in ROUND_KINDS:
        raise ValueError(f"collection plan round_kind must be one of {ROUND_KINDS}")
    actions = set(requested_actions)
    source_actions = {"range_exclusion", "scope_split"}
    if round_kind in {"P", "FINAL"} and (not actions or not actions <= source_actions):
        raise ValueError(
            f"{round_kind} round requires one or more source actions and forbids option action"
        )
    return round_kind


def _validate_identity_artifact(role, value, normalized):
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object")
    expectations = {
        "config_identity": {"config_fingerprint": normalized["config_fingerprint"]},
        "control_identity": {
            "control_fingerprint": normalized["control_fingerprint"],
            "round_kind": normalized["round_kind"],
        },
        "source_identity": {
            "source_revision": normalized["source_revision"],
            "source_fingerprint": normalized["source_fingerprint"],
        },
        "workload_identity": {
            "workload_fingerprint": normalized["workload_fingerprint"]
        },
    }
    for name, expected in expectations[role].items():
        if value.get(name) != expected:
            raise ValueError(f"{role}.{name} does not match collection plan")


def validate_collection_plan(plan, root, *, require_fingerprint=True):
    required = {
        "schema_version",
        "plan_id",
        "adapter_id",
        "round_kind",
        "requested_actions",
        "source_revision",
        "source_fingerprint",
        "config_fingerprint",
        "control_fingerprint",
        "workload_fingerprint",
        "experiment_artifacts",
        "bindings",
        "expected_outputs",
    }
    if require_fingerprint:
        required.add("collection_plan_fingerprint")
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError(f"collection plan must contain exactly {sorted(required)}")
    if plan.get("schema_version") != COLLECTION_PLAN_SCHEMA:
        raise ValueError(f"collection plan must use {COLLECTION_PLAN_SCHEMA}")

    requested_actions = plan.get("requested_actions")
    known_actions = set(ACTION_REQUIREMENTS_V3)
    if (
        not isinstance(requested_actions, list)
        or not requested_actions
        or requested_actions != sorted(set(requested_actions))
        or set(requested_actions) - known_actions
    ):
        raise ValueError(
            "collection plan requested_actions must be sorted unique known actions"
        )
    round_kind = _validate_round_actions(plan.get("round_kind"), requested_actions)

    normalized = {
        "schema_version": COLLECTION_PLAN_SCHEMA,
        "plan_id": _text(plan.get("plan_id"), "collection plan plan_id"),
        "adapter_id": _text(plan.get("adapter_id"), "collection plan adapter_id"),
        "round_kind": round_kind,
        "source_revision": _text(
            plan.get("source_revision"), "collection plan source_revision"
        ),
        "source_fingerprint": _sha256_text(
            plan.get("source_fingerprint"), "collection plan source_fingerprint"
        ),
        "config_fingerprint": _sha256_text(
            plan.get("config_fingerprint"), "collection plan config_fingerprint"
        ),
        "control_fingerprint": _sha256_text(
            plan.get("control_fingerprint"), "collection plan control_fingerprint"
        ),
        "workload_fingerprint": _sha256_text(
            plan.get("workload_fingerprint"), "collection plan workload_fingerprint"
        ),
        "requested_actions": requested_actions,
    }

    experiment_artifacts = plan.get("experiment_artifacts")
    if not isinstance(experiment_artifacts, list):
        raise ValueError("collection plan experiment_artifacts must be a list")
    normalized_artifacts = []
    seen_artifact_roles = set()
    for index, item in enumerate(experiment_artifacts):
        label = f"collection plan experiment_artifacts[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "role",
            "path",
            "file_fingerprint",
        }:
            raise ValueError(f"{label} fields are invalid")
        role = _text(item["role"], f"{label}.role")
        if role not in EXPERIMENT_IDENTITY_ROLES or role in seen_artifact_roles:
            raise ValueError(f"{label}.role is unknown or duplicated")
        path = _rooted(root, item["path"], f"{label}.path")
        actual = file_fingerprint(path)
        if item["file_fingerprint"] != actual:
            raise ValueError(f"{label}.file_fingerprint mismatch")
        _validate_identity_artifact(role, _load(path), normalized)
        seen_artifact_roles.add(role)
        normalized_artifacts.append(
            {
                "role": role,
                "path": item["path"],
                "file_fingerprint": actual,
            }
        )
    if seen_artifact_roles != set(EXPERIMENT_IDENTITY_ROLES):
        raise ValueError("collection plan must bind all experiment identity artifacts")
    if [item["role"] for item in normalized_artifacts] != sorted(seen_artifact_roles):
        raise ValueError("collection plan experiment_artifacts must be sorted by role")

    bindings = plan.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("collection plan bindings must be a list")
    normalized_bindings = []
    seen_roles = set()
    for index, item in enumerate(bindings):
        label = f"collection plan bindings[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "role",
            "path",
            "file_fingerprint",
        }:
            raise ValueError(f"{label} fields are invalid")
        role = _text(item["role"], f"{label}.role")
        if role not in COLLECTION_BINDING_ROLES or role in seen_roles:
            raise ValueError(f"{label}.role is unknown or duplicated")
        path = _rooted(root, item["path"], f"{label}.path")
        actual = file_fingerprint(path)
        if item["file_fingerprint"] != actual:
            raise ValueError(f"{label}.file_fingerprint mismatch")
        seen_roles.add(role)
        normalized_bindings.append(
            {
                "role": role,
                "path": item["path"],
                "file_fingerprint": actual,
            }
        )
    if [item["role"] for item in normalized_bindings] != sorted(seen_roles):
        raise ValueError("collection plan bindings must be sorted by role")

    expected_outputs = plan.get("expected_outputs")
    if (
        not isinstance(expected_outputs, list)
        or expected_outputs != sorted(set(expected_outputs))
        or set(expected_outputs) - set(COLLECTION_OUTPUTS)
    ):
        raise ValueError(
            "collection plan expected_outputs must be sorted unique known outputs"
        )

    normalized.update(
        {
            "experiment_artifacts": normalized_artifacts,
            "bindings": normalized_bindings,
            "expected_outputs": expected_outputs,
        }
    )
    normalized["collection_plan_fingerprint"] = fingerprint(normalized)
    if (
        require_fingerprint
        and plan["collection_plan_fingerprint"]
        != normalized["collection_plan_fingerprint"]
    ):
        raise ValueError("collection_plan_fingerprint mismatch")
    return normalized


def freeze_collection_plan(draft, root):
    if "collection_plan_fingerprint" in draft:
        raise ValueError(
            "collection plan draft must not contain collection_plan_fingerprint"
        )
    prepared = dict(draft)
    for field in ("experiment_artifacts", "bindings"):
        prepared_items = []
        for index, item in enumerate(draft.get(field, [])):
            if not isinstance(item, dict) or set(item) != {"role", "path"}:
                raise ValueError(
                    f"collection plan draft {field}[{index}] must contain role and path"
                )
            path = _rooted(root, item["path"], f"{field}[{index}].path")
            prepared_items.append({**item, "file_fingerprint": file_fingerprint(path)})
        prepared[field] = prepared_items
    return validate_collection_plan(prepared, root, require_fingerprint=False)


def _binding_semantic_blockers(plan, root):
    blockers = []
    bindings = {item["role"]: item for item in plan["bindings"]}
    if "model_run_spec" in bindings:
        value = _load(
            _rooted(root, bindings["model_run_spec"]["path"], "model_run_spec.path")
        )
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != MODEL_RUN_SPEC_SCHEMA
        ):
            blockers.append(
                {
                    "action": None,
                    "type": "collection_binding",
                    "name": "model_run_spec",
                    "code": "INVALID_COLLECTION_BINDING",
                    "detail": f"model_run_spec must use {MODEL_RUN_SPEC_SCHEMA}",
                }
            )
    if plan["round_kind"] in {"P", "FINAL"} and "round_action_manifest" in bindings:
        value = _load(
            _rooted(
                root,
                bindings["round_action_manifest"]["path"],
                "round_action_manifest.path",
            )
        )
        expected = {
            "schema_version": ROUND_ACTION_MANIFEST_SCHEMA,
            "round_kind": plan["round_kind"],
            "source_revision": plan["source_revision"],
            "requested_actions": plan["requested_actions"],
        }
        if value != expected:
            blockers.append(
                {
                    "action": None,
                    "type": "collection_binding",
                    "name": "round_action_manifest",
                    "code": "INVALID_COLLECTION_BINDING",
                    "detail": "round_action_manifest does not bind the current round/actions/source",
                }
            )
    return blockers


def _safe_receipt_text(value):
    return value if isinstance(value, str) and value.strip() else "unavailable"


def _blocked_preflight(
    preflight_id, matrix, collection_plan, adapter_id, requested_actions, error
):
    actions = (
        sorted(set(requested_actions))
        if isinstance(requested_actions, list)
        and all(isinstance(item, str) for item in requested_actions)
        else []
    )
    message = str(error)
    if "match exactly one matrix adapter" in message:
        code = "ADAPTER_NOT_IN_MATRIX"
    elif "unavailable in this matrix schema" in message:
        code = "ACTION_NOT_IN_MATRIX_SCHEMA"
    elif "collection plan" in message or "does not exist" in message:
        code = "COLLECTION_PLAN_INVALID"
    elif "support matrix" in message or "adapter" in message:
        code = "CAPABILITY_MATRIX_INVALID"
    else:
        code = "PREFLIGHT_INPUT_INVALID"
    result = {
        "schema_version": PREFLIGHT_SCHEMA,
        "preflight_id": _safe_receipt_text(preflight_id),
        "status": "blocked",
        "availability_scope": "adapter_and_run_specific_collection_readiness",
        "matrix_id": _safe_receipt_text(
            matrix.get("matrix_id") if isinstance(matrix, dict) else None
        ),
        "matrix_fingerprint": _safe_receipt_text(
            matrix.get("matrix_fingerprint") if isinstance(matrix, dict) else None
        ),
        "adapter_id": _safe_receipt_text(adapter_id),
        "adapter_fingerprint": "unavailable",
        "collection_plan_id": _safe_receipt_text(
            collection_plan.get("plan_id")
            if isinstance(collection_plan, dict)
            else None
        ),
        "collection_plan_fingerprint": _safe_receipt_text(
            collection_plan.get("collection_plan_fingerprint")
            if isinstance(collection_plan, dict)
            else None
        ),
        "round_kind": _safe_receipt_text(
            collection_plan.get("round_kind")
            if isinstance(collection_plan, dict)
            else None
        ),
        "requested_actions": actions,
        "action_checks": [],
        "blockers": [
            {
                "action": None,
                "type": "preflight_input",
                "name": "preflight_input",
                "code": code,
                "detail": message,
            }
        ],
    }
    result["preflight_fingerprint"] = fingerprint(result)
    return result


def _build_preflight(
    preflight_id, matrix, collection_plan, root, adapter_id, requested_actions
):
    validate_matrix(matrix, root)
    plan = validate_collection_plan(collection_plan, root)
    adapter_id = _text(adapter_id, "adapter_id")
    if (
        not isinstance(requested_actions, list)
        or not requested_actions
        or requested_actions != sorted(set(requested_actions))
    ):
        raise ValueError("requested actions must be sorted and unique")
    matches = [item for item in matrix["adapters"] if item["adapter_id"] == adapter_id]
    if len(matches) != 1:
        raise ValueError("preflight adapter_id must match exactly one matrix adapter")
    adapter = matches[0]
    action_index = {item["action"]: item for item in adapter["actions"]}
    unknown = set(requested_actions) - set(action_index)
    if unknown:
        raise ValueError(
            f"requested actions are unavailable in this matrix schema: {sorted(unknown)}"
        )
    if (
        plan["adapter_id"] != adapter_id
        or plan["requested_actions"] != requested_actions
    ):
        raise ValueError("collection plan adapter/actions do not match preflight")

    blockers = []
    action_checks = []
    for action in requested_actions:
        check = action_index[action]
        action_checks.append(
            {
                "action": action,
                "status": check["status"],
                "required_capabilities": check["required_capabilities"],
                "required_artifacts": check["required_artifacts"],
            }
        )
        blockers.extend({"action": action, **blocker} for blocker in check["blockers"])

    bound_roles = {item["role"] for item in plan["bindings"]}
    declared_outputs = set(plan["expected_outputs"])
    required_roles = set()
    required_outputs = set()
    for action in requested_actions:
        required_roles.update(COLLECTION_BINDING_REQUIREMENTS[action])
        required_outputs.update(COLLECTION_OUTPUT_REQUIREMENTS[action])
    if plan["round_kind"] in {"P", "FINAL"}:
        required_roles.update(ROUND_COLLECTION_BINDINGS)
        required_outputs.update(ROUND_EXPECTED_OUTPUTS)
    for role in sorted(required_roles - bound_roles):
        blockers.append(
            {
                "action": None,
                "type": "collection_binding",
                "name": role,
                "code": "MISSING_COLLECTION_BINDING",
                "detail": f"run-specific collection plan does not bind {role}",
            }
        )
    for output in sorted(required_outputs - declared_outputs):
        blockers.append(
            {
                "action": None,
                "type": "expected_output",
                "name": output,
                "code": "MISSING_EXPECTED_OUTPUT",
                "detail": f"run-specific collection plan does not require {output}",
            }
        )
    blockers.extend(_binding_semantic_blockers(plan, root))

    result = {
        "schema_version": PREFLIGHT_SCHEMA,
        "preflight_id": _text(preflight_id, "preflight_id"),
        "status": "ready" if not blockers else "blocked",
        "availability_scope": "adapter_and_run_specific_collection_readiness",
        "matrix_id": matrix["matrix_id"],
        "matrix_fingerprint": matrix["matrix_fingerprint"],
        "adapter_id": adapter_id,
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "collection_plan_id": plan["plan_id"],
        "collection_plan_fingerprint": plan["collection_plan_fingerprint"],
        "round_kind": plan["round_kind"],
        "requested_actions": requested_actions,
        "action_checks": action_checks,
        "blockers": blockers,
    }
    result["preflight_fingerprint"] = fingerprint(result)
    return result


def build_preflight(
    preflight_id, matrix, collection_plan, root, adapter_id, requested_actions
):
    try:
        return _build_preflight(
            preflight_id, matrix, collection_plan, root, adapter_id, requested_actions
        )
    except ValueError as error:
        return _blocked_preflight(
            preflight_id, matrix, collection_plan, adapter_id, requested_actions, error
        )


def validate_preflight(preflight, matrix, collection_plan, root):
    if not isinstance(preflight, dict):
        raise ValueError("preflight must be an object")
    actual = build_preflight(
        preflight.get("preflight_id"),
        matrix,
        collection_plan,
        root,
        preflight.get("adapter_id"),
        preflight.get("requested_actions"),
    )
    if preflight != actual:
        raise ValueError("preflight differs from deterministic replay")
    return {
        "valid": True,
        "status": preflight["status"],
        "preflight_fingerprint": preflight["preflight_fingerprint"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-adapter")
    freeze.add_argument("--draft", type=Path, required=True)
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--matrix-id", required=True)
    build.add_argument("--adapter", action="append", required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--matrix", type=Path, required=True)
    validate.add_argument("--root", type=Path, required=True)
    freeze_plan = commands.add_parser("freeze-collection-plan")
    freeze_plan.add_argument("--draft", type=Path, required=True)
    freeze_plan.add_argument("--root", type=Path, required=True)
    freeze_plan.add_argument("--out", type=Path, required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--preflight-id", required=True)
    preflight.add_argument("--matrix", type=Path, required=True)
    preflight.add_argument("--collection-plan", type=Path, required=True)
    preflight.add_argument("--adapter-id", required=True)
    preflight.add_argument("--action", action="append", required=True)
    preflight.add_argument("--root", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)
    validate_preflight_parser = commands.add_parser("validate-preflight")
    validate_preflight_parser.add_argument("--preflight", type=Path, required=True)
    validate_preflight_parser.add_argument("--matrix", type=Path, required=True)
    validate_preflight_parser.add_argument(
        "--collection-plan", type=Path, required=True
    )
    validate_preflight_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    exit_code = 0
    try:
        if args.command == "freeze-adapter":
            result = freeze_adapter(_load(args.draft), args.root)
            output = args.out
        elif args.command == "build":
            result = build_matrix(args.matrix_id, args.adapter, args.root)
            output = args.out
        elif args.command == "validate":
            result = validate_matrix(_load(args.matrix), args.root)
            output = None
        elif args.command == "freeze-collection-plan":
            result = freeze_collection_plan(_load(args.draft), args.root)
            output = args.out
        elif args.command == "preflight":
            result = build_preflight(
                args.preflight_id,
                _load(args.matrix),
                _load(args.collection_plan),
                args.root,
                args.adapter_id,
                sorted(set(args.action)),
            )
            output = args.out
            if result["status"] != "ready":
                exit_code = 2
        else:
            result = validate_preflight(
                _load(args.preflight),
                _load(args.matrix),
                _load(args.collection_plan),
                args.root,
            )
            output = None
        if output:
            if output.exists():
                raise ValueError(f"output already exists: {output}")
            _atomic(output, result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
