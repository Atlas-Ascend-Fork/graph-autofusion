#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Persist and validate provider-neutral SuperKernel Auto Tune sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SESSION_SCHEMA = "superkernel-auto-tune-session-v1"
RESULT_SCHEMA = "superkernel-auto-tune-phase-result-v1"
TASK_SCHEMA = "superkernel-auto-tune-phase-task-v1"
RECEIPT_SCHEMA = "superkernel-auto-tune-dispatch-receipt-v1"
EVIDENCE_IMPORT_SCHEMA = "superkernel-auto-tune-evidence-import-v1"
OPTIONAL_MODES = {"none", "multistream", "source-range", "both"}
ENTRYPOINTS = {"full", "optional-from-base", "source-range-from-smap"}
RESULT_STATUSES = {
    "succeeded",
    "accepted",
    "no_gain",
    "blocked",
    "failed",
    "invalid",
    "not_requested",
    "not_run",
}
FINAL_DETAIL_KEYS = (
    "environment",
    "s0",
    "stage_a",
    "stage_o",
    "base_profile_source_mapping",
    "optional_experiments",
    "final_e2e",
)
STEP_IDS = re.compile(r"^[a-z][a-z0-9-]{1,63}$")

PHASES = {
    "intake_preparation": (
        "sk-intake-preparation",
        "superkernel-intake-preparation",
    ),
    "s0_baseline": ("sk-s0-baseline", "superkernel-s0-baseline"),
    "stage_a_scope_selection": (
        "sk-stage-a-scope-selection",
        "superkernel-stage-a-scope-selection",
    ),
    "stage_o_option_tuning": (
        "sk-stage-o-option-tuning",
        "superkernel-stage-o-option-tuning",
    ),
    "base_profile_source_mapping": (
        "sk-base-profile-source-mapping",
        "superkernel-base-profile-source-mapping",
    ),
    "optional_experiments": (
        "sk-optional-experiments",
        "superkernel-optional-experiments",
    ),
    "final_e2e_report": (
        "sk-final-e2e-report",
        "superkernel-final-e2e-report",
    ),
}

PHASE_STATUSES = {
    "intake_preparation": {"succeeded", "blocked", "failed", "invalid"},
    "s0_baseline": {"succeeded", "blocked", "failed", "invalid"},
    "stage_a_scope_selection": {"accepted", "no_gain", "blocked", "failed", "invalid"},
    "stage_o_option_tuning": {"accepted", "no_gain", "blocked", "failed", "invalid"},
    "base_profile_source_mapping": {
        "succeeded",
        "no_gain",
        "blocked",
        "failed",
        "invalid",
    },
    "final_e2e_report": {
        "accepted",
        "no_gain",
        "blocked",
        "failed",
        "invalid",
        "not_run",
    },
}


def _allowed_statuses(step):
    if step["phase"] != "optional_experiments":
        return PHASE_STATUSES[step["phase"]]
    branch = step.get("optional_branch")
    if branch == "none":
        return {"not_requested"}
    if branch == "multistream":
        return {"accepted", "no_gain", "blocked", "failed", "invalid"}
    if branch == "source-range":
        return {"accepted", "no_gain", "blocked", "failed", "invalid", "not_run"}
    raise ValueError("optional step branch is invalid")


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _require_string(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _relative_artifact(value, name):
    value = _require_string(value, name)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{name} must be a relative artifact path")
    return value


def _load_ledger(artifact_root, relative_path):
    relative_path = _relative_artifact(relative_path, "ledger_path")
    path = Path(artifact_root) / relative_path
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"ledger_path cannot be read: {error}") from error
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 2:
        raise ValueError("ledger_path must reference a schema 2 ledger")
    if not isinstance(ledger.get("experiments"), dict):
        raise ValueError("ledger_path schema 2 ledger experiments must be an object")
    for experiment_id, experiment in ledger["experiments"].items():
        _require_string(experiment_id, "ledger experiment_id")
        if not isinstance(experiment, dict):
            raise ValueError("ledger experiment must be an object")
        if not isinstance(experiment.get("rounds", []), list):
            raise ValueError("ledger experiment rounds must be a list")
        if not isinstance(experiment.get("blockers", []), list):
            raise ValueError("ledger experiment blockers must be a list")
    _validate_final_e2e(ledger.get("final_e2e"))
    if "report_summary" in ledger:
        _validate_report_summary(
            ledger["report_summary"], ledger["final_e2e"], artifact_root
        )
    return relative_path, ledger


def _finite_number(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"final_e2e {name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"final_e2e {name} must be positive")
    return float(value)


def _validate_metric_set(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"final_e2e {name} must be an object")
    _finite_number(value.get("median_ms"), f"{name}.median_ms", positive=True)
    for optional in ("p90_ms", "stddev_ms"):
        if optional in value:
            _finite_number(
                value[optional], f"{name}.{optional}", positive=optional == "p90_ms"
            )
    for count_key in ("sample_count", "run_count"):
        if count_key in value and (
            isinstance(value[count_key], bool)
            or not isinstance(value[count_key], int)
            or value[count_key] <= 0
        ):
            raise ValueError(f"final_e2e {name}.{count_key} must be a positive integer")


def _validate_final_e2e(value):
    if not isinstance(value, dict):
        raise ValueError("ledger final_e2e must be an object")
    classification = value.get("classification")
    if classification not in {"beneficial", "no_gain", "not_run", "failed", "blocked"}:
        raise ValueError("ledger final_e2e classification is invalid")
    for key in ("candidate_id", "scope_strategy", "reason_zh"):
        _require_string(value.get(key), f"final_e2e {key}")
    if not isinstance(value.get("option_config"), dict):
        raise ValueError("final_e2e option_config must be an object")
    evidence = value.get("evidence_artifacts")
    if not isinstance(evidence, list):
        raise ValueError("final_e2e evidence_artifacts must be a list")
    for index, artifact in enumerate(evidence):
        _relative_artifact(artifact, f"final_e2e evidence_artifacts[{index}]")
    if classification in {"beneficial", "no_gain"}:
        _validate_metric_set(value.get("baseline"), "baseline")
        _validate_metric_set(value.get("candidate"), "candidate")
        declared = _finite_number(value.get("improvement_pct"), "improvement_pct")
        baseline = float(value["baseline"]["median_ms"])
        candidate = float(value["candidate"]["median_ms"])
        calculated = (baseline - candidate) / baseline * 100.0
        if not math.isclose(declared, calculated, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "final_e2e improvement_pct does not match clean E2E medians"
            )
        if classification == "beneficial" and declared <= 0:
            raise ValueError(
                "final_e2e beneficial classification requires positive improvement"
            )
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            )
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _step(step_id, phase, *, optional_branch=None, derived_family_rebase=False):
    agent_id, skill = PHASES[phase]
    return {
        "step_id": step_id,
        "phase": phase,
        "agent_id": agent_id,
        "skill": skill,
        "optional_branch": optional_branch,
        "derived_family_rebase": derived_family_rebase,
        "state": "pending",
    }


def _canonical_steps(optional_mode, *, entrypoint="full", multistream_accepted=False):
    if entrypoint == "full":
        steps = [
            _step("intake-preparation", "intake_preparation"),
            _step("s0-baseline", "s0_baseline"),
            _step("stage-a-scope-selection", "stage_a_scope_selection"),
            _step("stage-o-option-tuning", "stage_o_option_tuning"),
            _step("base-profile-source-mapping", "base_profile_source_mapping"),
        ]
    elif entrypoint == "source-range-from-smap":
        if optional_mode != "source-range":
            raise ValueError(
                "source-range-from-smap requires optional_mode source-range"
            )
        return [
            _step(
                "optional-source-range",
                "optional_experiments",
                optional_branch="source-range",
            ),
            _step("final-e2e-report", "final_e2e_report"),
        ]
    elif entrypoint == "optional-from-base":
        if optional_mode not in {"multistream", "both"}:
            raise ValueError("optional-from-base requires multistream or both")
        steps = []
    else:
        raise ValueError("session entrypoint is invalid")

    if optional_mode == "both":
        steps.extend(
            [
                _step(
                    "optional-multistream",
                    "optional_experiments",
                    optional_branch="multistream",
                ),
            ]
        )
        if multistream_accepted:
            steps.append(
                _step(
                    "base-profile-derived",
                    "base_profile_source_mapping",
                    derived_family_rebase=True,
                )
            )
        steps.append(
            _step(
                "optional-source-range",
                "optional_experiments",
                optional_branch="source-range",
            )
        )
    elif entrypoint == "full":
        steps.append(
            _step(
                "optional-experiments",
                "optional_experiments",
                optional_branch=optional_mode,
            )
        )
    else:
        steps.append(
            _step(
                "optional-multistream",
                "optional_experiments",
                optional_branch="multistream",
            )
        )
    steps.append(_step("final-e2e-report", "final_e2e_report"))
    return steps


def _initial_steps(optional_mode):
    return _canonical_steps(optional_mode, entrypoint="full")


def _validate_canonical_schedule(session):
    multistream = next(
        (
            step
            for step in session["steps"]
            if step.get("step_id") == "optional-multistream"
        ),
        None,
    )
    multistream_accepted = bool(
        multistream
        and multistream.get("state") == "sealed"
        and multistream.get("outcome_status") == "accepted"
    )
    expected = _canonical_steps(
        session["optional_mode"],
        entrypoint=session.get("entrypoint", "full"),
        multistream_accepted=multistream_accepted,
    )
    if len(expected) != len(session["steps"]):
        raise ValueError("session steps do not match the canonical schedule")
    keys = (
        "step_id",
        "phase",
        "agent_id",
        "skill",
        "optional_branch",
        "derived_family_rebase",
    )
    for actual, wanted in zip(session["steps"], expected):
        if any(actual.get(key) != wanted.get(key) for key in keys):
            raise ValueError("session steps do not match the canonical schedule")


def initialize_session(*, session_path, artifact_root, session_id, optional_mode):
    _require_string(session_id, "session_id")
    if optional_mode not in OPTIONAL_MODES:
        raise ValueError(f"optional_mode must be one of {sorted(OPTIONAL_MODES)}")
    session_path = Path(session_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    if session_path.exists():
        raise ValueError(f"session already exists: {session_path}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    session = {
        "schema_version": SESSION_SCHEMA,
        "session_id": session_id,
        "artifact_root": str(artifact_root),
        "optional_mode": optional_mode,
        "entrypoint": "full",
        "status": "active",
        "created_at": _now(),
        "steps": _initial_steps(optional_mode),
    }
    _atomic_json(session_path, session)
    return session


def _path_within(root, value, label):
    root = Path(root).resolve()
    try:
        path = Path(value).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside the parent artifact root") from error
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path, str(relative)


def _artifact_descriptor(root, value, label):
    path, relative = _path_within(root, value, label)
    return {
        "path": relative,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_object(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _analysis_fingerprint(value):
    unsigned = dict(value)
    unsigned.pop("analysis_content_fingerprint", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_import_analysis(path):
    analysis = _read_object(path, "profiling analysis")
    if analysis.get("schema_version") != "1.2":
        raise ValueError("profiling analysis must use schema_version 1.2")
    for key in ("experiment_id", "round_id", "candidate_name", "source_revision"):
        _require_string(analysis.get(key), f"profiling analysis {key}")
    declared = _require_string(
        analysis.get("analysis_content_fingerprint"),
        "profiling analysis analysis_content_fingerprint",
    )
    if declared != _analysis_fingerprint(analysis):
        raise ValueError("profiling analysis content fingerprint mismatch")
    source_mapping = analysis.get("source_scope_mapping")
    if source_mapping is not None and (
        not isinstance(source_mapping, dict)
        or source_mapping.get("protocol") != "source_scope_map_v2"
        or source_mapping.get("status") != "exact"
    ):
        raise ValueError("profiling analysis source_scope_mapping is not exact")
    return analysis


def _validate_import_ledger(path, analysis):
    ledger = _read_object(path, "experiment ledger")
    if ledger.get("schema_version") != 2 or not isinstance(
        ledger.get("experiments"), dict
    ):
        raise ValueError("experiment ledger must be a schema 2 ledger")
    experiment = ledger["experiments"].get(analysis["experiment_id"])
    if not isinstance(experiment, dict):
        raise ValueError("experiment ledger does not contain the analyzed experiment")
    if experiment.get("source_revision") != analysis["source_revision"]:
        raise ValueError(
            "experiment ledger source revision differs from profiling analysis"
        )
    rounds = experiment.get("rounds")
    if not isinstance(rounds, list) or not any(
        isinstance(item, dict) and item.get("round_id") == analysis["round_id"]
        for item in rounds
    ):
        raise ValueError("experiment ledger does not contain the analyzed BASE round")
    return ledger


def _source_map_fingerprint(value):
    unsigned = json.loads(json.dumps(value))
    provenance = unsigned.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("source_scope_map_content_fingerprint", None)
    return _digest(unsigned)


def _validate_import_source_map(path, analysis):
    source_map = _read_object(path, "source scope map")
    if source_map.get("protocol") != "source_scope_map_v2" or source_map.get(
        "schema_version"
    ) not in {"2.0", "2.1"}:
        raise ValueError(
            "source scope map must use source_scope_map_v2 schema 2.0 or 2.1"
        )
    provenance = source_map.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("source scope map provenance must be an object")
    declared = _require_string(
        provenance.get("source_scope_map_content_fingerprint"),
        "source scope map content fingerprint",
    )
    if declared != _source_map_fingerprint(source_map):
        raise ValueError("source scope map content fingerprint mismatch")
    source_revision = provenance.get(
        "source_revision", provenance.get("stable_marker_revision")
    )
    if source_revision != analysis["source_revision"]:
        raise ValueError(
            "source scope map source revision differs from profiling analysis"
        )
    ranges = source_map.get("source_ranges")
    if not isinstance(ranges, list) or not any(
        isinstance(item, dict) and item.get("relation") == "exact_cover"
        for item in ranges
    ):
        raise ValueError("source scope map has no actionable exact_cover range")
    return source_map


def _latest_base_handoff(parent, parent_root):
    candidates = [
        (index, step)
        for index, step in enumerate(parent["steps"])
        if step["phase"] == "base_profile_source_mapping"
        and step["state"] == "sealed"
        and step.get("outcome_status") in {"succeeded", "no_gain"}
    ]
    if not candidates:
        raise ValueError("parent session has no accepted BASE/SMAP handoff")
    index, step = candidates[-1]
    if any(
        later.get("step_id") == "optional-multistream"
        and later.get("state") == "sealed"
        and later.get("outcome_status") == "accepted"
        for later in parent["steps"][index + 1 :]
    ):
        raise ValueError(
            "accepted multistream requires a later derived BASE/SMAP handoff"
        )
    result_path = parent_root / step["result_path"]
    result = _read_object(result_path, "BASE/SMAP phase result")
    if _digest(result) != step["result_sha256"]:
        raise ValueError("BASE/SMAP phase result digest mismatch")
    return step, result_path, result


def _roots_overlap(first, second):
    return first == second or first in second.parents or second in first.parents


def derive_session(
    *,
    parent_session_path,
    session_path,
    artifact_root,
    session_id,
    optional_mode,
    profiling_analysis,
    ledger,
    source_scope_map=None,
    approve_imported_evidence=False,
):
    """Create a new optional-experiment session from immutable accepted evidence."""
    if not approve_imported_evidence:
        raise ValueError(
            "deriving a session requires explicit approval of imported evidence"
        )
    if optional_mode not in {"multistream", "source-range", "both"}:
        raise ValueError(
            "derived optional_mode must be multistream, source-range, or both"
        )
    _require_string(session_id, "session_id")
    parent_session_path = Path(parent_session_path).resolve()
    validation = verify_session(parent_session_path)
    if not validation["valid"]:
        raise ValueError(
            "parent session is invalid: " + "; ".join(validation["errors"])
        )
    parent = load_session(parent_session_path)
    if parent["status"] != "completed":
        raise ValueError("parent session must be completed")
    if session_id == parent["session_id"]:
        raise ValueError("derived session_id must differ from the parent session_id")
    parent_root = Path(parent["artifact_root"]).resolve()
    artifact_root = Path(artifact_root).resolve()
    session_path = Path(session_path).resolve()
    if _roots_overlap(parent_root, artifact_root):
        raise ValueError(
            "derived artifact root must be disjoint from the parent artifact root"
        )
    try:
        session_path.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError(
            "derived session path must be inside its artifact root"
        ) from error
    if session_path.exists():
        raise ValueError(f"session already exists: {session_path}")

    parent_session_file, parent_session_relative = _path_within(
        parent_root, parent_session_path, "parent session"
    )
    base_step, base_result_path, base_result = _latest_base_handoff(parent, parent_root)
    declared_artifacts = set(base_result["consumed_artifacts"]) | set(
        base_result["produced_artifacts"]
    )
    evidence_values = {
        "profiling_analysis": profiling_analysis,
        "ledger": ledger,
    }
    if optional_mode in {"source-range", "both"}:
        if source_scope_map is None:
            raise ValueError("source-range derivation requires source_scope_map")
        evidence_values["source_scope_map"] = source_scope_map
    elif source_scope_map is not None:
        evidence_values["source_scope_map"] = source_scope_map

    descriptors = {}
    resolved = {}
    for role, value in evidence_values.items():
        descriptor = _artifact_descriptor(parent_root, value, role)
        if descriptor["path"] not in declared_artifacts:
            raise ValueError(f"{role} is not bound by the selected BASE/SMAP handoff")
        descriptors[role] = descriptor
        resolved[role] = parent_root / descriptor["path"]

    analysis = _validate_import_analysis(resolved["profiling_analysis"])
    _validate_import_ledger(resolved["ledger"], analysis)
    if optional_mode in {"source-range", "both"}:
        _validate_import_source_map(resolved["source_scope_map"], analysis)

    entrypoint = (
        "source-range-from-smap"
        if optional_mode == "source-range"
        else "optional-from-base"
    )
    manifest = {
        "schema_version": EVIDENCE_IMPORT_SCHEMA,
        "authorization": {
            "approved": True,
            "source": "explicit-user-request",
            "approved_at": _now(),
        },
        "parent": {
            "session_id": parent["session_id"],
            "artifact_root": str(parent_root),
            "session_path": parent_session_relative,
            "session_sha256": _file_sha256(parent_session_file),
            "session_fingerprint": _digest(parent),
        },
        "base_handoff": {
            "step_id": base_step["step_id"],
            "result_path": str(base_result_path.relative_to(parent_root)),
            "result_sha256": base_step["result_sha256"],
            "file_sha256": _file_sha256(base_result_path),
            "outcome_status": base_step["outcome_status"],
        },
        "artifacts": descriptors,
        "identity": {
            key: analysis[key]
            for key in (
                "experiment_id",
                "round_id",
                "candidate_name",
                "source_revision",
            )
        },
    }
    manifest_relative = "imports/evidence-import.json"
    manifest_path = artifact_root / manifest_relative
    if session_path == manifest_path:
        raise ValueError(
            "derived session path must differ from the evidence import path"
        )
    if manifest_path.exists():
        raise ValueError(f"evidence import already exists: {manifest_path}")
    session = {
        "schema_version": SESSION_SCHEMA,
        "session_id": session_id,
        "artifact_root": str(artifact_root),
        "optional_mode": optional_mode,
        "entrypoint": entrypoint,
        "evidence_import": {
            "path": manifest_relative,
            "sha256": _digest(manifest),
        },
        "status": "active",
        "created_at": _now(),
        "steps": _canonical_steps(optional_mode, entrypoint=entrypoint),
    }
    _validate_session(session, validate_import=False)
    _atomic_json(manifest_path, manifest)
    _atomic_json(session_path, session)
    return load_session(session_path)


def load_session(session_path):
    try:
        session = json.loads(Path(session_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load session: {error}") from error
    _validate_session(session)
    return session


def next_step(session):
    _validate_session(session)
    for step in session["steps"]:
        if step["state"] == "pending":
            return step
    return None


def resume_session(session_path):
    session = load_session(session_path)
    validation = verify_session(session_path)
    if not validation["valid"]:
        raise ValueError(
            "session cannot be resumed: " + "; ".join(validation["errors"])
        )
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "next_step": next_step(session),
    }


def _validate_evidence_import(session):
    reference = session.get("evidence_import")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(
            "derived session requires one canonical evidence_import reference"
        )
    relative = _relative_artifact(reference.get("path"), "evidence_import path")
    declared_digest = _require_string(reference.get("sha256"), "evidence_import sha256")
    root = Path(session["artifact_root"])
    manifest = _read_object(root / relative, "evidence import")
    if _digest(manifest) != declared_digest:
        raise ValueError("evidence import digest mismatch")
    if manifest.get("schema_version") != EVIDENCE_IMPORT_SCHEMA:
        raise ValueError("evidence import schema_version is invalid")
    authorization = manifest.get("authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("approved") is not True
        or authorization.get("source") != "explicit-user-request"
        or not _requireable(authorization.get("approved_at"))
    ):
        raise ValueError("evidence import lacks explicit approval")
    parent = manifest.get("parent")
    if not isinstance(parent, dict):
        raise ValueError("evidence import parent must be an object")
    parent_root = Path(
        _require_string(parent.get("artifact_root"), "parent artifact_root")
    )
    if not parent_root.is_absolute():
        raise ValueError("parent artifact_root must be absolute")
    parent_session, _ = _path_within(
        parent_root,
        parent_root
        / _relative_artifact(parent.get("session_path"), "parent session_path"),
        "parent session",
    )
    if _file_sha256(parent_session) != parent.get("session_sha256"):
        raise ValueError("imported parent session digest mismatch")
    parent_value = _read_object(parent_session, "parent session")
    if _digest(parent_value) != parent.get("session_fingerprint"):
        raise ValueError("imported parent session fingerprint mismatch")
    if parent_value.get("session_id") != parent.get("session_id"):
        raise ValueError("imported parent session identity mismatch")

    base = manifest.get("base_handoff")
    if not isinstance(base, dict):
        raise ValueError("evidence import base_handoff must be an object")
    base_path, _ = _path_within(
        parent_root,
        parent_root / _relative_artifact(base.get("result_path"), "base result_path"),
        "imported BASE/SMAP result",
    )
    if _file_sha256(base_path) != base.get("file_sha256"):
        raise ValueError("imported BASE/SMAP result digest mismatch")
    base_result = _read_object(base_path, "imported BASE/SMAP result")
    if _digest(base_result) != base.get("result_sha256"):
        raise ValueError("imported BASE/SMAP semantic digest mismatch")
    matching_base = next(
        (
            step
            for step in parent_value.get("steps", [])
            if isinstance(step, dict) and step.get("step_id") == base.get("step_id")
        ),
        None,
    )
    if (
        not isinstance(matching_base, dict)
        or matching_base.get("phase") != "base_profile_source_mapping"
        or matching_base.get("state") != "sealed"
        or matching_base.get("result_path") != base.get("result_path")
        or matching_base.get("result_sha256") != base.get("result_sha256")
        or matching_base.get("outcome_status") != base.get("outcome_status")
    ):
        raise ValueError("imported BASE/SMAP handoff differs from the parent session")
    declared_artifacts = set(base_result.get("consumed_artifacts", [])) | set(
        base_result.get("produced_artifacts", [])
    )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("evidence import artifacts must be an object")
    if not set(artifacts).issubset(
        {"profiling_analysis", "ledger", "source_scope_map"}
    ):
        raise ValueError("evidence import contains an unsupported artifact role")
    required_roles = {"profiling_analysis", "ledger"}
    if session["optional_mode"] in {"source-range", "both"}:
        required_roles.add("source_scope_map")
    if not required_roles.issubset(artifacts):
        raise ValueError("evidence import is missing required artifacts")
    resolved = {}
    for role, descriptor in artifacts.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"imported artifact {role} descriptor must be an object")
        path, _ = _path_within(
            parent_root,
            parent_root / _relative_artifact(descriptor.get("path"), f"{role} path"),
            f"imported artifact {role}",
        )
        if _file_sha256(path) != descriptor.get("sha256"):
            raise ValueError(f"imported artifact {role} digest mismatch")
        if path.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError(f"imported artifact {role} size mismatch")
        if descriptor.get("path") not in declared_artifacts:
            raise ValueError(f"imported artifact {role} is not bound by BASE/SMAP")
        resolved[role] = path
    analysis = _validate_import_analysis(resolved["profiling_analysis"])
    _validate_import_ledger(resolved["ledger"], analysis)
    if session["optional_mode"] in {"source-range", "both"}:
        _validate_import_source_map(resolved["source_scope_map"], analysis)
    identity = manifest.get("identity")
    expected_identity = {
        key: analysis[key]
        for key in ("experiment_id", "round_id", "candidate_name", "source_revision")
    }
    if identity != expected_identity:
        raise ValueError("evidence import identity differs from profiling analysis")


def _validate_session(session, *, validate_import=True):
    if not isinstance(session, dict):
        raise ValueError("session must be an object")
    if session.get("schema_version") != SESSION_SCHEMA:
        raise ValueError("unsupported session schema_version")
    _require_string(session.get("session_id"), "session_id")
    if session.get("optional_mode") not in OPTIONAL_MODES:
        raise ValueError("session optional_mode is invalid")
    entrypoint = session.get("entrypoint", "full")
    if entrypoint not in ENTRYPOINTS:
        raise ValueError("session entrypoint is invalid")
    if session.get("status") not in {"active", "completed"}:
        raise ValueError("session status is invalid")
    _require_string(session.get("artifact_root"), "artifact_root")
    if entrypoint == "full":
        if "evidence_import" in session:
            raise ValueError("full session must not declare evidence_import")
    elif validate_import:
        _validate_evidence_import(session)
    if not isinstance(session.get("steps"), list) or not session["steps"]:
        raise ValueError("session steps must be a non-empty list")
    pending_seen = False
    seen_ids = set()
    for step in session["steps"]:
        if not isinstance(step, dict):
            raise ValueError("session step must be an object")
        step_id = _require_string(step.get("step_id"), "step_id")
        if not STEP_IDS.fullmatch(step_id) or step_id in seen_ids:
            raise ValueError("session step_id is invalid or duplicated")
        seen_ids.add(step_id)
        if step.get("phase") not in PHASES:
            raise ValueError("session step phase is invalid")
        expected_agent, expected_skill = PHASES[step["phase"]]
        if (
            step.get("agent_id") != expected_agent
            or step.get("skill") != expected_skill
        ):
            raise ValueError("session step agent or skill is invalid")
        if step.get("state") not in {"pending", "sealed"}:
            raise ValueError("session step state is invalid")
        if step["state"] == "pending":
            pending_seen = True
        elif pending_seen:
            raise ValueError("sealed step cannot follow a pending step")
        if step["state"] == "sealed":
            _relative_artifact(step.get("result_path"), "result_path")
            _require_string(step.get("result_sha256"), "result_sha256")
            if step.get("outcome_status") not in RESULT_STATUSES:
                raise ValueError("sealed step outcome_status is invalid")
            if "dispatch_receipt_path" in step:
                _relative_artifact(
                    step["dispatch_receipt_path"], "dispatch_receipt_path"
                )
                _require_string(
                    step.get("dispatch_receipt_sha256"), "dispatch_receipt_sha256"
                )
    if (
        session["status"] == "completed"
        and next(
            (step for step in session["steps"] if step["state"] == "pending"), None
        )
        is not None
    ):
        raise ValueError("completed session has pending steps")
    _validate_canonical_schedule(session)


def _validate_result(result, expected, session, *, allow_system_generated=False):
    if not isinstance(result, dict):
        raise ValueError("phase result must be an object")
    required = (
        "schema_version",
        "session_id",
        "step_id",
        "phase",
        "agent_id",
        "status",
        "summary_zh",
        "input_fingerprints",
        "consumed_artifacts",
        "produced_artifacts",
        "decisions",
        "blockers",
        "next_step_guidance_zh",
        "details_zh",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"phase result missing required fields: {', '.join(missing)}")
    if result["schema_version"] != RESULT_SCHEMA:
        raise ValueError("phase result schema_version is invalid")
    if result["session_id"] != session["session_id"]:
        raise ValueError("phase result session_id does not match the current session")
    for key in ("step_id", "phase", "agent_id"):
        if result[key] != expected[key]:
            raise ValueError(f"phase result {key} does not match the current step")
    if result["status"] not in RESULT_STATUSES:
        raise ValueError("phase result status is invalid")
    system_generated = result.get("system_generated_by") == "superkernel-auto-tune"
    if system_generated and not allow_system_generated:
        raise ValueError(
            "system_generated_by is reserved for controller-created handoffs"
        )
    if not system_generated and result["status"] not in _allowed_statuses(expected):
        raise ValueError(f"phase result status is invalid for {expected['phase']}")
    _require_string(result["summary_zh"], "summary_zh")
    _require_string(result["next_step_guidance_zh"], "next_step_guidance_zh")
    if not isinstance(result["input_fingerprints"], dict):
        raise ValueError("input_fingerprints must be an object")
    for name, fingerprint in result["input_fingerprints"].items():
        _require_string(name, "input fingerprint name")
        _require_string(fingerprint, "input fingerprint")
    for key in ("consumed_artifacts", "produced_artifacts"):
        if not isinstance(result[key], list):
            raise ValueError(f"{key} must be a list")
        for index, artifact in enumerate(result[key]):
            _relative_artifact(artifact, f"{key}[{index}]")
    for key in ("decisions", "blockers"):
        if not isinstance(result[key], list):
            raise ValueError(f"{key} must be a list")
    if not isinstance(result["details_zh"], dict):
        raise ValueError("details_zh must be an object")
    if expected["phase"] == "optional_experiments":
        branch = expected["optional_branch"]
        if branch == "none" and result["status"] != "not_requested":
            raise ValueError("optional mode none requires a not_requested handoff")
    if expected["phase"] == "final_e2e_report":
        missing_details = [
            key
            for key in FINAL_DETAIL_KEYS
            if not _requireable(result["details_zh"].get(key))
        ]
        if missing_details:
            raise ValueError(
                f"final details_zh is missing: {', '.join(missing_details)}"
            )
        if "ledger_path" not in result:
            raise ValueError("final phase result requires ledger_path")
        _, ledger = _load_ledger(session["artifact_root"], result["ledger_path"])
        expected_status = {
            "beneficial": "accepted",
            "no_gain": "no_gain",
            "not_run": "not_run",
            "failed": "failed",
            "blocked": "blocked",
        }[ledger["final_e2e"]["classification"]]
        if result["status"] != expected_status:
            raise ValueError(
                "final phase result status does not match ledger final_e2e classification"
            )


def _requireable(value):
    return isinstance(value, str) and bool(value.strip())


def _phase_report(result, expected):
    lines = [
        f"# {expected['phase']} 阶段报告",
        "",
        f"- Session: `{result['session_id']}`",
        f"- Step: `{result['step_id']}`",
        f"- Agent: `{result['agent_id']}`",
        f"- Status: `{result['status']}`",
        "",
        "## 结论",
        "",
        result["summary_zh"],
        "",
        "## 下一步",
        "",
        result["next_step_guidance_zh"],
    ]
    if result["blockers"]:
        lines.extend(["", "## 阻塞项", ""])
        lines.extend(
            f"- {json.dumps(item, ensure_ascii=False)}" for item in result["blockers"]
        )
    return "\n".join(lines) + "\n"


def _insert_derived_base(session, after_index):
    session["steps"].insert(
        after_index + 1,
        _step(
            "base-profile-derived",
            "base_profile_source_mapping",
            derived_family_rebase=True,
        ),
    )


def _must_short_circuit(step, status):
    if step["phase"] in {"intake_preparation", "s0_baseline"}:
        return status != "succeeded"
    if step["phase"] == "stage_a_scope_selection":
        return status != "accepted"
    if step["phase"] == "stage_o_option_tuning":
        return status not in {"accepted", "no_gain"}
    if step["phase"] == "base_profile_source_mapping":
        return status not in {"succeeded", "no_gain"}
    return False


def _seal_not_run_steps(session, reason_zh):
    root = Path(session["artifact_root"])
    for step in session["steps"]:
        if step["state"] != "pending" or step["phase"] == "final_e2e_report":
            continue
        status = "not_requested" if step.get("optional_branch") == "none" else "not_run"
        result = {
            "schema_version": RESULT_SCHEMA,
            "session_id": session["session_id"],
            "step_id": step["step_id"],
            "phase": step["phase"],
            "agent_id": step["agent_id"],
            "status": status,
            "summary_zh": f"未执行：{reason_zh}",
            "input_fingerprints": {"session_state": _digest(session)},
            "consumed_artifacts": [],
            "produced_artifacts": [],
            "decisions": [],
            "blockers": [{"reason_zh": reason_zh}],
            "next_step_guidance_zh": "控制 Agent 已短路至最终 E2E 报告阶段。",
            "details_zh": {},
            "system_generated_by": "superkernel-auto-tune",
        }
        _validate_result(result, step, session, allow_system_generated=True)
        phase_root = root / "phases" / step["step_id"]
        result_path = phase_root / "phase-result.json"
        report_path = phase_root / "PHASE_REPORT.md"
        if result_path.exists() or report_path.exists():
            raise ValueError(
                f"cannot short-circuit an existing phase scene: {step['step_id']}"
            )
        _atomic_json(result_path, result)
        _atomic_text(report_path, _phase_report(result, step))
        step["state"] = "sealed"
        step["result_path"] = str(result_path.relative_to(root))
        step["result_sha256"] = _digest(result)
        step["outcome_status"] = result["status"]
        step["sealed_at"] = _now()


def create_dispatch_task(session_path):
    """Build the only structured task a controller may give to a stage agent."""
    session = load_session(session_path)
    if session["status"] == "completed":
        raise ValueError("completed session has no dispatchable step")
    step = next_step(session)
    if step is None:
        raise ValueError("session has no dispatchable step")
    consumed = []
    for previous in session["steps"]:
        if previous is step:
            break
        if previous["state"] == "sealed":
            consumed.append(
                {
                    "step_id": previous["step_id"],
                    "result_path": previous["result_path"],
                    "result_sha256": previous["result_sha256"],
                    "outcome_status": previous["outcome_status"],
                }
            )
    task = {
        "schema_version": TASK_SCHEMA,
        "session_id": session["session_id"],
        "session_path": str(Path(session_path).resolve()),
        "artifact_root": session["artifact_root"],
        "session_fingerprint": _digest(session),
        "step": dict(step),
        "consumed_handoffs": consumed,
        "handoff": {
            "result_schema": RESULT_SCHEMA,
            "seal_command": [
                "python3",
                "scripts/auto_tune_session.py",
                "seal",
                "--session",
                str(Path(session_path).resolve()),
                "--result",
                "<phase-result.json>",
            ],
        },
    }
    if "evidence_import" in session:
        task["evidence_import"] = dict(session["evidence_import"])
    return task


def validate_dispatch_task(task, session_path):
    if not isinstance(task, dict) or task.get("schema_version") != TASK_SCHEMA:
        raise ValueError("dispatch task schema_version is invalid")
    current = create_dispatch_task(session_path)
    for key in (
        "session_id",
        "session_path",
        "artifact_root",
        "session_fingerprint",
        "step",
        "evidence_import",
    ):
        if task.get(key) != current.get(key):
            raise ValueError("stale dispatch task does not match the current session")
    if task.get("consumed_handoffs") != current["consumed_handoffs"]:
        raise ValueError(
            "stale dispatch task handoffs do not match the current session"
        )
    return current


def write_dispatch_task(session_path, output_path):
    task = create_dispatch_task(session_path)
    _atomic_json(output_path, task)
    return task


def _next_dispatch_root(artifact_root, step_id):
    dispatches = Path(artifact_root) / "dispatches"
    for attempt in range(1, 10000):
        candidate = dispatches / f"{step_id}-attempt-{attempt:03d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate, attempt
        except FileExistsError:
            continue
    raise ValueError(f"dispatch attempt limit reached for {step_id}")


def run_agent_step(session_path, runner_argv, *, timeout_seconds=3600):
    """Invoke one provider-neutral Agent host and seal its structured result."""
    if (
        not isinstance(runner_argv, (list, tuple))
        or not runner_argv
        or any(not isinstance(item, str) or not item for item in runner_argv)
    ):
        raise ValueError("runner_argv must be a non-empty string list")
    if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    session_path = Path(session_path).resolve()
    task = create_dispatch_task(session_path)
    step = task["step"]
    dispatch_root, attempt = _next_dispatch_root(task["artifact_root"], step["step_id"])
    task_path = dispatch_root / "phase-task.json"
    result_path = dispatch_root / "phase-result.json"
    stdout_path = dispatch_root / "runner.stdout.log"
    stderr_path = dispatch_root / "runner.stderr.log"
    receipt_path = dispatch_root / "dispatch-receipt.json"
    _atomic_json(task_path, task)
    started_at = _now()
    command = [*runner_argv, "--task", str(task_path), "--result", str(result_path)]
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "session_id": task["session_id"],
        "step_id": step["step_id"],
        "phase": step["phase"],
        "agent_id": step["agent_id"],
        "skill": step["skill"],
        "attempt": attempt,
        "started_at": started_at,
        "task_path": str(task_path.relative_to(task["artifact_root"])),
        "task_sha256": _file_sha256(task_path),
        "result_path": str(result_path.relative_to(task["artifact_root"])),
        "stdout_path": str(stdout_path.relative_to(task["artifact_root"])),
        "stderr_path": str(stderr_path.relative_to(task["artifact_root"])),
        "runner_argv_sha256": _digest(list(runner_argv)),
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
        _atomic_text(stdout_path, completed.stdout)
        _atomic_text(stderr_path, completed.stderr)
        receipt["return_code"] = completed.returncode
        if completed.returncode != 0:
            raise ValueError(f"Agent host exited with status {completed.returncode}")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Agent host did not produce a readable phase result: {error}"
            ) from error
        current = load_session(session_path)
        _validate_result(result, next_step(current), current)
        sealed = seal_handoff(session_path, result)
        receipt.update(
            {
                "runner_status": "passed",
                "result_sha256": _file_sha256(result_path),
                "finished_at": _now(),
            }
        )
        _atomic_json(receipt_path, receipt)
        sealed = load_session(session_path)
        sealed_step = next(
            item for item in sealed["steps"] if item["step_id"] == step["step_id"]
        )
        sealed_step["dispatch_receipt_path"] = str(
            receipt_path.relative_to(task["artifact_root"])
        )
        sealed_step["dispatch_receipt_sha256"] = _file_sha256(receipt_path)
        _atomic_json(session_path, sealed)
        return load_session(session_path)
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode()
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode()
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        _atomic_text(stdout_path, stdout)
        _atomic_text(stderr_path, stderr)
        receipt.update(
            {
                "runner_status": "timed_out",
                "error": f"Agent host timed out after {timeout_seconds} seconds",
                "finished_at": _now(),
            }
        )
        _atomic_json(receipt_path, receipt)
        raise ValueError(receipt["error"]) from error
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if not stdout_path.exists():
            _atomic_text(stdout_path, "")
        if not stderr_path.exists():
            _atomic_text(stderr_path, "")
        receipt.update(
            {
                "runner_status": "failed",
                "error": str(error),
                "finished_at": _now(),
            }
        )
        if result_path.is_file():
            receipt["result_sha256"] = _file_sha256(result_path)
        _atomic_json(receipt_path, receipt)
        raise


def seal_handoff(session_path, result):
    session_path = Path(session_path)
    session = load_session(session_path)
    if session["status"] == "completed":
        raise ValueError("completed session cannot accept a handoff")
    if isinstance(result, dict) and any(
        step["state"] == "sealed" and result.get("step_id") == step["step_id"]
        for step in session["steps"]
    ):
        raise ValueError(f"current step is already sealed: {result['step_id']}")
    expected = next_step(session)
    if expected is None:
        raise ValueError("session has no pending step")
    _validate_result(result, expected, session)
    phase_root = Path(session["artifact_root"]) / "phases" / expected["step_id"]
    result_path = phase_root / "phase-result.json"
    report_path = phase_root / "PHASE_REPORT.md"
    if result_path.exists() or report_path.exists():
        raise ValueError(f"current step is already sealed: {expected['step_id']}")
    _atomic_json(result_path, result)
    _atomic_text(report_path, _phase_report(result, expected))
    expected["state"] = "sealed"
    expected["result_path"] = str(result_path.relative_to(session["artifact_root"]))
    expected["result_sha256"] = _digest(result)
    expected["outcome_status"] = result["status"]
    expected["sealed_at"] = _now()
    index = session["steps"].index(expected)
    if (
        session["optional_mode"] == "both"
        and expected["step_id"] == "optional-multistream"
        and result["status"] == "accepted"
    ):
        _insert_derived_base(session, index)
    if _must_short_circuit(expected, result["status"]):
        _seal_not_run_steps(session, result["summary_zh"])
    if expected["phase"] == "final_e2e_report":
        session["status"] = "completed"
        _atomic_json(session_path, session)
        render_final_report(session_path)
        return load_session(session_path)
    _atomic_json(session_path, session)
    return load_session(session_path)


def verify_session(session_path):
    try:
        session = load_session(session_path)
        root = Path(session["artifact_root"])
        for step in session["steps"]:
            if step["state"] != "sealed":
                continue
            result_path = root / step["result_path"]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if _digest(result) != step["result_sha256"]:
                raise ValueError(
                    f"sealed phase result digest mismatch: {step['step_id']}"
                )
            _validate_result(result, step, session, allow_system_generated=True)
            if result["status"] != step["outcome_status"]:
                raise ValueError(
                    f"sealed phase result outcome mismatch: {step['step_id']}"
                )
            if not (result_path.parent / "PHASE_REPORT.md").is_file():
                raise ValueError(f"missing phase report: {step['step_id']}")
            if "dispatch_receipt_path" in step:
                receipt_path = root / step["dispatch_receipt_path"]
                if _file_sha256(receipt_path) != step["dispatch_receipt_sha256"]:
                    raise ValueError(
                        f"dispatch receipt digest mismatch: {step['step_id']}"
                    )
        if (
            session["status"] == "completed"
            and not (root / "FINAL_E2E_REPORT.md").is_file()
        ):
            raise ValueError("completed session is missing FINAL_E2E_REPORT.md")
        return {
            "valid": True,
            "session_id": session["session_id"],
            "status": session["status"],
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "errors": [str(error)]}


def _terminal_outcome_label(status):
    return {
        "accepted": "有收益",
        "no_gain": "无收益",
        "not_run": "未执行",
        "blocked": "失败或阻塞",
        "failed": "失败或阻塞",
        "invalid": "失败或阻塞",
    }[status]


SUMMARY_STAGES = {
    "stage_a": ("stage_a_scope_selection", "screening"),
    "stage_o": ("stage_o_option_tuning", "option"),
    "base_profile_source_mapping": ("base_profile_source_mapping", "profiling"),
    "optional_multistream": ("optional_experiments", "optional_clean"),
    "optional_source_range": ("optional_experiments", "optional_clean"),
    "optional_experiments": ("optional_experiments", "optional_clean"),
}


def _summary_evidence(value, root, name, *, required=False):
    artifacts = value.get("evidence_artifacts", [])
    if not isinstance(artifacts, list) or (required and not artifacts):
        raise ValueError(f"report_summary {name} requires evidence_artifacts")
    for artifact in artifacts:
        relative = _relative_artifact(artifact, f"report_summary {name} evidence")
        path = (Path(root) / relative).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
            raise ValueError(
                f"report_summary {name} evidence missing or outside artifact root: {artifact}"
            )


def _summary_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"report_summary {name} must be positive and finite")


def _validate_report_summary(summary, final_e2e, root):
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != "superkernel-report-summary-v1"
    ):
        raise ValueError("report_summary schema_version is invalid")
    baseline = summary.get("baseline")
    stages = summary.get("stages")
    if not isinstance(baseline, dict) or not isinstance(stages, list):
        raise ValueError("report_summary requires baseline object and stages list")
    for index, row in enumerate([baseline, *stages]):
        if not isinstance(row, dict):
            raise ValueError("report_summary measurement must be an object")
        _require_string(row.get("metric"), "report_summary metric")
        _require_string(row.get("reason_zh"), "report_summary reason_zh")
        for key in ("value_ms", "baseline_ms", "candidate_ms"):
            if row.get(key) is not None:
                _summary_number(row[key], key)
        for key in ("run_count", "selector_rank"):
            if row.get(key) is not None and (
                isinstance(row[key], bool)
                or not isinstance(row[key], int)
                or row[key] <= 0
            ):
                raise ValueError(f"report_summary {key} must be a positive integer")
        measured = row.get("value_ms" if index == 0 else "candidate_ms") is not None
        if measured and row.get("run_count") is None:
            raise ValueError("report_summary measured result requires run_count")
        _summary_evidence(
            row,
            root,
            "measurement",
            required=measured or row.get("baseline_ms") is not None,
        )
        if index == 0:
            continue
        stage = row.get("stage")
        if stage not in SUMMARY_STAGES or row.get("kind") != SUMMARY_STAGES[stage][1]:
            raise ValueError(
                "report_summary stage/kind is invalid; final clean comes only from final_e2e"
            )
        _require_string(row.get("candidate_id"), "report_summary candidate_id")
        _require_string(row.get("gates_zh"), "report_summary gates_zh")
        if row.get("status") not in RESULT_STATUSES or not isinstance(
            row.get("eligible"), bool
        ):
            raise ValueError("report_summary status/eligible is invalid")
        if row["eligible"] and (not measured or row["status"] != "accepted"):
            raise ValueError(
                "report_summary eligible candidate must be measured and accepted"
            )
        if row.get("baseline_ms") is not None and row["kind"] != "profiling":
            if row["metric"] != baseline["metric"] or baseline.get("value_ms") is None:
                raise ValueError(
                    "report_summary clean comparison metric must match frozen S0"
                )
            if not math.isclose(
                row["baseline_ms"], baseline["value_ms"], rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("report_summary baseline_ms must match frozen S0")
        if "improvement_pct" in row:
            if not measured or row.get("baseline_ms") is None:
                raise ValueError(
                    "report_summary improvement_pct requires paired measurements"
                )
            expected = (
                (row["baseline_ms"] - row["candidate_ms"]) / row["baseline_ms"] * 100
            )
            declared = row["improvement_pct"]
            if (
                isinstance(declared, bool)
                or not isinstance(declared, (int, float))
                or not math.isclose(declared, expected, rel_tol=1e-6, abs_tol=1e-6)
            ):
                raise ValueError(
                    "report_summary improvement_pct does not match measurements"
                )
    for key in ("final_gates_zh", "display_note_zh"):
        if key in summary:
            _require_string(summary[key], f"report_summary {key}")
    for role in ("winner", "fallback"):
        config = summary.get(role)
        if config is None:
            continue
        if not isinstance(config, dict):
            raise ValueError(f"report_summary {role} must be an object")
        for key in ("candidate_id", "scope_strategy", "reason_zh"):
            _require_string(config.get(key), f"report_summary {role}.{key}")
        for key in ("config_path", "config_fingerprint"):
            if key in config:
                _require_string(config[key], f"report_summary {role}.{key}")
        if "promotion_path" in config and config["promotion_path"] not in {
            "whole-scope",
            "FINAL",
        }:
            raise ValueError(
                "report_summary promotion_path must be whole-scope or FINAL"
            )
        _summary_evidence(config, root, role, required=True)
        for key in ("option_config", "debug_option_config"):
            if key in config and not isinstance(config[key], dict):
                raise ValueError(f"report_summary {role}.{key} must be an object")
        options = config.get("option_config", {})
        for key in ("super_kernel_optimize_options", "super_kernel_debug_options"):
            if key in options and not isinstance(options[key], dict):
                raise ValueError(f"report_summary {role}.{key} must be an object")
        if "super_kernel_debug_options" in options and "debug_option_config" in config:
            if options["super_kernel_debug_options"] != config["debug_option_config"]:
                raise ValueError("report_summary debug options disagree")
        if "config_path" in config:
            _summary_evidence(
                {"evidence_artifacts": [config["config_path"]]}, root, "config_path"
            )
        if role == "winner":
            if final_e2e["classification"] != "beneficial":
                raise ValueError("report_summary winner requires beneficial final_e2e")
            for key in ("candidate_id", "scope_strategy", "option_config"):
                if config.get(key) != final_e2e[key]:
                    raise ValueError(
                        f"report_summary winner {key} differs from final_e2e"
                    )
            if config.get("promotion_path") not in {"whole-scope", "FINAL"}:
                raise ValueError(
                    "report_summary winner requires whole-scope or FINAL promotion_path"
                )
    return summary


def _summary_cell(value):
    if value is None or value == "":
        return "N/A"
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def _summary_row(label, metric, baseline, candidate, count, status, evidence):
    delta = (
        candidate - baseline if baseline is not None and candidate is not None else None
    )
    improvement = -delta / baseline * 100 if delta is not None else None
    cells = [
        label,
        metric,
        f"{baseline:.6f}" if baseline is not None else None,
        f"{candidate:.6f}" if candidate is not None else None,
        f"{delta:+.6f}" if delta is not None else None,
        f"{improvement:+.6f}%" if improvement is not None else None,
        count,
        status,
        ", ".join(f"`{path}`" for path in evidence) or "N/A：未提供证据",
    ]
    return "| " + " | ".join(_summary_cell(cell) for cell in cells) + " |"


def _render_report_summary(session, result, final_e2e, summary):
    classification = final_e2e["classification"]
    measured_final = classification in {"beneficial", "no_gain"}
    outcome = {
        "beneficial": "调优成功，最终 clean E2E 有收益",
        "no_gain": "最终 clean E2E 无收益，未产生终局优胜者",
        "not_run": "未获得最终 E2E 验证的优胜者；最终 clean E2E 未执行",
        "failed": "最终验证失败，未产生终局优胜者",
        "blocked": "调优阻塞，未产生终局优胜者",
    }[classification]
    lines = [
        "## 结果速览",
        "",
        f"**{outcome}。**",
        "",
        f"- 最终 E2E 结论: {_terminal_outcome_label(result['status'])} (`{classification}`)",
        f"- 判定依据: {final_e2e['reason_zh']}",
        "",
        "## 关键数据对比",
        "",
        "耗时单位 ms；耗时差 = 实验 − 基线；改善率为正表示更快。不同测量口径不互算。",
        "screening / option / optional_clean 是阶段实验，profiling 是诊断，均不替代最终 clean E2E。",
        "",
        "| 阶段/候选 | 测量口径 | 基线耗时 ms | 实验耗时 ms | 耗时差 ms | 改善率 | 运行数 | 门禁/状态 | 证据 |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    summary = summary or {}
    baseline = summary.get("baseline", {})
    if not baseline and measured_final:
        baseline = {
            "metric": "final clean median",
            "value_ms": final_e2e["baseline"]["median_ms"],
            "run_count": final_e2e["baseline"].get("run_count"),
            "reason_zh": "来自 final_e2e；独立进程数未提供，不将 sample_count 当作运行数",
            "evidence_artifacts": final_e2e["evidence_artifacts"],
        }
    lines.append(
        _summary_row(
            "S0（冻结基线）",
            baseline.get("metric"),
            baseline.get("value_ms"),
            None,
            baseline.get("run_count"),
            baseline.get("reason_zh", "N/A：旧账本未提供结构化基线"),
            baseline.get("evidence_artifacts", []),
        )
    )
    gates = []
    stages = summary.get("stages", [])
    for stage, (phase, kind) in SUMMARY_STAGES.items():
        candidates = [row for row in stages if row["stage"] == stage]
        steps = [step for step in session["steps"] if step["phase"] == phase]
        if stage in {"optional_multistream", "optional_source_range"}:
            branch = stage.removeprefix("optional_").replace("_", "-")
            steps = [step for step in steps if step.get("optional_branch") == branch]
        elif stage == "optional_experiments":
            steps = [step for step in steps if step.get("optional_branch") == "none"]
        if not candidates:
            if not steps:
                continue
            if stage == "optional_experiments" and any(
                row["stage"].startswith("optional_") for row in stages
            ):
                continue
            status = "; ".join(
                f"{step['step_id']}: {step.get('outcome_status', step['state'])}"
                for step in steps
            )
            lines.append(
                _summary_row(
                    stage,
                    kind,
                    None,
                    None,
                    None,
                    status + "；N/A：未提供关键测量，原因见阶段正文",
                    [step["result_path"] for step in steps if step.get("result_path")],
                )
            )
            continue
        for metric in dict.fromkeys(row["metric"] for row in candidates):
            comparable = [row for row in candidates if row["metric"] == metric]
            chosen = min(
                comparable,
                key=lambda row: (
                    not row["eligible"],
                    row.get("candidate_ms") is None,
                    row.get("selector_rank") or math.inf,
                    row.get("candidate_ms")
                    if row.get("candidate_ms") is not None
                    else math.inf,
                    row["candidate_id"],
                ),
            )
            label = f"{stage} / {chosen['candidate_id']}"
            status = chosen["status"] + "；" + chosen["reason_zh"]
            if not chosen["eligible"] and chosen.get("candidate_ms") is not None:
                status += (
                    "；最佳已测尝试，未通过门禁，非优胜者"
                    if kind != "profiling"
                    else "；仅诊断，非优胜者"
                )
            lines.append(
                _summary_row(
                    label,
                    f"{kind} / {metric}",
                    chosen.get("baseline_ms"),
                    chosen.get("candidate_ms"),
                    chosen.get("run_count"),
                    status,
                    chosen.get("evidence_artifacts", []),
                )
            )
            gates.append(f"- {label}：{chosen['gates_zh']}")
    lines.append(
        _summary_row(
            f"最终 E2E / {final_e2e['candidate_id']}",
            "final clean / median",
            final_e2e["baseline"]["median_ms"] if measured_final else None,
            final_e2e["candidate"]["median_ms"] if measured_final else None,
            final_e2e["candidate"].get("run_count") if measured_final else None,
            classification + "；" + final_e2e["reason_zh"],
            final_e2e["evidence_artifacts"],
        )
    )
    lines.extend(
        [
            "",
            "### 门禁与统计说明",
            "",
            *gates,
            "- 最终阈值 / P90 / stddev："
            + summary.get(
                "final_gates_zh", "N/A：未提供结构化门禁说明，见最终 Clean E2E 正文。"
            ),
            "- 运行数仅指独立进程次数；旧 final_e2e.sample_count 不自动解释为运行数。",
            "",
            "## 配置与建议",
            "",
        ]
    )
    role = "winner" if classification == "beneficial" else "fallback"
    config = summary.get(role, {})
    if role == "winner":
        lines.append(f"- 优胜者：`{final_e2e['candidate_id']}`（最终 clean E2E 验证）")
        config = {**final_e2e, **config}
    else:
        lines.append("- 优胜者：无。最佳尝试不是推荐配置。")
        lines.append(
            f"- 回退配置：{config.get('candidate_id', 'N/A：未提供已验证回退配置，不推断默认 SK-off')}"
        )
    option_config = config.get("option_config")
    optimize = option_config
    debug = config.get("debug_option_config")
    if isinstance(option_config, dict) and any(
        key in option_config
        for key in ("super_kernel_optimize_options", "super_kernel_debug_options")
    ):
        optimize = option_config.get("super_kernel_optimize_options")
        debug = option_config.get("super_kernel_debug_options", debug)
    for label, value in (
        ("SK 框定方式", config.get("scope_strategy")),
        ("晋级路径", config.get("promotion_path")),
        ("Option 配置", option_config),
        ("Optimize options", optimize),
        ("Debug options", debug),
        ("配置文件", config.get("config_path")),
        ("配置 fingerprint", config.get("config_fingerprint")),
    ):
        text = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, dict)
            else value
        )
        lines.append(
            f"- {label}: `{text}`"
            if text is not None
            else f"- {label}: N/A（未提供证据，不猜测默认值）"
        )
    lines.extend(
        [
            f"- 配置依据: {config.get('reason_zh', 'N/A：缺少配置证据')}",
            "- 配置证据: "
            + (
                ", ".join(f"`{path}`" for path in config.get("evidence_artifacts", []))
                or "N/A"
            ),
        ]
    )
    if summary.get("display_note_zh"):
        lines.extend(["", "> " + _summary_cell(summary["display_note_zh"])])
    return lines


def render_final_report(session_path, output_path=None, summary_path=None):
    session = load_session(session_path)
    final = next(
        (
            step
            for step in session["steps"]
            if step["phase"] == "final_e2e_report" and step["state"] == "sealed"
        ),
        None,
    )
    if final is None:
        raise ValueError("final phase is not sealed")
    root = Path(session["artifact_root"])
    result = json.loads((root / final["result_path"]).read_text(encoding="utf-8"))
    ledger_path, ledger = _load_ledger(root, result["ledger_path"])
    final_e2e = ledger["final_e2e"]
    summary = ledger.get("report_summary")
    if summary_path is not None:
        summary = _read_object(Path(summary_path), "report_summary")
        _validate_report_summary(summary, final_e2e, root)
    target = Path(output_path) if output_path else root / "FINAL_E2E_REPORT.md"
    lines = [
        "# SuperKernel Auto Tune 最终 E2E 报告",
        "",
        *_render_report_summary(session, result, final_e2e, summary),
        "",
        "## 会话与原始交接摘要",
        "",
        f"- Session: `{session['session_id']}`",
        f"- Entrypoint: `{session.get('entrypoint', 'full')}`",
        f"- Optional mode: `{session['optional_mode']}`",
        f"- Final status: `{result['status']}`",
        "",
        result["summary_zh"],
        "",
        f"- 最终 E2E 结论: {_terminal_outcome_label(result['status'])} (`{final_e2e['classification']}`)",
        f"- Candidate: `{final_e2e['candidate_id']}`",
        f"- SK 框定方式: `{final_e2e['scope_strategy']}`",
        f"- Option 配置: `{json.dumps(final_e2e['option_config'], ensure_ascii=False, sort_keys=True)}`",
        f"- 判定依据: {final_e2e['reason_zh']}",
    ]
    if "evidence_import" in session:
        manifest = _read_object(
            root / session["evidence_import"]["path"], "evidence import"
        )
        lines.extend(
            [
                f"- Imported parent: `{manifest['parent']['session_id']}`",
                f"- Imported BASE/SMAP: `{manifest['base_handoff']['step_id']}` / "
                f"`{manifest['base_handoff']['result_sha256']}`",
            ]
        )
    if final_e2e["classification"] in {"beneficial", "no_gain"}:
        lines.extend(
            [
                f"- Baseline clean E2E median: `{final_e2e['baseline']['median_ms']}` ms",
                f"- Candidate clean E2E median: `{final_e2e['candidate']['median_ms']}` ms",
                f"- 收益率: `{final_e2e['improvement_pct']:.6f}%`",
            ]
        )
    evidence = final_e2e["evidence_artifacts"]
    lines.append(
        "- E2E evidence: " + (", ".join(f"`{item}`" for item in evidence) or "无")
    )
    labels = {
        "environment": "环境与准备",
        "s0": "S0 基线",
        "stage_a": "Stage A 框定方式",
        "stage_o": "Stage O Option",
        "base_profile_source_mapping": "BASE Profile 与 SMAP",
        "optional_experiments": "可选实验",
        "final_e2e": "最终 Clean E2E",
    }
    for key in FINAL_DETAIL_KEYS:
        lines.extend(["", f"## {labels[key]}", "", result["details_zh"][key]])
    lines.extend(["", "## Ledger 汇总", ""])
    lines.append(f"- Ledger: `{ledger_path}`")
    lines.append(f"- Active experiments: `{len(ledger['experiments'])}`")
    lines.extend(
        [
            "",
            "## 自动结论",
            "",
            f"- 最终 clean E2E 是唯一终局判定：{_terminal_outcome_label(result['status'])}。",
            "- 下表中的单 SK 分类仅用于解释与溯源，不可替代最终 E2E 结论。",
        ]
    )
    lines.extend(
        [
            "",
            "| Experiment | Frozen config | Rounds | Lifecycle | Options | SK classification | Blockers |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for experiment_id in sorted(ledger["experiments"]):
        experiment = ledger["experiments"][experiment_id]
        rounds = experiment.get("rounds", [])
        round_ids = []
        lifecycle = []
        options = []
        for round_data in rounds:
            if not isinstance(round_data, dict):
                continue
            round_ids.append(str(round_data.get("round_id", "unknown")))
            values = round_data.get("lifecycle", {})
            if isinstance(values, dict):
                lifecycle.append(
                    ", ".join(
                        f"{key}={values[key]}"
                        for key in ("correctness", "clean", "profiling")
                        if key in values
                    )
                )
            declared_options = round_data.get("declared_option_changes", [])
            if isinstance(declared_options, list):
                for change in declared_options:
                    if isinstance(change, dict) and isinstance(
                        change.get("pointer"), str
                    ):
                        options.append(change["pointer"])
        classifications = {}
        decisions = experiment.get("performance_scope_decisions", [])
        if isinstance(decisions, list):
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                classification = decision.get("classification")
                if isinstance(classification, str):
                    classifications[classification] = (
                        classifications.get(classification, 0) + 1
                    )
        config = "; ".join(
            f"{key}={experiment[key]}"
            for key in (
                "source_revision",
                "baseline_config_fingerprint",
                "control_fingerprint",
            )
            if isinstance(experiment.get(key), str)
        )
        classification_text = ", ".join(
            f"{key}={classifications[key]}" for key in sorted(classifications)
        )
        blockers = experiment.get("blockers", [])
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                experiment_id,
                config or "-",
                "<br>".join(round_ids) or "-",
                "<br>".join(lifecycle) or "-",
                "<br>".join(sorted(set(options))) or "-",
                classification_text or "-",
                str(len(blockers)),
            )
        )
    lines.extend(["", "## 阶段交接", ""])
    for step in session["steps"]:
        lines.append(f"- `{step['step_id']}` / `{step['agent_id']}`: `{step['state']}`")
    _atomic_text(target, "\n".join(lines) + "\n")
    return target


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--session", required=True, type=Path)
    initialize.add_argument("--artifact-root", required=True, type=Path)
    initialize.add_argument("--session-id", required=True)
    initialize.add_argument(
        "--optional-mode", required=True, choices=sorted(OPTIONAL_MODES)
    )
    derive = commands.add_parser("derive-session")
    source_range = commands.add_parser("source-range-from-smap")
    for derived_parser in (derive, source_range):
        derived_parser.add_argument("--parent-session", required=True, type=Path)
        derived_parser.add_argument("--session", required=True, type=Path)
        derived_parser.add_argument("--artifact-root", required=True, type=Path)
        derived_parser.add_argument("--session-id", required=True)
        derived_parser.add_argument("--profiling-analysis", required=True, type=Path)
        derived_parser.add_argument("--ledger", required=True, type=Path)
        derived_parser.add_argument("--approve-imported-evidence", action="store_true")
    derive.add_argument("--source-scope-map", type=Path)
    source_range.add_argument("--source-scope-map", required=True, type=Path)
    derive.add_argument(
        "--optional-mode",
        required=True,
        choices=("multistream", "source-range", "both"),
    )
    next_parser = commands.add_parser("next")
    next_parser.add_argument("--session", required=True, type=Path)
    dispatch = commands.add_parser("dispatch")
    dispatch.add_argument("--session", required=True, type=Path)
    dispatch.add_argument("--output", required=True, type=Path)
    run_next = commands.add_parser("run-next")
    run_next.add_argument("--session", required=True, type=Path)
    run_next.add_argument("--runner-command-json", required=True, type=Path)
    run_next.add_argument("--timeout-seconds", type=float, default=3600)
    resume = commands.add_parser("resume")
    resume.add_argument("--session", required=True, type=Path)
    seal = commands.add_parser("seal")
    seal.add_argument("--session", required=True, type=Path)
    seal.add_argument("--result", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--session", required=True, type=Path)
    report = commands.add_parser("render-final-report")
    report.add_argument("--session", required=True, type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument(
        "--summary",
        type=Path,
        help="Read-only report_summary JSON override for historical reports",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            output = initialize_session(
                session_path=args.session,
                artifact_root=args.artifact_root,
                session_id=args.session_id,
                optional_mode=args.optional_mode,
            )
        elif args.command in {"derive-session", "source-range-from-smap"}:
            output = derive_session(
                parent_session_path=args.parent_session,
                session_path=args.session,
                artifact_root=args.artifact_root,
                session_id=args.session_id,
                optional_mode=(
                    args.optional_mode
                    if args.command == "derive-session"
                    else "source-range"
                ),
                profiling_analysis=args.profiling_analysis,
                ledger=args.ledger,
                source_scope_map=args.source_scope_map,
                approve_imported_evidence=args.approve_imported_evidence,
            )
        elif args.command == "next":
            output = next_step(load_session(args.session))
        elif args.command == "dispatch":
            output = write_dispatch_task(args.session, args.output)
        elif args.command == "run-next":
            output = run_agent_step(
                args.session,
                _read_json(args.runner_command_json),
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "resume":
            output = resume_session(args.session)
        elif args.command == "seal":
            output = seal_handoff(args.session, _read_json(args.result))
        elif args.command == "verify":
            output = verify_session(args.session)
            print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if output["valid"] else 1
        else:
            output = {
                "report": str(
                    render_final_report(args.session, args.output, args.summary)
                )
            }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"valid": False, "errors": [str(error)]}, ensure_ascii=False))
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
