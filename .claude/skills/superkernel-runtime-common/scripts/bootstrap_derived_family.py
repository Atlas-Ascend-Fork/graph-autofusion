#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Register an accepted multi-stream result as a fresh derived-family SEED."""

import argparse
import fcntl
import hashlib
import json
import sys
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MULTISTREAM_SCRIPTS = (
    SCRIPT_DIR.parent.parent / "superkernel-multistream-performance-tuning" / "scripts"
)

sys.path.insert(0, str(MULTISTREAM_SCRIPTS))
import multistream_contract  # noqa: E402


REGISTRY_SCHEMA = "superkernel-derived-family-registry-v1"
LINEAGE_SCHEMA = "superkernel-derived-family-lineage-v1"
IDENTITY_FIELDS = (
    "candidate_name",
    "source_revision",
    "source_fingerprint",
    "config_fingerprint",
    "control_fingerprint",
    "workload_fingerprint",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_fingerprint(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _identity(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    identity = {}
    for field in IDENTITY_FIELDS:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}.{field} must be a non-empty string")
        identity[field] = item
    return identity


def _read_registry(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": REGISTRY_SCHEMA, "families": {}}
    registry = _load_json(path)
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != REGISTRY_SCHEMA
    ):
        raise ValueError(f"registry must use {REGISTRY_SCHEMA}")
    if not isinstance(registry.get("families"), dict):
        raise ValueError("registry.families must be an object")
    return registry


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _register_family(
    request_path,
    result_path,
    registry_path,
    family_root,
    request,
    result,
    validation,
    request_identity,
):
    artifact_root = request_path.parent
    selected = result["selected_candidate"]
    family_id = selected["derived_experiment_id"]
    if not SAFE_ID.fullmatch(family_id):
        raise ValueError("derived_experiment_id contains unsafe path characters")
    family_root = Path(family_root).resolve()
    family_root.mkdir(parents=True, exist_ok=True)
    destination = family_root / family_id
    registry = _read_registry(registry_path)

    selected_identity = _identity(selected, "selected candidate")
    accepted_trial = next(
        trial for trial in result["trials"] if trial["trial_id"] == selected["trial_id"]
    )
    execution_plan = _load_json(artifact_root / accepted_trial["execution_plan"])
    audit_artifacts = {
        field: {
            "path": accepted_trial[field],
            "file_fingerprint": _file_fingerprint(
                artifact_root / accepted_trial[field]
            ),
        }
        for field in (
            "action_manifest",
            "execution_state",
            "execution_plan",
            "model_adapter",
            "plan_compilation",
        )
    }
    audit_artifacts["phase_manifests"] = [
        {
            "phase_id": phase["phase_id"],
            "path": phase["manifest"],
            "file_fingerprint": _file_fingerprint(artifact_root / phase["manifest"]),
        }
        for phase in execution_plan["phases"]
    ]
    lineage = {
        "schema_version": LINEAGE_SCHEMA,
        "derived_experiment_id": family_id,
        "parent_experiment_id": request["parent_experiment_id"],
        "request_id": request["request_id"],
        "request_fingerprint": validation["request_fingerprint"],
        "result_fingerprint": _fingerprint(result),
        "accepted_trial_id": selected["trial_id"],
        "change_kind": accepted_trial["change_kind"],
        "source_artifact_root": str(artifact_root),
        "execution_audit_artifacts": audit_artifacts,
        "incumbent_identity": request_identity,
        "seed_identity": selected_identity,
        "evidence_invalidation": {
            "config_profile_binding": True,
            "source_revision_and_map": accepted_trial["change_kind"] != "option",
            "isolated_trial_not_base_evidence": True,
        },
        "fresh_base_required": True,
        "fresh_base_round_id": f"{family_id}-BASE",
    }
    lineage_fingerprint = _fingerprint(lineage)

    existing = registry["families"].get(family_id)
    if existing is not None:
        if existing.get("lineage_fingerprint") != lineage_fingerprint:
            raise ValueError(
                f"derived family ID already belongs to different lineage: {family_id}"
            )
        if not destination.is_dir():
            raise ValueError(
                "registry contains derived family but its directory is missing"
            )
        return existing

    if destination.exists():
        existing_lineage_path = destination / "lineage.json"
        if not existing_lineage_path.is_file():
            raise ValueError(
                f"derived family directory already exists without lineage: {destination}"
            )
        existing_lineage = _load_json(existing_lineage_path)
        if _fingerprint(existing_lineage) != lineage_fingerprint:
            raise ValueError(
                f"derived family directory has conflicting lineage: {destination}"
            )
    else:
        with tempfile.TemporaryDirectory(
            prefix=f".{family_id}.staging-", dir=family_root
        ) as staging:
            staging_path = Path(staging)
            seed = staging_path / "SEED"
            seed.mkdir()
            (staging_path / "lineage.json").write_text(
                json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            shutil.copy2(request_path, staging_path / "multistream-request.json")
            shutil.copy2(result_path, staging_path / "multistream-result.json")
            for field, name in (
                (
                    "config_snapshot",
                    "config-snapshot" + Path(selected["config_snapshot"]).suffix,
                ),
                ("config_manifest", "config-manifest.json"),
                ("source_manifest", "source-manifest.json"),
            ):
                shutil.copy2(artifact_root / selected[field], seed / name)
            shutil.copy2(
                artifact_root / accepted_trial["action_manifest"],
                seed / "action-manifest.json",
            )
            audit = seed / "execution-audit"
            audit.mkdir()
            for field, name in (
                ("execution_state", "execution-state.json"),
                ("execution_plan", "execution-plan.json"),
                ("model_adapter", "model-adapter.json"),
                ("plan_compilation", "plan-compilation.json"),
            ):
                shutil.copy2(artifact_root / accepted_trial[field], audit / name)
            phase_root = audit / "phase-manifests"
            phase_root.mkdir()
            for phase in execution_plan["phases"]:
                shutil.copy2(
                    artifact_root / phase["manifest"],
                    phase_root / f"{phase['phase_id']}.json",
                )
            os.replace(staging_path, destination)

    entry = {
        "derived_experiment_id": family_id,
        "status": "seed_registered",
        "family_path": str(destination),
        "lineage_fingerprint": lineage_fingerprint,
        "fresh_base_round_id": lineage["fresh_base_round_id"],
        "fresh_base_required": True,
        "ledger_merge_allowed": False,
        "registered_at": _now(),
    }
    registry["families"][family_id] = entry
    _atomic_json(registry_path, registry)
    return entry


def bootstrap(
    request_path, result_path, current_incumbent_path, registry_path, family_root
):
    request_path = Path(request_path).resolve()
    result_path = Path(result_path).resolve()
    artifact_root = request_path.parent
    request = _load_json(request_path)
    result = _load_json(result_path)
    validation = multistream_contract.validate_result(request, result, artifact_root)
    if validation["status"] != "accepted":
        raise ValueError(
            "only an accepted multi-stream result can create a derived family"
        )

    current_value = _load_json(current_incumbent_path)
    current = current_value.get("incumbent", current_value)
    current_identity = _identity(current, "current incumbent")
    request_identity = _identity(request["incumbent"], "request incumbent")
    if current_identity != request_identity:
        raise ValueError(
            "stale_result: current incumbent differs from request incumbent"
        )

    registry_path = Path(registry_path).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _register_family(
            request_path,
            result_path,
            registry_path,
            family_root,
            request,
            result,
            validation,
            request_identity,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--current-incumbent", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        entry = bootstrap(
            args.request,
            args.result,
            args.current_incumbent,
            args.registry,
            args.family_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.json_out:
        _atomic_json(args.json_out, entry)
    print(json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
