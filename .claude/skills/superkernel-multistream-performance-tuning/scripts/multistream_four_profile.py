#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run and seal incumbent/candidate x SK-off/SK-on mechanism profiles."""

import argparse
import json
from pathlib import Path

import multistream_runner


PLAN_SCHEMA = "superkernel-multistream-four-profile-plan-v1"
PROFILE_MANIFEST_SCHEMA = "superkernel-multistream-profile-role-manifest-v1"
SUMMARY_SCHEMA = "superkernel-multistream-four-profile-summary-v1"
ROLES = (
    "incumbent_sk_off",
    "incumbent_sk_on",
    "candidate_sk_off",
    "candidate_sk_on",
)


def _safe_output(root, relative, label):
    return multistream_runner._under(root, relative, label)


def _profile_logs(profile):
    manifest = Path(profile["manifest"])
    stem = manifest.name[:-5] if manifest.name.endswith(".json") else manifest.name
    parent = manifest.parent
    return {
        "command_stdout": (parent / f"{stem}.command.stdout.log").as_posix(),
        "command_stderr": (parent / f"{stem}.command.stderr.log").as_posix(),
        "validator_stdout": (parent / f"{stem}.validator.stdout.log").as_posix(),
        "validator_stderr": (parent / f"{stem}.validator.stderr.log").as_posix(),
    }


def _profile_environment(plan, profile):
    return multistream_runner.lease_environment(
        {
            **plan["environment"],
            **profile["environment_overrides"],
        },
        plan["device_ids"],
        plan["lease_root"],
    )


def _validate_programs(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = []
    seen = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "file_fingerprint"}:
            raise ValueError(f"{label}[{index}] fields are invalid")
        path = Path(item["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"{label}[{index}].path must be an existing absolute file")
        path = str(path.resolve())
        if path in seen:
            raise ValueError(f"{label} contains a duplicate path")
        seen.add(path)
        actual = multistream_runner.file_fingerprint(path)
        if item["file_fingerprint"] != actual:
            raise ValueError(f"{label}[{index}] fingerprint mismatch")
        result.append({"path": path, "file_fingerprint": actual})
    return sorted(result, key=lambda item: item["path"])


def validate_plan(value, *, require_fingerprint=True):
    required = {
        "schema_version",
        "plan_id",
        "trial_id",
        "request_fingerprint",
        "workspace_root",
        "artifact_root",
        "lease_root",
        "device_ids",
        "environment",
        "profiles",
        "summary",
    }
    if require_fingerprint:
        required.add("plan_fingerprint")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"four-profile plan must contain exactly {sorted(required)}")
    if value["schema_version"] != PLAN_SCHEMA:
        raise ValueError(f"four-profile plan must use {PLAN_SCHEMA}")
    for field in ("plan_id", "trial_id", "request_fingerprint"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"four-profile plan {field} must be non-empty")
    roots = {
        name: multistream_runner._absolute_root(value[name], name)
        for name in ("workspace_root", "artifact_root", "lease_root")
    }
    device_ids = value["device_ids"]
    if (
        not isinstance(device_ids, list)
        or not device_ids
        or device_ids != sorted(device_ids)
        or len(device_ids) != len(set(device_ids))
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in device_ids
        )
    ):
        raise ValueError("device_ids must be sorted unique non-negative integers")
    environment = multistream_runner._validate_environment(value["environment"])
    profiles = value["profiles"]
    if not isinstance(profiles, list) or [
        item.get("role") for item in profiles if isinstance(item, dict)
    ] != list(ROLES):
        raise ValueError(f"profiles must exactly follow {list(ROLES)}")
    normalized = []
    outputs = set()
    for index, profile in enumerate(profiles):
        label = f"profiles[{index}]"
        fields = {
            "role",
            "argv",
            "validator_argv",
            "cwd",
            "timeout_seconds",
            "validator_timeout_seconds",
            "environment_overrides",
            "program_files",
            "required_artifacts",
            "manifest",
        }
        if set(profile) != fields:
            raise ValueError(f"{label} must contain exactly {sorted(fields)}")
        for timeout in ("timeout_seconds", "validator_timeout_seconds"):
            if (
                isinstance(profile[timeout], bool)
                or not isinstance(profile[timeout], (int, float))
                or profile[timeout] <= 0
            ):
                raise ValueError(f"{label}.{timeout} must be positive")
        programs = _validate_programs(
            profile["program_files"], f"{label}.program_files"
        )
        program_paths = {item["path"] for item in programs}
        argv = multistream_runner._validate_argv(profile["argv"], f"{label}.argv")
        validator = multistream_runner._validate_argv(
            profile["validator_argv"], f"{label}.validator_argv"
        )
        for executable in (argv[0], validator[0]):
            path = Path(executable)
            if not path.is_absolute() or str(path.resolve()) not in program_paths:
                raise ValueError(f"{label} executable is not sealed in program_files")
        artifacts = profile["required_artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"{label}.required_artifacts must be non-empty")
        artifacts = [
            multistream_runner._safe_relative(item, f"{label}.required_artifacts[]")
            for item in artifacts
        ]
        manifest = multistream_runner._safe_relative(
            profile["manifest"], f"{label}.manifest"
        )
        role_outputs = artifacts + [manifest] + list(_profile_logs(profile).values())
        if len(role_outputs) != len(set(role_outputs)) or outputs & set(role_outputs):
            raise ValueError(f"{label} reuses an output path")
        outputs.update(role_outputs)
        normalized.append(
            {
                **profile,
                "argv": argv,
                "validator_argv": validator,
                "cwd": multistream_runner._safe_relative(
                    profile["cwd"], f"{label}.cwd"
                ),
                "environment_overrides": multistream_runner._validate_environment(
                    profile["environment_overrides"]
                ),
                "program_files": programs,
                "required_artifacts": artifacts,
                "manifest": manifest,
            }
        )
    summary = multistream_runner._safe_relative(value["summary"], "summary")
    if summary in outputs:
        raise ValueError("summary collides with a profile output")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": value["plan_id"],
        "trial_id": value["trial_id"],
        "request_fingerprint": value["request_fingerprint"],
        **roots,
        "device_ids": device_ids,
        "environment": environment,
        "profiles": normalized,
        "summary": summary,
    }
    plan["plan_fingerprint"] = multistream_runner.content_fingerprint(plan)
    if require_fingerprint and value["plan_fingerprint"] != plan["plan_fingerprint"]:
        raise ValueError("four-profile plan fingerprint mismatch")
    return plan


def freeze_plan(draft):
    if "plan_fingerprint" in draft:
        raise ValueError("draft four-profile plan must not contain plan_fingerprint")
    return validate_plan(draft, require_fingerprint=False)


def _lease_context(plan):
    try:
        held = multistream_runner.parent_lease_held(
            plan["device_ids"], plan["lease_root"]
        )
    except ValueError as error:
        raise ValueError(f"four-profile parent lease is invalid: {error}") from error
    if not held:
        raise ValueError("four-profile execution lacks matching parent lease markers")


def _sealed(path, relative):
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "file_fingerprint": multistream_runner.file_fingerprint(path),
    }


def validate_profile_manifest(
    value, plan, profile, artifact_root, *, require_passed=True
):
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PROFILE_MANIFEST_SCHEMA
    ):
        raise ValueError(f"profile role manifest must use {PROFILE_MANIFEST_SCHEMA}")
    expected = {
        "plan_id": plan["plan_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "role": profile["role"],
        "environment": _profile_environment(plan, profile),
    }
    for field, item in expected.items():
        if value.get(field) != item:
            raise ValueError(f"profile role manifest {field} mismatch")
    if require_passed and value.get("status") != "passed":
        raise ValueError("profile role manifest is not passed")
    for name, argv in (
        ("command", profile["argv"]),
        ("validator", profile["validator_argv"]),
    ):
        record = value.get(name)
        if not isinstance(record, dict) or record.get("argv") != argv:
            raise ValueError(f"profile role manifest {name} mismatch")
        if value.get("status") == "passed" and (
            record.get("return_code") != 0 or record.get("timed_out") is not False
        ):
            raise ValueError(f"passed profile role has unsuccessful {name}")
    files = value.get("sealed_files")
    if not isinstance(files, list):
        raise ValueError("profile role manifest sealed_files must be a list")
    indexed = {item.get("path"): item for item in files if isinstance(item, dict)}
    required = profile["required_artifacts"] + list(_profile_logs(profile).values())
    if value.get("status") == "passed" and set(indexed) != set(required):
        raise ValueError("passed profile role manifest has incomplete sealed files")
    for relative in required:
        path = _safe_output(artifact_root, relative, "profile sealed file")
        item = indexed.get(relative)
        if (
            not path.is_file()
            or item is None
            or item.get("size_bytes") != path.stat().st_size
            or item.get("file_fingerprint") != multistream_runner.file_fingerprint(path)
        ):
            raise ValueError(f"profile sealed file changed: {relative}")
    return value


def _write_manifest(plan, profile, artifact_root, status, reason, records):
    sealed = []
    for relative in profile["required_artifacts"] + list(
        _profile_logs(profile).values()
    ):
        path = _safe_output(artifact_root, relative, "profile output")
        if path.is_file():
            sealed.append(_sealed(path, relative))
    value = {
        "schema_version": PROFILE_MANIFEST_SCHEMA,
        "plan_id": plan["plan_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "role": profile["role"],
        "status": status,
        "reason": reason,
        "environment": _profile_environment(plan, profile),
        "command": records.get(
            "command",
            {"argv": profile["argv"], "return_code": None, "timed_out": False},
        ),
        "validator": records.get(
            "validator",
            {
                "argv": profile["validator_argv"],
                "return_code": None,
                "timed_out": False,
            },
        ),
        "sealed_files": sealed,
    }
    multistream_runner._atomic_write_json(
        _safe_output(artifact_root, profile["manifest"], "profile manifest"), value
    )
    return value


def _run_profile(plan, profile, workspace_root, artifact_root):
    manifest_path = _safe_output(artifact_root, profile["manifest"], "profile manifest")
    if manifest_path.is_file():
        return validate_profile_manifest(
            json.loads(manifest_path.read_text()), plan, profile, artifact_root
        )
    cwd = _safe_output(workspace_root, profile["cwd"], "profile cwd")
    if not cwd.is_dir():
        raise ValueError(f"profile cwd does not exist: {profile['cwd']}")
    logs = _profile_logs(profile)
    all_outputs = profile["required_artifacts"] + list(logs.values())
    paths = [
        _safe_output(artifact_root, item, "profile output") for item in all_outputs
    ]
    if any(path.exists() for path in paths):
        raise ValueError(f"profile {profile['role']} has outputs without a manifest")
    for path in paths + [manifest_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    environment = _profile_environment(plan, profile)
    try:
        records["command"] = multistream_runner._run_argv(
            profile["argv"],
            cwd,
            environment,
            _safe_output(artifact_root, logs["command_stdout"], "command stdout"),
            _safe_output(artifact_root, logs["command_stderr"], "command stderr"),
            profile["timeout_seconds"],
        )
    except OSError as error:
        return _write_manifest(
            plan,
            profile,
            artifact_root,
            "failed",
            f"process_start_failed: {error}",
            records,
        )
    if records["command"]["return_code"] != 0 or records["command"]["timed_out"]:
        return _write_manifest(
            plan, profile, artifact_root, "failed", "command_failed", records
        )
    try:
        records["validator"] = multistream_runner._run_argv(
            profile["validator_argv"],
            cwd,
            environment,
            _safe_output(artifact_root, logs["validator_stdout"], "validator stdout"),
            _safe_output(artifact_root, logs["validator_stderr"], "validator stderr"),
            profile["validator_timeout_seconds"],
        )
    except OSError as error:
        return _write_manifest(
            plan,
            profile,
            artifact_root,
            "failed",
            f"validator_start_failed: {error}",
            records,
        )
    if records["validator"]["return_code"] != 0 or records["validator"]["timed_out"]:
        return _write_manifest(
            plan, profile, artifact_root, "failed", "validator_failed", records
        )
    missing = [
        item
        for item, path in zip(profile["required_artifacts"], paths)
        if not path.is_file()
    ]
    if missing:
        return _write_manifest(
            plan,
            profile,
            artifact_root,
            "failed",
            f"required artifacts missing: {missing}",
            records,
        )
    manifest = _write_manifest(plan, profile, artifact_root, "passed", None, records)
    return validate_profile_manifest(manifest, plan, profile, artifact_root)


def _build_summary(plan, manifests, artifact_root):
    roles = {}
    for profile, manifest in zip(plan["profiles"], manifests):
        manifest_path = _safe_output(
            artifact_root, profile["manifest"], "profile manifest"
        )
        roles[profile["role"]] = {
            "manifest": profile["manifest"],
            "manifest_file_fingerprint": multistream_runner.file_fingerprint(
                manifest_path
            ),
            "artifacts": [
                item
                for item in manifest["sealed_files"]
                if item["path"] in profile["required_artifacts"]
            ],
        }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "plan_id": plan["plan_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "roles": roles,
        "comparisons": {
            "incumbent_fusion": ["incumbent_sk_off", "incumbent_sk_on"],
            "candidate_fusion": ["candidate_sk_off", "candidate_sk_on"],
            "sk_off_change": ["incumbent_sk_off", "candidate_sk_off"],
            "sk_on_change": ["incumbent_sk_on", "candidate_sk_on"],
        },
        "complete": True,
    }
    summary["summary_fingerprint"] = multistream_runner.content_fingerprint(summary)
    return summary


def validate_summary(summary_path, plan, artifact_root):
    value = json.loads(Path(summary_path).read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError(f"four-profile summary must use {SUMMARY_SCHEMA}")
    actual = value.get("summary_fingerprint")
    unsigned = {
        key: item for key, item in value.items() if key != "summary_fingerprint"
    }
    if actual != multistream_runner.content_fingerprint(unsigned):
        raise ValueError("four-profile summary fingerprint mismatch")
    manifests = []
    for profile in plan["profiles"]:
        path = _safe_output(artifact_root, profile["manifest"], "profile manifest")
        manifest = validate_profile_manifest(
            json.loads(path.read_text()), plan, profile, artifact_root
        )
        manifests.append(manifest)
    if _build_summary(plan, manifests, artifact_root) != value:
        raise ValueError("four-profile summary differs from deterministic replay")
    return {"valid": True, "complete": True, "summary_fingerprint": actual}


def run_plan(plan):
    plan = validate_plan(plan)
    _lease_context(plan)
    workspace_root = Path(plan["workspace_root"])
    artifact_root = Path(plan["artifact_root"])
    summary_path = _safe_output(artifact_root, plan["summary"], "four-profile summary")
    if summary_path.is_file():
        validate_summary(summary_path, plan, artifact_root)
        return json.loads(summary_path.read_text())
    manifests = []
    for profile in plan["profiles"]:
        manifest = _run_profile(plan, profile, workspace_root, artifact_root)
        manifests.append(manifest)
        if manifest["status"] != "passed":
            raise ValueError(
                f"four-profile role failed: {profile['role']}: {manifest['reason']}"
            )
    summary = _build_summary(plan, manifests, artifact_root)
    multistream_runner._atomic_write_json(summary_path, summary)
    validate_summary(summary_path, plan, artifact_root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-plan")
    freeze.add_argument("--draft", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    check = commands.add_parser("validate-summary")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = (
            validate_plan(json.loads(args.plan.read_text()))
            if hasattr(args, "plan")
            else None
        )
        if args.command == "freeze-plan":
            result = freeze_plan(json.loads(args.draft.read_text()))
            if args.out.exists():
                raise ValueError(f"plan output already exists: {args.out}")
            multistream_runner._atomic_write_json(args.out, result)
        elif args.command == "run":
            result = run_plan(plan)
        else:
            result = validate_summary(args.summary, plan, plan["artifact_root"])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
