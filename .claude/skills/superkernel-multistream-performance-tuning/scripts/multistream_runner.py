#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run a frozen multistream trial plan with device leases and sealed evidence."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import multistream_execution
import multistream_component_reorder
import multistream_event_stage_dispatch
import multistream_operator_order


PLAN_SCHEMA = "superkernel-multistream-execution-plan-v1"
PHASE_SCHEMA = "superkernel-multistream-phase-manifest-v1"
VALIDATOR_EXIT_ACTIONS = {"pass", "reject", "block", "fail"}
PHASE_STATES = (
    "correctness_passed",
    "profile_collected",
    "analysis_validated",
    "clean3_passed",
    "clean5_passed",
)
DEVICE_PHASE_STATES = {
    "correctness_passed",
    "profile_collected",
    "clean3_passed",
    "clean5_passed",
}
SECRET_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSWD|SECRET|PRIVATE|CREDENTIAL|API_KEY|ACCESS_KEY|AUTHORIZATION|COOKIE)",
    re.I,
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
LEASE_MARKERS = (
    "SUPERKERNEL_DEVICE_LEASE_HELD",
    "SUPERKERNEL_DEVICE_IDS",
    "SUPERKERNEL_DEVICE_LEASE_ROOT",
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


def file_fingerprint(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path, value):
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


def _safe_relative(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _under(root, relative, label):
    root = Path(root).resolve()
    candidate = (root / _safe_relative(relative, label)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    return candidate


def _absolute_root(value, label):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty absolute path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return str(path.resolve())


def _validate_argv(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty argv array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} entries must be non-empty strings")
    return list(value)


def _validate_environment(value):
    if not isinstance(value, dict):
        raise ValueError("environment must be an object")
    normalized = {}
    for name, item in value.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if SECRET_NAME.search(name):
            raise ValueError(f"secret-like environment variable is forbidden: {name}")
        if not isinstance(item, str) or "\x00" in item:
            raise ValueError(f"environment value must be a string without NUL: {name}")
        normalized[name] = item
    return dict(sorted(normalized.items()))


def lease_environment(environment, device_ids, lease_root):
    """Add canonical child markers and reject a conflicting inherited lease."""
    environment = dict(environment)
    expected = {
        "SUPERKERNEL_DEVICE_LEASE_HELD": "1",
        "SUPERKERNEL_DEVICE_IDS": ",".join(str(item) for item in sorted(device_ids)),
        "SUPERKERNEL_DEVICE_LEASE_ROOT": str(Path(lease_root).resolve()),
    }
    present = {name for name in LEASE_MARKERS if name in environment}
    if present and present != set(LEASE_MARKERS):
        raise ValueError("device lease environment contains partial parent markers")
    for name, value in expected.items():
        if name in environment and environment[name] != value:
            raise ValueError(f"device lease environment conflicts with {name}")
    environment.update(expected)
    return environment


def parent_lease_held(device_ids, lease_root, environment=None):
    environment = os.environ if environment is None else environment
    present = {name for name in LEASE_MARKERS if name in environment}
    if not present:
        return False
    if present != set(LEASE_MARKERS):
        raise ValueError("parent device lease markers are incomplete")
    expected = lease_environment({}, device_ids, lease_root)
    for name, value in expected.items():
        if environment.get(name) != value:
            raise ValueError(f"parent device lease marker mismatch: {name}")
    return True


def _phase_environment(plan, phase):
    environment = {**plan["environment"], **phase["environment_overrides"]}
    if phase["requires_device"]:
        environment = lease_environment(
            environment, plan["device_ids"], plan["lease_root"]
        )
    return environment


def _plan_payload(plan):
    return {key: value for key, value in plan.items() if key != "plan_fingerprint"}


def validate_plan(plan, require_fingerprint=True):
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"execution plan must use {PLAN_SCHEMA}")
    expected_keys = {
        "schema_version",
        "trial_id",
        "request_fingerprint",
        "environment",
        "workspace_root",
        "artifact_root",
        "lease_root",
        "device_ids",
        "lease_timeout_seconds",
        "phases",
    }
    if require_fingerprint:
        expected_keys.add("plan_fingerprint")
    optional_keys = {"pre_profile_evidence"}
    if (
        not expected_keys.issubset(set(plan))
        or set(plan) - expected_keys - optional_keys
    ):
        raise ValueError(f"execution plan must contain exactly {sorted(expected_keys)}")
    for field in ("trial_id", "request_fingerprint"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            raise ValueError(f"execution plan {field} must be non-empty")
    if not IDENTIFIER.fullmatch(plan["trial_id"]):
        raise ValueError("execution plan trial_id is unsafe")
    environment = _validate_environment(plan.get("environment"))
    device_ids = plan.get("device_ids")
    if (
        not isinstance(device_ids, list)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in device_ids
        )
        or len(set(device_ids)) != len(device_ids)
    ):
        raise ValueError("device_ids must be unique non-negative integers")
    if device_ids != sorted(device_ids):
        raise ValueError("device_ids must be sorted to guarantee lock ordering")
    lease_timeout = plan.get("lease_timeout_seconds")
    if (
        not isinstance(lease_timeout, (int, float))
        or isinstance(lease_timeout, bool)
        or lease_timeout <= 0
    ):
        raise ValueError("lease_timeout_seconds must be positive")
    phases = plan.get("phases")
    if not isinstance(phases, list) or [
        item.get("state_after") for item in phases if isinstance(item, dict)
    ] != list(PHASE_STATES):
        raise ValueError(
            f"execution plan phases must exactly follow {list(PHASE_STATES)}"
        )
    pre_profile = plan.get("pre_profile_evidence")
    normalized_pre_profile = None
    if pre_profile is not None:
        if not isinstance(pre_profile, dict) or set(pre_profile) != {"kind", "path"}:
            raise ValueError("pre_profile_evidence must contain kind and path")
        if pre_profile.get("kind") not in {
            "dispatch_order",
            "component_dispatch_order",
            "event_stage_dispatch",
        }:
            raise ValueError(
                "pre_profile_evidence.kind must be dispatch_order, component_dispatch_order, or event_stage_dispatch"
            )
        normalized_pre_profile = {
            "kind": pre_profile["kind"],
            "path": _safe_relative(
                pre_profile.get("path"), "pre_profile_evidence.path"
            ),
        }
    seen_paths = set()
    seen_phase_ids = set()
    normalized_phases = []
    for index, phase in enumerate(phases):
        required_keys = {
            "phase_id",
            "state_after",
            "argv",
            "validator_argv",
            "cwd",
            "timeout_seconds",
            "validator_timeout_seconds",
            "requires_device",
            "environment_overrides",
            "validator_exit_actions",
            "program_files",
            "required_artifacts",
            "manifest",
        }
        if set(phase) != required_keys:
            raise ValueError(
                f"phases[{index}] must contain exactly {sorted(required_keys)}"
            )
        if not isinstance(phase["phase_id"], str) or not IDENTIFIER.fullmatch(
            phase["phase_id"]
        ):
            raise ValueError(f"phases[{index}].phase_id is unsafe")
        if phase["phase_id"] in seen_phase_ids:
            raise ValueError(f"duplicate phase_id: {phase['phase_id']}")
        seen_phase_ids.add(phase["phase_id"])
        for timeout_name in ("timeout_seconds", "validator_timeout_seconds"):
            timeout = phase[timeout_name]
            if (
                not isinstance(timeout, (int, float))
                or isinstance(timeout, bool)
                or timeout <= 0
            ):
                raise ValueError(f"phases[{index}].{timeout_name} must be positive")
        if not isinstance(phase["requires_device"], bool):
            raise ValueError(f"phases[{index}].requires_device must be boolean")
        expected_requires_device = phase["state_after"] in DEVICE_PHASE_STATES
        if phase["requires_device"] is not expected_requires_device:
            raise ValueError(
                f"phases[{index}].requires_device must be {expected_requires_device} "
                f"for {phase['state_after']}"
            )
        if phase["requires_device"] and not device_ids:
            raise ValueError(
                f"phases[{index}] requires a device but device_ids is empty"
            )
        exit_actions = phase["validator_exit_actions"]
        if not isinstance(exit_actions, dict) or exit_actions.get("0") != "pass":
            raise ValueError(
                f"phases[{index}].validator_exit_actions must map exit code 0 to pass"
            )
        for exit_code, action in exit_actions.items():
            if (
                not isinstance(exit_code, str)
                or not exit_code.isdigit()
                or not 0 <= int(exit_code) <= 255
            ):
                raise ValueError(f"phases[{index}] has an invalid validator exit code")
            if action not in VALIDATOR_EXIT_ACTIONS:
                raise ValueError(
                    f"phases[{index}] has an invalid validator exit action"
                )
            if action == "pass" and exit_code != "0":
                raise ValueError(f"phases[{index}] only exit code 0 may map to pass")
        reject_codes = [
            code for code, action in exit_actions.items() if action == "reject"
        ]
        if (
            phase["state_after"] in {"clean3_passed", "clean5_passed"}
            and not reject_codes
        ):
            raise ValueError(
                f"phases[{index}] clean validator must define a reject exit code"
            )
        if (
            phase["state_after"] not in {"clean3_passed", "clean5_passed"}
            and reject_codes
        ):
            raise ValueError(f"phases[{index}] reject is allowed only for clean phases")
        program_files = phase["program_files"]
        if not isinstance(program_files, list) or not program_files:
            raise ValueError(f"phases[{index}].program_files must be non-empty")
        normalized_programs = []
        seen_programs = set()
        for program_index, program in enumerate(program_files):
            if not isinstance(program, dict) or set(program) != {
                "path",
                "file_fingerprint",
            }:
                raise ValueError(
                    f"phases[{index}].program_files[{program_index}] must contain path and file_fingerprint"
                )
            path = Path(program["path"])
            if not path.is_absolute() or not path.is_file():
                raise ValueError(
                    f"phases[{index}] program file must be an existing absolute file"
                )
            resolved = str(path.resolve())
            if resolved in seen_programs:
                raise ValueError(
                    f"phases[{index}] has duplicate program file: {resolved}"
                )
            seen_programs.add(resolved)
            actual = file_fingerprint(resolved)
            if program["file_fingerprint"] != actual:
                raise ValueError(
                    f"phases[{index}] program file fingerprint mismatch: {resolved}"
                )
            normalized_programs.append({"path": resolved, "file_fingerprint": actual})
        artifacts = phase["required_artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"phases[{index}].required_artifacts must be non-empty")
        normalized_artifacts = []
        for artifact_index, artifact in enumerate(artifacts):
            relative = _safe_relative(
                artifact, f"phases[{index}].required_artifacts[{artifact_index}]"
            )
            if relative in seen_paths:
                raise ValueError(f"artifact path is reused across phases: {relative}")
            seen_paths.add(relative)
            normalized_artifacts.append(relative)
        manifest = _safe_relative(phase["manifest"], f"phases[{index}].manifest")
        if manifest in seen_paths:
            raise ValueError(f"phase manifest collides with an artifact: {manifest}")
        seen_paths.add(manifest)
        normalized_phase = {
            **phase,
            "argv": _validate_argv(phase["argv"], f"phases[{index}].argv"),
            "validator_argv": _validate_argv(
                phase["validator_argv"], f"phases[{index}].validator_argv"
            ),
            "cwd": _safe_relative(phase["cwd"], f"phases[{index}].cwd"),
            "environment_overrides": _validate_environment(
                phase["environment_overrides"]
            ),
            "validator_exit_actions": dict(
                sorted(exit_actions.items(), key=lambda item: int(item[0]))
            ),
            "program_files": sorted(normalized_programs, key=lambda item: item["path"]),
            "required_artifacts": normalized_artifacts,
            "manifest": manifest,
        }
        for executable in (
            normalized_phase["argv"][0],
            normalized_phase["validator_argv"][0],
        ):
            executable_path = Path(executable)
            if not executable_path.is_absolute() or not executable_path.is_file():
                raise ValueError(
                    f"phases[{index}] command executables must be existing absolute files"
                )
            if str(executable_path.resolve()) not in seen_programs:
                raise ValueError(
                    f"phases[{index}] executable is missing from program_files: {executable}"
                )
        for log_path in _phase_log_paths(normalized_phase).values():
            if log_path in seen_paths:
                raise ValueError(
                    f"derived phase log path collides with another output: {log_path}"
                )
            seen_paths.add(log_path)
        normalized_phases.append(normalized_phase)
    normalized = {
        "schema_version": PLAN_SCHEMA,
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "workspace_root": _absolute_root(plan.get("workspace_root"), "workspace_root"),
        "artifact_root": _absolute_root(plan.get("artifact_root"), "artifact_root"),
        "lease_root": _absolute_root(plan.get("lease_root"), "lease_root"),
        "environment": environment,
        "device_ids": device_ids,
        "lease_timeout_seconds": lease_timeout,
        "phases": normalized_phases,
    }
    if normalized_pre_profile is not None:
        normalized["pre_profile_evidence"] = normalized_pre_profile
    fingerprint = content_fingerprint(normalized)
    if require_fingerprint and plan.get("plan_fingerprint") != fingerprint:
        raise ValueError("execution plan plan_fingerprint mismatch")
    normalized["plan_fingerprint"] = fingerprint
    return normalized


def freeze_plan(draft):
    if "plan_fingerprint" in draft:
        raise ValueError("draft execution plan must not contain plan_fingerprint")
    return validate_plan(draft, require_fingerprint=False)


def _phase_log_paths(phase):
    manifest = PurePosixPath(phase["manifest"])
    stem = manifest.name[:-5] if manifest.name.endswith(".json") else manifest.name
    parent = manifest.parent
    return {
        "command_stdout": (parent / f"{stem}.command.stdout.log").as_posix(),
        "command_stderr": (parent / f"{stem}.command.stderr.log").as_posix(),
        "validator_stdout": (parent / f"{stem}.validator.stdout.log").as_posix(),
        "validator_stderr": (parent / f"{stem}.validator.stderr.log").as_posix(),
    }


@contextlib.contextmanager
def device_leases(device_ids, lease_root, timeout_seconds):
    root = Path(lease_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    handles = []
    deadline = time.monotonic() + timeout_seconds
    try:
        for device_id in sorted(device_ids):
            path = root / f"npu-device-{device_id}.lock"
            handle = path.open("a+")
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        handle.close()
                        raise TimeoutError(
                            f"timed out acquiring NPU device lease {device_id}"
                        )
                    time.sleep(0.05)
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {"pid": os.getpid(), "device_id": device_id, "acquired_at": _now()}
                )
            )
            handle.flush()
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


@contextlib.contextmanager
def shared_device_leases(device_ids, lease_root, timeout_seconds, environment=None):
    """Reuse a matching parent lease or acquire the shared per-device locks."""
    devices = sorted(device_ids)
    if not devices:
        yield "not_required"
        return
    if parent_lease_held(devices, lease_root, environment=environment):
        yield "inherited"
        return
    with device_leases(devices, lease_root, timeout_seconds):
        yield "acquired"


def _run_argv(argv, cwd, environment, stdout_path, stderr_path, timeout_seconds):
    started_at = _now()
    timed_out = False
    with Path(stdout_path).open("xb") as stdout, Path(stderr_path).open("xb") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=min(5.0, timeout_seconds))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
    return {
        "argv": argv,
        "pid": process.pid,
        "started_at": started_at,
        "finished_at": _now(),
        "return_code": return_code,
        "timed_out": timed_out,
    }


def _sealed_file(path, relative):
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "fingerprint": file_fingerprint(path),
    }


def validate_phase_manifest(manifest, plan, phase, artifact_root, require_passed=True):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PHASE_SCHEMA:
        raise ValueError(f"phase manifest must use {PHASE_SCHEMA}")
    expected = {
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "phase_id": phase["phase_id"],
        "state_after": phase["state_after"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"phase manifest {field} mismatch")
    if require_passed and manifest.get("status") != "passed":
        raise ValueError("phase manifest is not passed")
    if manifest.get("status") == "passed":
        expected_keys = {
            "schema_version",
            "trial_id",
            "request_fingerprint",
            "plan_fingerprint",
            "phase_id",
            "state_after",
            "status",
            "reason",
            "cwd",
            "workspace_root",
            "artifact_root",
            "lease_root",
            "environment",
            "device_ids",
            "command",
            "validator",
            "sealed_files",
            "completed_at",
        }
        if set(manifest) != expected_keys:
            raise ValueError("passed phase manifest has unexpected or missing fields")
        if manifest.get("reason") is not None:
            raise ValueError("passed phase manifest reason must be null")
        if (
            not isinstance(manifest.get("completed_at"), str)
            or not manifest["completed_at"]
        ):
            raise ValueError("passed phase manifest completed_at must be non-empty")
    if manifest.get("cwd") != phase["cwd"] or manifest.get(
        "environment"
    ) != _phase_environment(plan, phase):
        raise ValueError("phase manifest execution context mismatch")
    for root_name in ("workspace_root", "artifact_root", "lease_root"):
        if manifest.get(root_name) != plan[root_name]:
            raise ValueError(f"phase manifest {root_name} mismatch")
    for key, argv in (
        ("command", phase["argv"]),
        ("validator", phase["validator_argv"]),
    ):
        record = manifest.get(key)
        if not isinstance(record, dict) or record.get("argv") != argv:
            raise ValueError(f"phase manifest {key} argv mismatch")
        if manifest.get("status") == "passed":
            expected_record = {
                "argv",
                "pid",
                "started_at",
                "finished_at",
                "return_code",
                "timed_out",
            }
            if set(record) != expected_record:
                raise ValueError(f"passed phase {key} record is incomplete")
            if (
                not isinstance(record.get("pid"), int)
                or isinstance(record.get("pid"), bool)
                or record["pid"] <= 0
            ):
                raise ValueError(f"passed phase {key} pid is invalid")
            for field in ("started_at", "finished_at"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"passed phase {key} {field} is invalid")
            if record.get("return_code") != 0 or record.get("timed_out") is not False:
                raise ValueError(f"passed phase has unsuccessful {key}")
    expected_devices = plan["device_ids"] if phase["requires_device"] else []
    if manifest.get("device_ids") != expected_devices:
        raise ValueError("phase manifest device_ids mismatch")
    files = manifest.get("sealed_files")
    if not isinstance(files, list):
        raise ValueError("phase manifest sealed_files must be a list")
    indexed = {item.get("path"): item for item in files if isinstance(item, dict)}
    required = phase["required_artifacts"] + list(_phase_log_paths(phase).values())
    if manifest.get("status") == "passed" and (
        len(files) != len(required) or set(indexed) != set(required)
    ):
        raise ValueError(
            "passed phase sealed_files must exactly match declared outputs"
        )
    for relative in required:
        path = _under(artifact_root, relative, "sealed file")
        if not path.is_file():
            raise ValueError(f"sealed file does not exist: {relative}")
        item = indexed.get(relative)
        if (
            item is None
            or item.get("fingerprint") != file_fingerprint(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"sealed file fingerprint mismatch: {relative}")
    return {"valid": True, "status": manifest.get("status")}


def _write_terminal_manifest(plan, phase, artifact_root, status, reason, records):
    logs = _phase_log_paths(phase)
    sealed = []
    for relative in logs.values():
        path = _under(artifact_root, relative, "phase log")
        if path.is_file():
            sealed.append(_sealed_file(path, relative))
    for relative in phase["required_artifacts"]:
        path = _under(artifact_root, relative, "required artifact")
        if path.is_file():
            sealed.append(_sealed_file(path, relative))
    manifest = {
        "schema_version": PHASE_SCHEMA,
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "phase_id": phase["phase_id"],
        "state_after": phase["state_after"],
        "status": status,
        "reason": reason,
        "cwd": phase["cwd"],
        "workspace_root": plan["workspace_root"],
        "artifact_root": plan["artifact_root"],
        "lease_root": plan["lease_root"],
        "environment": _phase_environment(plan, phase),
        "device_ids": plan["device_ids"] if phase["requires_device"] else [],
        "command": records.get(
            "command", {"argv": phase["argv"], "return_code": None, "timed_out": False}
        ),
        "validator": records.get(
            "validator",
            {"argv": phase["validator_argv"], "return_code": None, "timed_out": False},
        ),
        "sealed_files": sealed,
        "completed_at": _now(),
    }
    path = _under(artifact_root, phase["manifest"], "phase manifest")
    _atomic_write_json(path, manifest)
    return manifest


def _run_phase(plan, phase, workspace_root, artifact_root, lease_root):
    cwd = _under(workspace_root, phase["cwd"], "phase cwd")
    if not cwd.is_dir():
        raise ValueError(f"phase cwd does not exist: {phase['cwd']}")
    manifest_path = _under(artifact_root, phase["manifest"], "phase manifest")
    logs = _phase_log_paths(phase)
    output_paths = [
        _under(artifact_root, value, "phase log") for value in logs.values()
    ]
    artifact_paths = [
        _under(artifact_root, value, "required artifact")
        for value in phase["required_artifacts"]
    ]
    existing = [str(path) for path in output_paths + artifact_paths if path.exists()]
    if existing:
        raise ValueError(
            f"phase has pre-existing outputs without a manifest: {existing}"
        )
    for path in output_paths + artifact_paths + [manifest_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    environment = _phase_environment(plan, phase)
    records = {}
    lease_ids = plan["device_ids"] if phase["requires_device"] else []
    try:
        with shared_device_leases(lease_ids, lease_root, plan["lease_timeout_seconds"]):
            records["command"] = _run_argv(
                phase["argv"],
                cwd,
                environment,
                _under(artifact_root, logs["command_stdout"], "command stdout"),
                _under(artifact_root, logs["command_stderr"], "command stderr"),
                phase["timeout_seconds"],
            )
            if (
                records["command"]["timed_out"]
                or records["command"]["return_code"] != 0
            ):
                reason = (
                    "command_timeout"
                    if records["command"]["timed_out"]
                    else "command_failed"
                )
                return _write_terminal_manifest(
                    plan, phase, artifact_root, "failed", reason, records
                )
            records["validator"] = _run_argv(
                phase["validator_argv"],
                cwd,
                environment,
                _under(artifact_root, logs["validator_stdout"], "validator stdout"),
                _under(artifact_root, logs["validator_stderr"], "validator stderr"),
                phase["validator_timeout_seconds"],
            )
            if (
                records["validator"]["timed_out"]
                or records["validator"]["return_code"] != 0
            ):
                if records["validator"]["timed_out"]:
                    return _write_terminal_manifest(
                        plan,
                        phase,
                        artifact_root,
                        "failed",
                        "validator_timeout",
                        records,
                    )
                exit_code = str(records["validator"]["return_code"])
                action = phase["validator_exit_actions"].get(exit_code, "fail")
                status = {"reject": "rejected", "block": "blocked", "fail": "failed"}[
                    action
                ]
                return _write_terminal_manifest(
                    plan,
                    phase,
                    artifact_root,
                    status,
                    f"validator_{action}: exit_code={exit_code}",
                    records,
                )
    except TimeoutError as error:
        return _write_terminal_manifest(
            plan, phase, artifact_root, "blocked", str(error), records
        )
    except OSError as error:
        return _write_terminal_manifest(
            plan,
            phase,
            artifact_root,
            "failed",
            f"process_start_failed: {error}",
            records,
        )
    missing = [
        relative
        for relative, path in zip(phase["required_artifacts"], artifact_paths)
        if not path.is_file()
    ]
    if missing:
        return _write_terminal_manifest(
            plan,
            phase,
            artifact_root,
            "failed",
            f"required artifacts missing: {missing}",
            records,
        )
    sealed = [
        _sealed_file(path, relative)
        for relative, path in zip(phase["required_artifacts"], artifact_paths)
    ]
    for relative in logs.values():
        sealed.append(
            _sealed_file(_under(artifact_root, relative, "phase log"), relative)
        )
    manifest = {
        "schema_version": PHASE_SCHEMA,
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "phase_id": phase["phase_id"],
        "state_after": phase["state_after"],
        "status": "passed",
        "reason": None,
        "cwd": phase["cwd"],
        "workspace_root": plan["workspace_root"],
        "artifact_root": plan["artifact_root"],
        "lease_root": plan["lease_root"],
        "environment": environment,
        "device_ids": lease_ids,
        "command": records["command"],
        "validator": records["validator"],
        "sealed_files": sealed,
        "completed_at": _now(),
    }
    _atomic_write_json(manifest_path, manifest)
    validate_phase_manifest(manifest, plan, phase, artifact_root)
    return manifest


def run_plan(plan, state_path, workspace_root, artifact_root, lease_root):
    plan = validate_plan(plan)
    roots = {
        "workspace_root": str(Path(workspace_root).resolve()),
        "artifact_root": str(Path(artifact_root).resolve()),
        "lease_root": str(Path(lease_root).resolve()),
    }
    for name, value in roots.items():
        if plan[name] != value:
            raise ValueError(f"runtime {name} differs from the frozen execution plan")
    state_path = Path(state_path).resolve()
    try:
        state_path.relative_to(Path(artifact_root).resolve())
    except ValueError as error:
        raise ValueError("state path must be under the frozen artifact root") from error
    state = json.loads(state_path.read_text())
    multistream_execution.validate_state(
        state,
        trial_id=plan["trial_id"],
        request_fingerprint=plan["request_fingerprint"],
        artifact_root=artifact_root,
    )
    if state["state"] in multistream_execution.TERMINAL_STATES:
        raise ValueError(f"cannot run a terminal trial: {state['state']}")
    state_indices = {
        name: index for index, name in enumerate(multistream_execution.ORDERED_STATES)
    }
    if (
        state["state"] not in state_indices
        or state_indices[state["state"]] < state_indices["diff_verified"]
    ):
        raise ValueError("runner requires trial state diff_verified or later")
    for phase in plan["phases"]:
        phase_index = state_indices[phase["state_after"]]
        current_index = state_indices[state["state"]]
        manifest_path = _under(artifact_root, phase["manifest"], "phase manifest")
        if current_index >= phase_index:
            if not manifest_path.is_file():
                raise ValueError(
                    f"advanced state lacks phase manifest: {phase['manifest']}"
                )
            validate_phase_manifest(
                json.loads(manifest_path.read_text()), plan, phase, artifact_root
            )
            continue
        if phase_index != current_index + 1:
            raise ValueError(f"state cannot resume at phase {phase['phase_id']}")
        if phase["state_after"] == "profile_collected" and plan.get(
            "pre_profile_evidence"
        ):
            evidence_path = _under(
                artifact_root,
                plan["pre_profile_evidence"]["path"],
                "pre-profile evidence",
            )
            kind = plan["pre_profile_evidence"]["kind"]
            if not evidence_path.is_file():
                if kind in {"dispatch_order", "component_dispatch_order"}:
                    raise ValueError(
                        "reorder profile is blocked until dispatch_order_evidence exists"
                    )
                raise ValueError(
                    "profile is blocked until event_stage_dispatch evidence exists"
                )
            if kind == "dispatch_order":
                multistream_operator_order.validate_dispatch_evidence(
                    evidence_path,
                    artifact_root,
                    trial_id=plan["trial_id"],
                    request_fingerprint=plan["request_fingerprint"],
                )
            elif kind == "component_dispatch_order":
                multistream_component_reorder.validate_dispatch_evidence(
                    evidence_path,
                    artifact_root,
                    trial_id=plan["trial_id"],
                    request_fingerprint=plan["request_fingerprint"],
                )
            else:
                action = json.loads(Path(state["action_manifest"]).read_text())
                evidence = multistream_event_stage_dispatch.validate(
                    json.loads(evidence_path.read_text()),
                    action,
                    trial_id=plan["trial_id"],
                    request_fingerprint=plan["request_fingerprint"],
                )
                if evidence["decision"] != "effective":
                    raise ValueError(
                        "profile is blocked because event/stage dispatch action is not effective"
                    )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            validate_phase_manifest(manifest, plan, phase, artifact_root)
        else:
            manifest = _run_phase(
                plan, phase, workspace_root, artifact_root, lease_root
            )
        if manifest["status"] != "passed":
            terminal = {
                "blocked": "blocked",
                "rejected": "rejected",
                "failed": "failed",
            }[manifest["status"]]
            state = multistream_execution.advance_state(
                state, terminal, phase["manifest"]
            )
            _atomic_write_json(state_path, state)
            raise RuntimeError(
                f"phase {phase['phase_id']} {manifest['status']}: {manifest['reason']}"
            )
        state = multistream_execution.advance_state(
            state, phase["state_after"], phase["manifest"]
        )
        _atomic_write_json(state_path, state)
    return {
        "valid": True,
        "state": state["state"],
        "plan_fingerprint": plan["plan_fingerprint"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-plan")
    freeze.add_argument("--draft", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-plan")
    validate.add_argument("--plan", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--workspace-root", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--lease-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(
            (args.draft if args.command == "freeze-plan" else args.plan).read_text()
        )
        if args.command == "freeze-plan":
            result = freeze_plan(plan)
            if args.output.exists():
                raise ValueError(f"frozen plan output already exists: {args.output}")
            _atomic_write_json(args.output, result)
        elif args.command == "validate-plan":
            result = validate_plan(plan)
        else:
            result = run_plan(
                plan,
                args.state,
                args.workspace_root,
                args.artifact_root,
                args.lease_root,
            )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
