#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run one parent-flow command under the shared SuperKernel NPU lease protocol."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


MULTISTREAM_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "superkernel-multistream-performance-tuning"
    / "scripts"
)
sys.path.insert(0, str(MULTISTREAM_SCRIPTS))
import multistream_runner  # noqa: E402


MANIFEST_SCHEMA = "superkernel-shared-device-lease-command-v2"


def _fingerprint(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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


def run_command(
    argv,
    *,
    lease_root,
    device_ids,
    timeout_seconds,
    cwd,
    manifest_out,
    environment=None,
    command_id="parent-command",
    lease_timeout_seconds=None,
):
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ValueError("command argv must be a non-empty string array")
    executable = Path(argv[0])
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("command executable must be an existing absolute file")
    devices = sorted(device_ids)
    if (
        not devices
        or len(devices) != len(set(devices))
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in devices
        )
    ):
        raise ValueError("device_ids must be sorted unique non-negative integers")
    cwd = Path(cwd).resolve()
    if not cwd.is_dir():
        raise ValueError("command cwd must be an existing directory")
    lease_root = Path(lease_root).resolve()
    if lease_timeout_seconds is None:
        lease_timeout_seconds = timeout_seconds
    if (
        isinstance(lease_timeout_seconds, bool)
        or not isinstance(lease_timeout_seconds, (int, float))
        or lease_timeout_seconds <= 0
    ):
        raise ValueError("lease_timeout_seconds must be positive")
    manifest_out = Path(manifest_out).resolve()
    if manifest_out.exists():
        raise ValueError(f"lease command manifest already exists: {manifest_out}")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = manifest_out.with_suffix(".stdout.log")
    stderr_path = manifest_out.with_suffix(".stderr.log")
    if stdout_path.exists() or stderr_path.exists():
        raise ValueError("lease command log output already exists")
    if environment is None:
        environment = dict(os.environ)
        environment_mode = "inherited_parent_environment"
    else:
        environment = multistream_runner._validate_environment(environment)
        environment_mode = "explicit_environment"
    try:
        with multistream_runner.shared_device_leases(
            devices, lease_root, lease_timeout_seconds
        ) as lease_mode:
            child_environment = multistream_runner.lease_environment(
                environment, devices, lease_root
            )
            record = multistream_runner._run_argv(
                argv, cwd, child_environment, stdout_path, stderr_path, timeout_seconds
            )
        status = (
            "passed"
            if record["return_code"] == 0 and not record["timed_out"]
            else "failed"
        )
        reason = (
            None
            if status == "passed"
            else "command_timeout"
            if record["timed_out"]
            else "command_nonzero"
        )
    except TimeoutError as error:
        record = None
        status = "blocked"
        reason = str(error)
    except ValueError as error:
        record = None
        status = "blocked"
        reason = f"parent lease context invalid: {error}"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "command_id": command_id,
        "status": status,
        "reason": reason,
        "lease_root": str(lease_root),
        "device_ids": devices,
        "lease_timeout_seconds": lease_timeout_seconds,
        "cwd": str(cwd),
        "environment_mode": environment_mode,
        "lease_mode": lease_mode if record is not None else None,
        "executable": {
            "path": str(executable.resolve()),
            "file_fingerprint": multistream_runner.file_fingerprint(
                executable.resolve()
            ),
        },
        "command": record,
        "logs": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    }
    manifest["manifest_fingerprint"] = _fingerprint(manifest)
    _atomic_json(manifest_out, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--device-id", type=int, action="append", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--lease-timeout-seconds", type=float)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--command-id", default="parent-command")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    try:
        environment = (
            json.loads(args.environment_json.read_text())
            if args.environment_json
            else None
        )
        result = run_command(
            command,
            lease_root=args.lease_root,
            device_ids=args.device_id,
            timeout_seconds=args.timeout_seconds,
            cwd=args.cwd,
            manifest_out=args.manifest_out,
            environment=environment,
            command_id=args.command_id,
            lease_timeout_seconds=args.lease_timeout_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return (
        0 if result["status"] == "passed" else 2 if result["status"] == "blocked" else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
