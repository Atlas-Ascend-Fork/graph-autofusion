#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate one dispatched phase task and emit its executable tool plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SCRIPTS = REPOSITORY_ROOT / "superkernel-runtime-common" / "scripts"
sys.path.insert(0, str(RUNTIME_SCRIPTS))

import auto_tune_session  # noqa: E402 - load sibling scripts after sys.path setup
import device_lease_runner  # noqa: E402 - load sibling scripts after sys.path setup


def _load_manifest(path):
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "superkernel-phase-manifest-v1":
        raise ValueError("phase manifest schema_version is invalid")
    required = ("phase", "agent_id", "skill", "runtime_tools")
    if any(key not in manifest for key in required):
        raise ValueError("phase manifest is missing required fields")
    tools = manifest["runtime_tools"]
    if (
        not isinstance(tools, list)
        or not tools
        or any(not isinstance(tool, str) or Path(tool).name != tool for tool in tools)
    ):
        raise ValueError("phase manifest runtime_tools is invalid")
    missing = [tool for tool in tools if not (RUNTIME_SCRIPTS / tool).is_file()]
    if missing:
        raise ValueError(
            "phase manifest references missing runtime tools: " + ", ".join(missing)
        )
    return manifest


def run(*, manifest, argv=None):
    try:
        phase_manifest = _load_manifest(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"valid": False, "errors": [str(error)]}, ensure_ascii=False))
        return 1
    phase = phase_manifest["phase"]
    agent_id = phase_manifest["agent_id"]
    tools = phase_manifest["runtime_tools"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--tool", choices=tools)
    parser.add_argument("--lease-root", type=Path)
    parser.add_argument("--device-id", type=int, action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("tool_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        task = json.loads(args.task.read_text(encoding="utf-8"))
        current = auto_tune_session.validate_dispatch_task(
            task, task.get("session_path")
        )
        step = current["step"]
        if (
            step["phase"] != phase
            or step["agent_id"] != agent_id
            or step["skill"] != phase_manifest["skill"]
        ):
            raise ValueError("dispatch task is assigned to a different phase agent")
        if args.tool:
            if args.lease_root is None or not args.device_id:
                raise ValueError(
                    "--tool requires --lease-root and at least one --device-id"
                )
            tool_args = list(args.tool_args)
            if tool_args[:1] == ["--"]:
                tool_args.pop(0)
            manifest = args.manifest_out or (
                Path(current["artifact_root"])
                / "phase-executions"
                / step["step_id"]
                / f"{Path(args.tool).stem}.json"
            )
            result = device_lease_runner.run_command(
                [sys.executable, str(RUNTIME_SCRIPTS / args.tool), *tool_args],
                lease_root=args.lease_root,
                device_ids=args.device_id,
                timeout_seconds=args.timeout_seconds,
                cwd=Path(current["artifact_root"]),
                manifest_out=manifest,
                command_id=f"{step['step_id']}:{args.tool}",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result["status"] == "passed" else 1
        output = {
            "phase": phase,
            "agent_id": agent_id,
            "task_fingerprint": current["session_fingerprint"],
            "phase_manifest": str(Path(manifest).resolve()),
            "commands": [
                {"argv": ["python3", str(RUNTIME_SCRIPTS / tool), "--help"]}
                for tool in tools
            ],
            "handoff": {
                "schema_version": auto_tune_session.RESULT_SCHEMA,
                "session_path": current["session_path"],
                "step_id": step["step_id"],
                "seal_after_agent_result": True,
            },
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"valid": False, "errors": [str(error)]}, ensure_ascii=False))
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
