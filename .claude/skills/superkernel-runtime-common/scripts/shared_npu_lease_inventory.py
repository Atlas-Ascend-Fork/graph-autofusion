#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Audit every production process-launch path against the shared NPU lease protocol."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


SCHEMA = "superkernel-shared-npu-lease-inventory-v1"
SOURCE_ROOTS = (
    "superkernel-runtime-common/scripts",
    "superkernel-auto-tune/scripts",
    "superkernel-multistream-performance-tuning/scripts",
)
LAUNCH_CALLS = {
    "subprocess.Popen",
    "subprocess.run",
    "_run_argv",
    "multistream_runner._run_argv",
    "device_lease_runner.run_command",
}
KNOWN_CALLS = {
    (
        "superkernel-multistream-performance-tuning/scripts/multistream_runner.py",
        "_run_argv",
        "subprocess.Popen",
    ): (False, "lease_primitive"),
    (
        "superkernel-multistream-performance-tuning/scripts/multistream_runner.py",
        "_run_phase",
        "_run_argv",
    ): (True, "shared_device_leases"),
    (
        "superkernel-runtime-common/scripts/device_lease_runner.py",
        "run_command",
        "multistream_runner._run_argv",
    ): (True, "shared_device_leases+lease_environment"),
    (
        "superkernel-multistream-performance-tuning/scripts/multistream_four_profile.py",
        "_run_profile",
        "multistream_runner._run_argv",
    ): (True, "required_parent_marker"),
    (
        "superkernel-runtime-common/scripts/derived_family_lifecycle.py",
        "run_fresh_base",
        "device_lease_runner.run_command",
    ): (True, "delegated_parent_wrapper"),
    (
        "superkernel-auto-tune/scripts/phase_entrypoint.py",
        "run",
        "device_lease_runner.run_command",
    ): (True, "delegated_parent_wrapper"),
    (
        "superkernel-auto-tune/scripts/auto_tune_session.py",
        "run_agent_step",
        "subprocess.run",
    ): (False, "agent_host_dispatch"),
    (
        "superkernel-runtime-common/scripts/recommend_sk_strategy.py",
        "_run_read_only_reanalysis",
        "subprocess.run",
    ): (False, "offline_read_only_analysis"),
}
ENTRYPOINTS = (
    {
        "entrypoint": "parent_device_lease_runner",
        "path": "superkernel-runtime-common/scripts/device_lease_runner.py",
        "function": "run_command",
        "npu_capable": True,
        "lease_policy": "acquire_or_inherit_then_inject_child_markers",
        "required_tokens": ["shared_device_leases", "lease_environment"],
    },
    {
        "entrypoint": "multistream_execution_runner",
        "path": "superkernel-multistream-performance-tuning/scripts/multistream_runner.py",
        "function": "_run_phase",
        "npu_capable": True,
        "lease_policy": "device_phase_acquire_or_inherit",
        "required_tokens": ["shared_device_leases", "_phase_environment"],
    },
    {
        "entrypoint": "four_profile_orchestrator",
        "path": "superkernel-multistream-performance-tuning/scripts/multistream_four_profile.py",
        "function": "run_plan",
        "npu_capable": True,
        "lease_policy": "matching_parent_marker_required",
        "required_tokens": ["_lease_context", "parent_lease_held"],
    },
    {
        "entrypoint": "derived_family_fresh_base",
        "path": "superkernel-runtime-common/scripts/derived_family_lifecycle.py",
        "function": "run_fresh_base",
        "npu_capable": True,
        "lease_policy": "delegate_to_parent_device_lease_runner",
        "required_tokens": ["device_lease_runner.run_command"],
    },
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def file_fingerprint(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, relative):
        self.relative = relative
        self.functions = []
        self.calls = []

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        name = _call_name(node.func)
        if name in LAUNCH_CALLS:
            self.calls.append(
                {
                    "path": self.relative,
                    "function": self.functions[-1] if self.functions else "<module>",
                    "call": name,
                    "line": node.lineno,
                }
            )
        self.generic_visit(node)


def _discover_calls(repository_root):
    root = Path(repository_root).resolve()
    calls = []
    sources = {}
    for relative_root in SOURCE_ROOTS:
        for path in sorted((root / relative_root).glob("*.py")):
            relative = path.relative_to(root).as_posix()
            source = path.read_text()
            tree = ast.parse(source, filename=relative)
            visitor = _CallVisitor(relative)
            visitor.visit(tree)
            calls.extend(visitor.calls)
            sources[relative] = {
                "file_fingerprint": file_fingerprint(path),
                "source": source,
                "functions": {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                },
            }
    return calls, sources


def build_inventory(repository_root):
    root = Path(repository_root).resolve()
    calls, sources = _discover_calls(root)
    blockers = []
    classified = []
    observed = set()
    for call in calls:
        key = (call["path"], call["function"], call["call"])
        rule = KNOWN_CALLS.get(key)
        if rule is None:
            blockers.append(
                f"unregistered process launch: {call['path']}:{call['function']}:{call['call']}"
            )
            classified.append(
                {**call, "classification": "unregistered", "npu_capable": None}
            )
            continue
        observed.add(key)
        npu_capable, guard = rule
        classified.append({**call, "classification": guard, "npu_capable": npu_capable})
    blockers.extend(
        "registered launch no longer observed: " + ":".join(key)
        for key in sorted(set(KNOWN_CALLS) - observed)
    )

    entries = []
    for declaration in ENTRYPOINTS:
        source = sources.get(declaration["path"])
        problems = []
        if source is None:
            problems.append("source_missing")
        else:
            if declaration["function"] not in source["functions"]:
                problems.append("function_missing")
            problems.extend(
                f"required_token_missing:{token}"
                for token in declaration["required_tokens"]
                if token not in source["source"]
            )
        if problems:
            blockers.append(f"{declaration['entrypoint']}:" + ",".join(problems))
        entries.append(
            {
                **{
                    key: value
                    for key, value in declaration.items()
                    if key != "required_tokens"
                },
                "source_fingerprint": source["file_fingerprint"] if source else None,
                "status": "covered" if not problems else "blocked",
            }
        )
    report = {
        "schema_version": SCHEMA,
        "status": "passed" if not blockers else "blocked",
        "repository_root": str(root),
        "entrypoints": entries,
        "process_launches": classified,
        "blockers": sorted(blockers),
    }
    report["inventory_fingerprint"] = fingerprint(report)
    return report


def validate_inventory(report, repository_root):
    actual = build_inventory(repository_root)
    if report != actual:
        raise ValueError(
            "shared NPU lease inventory differs from deterministic repository scan"
        )
    if report["status"] != "passed":
        raise ValueError(
            "shared NPU lease inventory is blocked: " + "; ".join(report["blockers"])
        )
    return {
        "valid": True,
        "entrypoint_count": len(report["entrypoints"]),
        "inventory_fingerprint": report["inventory_fingerprint"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    try:
        if args.validate:
            result = validate_inventory(
                json.loads(args.validate.read_text()), args.repository_root
            )
        else:
            result = build_inventory(args.repository_root)
            if args.out:
                if args.out.exists():
                    raise ValueError(f"inventory output already exists: {args.out}")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(
                    json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n"
                )
    except (OSError, ValueError, SyntaxError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
