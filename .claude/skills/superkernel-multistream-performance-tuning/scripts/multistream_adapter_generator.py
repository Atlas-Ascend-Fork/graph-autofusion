#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate a reviewed five-phase model adapter from a compact model run spec."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import multistream_plan_compiler
import multistream_runner


SPEC_SCHEMA = "superkernel-multistream-model-run-spec-v1"
PHASE_KEYS = ("correctness", "profile", "analysis", "clean3", "clean5")
STATE_BY_PHASE = dict(zip(PHASE_KEYS, multistream_runner.PHASE_STATES))


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


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _templates(value, label, *, nonempty=True):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label} must be {'a non-empty ' if nonempty else ''}list")
    return [_text(item, f"{label}[]") for item in value]


def _phase(value, name):
    required = {
        "command_argv_template",
        "cwd_template",
        "timeout_seconds",
        "validator_timeout_seconds",
        "environment_overrides",
        "program_file_templates",
        "output_templates",
        "validator_inputs",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"phases.{name} must contain exactly {sorted(required)}")
    for field in ("timeout_seconds", "validator_timeout_seconds"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
            raise ValueError(f"phases.{name}.{field} must be positive")
    if not isinstance(value["environment_overrides"], dict):
        raise ValueError(f"phases.{name}.environment_overrides must be an object")
    if not isinstance(value["validator_inputs"], dict):
        raise ValueError(f"phases.{name}.validator_inputs must be an object")
    return {
        **value,
        "command_argv_template": _templates(
            value["command_argv_template"], f"phases.{name}.command_argv_template"
        ),
        "program_file_templates": _templates(
            value["program_file_templates"], f"phases.{name}.program_file_templates"
        ),
        "output_templates": _templates(
            value["output_templates"], f"phases.{name}.output_templates", nonempty=False
        ),
        "cwd_template": _text(value["cwd_template"], f"phases.{name}.cwd_template"),
    }


def _input(inputs, name, phase):
    if name not in inputs:
        raise ValueError(f"phases.{phase}.validator_inputs.{name} is required")
    return _text(inputs[name], f"phases.{phase}.validator_inputs.{name}")


def _validator_argv(phase, state, inputs, expected_ranks, warmup, evidence_out):
    common = [
        "{python_executable}",
        "{skill_root}/scripts/multistream_evidence.py",
    ]
    common_options = [
        "--artifact-root",
        "{artifact_root}",
        "--state-after",
        state,
        "--trial-id",
        "{trial_id}",
        "--request-fingerprint",
        "{request_fingerprint}",
        "--out",
        "{artifact_root}/" + evidence_out,
    ]
    if phase == "correctness":
        return (
            common
            + ["correctness"]
            + common_options
            + [
                "--run-root",
                _input(inputs, "run_root", phase),
                "--expected-ranks",
                str(expected_ranks),
            ]
        )
    if phase == "profile":
        argv = (
            common
            + ["profile"]
            + common_options
            + [
                "--baseline-manifest",
                _input(inputs, "baseline_manifest", phase),
                "--candidate-manifest",
                _input(inputs, "candidate_manifest", phase),
            ]
        )
        trace_analysis = inputs.get("trace_analysis")
        if trace_analysis is not None:
            argv.extend(
                [
                    "--trace-analysis",
                    _text(
                        trace_analysis, "phases.profile.validator_inputs.trace_analysis"
                    ),
                ]
            )
        return argv
    if phase == "analysis":
        return (
            common
            + ["analysis"]
            + common_options
            + [
                "--analysis-result",
                _input(inputs, "analysis_result", phase),
            ]
        )
    expected_runs = "3" if phase == "clean3" else "5"
    return (
        common
        + ["clean"]
        + common_options
        + [
            "--baseline-root",
            _input(inputs, "baseline_root", phase),
            "--candidate-root",
            _input(inputs, "candidate_root", phase),
            "--candidate-name",
            _input(inputs, "candidate_name", phase),
            "--expected-ranks",
            str(expected_ranks),
            "--warmup",
            str(warmup),
            "--expected-runs",
            expected_runs,
        ]
    )


def generate(spec):
    if not isinstance(spec, dict) or spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"model run spec must use {SPEC_SCHEMA}")
    required = {
        "schema_version",
        "adapter_id",
        "workspace_root",
        "artifact_root",
        "lease_root",
        "environment",
        "device_ids",
        "lease_timeout_seconds",
        "expected_ranks",
        "warmup",
        "phases",
    }
    if set(spec) != required:
        raise ValueError(f"model run spec must contain exactly {sorted(required)}")
    expected_ranks = spec["expected_ranks"]
    warmup = spec["warmup"]
    if (
        isinstance(expected_ranks, bool)
        or not isinstance(expected_ranks, int)
        or expected_ranks < 1
    ):
        raise ValueError("expected_ranks must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    phases = spec["phases"]
    if not isinstance(phases, dict) or set(phases) != set(PHASE_KEYS):
        raise ValueError(f"phases must contain exactly {list(PHASE_KEYS)}")
    adapter_phases = []
    for phase_name in PHASE_KEYS:
        phase = _phase(phases[phase_name], phase_name)
        state = STATE_BY_PHASE[phase_name]
        evidence = f"trials/{{trial_id}}/semantic/{state}.json"
        required_artifacts = list(phase["output_templates"])
        if evidence in required_artifacts:
            raise ValueError(
                f"phases.{phase_name}.output_templates duplicates semantic evidence"
            )
        required_artifacts.append(evidence)
        adapter_phases.append(
            {
                "state_after": state,
                "phase_id_template": "{trial_id}-" + phase_name,
                "argv_template": phase["command_argv_template"],
                "validator_argv_template": _validator_argv(
                    phase_name,
                    state,
                    phase["validator_inputs"],
                    expected_ranks,
                    warmup,
                    evidence,
                ),
                "cwd_template": phase["cwd_template"],
                "timeout_seconds": phase["timeout_seconds"],
                "validator_timeout_seconds": phase["validator_timeout_seconds"],
                "environment_overrides": phase["environment_overrides"],
                "validator_exit_actions": (
                    {"0": "pass", "10": "reject"}
                    if phase_name in {"clean3", "clean5"}
                    else {"0": "pass"}
                ),
                "program_file_templates": phase["program_file_templates"],
                "required_artifact_templates": required_artifacts,
                "manifest_template": f"trials/{{trial_id}}/phases/{phase_name}.json",
            }
        )
    draft = {
        "schema_version": multistream_plan_compiler.ADAPTER_SCHEMA,
        "adapter_id": spec["adapter_id"],
        "workspace_root": spec["workspace_root"],
        "artifact_root": spec["artifact_root"],
        "lease_root": spec["lease_root"],
        "environment": spec["environment"],
        "device_ids": spec["device_ids"],
        "lease_timeout_seconds": spec["lease_timeout_seconds"],
        "phases": adapter_phases,
    }
    return multistream_plan_compiler.freeze_adapter(draft)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"adapter output already exists: {args.out}")
        result = generate(json.loads(args.spec.read_text()))
        _atomic_json(args.out, result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
