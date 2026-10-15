#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile a frozen model adapter into one trial-bound execution plan."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import multistream_execution
import multistream_component_reorder
import multistream_runner
import multistream_source_transform


ADAPTER_SCHEMA = "superkernel-multistream-model-adapter-v1"
COMPILE_SCHEMA = "superkernel-multistream-plan-compilation-v1"
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SKILL_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PROGRAM = (SKILL_ROOT / "scripts" / "multistream_evidence.py").resolve()
VALIDATOR_COMMANDS = {
    "correctness_passed": "correctness",
    "profile_collected": "profile",
    "analysis_validated": "analysis",
    "clean3_passed": "clean",
    "clean5_passed": "clean",
}
VALIDATOR_REQUIRED_FLAGS = {
    "correctness": {
        "--artifact-root",
        "--state-after",
        "--trial-id",
        "--request-fingerprint",
        "--out",
        "--run-root",
        "--expected-ranks",
    },
    "profile": {
        "--artifact-root",
        "--state-after",
        "--trial-id",
        "--request-fingerprint",
        "--out",
        "--baseline-manifest",
        "--candidate-manifest",
    },
    "analysis": {
        "--artifact-root",
        "--state-after",
        "--trial-id",
        "--request-fingerprint",
        "--out",
        "--analysis-result",
    },
    "clean": {
        "--artifact-root",
        "--state-after",
        "--trial-id",
        "--request-fingerprint",
        "--out",
        "--baseline-root",
        "--candidate-root",
        "--candidate-name",
        "--expected-ranks",
        "--warmup",
        "--expected-runs",
    },
}


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


def _adapter_payload(adapter):
    return {
        key: value for key, value in adapter.items() if key != "adapter_fingerprint"
    }


def _template_text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty template string")
    return value


def _template_argv(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty argv template array")
    return [_template_text(item, f"{label}[]") for item in value]


def validate_adapter(adapter, require_fingerprint=True):
    if not isinstance(adapter, dict) or adapter.get("schema_version") != ADAPTER_SCHEMA:
        raise ValueError(f"model adapter must use {ADAPTER_SCHEMA}")
    expected_keys = {
        "schema_version",
        "adapter_id",
        "workspace_root",
        "artifact_root",
        "lease_root",
        "environment",
        "device_ids",
        "lease_timeout_seconds",
        "phases",
    }
    if require_fingerprint:
        expected_keys.add("adapter_fingerprint")
    if set(adapter) != expected_keys:
        raise ValueError(f"model adapter must contain exactly {sorted(expected_keys)}")
    adapter_id = adapter.get("adapter_id")
    if not isinstance(adapter_id, str) or not IDENTIFIER.fullmatch(adapter_id):
        raise ValueError("model adapter adapter_id is unsafe")
    roots = {
        name: multistream_runner._absolute_root(adapter.get(name), name)
        for name in ("workspace_root", "artifact_root", "lease_root")
    }
    environment = multistream_runner._validate_environment(adapter.get("environment"))
    device_ids = adapter.get("device_ids")
    if (
        not isinstance(device_ids, list)
        or device_ids != sorted(device_ids)
        or len(device_ids) != len(set(device_ids))
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in device_ids
        )
        or not device_ids
    ):
        raise ValueError(
            "model adapter device_ids must be sorted unique non-negative integers"
        )
    lease_timeout = adapter.get("lease_timeout_seconds")
    if (
        not isinstance(lease_timeout, (int, float))
        or isinstance(lease_timeout, bool)
        or lease_timeout <= 0
    ):
        raise ValueError("model adapter lease_timeout_seconds must be positive")
    phases = adapter.get("phases")
    if not isinstance(phases, list) or [
        item.get("state_after") for item in phases if isinstance(item, dict)
    ] != list(multistream_runner.PHASE_STATES):
        raise ValueError("model adapter phases must follow the runner phase sequence")
    normalized_phases = []
    for index, phase in enumerate(phases):
        required = {
            "state_after",
            "phase_id_template",
            "argv_template",
            "validator_argv_template",
            "cwd_template",
            "timeout_seconds",
            "validator_timeout_seconds",
            "environment_overrides",
            "validator_exit_actions",
            "program_file_templates",
            "required_artifact_templates",
            "manifest_template",
        }
        if not isinstance(phase, dict) or set(phase) != required:
            raise ValueError(
                f"adapter phases[{index}] must contain exactly {sorted(required)}"
            )
        timeout = phase["timeout_seconds"]
        validator_timeout = phase["validator_timeout_seconds"]
        for name, value in (
            ("timeout_seconds", timeout),
            ("validator_timeout_seconds", validator_timeout),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"adapter phases[{index}].{name} must be positive")
        overrides = phase["environment_overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(
                f"adapter phases[{index}].environment_overrides must be an object"
            )
        normalized_overrides = {
            name: _template_text(
                value, f"adapter phases[{index}].environment_overrides.{name}"
            )
            for name, value in overrides.items()
        }
        multistream_runner._validate_environment(
            {name: "template" for name in normalized_overrides}
        )
        exit_actions = phase["validator_exit_actions"]
        if not isinstance(exit_actions, dict):
            raise ValueError(
                f"adapter phases[{index}].validator_exit_actions must be an object"
            )
        if (
            phase["state_after"] in {"clean3_passed", "clean5_passed"}
            and exit_actions.get("10") != "reject"
        ):
            raise ValueError(
                f"adapter phases[{index}].validator_exit_actions must map standard "
                "clean no-gain exit code 10 to reject"
            )
        artifacts = phase["required_artifact_templates"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(
                f"adapter phases[{index}].required_artifact_templates must be non-empty"
            )
        program_templates = phase["program_file_templates"]
        if not isinstance(program_templates, list) or not program_templates:
            raise ValueError(
                f"adapter phases[{index}].program_file_templates must be non-empty"
            )
        normalized_phases.append(
            {
                **phase,
                "phase_id_template": _template_text(
                    phase["phase_id_template"], "phase_id_template"
                ),
                "argv_template": _template_argv(
                    phase["argv_template"], "argv_template"
                ),
                "validator_argv_template": _template_argv(
                    phase["validator_argv_template"], "validator_argv_template"
                ),
                "cwd_template": _template_text(phase["cwd_template"], "cwd_template"),
                "environment_overrides": dict(sorted(normalized_overrides.items())),
                "validator_exit_actions": dict(exit_actions),
                "program_file_templates": [
                    _template_text(item, "program_file_templates[]")
                    for item in program_templates
                ],
                "required_artifact_templates": [
                    _template_text(item, "required_artifact_templates[]")
                    for item in artifacts
                ],
                "manifest_template": _template_text(
                    phase["manifest_template"], "manifest_template"
                ),
            }
        )
    normalized = {
        "schema_version": ADAPTER_SCHEMA,
        "adapter_id": adapter_id,
        **roots,
        "environment": environment,
        "device_ids": device_ids,
        "lease_timeout_seconds": lease_timeout,
        "phases": normalized_phases,
    }
    fingerprint = content_fingerprint(normalized)
    if require_fingerprint and adapter.get("adapter_fingerprint") != fingerprint:
        raise ValueError("model adapter adapter_fingerprint mismatch")
    normalized["adapter_fingerprint"] = fingerprint
    return normalized


def freeze_adapter(draft):
    if "adapter_fingerprint" in draft:
        raise ValueError("draft model adapter must not contain adapter_fingerprint")
    return validate_adapter(draft, require_fingerprint=False)


def _render(value, variables, label):
    names = PLACEHOLDER.findall(value)
    unknown = sorted(set(names) - set(variables))
    if unknown:
        raise ValueError(f"{label} uses unknown placeholders: {', '.join(unknown)}")
    rendered = PLACEHOLDER.sub(lambda match: variables[match.group(1)], value)
    if "{" in rendered or "}" in rendered:
        raise ValueError(f"{label} contains unsupported brace syntax")
    if not rendered:
        raise ValueError(f"{label} rendered empty")
    return rendered


def _load_variables(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("compile variables must be a JSON object")
    normalized = {}
    for name, item in value.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError(f"invalid compile variable name: {name!r}")
        if not isinstance(item, str) or not item:
            raise ValueError(f"compile variable {name} must be a non-empty string")
        normalized[name] = item
    return normalized


def _validator_options(argv, state_after):
    command = VALIDATOR_COMMANDS[state_after]
    if len(argv) < 3 or argv[:3] != [sys.executable, str(EVIDENCE_PROGRAM), command]:
        raise ValueError(
            f"{state_after} validator must invoke the standard semantic evidence "
            f"program with subcommand {command}"
        )
    tail = argv[3:]
    if len(tail) % 2:
        raise ValueError(f"{state_after} validator options must use --name value pairs")
    options = {}
    for index in range(0, len(tail), 2):
        name, value = tail[index : index + 2]
        if not name.startswith("--") or name in options or not value:
            raise ValueError(f"{state_after} validator options are invalid")
        options[name] = value
    missing = sorted(VALIDATOR_REQUIRED_FLAGS[command] - set(options))
    if missing:
        raise ValueError(
            f"{state_after} validator is missing required options: {', '.join(missing)}"
        )
    expected_common = {
        "--artifact-root": None,
        "--state-after": state_after,
    }
    for name, expected in expected_common.items():
        if expected is not None and options.get(name) != expected:
            raise ValueError(f"{state_after} validator {name} mismatch")
    if state_after in {"clean3_passed", "clean5_passed"}:
        expected_runs = "3" if state_after == "clean3_passed" else "5"
        if options.get("--expected-runs") != expected_runs:
            raise ValueError(
                f"{state_after} validator --expected-runs must be {expected_runs}"
            )
    return options


def compile_plan(
    request,
    adapter,
    action_manifest,
    *,
    artifact_root,
    variables=None,
    candidate_config=None,
):
    import multistream_contract
    import multistream_critical_path_contract

    artifact_root = Path(artifact_root).resolve()
    if (
        request.get("schema_version")
        == multistream_critical_path_contract.REQUEST_SCHEMA
    ):
        request_summary = multistream_critical_path_contract.validate_request(
            request, artifact_root
        )
    else:
        request_summary = multistream_contract.validate_request(request, artifact_root)
    adapter = validate_adapter(adapter)
    if adapter["artifact_root"] != str(artifact_root):
        raise ValueError(
            "model adapter artifact_root differs from request artifact root"
        )
    action = action_manifest
    action_schema = action.get("schema_version") if isinstance(action, dict) else None
    if action_schema not in {
        multistream_execution.ACTION_SCHEMA,
        multistream_source_transform.ACTION_MANIFEST_SCHEMA,
    }:
        raise ValueError("action manifest schema is unsupported")
    if action.get("single_change_verified") is not True:
        raise ValueError("action manifest must prove one isolated change")
    trial_id = action.get("trial_id")
    if not isinstance(trial_id, str) or not IDENTIFIER.fullmatch(trial_id):
        raise ValueError("action manifest trial_id is unsafe")
    if candidate_config is None:
        candidate_config = action.get("materialized_config") or action.get(
            "materialized_source"
        )
    if not isinstance(candidate_config, str) or not candidate_config:
        raise ValueError("source action plan compilation requires candidate_config")
    candidate_path = Path(candidate_config)
    if not candidate_path.is_absolute():
        candidate_path = (artifact_root / candidate_path).resolve()
    try:
        candidate_path.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("candidate_config escapes artifact root") from error
    if not candidate_path.is_file():
        raise ValueError("candidate_config does not exist")
    context = {
        "trial_id": trial_id,
        "request_id": request["request_id"],
        "parent_experiment_id": request["parent_experiment_id"],
        "request_fingerprint": request_summary["request_fingerprint"],
        "candidate_config": str(candidate_path),
        "workspace_root": adapter["workspace_root"],
        "artifact_root": adapter["artifact_root"],
        "lease_root": adapter["lease_root"],
        "python_executable": sys.executable,
        "skill_root": str(SKILL_ROOT),
    }
    extras = _load_variables(variables)
    overlap = sorted(set(context) & set(extras))
    if overlap:
        raise ValueError(
            f"compile variables cannot override built-ins: {', '.join(overlap)}"
        )
    context.update(extras)
    is_reorder = action.get("change_kind") == multistream_execution.REORDER_CHANGE_KIND
    is_component_reorder = (
        action.get("change_kind") == multistream_component_reorder.CHANGE_KIND
    )
    is_event_stage = action.get("change_kind") in {
        "event_edge_refinement",
        "stage_split",
    }
    dispatch_evidence = extras.get("dispatch_order_evidence")
    component_dispatch_evidence = extras.get("component_dispatch_evidence")
    event_stage_evidence = extras.get("event_stage_dispatch_evidence")
    if is_reorder and not isinstance(dispatch_evidence, str):
        raise ValueError(
            "operator reorder plan requires compile variable dispatch_order_evidence"
        )
    if not is_reorder and dispatch_evidence is not None:
        raise ValueError(
            "dispatch_order_evidence compile variable is allowed only for operator reorder"
        )
    if is_component_reorder and not isinstance(component_dispatch_evidence, str):
        raise ValueError(
            "component reorder plan requires compile variable component_dispatch_evidence"
        )
    if not is_component_reorder and component_dispatch_evidence is not None:
        raise ValueError(
            "component_dispatch_evidence compile variable is allowed only for component reorder"
        )
    if is_event_stage and not isinstance(event_stage_evidence, str):
        raise ValueError(
            "event/stage plan requires compile variable event_stage_dispatch_evidence"
        )
    if not is_event_stage and event_stage_evidence is not None:
        raise ValueError(
            "event_stage_dispatch_evidence is allowed only for event/stage actions"
        )
    phases = []
    for index, template in enumerate(adapter["phases"]):

        def render(value, field):
            return _render(value, context, f"phases[{index}].{field}")

        validator_argv = [
            render(value, "validator_argv_template")
            for value in template["validator_argv_template"]
        ]
        validator_options = _validator_options(validator_argv, template["state_after"])
        if validator_options["--artifact-root"] != adapter["artifact_root"]:
            raise ValueError(
                f"{template['state_after']} validator --artifact-root differs from adapter"
            )
        if validator_options["--trial-id"] != trial_id:
            raise ValueError(f"{template['state_after']} validator --trial-id mismatch")
        if (
            validator_options["--request-fingerprint"]
            != request_summary["request_fingerprint"]
        ):
            raise ValueError(
                f"{template['state_after']} validator --request-fingerprint mismatch"
            )
        required_artifacts = [
            render(value, "required_artifact_templates")
            for value in template["required_artifact_templates"]
        ]
        output_path = Path(validator_options["--out"])
        output_path = (
            output_path.resolve()
            if output_path.is_absolute()
            else (Path(adapter["artifact_root"]) / output_path).resolve()
        )
        try:
            output_relative = str(
                output_path.relative_to(Path(adapter["artifact_root"]))
            )
        except ValueError as error:
            raise ValueError(
                f"{template['state_after']} validator --out escapes artifact root"
            ) from error
        if output_relative not in required_artifacts:
            raise ValueError(
                f"{template['state_after']} validator --out must be a required artifact"
            )
        program_paths = [
            render(value, "program_file_templates")
            for value in template["program_file_templates"]
        ]
        for standard_program in (sys.executable, str(EVIDENCE_PROGRAM)):
            if standard_program not in program_paths:
                program_paths.append(standard_program)
        phases.append(
            {
                "phase_id": render(template["phase_id_template"], "phase_id_template"),
                "state_after": template["state_after"],
                "argv": [
                    render(value, "argv_template")
                    for value in template["argv_template"]
                ],
                "validator_argv": validator_argv,
                "cwd": render(template["cwd_template"], "cwd_template"),
                "timeout_seconds": template["timeout_seconds"],
                "validator_timeout_seconds": template["validator_timeout_seconds"],
                "requires_device": template["state_after"]
                in multistream_runner.DEVICE_PHASE_STATES,
                "environment_overrides": {
                    name: render(value, f"environment_overrides.{name}")
                    for name, value in template["environment_overrides"].items()
                },
                "validator_exit_actions": template["validator_exit_actions"],
                "program_files": [
                    {
                        "path": value,
                        "file_fingerprint": multistream_runner.file_fingerprint(value),
                    }
                    for value in program_paths
                ],
                "required_artifacts": required_artifacts,
                "manifest": render(template["manifest_template"], "manifest_template"),
            }
        )
    plan_draft = {
        "schema_version": multistream_runner.PLAN_SCHEMA,
        "trial_id": trial_id,
        "request_fingerprint": request_summary["request_fingerprint"],
        "workspace_root": adapter["workspace_root"],
        "artifact_root": adapter["artifact_root"],
        "lease_root": adapter["lease_root"],
        "environment": adapter["environment"],
        "device_ids": adapter["device_ids"],
        "lease_timeout_seconds": adapter["lease_timeout_seconds"],
        "phases": phases,
    }
    if is_reorder:
        plan_draft["pre_profile_evidence"] = {
            "kind": "dispatch_order",
            "path": dispatch_evidence,
        }
    elif is_component_reorder:
        plan_draft["pre_profile_evidence"] = {
            "kind": "component_dispatch_order",
            "path": component_dispatch_evidence,
        }
    elif is_event_stage:
        plan_draft["pre_profile_evidence"] = {
            "kind": "event_stage_dispatch",
            "path": event_stage_evidence,
        }
    plan = multistream_runner.freeze_plan(plan_draft)
    compilation = {
        "schema_version": COMPILE_SCHEMA,
        "trial_id": trial_id,
        "request_fingerprint": request_summary["request_fingerprint"],
        "action_manifest_fingerprint": content_fingerprint(action),
        "adapter_id": adapter["adapter_id"],
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "candidate_config": str(candidate_path),
        "variables": dict(sorted(extras.items())),
    }
    compilation["compilation_fingerprint"] = content_fingerprint(compilation)
    return plan, compilation


def validate_compilation(
    compilation, plan, adapter, action_manifest, artifact_root, request=None
):
    if (
        not isinstance(compilation, dict)
        or compilation.get("schema_version") != COMPILE_SCHEMA
    ):
        raise ValueError(f"plan compilation must use {COMPILE_SCHEMA}")
    expected_keys = {
        "schema_version",
        "trial_id",
        "request_fingerprint",
        "action_manifest_fingerprint",
        "adapter_id",
        "adapter_fingerprint",
        "plan_fingerprint",
        "candidate_config",
        "variables",
        "compilation_fingerprint",
    }
    if set(compilation) != expected_keys:
        raise ValueError(
            f"plan compilation must contain exactly {sorted(expected_keys)}"
        )
    fingerprint = compilation["compilation_fingerprint"]
    unsigned = {
        key: value
        for key, value in compilation.items()
        if key != "compilation_fingerprint"
    }
    if fingerprint != content_fingerprint(unsigned):
        raise ValueError("plan compilation compilation_fingerprint mismatch")
    adapter = validate_adapter(adapter)
    plan = multistream_runner.validate_plan(plan)
    expected = {
        "trial_id": plan["trial_id"],
        "request_fingerprint": plan["request_fingerprint"],
        "action_manifest_fingerprint": content_fingerprint(action_manifest),
        "adapter_id": adapter["adapter_id"],
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
    }
    for field, value in expected.items():
        if compilation.get(field) != value:
            raise ValueError(f"plan compilation {field} mismatch")
    _load_variables(compilation.get("variables"))
    candidate = Path(compilation.get("candidate_config", ""))
    root = Path(artifact_root).resolve()
    if not candidate.is_absolute():
        raise ValueError("plan compilation candidate_config must be absolute")
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "plan compilation candidate_config escapes artifact root"
        ) from error
    if not candidate.is_file():
        raise ValueError("plan compilation candidate_config does not exist")
    if request is not None:
        regenerated_plan, regenerated_compilation = compile_plan(
            request,
            adapter,
            action_manifest,
            artifact_root=root,
            variables=compilation["variables"],
            candidate_config=str(candidate),
        )
        if _canonical_json(regenerated_plan) != _canonical_json(plan):
            raise ValueError(
                "execution plan differs from deterministic adapter compilation"
            )
        if _canonical_json(regenerated_compilation) != _canonical_json(compilation):
            raise ValueError("plan compilation differs from deterministic replay")
    return {
        "valid": True,
        "adapter_id": adapter["adapter_id"],
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "compilation_fingerprint": fingerprint,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-adapter")
    freeze.add_argument("--draft", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--request", type=Path, required=True)
    compile_parser.add_argument("--adapter", type=Path, required=True)
    compile_parser.add_argument("--action-manifest", type=Path, required=True)
    compile_parser.add_argument("--variables", type=Path)
    compile_parser.add_argument("--candidate-config")
    compile_parser.add_argument("--plan-out", type=Path, required=True)
    compile_parser.add_argument("--compilation-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-adapter":
            result = freeze_adapter(json.loads(args.draft.read_text()))
            if args.output.exists():
                raise ValueError(f"frozen adapter output already exists: {args.output}")
            _atomic_write_json(args.output, result)
        else:
            request = json.loads(args.request.read_text())
            adapter = json.loads(args.adapter.read_text())
            action = json.loads(args.action_manifest.read_text())
            variables = (
                json.loads(args.variables.read_text()) if args.variables else None
            )
            plan, result = compile_plan(
                request,
                adapter,
                action,
                artifact_root=args.request.parent,
                variables=variables,
                candidate_config=args.candidate_config,
            )
            if args.plan_out.exists() or args.compilation_out.exists():
                raise ValueError("plan or compilation output already exists")
            _atomic_write_json(args.plan_out, plan)
            _atomic_write_json(args.compilation_out, result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
