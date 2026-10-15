#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Seal and replay a complete real-NPU multistream tuning closure."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

import multistream_contract
import multistream_evidence
import multistream_execution
import multistream_four_profile
import multistream_operator_order


SCHEMA = "superkernel-multistream-npu-closure-v1"
OUTCOMES = {"accepted", "no_gain"}
ARTIFACT_SCHEMAS = {
    "request": multistream_contract.REQUEST_SCHEMA,
    "operator_order_capture": multistream_operator_order.CAPTURE_SCHEMA,
    "action_manifest": multistream_execution.ACTION_SCHEMA,
    "dispatch_order_evidence": multistream_operator_order.DISPATCH_EVIDENCE_SCHEMA,
    "four_profile_plan": multistream_four_profile.PLAN_SCHEMA,
    "four_profile_summary": multistream_four_profile.SUMMARY_SCHEMA,
    "clean_evidence": multistream_evidence.EVIDENCE_SCHEMA,
    "result": multistream_contract.RESULT_SCHEMA,
}


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


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
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
        raise ValueError(f"{label} escapes artifact_root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


def _validate_runtime(value):
    required = {
        "device_name",
        "device_count",
        "backend",
        "npugraph_ex",
        "static_kernel_compile",
        "super_kernel_scope",
        "stream_count",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"runtime must contain exactly {sorted(required)}")
    _text(value["device_name"], "runtime.device_name")
    if (
        isinstance(value["device_count"], bool)
        or not isinstance(value["device_count"], int)
        or value["device_count"] < 1
    ):
        raise ValueError("runtime.device_count must be a positive integer")
    if value["backend"] != "npugraph_ex":
        raise ValueError("runtime.backend must be npugraph_ex")
    for field in ("npugraph_ex", "static_kernel_compile", "super_kernel_scope"):
        if value[field] is not True:
            raise ValueError(f"runtime.{field} must be true")
    if (
        isinstance(value["stream_count"], bool)
        or not isinstance(value["stream_count"], int)
        or value["stream_count"] < 2
    ):
        raise ValueError("runtime.stream_count must be at least two")
    return dict(value)


def _artifact_records(paths, root):
    if not isinstance(paths, dict) or set(paths) != set(ARTIFACT_SCHEMAS):
        raise ValueError(f"artifacts must contain exactly {sorted(ARTIFACT_SCHEMAS)}")
    records = {}
    values = {}
    for name, expected_schema in ARTIFACT_SCHEMAS.items():
        item = paths[name]
        relative = item["path"] if isinstance(item, dict) else item
        path = _rooted(root, relative, f"artifacts.{name}")
        value = _load(path)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != expected_schema
        ):
            raise ValueError(f"artifacts.{name} must use {expected_schema}")
        actual = file_fingerprint(path)
        if isinstance(item, dict):
            expected = {"path", "schema_version", "file_fingerprint"}
            if set(item) != expected:
                raise ValueError(f"artifacts.{name} record fields are invalid")
            if (
                item["schema_version"] != expected_schema
                or item["file_fingerprint"] != actual
            ):
                raise ValueError(f"artifacts.{name} sealed identity mismatch")
        records[name] = {
            "path": relative,
            "schema_version": expected_schema,
            "file_fingerprint": actual,
        }
        values[name] = value
    return records, values


def _semantic_replay(manifest, root, values):
    request = values["request"]
    request_summary = multistream_contract.validate_request(request, root)
    request_fingerprint = request_summary["request_fingerprint"]
    if manifest["request_fingerprint"] != request_fingerprint:
        raise ValueError("closure request_fingerprint differs from request content")

    capture_path = root / manifest["artifacts"]["operator_order_capture"]["path"]
    analysis = multistream_operator_order.analyze(
        capture_path, root, expected_request_fingerprint=request_fingerprint
    )
    authorized = [
        item
        for item in analysis["targets"]
        if item.get("multistream_reorder_authorized") is True
    ]
    if not authorized:
        raise ValueError(
            "real NPU closure requires an analyzer-authorized reorder candidate"
        )

    action = values["action_manifest"]
    if action.get("trial_id") != manifest["trial_id"]:
        raise ValueError("action trial_id differs from closure")
    if action.get("change_kind") != multistream_execution.REORDER_CHANGE_KIND:
        raise ValueError(
            "real NPU closure action must be dependency-safe operator reorder"
        )
    if action.get("multistream_only_verified") is not True:
        raise ValueError("real NPU closure action lacks multistream-only verification")

    dispatch_path = root / manifest["artifacts"]["dispatch_order_evidence"]["path"]
    multistream_operator_order.validate_dispatch_evidence(
        dispatch_path,
        root,
        trial_id=manifest["trial_id"],
        request_fingerprint=request_fingerprint,
    )

    plan = multistream_four_profile.validate_plan(values["four_profile_plan"])
    if (
        plan["trial_id"] != manifest["trial_id"]
        or plan["request_fingerprint"] != request_fingerprint
    ):
        raise ValueError("four-profile plan identity differs from closure")
    summary_path = root / manifest["artifacts"]["four_profile_summary"]["path"]
    multistream_four_profile.validate_summary(summary_path, plan, root)

    clean_path = root / manifest["artifacts"]["clean_evidence"]["path"]
    clean = multistream_evidence.validate_evidence(
        clean_path,
        root,
        trial_id=manifest["trial_id"],
        request_fingerprint=request_fingerprint,
        state_after="clean3_passed",
    )
    result = values["result"]
    result_summary = multistream_contract.validate_result(request, result, root)
    if result_summary["status"] != manifest["outcome"]:
        raise ValueError("closure outcome differs from validated result")
    trials = [
        item
        for item in result.get("trials", [])
        if item.get("trial_id") == manifest["trial_id"]
    ]
    if len(trials) != 1:
        raise ValueError(
            "real NPU closure must contain exactly one matching executed trial"
        )
    expected_decision = "accepted" if manifest["outcome"] == "accepted" else "rejected"
    if trials[0].get("decision") != expected_decision:
        raise ValueError("trial decision differs from closure outcome")
    expected_clean = "pass" if manifest["outcome"] == "accepted" else "reject"
    if clean.get("decision") != expected_clean:
        raise ValueError("clean evidence decision differs from closure outcome")
    return {
        "authorized_range_ids": [item["range_id"] for item in authorized],
        "result_status": result_summary["status"],
        "trial_decision": trials[0]["decision"],
        "clean_decision": clean["decision"],
    }


def validate(manifest, artifact_root, *, require_fingerprint=True):
    root = Path(artifact_root).resolve()
    required = {
        "schema_version",
        "closure_id",
        "request_fingerprint",
        "trial_id",
        "runtime",
        "artifacts",
        "outcome",
    }
    if require_fingerprint:
        required.add("closure_fingerprint")
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError(f"closure manifest must contain exactly {sorted(required)}")
    if manifest["schema_version"] != SCHEMA:
        raise ValueError(f"closure manifest must use {SCHEMA}")
    _text(manifest["closure_id"], "closure_id")
    _text(manifest["request_fingerprint"], "request_fingerprint")
    _text(manifest["trial_id"], "trial_id")
    if manifest["outcome"] not in OUTCOMES:
        raise ValueError("closure outcome must be accepted or no_gain")
    runtime = _validate_runtime(manifest["runtime"])
    records, values = _artifact_records(manifest["artifacts"], root)
    normalized = {
        "schema_version": SCHEMA,
        "closure_id": manifest["closure_id"],
        "request_fingerprint": manifest["request_fingerprint"],
        "trial_id": manifest["trial_id"],
        "runtime": runtime,
        "artifacts": records,
        "outcome": manifest["outcome"],
    }
    replay = _semantic_replay(normalized, root, values)
    normalized["closure_fingerprint"] = fingerprint(normalized)
    if (
        require_fingerprint
        and manifest["closure_fingerprint"] != normalized["closure_fingerprint"]
    ):
        raise ValueError("closure_fingerprint mismatch")
    return {"valid": True, **replay, "closure": normalized}


def seal(draft, artifact_root):
    if not isinstance(draft, dict) or "closure_fingerprint" in draft:
        raise ValueError("closure draft must not contain closure_fingerprint")
    return validate(draft, artifact_root, require_fingerprint=False)["closure"]


def _write(path, value):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--draft", type=Path, required=True)
    seal_parser.add_argument("--artifact-root", type=Path, required=True)
    seal_parser.add_argument("--out", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "seal":
        result = seal(_load(args.draft), args.artifact_root)
        _write(args.out, result)
        print(
            json.dumps(
                {"valid": True, "closure_fingerprint": result["closure_fingerprint"]},
                sort_keys=True,
            )
        )
    else:
        result = validate(_load(args.manifest), args.artifact_root)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "closure"},
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
