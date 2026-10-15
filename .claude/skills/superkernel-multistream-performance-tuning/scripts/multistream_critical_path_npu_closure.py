#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Seal and replay a real-NPU event/stage critical-path tuning closure."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import multistream_cleanup
import multistream_critical_path_contract
import multistream_event_stage_dispatch
import multistream_evidence
import multistream_four_profile
import multistream_join_validation
import multistream_source_transform


SCHEMA = "superkernel-multistream-critical-path-npu-closure-v1"
ARTIFACT_SCHEMAS = {
    "request": multistream_critical_path_contract.REQUEST_SCHEMA,
    "action_manifest": multistream_source_transform.ACTION_MANIFEST_SCHEMA,
    "event_stage_dispatch": multistream_event_stage_dispatch.SCHEMA,
    "four_profile_plan": multistream_four_profile.PLAN_SCHEMA,
    "four_profile_summary": multistream_four_profile.SUMMARY_SCHEMA,
    "join_validation": multistream_join_validation.SCHEMA,
    "clean_evidence": multistream_evidence.EVIDENCE_SCHEMA,
    "result": multistream_critical_path_contract.RESULT_SCHEMA,
    "cleanup_plan": multistream_cleanup.PLAN_SCHEMA,
    "cleanup_receipt": multistream_cleanup.RECEIPT_SCHEMA,
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


def _rooted(root, value, label):
    value = _text(value, label)
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value in {"", "."}:
        raise ValueError(f"{label} must be a safe relative path")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist")
    return path


def _runtime(value):
    fields = {
        "device_name",
        "device_count",
        "backend",
        "npugraph_ex",
        "static_kernel_compile",
        "super_kernel_scope",
        "stream_count",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("critical path closure runtime fields are invalid")
    _text(value["device_name"], "runtime.device_name")
    if (
        isinstance(value["device_count"], bool)
        or not isinstance(value["device_count"], int)
        or value["device_count"] < 1
    ):
        raise ValueError("runtime.device_count must be positive")
    if value["backend"] != "npugraph_ex" or any(
        value[field] is not True
        for field in ("npugraph_ex", "static_kernel_compile", "super_kernel_scope")
    ):
        raise ValueError("runtime does not prove npugraph_ex SuperKernel execution")
    if (
        isinstance(value["stream_count"], bool)
        or not isinstance(value["stream_count"], int)
        or value["stream_count"] < 2
    ):
        raise ValueError("runtime.stream_count must be at least two")
    return dict(value)


def _artifacts(value, root):
    if not isinstance(value, dict) or set(value) != set(ARTIFACT_SCHEMAS):
        raise ValueError("critical path closure artifact catalog is incomplete")
    records, payloads = {}, {}
    for name, schema in ARTIFACT_SCHEMAS.items():
        item = value[name]
        relative = item["path"] if isinstance(item, dict) else item
        path = _rooted(root, relative, f"artifacts.{name}")
        payload = _load(path)
        if payload.get("schema_version") != schema:
            raise ValueError(f"artifacts.{name} must use {schema}")
        record = {
            "path": relative,
            "schema_version": schema,
            "file_fingerprint": file_fingerprint(path),
        }
        if isinstance(item, dict) and item != record:
            raise ValueError(f"artifacts.{name} sealed identity mismatch")
        records[name], payloads[name] = record, payload
    return records, payloads


def _replay(manifest, root, payloads):
    request = payloads["request"]
    request_summary = multistream_critical_path_contract.validate_request(request, root)
    request_fp = request_summary["request_fingerprint"]
    if request_fp != manifest["request_fingerprint"]:
        raise ValueError("closure request fingerprint mismatch")
    action = payloads["action_manifest"]
    if action.get("trial_id") != manifest["trial_id"] or action.get(
        "change_kind"
    ) not in {"event_edge_refinement", "stage_split"}:
        raise ValueError("closure action is not a matching event/stage action")
    if (
        action.get("multistream_only_verified") is not True
        or action.get("single_change_verified") is not True
    ):
        raise ValueError("closure action lacks source isolation gates")
    dispatch = multistream_event_stage_dispatch.validate(
        payloads["event_stage_dispatch"],
        action,
        trial_id=manifest["trial_id"],
        request_fingerprint=request_fp,
    )
    if dispatch["decision"] != "effective":
        raise ValueError("closure action dispatch did not change as authorized")
    plan = multistream_four_profile.validate_plan(payloads["four_profile_plan"])
    if (
        plan["trial_id"] != manifest["trial_id"]
        or plan["request_fingerprint"] != request_fp
    ):
        raise ValueError("closure four-profile identity mismatch")
    summary = multistream_four_profile.validate_summary(
        root / manifest["artifacts"]["four_profile_summary"]["path"], plan, root
    )
    join = multistream_join_validation.validate(payloads["join_validation"])
    if (
        join["trial_id"] != manifest["trial_id"]
        or join["request_fingerprint"] != request_fp
    ):
        raise ValueError("closure join validation identity mismatch")
    if join["action_manifest_fingerprint"] != multistream_source_transform.fingerprint(
        action
    ):
        raise ValueError("closure join validation does not bind action")
    if join["decision"] != "effective":
        raise ValueError("closure join mechanism is not effective")
    result_summary = multistream_critical_path_contract.validate_result(
        request, payloads["result"], root
    )
    trials = [
        item
        for item in payloads["result"].get("trials", [])
        if item.get("trial_id") == manifest["trial_id"]
    ]
    if len(trials) != 1 or trials[0].get("decision") not in {"accepted", "rejected"}:
        raise ValueError("closure requires one matching executed trial")
    clean = multistream_evidence.validate_evidence(
        root / manifest["artifacts"]["clean_evidence"]["path"],
        root,
        trial_id=manifest["trial_id"],
        request_fingerprint=request_fp,
        state_after=trials[0]["clean_state"],
    )
    expected_clean = "pass" if trials[0]["decision"] == "accepted" else "reject"
    if clean["decision"] != expected_clean:
        raise ValueError("closure clean decision differs from trial")
    multistream_cleanup.validate_receipt(
        payloads["cleanup_receipt"], payloads["cleanup_plan"]
    )
    expected_status = "accepted" if trials[0]["decision"] == "accepted" else "no_gain"
    if result_summary["status"] != expected_status:
        raise ValueError("closure result status differs from trial")
    return {
        "result_status": result_summary["status"],
        "trial_decision": trials[0]["decision"],
        "clean_decision": clean["decision"],
        "four_profile_complete": summary.get("complete", True),
    }


def validate(value, artifact_root, *, require_fingerprint=True):
    fields = {
        "schema_version",
        "closure_id",
        "request_fingerprint",
        "trial_id",
        "runtime",
        "artifacts",
        "action_closed",
        "four_profile_closed",
        "outcome",
    }
    if require_fingerprint:
        fields.add("closure_fingerprint")
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema_version") != SCHEMA
    ):
        raise ValueError(f"critical path closure must use {SCHEMA} with exact fields")
    root = Path(artifact_root).resolve()
    records, payloads = _artifacts(value["artifacts"], root)
    normalized = {
        "schema_version": SCHEMA,
        "closure_id": _text(value["closure_id"], "closure_id"),
        "request_fingerprint": _text(
            value["request_fingerprint"], "request_fingerprint"
        ),
        "trial_id": _text(value["trial_id"], "trial_id"),
        "runtime": _runtime(value["runtime"]),
        "artifacts": records,
        "action_closed": True,
        "four_profile_closed": True,
    }
    replay = _replay(normalized, root, payloads)
    normalized["outcome"] = replay
    normalized["closure_fingerprint"] = fingerprint(normalized)
    if value["action_closed"] is not True or value["four_profile_closed"] is not True:
        raise ValueError("critical path closure summary differs from semantic replay")
    if require_fingerprint and value["outcome"] != replay:
        raise ValueError("critical path closure outcome differs from semantic replay")
    if (
        require_fingerprint
        and value["closure_fingerprint"] != normalized["closure_fingerprint"]
    ):
        raise ValueError("critical path closure fingerprint mismatch")
    return {"valid": True, "closure": normalized, **replay}


def seal(draft, artifact_root):
    if not isinstance(draft, dict) or "closure_fingerprint" in draft:
        raise ValueError(
            "critical path closure draft must not contain closure_fingerprint"
        )
    prepared = dict(draft)
    prepared["action_closed"] = True
    prepared["four_profile_closed"] = True
    prepared.setdefault("outcome", {})
    return validate(prepared, artifact_root, require_fingerprint=False)["closure"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(_load(args.manifest), args.artifact_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {key: item for key, item in result.items() if key != "closure"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
