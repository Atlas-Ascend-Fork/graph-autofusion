#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate multiple network adapters against one unchanged generic core."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath


SKILL_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SCHEMA = "superkernel-multistream-conformance-adapter-v1"
PLAN_SCHEMA = "superkernel-multistream-network-conformance-plan-v1"
REPORT_SCHEMA = "superkernel-multistream-network-conformance-report-v1"
EVIDENCE_KINDS = {
    "real_npu_closure": "superkernel-multistream-real-npu-receipt-v1",
    "resource_screening": "superkernel-multistream-resource-screening-receipt-v1",
}
DECISIONS = {"accepted", "no_gain", "no_reorder_candidate", "blocked"}


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


def _rooted(root, value, label):
    root = Path(root).resolve()
    path = (root / _relative(value, label)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {value}")
    return path


def _atomic(path, value):
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


def _evidence_decision(kind, evidence):
    expected_schema = EVIDENCE_KINDS[kind]
    if evidence.get("schema_version") != expected_schema:
        raise ValueError(f"{kind} evidence must use {expected_schema}")
    if kind == "real_npu_closure":
        outcome = evidence.get("outcome")
        status = outcome.get("result_status") if isinstance(outcome, dict) else None
        if status not in {"accepted", "no_gain"} or outcome.get(
            "incumbent_unchanged"
        ) != (status != "accepted"):
            raise ValueError("real NPU closure receipt outcome is inconsistent")
        return status
    if (
        evidence.get("decision") != "no_reorder_candidate"
        or evidence.get("authorization") != "diagnostic_only"
        or evidence.get("stable_parent_match_count") != 0
    ):
        raise ValueError(
            "resource screening receipt must prove zero stable parent matches"
        )
    policy = evidence.get("resource_classification_policy")
    if policy != "accelerator_core_block_num_mix_block_num_v1":
        raise ValueError("resource screening receipt uses an unsupported classifier")
    return "no_reorder_candidate"


def validate_adapter(value, root, *, require_fingerprint=True):
    required = {
        "schema_version",
        "adapter_id",
        "network_class",
        "capabilities",
        "evidence",
        "expected_decision",
    }
    if require_fingerprint:
        required.add("adapter_fingerprint")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"conformance adapter must contain exactly {sorted(required)}")
    if value["schema_version"] != ADAPTER_SCHEMA:
        raise ValueError(f"conformance adapter must use {ADAPTER_SCHEMA}")
    adapter_id = _text(value["adapter_id"], "adapter_id")
    network_class = _text(value["network_class"], "network_class")
    capabilities = value["capabilities"]
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(not isinstance(item, str) or not item for item in capabilities)
        or capabilities != sorted(set(capabilities))
    ):
        raise ValueError("adapter capabilities must be a sorted unique non-empty list")
    if set(capabilities) - set(EVIDENCE_KINDS):
        raise ValueError("adapter declares an unknown conformance capability")
    evidence_records = value["evidence"]
    if not isinstance(evidence_records, list) or not evidence_records:
        raise ValueError("adapter evidence must be a non-empty list")
    normalized = []
    decisions = set()
    for index, item in enumerate(evidence_records):
        label = f"evidence[{index}]"
        required_item = {"kind", "path", "schema_version", "file_fingerprint"}
        if not isinstance(item, dict) or set(item) != required_item:
            raise ValueError(f"{label} fields are invalid")
        kind = item["kind"]
        if kind not in EVIDENCE_KINDS or kind not in capabilities:
            raise ValueError(f"{label}.kind is not declared by the adapter")
        path = _rooted(root, item["path"], f"{label}.path")
        evidence = _load(path)
        expected_schema = EVIDENCE_KINDS[kind]
        actual_fingerprint = file_fingerprint(path)
        if (
            item["schema_version"] != expected_schema
            or item["file_fingerprint"] != actual_fingerprint
        ):
            raise ValueError(f"{label} sealed identity mismatch")
        decisions.add(_evidence_decision(kind, evidence))
        normalized.append(
            {
                "kind": kind,
                "path": item["path"],
                "schema_version": expected_schema,
                "file_fingerprint": actual_fingerprint,
            }
        )
    expected_decision = value["expected_decision"]
    if expected_decision not in DECISIONS or decisions != {expected_decision}:
        raise ValueError("adapter expected_decision differs from evidence")
    adapter = {
        "schema_version": ADAPTER_SCHEMA,
        "adapter_id": adapter_id,
        "network_class": network_class,
        "capabilities": capabilities,
        "evidence": normalized,
        "expected_decision": expected_decision,
    }
    adapter["adapter_fingerprint"] = fingerprint(adapter)
    if (
        require_fingerprint
        and value["adapter_fingerprint"] != adapter["adapter_fingerprint"]
    ):
        raise ValueError("adapter_fingerprint mismatch")
    return adapter


def freeze_adapter(draft, root):
    if "adapter_fingerprint" in draft:
        raise ValueError("adapter draft must not contain adapter_fingerprint")
    prepared = dict(draft)
    records = []
    for item in draft.get("evidence", []):
        if not isinstance(item, dict) or set(item) != {"kind", "path"}:
            raise ValueError(
                "adapter draft evidence entries must contain kind and path"
            )
        path = _rooted(root, item["path"], "adapter draft evidence.path")
        kind = item["kind"]
        if kind not in EVIDENCE_KINDS:
            raise ValueError("adapter draft evidence kind is unknown")
        records.append(
            {
                **item,
                "schema_version": EVIDENCE_KINDS[kind],
                "file_fingerprint": file_fingerprint(path),
            }
        )
    prepared["evidence"] = records
    return validate_adapter(prepared, root, require_fingerprint=False)


def _core_records(paths):
    if not isinstance(paths, list) or not paths:
        raise ValueError("core_files must be a non-empty list")
    records = []
    seen = set()
    for index, item in enumerate(paths):
        relative = item["path"] if isinstance(item, dict) else item
        path = _rooted(SKILL_ROOT, relative, f"core_files[{index}]")
        if relative in seen:
            raise ValueError("core_files contains a duplicate path")
        seen.add(relative)
        actual = file_fingerprint(path)
        if isinstance(item, dict) and (
            set(item) != {"path", "file_fingerprint"}
            or item["file_fingerprint"] != actual
        ):
            raise ValueError(f"core_files[{index}] sealed identity mismatch")
        records.append({"path": relative, "file_fingerprint": actual})
    return sorted(records, key=lambda item: item["path"])


def validate_plan(value, root, *, require_fingerprint=True):
    required = {"schema_version", "suite_id", "core_files", "adapters"}
    if require_fingerprint:
        required.add("plan_fingerprint")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"conformance plan must contain exactly {sorted(required)}")
    if value["schema_version"] != PLAN_SCHEMA:
        raise ValueError(f"conformance plan must use {PLAN_SCHEMA}")
    suite_id = _text(value["suite_id"], "suite_id")
    core_files = _core_records(value["core_files"])
    adapters = value["adapters"]
    if not isinstance(adapters, list) or len(adapters) < 2:
        raise ValueError("conformance plan requires at least two network adapters")
    adapter_records = []
    values = []
    for index, item in enumerate(adapters):
        relative = item["path"] if isinstance(item, dict) else item
        path = _rooted(root, relative, f"adapters[{index}]")
        adapter = validate_adapter(_load(path), root)
        actual = file_fingerprint(path)
        if isinstance(item, dict) and (
            set(item) != {"path", "file_fingerprint"}
            or item["file_fingerprint"] != actual
        ):
            raise ValueError(f"adapters[{index}] sealed identity mismatch")
        adapter_records.append({"path": relative, "file_fingerprint": actual})
        values.append(adapter)
    if len({item["adapter_id"] for item in values}) != len(values):
        raise ValueError("conformance plan has duplicate adapter_id")
    if len({item["network_class"] for item in values}) < 2:
        raise ValueError("conformance plan requires at least two network classes")
    if not any("real_npu_closure" in item["capabilities"] for item in values):
        raise ValueError("conformance plan requires one real-NPU closure adapter")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "suite_id": suite_id,
        "core_files": core_files,
        "adapters": adapter_records,
    }
    plan["plan_fingerprint"] = fingerprint(plan)
    if require_fingerprint and value["plan_fingerprint"] != plan["plan_fingerprint"]:
        raise ValueError("plan_fingerprint mismatch")
    return plan, values


def freeze_plan(draft, root):
    if "plan_fingerprint" in draft:
        raise ValueError("plan draft must not contain plan_fingerprint")
    prepared = dict(draft)
    prepared["core_files"] = _core_records(draft.get("core_files"))
    records = []
    for item in draft.get("adapters", []):
        relative = item["path"] if isinstance(item, dict) else item
        path = _rooted(root, relative, "adapter path")
        records.append({"path": relative, "file_fingerprint": file_fingerprint(path)})
    prepared["adapters"] = records
    return validate_plan(prepared, root, require_fingerprint=False)[0]


def build_report(plan_path, root):
    plan, adapters = validate_plan(_load(plan_path), root)
    adapter_tokens = [item["adapter_id"].lower() for item in adapters]
    leaks = []
    for record in plan["core_files"]:
        text = (SKILL_ROOT / record["path"]).read_text(errors="replace").lower()
        for token in adapter_tokens:
            if token in text:
                leaks.append({"core_file": record["path"], "adapter_id": token})
    if leaks:
        raise ValueError(f"generic core contains adapter identities: {leaks}")
    report = {
        "schema_version": REPORT_SCHEMA,
        "suite_id": plan["suite_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "generic_core_fingerprint": fingerprint({"files": plan["core_files"]}),
        "adapter_count": len(adapters),
        "network_class_count": len({item["network_class"] for item in adapters}),
        "core_adapter_identity_leaks": [],
        "adapters": [
            {
                "adapter_id": item["adapter_id"],
                "network_class": item["network_class"],
                "capabilities": item["capabilities"],
                "decision": item["expected_decision"],
                "adapter_fingerprint": item["adapter_fingerprint"],
            }
            for item in adapters
        ],
        "decision": "pass",
    }
    report["report_fingerprint"] = fingerprint(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-adapter", "freeze-plan"):
        command = commands.add_parser(name)
        command.add_argument("--draft", type=Path, required=True)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-adapter":
            value = freeze_adapter(_load(args.draft), args.root)
        elif args.command == "freeze-plan":
            value = freeze_plan(_load(args.draft), args.root)
        else:
            value = build_report(args.plan, args.root)
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        _atomic(args.out, value)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
