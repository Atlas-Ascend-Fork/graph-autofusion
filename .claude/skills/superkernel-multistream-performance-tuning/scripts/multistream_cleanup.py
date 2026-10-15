#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Safely quarantine mutable state from a terminal multistream trial."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath


PLAN_SCHEMA = "superkernel-multistream-cleanup-plan-v1"
RECEIPT_SCHEMA = "superkernel-multistream-cleanup-receipt-v1"
TERMINAL_STATES = {"accepted", "rejected", "blocked", "failed"}


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


def _absolute_directory(value, label, *, must_exist):
    value = _text(value, label)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = path.resolve()
    if must_exist and not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory")
    return resolved


def _disjoint(left, right, label):
    if left == right or left in right.parents or right in left.parents:
        raise ValueError(f"{label} must be disjoint")


def _safe_child(root, relative, label):
    path = root / _relative(relative, label)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} traverses a symlink")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    return path


def snapshot(path):
    """Return a stable content identity and reject symlink-dependent state."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"cannot snapshot symlink: {path}")
    if path.is_file():
        records = [
            {
                "path": ".",
                "kind": "file",
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ]
        kind = "file"
    elif path.is_dir():
        records = []
        for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
            relative = item.relative_to(path).as_posix()
            if item.is_symlink():
                raise ValueError(f"cannot snapshot symlink: {item}")
            if item.is_dir():
                records.append({"path": relative, "kind": "directory"})
            elif item.is_file():
                records.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "size": item.stat().st_size,
                        "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                    }
                )
            else:
                raise ValueError(f"unsupported filesystem entry: {item}")
        kind = "directory"
    else:
        raise ValueError(f"cannot snapshot missing or unsupported path: {path}")
    return {
        "kind": kind,
        "entry_count": len(records),
        "fingerprint": fingerprint(records),
    }


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def _write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_plan(plan, *, require_fingerprint=True):
    required = {
        "schema_version",
        "cleanup_id",
        "trial_id",
        "terminal_state",
        "isolation_root",
        "incumbent_root",
        "quarantine_root",
        "incumbent_snapshot",
        "targets",
        "preserved_paths",
    }
    if require_fingerprint:
        required.add("plan_fingerprint")
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError(f"cleanup plan must contain exactly {sorted(required)}")
    if plan["schema_version"] != PLAN_SCHEMA:
        raise ValueError(f"cleanup plan must use {PLAN_SCHEMA}")
    cleanup_id = _relative(plan["cleanup_id"], "cleanup_id")
    if "/" in cleanup_id:
        raise ValueError("cleanup_id must be one path segment")
    trial_id = _text(plan["trial_id"], "trial_id")
    if plan["terminal_state"] not in TERMINAL_STATES:
        raise ValueError(f"terminal_state must be one of {sorted(TERMINAL_STATES)}")
    isolation = _absolute_directory(
        plan["isolation_root"], "isolation_root", must_exist=True
    )
    incumbent = _absolute_directory(
        plan["incumbent_root"], "incumbent_root", must_exist=True
    )
    quarantine = _absolute_directory(
        plan["quarantine_root"], "quarantine_root", must_exist=False
    )
    _disjoint(isolation, incumbent, "isolation_root and incumbent_root")
    _disjoint(isolation, quarantine, "isolation_root and quarantine_root")
    _disjoint(incumbent, quarantine, "incumbent_root and quarantine_root")
    if plan["incumbent_snapshot"] != snapshot(incumbent):
        raise ValueError("incumbent snapshot differs from current immutable incumbent")

    if not isinstance(plan["targets"], list) or not plan["targets"]:
        raise ValueError("targets must be a non-empty list")
    targets = []
    target_paths = []
    for index, target in enumerate(plan["targets"]):
        label = f"targets[{index}]"
        if not isinstance(target, dict) or set(target) != {"path", "snapshot"}:
            raise ValueError(f"{label} must contain exactly path and snapshot")
        relative = _relative(target["path"], f"{label}.path")
        source = _safe_child(isolation, relative, f"{label}.path")
        destination = _safe_child(quarantine / cleanup_id, relative, f"{label}.path")
        if source.exists() and destination.exists():
            raise ValueError(f"{label} exists in both isolation and quarantine")
        if source.exists():
            actual = snapshot(source)
        elif destination.exists():
            actual = snapshot(destination)
        else:
            raise ValueError(f"{label} exists in neither isolation nor quarantine")
        if target["snapshot"] != actual:
            raise ValueError(f"{label} snapshot mismatch")
        target_paths.append(PurePosixPath(relative))
        targets.append({"path": relative, "snapshot": actual})
    for index, left in enumerate(target_paths):
        for right in target_paths[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError("cleanup targets must not overlap")

    if not isinstance(plan["preserved_paths"], list):
        raise ValueError("preserved_paths must be a list")
    preserved = []
    for index, value in enumerate(plan["preserved_paths"]):
        relative = _relative(value, f"preserved_paths[{index}]")
        preserved_path = PurePosixPath(relative)
        if any(
            preserved_path == item
            or preserved_path in item.parents
            or item in preserved_path.parents
            for item in target_paths
        ):
            raise ValueError("preserved_paths must not overlap cleanup targets")
        _safe_child(isolation, relative, f"preserved_paths[{index}]")
        preserved.append(relative)
    if len(preserved) != len(set(preserved)):
        raise ValueError("preserved_paths must be unique")

    normalized = {
        "schema_version": PLAN_SCHEMA,
        "cleanup_id": cleanup_id,
        "trial_id": trial_id,
        "terminal_state": plan["terminal_state"],
        "isolation_root": str(isolation),
        "incumbent_root": str(incumbent),
        "quarantine_root": str(quarantine),
        "incumbent_snapshot": plan["incumbent_snapshot"],
        "targets": targets,
        "preserved_paths": preserved,
    }
    expected = fingerprint(normalized)
    if require_fingerprint and plan["plan_fingerprint"] != expected:
        raise ValueError("cleanup plan fingerprint mismatch")
    normalized["plan_fingerprint"] = expected
    return normalized


def freeze_plan(draft):
    draft = dict(draft)
    incumbent = _absolute_directory(
        draft.get("incumbent_root"), "incumbent_root", must_exist=True
    )
    isolation = _absolute_directory(
        draft.get("isolation_root"), "isolation_root", must_exist=True
    )
    raw_targets = draft.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("targets must be a list")
    draft["incumbent_snapshot"] = snapshot(incumbent)
    draft["targets"] = [
        {
            "path": _relative(value, f"targets[{index}]"),
            "snapshot": snapshot(_safe_child(isolation, value, f"targets[{index}]")),
        }
        for index, value in enumerate(raw_targets)
    ]
    return validate_plan(draft, require_fingerprint=False)


def validate_receipt(receipt, plan):
    plan = validate_plan(plan)
    required = {
        "schema_version",
        "cleanup_id",
        "trial_id",
        "terminal_state",
        "plan_fingerprint",
        "quarantined",
        "preserved_paths",
        "incumbent_snapshot",
        "receipt_fingerprint",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError(f"cleanup receipt must contain exactly {sorted(required)}")
    if receipt["schema_version"] != RECEIPT_SCHEMA:
        raise ValueError(f"cleanup receipt must use {RECEIPT_SCHEMA}")
    for field in (
        "cleanup_id",
        "trial_id",
        "terminal_state",
        "plan_fingerprint",
        "preserved_paths",
        "incumbent_snapshot",
    ):
        if receipt[field] != plan[field]:
            raise ValueError(f"cleanup receipt {field} differs from plan")
    if receipt["quarantined"] != plan["targets"]:
        raise ValueError("cleanup receipt target identities differ from plan")
    isolation = Path(plan["isolation_root"])
    quarantine = Path(plan["quarantine_root"]) / plan["cleanup_id"]
    for index, target in enumerate(plan["targets"]):
        source = _safe_child(isolation, target["path"], f"targets[{index}].path")
        destination = _safe_child(quarantine, target["path"], f"targets[{index}].path")
        if (
            source.exists()
            or not destination.exists()
            or snapshot(destination) != target["snapshot"]
        ):
            raise ValueError(
                f"cleanup receipt target state is invalid: {target['path']}"
            )
    if snapshot(Path(plan["incumbent_root"])) != plan["incumbent_snapshot"]:
        raise ValueError("incumbent changed after cleanup")
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_fingerprint"
    }
    if receipt["receipt_fingerprint"] != fingerprint(unsigned):
        raise ValueError("cleanup receipt fingerprint mismatch")
    return {
        "valid": True,
        "terminal_state": plan["terminal_state"],
        "target_count": len(plan["targets"]),
    }


def run(plan, receipt_path):
    plan = validate_plan(plan)
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        receipt = _load(receipt_path)
        validate_receipt(receipt, plan)
        return receipt
    isolation = Path(plan["isolation_root"])
    quarantine = Path(plan["quarantine_root"]) / plan["cleanup_id"]
    for index, target in enumerate(plan["targets"]):
        source = _safe_child(isolation, target["path"], f"targets[{index}].path")
        destination = _safe_child(quarantine, target["path"], f"targets[{index}].path")
        if destination.exists():
            if source.exists() or snapshot(destination) != target["snapshot"]:
                raise ValueError(
                    f"cannot recover partially quarantined target: {target['path']}"
                )
            continue
        if not source.exists() or snapshot(source) != target["snapshot"]:
            raise ValueError(
                f"cleanup target changed before quarantine: {target['path']}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    if snapshot(Path(plan["incumbent_root"])) != plan["incumbent_snapshot"]:
        raise ValueError("incumbent changed during cleanup")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "cleanup_id": plan["cleanup_id"],
        "trial_id": plan["trial_id"],
        "terminal_state": plan["terminal_state"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "quarantined": plan["targets"],
        "preserved_paths": plan["preserved_paths"],
        "incumbent_snapshot": plan["incumbent_snapshot"],
    }
    receipt["receipt_fingerprint"] = fingerprint(receipt)
    _write_atomic(receipt_path, receipt)
    validate_receipt(receipt, plan)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-plan")
    freeze.add_argument("--draft", required=True)
    freeze.add_argument("--out", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--receipt", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.command == "freeze-plan":
        value = freeze_plan(_load(args.draft))
        _write_atomic(args.out, value)
    elif args.command == "run":
        value = run(_load(args.plan), args.receipt)
    else:
        value = validate_receipt(_load(args.receipt), _load(args.plan))
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
