#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run fresh BASE and gate a derived family through ordinary ledger merge."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath

import bootstrap_derived_family
import device_lease_runner
import experiment_ledger


PLAN_SCHEMA = "superkernel-derived-fresh-base-plan-v1"
RECEIPT_SCHEMA = "superkernel-derived-fresh-base-receipt-v1"
REQUIRED_GATES = {"correctness", "profiling", "analysis", "clean_performance"}
REQUIRED_ARTIFACTS = {
    "config_snapshot",
    "baseline_profile_manifest",
    "candidate_profile_manifest",
    "profiling_analysis_result",
    "clean_performance_summary",
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


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


def _load(path):
    return json.loads(Path(path).read_text())


def _safe_relative(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value == ".":
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _under(root, relative, label, *, must_exist=False):
    root = Path(root).resolve()
    candidate = (root / _safe_relative(relative, label)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes family root") from error
    if must_exist and not candidate.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return candidate


def _identity(value, label):
    return bootstrap_derived_family._identity(value, label)


def freeze_plan(draft):
    required = {
        "schema_version",
        "family_id",
        "round_id",
        "family_root",
        "lease_root",
        "device_ids",
        "lease_timeout_seconds",
        "environment",
        "cwd",
        "command_argv",
        "timeout_seconds",
        "program_files",
        "receipt",
        "command_manifest",
    }
    if not isinstance(draft, dict) or draft.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"fresh BASE plan must use {PLAN_SCHEMA}")
    if set(draft) != required:
        raise ValueError(f"fresh BASE plan must contain exactly {sorted(required)}")
    for field in ("family_id", "round_id"):
        if not isinstance(draft[field], str) or not draft[field].strip():
            raise ValueError(f"fresh BASE plan {field} must be non-empty")
    roots = {}
    for field in ("family_root", "lease_root", "cwd"):
        path = Path(draft[field])
        if not path.is_absolute():
            raise ValueError(f"fresh BASE plan {field} must be absolute")
        path = path.resolve()
        if field in {"family_root", "cwd"} and not path.is_dir():
            raise ValueError(f"fresh BASE plan {field} must be an existing directory")
        roots[field] = str(path)
    try:
        Path(roots["cwd"]).relative_to(Path(roots["family_root"]))
    except ValueError as error:
        raise ValueError("fresh BASE cwd must be inside family_root") from error
    devices = draft["device_ids"]
    if (
        not isinstance(devices, list)
        or devices != sorted(devices)
        or len(devices) != len(set(devices))
        or not devices
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in devices
        )
    ):
        raise ValueError("fresh BASE plan device_ids are invalid")
    for field in ("lease_timeout_seconds", "timeout_seconds"):
        value = draft[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"fresh BASE plan {field} must be positive")
    environment = draft["environment"]
    multistream_scripts = (
        Path(__file__).resolve().parents[2]
        / "superkernel-multistream-performance-tuning"
        / "scripts"
    )
    sys.path.insert(0, str(multistream_scripts))
    import multistream_runner

    environment = multistream_runner._validate_environment(environment)
    argv = draft["command_argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ValueError("fresh BASE command_argv must be a non-empty array")
    executable = Path(argv[0])
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError(
            "fresh BASE command executable must be an existing absolute file"
        )
    programs = draft["program_files"]
    if not isinstance(programs, list) or not programs:
        raise ValueError("fresh BASE program_files must be non-empty")
    normalized_programs = []
    seen = set()
    for record in programs:
        if not isinstance(record, dict) or set(record) != {"path", "file_fingerprint"}:
            raise ValueError("fresh BASE program file record is invalid")
        path = Path(record["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "fresh BASE program file must be an existing absolute file"
            )
        resolved = str(path.resolve())
        if resolved in seen:
            raise ValueError("fresh BASE program_files contains duplicates")
        seen.add(resolved)
        actual = file_fingerprint(resolved)
        if record["file_fingerprint"] != actual:
            raise ValueError(f"fresh BASE program file changed: {resolved}")
        normalized_programs.append({"path": resolved, "file_fingerprint": actual})
    if str(executable.resolve()) not in seen:
        raise ValueError("fresh BASE command executable is missing from program_files")
    receipt = _safe_relative(draft["receipt"], "fresh BASE receipt")
    command_manifest = _safe_relative(
        draft["command_manifest"], "fresh BASE command_manifest"
    )
    if receipt == command_manifest:
        raise ValueError("fresh BASE receipt and command_manifest must differ")
    normalized = {
        "schema_version": PLAN_SCHEMA,
        "family_id": draft["family_id"],
        "round_id": draft["round_id"],
        **roots,
        "device_ids": devices,
        "lease_timeout_seconds": draft["lease_timeout_seconds"],
        "environment": environment,
        "command_argv": argv,
        "timeout_seconds": draft["timeout_seconds"],
        "program_files": sorted(normalized_programs, key=lambda item: item["path"]),
        "receipt": receipt,
        "command_manifest": command_manifest,
    }
    normalized["plan_fingerprint"] = fingerprint(normalized)
    return normalized


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"fresh BASE plan must use {PLAN_SCHEMA}")
    claimed = plan.get("plan_fingerprint")
    draft = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    normalized = freeze_plan(draft)
    if claimed != normalized["plan_fingerprint"]:
        raise ValueError("fresh BASE plan fingerprint mismatch")
    return normalized


def validate_receipt(path, family_root, lineage):
    family_root = Path(family_root).resolve()
    path = Path(path).resolve()
    try:
        path.relative_to(family_root)
    except ValueError as error:
        raise ValueError("fresh BASE receipt escapes family root") from error
    receipt = _load(path)
    required = {
        "schema_version",
        "derived_experiment_id",
        "round_id",
        "seed_identity",
        "gates",
        "clean_run_count",
        "stable",
        "artifacts",
        "source_files",
        "receipt_fingerprint",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError(f"fresh BASE receipt must contain exactly {sorted(required)}")
    if receipt["schema_version"] != RECEIPT_SCHEMA:
        raise ValueError(f"fresh BASE receipt must use {RECEIPT_SCHEMA}")
    if receipt["receipt_fingerprint"] != fingerprint(
        {key: value for key, value in receipt.items() if key != "receipt_fingerprint"}
    ):
        raise ValueError("fresh BASE receipt fingerprint mismatch")
    if (
        receipt["derived_experiment_id"] != lineage["derived_experiment_id"]
        or receipt["round_id"] != lineage["fresh_base_round_id"]
    ):
        raise ValueError("fresh BASE receipt family/round identity mismatch")
    if _identity(receipt["seed_identity"], "fresh BASE seed identity") != _identity(
        lineage["seed_identity"], "lineage seed identity"
    ):
        raise ValueError("fresh BASE receipt seed identity mismatch")
    gates = receipt["gates"]
    if (
        not isinstance(gates, dict)
        or set(gates) != REQUIRED_GATES
        or any(value != "passed" for value in gates.values())
    ):
        raise ValueError("fresh BASE receipt requires all gates passed")
    if receipt["clean_run_count"] != 5 or receipt["stable"] is not True:
        raise ValueError("fresh BASE receipt requires exactly five stable clean runs")
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError(
            f"fresh BASE artifacts must contain exactly {sorted(REQUIRED_ARTIFACTS)}"
        )
    artifact_paths = {
        name: _under(
            family_root, relative, f"fresh BASE artifact {name}", must_exist=True
        )
        for name, relative in artifacts.items()
    }
    records = receipt["source_files"]
    if not isinstance(records, list) or not records:
        raise ValueError("fresh BASE source_files must be non-empty")
    indexed = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "file_fingerprint",
        }:
            raise ValueError("fresh BASE source file record is invalid")
        source = _under(
            family_root, record["path"], "fresh BASE source file", must_exist=True
        )
        relative = str(source.relative_to(family_root))
        if relative in indexed:
            raise ValueError("fresh BASE source_files contains duplicates")
        if (
            source.stat().st_size != record["size_bytes"]
            or file_fingerprint(source) != record["file_fingerprint"]
        ):
            raise ValueError(f"fresh BASE source file changed: {relative}")
        indexed[relative] = record
    missing = sorted(
        str(path.relative_to(family_root))
        for path in artifact_paths.values()
        if str(path.relative_to(family_root)) not in indexed
    )
    if missing:
        raise ValueError(
            "fresh BASE artifacts are not bound by source_files: " + ", ".join(missing)
        )
    return receipt


def _locked_registry(registry_path):
    registry_path = Path(registry_path).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock = registry_path.with_name(registry_path.name + ".lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    return registry_path, lock


def _entry_and_lineage(registry, family_id):
    entry = registry.get("families", {}).get(family_id)
    if not isinstance(entry, dict):
        raise ValueError(f"derived family is not registered: {family_id}")
    family = Path(entry["family_path"]).resolve()
    lineage = _load(family / "lineage.json")
    if bootstrap_derived_family._fingerprint(lineage) != entry["lineage_fingerprint"]:
        raise ValueError("derived family lineage fingerprint mismatch")
    return entry, family, lineage


def run_fresh_base(plan_path, registry_path):
    plan = validate_plan(_load(plan_path))
    registry_path, lock = _locked_registry(registry_path)
    try:
        registry = bootstrap_derived_family._read_registry(registry_path)
        entry, family, lineage = _entry_and_lineage(registry, plan["family_id"])
        if (
            plan["family_root"] != str(family)
            or plan["round_id"] != lineage["fresh_base_round_id"]
        ):
            raise ValueError("fresh BASE plan differs from registered family lineage")
        if entry["status"] == "fresh_base_completed":
            validate_receipt(family / entry["fresh_base_receipt"], family, lineage)
            return entry
        if entry["status"] not in {"seed_registered", "fresh_base_failed"}:
            raise ValueError(f"fresh BASE cannot start from status {entry['status']}")
    finally:
        lock.close()
    receipt_path = _under(family, plan["receipt"], "fresh BASE receipt")
    command_manifest = _under(
        family, plan["command_manifest"], "fresh BASE command manifest"
    )
    if receipt_path.exists():
        raise ValueError("fresh BASE receipt existed before command execution")
    command = device_lease_runner.run_command(
        plan["command_argv"],
        lease_root=plan["lease_root"],
        device_ids=plan["device_ids"],
        timeout_seconds=plan["timeout_seconds"],
        cwd=plan["cwd"],
        manifest_out=command_manifest,
        environment=plan["environment"],
        command_id=plan["round_id"],
        lease_timeout_seconds=plan["lease_timeout_seconds"],
    )
    failure = None
    if command["status"] != "passed":
        failure = f"fresh BASE command {command['status']}: {command['reason']}"
    else:
        try:
            receipt = validate_receipt(receipt_path, family, lineage)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failure = f"fresh BASE receipt invalid: {error}"
    registry_path, lock = _locked_registry(registry_path)
    try:
        registry = bootstrap_derived_family._read_registry(registry_path)
        entry, _, _ = _entry_and_lineage(registry, plan["family_id"])
        if failure:
            entry.update(
                {
                    "status": "fresh_base_failed",
                    "fresh_base_required": True,
                    "ledger_merge_allowed": False,
                    "fresh_base_failure": failure,
                    "fresh_base_command_manifest": plan["command_manifest"],
                }
            )
        else:
            entry.update(
                {
                    "status": "fresh_base_completed",
                    "fresh_base_required": False,
                    "ledger_merge_allowed": False,
                    "fresh_base_receipt": plan["receipt"],
                    "fresh_base_receipt_fingerprint": receipt["receipt_fingerprint"],
                    "fresh_base_command_manifest": plan["command_manifest"],
                }
            )
            entry.pop("fresh_base_failure", None)
        bootstrap_derived_family._atomic_json(registry_path, registry)
    finally:
        lock.close()
    if failure:
        raise RuntimeError(failure)
    return entry


def complete_lifecycle(registry_path, family_id, result_path):
    result_path = Path(result_path).resolve()
    result = _load(result_path)
    validation = experiment_ledger.validate_experiment_result(
        result, artifact_root=result_path.parent
    )
    if not validation["valid"]:
        raise ValueError(
            "ordinary experiment result invalid: " + "; ".join(validation["errors"])
        )
    registry_path, lock = _locked_registry(registry_path)
    try:
        registry = bootstrap_derived_family._read_registry(registry_path)
        entry, family, lineage = _entry_and_lineage(registry, family_id)
        if entry["status"] not in {
            "fresh_base_completed",
            "ordinary_lifecycle_completed",
        }:
            raise ValueError("ordinary lifecycle requires fresh_base_completed")
        try:
            relative_result = str(result_path.relative_to(family))
        except ValueError as error:
            raise ValueError(
                "ordinary experiment result must be inside derived family"
            ) from error
        if (
            result.get("experiment_id") != family_id
            or result.get("parent_experiment_id") != family_id
        ):
            raise ValueError("ordinary experiment result lineage identity mismatch")
        rounds = result.get("rounds")
        if not isinstance(rounds, list) or not any(
            isinstance(item, dict)
            and item.get("round_id") == lineage["fresh_base_round_id"]
            and item.get("round_kind") in {"base", "automatic_aot"}
            for item in rounds
        ):
            raise ValueError(
                "ordinary experiment result does not contain the registered fresh BASE"
            )
        result_fp = file_fingerprint(result_path)
        if entry["status"] == "ordinary_lifecycle_completed":
            if entry.get("ordinary_result_fingerprint") != result_fp:
                raise ValueError(
                    "ordinary lifecycle already completed with another result"
                )
            return entry
        entry.update(
            {
                "status": "ordinary_lifecycle_completed",
                "ordinary_result": relative_result,
                "ordinary_result_fingerprint": result_fp,
                "ledger_merge_allowed": True,
            }
        )
        bootstrap_derived_family._atomic_json(registry_path, registry)
        return entry
    finally:
        lock.close()


def merge_ledger(registry_path, family_id, ledger_path, output_path):
    ledger_path = Path(ledger_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError(f"ledger output already exists: {output_path}")
    registry_path, lock = _locked_registry(registry_path)
    try:
        registry = bootstrap_derived_family._read_registry(registry_path)
        entry, family, _ = _entry_and_lineage(registry, family_id)
        if (
            entry["status"] != "ordinary_lifecycle_completed"
            or entry.get("ledger_merge_allowed") is not True
        ):
            raise ValueError("ledger merge requires ordinary_lifecycle_completed")
        result_path = family / entry["ordinary_result"]
        if file_fingerprint(result_path) != entry["ordinary_result_fingerprint"]:
            raise ValueError("ordinary experiment result changed before ledger merge")
        ledger = _load(ledger_path) if ledger_path.exists() else {}
        result = _load(result_path)
        merged = experiment_ledger.merge_experiment_result(
            ledger, result, artifact_root=result_path.parent
        )
        _atomic_json(output_path, merged)
        entry.update(
            {
                "status": "ledger_merged",
                "ledger_merge_allowed": False,
                "merged_ledger": str(output_path),
                "merged_ledger_fingerprint": file_fingerprint(output_path),
            }
        )
        bootstrap_derived_family._atomic_json(registry_path, registry)
        return entry
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-plan")
    freeze.add_argument("--draft", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run-base")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--registry", type=Path, required=True)
    complete = commands.add_parser("complete-lifecycle")
    complete.add_argument("--registry", type=Path, required=True)
    complete.add_argument("--family-id", required=True)
    complete.add_argument("--result", type=Path, required=True)
    merge = commands.add_parser("merge-ledger")
    merge.add_argument("--registry", type=Path, required=True)
    merge.add_argument("--family-id", required=True)
    merge.add_argument("--ledger", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-plan":
            if args.out.exists():
                raise ValueError(f"fresh BASE plan output already exists: {args.out}")
            result = freeze_plan(_load(args.draft))
            _atomic_json(args.out, result)
        elif args.command == "run-base":
            result = run_fresh_base(args.plan, args.registry)
        elif args.command == "complete-lifecycle":
            result = complete_lifecycle(args.registry, args.family_id, args.result)
        else:
            result = merge_ledger(
                args.registry, args.family_id, args.ledger, args.output
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
