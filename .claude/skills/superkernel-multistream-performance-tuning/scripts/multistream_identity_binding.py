#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build an auditable source/SK-off/SK-on identity registry for capture plugins."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


OBSERVATIONS_SCHEMA = "superkernel-multistream-identity-observations-v1"
REGISTRY_SCHEMA = "superkernel-multistream-identity-registry-v1"
DOMAINS = {"source", "sk_off", "sk_on"}
BINDING_METHODS = {
    "source": {"exact_source_span", "graph_debug_handle"},
    "sk_off": {"compiler_origin_uid", "projected_trace_exact"},
    "sk_on": {"compiler_origin_uid", "sk_meta_origin_exact"},
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
    if path.exists():
        raise ValueError(f"output already exists: {path}")
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
    relative = _relative(value, label)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes artifact root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


def _json_value(value, label):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _text(key, f"{label} key")
            _json_value(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} contains an unsupported JSON value")


def _validate_evidence_files(root, values):
    if not isinstance(values, list) or not values:
        raise ValueError("evidence_files must be a non-empty list")
    result = {}
    for index, record in enumerate(values):
        label = f"evidence_files[{index}]"
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "file_fingerprint",
        }:
            raise ValueError(f"{label} fields are invalid")
        path = _rooted(root, record["path"], f"{label}.path")
        relative = path.relative_to(Path(root).resolve()).as_posix()
        if relative in result:
            raise ValueError(f"duplicate evidence file: {relative}")
        actual = file_fingerprint(path)
        if (
            record["size_bytes"] != path.stat().st_size
            or record["file_fingerprint"] != actual
        ):
            raise ValueError(f"evidence file changed: {relative}")
        result[relative] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "file_fingerprint": actual,
        }
    return result


def _identity(value, label):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    _json_value(value, label)
    return json.loads(_canonical(value))


def _observation(value, label, evidence_files):
    required = {
        "domain",
        "identity",
        "operator_id",
        "statement_id",
        "alignment_id",
        "binding_method",
        "evidence_path",
        "evidence_locator",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    domain = _text(value["domain"], f"{label}.domain")
    if domain not in DOMAINS:
        raise ValueError(f"{label}.domain must be one of {sorted(DOMAINS)}")
    method = _text(value["binding_method"], f"{label}.binding_method")
    if method not in BINDING_METHODS[domain]:
        raise ValueError(
            f"{label}.binding_method is not an exact method allowed for {domain}"
        )
    alignment = value["alignment_id"]
    if domain == "source":
        if alignment is not None:
            raise ValueError(f"{label}.alignment_id must be null for source")
    else:
        alignment = _text(alignment, f"{label}.alignment_id")
    evidence_path = _relative(value["evidence_path"], f"{label}.evidence_path")
    if evidence_path not in evidence_files:
        raise ValueError(f"{label}.evidence_path is not bound by evidence_files")
    identity = _identity(value["identity"], f"{label}.identity")
    return {
        "domain": domain,
        "identity": identity,
        "identity_fingerprint": fingerprint(identity),
        "operator_id": _text(value["operator_id"], f"{label}.operator_id"),
        "statement_id": _text(value["statement_id"], f"{label}.statement_id"),
        "alignment_id": alignment,
        "binding_method": method,
        "evidence_path": evidence_path,
        "evidence_locator": _text(
            value["evidence_locator"], f"{label}.evidence_locator"
        ),
    }


def _blocker(code, **context):
    return {"code": code, "context": context}


def build(observations_path, artifact_root=None):
    observations_path = Path(observations_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else observations_path.parent
    try:
        observations_path.relative_to(root)
    except ValueError as error:
        raise ValueError("identity observations escape artifact root") from error
    value = json.loads(observations_path.read_text())
    required = {
        "schema_version",
        "observation_set_id",
        "request_fingerprint",
        "evidence_files",
        "observations",
        "observations_fingerprint",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"identity observations must contain exactly {sorted(required)}"
        )
    if value["schema_version"] != OBSERVATIONS_SCHEMA:
        raise ValueError(f"identity observations must use {OBSERVATIONS_SCHEMA}")
    unsigned = {
        key: item for key, item in value.items() if key != "observations_fingerprint"
    }
    if value["observations_fingerprint"] != fingerprint(unsigned):
        raise ValueError("identity observations fingerprint mismatch")
    evidence_files = _validate_evidence_files(root, value["evidence_files"])
    raw = value["observations"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("observations must be a non-empty list")
    observations = [
        _observation(item, f"observations[{index}]", evidence_files)
        for index, item in enumerate(raw)
    ]
    blockers = []
    raw_assignments = defaultdict(set)
    statement_assignments = defaultdict(set)
    pairs = defaultdict(lambda: defaultdict(list))
    for item in observations:
        raw_key = (item["domain"], item["identity_fingerprint"], item["alignment_id"])
        raw_assignments[raw_key].add((item["operator_id"], item["statement_id"]))
        statement_assignments[item["operator_id"]].add(item["statement_id"])
        pairs[(item["operator_id"], item["statement_id"])][item["domain"]].append(item)
    for (domain, identity_fp, alignment), assignments in sorted(
        raw_assignments.items()
    ):
        statement_ids = {item[1] for item in assignments}
        if len(statement_ids) > 1 or (domain != "source" and len(assignments) > 1):
            blockers.append(
                _blocker(
                    "ambiguous_raw_identity",
                    domain=domain,
                    identity_fingerprint=identity_fp,
                    alignment_id=alignment,
                    assignments=[list(item) for item in sorted(assignments)],
                )
            )
    for operator_id, statements in sorted(statement_assignments.items()):
        if len(statements) > 1:
            blockers.append(
                _blocker(
                    "operator_maps_to_multiple_statements",
                    operator_id=operator_id,
                    statement_ids=sorted(statements),
                )
            )
    entries = []
    for (operator_id, statement_id), domains in sorted(pairs.items()):
        missing = sorted(DOMAINS - set(domains))
        if missing:
            blockers.append(
                _blocker(
                    "identity_domain_missing",
                    operator_id=operator_id,
                    statement_id=statement_id,
                    missing_domains=missing,
                )
            )
        off_ids = {item["alignment_id"] for item in domains.get("sk_off", [])}
        on_ids = {item["alignment_id"] for item in domains.get("sk_on", [])}
        if len(off_ids) < 3 or len(on_ids) < 3:
            blockers.append(
                _blocker(
                    "aligned_occurrences_insufficient",
                    operator_id=operator_id,
                    statement_id=statement_id,
                    sk_off_count=len(off_ids),
                    sk_on_count=len(on_ids),
                )
            )
        elif off_ids != on_ids:
            blockers.append(
                _blocker(
                    "occurrence_alignment_mismatch",
                    operator_id=operator_id,
                    statement_id=statement_id,
                    sk_off_alignment_ids=sorted(off_ids),
                    sk_on_alignment_ids=sorted(on_ids),
                )
            )
        entry_domains = {}
        for domain in sorted(DOMAINS):
            entry_domains[domain] = sorted(
                domains.get(domain, []),
                key=lambda item: (
                    item["alignment_id"] or "",
                    item["identity_fingerprint"],
                    item["evidence_path"],
                    item["evidence_locator"],
                ),
            )
        entries.append(
            {
                "operator_id": operator_id,
                "statement_id": statement_id,
                "aligned_occurrence_ids": sorted(off_ids & on_ids),
                "domains": entry_domains,
            }
        )
    forward_index = []
    for (domain, identity_fp, alignment), assignments in sorted(
        raw_assignments.items()
    ):
        forward_index.append(
            {
                "domain": domain,
                "identity_fingerprint": identity_fp,
                "alignment_id": alignment,
                "assignments": [
                    {"operator_id": item[0], "statement_id": item[1]}
                    for item in sorted(assignments)
                ],
            }
        )
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "observation_set_id": _text(value["observation_set_id"], "observation_set_id"),
        "request_fingerprint": _text(
            value["request_fingerprint"], "request_fingerprint"
        ),
        "observations": {
            "path": observations_path.relative_to(root).as_posix(),
            "file_fingerprint": file_fingerprint(observations_path),
            "observations_fingerprint": value["observations_fingerprint"],
        },
        "evidence_files": [evidence_files[key] for key in sorted(evidence_files)],
        "entries": entries,
        "forward_index": forward_index,
        "blockers": sorted(blockers, key=_canonical),
    }
    registry["complete"] = not registry["blockers"]
    registry["registry_fingerprint"] = fingerprint(registry)
    return registry


def validate(registry_path, artifact_root=None, *, require_complete=False):
    registry_path = Path(registry_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else registry_path.parent
    value = json.loads(registry_path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError(f"identity registry must use {REGISTRY_SCHEMA}")
    actual = value.get("registry_fingerprint")
    unsigned = {
        key: item for key, item in value.items() if key != "registry_fingerprint"
    }
    if actual != fingerprint(unsigned):
        raise ValueError("identity registry fingerprint mismatch")
    observation_path = _rooted(
        root, value.get("observations", {}).get("path"), "registry.observations.path"
    )
    rebuilt = build(observation_path, root)
    if _canonical(rebuilt) != _canonical(value):
        raise ValueError("identity registry differs from deterministic replay")
    if require_complete and value["complete"] is not True:
        codes = sorted({item["code"] for item in value["blockers"]})
        raise ValueError(f"identity registry is incomplete: {', '.join(codes)}")
    return {
        "valid": True,
        "complete": value["complete"],
        "entry_count": len(value["entries"]),
        "blocker_count": len(value["blockers"]),
        "registry_fingerprint": actual,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--observations", type=Path, required=True)
    create.add_argument("--artifact-root", type=Path)
    create.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--registry", type=Path, required=True)
    check.add_argument("--artifact-root", type=Path)
    check.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.observations, args.artifact_root)
            _atomic_json(args.out, result)
        else:
            result = validate(
                args.registry,
                args.artifact_root,
                require_complete=args.require_complete,
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
