#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Merge audited graph/source dependency provider fragments into hard-edge evidence."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


FRAGMENT_SET_SCHEMA = "superkernel-multistream-dependency-fragment-set-v1"
EVIDENCE_SCHEMA = "superkernel-multistream-dependency-evidence-v1"
PROVIDER_API_VERSION = "superkernel-multistream-dependency-provider-v1"
PROVIDER_KINDS = {"fx_graph", "exported_graph", "source_review", "runtime_metadata"}
HARD_DEPENDENCY_KINDS = {
    "DATA",
    "STREAM_ORDER",
    "EVENT",
    "WAIT",
    "BARRIER",
    "COMMUNICATION",
    "CACHE_MUTATION",
    "SIDE_EFFECT",
    "CONTROL_FLOW",
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


def _file_record(root, value, label):
    if not isinstance(value, dict) or set(value) != {
        "path",
        "size_bytes",
        "file_fingerprint",
    }:
        raise ValueError(f"{label} fields are invalid")
    path = _rooted(root, value["path"], f"{label}.path")
    relative = path.relative_to(Path(root).resolve()).as_posix()
    actual = file_fingerprint(path)
    if (
        value["size_bytes"] != path.stat().st_size
        or value["file_fingerprint"] != actual
    ):
        raise ValueError(f"{label} file changed: {relative}")
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "file_fingerprint": actual,
    }


def source_file_record(path, artifact_root):
    root = Path(artifact_root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("provider file escapes artifact root") from error
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "file_fingerprint": file_fingerprint(path),
    }


def _has_cycle(nodes, edges):
    incoming = {node: 0 for node in nodes}
    children = defaultdict(set)
    for before, after in edges:
        if after not in children[before]:
            children[before].add(after)
            incoming[after] += 1
    ready = sorted(node for node, count in incoming.items() if count == 0)
    visited = 0
    while ready:
        node = ready.pop(0)
        visited += 1
        for child in sorted(children[node]):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
                ready.sort()
    return visited != len(nodes)


def _provider(root, value, label, statement_ids):
    required = {
        "provider_id",
        "provider_api_version",
        "provider_kind",
        "implementation",
        "input_files",
        "covered_dependency_kinds",
        "edges",
        "blockers",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    if value["provider_api_version"] != PROVIDER_API_VERSION:
        raise ValueError(f"{label} must use {PROVIDER_API_VERSION}")
    provider_id = _text(value["provider_id"], f"{label}.provider_id")
    provider_kind = _text(value["provider_kind"], f"{label}.provider_kind")
    if provider_kind not in PROVIDER_KINDS:
        raise ValueError(f"{label}.provider_kind is unsupported")
    implementation = _file_record(
        root, value["implementation"], f"{label}.implementation"
    )
    inputs = value["input_files"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{label}.input_files must be non-empty")
    input_files = [
        _file_record(root, item, f"{label}.input_files[{index}]")
        for index, item in enumerate(inputs)
    ]
    paths = [item["path"] for item in input_files]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label}.input_files contains duplicates")
    coverage = value["covered_dependency_kinds"]
    if not isinstance(coverage, list) or not coverage:
        raise ValueError(f"{label}.covered_dependency_kinds must be non-empty")
    coverage = [
        _text(item, f"{label}.covered_dependency_kinds[]").upper() for item in coverage
    ]
    if len(coverage) != len(set(coverage)) or set(coverage) - HARD_DEPENDENCY_KINDS:
        raise ValueError(f"{label}.covered_dependency_kinds is invalid")
    edges = value["edges"]
    if not isinstance(edges, list):
        raise ValueError(f"{label}.edges must be a list")
    normalized_edges = []
    seen = set()
    for index, edge in enumerate(edges):
        edge_label = f"{label}.edges[{index}]"
        if not isinstance(edge, dict) or set(edge) != {
            "before",
            "after",
            "kind",
            "evidence_locator",
        }:
            raise ValueError(f"{edge_label} fields are invalid")
        before = _text(edge["before"], f"{edge_label}.before")
        after = _text(edge["after"], f"{edge_label}.after")
        kind = _text(edge["kind"], f"{edge_label}.kind").upper()
        if before == after or before not in statement_ids or after not in statement_ids:
            raise ValueError(f"{edge_label} references an invalid statement")
        if kind not in coverage:
            raise ValueError(f"{edge_label}.kind is not covered by its provider")
        key = (before, after, kind)
        if key in seen:
            raise ValueError(f"{label}.edges contains a duplicate edge")
        seen.add(key)
        normalized_edges.append(
            {
                "before": before,
                "after": after,
                "kind": kind,
                "evidence_locator": _text(
                    edge["evidence_locator"], f"{edge_label}.evidence_locator"
                ),
            }
        )
    blockers = value["blockers"]
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in blockers
    ):
        raise ValueError(f"{label}.blockers must be canonical strings")
    return {
        "provider_id": provider_id,
        "provider_api_version": PROVIDER_API_VERSION,
        "provider_kind": provider_kind,
        "implementation": implementation,
        "input_files": sorted(input_files, key=lambda item: item["path"]),
        "covered_dependency_kinds": sorted(coverage),
        "edges": sorted(
            normalized_edges,
            key=lambda item: (item["before"], item["after"], item["kind"]),
        ),
        "blockers": sorted(set(blockers)),
    }


def build(fragment_set_path, artifact_root=None):
    fragment_set_path = Path(fragment_set_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else fragment_set_path.parent
    try:
        fragment_set_path.relative_to(root)
    except ValueError as error:
        raise ValueError("dependency fragment set escapes artifact root") from error
    value = json.loads(fragment_set_path.read_text())
    required = {
        "schema_version",
        "fragment_set_id",
        "request_fingerprint",
        "range_id",
        "graph_occurrence_fingerprint",
        "statement_ids",
        "required_dependency_kinds",
        "providers",
        "fragment_set_fingerprint",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"dependency fragment set must contain exactly {sorted(required)}"
        )
    if value["schema_version"] != FRAGMENT_SET_SCHEMA:
        raise ValueError(f"dependency fragment set must use {FRAGMENT_SET_SCHEMA}")
    unsigned = {
        key: item for key, item in value.items() if key != "fragment_set_fingerprint"
    }
    if value["fragment_set_fingerprint"] != fingerprint(unsigned):
        raise ValueError("dependency fragment set fingerprint mismatch")
    statement_ids = value["statement_ids"]
    if not isinstance(statement_ids, list) or len(statement_ids) < 2:
        raise ValueError("statement_ids requires at least two entries")
    statement_ids = [_text(item, "statement_ids[]") for item in statement_ids]
    if len(statement_ids) != len(set(statement_ids)):
        raise ValueError("statement_ids contains duplicates")
    required_kinds = value["required_dependency_kinds"]
    if (
        not isinstance(required_kinds, list)
        or set(required_kinds) != HARD_DEPENDENCY_KINDS
    ):
        raise ValueError(
            "required_dependency_kinds must contain every hard dependency kind"
        )
    providers = value["providers"]
    if not isinstance(providers, list) or not providers:
        raise ValueError("providers must be a non-empty list")
    providers = [
        _provider(root, item, f"providers[{index}]", set(statement_ids))
        for index, item in enumerate(providers)
    ]
    provider_ids = [item["provider_id"] for item in providers]
    if len(provider_ids) != len(set(provider_ids)):
        raise ValueError("providers contains duplicate provider_id")
    coverage = defaultdict(list)
    edge_sources = defaultdict(list)
    blockers = []
    for provider in providers:
        for kind in provider["covered_dependency_kinds"]:
            coverage[kind].append(provider["provider_id"])
        for blocker in provider["blockers"]:
            blockers.append(
                {
                    "code": "provider_blocked",
                    "provider_id": provider["provider_id"],
                    "detail": blocker,
                }
            )
        for edge in provider["edges"]:
            key = (edge["before"], edge["after"], edge["kind"])
            edge_sources[key].append(
                {
                    "provider_id": provider["provider_id"],
                    "evidence_locator": edge["evidence_locator"],
                }
            )
    missing = sorted(HARD_DEPENDENCY_KINDS - set(coverage))
    if missing:
        blockers.append({"code": "dependency_kind_coverage_missing", "kinds": missing})
    hard_dependencies = [
        {"before": key[0], "after": key[1], "kind": key[2]}
        for key in sorted(edge_sources)
    ]
    if _has_cycle(
        statement_ids, {(item["before"], item["after"]) for item in hard_dependencies}
    ):
        blockers.append({"code": "hard_dependency_cycle"})
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "fragment_set_id": _text(value["fragment_set_id"], "fragment_set_id"),
        "request_fingerprint": _text(
            value["request_fingerprint"], "request_fingerprint"
        ),
        "range_id": _text(value["range_id"], "range_id"),
        "graph_occurrence_fingerprint": _text(
            value["graph_occurrence_fingerprint"], "graph_occurrence_fingerprint"
        ),
        "statement_ids": statement_ids,
        "required_dependency_kinds": sorted(HARD_DEPENDENCY_KINDS),
        "coverage": {
            kind: sorted(coverage.get(kind, []))
            for kind in sorted(HARD_DEPENDENCY_KINDS)
        },
        "hard_dependencies": hard_dependencies,
        "edge_sources": [
            {
                "before": key[0],
                "after": key[1],
                "kind": key[2],
                "sources": sorted(
                    edge_sources[key],
                    key=lambda item: (item["provider_id"], item["evidence_locator"]),
                ),
            }
            for key in sorted(edge_sources)
        ],
        "providers": sorted(providers, key=lambda item: item["provider_id"]),
        "fragment_set": {
            "path": fragment_set_path.relative_to(root).as_posix(),
            "file_fingerprint": file_fingerprint(fragment_set_path),
            "fragment_set_fingerprint": value["fragment_set_fingerprint"],
        },
        "blockers": sorted(blockers, key=_canonical),
    }
    evidence["complete"] = not evidence["blockers"]
    evidence["evidence_fingerprint"] = fingerprint(evidence)
    return evidence


def validate(
    evidence_path,
    artifact_root=None,
    *,
    require_complete=False,
    request_fingerprint=None,
    range_id=None,
    graph_occurrence_fingerprint=None,
    statement_ids=None,
):
    evidence_path = Path(evidence_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else evidence_path.parent
    value = json.loads(evidence_path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError(f"dependency evidence must use {EVIDENCE_SCHEMA}")
    actual = value.get("evidence_fingerprint")
    unsigned = {
        key: item for key, item in value.items() if key != "evidence_fingerprint"
    }
    if actual != fingerprint(unsigned):
        raise ValueError("dependency evidence fingerprint mismatch")
    fragment_path = _rooted(
        root, value.get("fragment_set", {}).get("path"), "evidence.fragment_set.path"
    )
    rebuilt = build(fragment_path, root)
    if _canonical(rebuilt) != _canonical(value):
        raise ValueError("dependency evidence differs from deterministic replay")
    expected = {
        "request_fingerprint": request_fingerprint,
        "range_id": range_id,
        "graph_occurrence_fingerprint": graph_occurrence_fingerprint,
    }
    for field, wanted in expected.items():
        if wanted is not None and value[field] != wanted:
            raise ValueError(f"dependency evidence {field} mismatch")
    if statement_ids is not None and value["statement_ids"] != list(statement_ids):
        raise ValueError("dependency evidence statement_ids mismatch")
    if require_complete and value["complete"] is not True:
        codes = sorted({item["code"] for item in value["blockers"]})
        raise ValueError(f"dependency evidence is incomplete: {', '.join(codes)}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--fragments", type=Path, required=True)
    create.add_argument("--artifact-root", type=Path)
    create.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--evidence", type=Path, required=True)
    check.add_argument("--artifact-root", type=Path)
    check.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.fragments, args.artifact_root)
            _atomic_json(args.out, result)
        else:
            result = validate(
                args.evidence,
                args.artifact_root,
                require_complete=args.require_complete,
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
