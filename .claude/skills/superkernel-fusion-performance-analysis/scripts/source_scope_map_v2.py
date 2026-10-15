#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build and fail-closed validate source_scope_map_v2 evidence bundles."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict, deque
from pathlib import Path

from build_sk_source_map import build_sk_source_map
from marker_only_calibration import validate_marker_only_calibration
from project_calibration_graph import project_graphs
from source_calibration_common import (
    artifact_reference,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_hashed_json,
    load_json,
    require_list,
    require_object,
    require_sha256,
    require_text,
    resolve_evidence_path,
    safe_relative_path,
    validate_exact_keys,
    validate_byte_span,
)


ARTIFACT_KEYS = (
    "provenance_dag",
    "source_manifest",
    "sk_source_map",
    "block_instance_inventory",
    "calibration_projection",
    "original_fused_inventory",
    "unit_assignments",
    "original_candidate_collection_manifest",
    "calibration_collection_manifest",
    "baseline_collection_manifest",
)

AUTO_SOURCE_ARTIFACT_KEYS = (
    "calibration_source_manifest",
    "marker_only_calibration",
)


def _fingerprint_without(value, field):
    detached = copy.deepcopy(value)
    detached.pop(field, None)
    return canonical_sha256(detached)


def _content_fingerprint(value):
    detached = copy.deepcopy(value)
    detached["provenance"].pop("source_scope_map_content_fingerprint", None)
    return canonical_sha256(detached)


def _load_reference(root, provenance, key):
    if key not in provenance:
        raise ValueError(f"source_scope_map_v2 provenance is missing {key}")
    reference = validate_exact_keys(
        provenance[key],
        {"relative_path", "sha256", "fingerprint", "protocol"},
        f"provenance.{key}",
        required=("relative_path", "sha256"),
    )
    return load_hashed_json(root, reference, f"provenance.{key}")


def _validate_snapshot(root, provenance, source_manifest):
    reference = require_object(
        provenance.get("source_snapshot_root"), "provenance.source_snapshot_root"
    )
    relative = safe_relative_path(
        reference.get("relative_path"), "provenance.source_snapshot_root.relative_path"
    )
    snapshot_root = resolve_evidence_path(
        root, relative.as_posix(), "provenance.source_snapshot_root.relative_path"
    )
    if not snapshot_root.is_dir() or snapshot_root.is_symlink():
        raise ValueError("source snapshot root must be a regular directory")
    manifest_path = snapshot_root / "source-snapshot-manifest.json"
    expected = require_sha256(
        reference.get("manifest_sha256"),
        "provenance.source_snapshot_root.manifest_sha256",
    )
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or file_sha256(manifest_path) != expected
    ):
        raise ValueError("source snapshot manifest SHA-256 mismatch")
    snapshot_manifest = load_json(manifest_path, "source snapshot manifest")
    if snapshot_manifest.get("protocol") != "source_snapshot_manifest_v1":
        raise ValueError("source snapshot manifest protocol is unsupported")
    if snapshot_manifest.get("manifest_fingerprint") != _fingerprint_without(
        snapshot_manifest, "manifest_fingerprint"
    ):
        raise ValueError("source snapshot manifest fingerprint mismatch")
    observed = {}
    for index, item in enumerate(
        require_list(snapshot_manifest.get("files"), "snapshot files")
    ):
        item = require_object(item, f"snapshot files[{index}]")
        path = resolve_evidence_path(
            snapshot_root,
            item.get("relative_path"),
            f"snapshot files[{index}].relative_path",
        )
        sha = require_sha256(item.get("sha256"), f"snapshot files[{index}].sha256")
        if not path.is_file() or path.is_symlink() or file_sha256(path) != sha:
            raise ValueError(f"snapshot file evidence mismatch: {path}")
        observed[
            safe_relative_path(item["relative_path"], "snapshot path").as_posix()
        ] = path
    if set(observed) != set(source_manifest.get("file_sha256", {})):
        raise ValueError("source snapshot file set differs from source manifest")
    for name, expected_sha in source_manifest["file_sha256"].items():
        if file_sha256(observed[name]) != expected_sha:
            raise ValueError(f"source manifest file SHA-256 mismatch for {name}")
    for index, unit in enumerate(
        require_list(source_manifest.get("units"), "source units")
    ):
        path = observed.get(unit.get("source_file"))
        if path is None:
            raise ValueError(f"source unit {index} references an unarchived file")
        content = path.read_bytes()
        start, end = validate_byte_span(unit, len(content), f"source units[{index}]")
        import hashlib

        if hashlib.sha256(content[start:end]).hexdigest() != unit.get(
            "source_span_sha256"
        ):
            raise ValueError(f"source unit {index} byte span fingerprint mismatch")
        if canonical_sha256(
            require_object(
                unit.get("normalized_syntax"), f"source unit {index}.normalized_syntax"
            )
        ) != unit.get("syntax_tree_hash"):
            raise ValueError(f"source unit {index} syntax-tree fingerprint mismatch")
        if source_manifest.get("revision_role") != "stable_source":
            for name, kind in (("begin", "scope_begin"), ("end", "scope_end")):
                marker = require_object(
                    unit.get("marker_operations", {}).get(name),
                    f"source unit {index} {name}",
                )
                marker_start, marker_end = validate_byte_span(
                    marker, len(content), f"source unit {index} {name}"
                )
                if marker.get("operation_kind") != kind or not (
                    start <= marker_start < marker_end <= end
                ):
                    raise ValueError(f"source unit {index} {name} marker is invalid")
                if hashlib.sha256(
                    content[marker_start:marker_end]
                ).hexdigest() != marker.get("operation_sha256"):
                    raise ValueError(
                        f"source unit {index} {name} marker fingerprint mismatch"
                    )
    return snapshot_manifest


def _validate_dag(root, dag, references, projection, assignments, fused_inventory):
    if dag.get("protocol") != "source_mapping_provenance_dag_v1":
        raise ValueError("provenance DAG protocol is unsupported")
    nodes = {}
    for index, raw in enumerate(require_list(dag.get("nodes"), "provenance DAG nodes")):
        item = require_object(raw, f"provenance DAG nodes[{index}]")
        validate_exact_keys(
            item,
            {"artifact_id", "relative_path", "sha256", "depends_on"},
            f"provenance DAG nodes[{index}]",
            required=("artifact_id", "relative_path", "sha256", "depends_on"),
        )
        artifact_id = require_text(
            item.get("artifact_id"), f"DAG node {index}.artifact_id"
        )
        if artifact_id in nodes:
            raise ValueError(f"duplicate provenance DAG artifact_id {artifact_id}")
        nodes[artifact_id] = item
    by_path = {
        require_text(
            item.get("relative_path"), f"provenance DAG {key}.relative_path"
        ): item
        for key, item in nodes.items()
    }
    for artifact_id, item in nodes.items():
        artifact_path = resolve_evidence_path(
            root,
            item.get("relative_path"),
            f"provenance DAG {artifact_id}.relative_path",
        )
        expected_sha = require_sha256(
            item.get("sha256"), f"provenance DAG {artifact_id}.sha256"
        )
        if (
            not artifact_path.is_file()
            or artifact_path.is_symlink()
            or file_sha256(artifact_path) != expected_sha
        ):
            raise ValueError(f"provenance DAG artifact mismatch: {artifact_id}")
    for key, reference in references.items():
        path = reference.get("relative_path")
        if path not in by_path or by_path[path].get("sha256") != reference.get(
            "sha256"
        ):
            raise ValueError(f"provenance DAG does not seal {key}")
    dependencies = {}
    reverse = defaultdict(set)
    for artifact_id, item in nodes.items():
        deps = set(
            require_list(item.get("depends_on", []), f"DAG {artifact_id}.depends_on")
        )
        if artifact_id in deps or not deps.issubset(nodes):
            raise ValueError(f"DAG {artifact_id} has an invalid dependency")
        dependencies[artifact_id] = deps
        for dep in deps:
            reverse[dep].add(artifact_id)
    original_dependencies = copy.deepcopy(dependencies)
    queue = deque(sorted(key for key, deps in dependencies.items() if not deps))
    visited = []
    while queue:
        key = queue.popleft()
        visited.append(key)
        for owner in sorted(reverse[key]):
            dependencies[owner].remove(key)
            if not dependencies[owner]:
                queue.append(owner)
    if len(visited) != len(nodes):
        raise ValueError("provenance DAG contains a cycle")
    catalog = require_object(
        projection.get("evidence_catalog"), "projection.evidence_catalog"
    )
    required_catalog = {"original_graph", "calibration_graph", "runtime_validation"}
    if not required_catalog.issubset(catalog):
        raise ValueError("projection evidence_catalog is incomplete")

    def referenced_paths(value):
        paths = set()

        def collect(child):
            if isinstance(child, dict):
                if "relative_path" in child and "sha256" in child:
                    paths.add(child["relative_path"])
                for descendant in child.values():
                    collect(descendant)
            elif isinstance(child, list):
                for descendant in child:
                    collect(descendant)

        collect(value)
        return paths

    projection_evidence_paths = referenced_paths(
        [
            projection.get("evidence_catalog", {}),
            projection.get("runtime_validation", []),
        ]
    )
    assignment_evidence_paths = referenced_paths(
        assignments.get("evidence_catalog", {})
    )
    fused_evidence_paths = referenced_paths(fused_inventory.get("evidence_catalog", {}))
    evidence_paths = (
        projection_evidence_paths | assignment_evidence_paths | fused_evidence_paths
    )

    missing = sorted(evidence_paths - set(by_path))
    if missing:
        raise ValueError(f"provenance DAG omits projection evidence: {missing}")
    snapshot_reference = require_object(
        references.get("source_snapshot_manifest"), "source snapshot DAG reference"
    )
    snapshot_path = snapshot_reference.get("relative_path")
    if snapshot_path not in by_path or by_path[snapshot_path].get(
        "sha256"
    ) != snapshot_reference.get("sha256"):
        raise ValueError("provenance DAG does not seal the source snapshot manifest")
    declared_paths = {reference["relative_path"] for reference in references.values()}
    unused = sorted(set(by_path) - declared_paths - evidence_paths)
    if unused:
        raise ValueError(f"provenance DAG contains unreachable artifacts: {unused}")

    ids_by_path = {
        item["relative_path"]: artifact_id for artifact_id, item in nodes.items()
    }

    def depends_transitively(owner, dependency):
        pending = list(original_dependencies[owner])
        visited_dependencies = set()
        while pending:
            current = pending.pop()
            if current == dependency:
                return True
            if current not in visited_dependencies:
                visited_dependencies.add(current)
                pending.extend(original_dependencies[current])
        return False

    projection_id = ids_by_path[references["calibration_projection"]["relative_path"]]
    for evidence_path in projection_evidence_paths:
        if not depends_transitively(projection_id, ids_by_path[evidence_path]):
            raise ValueError(
                "projection evidence is not a dependency of calibration projection"
            )
    assignment_id = ids_by_path[references["unit_assignments"]["relative_path"]]
    for evidence_path in assignment_evidence_paths:
        if not depends_transitively(assignment_id, ids_by_path[evidence_path]):
            raise ValueError(
                "assignment evidence is not a dependency of unit assignments"
            )
    fused_id = ids_by_path[references["original_fused_inventory"]["relative_path"]]
    for evidence_path in fused_evidence_paths:
        if not depends_transitively(fused_id, ids_by_path[evidence_path]):
            raise ValueError(
                "fused log evidence is not a dependency of fused inventory"
            )
    source_manifest_id = ids_by_path[references["source_manifest"]["relative_path"]]
    if not depends_transitively(source_manifest_id, ids_by_path[snapshot_path]):
        raise ValueError("source snapshot is not a dependency of source manifest")
    sk_map_id = ids_by_path[references["sk_source_map"]["relative_path"]]
    for key in (
        "source_manifest",
        "block_instance_inventory",
        "calibration_projection",
        "original_fused_inventory",
        "unit_assignments",
    ):
        dependency_id = ids_by_path[references[key]["relative_path"]]
        if not depends_transitively(sk_map_id, dependency_id):
            raise ValueError(f"{key} is not a dependency of sk-source-map")


def _validate_normalized_record_evidence(
    root, artifact, items, *, label, identity_fields
):
    catalog = require_object(
        artifact.get("evidence_catalog"), f"{label}.evidence_catalog"
    )
    loaded_records = {}
    for artifact_id, reference in catalog.items():
        require_text(artifact_id, f"{label} evidence artifact_id")
        _, evidence = load_hashed_json(
            root, reference, f"{label}.evidence_catalog.{artifact_id}"
        )
        if evidence.get("protocol") != "normalized_log_records_v1":
            raise ValueError(f"{label} evidence {artifact_id} protocol is unsupported")
        records = {}
        for index, raw in enumerate(
            require_list(evidence.get("records"), f"{label} evidence records")
        ):
            record = require_object(
                raw, f"{label} evidence {artifact_id} record {index}"
            )
            record_id = require_text(
                record.get("record_id"), f"{label} evidence record_id"
            )
            if record_id in records:
                raise ValueError(
                    f"{label} evidence {artifact_id} has duplicate record_id"
                )
            fingerprint = require_sha256(
                record.get("record_fingerprint"), f"{label} evidence record fingerprint"
            )
            if fingerprint != _fingerprint_without(record, "record_fingerprint"):
                raise ValueError(f"{label} evidence record fingerprint mismatch")
            records[record_id] = record
        loaded_records[artifact_id] = records
    for index, item in enumerate(items):
        matched = False
        for evidence_index, reference in enumerate(
            require_list(
                item.get("evidence_records"), f"{label}[{index}].evidence_records"
            )
        ):
            reference = require_object(
                reference, f"{label}[{index}].evidence_records[{evidence_index}]"
            )
            artifact_id = require_text(
                reference.get("artifact_id"), "evidence artifact_id"
            )
            record_id = require_text(reference.get("record_id"), "evidence record_id")
            expected = require_sha256(
                reference.get("record_fingerprint"), "evidence record fingerprint"
            )
            record = loaded_records.get(artifact_id, {}).get(record_id)
            if record is None or record.get("record_fingerprint") != expected:
                raise ValueError(f"{label}[{index}] evidence record cannot be resolved")
            if all(record.get(field) == item.get(field) for field in identity_fields):
                matched = True
        if not matched:
            raise ValueError(f"{label}[{index}] lacks an identity-matching log record")


def _replay_calibration_projection(root, projection):
    catalog = require_object(
        projection.get("evidence_catalog"), "projection.evidence_catalog"
    )
    artifacts = {}
    for key in ("original_graph", "calibration_graph", "runtime_validation"):
        if key not in catalog:
            raise ValueError(f"projection evidence_catalog is missing {key}")
        _, artifacts[key] = load_hashed_json(
            root, catalog[key], f"projection.evidence_catalog.{key}"
        )
    replayed = project_graphs(
        artifacts["original_graph"],
        artifacts["calibration_graph"],
        runtime_validation=artifacts["runtime_validation"],
        evidence_catalog=catalog,
        original_collection_fingerprint=projection.get(
            "original_collection_fingerprint"
        ),
        calibration_collection_fingerprint=projection.get(
            "calibration_collection_fingerprint"
        ),
    )
    if replayed != projection:
        raise ValueError(
            "calibration projection cannot be reproduced from sealed graph evidence"
        )


def _validate_current_source(source_root, source_manifest):
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise ValueError("current source root must be an existing directory")
    for name, expected_sha in source_manifest["file_sha256"].items():
        source_file = resolve_evidence_path(
            source_root, name, f"current source file {name}"
        )
        if (
            not source_file.is_file()
            or source_file.is_symlink()
            or file_sha256(source_file) != expected_sha
        ):
            raise ValueError(
                f"current source differs from archived source snapshot: {name}"
            )


def _snapshot_root(root, reference, label):
    reference = require_object(reference, label)
    relative = safe_relative_path(
        reference.get("relative_path"), f"{label}.relative_path"
    )
    return resolve_evidence_path(root, relative.as_posix(), f"{label}.relative_path")


def load_source_scope_map_v2(
    path,
    *,
    expected_source_revision,
    expected_candidate_manifest_sha256=None,
    expected_baseline_manifest_sha256=None,
    expected_source_root=None,
):
    path = Path(path)
    value = load_json(path, "source scope map")
    if not isinstance(value, dict) or value.get("protocol") != "source_scope_map_v2":
        return {
            "protocol": "legacy_source_scope_map",
            "status": "diagnostic_only",
            "lookup": {},
            "blockers": ["legacy_task_range_source_map_not_exact"],
        }
    schema_version = value.get("schema_version")
    if schema_version not in {"2.0", "2.1"}:
        raise ValueError("source_scope_map_v2 schema_version must be 2.0 or 2.1")
    validate_exact_keys(
        value,
        {"schema_version", "protocol", "provenance", "source_ranges"},
        "source_scope_map_v2",
        required=("schema_version", "protocol", "provenance", "source_ranges"),
    )
    provenance_allowed = {
        "stable_marker_revision",
        "source_revision",
        "source_revision_role",
        "source_snapshot_root",
        "calibration_source_snapshot_root",
        "source_scope_map_content_fingerprint",
        *ARTIFACT_KEYS,
        *AUTO_SOURCE_ARTIFACT_KEYS,
    }
    provenance_required = {
        "source_snapshot_root",
        "source_scope_map_content_fingerprint",
        *ARTIFACT_KEYS,
    }
    if schema_version == "2.0":
        provenance_required.add("stable_marker_revision")
    else:
        provenance_required.update({"source_revision", "source_revision_role"})
    provenance = validate_exact_keys(
        value.get("provenance"),
        provenance_allowed,
        "provenance",
        required=provenance_required,
    )
    if provenance.get("source_scope_map_content_fingerprint") != _content_fingerprint(
        value
    ):
        raise ValueError("source_scope_map_v2 content fingerprint mismatch")
    source_revision_role = (
        "stable_marker"
        if schema_version == "2.0"
        else require_text(
            provenance.get("source_revision_role"),
            "provenance.source_revision_role",
        )
    )
    if source_revision_role not in {"stable_marker", "stable_source"}:
        raise ValueError("source_scope_map_v2 source revision role is unsupported")
    source_revision = require_text(
        provenance.get("source_revision", provenance.get("stable_marker_revision")),
        "provenance.source_revision",
    )
    if source_revision != require_text(
        expected_source_revision, "expected source revision"
    ):
        raise ValueError(
            "source_scope_map_v2 source revision differs from --source-revision"
        )
    if source_revision_role == "stable_marker":
        stable_marker_revision = require_text(
            provenance.get("stable_marker_revision"),
            "provenance.stable_marker_revision",
        )
        if stable_marker_revision != source_revision:
            raise ValueError("stable marker revision differs from source revision")
        active_artifact_keys = ARTIFACT_KEYS
    else:
        if schema_version != "2.1":
            raise ValueError("stable_source source maps require schema_version 2.1")
        if provenance.get("stable_marker_revision") is not None:
            raise ValueError(
                "stable_source source maps must not declare stable_marker_revision"
            )
        missing_auto = sorted(
            {
                "calibration_source_snapshot_root",
                *AUTO_SOURCE_ARTIFACT_KEYS,
            }
            - set(provenance)
        )
        if missing_auto:
            raise ValueError(
                "stable_source source map is missing marker-only evidence: "
                + ", ".join(missing_auto)
            )
        active_artifact_keys = ARTIFACT_KEYS + AUTO_SOURCE_ARTIFACT_KEYS
    root = path.parent.resolve()
    loaded = {
        key: _load_reference(root, provenance, key) for key in active_artifact_keys
    }
    references = {key: provenance[key] for key in active_artifact_keys}
    snapshot_reference = require_object(
        provenance.get("source_snapshot_root"), "provenance.source_snapshot_root"
    )
    references["source_snapshot_manifest"] = {
        "relative_path": (
            Path(snapshot_reference["relative_path"]) / "source-snapshot-manifest.json"
        ).as_posix(),
        "sha256": snapshot_reference.get("manifest_sha256"),
    }
    source_manifest = loaded["source_manifest"][1]
    source_manifest_valid = (
        source_manifest.get("revision_role") == source_revision_role
        and source_manifest.get("source_revision") == source_revision
        and source_manifest.get("manifest_fingerprint")
        == _fingerprint_without(source_manifest, "manifest_fingerprint")
    )
    if source_revision_role == "stable_marker":
        source_manifest_valid = source_manifest_valid and (
            source_manifest.get("stable_marker_revision") == source_revision
        )
    else:
        source_manifest_valid = source_manifest_valid and (
            source_manifest.get("stable_marker_revision") is None
        )
    if not source_manifest_valid:
        raise ValueError(
            f"source manifest is not bound to the {source_revision_role} revision"
        )
    _validate_snapshot(root, provenance, source_manifest)
    if source_revision_role == "stable_source":
        calibration_manifest = loaded["calibration_source_manifest"][1]
        calibration_snapshot_reference = require_object(
            provenance.get("calibration_source_snapshot_root"),
            "provenance.calibration_source_snapshot_root",
        )
        references["calibration_source_snapshot_manifest"] = {
            "relative_path": (
                Path(calibration_snapshot_reference["relative_path"])
                / "source-snapshot-manifest.json"
            ).as_posix(),
            "sha256": calibration_snapshot_reference.get("manifest_sha256"),
        }
        _validate_snapshot(
            root,
            {"source_snapshot_root": calibration_snapshot_reference},
            calibration_manifest,
        )
        validate_marker_only_calibration(
            loaded["marker_only_calibration"][1],
            stable_manifest=source_manifest,
            calibration_manifest=calibration_manifest,
            stable_snapshot_root=_snapshot_root(
                root,
                provenance["source_snapshot_root"],
                "provenance.source_snapshot_root",
            ),
            calibration_snapshot_root=_snapshot_root(
                root,
                calibration_snapshot_reference,
                "provenance.calibration_source_snapshot_root",
            ),
        )
    if expected_source_root is not None:
        _validate_current_source(expected_source_root, source_manifest)
    block_inventory = loaded["block_instance_inventory"][1]
    projection = loaded["calibration_projection"][1]
    sk_map = loaded["sk_source_map"][1]
    fused_inventory = loaded["original_fused_inventory"][1]
    assignments = loaded["unit_assignments"][1]
    _validate_dag(
        root,
        loaded["provenance_dag"][1],
        {key: value for key, value in references.items() if key != "provenance_dag"},
        projection,
        assignments,
        fused_inventory,
    )
    _replay_calibration_projection(root, projection)
    _validate_normalized_record_evidence(
        root,
        assignments,
        require_list(assignments.get("assignments"), "assignments"),
        label="assignments",
        identity_fields=("calibration_node_key", "block_instance_id", "unit_id"),
    )
    _validate_normalized_record_evidence(
        root,
        fused_inventory,
        require_list(fused_inventory.get("sk_groups"), "sk_groups"),
        label="sk_groups",
        identity_fields=(
            "device_id",
            "model_id",
            "block_instance_id",
            "candidate_source_scope",
            "sk_occurrence_fingerprint",
            "original_child_node_keys",
        ),
    )
    for key in (
        "original_candidate_collection_manifest",
        "calibration_collection_manifest",
        "baseline_collection_manifest",
    ):
        manifest = loaded[key][1]
        expected_fingerprint = require_sha256(
            provenance[key].get("fingerprint"), f"provenance.{key}.fingerprint"
        )
        if manifest.get("manifest_fingerprint") != expected_fingerprint:
            raise ValueError(f"{key} fingerprint mismatch")
    if (
        projection.get("original_collection_fingerprint")
        != provenance["original_candidate_collection_manifest"]["fingerprint"]
    ):
        raise ValueError(
            "calibration projection original collection fingerprint mismatch"
        )
    if (
        projection.get("calibration_collection_fingerprint")
        != provenance["calibration_collection_manifest"]["fingerprint"]
    ):
        raise ValueError(
            "calibration projection calibration collection fingerprint mismatch"
        )
    rebuilt = build_sk_source_map(
        projection,
        source_manifest,
        block_inventory,
        fused_inventory,
        assignments,
        calibration_source_manifest=(
            loaded["calibration_source_manifest"][1]
            if source_revision_role == "stable_source"
            else None
        ),
    )
    if rebuilt != sk_map:
        raise ValueError("sk-source-map cannot be reproduced from occurrence evidence")
    if expected_candidate_manifest_sha256 is not None:
        expected = require_sha256(
            expected_candidate_manifest_sha256,
            "expected candidate collection manifest SHA-256",
        )
        if references["original_candidate_collection_manifest"]["sha256"] != expected:
            raise ValueError(
                "source map is not bound to the current candidate collection manifest"
            )
    if expected_baseline_manifest_sha256 is not None:
        expected = require_sha256(
            expected_baseline_manifest_sha256,
            "expected baseline collection manifest SHA-256",
        )
        if references["baseline_collection_manifest"]["sha256"] != expected:
            raise ValueError(
                "source map is not bound to the current baseline collection manifest"
            )
    mapping_by_fingerprint = {
        item["sk_occurrence_fingerprint"]: item for item in sk_map["mappings"]
    }
    lookup = {}
    for index, raw in enumerate(
        require_list(value.get("source_ranges"), "source_ranges")
    ):
        item = validate_exact_keys(
            raw,
            {
                "device_id",
                "model_id",
                "block_instance_id",
                "structural_family",
                "source_scope",
                "candidate_source_scope",
                "unit_id",
                "sk_occurrence_fingerprint",
                "baseline_projection_fingerprint",
                "relation",
                "boundary",
                "ordered_child_op_sequence",
            },
            f"source_ranges[{index}]",
            required=(
                "device_id",
                "model_id",
                "block_instance_id",
                "structural_family",
                "source_scope",
                "candidate_source_scope",
                "unit_id",
                "sk_occurrence_fingerprint",
                "baseline_projection_fingerprint",
                "relation",
                "boundary",
                "ordered_child_op_sequence",
            ),
        )
        fingerprint = require_sha256(
            item.get("sk_occurrence_fingerprint"),
            f"source_ranges[{index}].sk_occurrence_fingerprint",
        )
        mapping = mapping_by_fingerprint.get(fingerprint)
        if mapping is None or mapping.get("relation") != "exact_cover":
            raise ValueError(f"source range {index} is not backed by exact_cover")
        if mapping.get("confidence") not in {
            "source_unit_exact",
            "source_unit_template_exact",
        }:
            raise ValueError(f"source range {index} lacks source-unit exact confidence")
        if (
            len(mapping.get("covered_unit_ids", [])) != 1
            or item.get("unit_id") != mapping["covered_unit_ids"][0]
        ):
            raise ValueError(f"source range {index} unit identity mismatch")
        for field in (
            "device_id",
            "model_id",
            "block_instance_id",
            "structural_family",
            "source_scope",
            "candidate_source_scope",
            "relation",
        ):
            if str(item.get(field)) != str(mapping.get(field)):
                raise ValueError(f"source range {index} {field} mismatch")
        expected_ops = [
            occurrence[2] for occurrence in mapping["original_child_occurrences"]
        ]
        if item.get("ordered_child_op_sequence") != expected_ops:
            raise ValueError(f"source range {index} child op sequence mismatch")
        if item.get("baseline_projection_fingerprint") != mapping.get(
            "baseline_projection_fingerprint"
        ):
            raise ValueError(
                f"source range {index} baseline projection fingerprint mismatch"
            )
        boundary = validate_exact_keys(
            item.get("boundary"),
            {"start_op", "end_op", "source_file", "start_offset", "end_offset"},
            f"source_ranges[{index}].boundary",
            required=(
                "start_op",
                "end_op",
                "source_file",
                "start_offset",
                "end_offset",
            ),
        )
        interval = mapping["source_intervals"][0]
        if any(
            boundary.get(field) != interval.get(field)
            for field in ("source_file", "start_offset", "end_offset")
        ):
            raise ValueError(
                f"source range {index} boundary differs from source manifest"
            )
        if (
            boundary.get("start_op") != expected_ops[0]
            or boundary.get("end_op") != expected_ops[-1]
        ):
            raise ValueError(f"source range {index} fusion boundary mismatch")
        if len(set(mapping["consensus"]["validated_step_ids"])) < 3:
            raise ValueError(f"source range {index} lacks three validated steps")
        if fingerprint in lookup:
            raise ValueError(f"duplicate source range fingerprint {fingerprint}")
        lookup[fingerprint] = copy.deepcopy(item)
    return {
        "protocol": "source_scope_map_v2",
        "status": "exact",
        "lookup": lookup,
        "blockers": [],
        "source_revision": source_revision,
        "source_revision_role": source_revision_role,
        **(
            {"stable_marker_revision": source_revision}
            if source_revision_role == "stable_marker"
            else {}
        ),
    }


def build_source_scope_map_v2(
    *,
    artifact_root,
    stable_marker_revision=None,
    stable_source_revision=None,
    provenance_dag,
    source_manifest,
    source_snapshot_root,
    sk_source_map,
    block_instance_inventory,
    calibration_projection,
    original_fused_inventory,
    unit_assignments,
    original_candidate_collection_manifest,
    calibration_collection_manifest,
    baseline_collection_manifest,
    calibration_source_manifest=None,
    marker_only_calibration=None,
    calibration_source_snapshot_root=None,
):
    root = Path(artifact_root).resolve()
    if (stable_marker_revision is None) == (stable_source_revision is None):
        raise ValueError(
            "declare exactly one of stable_marker_revision or stable_source_revision"
        )
    source_revision_role = (
        "stable_source" if stable_source_revision is not None else "stable_marker"
    )
    source_revision = require_text(
        stable_source_revision
        if stable_source_revision is not None
        else stable_marker_revision,
        f"{source_revision_role}_revision",
    )
    paths = {
        "provenance_dag": Path(provenance_dag),
        "source_manifest": Path(source_manifest),
        "sk_source_map": Path(sk_source_map),
        "block_instance_inventory": Path(block_instance_inventory),
        "calibration_projection": Path(calibration_projection),
        "original_fused_inventory": Path(original_fused_inventory),
        "unit_assignments": Path(unit_assignments),
        "original_candidate_collection_manifest": Path(
            original_candidate_collection_manifest
        ),
        "calibration_collection_manifest": Path(calibration_collection_manifest),
        "baseline_collection_manifest": Path(baseline_collection_manifest),
    }
    if source_revision_role == "stable_source":
        missing = [
            name
            for name, value in (
                ("calibration_source_manifest", calibration_source_manifest),
                ("marker_only_calibration", marker_only_calibration),
                (
                    "calibration_source_snapshot_root",
                    calibration_source_snapshot_root,
                ),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "stable_source source map is missing marker-only evidence: "
                + ", ".join(missing)
            )
        paths.update(
            {
                "calibration_source_manifest": Path(calibration_source_manifest),
                "marker_only_calibration": Path(marker_only_calibration),
            }
        )
    provenance = {
        **{
            key: artifact_reference(path, relative_to=root)
            for key, path in paths.items()
        },
    }
    if source_revision_role == "stable_source":
        provenance.update(
            {
                "source_revision": source_revision,
                "source_revision_role": source_revision_role,
            }
        )
    else:
        provenance["stable_marker_revision"] = source_revision
    for key in (
        "original_candidate_collection_manifest",
        "calibration_collection_manifest",
        "baseline_collection_manifest",
    ):
        manifest = load_json(paths[key], key)
        provenance[key]["fingerprint"] = require_sha256(
            manifest.get("manifest_fingerprint"), f"{key}.manifest_fingerprint"
        )
    snapshot_root = Path(source_snapshot_root).resolve()
    provenance["source_snapshot_root"] = {
        "relative_path": snapshot_root.relative_to(root).as_posix(),
        "manifest_sha256": file_sha256(snapshot_root / "source-snapshot-manifest.json"),
    }
    if source_revision_role == "stable_source":
        calibration_snapshot_root = Path(calibration_source_snapshot_root).resolve()
        provenance["calibration_source_snapshot_root"] = {
            "relative_path": calibration_snapshot_root.relative_to(root).as_posix(),
            "manifest_sha256": file_sha256(
                calibration_snapshot_root / "source-snapshot-manifest.json"
            ),
        }
    sk_map = load_json(paths["sk_source_map"])
    source_manifest_value = load_json(paths["source_manifest"])
    units = {item["unit_id"]: item for item in source_manifest_value["units"]}
    source_ranges = []
    for mapping in sk_map["mappings"]:
        if mapping["relation"] != "exact_cover" or mapping["confidence"] not in {
            "source_unit_exact",
            "source_unit_template_exact",
        }:
            continue
        unit_id = mapping["covered_unit_ids"][0]
        interval = units[unit_id]
        source_ranges.append(
            {
                "device_id": mapping["device_id"],
                "model_id": mapping["model_id"],
                "block_instance_id": mapping["block_instance_id"],
                "structural_family": mapping["structural_family"],
                "source_scope": mapping["source_scope"],
                "candidate_source_scope": mapping["candidate_source_scope"],
                "unit_id": unit_id,
                "sk_occurrence_fingerprint": mapping["sk_occurrence_fingerprint"],
                "baseline_projection_fingerprint": mapping[
                    "baseline_projection_fingerprint"
                ],
                "relation": "exact_cover",
                "boundary": {
                    "start_op": mapping["original_child_occurrences"][0][2],
                    "end_op": mapping["original_child_occurrences"][-1][2],
                    "source_file": interval["source_file"],
                    "start_offset": interval["start_offset"],
                    "end_offset": interval["end_offset"],
                },
                "ordered_child_op_sequence": [
                    item[2] for item in mapping["original_child_occurrences"]
                ],
            }
        )
    value = {
        "schema_version": ("2.1" if source_revision_role == "stable_source" else "2.0"),
        "protocol": "source_scope_map_v2",
        "provenance": provenance,
        "source_ranges": sorted(
            source_ranges, key=lambda item: item["sk_occurrence_fingerprint"]
        ),
    }
    provenance["source_scope_map_content_fingerprint"] = _content_fingerprint(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--artifact-root", required=True)
    revision = build.add_mutually_exclusive_group(required=True)
    revision.add_argument("--stable-marker-revision")
    revision.add_argument("--stable-source-revision")
    for option in ARTIFACT_KEYS:
        build.add_argument(f"--{option.replace('_', '-')}", required=True)
    for option in AUTO_SOURCE_ARTIFACT_KEYS:
        build.add_argument(f"--{option.replace('_', '-')}")
    build.add_argument("--source-snapshot-root", required=True)
    build.add_argument("--calibration-source-snapshot-root")
    build.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate", allow_abbrev=False)
    validate.add_argument("--map", required=True)
    validate.add_argument("--source-revision", required=True)
    validate.add_argument("--candidate-manifest-sha256")
    validate.add_argument("--baseline-manifest-sha256")
    validate.add_argument("--source-root")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            values = vars(args)
            result = build_source_scope_map_v2(
                artifact_root=values["artifact_root"],
                stable_marker_revision=values["stable_marker_revision"],
                stable_source_revision=values["stable_source_revision"],
                source_snapshot_root=values["source_snapshot_root"],
                calibration_source_snapshot_root=values[
                    "calibration_source_snapshot_root"
                ],
                **{
                    key: values[key]
                    for key in ARTIFACT_KEYS + AUTO_SOURCE_ARTIFACT_KEYS
                },
            )
            atomic_write_json(args.output, result)
        else:
            result = load_source_scope_map_v2(
                args.map,
                expected_source_revision=args.source_revision,
                expected_candidate_manifest_sha256=args.candidate_manifest_sha256,
                expected_baseline_manifest_sha256=args.baseline_manifest_sha256,
                expected_source_root=args.source_root,
            )
            print(result["status"])
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
