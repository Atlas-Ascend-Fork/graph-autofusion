#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build an auditable many-to-many SK-to-source-unit map."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict

from project_calibration_graph import PROTOCOL_VERSIONS
from source_calibration_common import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    require_integer,
    require_list,
    require_object,
    require_scalar_id,
    require_sha256,
    require_text,
)


def _fingerprint_without(value, field):
    detached = copy.deepcopy(value)
    detached.pop(field, None)
    return canonical_sha256(detached)


def _validated_steps_for_targets(projection, calibration_node_keys):
    targets = set(calibration_node_keys)
    valid_steps = set(projection.get("validated_step_ids", []))
    for raw in projection.get("runtime_validation", []):
        item = require_object(raw, "projection runtime validation")
        if item.get("passed") is True:
            valid_steps.add(require_integer(item.get("step_id"), "runtime step_id"))
            continue
        if item.get("assignment_partition_complete") is not True:
            continue
        exact = set(
            require_list(
                item.get("exact_assigned_calibration_node_keys"),
                "runtime exact assigned calibration nodes",
            )
        )
        skipped = set(
            require_list(
                item.get("skipped_calibration_node_keys"),
                "runtime skipped calibration nodes",
            )
        )
        conflicting = set(
            require_list(
                item.get("conflicting_calibration_node_keys"),
                "runtime conflicting calibration nodes",
            )
        )
        if targets <= exact and not targets & (skipped | conflicting):
            valid_steps.add(require_integer(item.get("step_id"), "runtime step_id"))
    return sorted(valid_steps)


def build_sk_source_map(
    projection,
    source_manifest,
    block_inventory,
    fused_inventory,
    unit_assignments,
    calibration_source_manifest=None,
):
    projection = require_object(projection, "calibration projection")
    if projection.get("protocol") != "source_calibration_projection_v2":
        raise ValueError("calibration projection protocol is unsupported")
    if projection.get("protocol_versions") != PROTOCOL_VERSIONS:
        raise ValueError("calibration projection algorithm versions are unsupported")
    if projection.get("alternative_stream_assignment_count") != 0:
        raise ValueError("calibration projection stream assignment is ambiguous")
    if projection.get("alternative_node_correspondence_count") != 0:
        raise ValueError("calibration projection is ambiguous")
    if projection.get("nontrivial_automorphism_count") != 0:
        raise ValueError("calibration projection has a nontrivial automorphism")
    if (
        projection.get("business_kernel_sequence_equal") is not True
        or projection.get("typed_business_edge_set_equal") is not True
    ):
        raise ValueError("calibration projection does not preserve the business graph")
    expected_projection = projection.get("projection_fingerprint")
    if expected_projection != _fingerprint_without(
        projection, "projection_fingerprint"
    ):
        raise ValueError("calibration projection fingerprint mismatch")

    source_manifest = require_object(source_manifest, "source manifest")
    expected_source = source_manifest.get("manifest_fingerprint")
    if expected_source != _fingerprint_without(source_manifest, "manifest_fingerprint"):
        raise ValueError("source manifest fingerprint mismatch")
    units = {}
    for index, raw in enumerate(
        require_list(source_manifest.get("units"), "source units")
    ):
        item = require_object(raw, f"source units[{index}]")
        unit_id = require_text(item.get("unit_id"), f"source unit {index}.unit_id")
        if unit_id in units:
            raise ValueError(f"duplicate source unit {unit_id}")
        units[unit_id] = item
    calibration_source_fingerprint = None
    if calibration_source_manifest is not None:
        calibration_source_manifest = require_object(
            calibration_source_manifest, "calibration source manifest"
        )
        calibration_source_fingerprint = calibration_source_manifest.get(
            "manifest_fingerprint"
        )
        if calibration_source_fingerprint != _fingerprint_without(
            calibration_source_manifest, "manifest_fingerprint"
        ):
            raise ValueError("calibration source manifest fingerprint mismatch")
        if source_manifest.get("revision_role") != "stable_source":
            raise ValueError(
                "calibration source manifest is only valid with stable_source"
            )
        if calibration_source_manifest.get("revision_role") != "calibration":
            raise ValueError("calibration source manifest role is invalid")
        calibration_units = {
            require_text(item.get("unit_id"), "calibration source unit_id"): item
            for item in require_list(
                calibration_source_manifest.get("units"),
                "calibration source units",
            )
        }
        if set(calibration_units) != set(units):
            raise ValueError("stable and calibration source unit sets differ")
        for unit_id, stable_unit in units.items():
            calibration_unit = calibration_units[unit_id]
            for field in (
                "block_template_id",
                "source_symbol",
                "source_file",
                "syntax_tree_hash",
            ):
                if stable_unit.get(field) != calibration_unit.get(field):
                    raise ValueError(
                        f"stable and calibration source unit {unit_id} {field} differs"
                    )
    template_ids = set()
    for index, raw in enumerate(
        require_list(source_manifest.get("block_templates"), "block templates")
    ):
        item = require_object(raw, f"block templates[{index}]")
        template_id = require_text(
            item.get("block_template_id"), f"block templates[{index}].block_template_id"
        )
        if template_id in template_ids:
            raise ValueError(f"duplicate block template {template_id}")
        template_ids.add(template_id)

    block_inventory = require_object(block_inventory, "block inventory")
    expected_inventory_source = calibration_source_fingerprint or expected_source
    if block_inventory.get("source_manifest_fingerprint") != expected_inventory_source:
        raise ValueError("block inventory source manifest fingerprint mismatch")
    adapter = require_object(
        source_manifest.get("model_adapter"), "source model adapter"
    )
    if block_inventory.get("model_adapter_fingerprint") != adapter.get("fingerprint"):
        raise ValueError("block inventory model adapter fingerprint mismatch")
    instances = {}
    for index, raw in enumerate(
        require_list(block_inventory.get("instances"), "block instances")
    ):
        item = require_object(raw, f"block instances[{index}]")
        block_id = require_text(
            item.get("block_instance_id"), f"block instances[{index}].block_instance_id"
        )
        if block_id in instances:
            raise ValueError(f"duplicate block instance {block_id}")
        template_id = require_text(
            item.get("block_template_id"),
            f"block instances[{index}].block_template_id",
        )
        if template_id not in template_ids:
            raise ValueError(
                f"block instance {block_id} references an unknown template"
            )
        binding = require_object(
            item.get("runtime_scope_binding"),
            f"block instances[{index}].runtime_scope_binding",
        )
        if binding.get("method") != "exact_scope_name":
            raise ValueError(
                f"block instance {block_id} uses an unsupported scope binding"
            )
        require_text(binding.get("value"), f"block instance {block_id} scope binding")
        require_text(
            item.get("control_flow_path"),
            f"block instance {block_id} control_flow_path",
        )
        instances[block_id] = item

    correspondence = {}
    coordinates = {}
    for index, raw in enumerate(
        require_list(projection.get("occurrences"), "projection occurrences")
    ):
        item = require_object(raw, f"projection occurrences[{index}]")
        original = require_text(
            item.get("original_node_key"),
            f"projection occurrences[{index}].original_node_key",
        )
        calibration = require_text(
            item.get("calibration_node_key"),
            f"projection occurrences[{index}].calibration_node_key",
        )
        if original in correspondence or calibration in coordinates:
            raise ValueError("projection correspondence is not bijective")
        correspondence[original] = calibration
        coordinates[calibration] = copy.deepcopy(item.get("original"))

    unit_assignments = require_object(unit_assignments, "unit assignments")
    if unit_assignments.get("protocol") != "calibration_unit_assignments_v1":
        raise ValueError("unit assignments protocol is unsupported")
    assignment_by_node = {}
    unit_nodes = defaultdict(set)
    block_nodes = defaultdict(set)
    control_flow = {}
    for index, raw in enumerate(
        require_list(unit_assignments.get("assignments"), "assignments")
    ):
        item = require_object(raw, f"assignments[{index}]")
        evidence_records = require_list(
            item.get("evidence_records"), f"assignments[{index}].evidence_records"
        )
        if not evidence_records:
            raise ValueError(f"assignments[{index}] requires normalized log evidence")
        for evidence_index, evidence in enumerate(evidence_records):
            evidence = require_object(
                evidence, f"assignments[{index}].evidence_records[{evidence_index}]"
            )
            require_text(evidence.get("artifact_id"), "assignment evidence artifact_id")
            require_text(evidence.get("record_id"), "assignment evidence record_id")
            require_sha256(
                evidence.get("record_fingerprint"),
                "assignment evidence record_fingerprint",
            )
        node_key = require_text(
            item.get("calibration_node_key"),
            f"assignments[{index}].calibration_node_key",
        )
        block_id = require_text(
            item.get("block_instance_id"), f"assignments[{index}].block_instance_id"
        )
        unit_id = require_text(item.get("unit_id"), f"assignments[{index}].unit_id")
        if node_key in assignment_by_node or node_key not in coordinates:
            raise ValueError(
                f"assignment node {node_key} is duplicate or not projected"
            )
        if block_id not in instances or unit_id not in units:
            raise ValueError(
                f"assignment {node_key} references an unknown block or unit"
            )
        if units[unit_id].get("block_template_id") != instances[block_id].get(
            "block_template_id"
        ):
            raise ValueError(f"assignment {node_key} crosses block templates")
        assignment_by_node[node_key] = (block_id, unit_id)
        unit_nodes[(block_id, unit_id)].add(node_key)
        block_nodes[block_id].add(node_key)
        path = require_text(
            item.get("control_flow_path"), f"assignments[{index}].control_flow_path"
        )
        if path != instances[block_id].get("control_flow_path"):
            raise ValueError(f"assignment {node_key} control-flow path mismatch")
        previous = control_flow.setdefault(block_id, path)
        if previous != path:
            raise ValueError(f"block {block_id} has conflicting control-flow paths")
    unassigned_calibration_nodes = set(coordinates) - set(assignment_by_node)

    fused_inventory = require_object(fused_inventory, "original fused inventory")
    if fused_inventory.get("protocol") != "original_fused_inventory_v1":
        raise ValueError("original fused inventory protocol is unsupported")
    sk_groups = require_list(fused_inventory.get("sk_groups"), "sk_groups")
    normalized_groups = []
    for index, raw in enumerate(sk_groups):
        group = require_object(raw, f"sk_groups[{index}]")
        evidence_records = require_list(
            group.get("evidence_records"), f"sk_groups[{index}].evidence_records"
        )
        if not evidence_records:
            raise ValueError(f"sk_groups[{index}] requires normalized log evidence")
        for evidence_index, evidence in enumerate(evidence_records):
            evidence = require_object(
                evidence, f"sk_groups[{index}].evidence_records[{evidence_index}]"
            )
            require_text(evidence.get("artifact_id"), "fused evidence artifact_id")
            require_text(evidence.get("record_id"), "fused evidence record_id")
            require_sha256(
                evidence.get("record_fingerprint"), "fused evidence record_fingerprint"
            )
        block_id = require_text(
            group.get("block_instance_id"), f"sk_groups[{index}].block_instance_id"
        )
        children = [
            require_text(key, f"sk_groups[{index}].original_child_node_keys")
            for key in require_list(
                group.get("original_child_node_keys"),
                f"sk_groups[{index}].original_child_node_keys",
            )
        ]
        if (
            not children
            or len(children) != len(set(children))
            or any(key not in correspondence for key in children)
        ):
            raise ValueError(f"sk_groups[{index}] child occurrence set is invalid")
        if block_id not in instances:
            raise ValueError(f"sk_groups[{index}] references an unknown block instance")
        binding = instances[block_id]["runtime_scope_binding"]
        if group.get("source_scope") != binding.get("value"):
            raise ValueError(f"sk_groups[{index}] source scope binding mismatch")
        calibration_children = {correspondence[key] for key in children}
        missing_assignment_nodes = sorted(
            calibration_children - set(assignment_by_node)
        )
        cross_block_assignment_nodes = sorted(
            key
            for key in calibration_children - set(missing_assignment_nodes)
            if assignment_by_node[key][0] != block_id
        )
        validated_step_ids = _validated_steps_for_targets(
            projection, calibration_children
        )
        target_assignment_complete = (
            not missing_assignment_nodes
            and not cross_block_assignment_nodes
            and len(validated_step_ids) >= 3
        )
        expected_ops = [coordinates[correspondence[key]][2] for key in children]
        if group.get("ordered_child_ops") != expected_ops:
            raise ValueError(f"sk_groups[{index}] child operator sequence mismatch")
        normalized = {
            "raw": group,
            "block_id": block_id,
            "children": children,
            "calibration_children": calibration_children,
            "missing_assignment_nodes": missing_assignment_nodes,
            "cross_block_assignment_nodes": cross_block_assignment_nodes,
            "validated_step_ids": validated_step_ids,
            "target_assignment_complete": target_assignment_complete,
            "ordinal": require_integer(
                group.get("block_local_sk_ordinal"),
                f"sk_groups[{index}].block_local_sk_ordinal",
            ),
        }
        normalized_groups.append(normalized)
    structural_family = {}
    block_local_coordinates = {}
    for block_id, nodes in block_nodes.items():
        streams = defaultdict(list)
        for node_key in nodes:
            role, ordinal, op, core = coordinates[node_key]
            streams[role].append((ordinal, node_key, op, core))
        stream_signatures = {
            role: tuple((op, core) for _, _, op, core in sorted(sequence))
            for role, sequence in streams.items()
        }
        signature_counts = defaultdict(int)
        for signature in stream_signatures.values():
            signature_counts[signature] += 1
        stream_ambiguous = any(count > 1 for count in signature_counts.values())
        ordered_streams = sorted(
            streams, key=lambda role: (stream_signatures[role], role)
        )
        for local_role, role in enumerate(ordered_streams):
            for local_ordinal, (_, node_key, op, core) in enumerate(
                sorted(streams[role])
            ):
                block_local_coordinates[node_key] = [
                    f"block-stream-role-{local_role}",
                    local_ordinal,
                    op,
                    core,
                ]
        local_groups = [
            item for item in normalized_groups if item["block_id"] == block_id
        ]
        if len({item["ordinal"] for item in local_groups}) != len(local_groups):
            raise ValueError(f"block {block_id} has duplicate block-local SK ordinals")
        seed = {
            "business_occurrences": sorted(
                block_local_coordinates[key] for key in nodes
            ),
            "sk_groups": sorted(
                [
                    {
                        "block_local_sk_ordinal": group["ordinal"],
                        "child_occurrences": sorted(
                            block_local_coordinates[key]
                            for key in group["calibration_children"]
                        ),
                    }
                    for group in local_groups
                    if group["calibration_children"] <= nodes
                ],
                key=lambda item: item["block_local_sk_ordinal"],
            ),
        }
        family_hash = canonical_sha256(seed)
        structural_family[block_id] = (
            f"family:{family_hash}"
            if not stream_ambiguous
            else f"family-instance-only:{family_hash}:{canonical_sha256(block_id)}"
        )

    provisional = []
    consensus_observations = defaultdict(list)
    for item in normalized_groups:
        group = item["raw"]
        block_id = item["block_id"]
        child_set = item["calibration_children"]
        intersecting = (
            sorted(
                unit_id
                for (candidate_block, unit_id), nodes in unit_nodes.items()
                if candidate_block == block_id and nodes & child_set
            )
            if item["target_assignment_complete"]
            else []
        )
        exact_units = (
            sorted(
                unit_id
                for unit_id in intersecting
                if unit_nodes[(block_id, unit_id)] == child_set
            )
            if item["target_assignment_complete"]
            else []
        )
        if not item["target_assignment_complete"]:
            relation = "unmapped"
            covered = []
            confidence = "diagnostic_only"
        elif len(exact_units) == 1:
            relation = "exact_cover"
            covered = exact_units
            confidence = "source_unit_exact"
        elif intersecting:
            relation = "partial_intersection"
            covered = intersecting
            confidence = "diagnostic_only"
        else:
            relation = "unmapped"
            covered = []
            confidence = "diagnostic_only"
        family = structural_family.get(
            block_id,
            f"family-unmapped:{canonical_sha256(block_id)}",
        )
        consensus_key = (
            family,
            item["ordinal"],
            canonical_sha256(group.get("ordered_child_ops", [])),
        )
        mapping = {
            "device_id": require_integer(group.get("device_id"), "sk group device_id"),
            "model_id": require_scalar_id(group.get("model_id"), "sk group model_id"),
            "block_instance_id": block_id,
            "source_scope": require_text(
                group.get("source_scope"), "sk group source_scope"
            ),
            "candidate_source_scope": require_text(
                group.get("candidate_source_scope"), "sk group candidate_source_scope"
            ),
            "sk_occurrence_fingerprint": require_sha256(
                group.get("sk_occurrence_fingerprint"),
                "sk group occurrence fingerprint",
            ),
            "baseline_projection_fingerprint": require_sha256(
                group.get("baseline_projection_fingerprint"),
                "baseline projection fingerprint",
            ),
            "structural_family": family,
            "block_local_sk_ordinal": item["ordinal"],
            "original_child_node_keys": item["children"],
            "original_child_occurrences": [
                coordinates[correspondence[key]] for key in item["children"]
            ],
            "covered_unit_ids": covered,
            "relation": relation,
            "mapping_group_id": None,
            "confidence": confidence,
            "processing_status": (
                "processed" if item["target_assignment_complete"] else "skipped"
            ),
            "mapping_blockers": [
                blocker
                for blocker, active in (
                    (
                        "target_child_assignment_missing",
                        bool(item["missing_assignment_nodes"]),
                    ),
                    (
                        "target_child_block_mismatch",
                        bool(item["cross_block_assignment_nodes"]),
                    ),
                    (
                        "target_child_three_step_validation_missing",
                        len(item["validated_step_ids"]) < 3,
                    ),
                )
                if active
            ],
            "missing_calibration_node_keys": item["missing_assignment_nodes"],
            "cross_block_calibration_node_keys": item["cross_block_assignment_nodes"],
            "consensus": {
                "supporting_block_instances": [block_id],
                "counterexample_block_instances": [],
                "validated_step_ids": item["validated_step_ids"],
            },
            "source_intervals": [
                {
                    field: units[unit_id][field]
                    for field in (
                        "source_file",
                        "start_offset",
                        "end_offset",
                        "syntax_tree_hash",
                    )
                }
                for unit_id in covered
            ],
        }
        mapping["mapped_calibration_node_keys"] = sorted(child_set)
        provisional.append((consensus_key, mapping))
        consensus_observations[consensus_key].append(mapping)
    mappings = []
    for key, mapping in provisional:
        observations = consensus_observations[key]
        exact_unit_sets = {
            tuple(item["covered_unit_ids"])
            for item in observations
            if item["relation"] == "exact_cover"
        }
        consensus_exact = len(exact_unit_sets) == 1 and all(
            item["relation"] == "exact_cover" for item in observations
        )
        support = (
            sorted({item["block_instance_id"] for item in observations})
            if consensus_exact
            else []
        )
        mapping["consensus"]["supporting_block_instances"] = support or [
            mapping["block_instance_id"]
        ]
        mapping["consensus"]["counterexample_block_instances"] = (
            []
            if consensus_exact
            else sorted(
                {
                    item["block_instance_id"]
                    for item in observations
                    if tuple(item["covered_unit_ids"])
                    != tuple(mapping["covered_unit_ids"])
                    or item["relation"] != "exact_cover"
                }
            )
        )
        if mapping["relation"] == "exact_cover" and len(support) >= 3:
            mapping["confidence"] = "source_unit_template_exact"
        mapping["mapping_fingerprint"] = canonical_sha256(mapping)
        mappings.append(mapping)
    result = {
        "schema_version": "1.0",
        "protocol": "sk_source_map_v1",
        "base_revision": source_manifest["base_revision"],
        "calibration_revision": source_manifest["calibration_revision"],
        "stable_marker_revision": source_manifest.get("stable_marker_revision"),
        "source_manifest_fingerprint": expected_source,
        **(
            {"calibration_source_manifest_fingerprint": calibration_source_fingerprint}
            if calibration_source_fingerprint is not None
            else {}
        ),
        "calibration_projection_fingerprint": expected_projection,
        "original_collection_fingerprint": projection[
            "original_collection_fingerprint"
        ],
        "calibration_collection_fingerprint": projection[
            "calibration_collection_fingerprint"
        ],
        "algorithm_versions": {
            "set_relation": "target-business-occurrence-set-equality-v2",
            "structural_family": "block-local-business-structure-v1",
            "consensus": "three-instance-no-counterexample-v1",
            "assignment_filter": "per-sk-target-exact-assignment-v1",
        },
        "assignment_coverage": {
            "projected_calibration_node_count": len(coordinates),
            "assigned_calibration_node_count": len(assignment_by_node),
            "skipped_calibration_node_count": len(unassigned_calibration_nodes),
            "skipped_calibration_node_keys": sorted(unassigned_calibration_nodes),
            "processed_sk_group_count": sum(
                item["processing_status"] == "processed" for item in mappings
            ),
            "skipped_sk_group_count": sum(
                item["processing_status"] == "skipped" for item in mappings
            ),
        },
        "mappings": sorted(
            mappings, key=lambda item: item["sk_occurrence_fingerprint"]
        ),
        "mapping_groups": [],
    }
    result["map_fingerprint"] = canonical_sha256(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--calibration-projection", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--calibration-source-manifest")
    parser.add_argument("--block-instance-inventory", required=True)
    parser.add_argument("--original-fused-inventory", required=True)
    parser.add_argument("--unit-assignments", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = build_sk_source_map(
            load_json(args.calibration_projection),
            load_json(args.source_manifest),
            load_json(args.block_instance_inventory),
            load_json(args.original_fused_inventory),
            load_json(args.unit_assignments),
            calibration_source_manifest=(
                load_json(args.calibration_source_manifest)
                if args.calibration_source_manifest
                else None
            ),
        )
        atomic_write_json(args.output, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
