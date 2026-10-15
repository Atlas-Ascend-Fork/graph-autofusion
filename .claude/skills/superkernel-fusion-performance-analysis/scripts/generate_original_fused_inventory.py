#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate original_fused_inventory_v1 from exact projection evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from projected_trace_mapping import graph_occurrence_fingerprint
from source_calibration_common import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_json,
    require_list,
    require_object,
    require_sha256,
    require_text,
)


def _fingerprint_without(value, field):
    return canonical_sha256({key: item for key, item in value.items() if key != field})


def build_inventory(trace, projection, block_inventory, unit_assignments):
    trace = require_object(trace, "projection trace")
    if trace.get("protocol") != "kernel_projection_trace_v2":
        raise ValueError("projection trace protocol is unsupported")
    trace_fingerprint = require_sha256(
        trace.get("mapping_fingerprint"), "projection trace mapping_fingerprint"
    )
    if trace_fingerprint != _fingerprint_without(trace, "mapping_fingerprint"):
        raise ValueError("projection trace fingerprint mismatch")

    projection = require_object(projection, "calibration projection")
    if projection.get("protocol") != "source_calibration_projection_v2":
        raise ValueError("calibration projection protocol is unsupported")
    projection_fingerprint = require_sha256(
        projection.get("projection_fingerprint"),
        "calibration projection fingerprint",
    )
    if projection_fingerprint != _fingerprint_without(
        projection, "projection_fingerprint"
    ):
        raise ValueError("calibration projection fingerprint mismatch")

    correspondence = {}
    original_coordinates = {}
    for raw in require_list(projection.get("occurrences"), "projection occurrences"):
        item = require_object(raw, "projection occurrence")
        original = require_text(item.get("original_node_key"), "original node key")
        calibration = require_text(
            item.get("calibration_node_key"), "calibration node key"
        )
        coordinate = require_list(item.get("original"), "original coordinate")
        if original in correspondence:
            raise ValueError(f"duplicate original projection node {original}")
        correspondence[original] = calibration
        original_coordinates[original] = tuple(coordinate)

    block_inventory = require_object(block_inventory, "block inventory")
    bindings = {}
    for raw in require_list(block_inventory.get("instances"), "block instances"):
        item = require_object(raw, "block instance")
        block_id = require_text(item.get("block_instance_id"), "block instance id")
        binding = require_object(item.get("runtime_scope_binding"), "scope binding")
        if binding.get("method") != "exact_scope_name":
            raise ValueError(f"block {block_id} has unsupported scope binding")
        bindings[block_id] = require_text(binding.get("value"), "scope binding value")

    unit_assignments = require_object(unit_assignments, "unit assignments")
    if unit_assignments.get("protocol") != "calibration_unit_assignments_v1":
        raise ValueError("unit assignments protocol is unsupported")
    assigned_blocks = {}
    for raw in require_list(unit_assignments.get("assignments"), "assignments"):
        item = require_object(raw, "unit assignment")
        node_key = require_text(item.get("calibration_node_key"), "assignment node")
        block_id = require_text(item.get("block_instance_id"), "assignment block")
        if node_key in assigned_blocks:
            raise ValueError(f"duplicate unit assignment for {node_key}")
        if block_id not in bindings:
            raise ValueError(f"assignment references unknown block {block_id}")
        assigned_blocks[node_key] = block_id

    candidates = defaultdict(list)
    skipped = Counter()
    for raw in require_list(trace.get("mappings"), "projection mappings"):
        group = require_object(raw, "projection mapping")
        if group.get("status") != "exact":
            skipped["projection_not_exact"] += 1
            continue
        children = [
            require_text(key, "original child node key")
            for key in require_list(
                group.get("ordered_child_node_keys"), "ordered child node keys"
            )
        ]
        if not children or len(children) != len(set(children)):
            raise ValueError("projection mapping child keys are empty or duplicated")
        if any(key not in correspondence for key in children):
            skipped["projection_correspondence_missing"] += 1
            continue
        calibration_children = [correspondence[key] for key in children]
        blocks = {
            assigned_blocks[key]
            for key in calibration_children
            if key in assigned_blocks
        }
        if not blocks:
            skipped["source_block_unassigned"] += 1
            continue
        if len(blocks) != 1:
            skipped["source_block_ambiguous"] += 1
            continue
        block_id = next(iter(blocks))
        sort_coordinate = min(original_coordinates[key] for key in children)
        candidates[block_id].append((sort_coordinate, group, children))

    records = []
    groups = []
    for block_id in sorted(
        candidates, key=lambda value: (not value.isdecimal(), value)
    ):
        ordered = sorted(
            candidates[block_id], key=lambda item: (item[0], int(item[1]["sk_id"]))
        )
        for ordinal, (_, source_group, children) in enumerate(ordered):
            identity = {
                "device_id": source_group["device_id"],
                "model_id": source_group["model_id"],
                "sk_id": source_group["sk_id"],
                "block_instance_id": block_id,
                "candidate_source_scope": require_text(
                    source_group.get("candidate_source_scope"),
                    "candidate source scope",
                ),
                "original_child_node_keys": children,
                "metadata_fingerprint": source_group.get("metadata_fingerprint"),
            }
            occurrence_fingerprint = graph_occurrence_fingerprint(
                trace_fingerprint,
                device_id=source_group["device_id"],
                model_id=source_group["model_id"],
                sk_id=source_group["sk_id"],
                ordered_child_node_keys=children,
            )
            record = {
                "record_id": f"sk:{source_group['sk_id']}",
                **identity,
                "sk_occurrence_fingerprint": occurrence_fingerprint,
            }
            record["record_fingerprint"] = canonical_sha256(record)
            records.append(record)
            groups.append(
                {
                    "device_id": source_group["device_id"],
                    "model_id": source_group["model_id"],
                    "block_instance_id": block_id,
                    "source_scope": bindings[block_id],
                    "candidate_source_scope": identity["candidate_source_scope"],
                    "sk_occurrence_fingerprint": occurrence_fingerprint,
                    "baseline_projection_fingerprint": trace_fingerprint,
                    "block_local_sk_ordinal": ordinal,
                    "ordered_child_ops": list(source_group["ordered_child_ops"]),
                    "original_child_node_keys": children,
                    "evidence_records": [
                        {
                            "artifact_id": "original-fused-projection-trace",
                            "record_id": record["record_id"],
                            "record_fingerprint": record["record_fingerprint"],
                        }
                    ],
                }
            )

    inventory = {
        "protocol": "original_fused_inventory_v1",
        "baseline_projection_fingerprint": trace_fingerprint,
        "calibration_projection_fingerprint": projection_fingerprint,
        "generation_summary": {
            "input_sk_group_count": len(trace["mappings"]),
            "emitted_sk_group_count": len(groups),
            "skipped_sk_group_count": sum(skipped.values()),
            "skip_reasons": dict(sorted(skipped.items())),
        },
        "sk_groups": groups,
    }
    evidence = {
        "protocol": "normalized_log_records_v1",
        "records": records,
    }
    return inventory, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--projection-trace", required=True)
    parser.add_argument("--calibration-projection", required=True)
    parser.add_argument("--block-instance-inventory", required=True)
    parser.add_argument("--unit-assignments", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        inventory, evidence = build_inventory(
            load_json(args.projection_trace),
            load_json(args.calibration_projection),
            load_json(args.block_instance_inventory),
            load_json(args.unit_assignments),
        )
        evidence_path = Path(args.evidence_output)
        output_path = Path(args.output)
        artifact_root = Path(args.artifact_root or output_path.parent).resolve()
        atomic_write_json(evidence_path, evidence)
        evidence_path = evidence_path.resolve()
        try:
            evidence_relative_path = evidence_path.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError("evidence output must be under artifact_root") from error
        inventory["evidence_catalog"] = {
            "original-fused-projection-trace": {
                "relative_path": evidence_relative_path.as_posix(),
                "sha256": file_sha256(evidence_path),
            }
        }
        atomic_write_json(output_path, inventory)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
