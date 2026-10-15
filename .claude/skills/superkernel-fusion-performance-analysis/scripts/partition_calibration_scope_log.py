#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Partition calibration business nodes by semantic scope-log markers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from source_calibration_common import atomic_write_json, canonical_sha256


BEGIN = "Scope split results begin: pass=InitialScopeSplitPass"
END = "Scope split results end: pass=InitialScopeSplitPass"
SCOPE = re.compile(r"Scope ([0-9]+) .*scopeNames=\[([^]]*)\]")
NODE = re.compile(r"\[nodeId:([0-9]+),")
TRIGGER = re.compile(r"BreakInfo:.*triggerNode=([0-9]+),")
BLOCK = re.compile(r"(?:^|_)skcal\.block\.([A-Za-z0-9-]+)(?=_skcal\.|$)")
UNIT = re.compile(
    r"(?:^|_)skcal\.unit\.([A-Za-z0-9-]+)\.([A-Za-z0-9_.-]+?)(?=_skcal\.|$)"
)


def semantic_unit_matches(scope_name: str, unit_ids: set[str]) -> list[tuple[str, str]]:
    """Resolve unit tokens against the manifest despite outer-scope suffixes."""
    prefixes = re.finditer(r"(?:^|_)skcal\.unit\.([A-Za-z0-9-]+)\.", scope_name)
    matches = []
    ordered_ids = sorted(unit_ids, key=lambda value: (-len(value), value))
    for prefix in prefixes:
        tail = scope_name[prefix.end() :]
        candidates = [
            unit_id
            for unit_id in ordered_ids
            if tail == unit_id or tail.startswith(f"{unit_id}_")
        ]
        if candidates:
            matches.append((prefix.group(1), candidates[0]))
    return matches


def first_initial_pass(path: Path) -> list[dict]:
    scopes: list[dict] = []
    active = False
    current = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        if not active:
            if BEGIN in line:
                active = True
            continue
        if END in line:
            break
        header = SCOPE.search(line)
        if header:
            current = {
                "scope_index": int(header.group(1)),
                "scope_name": header.group(2),
                "header_line": line_number,
                "node_ids": [],
                "trigger_node_ids": [],
            }
            scopes.append(current)
            continue
        node = NODE.search(line)
        if node and current is not None:
            current["node_ids"].append(int(node.group(1)))
        trigger = TRIGGER.search(line)
        if trigger and current is not None:
            current["trigger_node_ids"].append(int(trigger.group(1)))
    if not scopes:
        raise ValueError("first InitialScopeSplitPass contains no scopes")
    return scopes


def partition_scope_log(
    scope_path: Path,
    graph: dict,
    manifest: dict,
    steps: list[int],
    output_dir: Path,
    artifact_root: Path | None = None,
) -> dict:
    unit_ids = {item["unit_id"] for item in manifest["units"]}
    template_ids = {item["block_template_id"] for item in manifest["block_templates"]}
    if len(template_ids) != 1:
        raise ValueError("this marker run must bind exactly one block template")
    template_id = next(iter(template_ids))
    node_id_to_key = {
        int(item["provenance"]["node_id"]): item["node_key"] for item in graph["nodes"]
    }
    all_node_keys = set(node_id_to_key.values())

    candidates: dict[str, list[dict]] = defaultdict(list)
    normalized_records = []
    observed_blocks = set()
    ignored_trigger_node_ids = []
    scopes = first_initial_pass(scope_path)
    for scope_index, scope in enumerate(scopes):
        block_matches = BLOCK.findall(scope["scope_name"])
        unit_matches = semantic_unit_matches(scope["scope_name"], unit_ids)
        observed_blocks.update(block_matches)
        if not unit_matches:
            ignored_trigger_node_ids.extend(scope["trigger_node_ids"])
            continue
        if len(set(block_matches)) != 1 or len(set(unit_matches)) != 1:
            raise ValueError(
                f"scope has ambiguous marker tokens: {scope['scope_name']}"
            )
        block_id = block_matches[0]
        unit_block_id, unit_id = unit_matches[0]
        if unit_block_id != block_id:
            raise ValueError(
                f"scope has inconsistent parent/unit binding: {scope['scope_name']}"
            )
        if unit_id not in unit_ids:
            raise ValueError(f"scope references unknown semantic unit: {unit_id}")

        next_scope = scopes[scope_index + 1] if scope_index + 1 < len(scopes) else None
        continuation = (
            next_scope is not None and next_scope["scope_name"] == scope["scope_name"]
        )
        scoped_nodes = [(node_id, "scope_member") for node_id in scope["node_ids"]]
        if continuation:
            scoped_nodes.extend(
                (node_id, "break_trigger_continuation")
                for node_id in scope["trigger_node_ids"]
            )
        else:
            ignored_trigger_node_ids.extend(scope["trigger_node_ids"])

        # A break trigger belongs to this unit only when the immediately following
        # scope is a continuation with the exact same marker identity. A terminal
        # trigger occurs after the marker end and belongs to neither this unit nor
        # its final PrintScopeNodes segment.
        for node_id, evidence_kind in scoped_nodes:
            node_key = node_id_to_key.get(node_id)
            if node_key is None:
                continue
            record = {
                "artifact_id": "fine-calibration-initial-scope-log",
                "record_id": f"scope:{scope['scope_index']}:node:{node_id}",
                "scope_index": scope["scope_index"],
                "scope_name": scope["scope_name"],
                "header_line": scope["header_line"],
                "node_id": node_id,
                "evidence_kind": evidence_kind,
                "calibration_node_key": node_key,
                "block_instance_id": block_id,
                "unit_id": unit_id,
            }
            record["record_fingerprint"] = canonical_sha256(record)
            normalized_records.append(record)
            candidates[node_key].append(record)

    exact = {}
    conflicting = {}
    for node_key, records in candidates.items():
        identities = {(item["block_instance_id"], item["unit_id"]) for item in records}
        if len(identities) == 1:
            exact[node_key] = records
        else:
            conflicting[node_key] = records
    skipped = sorted(all_node_keys - set(exact) - set(conflicting))

    assignments = []
    for node_key in sorted(exact):
        records = exact[node_key]
        block_id, unit_id = records[0]["block_instance_id"], records[0]["unit_id"]
        assignments.append(
            {
                "calibration_node_key": node_key,
                "block_instance_id": block_id,
                "unit_id": unit_id,
                "control_flow_path": "scope-log:first-InitialScopeSplitPass",
                "evidence_records": [
                    {
                        "artifact_id": record["artifact_id"],
                        "record_id": record["record_id"],
                        "record_fingerprint": record["record_fingerprint"],
                    }
                    for record in records
                ],
            }
        )

    def block_sort(value: str):
        return (0, int(value)) if value.isdigit() else (1, value)

    instances = [
        {
            "block_instance_id": block_id,
            "block_template_id": template_id,
            "runtime_scope_binding": {
                "method": "exact_scope_name",
                "value": f"skcal.block.{block_id}",
            },
            "control_flow_path": "scope-log:first-InitialScopeSplitPass",
            "diagnostic_labels": {"display": f"block {block_id}"},
        }
        for block_id in sorted(observed_blocks, key=block_sort)
    ]
    block_inventory = {
        "schema_version": "1.0",
        "model_adapter_fingerprint": manifest["model_adapter"]["fingerprint"],
        "source_manifest_fingerprint": manifest["manifest_fingerprint"],
        "instances": instances,
    }

    validations = [
        {
            "step_id": step_id,
            "business_occurrence_count": len(all_node_keys),
            "assigned_occurrence_count": len(exact),
            "unscoped_occurrence_count": len(skipped),
            "conflicting_assignment_count": len(conflicting),
            "parent_binding_exact": len(skipped) == 0 and len(conflicting) == 0,
            "unit_assignment_complete": len(skipped) == 0 and len(conflicting) == 0,
            "exact_assigned_calibration_node_keys": sorted(exact),
            "skipped_calibration_node_keys": skipped,
            "conflicting_calibration_node_keys": sorted(conflicting),
            "scope_partition_source": "compiled graph; invariant across runtime steps",
        }
        for step_id in steps
    ]

    catalog = {
        "fine-calibration-initial-scope-log": {
            "path": str(scope_path),
            "sha256": hashlib.sha256(scope_path.read_bytes()).hexdigest(),
            "pass": "first InitialScopeSplitPass",
            "records_path": "normalized-scope-records.json",
            "record_count": len(normalized_records),
            "membership_evidence": (
                "PrintScopeNodes plus break triggers followed by an exact-name "
                "continuation scope"
            ),
        }
    }
    summary = {
        "business_node_count": len(all_node_keys),
        "exact_assigned_node_count": len(exact),
        "skipped_node_count": len(skipped),
        "conflicting_node_count": len(conflicting),
        "block_instance_count": len(instances),
        "source_unit_count": len(unit_ids),
        "normalized_record_count": len(normalized_records),
        "ignored_break_trigger_count": len(ignored_trigger_node_ids),
        "ignored_break_trigger_business_node_count": len(
            set(ignored_trigger_node_ids) & set(node_id_to_key)
        ),
        "steps": steps,
    }
    records_path = output_dir / "normalized-scope-records.json"
    atomic_write_json(
        records_path,
        {"protocol": "normalized_log_records_v1", "records": normalized_records},
    )
    atomic_write_json(output_dir / "evidence-catalog.json", catalog)
    evidence_root = Path(artifact_root or output_dir).resolve()
    try:
        records_relative_path = records_path.resolve().relative_to(evidence_root)
    except ValueError as error:
        raise ValueError("normalized records must be under artifact_root") from error
    atomic_write_json(
        output_dir / "unit-assignments.json",
        {
            "protocol": "calibration_unit_assignments_v1",
            "evidence_catalog": {
                "fine-calibration-initial-scope-log": {
                    "relative_path": records_relative_path.as_posix(),
                    "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                }
            },
            "assignments": assignments,
        },
    )
    atomic_write_json(output_dir / "block-instance-inventory.json", block_inventory)
    atomic_write_json(output_dir / "runtime-validation.json", validations)
    atomic_write_json(output_dir / "partition-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-log", required=True)
    parser.add_argument("--business-graph", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()

    scope_path = Path(args.scope_log).resolve(strict=True)
    graph = json.loads(Path(args.business_graph).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    summary = partition_scope_log(
        scope_path,
        graph,
        manifest,
        args.steps,
        Path(args.output_dir),
        Path(args.artifact_root) if args.artifact_root else None,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
