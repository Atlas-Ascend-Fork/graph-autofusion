#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Map fused SuperKernel children to SK-off rows through kernel projection.

The mapper deliberately avoids joining host profiler timestamps with device
``sk_prof`` timestamps.  It projects the profile-process origin graph to the
kernel nodes visible in ``kernel_details.csv``, aligns complete per-stream
sequences for every measured step, and then materializes each fused child by
its graph stream role and kernel ordinal.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from artifact_contract import canonical_json
from structural_association import (
    FusedGroups,
    ProfileGraphSet,
    _parse_candidate_fused_metadata_core,
    _read_manifest_artifact,
    _standard_json_from_bytes,
    canonical_core_family,
    canonical_op,
    graph_fingerprint,
    load_candidate_kernel_rows,
    load_candidate_profile_manifest,
    load_profile_fused_groups,
    load_profile_graphs,
)


PROTOCOL = "kernel_projection_trace_v2"
MAPPING_METHOD = "kernel_projection_structural"
EXACT_CONFIDENCE = "exact_projected_trace"
SENTINEL_EXCLUSION_PROTOCOL = "scope_sentinel_exclusion_v2"
_SCOPE_SENTINEL_FUNCTION = re.compile(
    r"^sk_(?P<kind>scope_kernel_begin|placeholder_kernel|scope_kernel_end)_dav_"
    r"(?P<tag>[1-9][0-9]*)$"
)
_SCOPE_SENTINEL_KINDS = (
    "scope_kernel_begin",
    "placeholder_kernel",
    "placeholder_kernel",
    "placeholder_kernel",
    "placeholder_kernel",
    "scope_kernel_end",
)
_SCOPE_SENTINEL_PROJECTED_OPS = frozenset(
    {
        "sk_scope_kernel_begin_dav",
        "sk_placeholder_kernel_dav",
        "sk_scope_kernel_end_dav",
    }
)
REQUIRED_BASELINE_FIELDS = {
    "Step Id",
    "Device_id",
    "Model ID",
    "Task ID",
    "Stream ID",
    "Name",
    "Type",
    "Accelerator Core",
    "Start Time(us)",
    "Duration(us)",
}


def graph_occurrence_fingerprint(
    mapping_fingerprint, *, device_id, model_id, sk_id, ordered_child_node_keys
):
    """Return the canonical join key for one fused graph occurrence."""
    return hashlib.sha256(
        canonical_json(
            {
                "mapping_fingerprint": mapping_fingerprint,
                "device_id": device_id,
                "model_id": model_id,
                "sk_id": sk_id,
                "children": list(ordered_child_node_keys),
            }
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value, label, *, positive=False):
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} is not a finite decimal") from error
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        raise ValueError(f"{label} is not a finite decimal")
    return result


def projection_op(name, core_family=None):
    """Return the observable operator token shared by graph and profiler rows."""
    core = canonical_core_family(core_family)
    if core == "COMMUNICATION":
        return "COMMUNICATION"
    op = canonical_op(name)
    if op.lower().startswith("aclnn") and "_" in op:
        op = op.rsplit("_", 1)[-1]
    return op


def projection_core(value, op=None):
    core = canonical_core_family(value)
    if core == "UNKNOWN" and op is not None:
        core = canonical_core_family(op)
    return core


def load_baseline_rows(path, *, device_id, model_id):
    result = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not REQUIRED_BASELINE_FIELDS <= set(
            reader.fieldnames
        ):
            raise ValueError("baseline kernel_details headers are incomplete")
        for source_row, row in enumerate(reader, start=2):
            if str(row["Device_id"]).strip() != str(device_id):
                continue
            if str(row["Model ID"]).strip() != str(model_id):
                continue
            stream_text = str(row["Stream ID"]).strip()
            if not stream_text.isdecimal():
                continue
            step_text = str(row["Step Id"]).strip()
            if not step_text.isdecimal():
                raise ValueError(f"baseline row {source_row} has invalid Step Id")
            task_text = str(row["Task ID"]).strip()
            result.append(
                {
                    "source_row": source_row,
                    "step_id": int(step_text),
                    "device_id": int(str(row["Device_id"]).strip()),
                    "model_id": int(str(row["Model ID"]).strip()),
                    "stream_id": int(stream_text),
                    "task_id": int(task_text) if task_text.isdecimal() else None,
                    "name": str(row["Name"]).strip(),
                    "type": str(row["Type"]).strip(),
                    "core_family": projection_core(
                        row["Accelerator Core"], row["Name"]
                    ),
                    "projected_op": projection_op(row["Name"], row["Accelerator Core"]),
                    "start_us": _decimal(
                        row["Start Time(us)"], f"baseline row {source_row} start"
                    ),
                    "duration_us": _decimal(
                        row["Duration(us)"],
                        f"baseline row {source_row} duration",
                        positive=True,
                    ),
                }
            )
    if not result:
        raise ValueError("baseline kernel projection is empty")
    return result


def source_kernel_projection(graph, excluded_node_keys=()):
    excluded = frozenset(excluded_node_keys)
    streams = defaultdict(list)
    node_locations = {}
    for node in graph.nodes:
        if node.topology_only or node.key in excluded:
            continue
        streams[node.stream_key].append(node)
    projected = {}
    for stream_key, nodes in streams.items():
        nodes.sort(key=lambda node: node.stream_ordinal)
        values = []
        for kernel_ordinal, node in enumerate(nodes):
            token = (
                projection_op(node.canonical_op, node.core_family),
                projection_core(node.core_family, node.canonical_op),
            )
            values.append(token)
            node_locations[node.key] = (stream_key, kernel_ordinal, token)
        projected[stream_key] = tuple(values)
    if not projected:
        raise ValueError("source graph kernel projection is empty")
    return projected, node_locations


def _node_provenance(node):
    return dict(node.provenance)


def _sentinel_match(node):
    func_name = _node_provenance(node).get("func_name")
    if not isinstance(func_name, str):
        return None
    return _SCOPE_SENTINEL_FUNCTION.fullmatch(func_name)


def _sentinel_chain_proposal(graph):
    """Propose concrete origin node keys; never changes projection by itself."""
    marker_nodes = [node for node in graph.nodes if _sentinel_match(node)]
    if not marker_nodes:
        return {
            "status": "not_present",
            "node_keys": [],
            "function_names": [],
            "stream_role": None,
            "tags": [],
            "occurrence_count": 0,
            "signature": None,
            "blockers": [],
        }

    blockers = set()
    if any(
        node.task_kind != "KERNEL" or node.core_family != "AI_VECTOR_CORE"
        for node in marker_nodes
    ):
        blockers.add("scope_sentinel_not_vector_kernel")
    stream_keys = {node.stream_key for node in marker_nodes}
    ordered_nodes = sorted(
        marker_nodes, key=lambda node: (node.stream_key, node.stream_ordinal)
    )
    matches = [_sentinel_match(node) for node in ordered_nodes]
    tag_counts = defaultdict(Counter)
    for match in matches:
        tag_counts[match.group("tag")][match.group("kind")] += 1
    for counts in tag_counts.values():
        begins = counts["scope_kernel_begin"]
        if (
            begins <= 0
            or counts["scope_kernel_end"] != begins
            or counts["placeholder_kernel"] != 4 * begins
        ):
            blockers.add("scope_sentinel_chain_incomplete")

    occurrence_count = sum(
        counts["scope_kernel_begin"] for counts in tag_counts.values()
    )
    signature = []
    simple_chain_layout = len(stream_keys) == 1 and all(
        len(nodes) % len(_SCOPE_SENTINEL_KINDS) == 0
        for nodes in (
            [node for node in ordered_nodes if node.stream_key == stream_key]
            for stream_key in stream_keys
        )
    )
    if simple_chain_layout:
        width = len(_SCOPE_SENTINEL_KINDS)
        occurrences = [
            ordered_nodes[offset : offset + width]
            for offset in range(0, len(ordered_nodes), width)
        ]
        for occurrence_index, nodes in enumerate(occurrences):
            kinds = tuple(_sentinel_match(node).group("kind") for node in nodes)
            tags = tuple(_sentinel_match(node).group("tag") for node in nodes)
            ordinals = tuple(node.stream_ordinal for node in nodes)
            if kinds != _SCOPE_SENTINEL_KINDS:
                blockers.add("scope_sentinel_chain_incomplete")
            if len(set(tags)) != 1:
                blockers.add("scope_sentinel_chain_tag_inconsistent")
            if tuple(ordinals[:3]) != tuple(
                range(ordinals[0], ordinals[0] + 3)
            ) or tuple(ordinals[3:]) != tuple(range(ordinals[3], ordinals[3] + 3)):
                blockers.add("scope_sentinel_chain_noncontiguous")
            if ordinals[3] <= ordinals[2] + 1:
                blockers.add("scope_sentinel_business_interval_missing")
            signature.append(
                {
                    "occurrence_ordinal": occurrence_index,
                    "tag": tags[0] if len(set(tags)) == 1 else None,
                    "kinds": list(kinds),
                    "stream_ordinals": list(ordinals),
                    "core_families": [node.core_family for node in nodes],
                }
            )
    else:
        for stream_key in sorted(stream_keys):
            nodes = [node for node in ordered_nodes if node.stream_key == stream_key]
            signature.append(
                {
                    "stream_role": stream_key,
                    "markers": [
                        {
                            "kind": _sentinel_match(node).group("kind"),
                            "tag": _sentinel_match(node).group("tag"),
                            "stream_ordinal": node.stream_ordinal,
                            "core_family": node.core_family,
                        }
                        for node in nodes
                    ],
                }
            )

    tags = sorted(tag_counts, key=int)
    return {
        "status": "proposed" if not blockers else "rejected",
        "node_keys": [node.key for node in ordered_nodes],
        "function_names": [
            _node_provenance(node)["func_name"] for node in ordered_nodes
        ],
        "stream_role": next(iter(stream_keys)) if len(stream_keys) == 1 else None,
        "stream_roles": sorted(stream_keys),
        "tags": tags,
        "occurrence_count": occurrence_count,
        "signature": signature,
        "blockers": sorted(blockers),
    }


def _same_model_origin_proposals(graph_set, model_id):
    proposals = []
    for item in graph_set.graphs:
        if item.model_id.split("_", 1)[0] != str(model_id):
            continue
        proposals.append((item.device_id, _sentinel_chain_proposal(item.graph)))
    return proposals


def _json_marker_function_names(value):
    names = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "funcName" and isinstance(child, str):
                if _SCOPE_SENTINEL_FUNCTION.fullmatch(child):
                    names.append(child)
            else:
                names.extend(_json_marker_function_names(child))
    elif isinstance(value, list):
        for child in value:
            names.extend(_json_marker_function_names(child))
    return names


def _updated_graph_marker_evidence(manifest, model_id):
    checked = 0
    visible = []
    for record in manifest.records_for("sk_graph_updated"):
        data, _ = _read_manifest_artifact(manifest, record, "sk_graph_updated")
        value = _standard_json_from_bytes(data, "sk_graph_updated")
        observed_model = str(value.get("modelId", "")).split("_", 1)[0]
        if observed_model != str(model_id):
            continue
        checked += 1
        visible.extend(_json_marker_function_names(value))
    return checked, sorted(set(visible))


def _candidate_profiler_marker_evidence(manifest, *, model_id, required_steps):
    checked_rows = 0
    visible = []
    for record in manifest.records_for("kernel_details"):
        data, _ = _read_manifest_artifact(manifest, record, "kernel_details")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("candidate kernel_details must be UTF-8 CSV") from error
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        required = {"Step Id", "Device_id", "Model ID", "Name"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("candidate kernel_details headers are incomplete")
        for row in reader:
            if (
                str(row["Model ID"]).strip() != str(model_id)
                or not str(row["Step Id"]).strip().isdecimal()
                or int(str(row["Step Id"]).strip()) not in required_steps
            ):
                continue
            checked_rows += 1
            projected = projection_op(row["Name"], row.get("Accelerator Core"))
            if projected in _SCOPE_SENTINEL_PROJECTED_OPS:
                visible.append(str(row["Name"]).strip())
    return checked_rows, sorted(set(visible))


def _calibration_marker_namespace(manifest):
    manifest_value = getattr(manifest, "manifest_value", None)
    if not isinstance(manifest_value, dict) and not hasattr(manifest_value, "get"):
        return None
    config = manifest_value.get("config")
    if not isinstance(config, dict) and not hasattr(config, "get"):
        return None
    namespace = config.get("marker_namespace")
    if not isinstance(namespace, str) or not namespace.strip():
        return None
    return namespace.strip()


def _assignments_identical(assignments, required_steps):
    if len(assignments) != len(required_steps):
        return False
    values = [assignments[step] for step in required_steps]
    return all(value == values[0] for value in values[1:])


def _all_fused_children_project(
    groups, node_locations, baseline_steps, assignments, required_steps
):
    for group in groups:
        for child in group.kernel_nodes:
            location = node_locations.get(child.node_key)
            if location is None:
                return False
            source_stream, ordinal, token = location
            for step in required_steps:
                baseline_stream = assignments[step].get(source_stream)
                if baseline_stream is None:
                    return False
                rows = baseline_steps[step][baseline_stream]["rows"]
                if ordinal >= len(rows):
                    return False
                row = rows[ordinal]
                if (row["projected_op"], row["core_family"]) != token:
                    return False
    return True


def guarded_sentinel_exclusion(
    graph,
    graph_set,
    groups,
    manifest,
    baseline_rows,
    baseline_steps,
    required_steps,
    *,
    device_id,
    model_id,
):
    proposal = _sentinel_chain_proposal(graph)
    evidence = {
        "protocol": SENTINEL_EXCLUSION_PROTOCOL,
        "status": proposal["status"],
        "candidate_node_keys": proposal["node_keys"],
        "excluded_node_keys": [],
        "function_names": proposal["function_names"],
        "stream_role": proposal["stream_role"],
        "stream_roles": proposal.get("stream_roles", []),
        "tags": proposal["tags"],
        "occurrence_count": proposal["occurrence_count"],
        "origin_graph_count": 0,
        "updated_graph_count": 0,
        "candidate_profiler_rows_checked": 0,
        "gates": {},
        "blockers": list(proposal["blockers"]),
    }
    if proposal["status"] == "not_present":
        return frozenset(), evidence
    if proposal["status"] != "proposed":
        return frozenset(), evidence

    blockers = set()
    gates = evidence["gates"]
    gates["complete_occurrence_partition"] = True

    origin_proposals = _same_model_origin_proposals(graph_set, model_id)
    evidence["origin_graph_count"] = len(origin_proposals)
    origin_consistent = bool(origin_proposals) and all(
        item["status"] == "proposed" and item["signature"] == proposal["signature"]
        for _, item in origin_proposals
    )
    gates["all_origin_graphs_consistent"] = origin_consistent
    if not origin_consistent:
        blockers.add("scope_sentinel_origin_graph_inconsistent")

    referenced = {node.node_key for group in groups for node in group.nodes} & set(
        proposal["node_keys"]
    )
    gates["absent_from_fused_children"] = not referenced
    if referenced:
        blockers.add("scope_sentinel_referenced_by_fused_group")

    baseline_visible = sorted(
        {
            row["name"]
            for row in baseline_rows
            if row["step_id"] in required_steps
            and row["projected_op"] in _SCOPE_SENTINEL_PROJECTED_OPS
        }
    )
    gates["absent_from_baseline_profiler"] = not baseline_visible
    if baseline_visible:
        blockers.add("scope_sentinel_visible_in_baseline_profiler")

    checked_rows, candidate_visible = _candidate_profiler_marker_evidence(
        manifest,
        model_id=model_id,
        required_steps=set(required_steps),
    )
    evidence["candidate_profiler_rows_checked"] = checked_rows
    marker_namespace = _calibration_marker_namespace(manifest)
    evidence["marker_namespace"] = marker_namespace
    if marker_namespace is None:
        evidence["candidate_profiler_policy"] = "production_absence_required"
        gates["absent_from_candidate_profiler"] = not candidate_visible
        if candidate_visible:
            blockers.add("scope_sentinel_visible_in_candidate_profiler")
    else:
        evidence["candidate_profiler_policy"] = "calibration_visibility_allowed"
        gates["calibration_profiler_marker_policy"] = checked_rows > 0
        if checked_rows <= 0:
            blockers.add("scope_sentinel_candidate_profiler_evidence_missing")

    updated_count, updated_visible = _updated_graph_marker_evidence(manifest, model_id)
    evidence["updated_graph_count"] = updated_count
    updated_absent = updated_count == len(origin_proposals) and not updated_visible
    gates["absent_from_all_updated_graphs"] = updated_absent
    if not updated_absent:
        blockers.add("scope_sentinel_updated_graph_evidence_incomplete")

    source_streams, node_locations = source_kernel_projection(
        graph, proposal["node_keys"]
    )
    source_count = sum(len(values) for values in source_streams.values())
    count_match = all(
        step in baseline_steps
        and len(source_streams) == len(baseline_steps[step])
        and source_count
        == sum(len(value["rows"]) for value in baseline_steps[step].values())
        for step in required_steps
    )
    gates["post_exclusion_counts_match"] = count_match
    if not count_match:
        blockers.add("scope_sentinel_post_exclusion_count_mismatch")

    assignments, alignment_blockers, alternatives = align_projection_steps(
        source_streams, baseline_steps, required_steps
    )
    unique = not alignment_blockers and all(
        alternatives.get(str(step)) == 0 for step in required_steps
    )
    gates["unique_full_sequence_assignment"] = unique
    if not unique:
        blockers.add("scope_sentinel_post_exclusion_mapping_not_unique")
    identical = unique and _assignments_identical(assignments, required_steps)
    gates["assignment_identical_across_steps"] = identical
    if not identical:
        blockers.add("scope_sentinel_step_assignment_inconsistent")

    children_exact = unique and _all_fused_children_project(
        groups, node_locations, baseline_steps, assignments, required_steps
    )
    gates["all_fused_children_project_exactly"] = children_exact
    if not children_exact:
        blockers.add("scope_sentinel_fused_child_projection_failed")

    evidence["blockers"] = sorted(blockers)
    if blockers:
        evidence["status"] = "rejected"
        return frozenset(), evidence
    evidence["status"] = "accepted"
    evidence["excluded_node_keys"] = proposal["node_keys"]
    return frozenset(proposal["node_keys"]), evidence


def baseline_step_projection(rows):
    by_step_stream = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_step_stream[row["step_id"]][row["stream_id"]].append(row)
    result = {}
    for step_id, streams in by_step_stream.items():
        projected_streams = {}
        for stream_id, stream_rows in streams.items():
            stream_rows.sort(
                key=lambda row: (
                    row["start_us"],
                    row["task_id"] is None,
                    row["task_id"] if row["task_id"] is not None else -1,
                    row["source_row"],
                )
            )
            projected_streams[stream_id] = {
                "signature": tuple(
                    (row["projected_op"], row["core_family"]) for row in stream_rows
                ),
                "rows": tuple(stream_rows),
            }
        result[step_id] = projected_streams
    return result


def _stream_candidates(source_streams, baseline_streams):
    return {
        source_stream: tuple(
            sorted(
                baseline_stream
                for baseline_stream, evidence in baseline_streams.items()
                if evidence["signature"] == signature
            )
        )
        for source_stream, signature in source_streams.items()
    }


def solve_unique_stream_assignment(source_streams, baseline_streams):
    if len(source_streams) != len(baseline_streams):
        return None, "kernel_projection_stream_count_mismatch", 0
    candidates = _stream_candidates(source_streams, baseline_streams)
    if any(not values for values in candidates.values()):
        return None, "kernel_projection_stream_unmapped", 0
    ordered = sorted(candidates, key=lambda key: (len(candidates[key]), key))
    solutions = []

    def visit(index, used, current):
        if len(solutions) >= 2:
            return
        if index == len(ordered):
            solutions.append(dict(current))
            return
        source_stream = ordered[index]
        for baseline_stream in candidates[source_stream]:
            if baseline_stream in used:
                continue
            used.add(baseline_stream)
            current[source_stream] = baseline_stream
            visit(index + 1, used, current)
            current.pop(source_stream)
            used.remove(baseline_stream)

    visit(0, set(), {})
    if not solutions:
        return None, "kernel_projection_injective_assignment_missing", 0
    if len(solutions) > 1:
        return None, "kernel_projection_stream_assignment_ambiguous", len(solutions)
    return solutions[0], None, 1


def align_projection_steps(source_streams, baseline_steps, required_steps):
    assignments = {}
    blockers = []
    alternatives = {}
    for step_id in required_steps:
        streams = baseline_steps.get(step_id)
        if streams is None:
            blockers.append("kernel_projection_step_missing")
            continue
        assignment, blocker, solution_count = solve_unique_stream_assignment(
            source_streams, streams
        )
        alternatives[str(step_id)] = max(0, solution_count - 1)
        if blocker:
            blockers.append(blocker)
        else:
            assignments[step_id] = assignment
    if len(assignments) == len(required_steps):
        role_signatures = {
            source_stream: tuple(
                baseline_steps[step][assignments[step][source_stream]]["signature"]
                for step in required_steps
            )
            for source_stream in source_streams
        }
        if any(len(set(signatures)) != 1 for signatures in role_signatures.values()):
            blockers.append("kernel_projection_step_inconsistent")
    return assignments, sorted(set(blockers)), alternatives


def _union_duration(intervals):
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = Decimal(0)
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("timing value is not finite")
    return result


def _decimal_text(value):
    return format(value, "f")


def materialize_group(
    group,
    candidate_rows,
    node_locations,
    baseline_steps,
    assignments,
    required_steps,
):
    blockers = set(group.blockers)
    children = list(group.kernel_nodes)
    if not children:
        blockers.add("metadata_kernel_nodes_missing")
    occurrences = [
        row
        for row in candidate_rows
        if row.device_id == group.device_id
        and row.model_id == group.model_id
        and row.sk_id == group.sk_id
        and row.step_id in required_steps
    ]
    occurrences.sort(key=lambda row: (row.step_id, row.start_us))
    if len(occurrences) != len(required_steps) or {
        row.step_id for row in occurrences
    } != set(required_steps):
        blockers.add("candidate_projection_occurrence_domain_mismatch")

    child_locations = []
    for child in children:
        location = node_locations.get(child.node_key)
        if location is None:
            blockers.add("metadata_origin_node_not_in_kernel_projection")
        else:
            child_locations.append((child, location))

    baseline_occurrences = []
    if not blockers:
        for step_id in required_steps:
            mapped_rows = []
            for child, (source_stream, kernel_ordinal, token) in child_locations:
                baseline_stream = assignments[step_id][source_stream]
                rows = baseline_steps[step_id][baseline_stream]["rows"]
                if kernel_ordinal >= len(rows):
                    blockers.add("baseline_projection_ordinal_missing")
                    break
                row = rows[kernel_ordinal]
                observed = (row["projected_op"], row["core_family"])
                if observed != token:
                    blockers.add("baseline_projection_child_mismatch")
                    break
                mapped_rows.append(row)
            if blockers:
                break
            intervals = [
                (row["start_us"], row["start_us"] + row["duration_us"])
                for row in mapped_rows
            ]
            interval_start = min(start for start, _ in intervals)
            interval_end = max(end for _, end in intervals)
            baseline_occurrences.append(
                {
                    "step_id": step_id,
                    "interval_us": _number(interval_end - interval_start),
                    "duration_sum_us": _number(
                        sum((row["duration_us"] for row in mapped_rows), Decimal(0))
                    ),
                    "union_duration_us": _number(_union_duration(intervals)),
                    "children": [
                        {
                            "origin_node_key": child.node_key,
                            "source_stream_role": source_stream,
                            "kernel_ordinal": kernel_ordinal,
                            "baseline_stream_id": row["stream_id"],
                            "baseline_task_id": row["task_id"],
                            "baseline_source_row": row["source_row"],
                            "baseline_name": row["name"],
                            "projected_op": row["projected_op"],
                            "core_family": row["core_family"],
                            "start_us": _decimal_text(row["start_us"]),
                            "duration_us": _number(row["duration_us"]),
                        }
                        for (child, (source_stream, kernel_ordinal, _)), row in zip(
                            child_locations, mapped_rows
                        )
                    ],
                }
            )

    exact = not blockers and len(baseline_occurrences) == len(required_steps)
    return {
        "status": "exact" if exact else "blocked",
        "mapping_method": MAPPING_METHOD,
        "mapping_confidence": EXACT_CONFIDENCE if exact else "diagnostic_only",
        "mapping_blockers": sorted(blockers),
        "device_id": group.device_id,
        "model_id": group.model_id,
        "sk_id": group.sk_id,
        "name": group.name,
        "candidate_source_scope": getattr(
            getattr(group, "boundary", None), "source_scope", None
        ),
        "child_count": len(children),
        "ordered_child_node_keys": [child.node_key for child in children],
        "ordered_child_ops": [child.canonical_op for child in children],
        "candidate_occurrences": [
            {
                "step_id": row.step_id,
                "start_us": _decimal_text(row.start_us),
                "duration_us": _number(row.duration_us),
            }
            for row in occurrences
        ],
        "baseline_occurrences": baseline_occurrences,
        "metadata_fingerprint": group.fused_fingerprint,
    }


def build_mapping(baseline_path, profile_manifest_path, *, device_id, model_id):
    manifest = load_candidate_profile_manifest(profile_manifest_path)
    graph_set = load_profile_graphs(manifest)
    fused = load_profile_fused_groups(manifest)
    candidate_rows = load_candidate_kernel_rows(manifest)
    target_graphs = tuple(
        item
        for item in graph_set.graphs
        if item.device_id == device_id
        and item.model_id.split("_", 1)[0] == str(model_id)
    )
    if len(target_graphs) != 1:
        raise ValueError("profile origin graph identity is ambiguous")
    filtered_graph_set = ProfileGraphSet(
        graph_set.binding, target_graphs, graph_set._seal
    )
    target_graph_parent = str(
        Path(target_graphs[0].source.record.relative_path).parent
    ).replace("\\", "/")
    target_sources = tuple(
        source
        for source in fused.sources
        if str(Path(source.record.relative_path).parent).replace("\\", "/")
        == target_graph_parent
    )
    filtered_fused = FusedGroups(fused.binding, target_sources, fused._seal)
    groups = _parse_candidate_fused_metadata_core(filtered_graph_set, filtered_fused)
    groups = [
        group
        for group in groups
        if group.device_id == device_id and group.model_id == model_id
    ]
    if not groups:
        raise ValueError("profile fused metadata groups are missing")

    graph = target_graphs[0].graph
    baseline_rows = load_baseline_rows(
        baseline_path, device_id=device_id, model_id=model_id
    )
    baseline_steps = baseline_step_projection(baseline_rows)
    required_steps = sorted(
        {
            row.step_id
            for row in candidate_rows.rows
            if row.device_id == device_id and row.model_id == model_id
        }
    )
    if len(required_steps) < 3:
        raise ValueError("candidate kernel projection has fewer than three steps")
    excluded_node_keys, sentinel_exclusion = guarded_sentinel_exclusion(
        graph,
        graph_set,
        groups,
        manifest,
        baseline_rows,
        baseline_steps,
        required_steps,
        device_id=device_id,
        model_id=model_id,
    )
    source_streams, node_locations = source_kernel_projection(graph, excluded_node_keys)
    assignments, alignment_blockers, alternatives = align_projection_steps(
        source_streams, baseline_steps, required_steps
    )

    mappings = []
    if alignment_blockers:
        for group in groups:
            mappings.append(
                {
                    "status": "blocked",
                    "mapping_method": MAPPING_METHOD,
                    "mapping_confidence": "diagnostic_only",
                    "mapping_blockers": alignment_blockers,
                    "device_id": group.device_id,
                    "model_id": group.model_id,
                    "sk_id": group.sk_id,
                    "name": group.name,
                    "candidate_source_scope": getattr(
                        getattr(group, "boundary", None), "source_scope", None
                    ),
                    "child_count": len(group.kernel_nodes),
                    "ordered_child_node_keys": [
                        child.node_key for child in group.kernel_nodes
                    ],
                    "ordered_child_ops": [
                        child.canonical_op for child in group.kernel_nodes
                    ],
                    "candidate_occurrences": [],
                    "baseline_occurrences": [],
                    "metadata_fingerprint": group.fused_fingerprint,
                }
            )
    else:
        for group in sorted(groups, key=lambda item: item.sk_id):
            mappings.append(
                materialize_group(
                    group,
                    candidate_rows.rows,
                    node_locations,
                    baseline_steps,
                    assignments,
                    required_steps,
                )
            )

    blocker_counts = Counter(
        blocker for item in mappings for blocker in item["mapping_blockers"]
    )
    exact = sum(item["status"] == "exact" for item in mappings)
    stream_mapping = {
        str(step): {
            source_stream: baseline_stream
            for source_stream, baseline_stream in sorted(assignment.items())
        }
        for step, assignment in sorted(assignments.items())
    }
    result = {
        "schema_version": "2.0",
        "protocol": PROTOCOL,
        "mapping_method": MAPPING_METHOD,
        "status": "exact" if exact == len(mappings) else "partial",
        "identity": {"device_id": device_id, "model_id": model_id},
        "steps": required_steps,
        "summary": {
            "source_kernel_nodes": sum(len(value) for value in source_streams.values()),
            "source_kernel_streams": len(source_streams),
            "baseline_rows_per_step": {
                str(step): sum(
                    len(value["rows"]) for value in baseline_steps[step].values()
                )
                for step in required_steps
                if step in baseline_steps
            },
            "total_sk_ids": len(mappings),
            "exact_sk_ids": exact,
            "blocked_sk_ids": len(mappings) - exact,
            "coverage_pct": round(100.0 * exact / len(mappings), 4),
            "blocker_counts": dict(sorted(blocker_counts.items())),
        },
        "stream_role_mapping": stream_mapping,
        "alternative_solution_count_by_step": alternatives,
        "projection_exclusions": sentinel_exclusion,
        "evidence": {
            "baseline_kernel_details_sha256": _sha256(baseline_path),
            "profile_manifest_fingerprint": manifest.manifest_fingerprint,
            "profile_origin_graph_fingerprint": graph_fingerprint(graph),
        },
        "mappings": mappings,
    }
    result["mapping_fingerprint"] = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def render_markdown(result):
    summary = result["summary"]
    lines = [
        "# SuperKernel 融合前子算子映射报告",
        "",
        f"- 协议：`{result['protocol']}`",
        f"- 状态：`{result['status']}`",
        f"- 设备/模型：`{result['identity']['device_id']}` / "
        f"`{result['identity']['model_id']}`",
        f"- 动态 step：`{', '.join(map(str, result['steps']))}`",
        f"- 原图可观测 kernel：`{summary['source_kernel_nodes']}`，"
        f"跨 `{summary['source_kernel_streams']}` 个 stream role",
        f"- SK 映射：`{summary['exact_sk_ids']}/{summary['total_sk_ids']}` "
        f"(`{summary['coverage_pct']:.4f}%`)",
        "",
        "## 方法结论",
        "",
        "该映射不再把静态控制图直接同 profiler 动态 trace 做全图同构。"
        "它先移除 profiler 不可稳定观测的 control/event 节点，再以完整流内 kernel "
        "序列求唯一 stream-role 注入，最后用 fused metadata 中的 origin nodeId 对应"
        "到每个 step 的 baseline kernel ordinal。主机时间戳与设备 sk_prof 时间戳不做"
        "跨时钟域偏移求解。",
        "",
        "## Step 覆盖",
        "",
        "| Step | baseline kernel 行数 | stream role 映射数 | 替代解数 |",
        "|---:|---:|---:|---:|",
    ]
    for step in result["steps"]:
        lines.append(
            f"| {step} | {summary['baseline_rows_per_step'].get(str(step), 0)} | "
            f"{len(result['stream_role_mapping'].get(str(step), {}))} | "
            f"{result['alternative_solution_count_by_step'].get(str(step), 0)} |"
        )
    exclusion = result["projection_exclusions"]
    lines.extend(
        [
            "",
            "## Projection 排除门禁",
            "",
            f"- 协议：`{exclusion['protocol']}`",
            f"- 状态：`{exclusion['status']}`",
            f"- 候选/已排除 node key：`{len(exclusion['candidate_node_keys'])}` / "
            f"`{len(exclusion['excluded_node_keys'])}`",
            f"- scope sentinel occurrence：`{exclusion['occurrence_count']}`",
            f"- 同模型 origin/updated graph：`{exclusion['origin_graph_count']}` / "
            f"`{exclusion['updated_graph_count']}`",
            f"- blockers：`{', '.join(exclusion['blockers']) if exclusion['blockers'] else 'none'}`",
        ]
    )
    lines.extend(
        [
            "",
            "## 映射清单",
            "",
            "| SK | child | 状态 | P50 baseline interval (us) |",
            "|---:|---:|---|---:|",
        ]
    )
    for item in result["mappings"]:
        intervals = sorted(
            occurrence["interval_us"] for occurrence in item["baseline_occurrences"]
        )
        p50 = intervals[len(intervals) // 2] if intervals else None
        lines.append(
            f"| {item['sk_id']} | {item['child_count']} | "
            f"`{item['mapping_confidence']}` | "
            f"{p50:.6f} |"
            if p50 is not None
            else f"| {item['sk_id']} | {item['child_count']} | "
            f"`{item['mapping_confidence']}` | N/A |"
        )
    if summary["blocker_counts"]:
        lines.extend(["", "## Blockers", ""])
        for blocker, count in summary["blocker_counts"].items():
            lines.append(f"- `{blocker}`: {count}")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "该结果证明的是 baseline 动态 kernel occurrence，不证明 Python 源码区间。"
            "`sk_prof` 仅用于融合后子节点调度分析；缺失或跨时钟域不影响本映射，"
            "但会限制 Cube/Vector 内部调度归因。",
            "",
            f"映射指纹：`{result['mapping_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-kernel-details", required=True)
    parser.add_argument("--profile-collection-manifest", required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = build_mapping(
        args.baseline_kernel_details,
        args.profile_collection_manifest,
        device_id=args.device_id,
        model_id=args.model_id,
    )
    json_path = Path(args.json_out)
    markdown_path = Path(args.markdown_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        print(f"projected trace mapping failed: {error}", file=sys.stderr)
        raise SystemExit(2)
