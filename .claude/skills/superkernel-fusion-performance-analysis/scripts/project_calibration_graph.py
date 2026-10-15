#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Project an original business graph onto a calibration graph uniquely."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict

from source_calibration_common import (
    atomic_write_json,
    canonical_sha256,
    load_json,
    require_integer,
    require_list,
    require_object,
    require_scalar_id,
    require_text,
)


PROTOCOL_VERSIONS = {
    "graph_projection": "labeled-business-multigraph-v2",
    "canonical_op": "canonical-op-v2",
    "core_family": "canonical-core-family-v2",
    "topology_filter": "topology-only-filter-v2",
    "sentinel_exclusion": "scope-sentinel-exclusion-v2",
    "node_label": "observable-node-label-v2",
    "dependency_edge": "typed-business-edge-v2",
    "isomorphism": "unique-labeled-stream-chain-isomorphism-v3",
    "runtime_assignment": "per-step-node-partition-v1",
    "fingerprint": "canonical-json-sha256-v1",
}
EDGE_KINDS = frozenset(("DATA", "BUSINESS_CONTROL", "STREAM_ORDER"))


def _node_label(node):
    return (
        node["canonical_op"],
        node["core_family"],
        canonical_sha256(node.get("observable_signature", {})),
    )


def _normalize_graph(value, label):
    graph = require_object(value, label)
    if graph.get("protocol") != "normalized_business_graph_v2":
        raise ValueError(f"{label}.protocol must be normalized_business_graph_v2")
    identity = require_object(graph.get("identity"), f"{label}.identity")
    device_id = require_integer(
        identity.get("device_id"), f"{label}.identity.device_id"
    )
    model_id = require_scalar_id(identity.get("model_id"), f"{label}.identity.model_id")
    nodes = {}
    streams = defaultdict(list)
    for index, raw in enumerate(require_list(graph.get("nodes"), f"{label}.nodes")):
        node = require_object(raw, f"{label}.nodes[{index}]")
        key = require_text(node.get("node_key"), f"{label}.nodes[{index}].node_key")
        if key in nodes:
            raise ValueError(f"{label} has duplicate node_key {key}")
        normalized = {
            "node_key": key,
            "stream_role": require_text(
                node.get("stream_role"), f"{label}.nodes[{index}].stream_role"
            ),
            "stream_ordinal": require_integer(
                node.get("stream_ordinal"), f"{label}.nodes[{index}].stream_ordinal"
            ),
            "canonical_op": require_text(
                node.get("canonical_op"), f"{label}.nodes[{index}].canonical_op"
            ),
            "core_family": require_text(
                node.get("core_family"), f"{label}.nodes[{index}].core_family"
            ),
            "observable_signature": require_object(
                node.get("observable_signature", {}),
                f"{label}.nodes[{index}].observable_signature",
            ),
            "provenance": copy.deepcopy(node.get("provenance", {})),
        }
        nodes[key] = normalized
        streams[normalized["stream_role"]].append(normalized)
    for role, sequence in streams.items():
        ordinals = sorted(node["stream_ordinal"] for node in sequence)
        if ordinals != list(range(len(sequence))):
            raise ValueError(
                f"{label} stream {role} ordinals must be contiguous from zero"
            )
    edge_counter = Counter()
    edge_records = []
    for index, raw in enumerate(require_list(graph.get("edges"), f"{label}.edges")):
        edge = require_object(raw, f"{label}.edges[{index}]")
        source = require_text(edge.get("source"), f"{label}.edges[{index}].source")
        target = require_text(edge.get("target"), f"{label}.edges[{index}].target")
        kind = require_text(edge.get("kind"), f"{label}.edges[{index}].kind")
        if source not in nodes or target not in nodes:
            raise ValueError(f"{label}.edges[{index}] references an unknown node")
        if kind not in EDGE_KINDS:
            raise ValueError(f"{label}.edges[{index}].kind is unsupported")
        signature = canonical_sha256(edge.get("ports", {}))
        token = (source, target, kind, signature)
        edge_counter[token] += 1
        edge_records.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "ports": edge.get("ports", {}),
            }
        )
    normalized_graph = {
        "protocol": "normalized_business_graph_v2",
        "identity": {"device_id": device_id, "model_id": model_id},
        "nodes": sorted(nodes.values(), key=lambda item: item["node_key"]),
        "edges": sorted(
            edge_records,
            key=lambda item: (
                item["source"],
                item["target"],
                item["kind"],
                canonical_sha256(item["ports"]),
            ),
        ),
    }
    return {
        "identity": normalized_graph["identity"],
        "nodes": nodes,
        "streams": streams,
        "edges": edge_counter,
        "fingerprint": canonical_sha256(normalized_graph),
    }


def _incident_signature(graph, key):
    incoming = Counter()
    outgoing = Counter()
    for (source, target, kind, ports), count in graph["edges"].items():
        if source == key:
            outgoing[(kind, ports, _node_label(graph["nodes"][target]))] += count
        if target == key:
            incoming[(kind, ports, _node_label(graph["nodes"][source]))] += count
    return canonical_sha256(
        {
            "incoming": sorted((repr(k), v) for k, v in incoming.items()),
            "outgoing": sorted((repr(k), v) for k, v in outgoing.items()),
        }
    )


def _stream_candidates(original, calibration):
    result = {}
    for role, nodes in original["streams"].items():
        labels = Counter(_node_label(node) for node in nodes)
        candidates = [
            other_role
            for other_role, other_nodes in calibration["streams"].items()
            if Counter(_node_label(node) for node in other_nodes) == labels
        ]
        if not candidates:
            return None
        result[role] = sorted(candidates)
    return result


def _edge_consistent(original, calibration, mapping, source, target):
    reverse = {value: key for key, value in mapping.items()}
    for (left, right, kind, ports), count in original["edges"].items():
        if left in mapping and right in mapping:
            if (
                calibration["edges"][(mapping[left], mapping[right], kind, ports)]
                != count
            ):
                return False
    for (left, right, kind, ports), count in calibration["edges"].items():
        if left in reverse and right in reverse:
            if original["edges"][(reverse[left], reverse[right], kind, ports)] != count:
                return False
    return True


def _find_node_solutions(original, calibration, *, limit=2):
    stream_candidates = _stream_candidates(original, calibration)
    if stream_candidates is None:
        return []
    original_keys = set(original["nodes"])
    solutions = []

    def search(mapping, used):
        if len(solutions) >= limit:
            return
        if len(mapping) == len(original_keys):
            if len(used) == len(calibration["nodes"]):
                solutions.append(dict(mapping))
            return
        choices = []
        for key in original_keys - set(mapping):
            node = original["nodes"][key]
            candidates = []
            for candidate_key, candidate in calibration["nodes"].items():
                if candidate_key in used or _node_label(node) != _node_label(candidate):
                    continue
                if (
                    candidate["stream_role"]
                    not in stream_candidates[node["stream_role"]]
                ):
                    continue
                existing_stream = {
                    calibration["nodes"][mapped]["stream_role"]
                    for source, mapped in mapping.items()
                    if original["nodes"][source]["stream_role"] == node["stream_role"]
                }
                if existing_stream and candidate["stream_role"] not in existing_stream:
                    continue
                owners = {
                    original["nodes"][source]["stream_role"]
                    for source, mapped in mapping.items()
                    if calibration["nodes"][mapped]["stream_role"]
                    == candidate["stream_role"]
                }
                if owners and node["stream_role"] not in owners:
                    continue
                tentative = {**mapping, key: candidate_key}
                if _edge_consistent(
                    original, calibration, tentative, key, candidate_key
                ):
                    candidates.append(candidate_key)
            choices.append((len(candidates), key, sorted(candidates)))
        _, key, candidates = min(choices)
        for candidate_key in candidates:
            mapping[key] = candidate_key
            used.add(candidate_key)
            search(mapping, used)
            used.remove(candidate_key)
            del mapping[key]

    search({}, set())
    return solutions


def _stream_order_chains(graph):
    """Return topology-proven stream chains, or None for conservative fallback."""
    chains = {
        role: tuple(
            node["node_key"]
            for node in sorted(nodes, key=lambda item: item["stream_ordinal"])
        )
        for role, nodes in graph["streams"].items()
    }
    expected = Counter()
    empty_ports = canonical_sha256({})
    for chain in chains.values():
        expected.update(
            (left, right, "STREAM_ORDER", empty_ports)
            for left, right in zip(chain, chain[1:])
        )
    observed = Counter(
        {
            edge: count
            for edge, count in graph["edges"].items()
            if edge[2] == "STREAM_ORDER"
        }
    )
    return chains if observed == expected else None


def _mapping_preserves_all_edges(original, calibration, mapping):
    projected = Counter(
        (mapping[source], mapping[target], kind, ports)
        for (source, target, kind, ports), count in original["edges"].items()
        for _ in range(count)
    )
    return projected == calibration["edges"]


def _find_stream_chain_solutions(original, calibration, *, limit=2):
    original_chains = _stream_order_chains(original)
    calibration_chains = _stream_order_chains(calibration)
    if original_chains is None or calibration_chains is None:
        return None

    original_signatures = {
        role: tuple(_node_label(original["nodes"][key]) for key in chain)
        for role, chain in original_chains.items()
    }
    calibration_signatures = {
        role: tuple(_node_label(calibration["nodes"][key]) for key in chain)
        for role, chain in calibration_chains.items()
    }
    candidates = {
        role: sorted(
            other_role
            for other_role, signature in calibration_signatures.items()
            if signature == original_signatures[role]
        )
        for role in original_chains
    }
    if any(not roles for roles in candidates.values()):
        return []

    solutions = []

    def search(role_mapping, used_roles, node_mapping):
        if len(solutions) >= limit:
            return
        if len(role_mapping) == len(original_chains):
            if _mapping_preserves_all_edges(original, calibration, node_mapping):
                solutions.append(dict(node_mapping))
            return
        choices = [
            (
                len(
                    [
                        candidate
                        for candidate in candidates[role]
                        if candidate not in used_roles
                    ]
                ),
                role,
            )
            for role in original_chains
            if role not in role_mapping
        ]
        _, role = min(choices)
        for candidate_role in candidates[role]:
            if candidate_role in used_roles:
                continue
            additions = dict(
                zip(original_chains[role], calibration_chains[candidate_role])
            )
            tentative = {**node_mapping, **additions}
            if not _edge_consistent(original, calibration, tentative, "", ""):
                continue
            role_mapping[role] = candidate_role
            used_roles.add(candidate_role)
            search(role_mapping, used_roles, tentative)
            used_roles.remove(candidate_role)
            del role_mapping[role]

    search({}, set(), {})
    return solutions


def _find_solutions(original, calibration, *, limit=2):
    stream_solutions = _find_stream_chain_solutions(original, calibration, limit=limit)
    if stream_solutions is not None:
        return stream_solutions
    return _find_node_solutions(original, calibration, limit=limit)


def project_graphs(
    original_value,
    calibration_value,
    *,
    runtime_validation=None,
    evidence_catalog=None,
    original_collection_fingerprint=None,
    calibration_collection_fingerprint=None,
):
    original = _normalize_graph(original_value, "original graph")
    calibration = _normalize_graph(calibration_value, "calibration graph")
    if original["identity"] != calibration["identity"]:
        raise ValueError("original and calibration graph identities differ")
    if len(original["nodes"]) != len(calibration["nodes"]):
        raise ValueError("business kernel count differs")
    if Counter(
        (kind, ports) for _, _, kind, ports in original["edges"].elements()
    ) != Counter(
        (kind, ports) for _, _, kind, ports in calibration["edges"].elements()
    ):
        raise ValueError("typed business edge counts differ")
    solutions = _find_solutions(original, calibration, limit=2)
    if len(solutions) != 1:
        raise ValueError(
            "graph correspondence is missing or ambiguous; exact projection requires one solution"
        )
    mapping = solutions[0]
    validations = require_list(runtime_validation or [], "runtime_validation")
    valid_steps = []
    normalized_validations = []
    calibration_node_keys = set(calibration["nodes"])
    for index, item in enumerate(validations):
        item = require_object(item, f"runtime_validation[{index}]")
        assignment_partitions = {}
        partition_fields = (
            "exact_assigned_calibration_node_keys",
            "skipped_calibration_node_keys",
            "conflicting_calibration_node_keys",
        )
        provided_partition_fields = [
            field for field in partition_fields if field in item
        ]
        if provided_partition_fields and len(provided_partition_fields) != len(
            partition_fields
        ):
            raise ValueError(
                f"runtime_validation[{index}] must provide all assignment partition fields"
            )
        if provided_partition_fields:
            for field in partition_fields:
                values = [
                    require_text(value, f"runtime_validation[{index}].{field}")
                    for value in require_list(
                        item.get(field), f"runtime_validation[{index}].{field}"
                    )
                ]
                if len(values) != len(set(values)):
                    raise ValueError(
                        f"runtime_validation[{index}].{field} contains duplicates"
                    )
                if set(values) - calibration_node_keys:
                    raise ValueError(
                        f"runtime_validation[{index}].{field} references unknown nodes"
                    )
                assignment_partitions[field] = set(values)
            partition_union = set().union(*assignment_partitions.values())
            partition_total = sum(
                len(values) for values in assignment_partitions.values()
            )
            if partition_total != len(partition_union):
                raise ValueError(
                    f"runtime_validation[{index}] assignment partitions overlap"
                )
            if partition_union != calibration_node_keys:
                raise ValueError(
                    f"runtime_validation[{index}] assignment partitions must cover the calibration graph"
                )
        passed = (
            item.get("parent_binding_exact") is True
            and item.get("unit_assignment_complete") is True
            and require_integer(
                item.get("business_occurrence_count"),
                f"runtime_validation[{index}].business_occurrence_count",
            )
            == len(original["nodes"])
            and require_integer(
                item.get("assigned_occurrence_count"),
                f"runtime_validation[{index}].assigned_occurrence_count",
            )
            == len(calibration["nodes"])
            and require_integer(
                item.get("unscoped_occurrence_count"),
                f"runtime_validation[{index}].unscoped_occurrence_count",
            )
            == 0
            and require_integer(
                item.get("conflicting_assignment_count"),
                f"runtime_validation[{index}].conflicting_assignment_count",
            )
            == 0
        )
        step_id = require_integer(
            item.get("step_id"), f"runtime_validation[{index}].step_id"
        )
        normalized = {**copy.deepcopy(item), "passed": passed}
        if assignment_partitions:
            if require_integer(
                item.get("business_occurrence_count"),
                f"runtime_validation[{index}].business_occurrence_count",
            ) != len(calibration_node_keys):
                raise ValueError(
                    f"runtime_validation[{index}] business occurrence count mismatch"
                )
            count_fields = {
                "exact_assigned_calibration_node_keys": "assigned_occurrence_count",
                "skipped_calibration_node_keys": "unscoped_occurrence_count",
                "conflicting_calibration_node_keys": "conflicting_assignment_count",
            }
            for field, count_field in count_fields.items():
                expected_count = require_integer(
                    item.get(count_field), f"runtime_validation[{index}].{count_field}"
                )
                if len(assignment_partitions[field]) != expected_count:
                    raise ValueError(
                        f"runtime_validation[{index}].{field} count mismatch"
                    )
                normalized[field] = sorted(assignment_partitions[field])
            normalized["assignment_partition_complete"] = True
        else:
            normalized["assignment_partition_complete"] = False
        normalized["validation_fingerprint"] = canonical_sha256(normalized)
        normalized_validations.append(normalized)
        if passed:
            valid_steps.append(step_id)
    occurrences = []
    for source in sorted(mapping):
        target = mapping[source]
        source_node = original["nodes"][source]
        target_node = calibration["nodes"][target]
        node_label_fingerprint = canonical_sha256(_node_label(source_node))
        incident_edge_fingerprint = canonical_sha256(
            [
                _incident_signature(original, source),
                _incident_signature(calibration, target),
            ]
        )
        occurrence = {
            "original_node_key": source,
            "calibration_node_key": target,
            "original": [
                source_node["stream_role"],
                source_node["stream_ordinal"],
                source_node["canonical_op"],
                source_node["core_family"],
            ],
            "calibration": [
                target_node["stream_role"],
                target_node["stream_ordinal"],
                target_node["canonical_op"],
                target_node["core_family"],
            ],
            "node_label_fingerprint": node_label_fingerprint,
            "incident_edge_fingerprint": incident_edge_fingerprint,
        }
        occurrence["correspondence_fingerprint"] = canonical_sha256(occurrence)
        occurrences.append(occurrence)
    stream_assignment = {}
    for source, target in mapping.items():
        left = original["nodes"][source]["stream_role"]
        right = calibration["nodes"][target]["stream_role"]
        previous = stream_assignment.setdefault(left, right)
        if previous != right:
            raise ValueError("node mapping does not define a stream-role injection")
    result = {
        "protocol": "source_calibration_projection_v2",
        "protocol_versions": copy.deepcopy(PROTOCOL_VERSIONS),
        "original_collection_fingerprint": require_text(
            original_collection_fingerprint
            or original_value.get("collection_fingerprint"),
            "original_collection_fingerprint",
        ),
        "calibration_collection_fingerprint": require_text(
            calibration_collection_fingerprint
            or calibration_value.get("collection_fingerprint"),
            "calibration_collection_fingerprint",
        ),
        "device_id": original["identity"]["device_id"],
        "model_id": original["identity"]["model_id"],
        "original_graph_fingerprint": original["fingerprint"],
        "calibration_graph_fingerprint": calibration["fingerprint"],
        "stream_role_assignment": dict(sorted(stream_assignment.items())),
        "alternative_stream_assignment_count": 0,
        "alternative_node_correspondence_count": 0,
        "nontrivial_automorphism_count": 0,
        "business_kernel_sequence_equal": True,
        "typed_business_edge_set_equal": True,
        "evidence_catalog": copy.deepcopy(
            require_object(evidence_catalog or {}, "evidence_catalog")
        ),
        "runtime_validation": normalized_validations,
        "validated_step_ids": sorted(set(valid_steps)),
        "occurrences": occurrences,
    }
    result["node_correspondence_fingerprint"] = canonical_sha256(occurrences)
    result["projection_fingerprint"] = canonical_sha256(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--original-graph", required=True)
    parser.add_argument("--calibration-graph", required=True)
    parser.add_argument("--runtime-validation")
    parser.add_argument("--evidence-catalog")
    parser.add_argument("--original-collection-fingerprint")
    parser.add_argument("--calibration-collection-fingerprint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = project_graphs(
            load_json(args.original_graph, "original graph"),
            load_json(args.calibration_graph, "calibration graph"),
            runtime_validation=(
                load_json(args.runtime_validation) if args.runtime_validation else []
            ),
            evidence_catalog=(
                load_json(args.evidence_catalog) if args.evidence_catalog else {}
            ),
            original_collection_fingerprint=args.original_collection_fingerprint,
            calibration_collection_fingerprint=args.calibration_collection_fingerprint,
        )
        atomic_write_json(args.output, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
