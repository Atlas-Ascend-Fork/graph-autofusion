#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Normalize one sk_graph_origin artifact into a calibration business graph."""

from __future__ import annotations

import argparse
import hashlib
import re

from artifact_contract import canonical_json as projection_canonical_json
from projected_trace_mapping import PROTOCOL as PROJECTION_PROTOCOL
from structural_association import graph_fingerprint, graph_from_sk_origin
from source_calibration_common import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_json,
    require_integer,
    require_object,
    require_scalar_id,
    require_sha256,
    require_text,
)


NORMALIZATION_PROTOCOL = "normalized_business_graph_v2"
SENTINEL_EXCLUSION_PROTOCOL = "scope_sentinel_exclusion_v2"
_SCOPE_SENTINEL_FUNCTION = re.compile(
    r"^sk_(?:scope_kernel_begin|placeholder_kernel|scope_kernel_end)_dav_"
    r"[1-9][0-9]*$"
)


def _base_model_id(value, label):
    """Canonicalize CANN model-instance ids such as ``48_1`` to ``48``."""
    model_id = require_scalar_id(value, label)
    base_model_id = model_id.split("_", 1)[0]
    try:
        return str(int(base_model_id))
    except ValueError as error:
        raise ValueError(f"{label} must start with an integer model id") from error


def _validated_projection_exclusions(value, graph, *, device_id, model_id):
    trace = require_object(value, "projection trace")
    if trace.get("protocol") != PROJECTION_PROTOCOL:
        raise ValueError(f"projection trace.protocol must be {PROJECTION_PROTOCOL}")
    expected_fingerprint = require_sha256(
        trace.get("mapping_fingerprint"), "projection trace.mapping_fingerprint"
    )
    payload = {key: item for key, item in trace.items() if key != "mapping_fingerprint"}
    observed_fingerprint = hashlib.sha256(
        projection_canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if observed_fingerprint != expected_fingerprint:
        raise ValueError("projection trace mapping_fingerprint mismatch")
    identity = require_object(trace.get("identity"), "projection trace.identity")
    observed_device_id = require_integer(
        identity.get("device_id"), "projection trace.identity.device_id"
    )
    observed_model_id = _base_model_id(
        identity.get("model_id"), "projection trace.identity.model_id"
    )
    if observed_device_id != device_id or observed_model_id != model_id:
        raise ValueError("projection trace identity does not match the origin graph")
    evidence = require_object(trace.get("evidence"), "projection trace.evidence")
    if evidence.get("profile_origin_graph_fingerprint") != graph_fingerprint(graph):
        raise ValueError("projection trace does not bind the supplied origin graph")
    exclusion = require_object(
        trace.get("projection_exclusions"), "projection trace.projection_exclusions"
    )
    if exclusion.get("protocol") != SENTINEL_EXCLUSION_PROTOCOL:
        raise ValueError(
            "projection trace uses an unsupported sentinel exclusion protocol"
        )
    excluded = exclusion.get("excluded_node_keys")
    if not isinstance(excluded, list) or any(
        not isinstance(key, str) for key in excluded
    ):
        raise ValueError("projection trace excluded_node_keys must be a string list")
    if excluded and exclusion.get("status") != "accepted":
        raise ValueError("projection trace sentinel exclusion was not accepted")
    return frozenset(excluded), expected_fingerprint


def _marker_keys(graph):
    return {
        node.key
        for node in graph.nodes
        if _SCOPE_SENTINEL_FUNCTION.fullmatch(
            dict(node.provenance).get("func_name", "")
        )
    }


def normalize_origin_graph(
    origin_graph_path,
    *,
    collection_fingerprint,
    model_role="candidate",
    projection_trace=None,
):
    """Return a fail-closed business graph derived from one origin graph."""
    collection_fingerprint = require_sha256(
        collection_fingerprint, "collection_fingerprint"
    )
    model_role = require_text(model_role, "model_role")
    origin_sha256 = file_sha256(origin_graph_path)
    graph = graph_from_sk_origin(origin_graph_path, model_role=model_role)
    raw = require_object(
        load_json(origin_graph_path, "sk_graph_origin"), "sk_graph_origin"
    )
    try:
        device_id = int(str(raw.get("deviceId")))
    except (TypeError, ValueError) as error:
        raise ValueError("sk_graph_origin.deviceId must be an integer") from error
    origin_model_id = require_scalar_id(raw.get("modelId"), "sk_graph_origin.modelId")
    model_id = _base_model_id(origin_model_id, "sk_graph_origin.modelId")

    excluded = frozenset()
    projection_fingerprint = None
    if projection_trace is not None:
        excluded, projection_fingerprint = _validated_projection_exclusions(
            projection_trace, graph, device_id=device_id, model_id=model_id
        )
    known_keys = {node.key for node in graph.nodes}
    if not excluded <= known_keys:
        raise ValueError("projection trace excludes unknown origin graph nodes")
    markers = _marker_keys(graph)
    if markers != set(excluded):
        raise ValueError(
            "scope sentinel nodes require an accepted, complete projection exclusion"
        )
    if any(
        not node.topology_only and node.key in excluded and node.key not in markers
        for node in graph.nodes
    ):
        raise ValueError("projection trace attempts to exclude a business kernel")

    retained = {
        node.key: node
        for node in graph.nodes
        if not node.topology_only and node.key not in excluded
    }
    if not retained:
        raise ValueError("normalized business graph contains no business nodes")

    stream_sequences = {}
    locations = {}
    nodes = []
    for stream_key, sequence in graph.streams:
        business_keys = [key for key in sequence if key in retained]
        if business_keys:
            stream_sequences[stream_key] = business_keys
        for position, key in enumerate(sequence):
            locations[key] = (stream_key, position, sequence)
        for ordinal, key in enumerate(business_keys):
            node = retained[key]
            nodes.append(
                {
                    "node_key": key,
                    "stream_role": stream_key,
                    "stream_ordinal": ordinal,
                    "canonical_op": node.canonical_op,
                    "core_family": node.core_family,
                    "observable_signature": {
                        "shape_dtype": list(node.shape_dtype),
                    },
                    "provenance": dict(node.provenance),
                }
            )

    edges = []
    for stream_key in sorted(stream_sequences):
        sequence = stream_sequences[stream_key]
        edges.extend(
            {"source": left, "target": right, "kind": "STREAM_ORDER", "ports": {}}
            for left, right in zip(sequence, sequence[1:])
        )

    # Origin graphs express cross-stream business control through NOTIFY/WAIT nodes.
    # Contract each event edge onto its nearest surrounding business kernels.
    for edge in graph.edges:
        if edge.kind != "event":
            continue
        _, source_position, source_sequence = locations[edge.source]
        _, target_position, target_sequence = locations[edge.target]
        source = next(
            (
                key
                for key in reversed(source_sequence[:source_position])
                if key in retained
            ),
            None,
        )
        target = next(
            (key for key in target_sequence[target_position + 1 :] if key in retained),
            None,
        )
        if source is not None and target is not None:
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "kind": "BUSINESS_CONTROL",
                    "ports": {},
                }
            )

    result = {
        "protocol": NORMALIZATION_PROTOCOL,
        "collection_fingerprint": collection_fingerprint,
        "identity": {
            "device_id": device_id,
            "model_id": model_id,
        },
        "algorithm_versions": {
            "origin_graph_parser": "structural-association-origin-graph-v2",
            "kernel_projection_trace": PROJECTION_PROTOCOL,
            "canonical_op": "canonical-op-v2",
            "core_family": "canonical-core-family-v2",
            "topology_filter": "topology-only-filter-v2",
            "sentinel_exclusion": SENTINEL_EXCLUSION_PROTOCOL,
            "control_contraction": "notify-wait-business-control-v1",
        },
        "nodes": sorted(nodes, key=lambda item: item["node_key"]),
        "edges": sorted(
            edges,
            key=lambda item: (
                item["source"],
                item["target"],
                item["kind"],
                canonical_sha256(item["ports"]),
            ),
        ),
        "normalization_evidence": {
            "origin_graph_sha256": origin_sha256,
            "origin_graph_fingerprint": graph_fingerprint(graph),
            "origin_model_id": origin_model_id,
            "projection_mapping_fingerprint": projection_fingerprint,
            "excluded_node_keys": sorted(excluded),
        },
    }
    if file_sha256(origin_graph_path) != origin_sha256:
        raise ValueError("sk_graph_origin changed while it was being normalized")
    result["normalization_fingerprint"] = canonical_sha256(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--origin-graph", required=True)
    parser.add_argument("--collection-fingerprint", required=True)
    parser.add_argument("--model-role", default="candidate")
    parser.add_argument("--projection-trace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = normalize_origin_graph(
            args.origin_graph,
            collection_fingerprint=args.collection_fingerprint,
            model_role=args.model_role,
            projection_trace=(
                load_json(args.projection_trace, "projection trace")
                if args.projection_trace
                else None
            ),
        )
        atomic_write_json(args.output, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
