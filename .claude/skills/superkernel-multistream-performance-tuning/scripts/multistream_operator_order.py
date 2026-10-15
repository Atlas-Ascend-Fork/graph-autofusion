#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Analyze multistream execution/dispatch order and plan dependency-safe reorders."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath

import multistream_core_family
import multistream_dependency_evidence


CAPTURE_SCHEMA = "superkernel-multistream-operator-order-capture-v3"
ANALYSIS_SCHEMA = "superkernel-multistream-operator-order-analysis-v1"
DISPATCH_CAPTURE_SCHEMA = "superkernel-multistream-post-reorder-dispatch-capture-v1"
DISPATCH_EVIDENCE_SCHEMA = "superkernel-multistream-dispatch-order-evidence-v1"
CORE_FAMILIES = multistream_core_family.CORE_FAMILIES
RESOURCE_COMPLEMENTARY_FAMILIES = frozenset({"CUBE", "VECTOR"})
RESOURCE_PAIR_POLICY = "cube_vector_only"
HARD_DEPENDENCY_KINDS = multistream_dependency_evidence.HARD_DEPENDENCY_KINDS


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


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if value < 0 or (positive and value <= 0):
        raise ValueError(
            f"{label} must be {'positive' if positive else 'non-negative'}"
        )
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


def _validate_source_files(root, records):
    if not isinstance(records, list) or not records:
        raise ValueError("source_files must be a non-empty list")
    normalized = []
    seen = set()
    for index, record in enumerate(records):
        label = f"source_files[{index}]"
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "file_fingerprint",
        }:
            raise ValueError(f"{label} fields are invalid")
        path = _rooted(root, record["path"], f"{label}.path")
        relative = path.relative_to(Path(root).resolve()).as_posix()
        if relative in seen:
            raise ValueError(f"duplicate source file: {relative}")
        seen.add(relative)
        actual = file_fingerprint(path)
        if (
            record["size_bytes"] != path.stat().st_size
            or record["file_fingerprint"] != actual
        ):
            raise ValueError(f"source file changed: {relative}")
        normalized.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "file_fingerprint": actual,
            }
        )
    return sorted(normalized, key=lambda item: item["path"])


def _reachable(edges, start, goal):
    children = defaultdict(set)
    for left, right in edges:
        children[left].add(right)
    pending = [start]
    seen = set()
    while pending:
        current = pending.pop()
        if current == goal:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(children[current] - seen)
    return False


def _topological_order(nodes, hard_edges, preferences, current_order):
    combined = set(hard_edges) | set(preferences)
    incoming = {node: 0 for node in nodes}
    children = defaultdict(set)
    for before, after in combined:
        if before == after or before not in incoming or after not in incoming:
            raise ValueError("dependency edge references an invalid statement")
        if after not in children[before]:
            children[before].add(after)
            incoming[after] += 1
    preference_rank = {node: 0 for node in nodes}
    for before, after in preferences:
        preference_rank[before] -= 1
        preference_rank[after] += 1
    original_rank = {node: index for index, node in enumerate(current_order)}
    ready = [node for node, count in incoming.items() if count == 0]
    result = []
    while ready:
        ready.sort(key=lambda node: (preference_rank[node], original_rank[node], node))
        node = ready.pop(0)
        result.append(node)
        for child in sorted(children[node]):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    return result if len(result) == len(nodes) else None


def _bounded_orders(nodes, hard_edges, preferred, current, *, maximum=5):
    """Enumerate a small deterministic prefix of legal topological orders."""
    children = defaultdict(set)
    base_incoming = {node: 0 for node in nodes}
    for before, after in hard_edges:
        if after not in children[before]:
            children[before].add(after)
            base_incoming[after] += 1
    preferred_rank = {node: index for index, node in enumerate(preferred)}
    current_rank = {node: index for index, node in enumerate(current)}
    queue = deque([([], base_incoming)])
    orders = []
    while queue and len(orders) < maximum * 8:
        prefix, incoming = queue.popleft()
        if len(prefix) == len(nodes):
            if prefix not in orders:
                orders.append(prefix)
            continue
        ready = [node for node in nodes if node not in prefix and incoming[node] == 0]
        ready.sort(key=lambda node: (preferred_rank[node], current_rank[node], node))
        for node in ready[:maximum]:
            updated = dict(incoming)
            updated[node] = -1
            for child in children[node]:
                updated[child] -= 1
            queue.append((prefix + [node], updated))
    orders.sort(
        key=lambda order: (
            sum(left != right for left, right in zip(order, preferred)),
            sum(left != right for left, right in zip(order, current)),
            order,
        )
    )
    return orders[:maximum]


def _validate_statements(target, source_size, label):
    statements = target.get("statements")
    if not isinstance(statements, list) or len(statements) < 2:
        raise ValueError(f"{label}.statements requires at least two entries")
    normalized = []
    seen = set()
    for index, item in enumerate(statements):
        item_label = f"{label}.statements[{index}]"
        required = {
            "statement_id",
            "start_offset",
            "end_offset",
            "movable",
            "side_effect_free",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"{item_label} fields are invalid")
        statement_id = _text(item["statement_id"], f"{item_label}.statement_id")
        if statement_id in seen:
            raise ValueError(f"{label} has duplicate statement_id: {statement_id}")
        seen.add(statement_id)
        start = _integer(item["start_offset"], f"{item_label}.start_offset")
        end = _integer(item["end_offset"], f"{item_label}.end_offset", minimum=1)
        if not 0 <= start < end <= source_size:
            raise ValueError(f"{item_label} byte span is invalid")
        if item["movable"] is not True or item["side_effect_free"] is not True:
            raise ValueError(
                f"{item_label} must be explicitly movable and side-effect-free"
            )
        normalized.append({**item, "statement_id": statement_id})
    normalized.sort(key=lambda item: item["start_offset"])
    for left, right in zip(normalized, normalized[1:]):
        if left["end_offset"] != right["start_offset"]:
            raise ValueError(f"{label} statement spans must form one contiguous region")
    return normalized


def _validate_dependencies(value, statement_ids, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    normalized = []
    seen = set()
    for index, edge in enumerate(value):
        edge_label = f"{label}[{index}]"
        if not isinstance(edge, dict) or set(edge) != {"before", "after", "kind"}:
            raise ValueError(f"{edge_label} fields are invalid")
        before = _text(edge["before"], f"{edge_label}.before")
        after = _text(edge["after"], f"{edge_label}.after")
        kind = _text(edge["kind"], f"{edge_label}.kind").upper()
        if before not in statement_ids or after not in statement_ids or before == after:
            raise ValueError(f"{edge_label} references an invalid statement")
        if kind not in HARD_DEPENDENCY_KINDS:
            raise ValueError(f"{edge_label}.kind is not a hard dependency kind")
        key = (before, after, kind)
        if key in seen:
            raise ValueError(f"{label} contains a duplicate edge")
        seen.add(key)
        normalized.append({"before": before, "after": after, "kind": kind})
    pairs = {(item["before"], item["after"]) for item in normalized}
    if _topological_order(statement_ids, pairs, set(), list(statement_ids)) is None:
        raise ValueError(f"{label} contains a cycle")
    return normalized


def _validate_occurrences(value, statement_ids, label):
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{label} requires at least three occurrences")
    normalized = []
    alignment_ids = set()
    operator_to_statement = None
    for index, occurrence in enumerate(value):
        occurrence_label = f"{label}[{index}]"
        if not isinstance(occurrence, dict) or set(occurrence) != {
            "alignment_id",
            "sk_off_operators",
            "sk_on_dispatch_order",
        }:
            raise ValueError(f"{occurrence_label} fields are invalid")
        alignment_id = _text(
            occurrence["alignment_id"], f"{occurrence_label}.alignment_id"
        )
        if alignment_id in alignment_ids:
            raise ValueError(f"{label} has duplicate alignment_id")
        alignment_ids.add(alignment_id)
        raw_operators = occurrence["sk_off_operators"]
        if not isinstance(raw_operators, list) or len(raw_operators) < 2:
            raise ValueError(
                f"{occurrence_label}.sk_off_operators requires at least two entries"
            )
        operators = []
        mapping = {}
        for op_index, operator in enumerate(raw_operators):
            op_label = f"{occurrence_label}.sk_off_operators[{op_index}]"
            required = {
                "operator_id",
                "statement_id",
                "stream_id",
                "accelerator_core",
                "block_num",
                "mix_block_num",
                "start_us",
                "duration_us",
            }
            if not isinstance(operator, dict) or set(operator) != required:
                raise ValueError(f"{op_label} fields are invalid")
            operator_id = _text(operator["operator_id"], f"{op_label}.operator_id")
            statement_id = _text(operator["statement_id"], f"{op_label}.statement_id")
            if operator_id in mapping or statement_id not in statement_ids:
                raise ValueError(f"{op_label} identity is invalid")
            stream_id = _integer(operator["stream_id"], f"{op_label}.stream_id")
            resource = multistream_core_family.classify_profile_identity(
                operator["accelerator_core"],
                operator["block_num"],
                operator["mix_block_num"],
            )
            start = _number(operator["start_us"], f"{op_label}.start_us")
            duration = _number(
                operator["duration_us"], f"{op_label}.duration_us", positive=True
            )
            mapping[operator_id] = statement_id
            operators.append(
                {
                    **operator,
                    **resource,
                    "start_us": start,
                    "duration_us": duration,
                    "end_us": start + duration,
                    "stream_id": stream_id,
                }
            )
        if operator_to_statement is None:
            operator_to_statement = mapping
        elif mapping != operator_to_statement:
            raise ValueError(f"{label} operator-to-statement identity is unstable")
        dispatch = occurrence["sk_on_dispatch_order"]
        if not isinstance(dispatch, list) or len(dispatch) != len(mapping):
            raise ValueError(f"{occurrence_label}.sk_on_dispatch_order is invalid")
        if set(dispatch) != set(mapping) or len(dispatch) != len(set(dispatch)):
            raise ValueError(
                f"{occurrence_label}.sk_on_dispatch_order identity differs"
            )
        normalized.append(
            {
                "alignment_id": alignment_id,
                "operators": operators,
                "dispatch_order": dispatch,
            }
        )
    return normalized, operator_to_statement


def _resource_pair_kind(left_family, right_family):
    """Classify whether two independently streamed kernels can share compute resources."""
    families = frozenset((left_family, right_family))
    if families == RESOURCE_COMPLEMENTARY_FAMILIES:
        return "CUBE_VECTOR"
    if "MIX" in families:
        return "MIX_EXCLUDED"
    if left_family == right_family and left_family in RESOURCE_COMPLEMENTARY_FAMILIES:
        return "SAME_ENGINE_EXCLUDED"
    return "NON_COMPLEMENTARY_EXCLUDED"


def _analyze_target(
    target, source_records, root, label, *, request_fingerprint, max_route3_candidates
):
    required = {
        "range_id",
        "graph_occurrence_fingerprint",
        "source_file",
        "range_start_offset",
        "range_end_offset",
        "statements",
        "hard_dependencies",
        "dependency_evidence",
        "occurrences",
    }
    if not isinstance(target, dict) or set(target) != required:
        raise ValueError(f"{label} fields are invalid")
    range_id = _text(target["range_id"], f"{label}.range_id")
    occurrence_fingerprint = _text(
        target["graph_occurrence_fingerprint"], f"{label}.graph_occurrence_fingerprint"
    )
    source_file = _relative(target["source_file"], f"{label}.source_file")
    source_record = next(
        (item for item in source_records if item["path"] == source_file), None
    )
    if source_record is None:
        raise ValueError(f"{label}.source_file is not sealed in source_files")
    source_path = _rooted(root, source_file, f"{label}.source_file")
    statements = _validate_statements(target, source_path.stat().st_size, label)
    current_order = [item["statement_id"] for item in statements]
    statement_ids = set(current_order)
    range_start = _integer(target["range_start_offset"], f"{label}.range_start_offset")
    range_end = _integer(
        target["range_end_offset"], f"{label}.range_end_offset", minimum=1
    )
    if (
        range_start != statements[0]["start_offset"]
        or range_end != statements[-1]["end_offset"]
    ):
        raise ValueError(
            f"{label} range must exactly cover the contiguous statement region"
        )
    dependencies = _validate_dependencies(
        target["hard_dependencies"], statement_ids, f"{label}.hard_dependencies"
    )
    dependency_path = _rooted(
        root, target["dependency_evidence"], f"{label}.dependency_evidence"
    )
    dependency_evidence = multistream_dependency_evidence.validate(
        dependency_path,
        root,
        require_complete=True,
        request_fingerprint=request_fingerprint,
        range_id=range_id,
        graph_occurrence_fingerprint=occurrence_fingerprint,
        statement_ids=current_order,
    )
    if dependencies != dependency_evidence["hard_dependencies"]:
        raise ValueError(f"{label}.hard_dependencies differs from dependency evidence")
    hard_pairs = {(item["before"], item["after"]) for item in dependencies}
    occurrences, operator_to_statement = _validate_occurrences(
        target["occurrences"], statement_ids, f"{label}.occurrences"
    )

    pair_observations = defaultdict(list)
    for occurrence in occurrences:
        operators = {item["operator_id"]: item for item in occurrence["operators"]}
        # kernel_details.csv row order is the execution sequence. Start timestamps are
        # retained for overlap only; sub-microsecond Cube/Vector starts may tie or
        # appear reversed without changing the profiler's execution ordering.
        operator_ids = [item["operator_id"] for item in occurrence["operators"]]
        dispatch_rank = {
            item: index for index, item in enumerate(occurrence["dispatch_order"])
        }
        for left_index, left_id in enumerate(operator_ids):
            for right_id in operator_ids[left_index + 1 :]:
                left, right = operators[left_id], operators[right_id]
                first, second = left, right
                overlap = min(left["end_us"], right["end_us"]) > max(
                    left["start_us"], right["start_us"]
                )
                cross_stream = left["stream_id"] != right["stream_id"]
                pair_observations[(first["operator_id"], second["operator_id"])].append(
                    {
                        "overlap": overlap,
                        "cross_stream": cross_stream,
                        "resource_pair": _resource_pair_kind(
                            first["core_family"], second["core_family"]
                        ),
                        "dispatch_inverted": dispatch_rank[first["operator_id"]]
                        > dispatch_rank[second["operator_id"]],
                    }
                )

    required_count = len(occurrences)
    parallel_pairs = []
    resource_complementary_pairs = []
    excluded_resource_pairs = []
    preference_pairs = []
    inversions = []
    for (before_op, after_op), observations in sorted(pair_observations.items()):
        if len(observations) != required_count:
            continue
        before = operator_to_statement[before_op]
        after = operator_to_statement[after_op]
        if before == after:
            continue
        if all(item["overlap"] and item["cross_stream"] for item in observations):
            parallel_pairs.append((before, after))
            if all(item["resource_pair"] == "CUBE_VECTOR" for item in observations):
                resource_complementary_pairs.append((before, after))
                preference_pairs.append((before, after))
                if all(item["dispatch_inverted"] for item in observations):
                    inversions.append((before, after))
            else:
                excluded_resource_pairs.append(
                    {
                        "before": before,
                        "after": after,
                        "resource_pair_kinds": sorted(
                            {item["resource_pair"] for item in observations}
                        ),
                    }
                )

    blockers = []
    if not parallel_pairs:
        blockers.append("direct_multistream_evidence_missing")
    if parallel_pairs and not resource_complementary_pairs:
        blockers.append("resource_complementary_pair_missing")
        excluded_kinds = {
            kind
            for pair in excluded_resource_pairs
            for kind in pair["resource_pair_kinds"]
        }
        if "MIX_EXCLUDED" in excluded_kinds:
            blockers.append("mix_resource_pair_excluded")
        if "SAME_ENGINE_EXCLUDED" in excluded_kinds:
            blockers.append("same_engine_pair_excluded")
    if resource_complementary_pairs and not inversions:
        blockers.append("stable_dispatch_inversion_missing")
    stable_preferences = sorted(set(preference_pairs))
    conflicts = [
        pair for pair in stable_preferences if _reachable(hard_pairs, pair[1], pair[0])
    ]
    if conflicts:
        blockers.append("stable_preference_conflicts_with_hard_dependency")
    usable_preferences = [pair for pair in stable_preferences if pair not in conflicts]
    route2 = None
    if not blockers:
        route2 = _topological_order(
            statement_ids, hard_pairs, usable_preferences, current_order
        )
        if route2 is None:
            blockers.append("preferred_order_cycle")
        elif route2 == current_order:
            route2 = None
            blockers.append("source_order_already_matches_stable_preference")
    route3 = []
    if route2 is not None:
        route3 = [
            order
            for order in _bounded_orders(
                list(statement_ids),
                hard_pairs | set(usable_preferences),
                route2,
                current_order,
                maximum=max_route3_candidates + 1,
            )
            if order != route2 and order != current_order
        ][:max_route3_candidates]
    return {
        "range_id": range_id,
        "graph_occurrence_fingerprint": occurrence_fingerprint,
        "source_file": source_file,
        "source_file_fingerprint": source_record["file_fingerprint"],
        "range_start_offset": range_start,
        "range_end_offset": range_end,
        "aligned_occurrence_count": required_count,
        "current_source_order": current_order,
        "statements": statements,
        "hard_dependencies": dependencies,
        "dependency_evidence": {
            "path": dependency_path.relative_to(Path(root).resolve()).as_posix(),
            "file_fingerprint": file_fingerprint(dependency_path),
            "evidence_fingerprint": dependency_evidence["evidence_fingerprint"],
        },
        "stable_execution_preferences": [list(item) for item in stable_preferences],
        "stable_parallel_pairs": [list(item) for item in sorted(set(parallel_pairs))],
        "stable_resource_complementary_pairs": [
            list(item) for item in sorted(set(resource_complementary_pairs))
        ],
        "excluded_resource_pairs": excluded_resource_pairs,
        "stable_dispatch_inversions": [list(item) for item in sorted(set(inversions))],
        "multistream_reorder_authorized": route2 is not None,
        "route2_order": route2,
        "route3_orders": route3,
        "blockers": sorted(set(blockers)),
    }


def analyze(
    capture_path,
    artifact_root=None,
    *,
    expected_request_fingerprint=None,
    max_route3_candidates=5,
):
    capture_path = Path(capture_path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else capture_path.parent
    capture = json.loads(capture_path.read_text())
    required = {
        "schema_version",
        "capture_id",
        "request_fingerprint",
        "source_files",
        "targets",
        "capture_fingerprint",
    }
    if not isinstance(capture, dict) or set(capture) != required:
        raise ValueError(
            f"operator order capture must contain exactly {sorted(required)}"
        )
    if capture["schema_version"] != CAPTURE_SCHEMA:
        raise ValueError(f"operator order capture must use {CAPTURE_SCHEMA}")
    unsigned = {
        key: value for key, value in capture.items() if key != "capture_fingerprint"
    }
    if capture["capture_fingerprint"] != fingerprint(unsigned):
        raise ValueError("operator order capture fingerprint mismatch")
    request_fingerprint = _text(capture["request_fingerprint"], "request_fingerprint")
    if (
        expected_request_fingerprint is not None
        and request_fingerprint != expected_request_fingerprint
    ):
        raise ValueError("operator order capture request fingerprint mismatch")
    _text(capture["capture_id"], "capture_id")
    source_files = _validate_source_files(root, capture["source_files"])
    targets = capture["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("operator order capture targets must be non-empty")
    results = [
        _analyze_target(
            target,
            source_files,
            root,
            f"targets[{index}]",
            request_fingerprint=request_fingerprint,
            max_route3_candidates=max_route3_candidates,
        )
        for index, target in enumerate(targets)
    ]
    range_ids = [item["range_id"] for item in results]
    if len(range_ids) != len(set(range_ids)):
        raise ValueError("operator order capture has duplicate range_id")
    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "request_fingerprint": request_fingerprint,
        "capture": {
            "path": capture_path.relative_to(root).as_posix(),
            "file_fingerprint": file_fingerprint(capture_path),
            "capture_fingerprint": capture["capture_fingerprint"],
        },
        "thresholds": {
            "minimum_aligned_occurrences": 3,
            "direct_multistream_required": True,
            "resource_pair_policy": RESOURCE_PAIR_POLICY,
            "max_route3_candidates": max_route3_candidates,
        },
        "targets": sorted(results, key=lambda item: item["range_id"]),
    }
    analysis["analysis_fingerprint"] = fingerprint(analysis)
    return analysis


def validate_analysis(path, artifact_root=None, *, expected_request_fingerprint=None):
    path = Path(path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else path.parent
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError(f"operator order analysis must use {ANALYSIS_SCHEMA}")
    actual = value.get("analysis_fingerprint")
    if actual != fingerprint(
        {key: item for key, item in value.items() if key != "analysis_fingerprint"}
    ):
        raise ValueError("operator order analysis fingerprint mismatch")
    if (
        expected_request_fingerprint is not None
        and value.get("request_fingerprint") != expected_request_fingerprint
    ):
        raise ValueError("operator order analysis request fingerprint mismatch")
    capture = value.get("capture")
    if not isinstance(capture, dict) or set(capture) != {
        "path",
        "file_fingerprint",
        "capture_fingerprint",
    }:
        raise ValueError("operator order analysis capture binding is invalid")
    capture_path = _rooted(root, capture["path"], "analysis.capture.path")
    if file_fingerprint(capture_path) != capture["file_fingerprint"]:
        raise ValueError("operator order analysis capture file changed")
    rebuilt = analyze(
        capture_path,
        root,
        expected_request_fingerprint=value["request_fingerprint"],
        max_route3_candidates=value["thresholds"]["max_route3_candidates"],
    )
    if _canonical(rebuilt) != _canonical(value):
        raise ValueError("operator order analysis differs from deterministic replay")
    return {
        "valid": True,
        "analysis_fingerprint": actual,
        "targets": len(value["targets"]),
    }


def build_dispatch_evidence(
    action_manifest_path, dispatch_capture_path, artifact_root=None
):
    root = (
        Path(artifact_root).resolve()
        if artifact_root
        else Path(action_manifest_path).resolve().parent
    )
    action_path = Path(action_manifest_path).resolve()
    capture_path = Path(dispatch_capture_path).resolve()
    action = json.loads(action_path.read_text())
    if action.get("change_kind") != "dependency_safe_operator_reorder":
        raise ValueError("dispatch verification requires an operator reorder action")
    if action.get("multistream_only_verified") is not True:
        raise ValueError("operator reorder action lacks multistream-only verification")
    capture = json.loads(capture_path.read_text())
    required = {
        "schema_version",
        "trial_id",
        "request_fingerprint",
        "action_manifest_fingerprint",
        "range_id",
        "child_set_preserved",
        "stream_identity_complete",
        "occurrences",
        "source_files",
        "capture_fingerprint",
    }
    if not isinstance(capture, dict) or set(capture) != required:
        raise ValueError(
            f"post-reorder dispatch capture must contain exactly {sorted(required)}"
        )
    if capture["schema_version"] != DISPATCH_CAPTURE_SCHEMA:
        raise ValueError(
            f"post-reorder dispatch capture must use {DISPATCH_CAPTURE_SCHEMA}"
        )
    if capture["capture_fingerprint"] != fingerprint(
        {key: item for key, item in capture.items() if key != "capture_fingerprint"}
    ):
        raise ValueError("post-reorder dispatch capture fingerprint mismatch")
    if capture["trial_id"] != action.get("trial_id"):
        raise ValueError("post-reorder dispatch trial_id differs from action")
    if capture["action_manifest_fingerprint"] != fingerprint(action):
        raise ValueError("post-reorder dispatch action fingerprint mismatch")
    change = action["only_change"]
    if capture["range_id"] != change["range_id"]:
        raise ValueError("post-reorder dispatch range_id differs from action")
    if capture["child_set_preserved"] is not True:
        raise ValueError("post-reorder dispatch child set changed")
    if capture["stream_identity_complete"] is not True:
        raise ValueError("post-reorder dispatch stream identity is incomplete")
    source_files = _validate_source_files(root, capture["source_files"])
    occurrences = capture["occurrences"]
    if not isinstance(occurrences, list) or len(occurrences) < 3:
        raise ValueError("post-reorder dispatch requires at least three occurrences")
    expected = change["after_order"]
    normalized = []
    seen = set()
    for index, occurrence in enumerate(occurrences):
        label = f"post-reorder dispatch occurrences[{index}]"
        if not isinstance(occurrence, dict) or set(occurrence) != {
            "alignment_id",
            "observed_statement_order",
            "stream_ids",
        }:
            raise ValueError(f"{label} fields are invalid")
        alignment_id = _text(occurrence["alignment_id"], f"{label}.alignment_id")
        if alignment_id in seen:
            raise ValueError("post-reorder dispatch has duplicate alignment_id")
        seen.add(alignment_id)
        order = occurrence["observed_statement_order"]
        streams = occurrence["stream_ids"]
        if order != expected:
            raise ValueError(
                "post-reorder dispatch order does not match the planned order"
            )
        if (
            not isinstance(streams, list)
            or len(streams) != len(expected)
            or len(set(streams)) < 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in streams
            )
        ):
            raise ValueError(
                "post-reorder dispatch occurrence lacks one stream identity per statement "
                "or is not multistream"
            )
        normalized.append(
            {
                "alignment_id": alignment_id,
                "observed_statement_order": order,
                "stream_ids": streams,
            }
        )
    evidence = {
        "schema_version": DISPATCH_EVIDENCE_SCHEMA,
        "trial_id": action["trial_id"],
        "request_fingerprint": _text(
            capture["request_fingerprint"], "request_fingerprint"
        ),
        "range_id": change["range_id"],
        "action_manifest": action_path.relative_to(root).as_posix(),
        "action_manifest_fingerprint": fingerprint(action),
        "dispatch_capture": capture_path.relative_to(root).as_posix(),
        "dispatch_capture_fingerprint": capture["capture_fingerprint"],
        "source_files": source_files,
        "aligned_occurrence_count": len(normalized),
        "expected_statement_order": expected,
        "child_set_preserved": True,
        "stream_identity_complete": True,
        "decision": "pass",
    }
    evidence["evidence_fingerprint"] = fingerprint(evidence)
    return evidence


def validate_dispatch_evidence(
    path, artifact_root=None, *, trial_id=None, request_fingerprint=None
):
    path = Path(path).resolve()
    root = Path(artifact_root).resolve() if artifact_root else path.parent
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != DISPATCH_EVIDENCE_SCHEMA
    ):
        raise ValueError(f"dispatch evidence must use {DISPATCH_EVIDENCE_SCHEMA}")
    if value.get("evidence_fingerprint") != fingerprint(
        {key: item for key, item in value.items() if key != "evidence_fingerprint"}
    ):
        raise ValueError("dispatch evidence fingerprint mismatch")
    if trial_id is not None and value.get("trial_id") != trial_id:
        raise ValueError("dispatch evidence trial_id mismatch")
    if (
        request_fingerprint is not None
        and value.get("request_fingerprint") != request_fingerprint
    ):
        raise ValueError("dispatch evidence request fingerprint mismatch")
    action_path = _rooted(
        root, value["action_manifest"], "dispatch evidence action_manifest"
    )
    capture_path = _rooted(root, value["dispatch_capture"], "dispatch evidence capture")
    rebuilt = build_dispatch_evidence(action_path, capture_path, root)
    if _canonical(rebuilt) != _canonical(value):
        raise ValueError("dispatch evidence differs from deterministic replay")
    return {
        "valid": True,
        "decision": "pass",
        "aligned_occurrence_count": value["aligned_occurrence_count"],
    }


def validate_route2_no_gain_evidence(
    path, artifact_root, *, trial_id, request_fingerprint
):
    """Accept only a stable, non-regressing clean3 miss as route3 authorization."""
    import multistream_evidence

    clean = multistream_evidence.validate_evidence(
        path,
        artifact_root,
        trial_id=trial_id,
        request_fingerprint=request_fingerprint,
        state_after="clean3_passed",
    )
    if clean["decision"] != "reject":
        raise ValueError("route3 requires rejected route2 clean3 evidence")
    try:
        evaluation = clean["semantic_result"]["candidate"]["option_trial_evaluation"]
        checks = evaluation["checks"]
        mean_improvement = evaluation["mean_improvement_pct"]
        median_improvement = evaluation["median_run_improvement_pct"]
    except (KeyError, TypeError) as error:
        raise ValueError("route2 clean3 evidence lacks promotion checks") from error
    required_passes = {
        "baseline_stable",
        "baseline_min_runs",
        "candidate_min_runs",
        "p90_no_material_regression",
        "stddev_no_material_regression",
    }
    if any(checks.get(name) is not True for name in required_passes):
        raise ValueError(
            "route3 cannot follow unstable or tail/variance-regressing route2"
        )
    if checks.get("mean_improvement") is not False:
        raise ValueError(
            "route3 requires route2 to miss the clean improvement threshold"
        )
    if (
        isinstance(mean_improvement, bool)
        or not isinstance(mean_improvement, (int, float))
        or mean_improvement < 0
        or isinstance(median_improvement, bool)
        or not isinstance(median_improvement, (int, float))
        or median_improvement < 0
    ):
        raise ValueError("route3 cannot follow a mean or median clean regression")
    return clean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("analyze")
    create.add_argument("--capture", type=Path, required=True)
    create.add_argument("--artifact-root", type=Path)
    create.add_argument("--request-fingerprint")
    create.add_argument("--max-route3-candidates", type=int, default=5)
    create.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--analysis", type=Path, required=True)
    validate.add_argument("--artifact-root", type=Path)
    validate.add_argument("--request-fingerprint")
    dispatch = commands.add_parser("verify-dispatch")
    dispatch.add_argument("--action-manifest", type=Path, required=True)
    dispatch.add_argument("--dispatch-capture", type=Path, required=True)
    dispatch.add_argument("--artifact-root", type=Path)
    dispatch.add_argument("--out", type=Path, required=True)
    validate_dispatch = commands.add_parser("validate-dispatch")
    validate_dispatch.add_argument("--evidence", type=Path, required=True)
    validate_dispatch.add_argument("--artifact-root", type=Path)
    validate_dispatch.add_argument("--trial-id")
    validate_dispatch.add_argument("--request-fingerprint")
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            result = analyze(
                args.capture,
                args.artifact_root,
                expected_request_fingerprint=args.request_fingerprint,
                max_route3_candidates=args.max_route3_candidates,
            )
            if args.out.exists():
                raise ValueError(
                    f"operator order analysis output already exists: {args.out}"
                )
            _atomic_json(args.out, result)
        elif args.command == "validate":
            result = validate_analysis(
                args.analysis,
                args.artifact_root,
                expected_request_fingerprint=args.request_fingerprint,
            )
        elif args.command == "verify-dispatch":
            result = build_dispatch_evidence(
                args.action_manifest, args.dispatch_capture, args.artifact_root
            )
            if args.out.exists():
                raise ValueError(f"dispatch evidence output already exists: {args.out}")
            _atomic_json(args.out, result)
        else:
            result = validate_dispatch_evidence(
                args.evidence,
                args.artifact_root,
                trial_id=args.trial_id,
                request_fingerprint=args.request_fingerprint,
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
