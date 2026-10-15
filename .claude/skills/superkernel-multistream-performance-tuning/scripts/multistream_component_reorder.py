#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate and materialize reviewed MIX-component cross-function reorders."""

import argparse
import ast
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


SOURCE_MAP_SCHEMA = "superkernel-multistream-component-source-map-v1"
CAPTURE_SCHEMA = "superkernel-multistream-component-order-capture-v1"
TRANSFORM_SCHEMA = "superkernel-multistream-component-source-transform-v1"
REVIEW_SCHEMA = "superkernel-multistream-component-review-v1"
AUDIT_SCHEMA = "superkernel-multistream-component-post-transform-audit-v1"
DISPATCH_CAPTURE_SCHEMA = "superkernel-multistream-component-dispatch-capture-v1"
DISPATCH_EVIDENCE_SCHEMA = "superkernel-multistream-component-dispatch-evidence-v1"
ACTION_MANIFEST_SCHEMA = "superkernel-multistream-action-manifest-v2"
CHANGE_KIND = "component_overlap_reorder"
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
SPAN_RELATIONS = {
    "operator_call",
    "producer",
    "synchronization",
    "join",
    "transform_region",
}
COMPONENT_ENGINES = {"AIC", "AIV"}
SAFETY_PROOFS = {
    "all_components_bound",
    "hard_dependency_dag_permits",
    "events_waits_preserved",
    "communication_preserved",
    "cache_mutation_preserved",
    "control_flow_preserved",
    "single_stream_projection_preserved",
    "child_set_expected_unchanged",
    "reviewed_adapter_transform",
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


def bytes_fingerprint(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _sha(value, label):
    value = _text(value, label)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 fingerprint")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a sha256 fingerprint") from error
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


def _rooted(root, relative, label):
    root = Path(root).resolve()
    path = (root / _relative(relative, label)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _line_offsets(raw):
    offsets = [0]
    for index, value in enumerate(raw):
        if value == 10:
            offsets.append(index + 1)
    return offsets


def _node_bounds(node, offsets, raw_size):
    if not all(
        hasattr(node, field)
        for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        return None
    start = offsets[node.lineno - 1] + node.col_offset
    end_line = node.end_lineno - 1
    if end_line >= len(offsets):
        return None
    end = offsets[end_line] + node.end_col_offset
    if not 0 <= start < end <= raw_size:
        return None
    return start, end


class _QualifiedNodes(ast.NodeVisitor):
    def __init__(self, offsets, raw_size):
        self.offsets = offsets
        self.raw_size = raw_size
        self.scope = []
        self.records = []

    def _record(self, node):
        bounds = _node_bounds(node, self.offsets, self.raw_size)
        if bounds is not None:
            self.records.append(
                {
                    "function_qualname": ".".join(self.scope),
                    "node_type": type(node).__name__,
                    "start_offset": bounds[0],
                    "end_offset": bounds[1],
                }
            )

    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            self._record(node)
        super().generic_visit(node)

    def visit_ClassDef(self, node):
        self._record(node)
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    def _visit_function(self, node):
        self.scope.append(node.name)
        self._record(node)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function


def _ast_records(path):
    raw = Path(path).read_bytes()
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"source map requires valid UTF-8 Python: {error}") from error
    visitor = _QualifiedNodes(_line_offsets(raw), len(raw))
    visitor.visit(tree)
    return raw, visitor.records


def _validate_source_files(root, records):
    if not isinstance(records, list) or not records:
        raise ValueError("component source map requires source_files")
    normalized = []
    seen = set()
    for index, item in enumerate(records):
        label = f"source_files[{index}]"
        if not isinstance(item, dict) or set(item) != {"path", "file_fingerprint"}:
            raise ValueError(f"{label} fields are invalid")
        path = _rooted(root, item["path"], f"{label}.path")
        relative = path.relative_to(Path(root).resolve()).as_posix()
        if relative in seen:
            raise ValueError(f"duplicate source file: {relative}")
        seen.add(relative)
        actual = file_fingerprint(path)
        if item["file_fingerprint"] != actual:
            raise ValueError(f"{label}.file_fingerprint mismatch")
        normalized.append({"path": relative, "file_fingerprint": actual})
    return sorted(normalized, key=lambda item: item["path"])


def validate_source_map(value, source_root):
    required = {
        "schema_version",
        "source_revision",
        "target_set_id",
        "source_files",
        "spans",
        "target_bindings",
        "hard_dependencies",
        "dependency_coverage",
        "mapping_fingerprint",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != SOURCE_MAP_SCHEMA
    ):
        raise ValueError(
            f"component source map must use {SOURCE_MAP_SCHEMA} with exact fields"
        )
    if value["mapping_fingerprint"] != fingerprint(
        {key: item for key, item in value.items() if key != "mapping_fingerprint"}
    ):
        raise ValueError("component source map fingerprint mismatch")
    source_revision = _text(value["source_revision"], "source_revision")
    target_set_id = _text(value["target_set_id"], "target_set_id")
    source_files = _validate_source_files(source_root, value["source_files"])
    file_records = {item["path"]: item for item in source_files}
    ast_cache = {}
    spans = value["spans"]
    if not isinstance(spans, list) or len(spans) < 2:
        raise ValueError("component source map requires at least two exact spans")
    normalized_spans = []
    span_ids = set()
    for index, item in enumerate(spans):
        label = f"spans[{index}]"
        required_span = {
            "span_id",
            "source_file",
            "function_qualname",
            "node_type",
            "start_offset",
            "end_offset",
            "before_fingerprint",
            "relation",
            "operator_ids",
        }
        if not isinstance(item, dict) or set(item) != required_span:
            raise ValueError(f"{label} fields are invalid")
        span_id = _text(item["span_id"], f"{label}.span_id")
        if span_id in span_ids:
            raise ValueError(f"duplicate span_id: {span_id}")
        span_ids.add(span_id)
        source_file = _relative(item["source_file"], f"{label}.source_file")
        if source_file not in file_records:
            raise ValueError(f"{label}.source_file is not sealed")
        function = _text(item["function_qualname"], f"{label}.function_qualname")
        node_type = _text(item["node_type"], f"{label}.node_type")
        relation = _text(item["relation"], f"{label}.relation")
        if relation not in SPAN_RELATIONS:
            raise ValueError(f"{label}.relation is unsupported")
        start = _integer(item["start_offset"], f"{label}.start_offset")
        end = _integer(item["end_offset"], f"{label}.end_offset", minimum=1)
        if source_file not in ast_cache:
            ast_cache[source_file] = _ast_records(
                _rooted(source_root, source_file, source_file)
            )
        raw, nodes = ast_cache[source_file]
        if not 0 <= start < end <= len(raw):
            raise ValueError(f"{label} byte range is invalid")
        if not any(
            record
            == {
                "function_qualname": function,
                "node_type": node_type,
                "start_offset": start,
                "end_offset": end,
            }
            for record in nodes
        ):
            raise ValueError(f"{label} is not an exact AST node span")
        before = bytes_fingerprint(raw[start:end])
        if item["before_fingerprint"] != before:
            raise ValueError(f"{label}.before_fingerprint mismatch")
        operators = item["operator_ids"]
        if (
            not isinstance(operators, list)
            or operators != sorted(set(operators))
            or any(
                not isinstance(operator, str) or not operator.strip()
                for operator in operators
            )
        ):
            raise ValueError(f"{label}.operator_ids must be sorted unique strings")
        if relation == "operator_call" and not operators:
            raise ValueError(f"{label} operator_call requires operator_ids")
        normalized_spans.append({**item, "source_file": source_file})

    bindings = value["target_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("component source map requires target_bindings")
    normalized_bindings = []
    range_ids = set()
    for index, item in enumerate(bindings):
        label = f"target_bindings[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "range_id",
            "graph_occurrence_fingerprint",
            "operator_spans",
        }:
            raise ValueError(f"{label} fields are invalid")
        range_id = _text(item["range_id"], f"{label}.range_id")
        if range_id in range_ids:
            raise ValueError(f"duplicate range_id: {range_id}")
        range_ids.add(range_id)
        _text(
            item["graph_occurrence_fingerprint"],
            f"{label}.graph_occurrence_fingerprint",
        )
        operator_spans = item["operator_spans"]
        if not isinstance(operator_spans, dict) or len(operator_spans) < 2:
            raise ValueError(f"{label}.operator_spans requires at least two operators")
        for operator, span_id in operator_spans.items():
            _text(operator, f"{label}.operator_spans key")
            if span_id not in span_ids:
                raise ValueError(f"{label}.operator_spans references unknown span")
        normalized_bindings.append(dict(item))

    dependencies = value["hard_dependencies"]
    if not isinstance(dependencies, list):
        raise ValueError("hard_dependencies must be a list")
    normalized_dependencies = []
    edge_pairs = set()
    for index, edge in enumerate(dependencies):
        label = f"hard_dependencies[{index}]"
        if not isinstance(edge, dict) or set(edge) != {"before", "after", "kind"}:
            raise ValueError(f"{label} fields are invalid")
        before = _text(edge["before"], f"{label}.before")
        after = _text(edge["after"], f"{label}.after")
        kind = _text(edge["kind"], f"{label}.kind").upper()
        if before not in span_ids or after not in span_ids or before == after:
            raise ValueError(f"{label} references an invalid span")
        if kind not in HARD_DEPENDENCY_KINDS:
            raise ValueError(f"{label}.kind is unsupported")
        if (before, after) in edge_pairs:
            raise ValueError(f"duplicate hard dependency pair: {before}->{after}")
        edge_pairs.add((before, after))
        normalized_dependencies.append({"before": before, "after": after, "kind": kind})
    children = defaultdict(set)
    incoming = {span_id: 0 for span_id in span_ids}
    for before, after in edge_pairs:
        if after not in children[before]:
            children[before].add(after)
            incoming[after] += 1
    ready = sorted(span_id for span_id, count in incoming.items() if count == 0)
    visited = []
    while ready:
        current = ready.pop(0)
        visited.append(current)
        for child in sorted(children[current]):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
                ready.sort()
    if len(visited) != len(span_ids):
        raise ValueError("component source map hard dependencies contain a cycle")

    coverage = value["dependency_coverage"]
    if not isinstance(coverage, dict) or set(coverage) != {"covered_kinds", "complete"}:
        raise ValueError("dependency_coverage fields are invalid")
    if coverage["complete"] is not True or coverage["covered_kinds"] != sorted(
        HARD_DEPENDENCY_KINDS
    ):
        raise ValueError("component source map dependency coverage is incomplete")
    return {
        **value,
        "source_revision": source_revision,
        "target_set_id": target_set_id,
        "source_files": source_files,
        "spans": normalized_spans,
        "target_bindings": normalized_bindings,
        "hard_dependencies": normalized_dependencies,
    }


def validate_capture(value, artifact_root, source_map):
    required = {
        "schema_version",
        "request_fingerprint",
        "target_set_id",
        "source_map",
        "source_map_fingerprint",
        "component_policy",
        "targets",
        "source_files",
        "capture_fingerprint",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != CAPTURE_SCHEMA
    ):
        raise ValueError(
            f"component capture must use {CAPTURE_SCHEMA} with exact fields"
        )
    if value["capture_fingerprint"] != fingerprint(
        {key: item for key, item in value.items() if key != "capture_fingerprint"}
    ):
        raise ValueError("component capture fingerprint mismatch")
    _text(value["request_fingerprint"], "request_fingerprint")
    if value["target_set_id"] != source_map["target_set_id"]:
        raise ValueError("component capture target_set_id differs from source map")
    if value["source_map_fingerprint"] != source_map["mapping_fingerprint"]:
        raise ValueError("component capture source map fingerprint mismatch")
    _validate_source_files(artifact_root, value["source_files"])
    policy = value["component_policy"]
    required_policy = {
        "mix_statement_id",
        "complementary_statement_id",
        "mix_overlap_engine",
        "complementary_engine",
        "mix_same_engine",
        "expected_component_counts",
        "before_order",
        "after_order",
    }
    if not isinstance(policy, dict) or set(policy) != required_policy:
        raise ValueError("component_policy fields are invalid")
    mix_id = _text(policy["mix_statement_id"], "mix_statement_id")
    complement_id = _text(
        policy["complementary_statement_id"], "complementary_statement_id"
    )
    if mix_id == complement_id:
        raise ValueError("component policy requires two statements")
    for field in ("mix_overlap_engine", "complementary_engine", "mix_same_engine"):
        if policy[field] not in COMPONENT_ENGINES:
            raise ValueError(f"component_policy.{field} is invalid")
    if policy["mix_overlap_engine"] == policy["mix_same_engine"]:
        raise ValueError("MIX overlap and same-engine components must differ")
    if policy["before_order"] != [mix_id, complement_id] or policy["after_order"] != [
        complement_id,
        mix_id,
    ]:
        raise ValueError(
            "component policy must reverse MIX/complementary dispatch order"
        )
    counts = policy["expected_component_counts"]
    if not isinstance(counts, dict) or set(counts) != {mix_id, complement_id}:
        raise ValueError(
            "component policy expected_component_counts statement set is invalid"
        )
    expected_engines = {
        mix_id: COMPONENT_ENGINES,
        complement_id: {policy["complementary_engine"]},
    }
    for statement_id, engines in expected_engines.items():
        statement_counts = counts[statement_id]
        if not isinstance(statement_counts, dict) or set(statement_counts) != engines:
            raise ValueError(
                "component policy expected_component_counts engine set is invalid"
            )
        for engine, count in statement_counts.items():
            _integer(
                count, f"expected_component_counts.{statement_id}.{engine}", minimum=1
            )

    expected_ranges = {item["range_id"] for item in source_map["target_bindings"]}
    targets = value["targets"]
    if (
        not isinstance(targets, list)
        or {item.get("range_id") for item in targets if isinstance(item, dict)}
        != expected_ranges
    ):
        raise ValueError("component capture targets differ from source map")
    total_occurrences = 0
    overlap_values = []
    contention_values = []
    for target_index, target in enumerate(targets):
        label = f"targets[{target_index}]"
        if not isinstance(target, dict) or set(target) != {"range_id", "occurrences"}:
            raise ValueError(f"{label} fields are invalid")
        occurrences = target["occurrences"]
        if not isinstance(occurrences, list) or len(occurrences) < 3:
            raise ValueError(f"{label} requires at least three occurrences")
        seen = set()
        for occurrence_index, occurrence in enumerate(occurrences):
            occurrence_label = f"{label}.occurrences[{occurrence_index}]"
            if not isinstance(occurrence, dict) or set(occurrence) != {
                "alignment_id",
                "dispatch_order",
                "operators",
            }:
                raise ValueError(f"{occurrence_label} fields are invalid")
            alignment_id = _text(
                occurrence["alignment_id"], f"{occurrence_label}.alignment_id"
            )
            if alignment_id in seen:
                raise ValueError(f"{label} contains duplicate alignment_id")
            seen.add(alignment_id)
            if occurrence["dispatch_order"] != policy["before_order"]:
                raise ValueError(
                    "component capture does not prove the dispatch inversion"
                )
            operators = occurrence["operators"]
            if not isinstance(operators, list) or len(operators) != 2:
                raise ValueError(
                    f"{occurrence_label}.operators must contain exactly two operators"
                )
            by_statement = {}
            for operator_index, operator in enumerate(operators):
                operator_label = f"{occurrence_label}.operators[{operator_index}]"
                if not isinstance(operator, dict) or set(operator) != {
                    "operator_id",
                    "statement_id",
                    "kernel_type",
                    "stream_id",
                    "components",
                }:
                    raise ValueError(f"{operator_label} fields are invalid")
                statement_id = _text(
                    operator["statement_id"], f"{operator_label}.statement_id"
                )
                if statement_id in by_statement:
                    raise ValueError(f"{occurrence_label} has duplicate statement_id")
                _text(operator["operator_id"], f"{operator_label}.operator_id")
                kernel_type = _text(
                    operator["kernel_type"], f"{operator_label}.kernel_type"
                )
                stream_id = _integer(
                    operator["stream_id"], f"{operator_label}.stream_id"
                )
                components = operator["components"]
                if not isinstance(components, list) or not components:
                    raise ValueError(f"{operator_label}.components is empty")
                normalized_components = []
                engines = set()
                for component_index, component in enumerate(components):
                    component_label = f"{operator_label}.components[{component_index}]"
                    if not isinstance(component, dict) or set(component) != {
                        "component_id",
                        "engine",
                        "start_us",
                        "duration_us",
                    }:
                        raise ValueError(f"{component_label} fields are invalid")
                    engine = component["engine"]
                    if engine not in COMPONENT_ENGINES:
                        raise ValueError(f"{component_label}.engine is invalid")
                    engines.add(engine)
                    start = _number(
                        component["start_us"], f"{component_label}.start_us"
                    )
                    duration = _number(
                        component["duration_us"],
                        f"{component_label}.duration_us",
                        positive=True,
                    )
                    normalized_components.append(
                        {
                            **component,
                            "start_us": start,
                            "duration_us": duration,
                            "end_us": start + duration,
                        }
                    )
                if statement_id == mix_id and (
                    not kernel_type.startswith("MIX") or engines != COMPONENT_ENGINES
                ):
                    raise ValueError("MIX statement lacks complete AIC/AIV components")
                if (
                    statement_id == complement_id
                    and policy["complementary_engine"] not in engines
                ):
                    raise ValueError(
                        "complementary statement lacks its required component"
                    )
                by_statement[statement_id] = {
                    **operator,
                    "stream_id": stream_id,
                    "components": normalized_components,
                }
            if set(by_statement) != {mix_id, complement_id}:
                raise ValueError("component capture statement set differs from policy")
            if (
                by_statement[mix_id]["stream_id"]
                == by_statement[complement_id]["stream_id"]
            ):
                raise ValueError("component capture pair is not on distinct streams")
            by_engine = {}
            for statement_id, operator in by_statement.items():
                grouped = defaultdict(list)
                for component in operator["components"]:
                    grouped[component["engine"]].append(component)
                actual_counts = {
                    engine: len(items) for engine, items in grouped.items()
                }
                if actual_counts != counts[statement_id]:
                    raise ValueError(
                        f"component capture count mismatch for {statement_id}: "
                        f"{actual_counts} != {counts[statement_id]}"
                    )
                by_engine[statement_id] = {
                    engine: {
                        "start_us": min(item["start_us"] for item in items),
                        "end_us": max(item["end_us"] for item in items),
                        "duration_us": max(item["end_us"] for item in items)
                        - min(item["start_us"] for item in items),
                    }
                    for engine, items in grouped.items()
                }
            mix_overlap = by_engine[mix_id][policy["mix_overlap_engine"]]
            complement = by_engine[complement_id][policy["complementary_engine"]]
            mix_same = by_engine[mix_id][policy["mix_same_engine"]]
            overlap = max(
                0.0,
                min(mix_overlap["end_us"], complement["end_us"])
                - max(mix_overlap["start_us"], complement["start_us"]),
            )
            contention = max(
                0.0,
                min(mix_same["end_us"], complement["end_us"])
                - max(mix_same["start_us"], complement["start_us"]),
            )
            if overlap != 0.0:
                raise ValueError(
                    "component capture does not prove zero current complementary overlap"
                )
            if complement["duration_us"] <= mix_overlap["duration_us"]:
                raise ValueError(
                    "component capture does not prove longer complementary window first"
                )
            overlap_values.append(overlap)
            contention_values.append(contention)
            total_occurrences += 1
    return {
        "valid": True,
        "authorized": True,
        "target_count": len(targets),
        "occurrence_count": total_occurrences,
        "before_order": policy["before_order"],
        "after_order": policy["after_order"],
        "current_complementary_overlap_max_us": max(overlap_values),
        "current_same_engine_contention_max_us": max(contention_values),
    }


def validate_transform(value, source_map, capture, source_root, artifact_root):
    required = {
        "schema_version",
        "trial_id",
        "request_fingerprint",
        "target_set_id",
        "source_map",
        "source_map_fingerprint",
        "component_capture",
        "component_capture_fingerprint",
        "source_revision",
        "review_policy_id",
        "review_record",
        "reviewer_fingerprint",
        "single_stream_projection_fingerprint_before",
        "single_stream_projection_fingerprint_after",
        "dependency_contract_fingerprint_before",
        "dependency_contract_fingerprint_after",
        "post_transform_audit",
        "post_transform_audit_fingerprint",
        "safety_proofs",
        "replacements",
        "transform_fingerprint",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != TRANSFORM_SCHEMA
    ):
        raise ValueError(
            f"component transform must use {TRANSFORM_SCHEMA} with exact fields"
        )
    if value["transform_fingerprint"] != fingerprint(
        {key: item for key, item in value.items() if key != "transform_fingerprint"}
    ):
        raise ValueError("component transform fingerprint mismatch")
    _text(value["trial_id"], "trial_id")
    _text(value["request_fingerprint"], "request_fingerprint")
    if (
        value["target_set_id"] != source_map["target_set_id"]
        or value["source_revision"] != source_map["source_revision"]
    ):
        raise ValueError("component transform source/target identity mismatch")
    if value["source_map_fingerprint"] != source_map["mapping_fingerprint"]:
        raise ValueError("component transform source map fingerprint mismatch")
    if value["component_capture_fingerprint"] != capture["capture_fingerprint"]:
        raise ValueError("component transform capture fingerprint mismatch")
    if value["request_fingerprint"] != capture["request_fingerprint"]:
        raise ValueError("component transform request fingerprint mismatch")
    review_policy_id = _text(value["review_policy_id"], "review_policy_id")
    before_projection = _sha(
        value["single_stream_projection_fingerprint_before"],
        "single_stream_projection_fingerprint_before",
    )
    after_projection = _sha(
        value["single_stream_projection_fingerprint_after"],
        "single_stream_projection_fingerprint_after",
    )
    if before_projection != after_projection:
        raise ValueError("component transform must preserve single-stream projection")
    before_dependency = _sha(
        value["dependency_contract_fingerprint_before"],
        "dependency_contract_fingerprint_before",
    )
    after_dependency = _sha(
        value["dependency_contract_fingerprint_after"],
        "dependency_contract_fingerprint_after",
    )
    if before_dependency != after_dependency:
        raise ValueError(
            "component transform must preserve the hard dependency contract"
        )
    proofs = value["safety_proofs"]
    if (
        not isinstance(proofs, dict)
        or set(proofs) != SAFETY_PROOFS
        or any(item is not True for item in proofs.values())
    ):
        raise ValueError("component transform lacks complete safety proofs")
    capture_validation = validate_capture(capture, artifact_root, source_map)
    if capture_validation["authorized"] is not True:
        raise ValueError("component capture does not authorize a transform")

    spans = {item["span_id"]: item for item in source_map["spans"]}
    replacements = value["replacements"]
    if not isinstance(replacements, list) or len(replacements) < 2:
        raise ValueError("component transform requires at least two replacement hunks")
    normalized = []
    functions = set()
    source_files = set()
    previous_by_file = {}
    for index, replacement in enumerate(
        sorted(
            replacements,
            key=lambda item: (
                item.get("source_file", ""),
                item.get("start_offset", -1),
            ),
        )
    ):
        label = f"replacements[{index}]"
        fields = {
            "span_id",
            "source_file",
            "start_offset",
            "end_offset",
            "before_fingerprint",
            "replacement",
            "after_fingerprint",
        }
        if not isinstance(replacement, dict) or set(replacement) != fields:
            raise ValueError(f"{label} fields are invalid")
        span_id = _text(replacement["span_id"], f"{label}.span_id")
        span = spans.get(span_id)
        if span is None or span["relation"] != "transform_region":
            raise ValueError(f"{label} must reference a transform_region span")
        source_file = _relative(replacement["source_file"], f"{label}.source_file")
        start = _integer(replacement["start_offset"], f"{label}.start_offset")
        end = _integer(replacement["end_offset"], f"{label}.end_offset", minimum=1)
        if (source_file, start, end, replacement["before_fingerprint"]) != (
            span["source_file"],
            span["start_offset"],
            span["end_offset"],
            span["before_fingerprint"],
        ):
            raise ValueError(f"{label} differs from exact source map span")
        source = _rooted(source_root, source_file, f"{label}.source_file")
        raw = source.read_bytes()
        if bytes_fingerprint(raw[start:end]) != replacement["before_fingerprint"]:
            raise ValueError(f"{label}.before_fingerprint differs from source")
        text = replacement["replacement"]
        if not isinstance(text, str) or not text:
            raise ValueError(f"{label}.replacement must be non-empty text")
        after = bytes_fingerprint(text.encode("utf-8"))
        if (
            replacement["after_fingerprint"] != after
            or after == replacement["before_fingerprint"]
        ):
            raise ValueError(f"{label}.after_fingerprint is invalid or unchanged")
        if start < previous_by_file.get(source_file, -1):
            raise ValueError("component transform replacement hunks overlap")
        previous_by_file[source_file] = end
        functions.add(span["function_qualname"])
        source_files.add(source_file)
        normalized.append(dict(replacement))
    if len(functions) < 2:
        raise ValueError("component transform must be cross-function")
    if len(source_files) != 1:
        raise ValueError(
            "component transform v1 requires all hunks in one sealed source file"
        )
    replacement_plan_fingerprint = fingerprint(normalized)
    review_path = _rooted(artifact_root, value["review_record"], "review_record")
    review = json.loads(review_path.read_text())
    review_required = {
        "schema_version",
        "review_policy_id",
        "allowed_change_kind",
        "source_revision",
        "target_set_id",
        "source_map_fingerprint",
        "component_capture_fingerprint",
        "replacement_plan_fingerprint",
        "approved_safety_proofs",
        "review_fingerprint",
    }
    if (
        not isinstance(review, dict)
        or set(review) != review_required
        or review.get("schema_version") != REVIEW_SCHEMA
    ):
        raise ValueError(f"component review must use {REVIEW_SCHEMA} with exact fields")
    if review["review_fingerprint"] != fingerprint(
        {key: item for key, item in review.items() if key != "review_fingerprint"}
    ):
        raise ValueError("component review fingerprint mismatch")
    if value["reviewer_fingerprint"] != file_fingerprint(review_path):
        raise ValueError("component transform reviewer fingerprint mismatch")
    review_expected = {
        "review_policy_id": review_policy_id,
        "allowed_change_kind": CHANGE_KIND,
        "source_revision": source_map["source_revision"],
        "target_set_id": source_map["target_set_id"],
        "source_map_fingerprint": source_map["mapping_fingerprint"],
        "component_capture_fingerprint": capture["capture_fingerprint"],
        "replacement_plan_fingerprint": replacement_plan_fingerprint,
        "approved_safety_proofs": proofs,
    }
    for field, expected in review_expected.items():
        if review.get(field) != expected:
            raise ValueError(f"component review {field} differs from transform")

    source_file = next(iter(source_files))
    raw = _rooted(source_root, source_file, "transform source_file").read_bytes()
    cursor = 0
    chunks = []
    for replacement in normalized:
        chunks.append(raw[cursor : replacement["start_offset"]])
        chunks.append(replacement["replacement"].encode("utf-8"))
        cursor = replacement["end_offset"]
    chunks.append(raw[cursor:])
    expected_output_fingerprint = bytes_fingerprint(b"".join(chunks))
    audit_path = _rooted(
        artifact_root, value["post_transform_audit"], "post_transform_audit"
    )
    audit = json.loads(audit_path.read_text())
    audit_required = {
        "schema_version",
        "source_revision",
        "target_set_id",
        "source_map_fingerprint",
        "component_capture_fingerprint",
        "replacement_plan_fingerprint",
        "expected_output_fingerprint",
        "single_stream_projection_fingerprint_before",
        "single_stream_projection_fingerprint_after",
        "dependency_contract_fingerprint_before",
        "dependency_contract_fingerprint_after",
        "safety_proofs",
        "audit_fingerprint",
    }
    if (
        not isinstance(audit, dict)
        or set(audit) != audit_required
        or audit.get("schema_version") != AUDIT_SCHEMA
    ):
        raise ValueError(
            f"post-transform audit must use {AUDIT_SCHEMA} with exact fields"
        )
    if audit["audit_fingerprint"] != fingerprint(
        {key: item for key, item in audit.items() if key != "audit_fingerprint"}
    ):
        raise ValueError("post-transform audit fingerprint mismatch")
    if value["post_transform_audit_fingerprint"] != file_fingerprint(audit_path):
        raise ValueError(
            "component transform post-transform audit fingerprint mismatch"
        )
    audit_expected = {
        "source_revision": source_map["source_revision"],
        "target_set_id": source_map["target_set_id"],
        "source_map_fingerprint": source_map["mapping_fingerprint"],
        "component_capture_fingerprint": capture["capture_fingerprint"],
        "replacement_plan_fingerprint": replacement_plan_fingerprint,
        "expected_output_fingerprint": expected_output_fingerprint,
        "single_stream_projection_fingerprint_before": before_projection,
        "single_stream_projection_fingerprint_after": after_projection,
        "dependency_contract_fingerprint_before": before_dependency,
        "dependency_contract_fingerprint_after": after_dependency,
        "safety_proofs": proofs,
    }
    for field, expected in audit_expected.items():
        if audit.get(field) != expected:
            raise ValueError(f"post-transform audit {field} differs from transform")
    return {**value, "replacements": normalized}, capture_validation


def materialize(
    transform, source_map, capture, source_root, artifact_root, output_path
):
    normalized, capture_validation = validate_transform(
        transform, source_map, capture, source_root, artifact_root
    )
    source_file = normalized["replacements"][0]["source_file"]
    source = _rooted(source_root, source_file, "source_file")
    output_path = Path(output_path).resolve()
    if output_path == source:
        raise ValueError("component transform output must differ from immutable input")
    if output_path.exists() and file_fingerprint(output_path) != file_fingerprint(
        source
    ):
        raise ValueError(
            "existing component transform output differs from immutable input"
        )
    raw = source.read_bytes()
    cursor = 0
    chunks = []
    for replacement in normalized["replacements"]:
        chunks.append(raw[cursor : replacement["start_offset"]])
        chunks.append(replacement["replacement"].encode("utf-8"))
        cursor = replacement["end_offset"]
    chunks.append(raw[cursor:])
    materialized = b"".join(chunks)
    try:
        ast.parse(materialized.decode("utf-8"), filename=str(output_path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            f"component transform does not preserve valid Python syntax: {error}"
        ) from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(materialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    policy = capture["component_policy"]
    target_range_ids = sorted(
        item["range_id"] for item in source_map["target_bindings"]
    )
    return {
        "schema_version": ACTION_MANIFEST_SCHEMA,
        "trial_id": normalized["trial_id"],
        "change_kind": CHANGE_KIND,
        "immutable_input_source": source_file,
        "materialized_source": str(output_path),
        "input_source_fingerprint": file_fingerprint(source),
        "output_source_fingerprint": file_fingerprint(output_path),
        "component_source_map": normalized["source_map"],
        "component_source_map_fingerprint": source_map["mapping_fingerprint"],
        "component_capture": normalized["component_capture"],
        "component_capture_fingerprint": capture["capture_fingerprint"],
        "component_capture_summary": capture_validation,
        "source_transform_fingerprint": normalized["transform_fingerprint"],
        "only_change": {
            "source_file": source_file,
            "target_set_id": source_map["target_set_id"],
            "target_range_ids": target_range_ids,
            "span_ids": [item["span_id"] for item in normalized["replacements"]],
            "before_order": policy["before_order"],
            "after_order": policy["after_order"],
            "hard_dependencies": source_map["hard_dependencies"],
        },
        "replacements": normalized["replacements"],
        "review_policy_id": normalized["review_policy_id"],
        "review_record": normalized["review_record"],
        "reviewer_fingerprint": normalized["reviewer_fingerprint"],
        "single_stream_projection_fingerprint": normalized[
            "single_stream_projection_fingerprint_before"
        ],
        "dependency_contract_fingerprint": normalized[
            "dependency_contract_fingerprint_before"
        ],
        "post_transform_audit_fingerprint": normalized[
            "post_transform_audit_fingerprint"
        ],
        "post_transform_audit": normalized["post_transform_audit"],
        "safety_proofs": normalized["safety_proofs"],
        "component_aware_verified": True,
        "multistream_only_verified": True,
        "single_change_verified": True,
        "source_adapter_validation": "passed",
    }


def build_dispatch_evidence(action, capture, artifact_root):
    root = Path(artifact_root).resolve()
    action_path = Path(action).resolve()
    capture_path = Path(capture).resolve()
    action_value = json.loads(action_path.read_text())
    if (
        action_value.get("change_kind") != CHANGE_KIND
        or action_value.get("component_aware_verified") is not True
    ):
        raise ValueError(
            "component dispatch verification requires a component-aware action"
        )
    value = json.loads(capture_path.read_text())
    required = {
        "schema_version",
        "trial_id",
        "request_fingerprint",
        "action_manifest_fingerprint",
        "target_set_id",
        "target_range_ids",
        "child_set_preserved",
        "component_lanes_complete",
        "stream_identity_complete",
        "targets",
        "source_files",
        "capture_fingerprint",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != DISPATCH_CAPTURE_SCHEMA
    ):
        raise ValueError(
            f"component dispatch capture must use {DISPATCH_CAPTURE_SCHEMA} with exact fields"
        )
    if value["capture_fingerprint"] != fingerprint(
        {key: item for key, item in value.items() if key != "capture_fingerprint"}
    ):
        raise ValueError("component dispatch capture fingerprint mismatch")
    if value["trial_id"] != action_value["trial_id"] or value[
        "action_manifest_fingerprint"
    ] != fingerprint(action_value):
        raise ValueError("component dispatch action identity mismatch")
    change = action_value["only_change"]
    if (
        value["target_set_id"] != change["target_set_id"]
        or value["target_range_ids"] != change["target_range_ids"]
    ):
        raise ValueError("component dispatch target set differs from action")
    if any(
        value[field] is not True
        for field in (
            "child_set_preserved",
            "component_lanes_complete",
            "stream_identity_complete",
        )
    ):
        raise ValueError(
            "component dispatch did not preserve child/component/stream evidence"
        )
    _validate_source_files(root, value["source_files"])
    targets = value["targets"]
    if (
        not isinstance(targets, list)
        or sorted(item.get("range_id") for item in targets if isinstance(item, dict))
        != change["target_range_ids"]
    ):
        raise ValueError("component dispatch targets differ from action")
    total = 0
    for target in targets:
        if (
            set(target) != {"range_id", "occurrences"}
            or not isinstance(target["occurrences"], list)
            or len(target["occurrences"]) < 3
        ):
            raise ValueError(
                "component dispatch target requires at least three occurrences"
            )
        seen = set()
        for occurrence in target["occurrences"]:
            if not isinstance(occurrence, dict) or set(occurrence) != {
                "alignment_id",
                "observed_statement_order",
                "stream_ids",
            }:
                raise ValueError("component dispatch occurrence fields are invalid")
            alignment = _text(occurrence["alignment_id"], "alignment_id")
            if alignment in seen:
                raise ValueError("component dispatch has duplicate alignment_id")
            seen.add(alignment)
            if occurrence["observed_statement_order"] != change["after_order"]:
                raise ValueError(
                    "component dispatch order does not match planned order"
                )
            streams = occurrence["stream_ids"]
            if (
                not isinstance(streams, list)
                or len(streams) != len(change["after_order"])
                or len(set(streams)) < 2
            ):
                raise ValueError("component dispatch occurrence is not multistream")
            total += 1
    evidence = {
        "schema_version": DISPATCH_EVIDENCE_SCHEMA,
        "trial_id": action_value["trial_id"],
        "request_fingerprint": _text(
            value["request_fingerprint"], "request_fingerprint"
        ),
        "target_set_id": change["target_set_id"],
        "target_range_ids": change["target_range_ids"],
        "action_manifest": action_path.relative_to(root).as_posix(),
        "action_manifest_fingerprint": fingerprint(action_value),
        "dispatch_capture": capture_path.relative_to(root).as_posix(),
        "dispatch_capture_fingerprint": value["capture_fingerprint"],
        "expected_statement_order": change["after_order"],
        "aligned_occurrence_count": total,
        "child_set_preserved": True,
        "component_lanes_complete": True,
        "stream_identity_complete": True,
        "decision": "pass",
    }
    evidence["evidence_fingerprint"] = fingerprint(evidence)
    return evidence


def validate_dispatch_evidence(
    path, artifact_root, *, trial_id=None, request_fingerprint=None
):
    root = Path(artifact_root).resolve()
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != DISPATCH_EVIDENCE_SCHEMA
    ):
        raise ValueError(
            f"component dispatch evidence must use {DISPATCH_EVIDENCE_SCHEMA}"
        )
    if value.get("evidence_fingerprint") != fingerprint(
        {key: item for key, item in value.items() if key != "evidence_fingerprint"}
    ):
        raise ValueError("component dispatch evidence fingerprint mismatch")
    if trial_id is not None and value.get("trial_id") != trial_id:
        raise ValueError("component dispatch trial_id mismatch")
    if (
        request_fingerprint is not None
        and value.get("request_fingerprint") != request_fingerprint
    ):
        raise ValueError("component dispatch request fingerprint mismatch")
    rebuilt = build_dispatch_evidence(
        _rooted(root, value["action_manifest"], "action_manifest"),
        _rooted(root, value["dispatch_capture"], "dispatch_capture"),
        root,
    )
    if _canonical(rebuilt) != _canonical(value):
        raise ValueError(
            "component dispatch evidence differs from deterministic replay"
        )
    return {
        "valid": True,
        "decision": "pass",
        "aligned_occurrence_count": value["aligned_occurrence_count"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    validate_map = commands.add_parser("validate-map")
    validate_map.add_argument("--map", type=Path, required=True)
    validate_map.add_argument("--source-root", type=Path, required=True)
    validate_capture_parser = commands.add_parser("validate-capture")
    validate_capture_parser.add_argument("--capture", type=Path, required=True)
    validate_capture_parser.add_argument("--map", type=Path, required=True)
    validate_capture_parser.add_argument("--source-root", type=Path, required=True)
    validate_capture_parser.add_argument("--artifact-root", type=Path, required=True)
    materialize_parser = commands.add_parser("materialize")
    materialize_parser.add_argument("--transform", type=Path, required=True)
    materialize_parser.add_argument("--map", type=Path, required=True)
    materialize_parser.add_argument("--capture", type=Path, required=True)
    materialize_parser.add_argument("--source-root", type=Path, required=True)
    materialize_parser.add_argument("--artifact-root", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--manifest-out", type=Path, required=True)
    dispatch = commands.add_parser("verify-dispatch")
    dispatch.add_argument("--action-manifest", type=Path, required=True)
    dispatch.add_argument("--dispatch-capture", type=Path, required=True)
    dispatch.add_argument("--artifact-root", type=Path, required=True)
    dispatch.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-map":
            result = validate_source_map(
                json.loads(args.map.read_text()), args.source_root
            )
        elif args.command == "validate-capture":
            source_map = validate_source_map(
                json.loads(args.map.read_text()), args.source_root
            )
            result = validate_capture(
                json.loads(args.capture.read_text()), args.artifact_root, source_map
            )
        elif args.command == "materialize":
            source_map = validate_source_map(
                json.loads(args.map.read_text()), args.source_root
            )
            capture = json.loads(args.capture.read_text())
            result = materialize(
                json.loads(args.transform.read_text()),
                source_map,
                capture,
                args.source_root,
                args.artifact_root,
                args.output,
            )
            if args.manifest_out.exists():
                raise ValueError(
                    f"component action manifest output already exists: {args.manifest_out}"
                )
            _atomic_json(args.manifest_out, result)
        else:
            result = build_dispatch_evidence(
                args.action_manifest, args.dispatch_capture, args.artifact_root
            )
            if args.out.exists():
                raise ValueError(
                    f"component dispatch evidence output already exists: {args.out}"
                )
            _atomic_json(args.out, result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
