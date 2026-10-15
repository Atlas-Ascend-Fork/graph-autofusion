#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Materialize isolated option trials and maintain an auditable trial state."""

import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

import multistream_operator_order


ACTION_SCHEMA = "superkernel-multistream-action-manifest-v1"
ACTION_SCHEMA_V2 = "superkernel-multistream-action-manifest-v2"
STATE_SCHEMA = "superkernel-multistream-trial-state-v1"
SOURCE_SNAPSHOT_SCHEMA = "superkernel-source-snapshot-manifest-v1"
ORDERED_STATES = (
    "planned",
    "materialized",
    "diff_verified",
    "correctness_passed",
    "profile_collected",
    "analysis_validated",
    "clean3_passed",
    "clean5_passed",
    "accepted",
)
TERMINAL_STATES = {"accepted", "rejected", "blocked", "failed"}
SOURCE_CHANGE_KINDS = {"scope_split", "range_exclusion"}
REORDER_CHANGE_KIND = "dependency_safe_operator_reorder"
SCOPE_CALL = re.compile(
    r"^[ \t]*torch\.npu\.super_kernel_scope_(begin|end)"
    r"\((?P<arg>None|(?P<quote>['\"])[A-Za-z0-9_.:-]+(?P=quote))\)[ \t]*$"
)


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def file_fingerprint(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source_snapshot(manifest, artifact_root):
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA
    ):
        raise ValueError(f"source manifest must use {SOURCE_SNAPSHOT_SCHEMA}")
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("source manifest source_revision must be a non-empty string")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source manifest files must be a non-empty list")
    root = Path(artifact_root).resolve()
    normalized = []
    seen = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {"path", "file_fingerprint"}:
            raise ValueError(
                f"source manifest files[{index}] must contain path and file_fingerprint"
            )
        relative = _safe_relative_artifact(item["path"])
        if relative in seen:
            raise ValueError(f"source manifest has duplicate path: {relative}")
        seen.add(relative)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("source manifest file escapes artifact root") from error
        if not candidate.is_file():
            raise ValueError(f"source manifest file does not exist: {relative}")
        actual = file_fingerprint(candidate)
        if item["file_fingerprint"] != actual:
            raise ValueError(f"source manifest file fingerprint mismatch: {relative}")
        normalized.append({"path": relative, "file_fingerprint": actual})
    normalized.sort(key=lambda item: item["path"])
    fingerprint = content_fingerprint({"files": normalized})
    if manifest.get("source_fingerprint") != fingerprint:
        raise ValueError("source manifest source_fingerprint mismatch")
    return {
        "source_revision": revision,
        "source_fingerprint": fingerprint,
        "files": normalized,
    }


def generate_source_snapshot(source_root, artifact_root, source_revision, files):
    source_root = Path(source_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    if not source_root.is_dir():
        raise ValueError("source snapshot root must be an existing directory")
    try:
        source_root.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError("source snapshot root must be inside artifact root") from error
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("source snapshot source_revision must be non-empty")
    if not isinstance(files, list) or not files:
        raise ValueError("source snapshot file list must be non-empty")
    normalized = []
    seen = set()
    for index, value in enumerate(files):
        if not isinstance(value, str) or not value:
            raise ValueError(f"source snapshot files[{index}] must be a relative path")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or value in {"", "."}:
            raise ValueError(f"source snapshot files[{index}] is unsafe")
        path = (source_root / value).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ValueError("source snapshot file escapes source root") from error
        if not path.is_file():
            raise ValueError(f"source snapshot file does not exist: {value}")
        artifact_relative = path.relative_to(artifact_root).as_posix()
        if artifact_relative in seen:
            raise ValueError(
                f"source snapshot contains duplicate file: {artifact_relative}"
            )
        seen.add(artifact_relative)
        normalized.append(
            {"path": artifact_relative, "file_fingerprint": file_fingerprint(path)}
        )
    normalized.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "source_revision": source_revision,
        "source_fingerprint": content_fingerprint({"files": normalized}),
        "files": normalized,
    }
    validate_source_snapshot(manifest, artifact_root)
    return manifest


def _load_structured(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text())
        else:
            value = yaml.safe_load(path.read_text())
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load structured config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return value


def _atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path, value):
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_structured(path, value):
    path = Path(path)
    if path.suffix.lower() == ".json":
        rendered = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    else:
        rendered = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    _atomic_write_text(path, rendered)


def _decode_pointer(pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        raise ValueError("json_pointer must be a non-root RFC6901 pointer")
    tokens = []
    for raw in pointer[1:].split("/"):
        index = 0
        decoded = []
        while index < len(raw):
            if raw[index] != "~":
                decoded.append(raw[index])
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in "01":
                raise ValueError(f"json_pointer has an invalid escape: {pointer}")
            decoded.append("~" if raw[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(decoded))
    return tokens


def _resolve_parent(value, tokens):
    current = value
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"json_pointer component does not exist: {token}")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as error:
                raise ValueError(
                    f"json_pointer list index is invalid: {token}"
                ) from error
            if index < 0 or index >= len(current):
                raise ValueError(f"json_pointer list index is out of range: {token}")
            current = current[index]
        else:
            raise ValueError(f"json_pointer traverses a scalar at: {token}")
    return current, tokens[-1]


def _get_pointer(value, tokens):
    parent, leaf = _resolve_parent(value, tokens)
    if isinstance(parent, dict):
        if leaf not in parent:
            raise ValueError(f"json_pointer target does not exist: {leaf}")
        return parent[leaf]
    if isinstance(parent, list):
        try:
            index = int(leaf)
        except ValueError as error:
            raise ValueError(f"json_pointer list index is invalid: {leaf}") from error
        if index < 0 or index >= len(parent):
            raise ValueError(f"json_pointer list index is out of range: {leaf}")
        return parent[index]
    raise ValueError("json_pointer target parent is a scalar")


def _set_pointer(value, tokens, replacement):
    parent, leaf = _resolve_parent(value, tokens)
    if isinstance(parent, dict):
        if leaf not in parent:
            raise ValueError(f"json_pointer target does not exist: {leaf}")
        parent[leaf] = replacement
        return
    if isinstance(parent, list):
        try:
            index = int(leaf)
        except ValueError as error:
            raise ValueError(f"json_pointer list index is invalid: {leaf}") from error
        if index < 0 or index >= len(parent):
            raise ValueError(f"json_pointer list index is out of range: {leaf}")
        parent[index] = replacement
        return
    raise ValueError("json_pointer target parent is a scalar")


def _changed_pointers(before, after, prefix=""):
    if type(before) is not type(after):
        return [prefix or "/"]
    if isinstance(before, dict):
        changed = []
        for key in sorted(set(before) | set(after)):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{prefix}/{escaped}"
            if key not in before or key not in after:
                changed.append(child)
            else:
                changed.extend(_changed_pointers(before[key], after[key], child))
        return changed
    if isinstance(before, list):
        if len(before) != len(after):
            return [prefix or "/"]
        changed = []
        for index, (left, right) in enumerate(zip(before, after)):
            changed.extend(_changed_pointers(left, right, f"{prefix}/{index}"))
        return changed
    return [] if before == after else [prefix or "/"]


def _manifest_path(path, artifact_root):
    path = Path(path).resolve()
    if artifact_root is None:
        return str(path)
    root = Path(artifact_root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact path escapes artifact root: {path}") from error


def materialize_option(
    input_path, output_path, pointer, before, after, trial_id, artifact_root=None
):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("output config must differ from immutable input config")
    if output_path.exists():
        raise ValueError(f"output config already exists: {output_path}")
    config = _load_structured(input_path)
    tokens = _decode_pointer(pointer)
    actual_before = _get_pointer(config, tokens)
    if _canonical_json(actual_before) != _canonical_json(before):
        raise ValueError("declared before value does not match the input config")
    if _canonical_json(before) == _canonical_json(after):
        raise ValueError("before and after values must differ")

    before_config = json.loads(_canonical_json(config))
    _set_pointer(config, tokens, after)
    changed = _changed_pointers(before_config, config)
    if changed != [pointer]:
        raise ValueError(f"option materialization changed unexpected paths: {changed}")
    _write_structured(output_path, config)
    replayed = _load_structured(output_path)
    if _canonical_json(replayed) != _canonical_json(config):
        raise ValueError("serialized output config does not round-trip exactly")

    return {
        "schema_version": ACTION_SCHEMA,
        "trial_id": trial_id,
        "change_kind": "option",
        "immutable_input_config": _manifest_path(input_path, artifact_root),
        "materialized_config": _manifest_path(output_path, artifact_root),
        "input_file_fingerprint": file_fingerprint(input_path),
        "output_file_fingerprint": file_fingerprint(output_path),
        "input_content_fingerprint": content_fingerprint(before_config),
        "output_content_fingerprint": content_fingerprint(config),
        "only_change": {
            "json_pointer": pointer,
            "before": before,
            "after": after,
        },
        "changed_pointers": changed,
        "single_change_verified": True,
    }


def _scope_calls(text):
    calls = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = SCOPE_CALL.fullmatch(line)
        if match is None:
            raise ValueError("source insertion contains a non-scope statement")
        raw_argument = match.group("arg")
        argument = None if raw_argument == "None" else ast.literal_eval(raw_argument)
        calls.append((match.group(1), argument))
    if not calls:
        raise ValueError("source insertion must contain at least one scope call")
    return calls


def _validate_source_action(kind, start, end, insertions):
    if kind not in SOURCE_CHANGE_KINDS:
        raise ValueError("source change_kind must be scope_split or range_exclusion")
    offsets = [item["offset"] for item in insertions]
    if any(offset not in {start, end} for offset in offsets):
        raise ValueError("source insertions are allowed only at exact range boundaries")
    calls_by_offset = {start: [], end: []}
    for insertion in insertions:
        calls_by_offset[insertion["offset"]].extend(_scope_calls(insertion["text"]))

    if kind == "scope_split":
        populated = [calls for calls in calls_by_offset.values() if calls]
        if len(populated) != 1 or len(populated[0]) != 2:
            raise ValueError("scope_split requires one boundary with end then begin")
        if [call[0] for call in populated[0]] != ["end", "begin"]:
            raise ValueError(
                "scope_split requires one scope_end followed by scope_begin"
            )
        if any(call[1] is None for call in populated[0]):
            raise ValueError("scope_split requires named scopes")
        return "dependency_aligned_split"

    start_calls = calls_by_offset[start]
    end_calls = calls_by_offset[end]
    if start_calls == [("begin", None)] and end_calls == [("end", None)]:
        return "explicit_none_exclusion"
    if (
        len(start_calls) == 1
        and len(end_calls) == 1
        and start_calls[0][0] == "end"
        and end_calls[0][0] == "begin"
        and start_calls[0][1] is not None
        and start_calls[0][1] == end_calls[0][1]
    ):
        return "close_reopen_exclusion"
    raise ValueError(
        "range_exclusion requires balanced None exclusion or matching close/reopen"
    )


def materialize_source_action(
    input_path,
    output_path,
    source_file,
    start_offset,
    end_offset,
    boundary_change,
    kind,
    insertions,
    trial_id,
    artifact_root=None,
):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("output source must differ from immutable input source")
    if output_path.exists():
        raise ValueError(f"output source already exists: {output_path}")
    original = input_path.read_bytes()
    try:
        original_text = original.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("source action requires UTF-8 source") from error
    if not (0 <= start_offset < end_offset <= len(original)):
        raise ValueError("source half-open byte range is invalid")
    for offset in (start_offset, end_offset):
        try:
            original[:offset].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("source range offset splits a UTF-8 character") from error
    if not isinstance(insertions, list) or not insertions:
        raise ValueError("source action requires a non-empty insertion list")
    normalized = []
    for index, insertion in enumerate(insertions):
        if not isinstance(insertion, dict) or set(insertion) != {"offset", "text"}:
            raise ValueError(f"insertions[{index}] must contain offset and text")
        offset = insertion["offset"]
        text = insertion["text"]
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError(f"insertions[{index}].offset must be an integer")
        if not isinstance(text, str) or not text:
            raise ValueError(f"insertions[{index}].text must be non-empty")
        normalized.append({"offset": offset, "text": text})
    mechanism = _validate_source_action(kind, start_offset, end_offset, normalized)

    grouped = {}
    for insertion in normalized:
        grouped.setdefault(insertion["offset"], []).append(insertion["text"].encode())
    cursor = 0
    output_chunks = []
    for offset in sorted(grouped):
        output_chunks.append(original[cursor:offset])
        output_chunks.extend(grouped[offset])
        cursor = offset
    output_chunks.append(original[cursor:])
    materialized = b"".join(output_chunks)
    try:
        ast.parse(original_text, filename=str(input_path))
        ast.parse(materialized.decode("utf-8"), filename=str(output_path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(
            f"source action does not preserve valid Python syntax: {error}"
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
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    return {
        "schema_version": ACTION_SCHEMA,
        "trial_id": trial_id,
        "change_kind": kind,
        "immutable_input_source": _manifest_path(input_path, artifact_root),
        "materialized_source": _manifest_path(output_path, artifact_root),
        "input_source_fingerprint": file_fingerprint(input_path),
        "output_source_fingerprint": file_fingerprint(output_path),
        "only_change": {
            "source_file": source_file,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "boundary_change": boundary_change,
        },
        "insertions": normalized,
        "mechanism": mechanism,
        "single_change_verified": True,
        "source_adapter_validation": "passed",
    }


def materialize_operator_reorder(
    input_path,
    output_path,
    order_analysis_path,
    range_id,
    after_order,
    trial_id,
    artifact_root=None,
    route2_dispatch_evidence=None,
    route2_clean_evidence=None,
):
    """Reorder one contiguous, dependency-safe statement region in an isolated source."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    analysis_path = Path(order_analysis_path).resolve()
    root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else analysis_path.parent
    )
    if input_path == output_path:
        raise ValueError("output source must differ from immutable input source")
    if output_path.exists():
        raise ValueError(f"output source already exists: {output_path}")
    multistream_operator_order.validate_analysis(analysis_path, root)
    analysis = json.loads(analysis_path.read_text())
    matches = [item for item in analysis["targets"] if item["range_id"] == range_id]
    if len(matches) != 1:
        raise ValueError(
            "operator reorder range_id must match exactly one analysis target"
        )
    target = matches[0]
    if target.get("multistream_reorder_authorized") is not True:
        raise ValueError(
            "business operator reorder requires direct multistream authorization"
        )
    legal_orders = [target["route2_order"], *target["route3_orders"]]
    if after_order not in legal_orders:
        raise ValueError(
            "after_order is not an analyzer-generated legal route2/route3 order"
        )
    route = "route2" if after_order == target["route2_order"] else "route3"
    route3_prerequisite = None
    if route == "route3":
        if route2_dispatch_evidence is None or route2_clean_evidence is None:
            raise ValueError(
                "route3 requires route2 dispatch evidence and rejected clean3 evidence"
            )
        dispatch_path = Path(route2_dispatch_evidence).resolve()
        clean_path = Path(route2_clean_evidence).resolve()
        dispatch_validation = multistream_operator_order.validate_dispatch_evidence(
            dispatch_path,
            root,
            request_fingerprint=analysis["request_fingerprint"],
        )
        dispatch = json.loads(dispatch_path.read_text())
        route2_action_path = root / dispatch["action_manifest"]
        route2_action = json.loads(route2_action_path.read_text())
        route2_change = route2_action.get("only_change", {})
        if (
            route2_action.get("change_kind") != REORDER_CHANGE_KIND
            or route2_action.get("operator_order_analysis_fingerprint")
            != analysis["analysis_fingerprint"]
            or route2_change.get("range_id") != range_id
            or route2_change.get("route") != "route2"
            or route2_change.get("after_order") != target["route2_order"]
        ):
            raise ValueError("route3 prerequisite is not the matching route2 action")
        clean = multistream_operator_order.validate_route2_no_gain_evidence(
            clean_path,
            root,
            trial_id=route2_action["trial_id"],
            request_fingerprint=analysis["request_fingerprint"],
        )
        route3_prerequisite = {
            "route2_trial_id": route2_action["trial_id"],
            "route2_dispatch_evidence": _manifest_path(dispatch_path, artifact_root),
            "route2_dispatch_evidence_fingerprint": dispatch["evidence_fingerprint"],
            "route2_dispatch_occurrence_count": dispatch_validation[
                "aligned_occurrence_count"
            ],
            "route2_clean3_evidence": _manifest_path(clean_path, artifact_root),
            "route2_clean3_evidence_fingerprint": clean["evidence_fingerprint"],
            "route2_clean3_decision": "reject",
        }
    try:
        input_relative = input_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("input source must be inside the artifact root") from error
    if input_relative != target["source_file"]:
        raise ValueError("input source does not match the analyzed source file")
    if file_fingerprint(input_path) != target["source_file_fingerprint"]:
        raise ValueError(
            "input source fingerprint differs from operator order analysis"
        )

    original = input_path.read_bytes()
    try:
        original_text = original.decode("utf-8")
        ast.parse(original_text, filename=str(input_path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            f"operator reorder requires valid UTF-8 Python source: {error}"
        ) from error
    start = target["range_start_offset"]
    end = target["range_end_offset"]
    if not 0 <= start < end <= len(original):
        raise ValueError("operator reorder range is outside the input source")
    statements = target["statements"]
    before_order = target["current_source_order"]
    if [item["statement_id"] for item in statements] != before_order:
        raise ValueError("operator reorder statement order differs from analysis")
    if set(after_order) != set(before_order) or len(after_order) != len(
        set(after_order)
    ):
        raise ValueError(
            "operator reorder must be a permutation of the analyzed statements"
        )
    if after_order == before_order:
        raise ValueError("operator reorder must change statement order")
    blocks = {}
    block_records = []
    for statement in statements:
        statement_id = statement["statement_id"]
        block = original[statement["start_offset"] : statement["end_offset"]]
        if not block:
            raise ValueError(
                f"operator reorder statement block is empty: {statement_id}"
            )
        try:
            block.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("operator reorder statement span splits UTF-8") from error
        blocks[statement_id] = block
        block_records.append(
            {
                **statement,
                "block_fingerprint": "sha256:" + hashlib.sha256(block).hexdigest(),
            }
        )
    hard_edges = [
        (item["before"], item["after"]) for item in target["hard_dependencies"]
    ]
    after_rank = {statement_id: index for index, statement_id in enumerate(after_order)}
    violated = [
        edge for edge in hard_edges if after_rank[edge[0]] >= after_rank[edge[1]]
    ]
    if violated:
        raise ValueError(f"operator reorder violates hard dependencies: {violated}")

    replacement = b"".join(blocks[statement_id] for statement_id in after_order)
    if len(replacement) != end - start:
        raise ValueError("operator reorder must preserve the exact source region size")
    materialized = original[:start] + replacement + original[end:]
    try:
        ast.parse(materialized.decode("utf-8"), filename=str(output_path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            f"operator reorder does not preserve valid Python syntax: {error}"
        ) from error
    if sorted(blocks.values()) != sorted(
        replacement[
            sum(len(blocks[item]) for item in after_order[:index]) : sum(
                len(blocks[item]) for item in after_order[: index + 1]
            )
        ]
        for index in range(len(after_order))
    ):
        raise ValueError("operator reorder changed statement block contents")

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
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    manifest = {
        "schema_version": ACTION_SCHEMA,
        "trial_id": trial_id,
        "change_kind": REORDER_CHANGE_KIND,
        "immutable_input_source": _manifest_path(input_path, artifact_root),
        "materialized_source": _manifest_path(output_path, artifact_root),
        "input_source_fingerprint": file_fingerprint(input_path),
        "output_source_fingerprint": file_fingerprint(output_path),
        "operator_order_analysis": _manifest_path(analysis_path, artifact_root),
        "operator_order_analysis_fingerprint": analysis["analysis_fingerprint"],
        "only_change": {
            "source_file": target["source_file"],
            "range_id": range_id,
            "start_offset": start,
            "end_offset": end,
            "before_order": before_order,
            "after_order": after_order,
            "route": route,
            "hard_dependencies": target["hard_dependencies"],
        },
        "statement_blocks": block_records,
        "stable_parallel_pairs": target["stable_parallel_pairs"],
        "stable_resource_complementary_pairs": target[
            "stable_resource_complementary_pairs"
        ],
        "stable_dispatch_inversions": target["stable_dispatch_inversions"],
        "resource_pair_policy": multistream_operator_order.RESOURCE_PAIR_POLICY,
        "multistream_only_verified": True,
        "single_change_verified": True,
        "source_adapter_validation": "passed",
    }
    if route3_prerequisite is not None:
        manifest["route3_prerequisite"] = route3_prerequisite
    return manifest


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def initialize_state(trial_id, request_fingerprint, action_manifest):
    manifest = json.loads(Path(action_manifest).read_text())
    if manifest.get("schema_version") not in {ACTION_SCHEMA, ACTION_SCHEMA_V2}:
        raise ValueError(
            f"action manifest must use {ACTION_SCHEMA} or {ACTION_SCHEMA_V2}"
        )
    if manifest.get("trial_id") != trial_id:
        raise ValueError("action manifest trial_id mismatch")
    return {
        "schema_version": STATE_SCHEMA,
        "trial_id": trial_id,
        "request_fingerprint": request_fingerprint,
        "state": "planned",
        "terminal": False,
        "action_manifest": str(Path(action_manifest).resolve()),
        "events": [{"state": "planned", "at": _now(), "evidence": None}],
    }


def _safe_relative_artifact(value):
    if value is None:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("evidence must be a safe relative path")
    return value


def advance_state(state, target, evidence=None):
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"state must use {STATE_SCHEMA}")
    current = state.get("state")
    if current in TERMINAL_STATES:
        raise ValueError(f"terminal trial state cannot advance: {current}")
    if target in TERMINAL_STATES - {"accepted"}:
        pass
    else:
        if current not in ORDERED_STATES or target not in ORDERED_STATES:
            raise ValueError(f"unknown trial state transition: {current} -> {target}")
        current_index = ORDERED_STATES.index(current)
        if ORDERED_STATES[current_index + 1] != target:
            raise ValueError(
                f"trial state must advance one gate at a time: {current} -> {target}"
            )
    if target not in {"planned", "materialized"} and evidence is None:
        raise ValueError(f"transition to {target} requires an evidence path")
    evidence = _safe_relative_artifact(evidence)
    updated = json.loads(_canonical_json(state))
    updated["state"] = target
    updated["terminal"] = target in TERMINAL_STATES
    updated.setdefault("events", []).append(
        {"state": target, "at": _now(), "evidence": evidence}
    )
    return updated


def validate_state(
    state,
    *,
    trial_id=None,
    request_fingerprint=None,
    require_accepted=False,
    artifact_root=None,
):
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"state must use {STATE_SCHEMA}")
    if trial_id is not None and state.get("trial_id") != trial_id:
        raise ValueError("trial state trial_id mismatch")
    if (
        request_fingerprint is not None
        and state.get("request_fingerprint") != request_fingerprint
    ):
        raise ValueError("trial state request_fingerprint mismatch")
    events = state.get("events")
    if (
        not isinstance(events, list)
        or not events
        or events[0].get("state") != "planned"
    ):
        raise ValueError("trial state must start with planned")
    replay = "planned"
    terminal = False
    for event in events[1:]:
        if terminal:
            raise ValueError("trial state has events after a terminal state")
        target = event.get("state")
        evidence = event.get("evidence")
        if evidence is not None and artifact_root is not None:
            relative = _safe_relative_artifact(evidence)
            root = Path(artifact_root).resolve()
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "trial state evidence escapes artifact root"
                ) from error
            if not candidate.is_file():
                raise ValueError(f"trial state evidence does not exist: {relative}")
        holder = {"schema_version": STATE_SCHEMA, "state": replay, "events": []}
        advanced = advance_state(holder, target, evidence)
        replay = advanced["state"]
        terminal = advanced["terminal"]
    if state.get("state") != replay or state.get("terminal") is not terminal:
        raise ValueError("trial state summary differs from its event history")
    if require_accepted and replay != "accepted":
        raise ValueError("accepted result requires trial state accepted")
    return {"valid": True, "state": replay, "terminal": terminal}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-option")
    materialize.add_argument("--input", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--json-pointer", required=True)
    materialize.add_argument("--before-json", required=True)
    materialize.add_argument("--after-json", required=True)
    materialize.add_argument("--trial-id", required=True)
    materialize.add_argument("--artifact-root", type=Path, required=True)
    materialize.add_argument("--manifest-out", type=Path, required=True)

    source = subparsers.add_parser("materialize-source")
    source.add_argument("--input", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--source-file", required=True)
    source.add_argument("--start-offset", type=int, required=True)
    source.add_argument("--end-offset", type=int, required=True)
    source.add_argument("--boundary-change", required=True)
    source.add_argument(
        "--change-kind", choices=sorted(SOURCE_CHANGE_KINDS), required=True
    )
    source.add_argument("--insertions", type=Path, required=True)
    source.add_argument("--trial-id", required=True)
    source.add_argument("--artifact-root", type=Path, required=True)
    source.add_argument("--manifest-out", type=Path, required=True)

    reorder = subparsers.add_parser("materialize-reorder")
    reorder.add_argument("--input", type=Path, required=True)
    reorder.add_argument("--output", type=Path, required=True)
    reorder.add_argument("--operator-order-analysis", type=Path, required=True)
    reorder.add_argument("--range-id", required=True)
    reorder.add_argument("--after-order", type=Path, required=True)
    reorder.add_argument("--trial-id", required=True)
    reorder.add_argument("--artifact-root", type=Path, required=True)
    reorder.add_argument("--route2-dispatch-evidence", type=Path)
    reorder.add_argument("--route2-clean-evidence", type=Path)
    reorder.add_argument("--manifest-out", type=Path, required=True)

    initialize = subparsers.add_parser("init-state")
    initialize.add_argument("--trial-id", required=True)
    initialize.add_argument("--request-fingerprint", required=True)
    initialize.add_argument("--action-manifest", type=Path, required=True)
    initialize.add_argument("--state-out", type=Path, required=True)

    advance = subparsers.add_parser("advance-state")
    advance.add_argument("--state", type=Path, required=True)
    advance.add_argument("--to", required=True)
    advance.add_argument("--evidence")

    validate = subparsers.add_parser("validate-state")
    validate.add_argument("--state", type=Path, required=True)
    validate.add_argument("--trial-id")
    validate.add_argument("--request-fingerprint")
    validate.add_argument("--require-accepted", action="store_true")

    snapshot = subparsers.add_parser("snapshot-source")
    snapshot.add_argument("--source-root", type=Path, required=True)
    snapshot.add_argument("--artifact-root", type=Path, required=True)
    snapshot.add_argument("--source-revision", required=True)
    snapshot.add_argument("--file-list", type=Path, required=True)
    snapshot.add_argument("--manifest-out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "materialize-option":
            result = materialize_option(
                args.input,
                args.output,
                args.json_pointer,
                json.loads(args.before_json),
                json.loads(args.after_json),
                args.trial_id,
                args.artifact_root,
            )
            _write_json(args.manifest_out, result)
        elif args.command == "materialize-source":
            result = materialize_source_action(
                args.input,
                args.output,
                args.source_file,
                args.start_offset,
                args.end_offset,
                args.boundary_change,
                args.change_kind,
                json.loads(args.insertions.read_text()),
                args.trial_id,
                args.artifact_root,
            )
            _write_json(args.manifest_out, result)
        elif args.command == "materialize-reorder":
            after_order = json.loads(args.after_order.read_text())
            if not isinstance(after_order, list):
                raise ValueError("--after-order must contain a JSON list")
            result = materialize_operator_reorder(
                args.input,
                args.output,
                args.operator_order_analysis,
                args.range_id,
                after_order,
                args.trial_id,
                args.artifact_root,
                args.route2_dispatch_evidence,
                args.route2_clean_evidence,
            )
            _write_json(args.manifest_out, result)
        elif args.command == "init-state":
            result = initialize_state(
                args.trial_id, args.request_fingerprint, args.action_manifest
            )
            _write_json(args.state_out, result)
        elif args.command == "advance-state":
            state = json.loads(args.state.read_text())
            result = advance_state(state, args.to, args.evidence)
            _write_json(args.state, result)
        elif args.command == "validate-state":
            result = validate_state(
                json.loads(args.state.read_text()),
                trial_id=args.trial_id,
                request_fingerprint=args.request_fingerprint,
                require_accepted=args.require_accepted,
            )
        else:
            if args.manifest_out.exists():
                raise ValueError(
                    f"source snapshot manifest already exists: {args.manifest_out}"
                )
            result = generate_source_snapshot(
                args.source_root,
                args.artifact_root,
                args.source_revision,
                json.loads(args.file_list.read_text()),
            )
            _write_json(args.manifest_out, result)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
