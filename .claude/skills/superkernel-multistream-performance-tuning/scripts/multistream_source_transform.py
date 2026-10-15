#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Materialize analyzer-authorized event/stage source transforms without free-form patches."""

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

import multistream_event_stage_action


TRANSFORM_SCHEMA = "superkernel-multistream-source-transform-v1"
ACTION_MANIFEST_SCHEMA = "superkernel-multistream-action-manifest-v2"


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
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
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
        raise ValueError(f"{label} escapes source root") from error
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


def _validate_action(catalog, action_id):
    catalog = multistream_event_stage_action.validate(catalog)
    action_id = _text(action_id, "action_id")
    matches = [item for item in catalog["actions"] if item["action_id"] == action_id]
    if len(matches) != 1:
        raise ValueError("source transform action_id is not analyzer-authorized")
    return catalog, matches[0]


def validate(value, catalog, source_root):
    required = {
        "schema_version",
        "trial_id",
        "action_catalog_fingerprint",
        "action_id",
        "source_file",
        "input_source_fingerprint",
        "single_stream_projection_fingerprint_before",
        "single_stream_projection_fingerprint_after",
        "allowed_multistream_ranges",
        "dependency_evidence_fingerprint_before",
        "dependency_evidence_fingerprint_after",
        "safety_proofs",
        "stage_state_contract",
        "replacements",
        "transform_fingerprint",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != TRANSFORM_SCHEMA
    ):
        raise ValueError(f"source transform must use {TRANSFORM_SCHEMA}")
    unsigned = {
        key: item for key, item in value.items() if key != "transform_fingerprint"
    }
    if value["transform_fingerprint"] != fingerprint(unsigned):
        raise ValueError("source transform fingerprint mismatch")
    catalog, action = _validate_action(catalog, value["action_id"])
    if value["action_catalog_fingerprint"] != catalog["action_catalog_fingerprint"]:
        raise ValueError("source transform action catalog fingerprint mismatch")
    if action["change_kind"] not in {"event_edge_refinement", "stage_split"}:
        raise ValueError("source transform action kind is unsupported")
    if action["change_kind"] == "event_edge_refinement":
        required_proofs = {
            "producer_before_record",
            "consumer_after_wait",
            "event_reuse_safe",
            "record_stream_lifetime_preserved",
            "modified_dependency_complete",
        }
        if value["stage_state_contract"] is not None:
            raise ValueError(
                "event source transform cannot declare stage_state_contract"
            )
    else:
        required_proofs = {
            "alias_safe",
            "lifetime_preserved",
            "modified_dependency_complete",
            "stage_state_explicit",
        }
        state = value["stage_state_contract"]
        if not isinstance(state, dict) or set(state) != {
            "inputs",
            "outputs",
            "preserved_values",
            "forbidden_effects",
        }:
            raise ValueError("stage split requires an explicit stage_state_contract")
        for field in ("inputs", "outputs", "preserved_values"):
            if (
                not isinstance(state[field], list)
                or not state[field]
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in state[field]
                )
            ):
                raise ValueError(
                    f"stage_state_contract.{field} must be non-empty identifiers"
                )
        if state["forbidden_effects"] != []:
            raise ValueError("stage_state_contract cannot cross forbidden effects")
    proofs = value["safety_proofs"]
    if (
        not isinstance(proofs, dict)
        or set(proofs) != required_proofs
        or any(item is not True for item in proofs.values())
    ):
        raise ValueError(
            "source transform lacks complete action-specific safety proofs"
        )
    if (
        value["single_stream_projection_fingerprint_before"]
        != value["single_stream_projection_fingerprint_after"]
    ):
        raise ValueError(
            "source transform must preserve single_stream_projection fingerprint"
        )
    source = _rooted(source_root, value["source_file"], "source_file")
    if value["input_source_fingerprint"] != file_fingerprint(source):
        raise ValueError("source transform input source fingerprint mismatch")
    original = source.read_bytes()
    try:
        ast.parse(original.decode("utf-8"), filename=str(source))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            f"source transform requires valid UTF-8 Python input: {error}"
        ) from error
    replacements = value["replacements"]
    allowed_ranges = value["allowed_multistream_ranges"]
    if not isinstance(allowed_ranges, list) or not allowed_ranges:
        raise ValueError("source transform requires allowed_multistream_ranges")
    normalized_ranges = []
    previous_range_end = -1
    for index, item in enumerate(allowed_ranges):
        if not isinstance(item, dict) or set(item) != {"start_offset", "end_offset"}:
            raise ValueError(f"allowed_multistream_ranges[{index}] fields are invalid")
        start, end = item["start_offset"], item["end_offset"]
        if any(
            isinstance(entry, bool) or not isinstance(entry, int)
            for entry in (start, end)
        ) or not 0 <= start < end <= len(original):
            raise ValueError(
                f"allowed_multistream_ranges[{index}] byte range is invalid"
            )
        if start < previous_range_end:
            raise ValueError(
                "allowed_multistream_ranges must be sorted and non-overlapping"
            )
        previous_range_end = end
        normalized_ranges.append({"start_offset": start, "end_offset": end})
    for field in (
        "dependency_evidence_fingerprint_before",
        "dependency_evidence_fingerprint_after",
    ):
        _text(value[field], field)
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("source transform replacements must be non-empty")
    normalized = []
    previous_end = -1
    changed = False
    for index, replacement in enumerate(
        sorted(
            replacements,
            key=lambda item: item.get("start_offset", -1)
            if isinstance(item, dict)
            else -1,
        )
    ):
        fields = {"start_offset", "end_offset", "before_fingerprint", "replacement"}
        if not isinstance(replacement, dict) or set(replacement) != fields:
            raise ValueError(f"replacements[{index}] fields are invalid")
        start, end = replacement["start_offset"], replacement["end_offset"]
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in (start, end)
        ) or not 0 <= start < end <= len(original):
            raise ValueError(f"replacements[{index}] byte range is invalid")
        if start < previous_end:
            raise ValueError("source transform replacement ranges overlap")
        if not any(
            start >= allowed["start_offset"] and end <= allowed["end_offset"]
            for allowed in normalized_ranges
        ):
            raise ValueError(
                "source transform replacement is outside declared multistream ranges"
            )
        previous_end = end
        try:
            original[:start].decode("utf-8")
            original[:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("source transform replacement splits UTF-8") from error
        before = original[start:end]
        expected = "sha256:" + hashlib.sha256(before).hexdigest()
        if replacement["before_fingerprint"] != expected:
            raise ValueError(
                f"replacements[{index}].before_fingerprint does not match source"
            )
        text = replacement["replacement"]
        if not isinstance(text, str):
            raise ValueError(f"replacements[{index}].replacement must be a string")
        encoded = text.encode("utf-8")
        changed = changed or encoded != before
        normalized.append(
            {
                "start_offset": start,
                "end_offset": end,
                "before_fingerprint": expected,
                "replacement": text,
            }
        )
    if not changed:
        raise ValueError("source transform must change source bytes")
    normalized_value = {
        **value,
        "source_file": source.relative_to(Path(source_root).resolve()).as_posix(),
        "allowed_multistream_ranges": normalized_ranges,
        "replacements": normalized,
    }
    return normalized_value, action, source


def materialize(value, catalog, source_root, output_path):
    transform, action, source = validate(value, catalog, source_root)
    output_path = Path(output_path).resolve()
    if output_path == source or output_path.exists():
        raise ValueError("source transform output must be distinct and absent")
    original = source.read_bytes()
    cursor = 0
    chunks = []
    for replacement in transform["replacements"]:
        chunks.append(original[cursor : replacement["start_offset"]])
        chunks.append(replacement["replacement"].encode("utf-8"))
        cursor = replacement["end_offset"]
    chunks.append(original[cursor:])
    materialized = b"".join(chunks)
    try:
        ast.parse(materialized.decode("utf-8"), filename=str(output_path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(
            f"source transform does not preserve valid Python syntax: {error}"
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
        "schema_version": ACTION_MANIFEST_SCHEMA,
        "trial_id": transform["trial_id"],
        "change_kind": action["change_kind"],
        "action_id": action["action_id"],
        "action_catalog_fingerprint": transform["action_catalog_fingerprint"],
        "expected_dispatch_change": action["expected_dispatch_change"],
        "immutable_input_source": transform["source_file"],
        "materialized_source": str(output_path),
        "input_source_fingerprint": file_fingerprint(source),
        "output_source_fingerprint": file_fingerprint(output_path),
        "source_transform_fingerprint": transform["transform_fingerprint"],
        "replacements": transform["replacements"],
        "single_stream_projection_fingerprint": transform[
            "single_stream_projection_fingerprint_before"
        ],
        "dependency_evidence_fingerprint_before": transform[
            "dependency_evidence_fingerprint_before"
        ],
        "dependency_evidence_fingerprint_after": transform[
            "dependency_evidence_fingerprint_after"
        ],
        "safety_proofs": transform["safety_proofs"],
        "stage_state_contract": transform["stage_state_contract"],
        "allowed_multistream_ranges": transform["allowed_multistream_ranges"],
        "multistream_only_verified": True,
        "single_change_verified": True,
    }


def wrap_scope_derivative(
    scope_action_manifest,
    candidate,
    parent_action_manifest,
    *,
    single_stream_projection_fingerprint_before,
    single_stream_projection_fingerprint_after,
    dependency_evidence_fingerprint_before,
    dependency_evidence_fingerprint_after,
):
    """Promote one already-materialized scope action into result-v3 lineage."""
    if (
        scope_action_manifest.get("schema_version")
        != "superkernel-multistream-action-manifest-v1"
    ):
        raise ValueError("scope derivative base action must use action-manifest-v1")
    if scope_action_manifest.get("change_kind") not in {
        "scope_split",
        "range_exclusion",
    }:
        raise ValueError("scope derivative base action must be one exact scope action")
    if (
        scope_action_manifest.get("single_change_verified") is not True
        or scope_action_manifest.get("source_adapter_validation") != "passed"
    ):
        raise ValueError("scope derivative base action lacks materializer verification")
    if (
        not isinstance(candidate, dict)
        or candidate.get("change_kind") != "scope_event_derivative"
    ):
        raise ValueError("scope derivative candidate kind is invalid")
    source_action = candidate.get("source_action")
    if (
        not isinstance(source_action, dict)
        or source_action.get("change_kind") != scope_action_manifest["change_kind"]
    ):
        raise ValueError(
            "scope derivative candidate differs from materialized scope factor"
        )
    if parent_action_manifest.get("schema_version") != ACTION_MANIFEST_SCHEMA:
        raise ValueError("scope derivative parent must use action-manifest-v2")
    if candidate.get("parent_action_id") != parent_action_manifest.get("action_id"):
        raise ValueError("scope derivative parent action identity mismatch")
    before = _text(
        single_stream_projection_fingerprint_before,
        "single_stream_projection_fingerprint_before",
    )
    after = _text(
        single_stream_projection_fingerprint_after,
        "single_stream_projection_fingerprint_after",
    )
    if before != after:
        raise ValueError("scope derivative must preserve single-stream projection")
    dependency_before = _text(
        dependency_evidence_fingerprint_before, "dependency_evidence_fingerprint_before"
    )
    dependency_after = _text(
        dependency_evidence_fingerprint_after, "dependency_evidence_fingerprint_after"
    )
    if dependency_before == dependency_after:
        raise ValueError(
            "scope derivative requires independently sealed post-transform dependency evidence"
        )
    result = {
        "schema_version": ACTION_MANIFEST_SCHEMA,
        "trial_id": _text(
            scope_action_manifest.get("trial_id"), "scope_action_manifest.trial_id"
        ),
        "change_kind": "scope_event_derivative",
        "action_id": _text(candidate.get("action_id"), "candidate.action_id"),
        "parent_action_id": candidate["parent_action_id"],
        "source_action": source_action,
        "base_scope_action_manifest_fingerprint": fingerprint(scope_action_manifest),
        "parent_action_manifest_fingerprint": fingerprint(parent_action_manifest),
        "immutable_input_source": scope_action_manifest.get("immutable_input_source"),
        "materialized_source": scope_action_manifest.get("materialized_source"),
        "input_source_fingerprint": scope_action_manifest.get(
            "input_source_fingerprint"
        ),
        "output_source_fingerprint": scope_action_manifest.get(
            "output_source_fingerprint"
        ),
        "single_stream_projection_fingerprint": before,
        "dependency_evidence_fingerprint_before": dependency_before,
        "dependency_evidence_fingerprint_after": dependency_after,
        "multistream_only_verified": True,
        "single_change_verified": True,
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize(
            json.loads(args.transform.read_text()),
            json.loads(args.catalog.read_text()),
            args.source_root,
            args.output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
