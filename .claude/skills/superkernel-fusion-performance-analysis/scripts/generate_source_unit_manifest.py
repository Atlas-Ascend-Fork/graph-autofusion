#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate a model-independent source-unit manifest from adapter output."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from source_calibration_common import (
    atomic_write_json,
    bytes_sha256,
    canonical_sha256,
    load_json,
    require_list,
    require_object,
    require_sha256,
    require_text,
    safe_relative_path,
    validate_byte_span,
)


def _inside(inner, outer):
    return outer[0] <= inner[0] < inner[1] <= outer[1]


def build_manifest(spec, source_root, *, snapshot_root=None):
    spec = require_object(spec, "adapter output")
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise ValueError("source_root must be an existing directory")
    adapter = require_object(spec.get("model_adapter"), "model_adapter")
    normalized_adapter = {
        "name": require_text(adapter.get("name"), "model_adapter.name"),
        "version": require_text(adapter.get("version"), "model_adapter.version"),
        "fingerprint": require_sha256(
            adapter.get("fingerprint"), "model_adapter.fingerprint"
        ),
    }
    revision_role = require_text(spec.get("revision_role"), "revision_role")
    if revision_role not in {"calibration", "stable_marker", "stable_source"}:
        raise ValueError(
            "revision_role must be calibration, stable_marker, or stable_source"
        )
    source_revision = require_text(spec.get("source_revision"), "source_revision")
    stable_marker_revision = spec.get("stable_marker_revision")
    if stable_marker_revision is not None:
        stable_marker_revision = require_text(
            stable_marker_revision, "stable_marker_revision"
        )
    if revision_role == "stable_marker" and stable_marker_revision != source_revision:
        raise ValueError(
            "stable_marker revision requires source_revision == stable_marker_revision"
        )
    if revision_role == "calibration" and stable_marker_revision is not None:
        raise ValueError("calibration revision must not declare stable_marker_revision")
    if revision_role == "stable_source" and stable_marker_revision is not None:
        raise ValueError(
            "stable_source revision must not declare stable_marker_revision"
        )
    units = []
    file_bytes = {}
    unit_ids = set()
    spans_by_file = {}
    for index, raw in enumerate(require_list(spec.get("units"), "units")):
        unit = require_object(raw, f"units[{index}]")
        unit_id = require_text(unit.get("unit_id"), f"units[{index}].unit_id")
        if unit_id in unit_ids:
            raise ValueError(f"duplicate unit_id {unit_id}")
        unit_ids.add(unit_id)
        relative = safe_relative_path(
            unit.get("source_file"), f"units[{index}].source_file"
        )
        source_candidate = source_root.joinpath(relative)
        if source_candidate.is_symlink():
            raise ValueError(f"units[{index}].source_file must not be a symlink")
        source_file = source_candidate.resolve()
        if (
            source_root not in source_file.parents
            or not source_file.is_file()
            or source_file.is_symlink()
        ):
            raise ValueError(
                f"units[{index}].source_file is not a regular file under source_root"
            )
        content = file_bytes.setdefault(relative.as_posix(), source_file.read_bytes())
        envelope = validate_byte_span(unit, len(content), f"units[{index}]")
        normalized_markers = None
        if revision_role == "stable_source":
            if unit.get("marker_operations") is not None:
                raise ValueError(
                    f"units[{index}] stable_source must not contain marker operations"
                )
        else:
            marker_operations = require_object(
                unit.get("marker_operations"), f"units[{index}].marker_operations"
            )
            normalized_markers = {}
            for name, operation_kind in (
                ("begin", "scope_begin"),
                ("end", "scope_end"),
            ):
                marker = require_object(
                    marker_operations.get(name),
                    f"units[{index}].marker_operations.{name}",
                )
                marker_span = validate_byte_span(
                    marker, len(content), f"units[{index}].marker_operations.{name}"
                )
                if marker.get("operation_kind") != operation_kind or not _inside(
                    marker_span, envelope
                ):
                    raise ValueError(
                        f"units[{index}] has an invalid {name} marker operation"
                    )
                normalized_marker = {
                    "operation_kind": operation_kind,
                    "parser_node_id": require_text(
                        marker.get("parser_node_id"),
                        f"units[{index}].marker_operations.{name}.parser_node_id",
                    ),
                    "start_offset": marker_span[0],
                    "end_offset": marker_span[1],
                    "operation_sha256": bytes_sha256(
                        content[marker_span[0] : marker_span[1]]
                    ),
                }
                insertion_fields = (
                    marker.get("insertion_start_offset"),
                    marker.get("insertion_end_offset"),
                )
                if any(value is not None for value in insertion_fields):
                    insertion_span = validate_byte_span(
                        {
                            "start_offset": insertion_fields[0],
                            "end_offset": insertion_fields[1],
                        },
                        len(content),
                        f"units[{index}].marker_operations.{name}.insertion_span",
                    )
                    if not _inside(marker_span, insertion_span):
                        raise ValueError(
                            f"units[{index}] {name} operation is outside its insertion span"
                        )
                    normalized_marker.update(
                        {
                            "insertion_start_offset": insertion_span[0],
                            "insertion_end_offset": insertion_span[1],
                            "insertion_sha256": bytes_sha256(
                                content[insertion_span[0] : insertion_span[1]]
                            ),
                        }
                    )
                normalized_markers[name] = normalized_marker
            if (
                normalized_markers["begin"]["end_offset"]
                > normalized_markers["end"]["start_offset"]
            ):
                raise ValueError(
                    f"units[{index}] marker operations overlap or are reversed"
                )
        syntax = require_object(
            unit.get("normalized_syntax"), f"units[{index}].normalized_syntax"
        )
        normalized = {
            "unit_id": unit_id,
            "block_template_id": require_text(
                unit.get("block_template_id"), f"units[{index}].block_template_id"
            ),
            "source_symbol": require_text(
                unit.get("source_symbol"), f"units[{index}].source_symbol"
            ),
            "source_file": relative.as_posix(),
            "start_offset": envelope[0],
            "end_offset": envelope[1],
            "source_span_sha256": bytes_sha256(content[envelope[0] : envelope[1]]),
            "normalized_syntax": copy.deepcopy(syntax),
            "syntax_tree_hash": canonical_sha256(syntax),
            "applicable_structural_families": sorted(
                require_text(item, f"units[{index}].applicable_structural_families")
                for item in require_list(
                    unit.get("applicable_structural_families", []),
                    f"units[{index}].applicable_structural_families",
                )
            ),
        }
        if revision_role != "stable_source":
            normalized["scope_name_template"] = require_text(
                unit.get("scope_name_template"), f"units[{index}].scope_name_template"
            )
            normalized["marker_operations"] = normalized_markers
        units.append(normalized)
        spans_by_file.setdefault(relative.as_posix(), []).append((envelope, unit_id))
    for source_file, spans in spans_by_file.items():
        for index, (left, left_id) in enumerate(sorted(spans)):
            for right, right_id in sorted(spans)[index + 1 :]:
                if right[0] < left[1]:
                    raise ValueError(
                        f"source units {left_id} and {right_id} overlap in {source_file}"
                    )
    templates = []
    template_ids = set()
    for index, raw in enumerate(
        require_list(spec.get("block_templates"), "block_templates")
    ):
        item = require_object(raw, f"block_templates[{index}]")
        template_id = require_text(
            item.get("block_template_id"), f"block_templates[{index}].block_template_id"
        )
        if template_id in template_ids:
            raise ValueError(f"duplicate block_template_id {template_id}")
        template_ids.add(template_id)
        templates.append(
            {
                "block_template_id": template_id,
                "source_symbol": require_text(
                    item.get("source_symbol"), f"block_templates[{index}].source_symbol"
                ),
                "instance_binding": require_text(
                    item.get("instance_binding"),
                    f"block_templates[{index}].instance_binding",
                ),
                "binding_fingerprint": require_sha256(
                    item.get("binding_fingerprint"),
                    f"block_templates[{index}].binding_fingerprint",
                ),
            }
        )
    unknown_templates = sorted(
        {unit["block_template_id"] for unit in units} - template_ids
    )
    if unknown_templates:
        raise ValueError(
            f"units reference unknown block templates: {unknown_templates}"
        )
    manifest = {
        "schema_version": "1.0",
        "revision_role": revision_role,
        "base_revision": require_text(spec.get("base_revision"), "base_revision"),
        "source_revision": source_revision,
        "calibration_revision": require_text(
            spec.get("calibration_revision"), "calibration_revision"
        ),
        "stable_marker_revision": stable_marker_revision,
        "offset_encoding": "utf-8-byte-offset-v1",
        "source_language": require_text(spec.get("source_language"), "source_language"),
        "model_adapter": normalized_adapter,
        "file_sha256": {
            name: bytes_sha256(value) for name, value in sorted(file_bytes.items())
        },
        "algorithm_versions": copy.deepcopy(
            require_object(
                spec.get(
                    "algorithm_versions",
                    {
                        "source_parser": "adapter-parser-v1",
                        "syntax_tree_hash": "canonical-adapter-syntax-v1",
                        "marker_operation": "adapter-marker-operation-v1",
                        "source_span": "parser-node-byte-span-v1",
                    },
                ),
                "algorithm_versions",
            )
        ),
        "block_templates": sorted(
            templates, key=lambda item: item["block_template_id"]
        ),
        "units": sorted(units, key=lambda item: item["unit_id"]),
    }
    manifest["manifest_fingerprint"] = canonical_sha256(manifest)
    snapshot_manifest = {
        "schema_version": "1.0",
        "protocol": "source_snapshot_manifest_v1",
        "source_revision": source_revision,
        "files": [
            {
                "relative_path": name,
                "sha256": bytes_sha256(content),
                "size_bytes": len(content),
            }
            for name, content in sorted(file_bytes.items())
        ],
    }
    snapshot_manifest["manifest_fingerprint"] = canonical_sha256(snapshot_manifest)
    for name, content in file_bytes.items():
        if source_root.joinpath(name).read_bytes() != content:
            raise ValueError(
                f"source file changed while manifest was generated: {name}"
            )
    if snapshot_root is not None:
        snapshot_root = Path(snapshot_root)
        if snapshot_root.exists():
            raise ValueError("snapshot_root must not already exist")
        snapshot_root.mkdir(parents=True)
        for name in sorted(file_bytes):
            destination = snapshot_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file_bytes[name])
        atomic_write_json(
            snapshot_root / "source-snapshot-manifest.json", snapshot_manifest
        )
    return manifest, snapshot_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--adapter-output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest, _ = build_manifest(
            load_json(args.adapter_output, "adapter output"),
            args.source_root,
            snapshot_root=args.snapshot_root,
        )
        atomic_write_json(args.output, manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
