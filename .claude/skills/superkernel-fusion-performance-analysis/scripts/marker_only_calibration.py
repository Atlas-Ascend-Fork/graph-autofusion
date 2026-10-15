#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build and validate marker-only calibration evidence for an unmarked source tree."""

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
    resolve_evidence_path,
)


def _fingerprint_without(value, field):
    detached = copy.deepcopy(value)
    detached.pop(field, None)
    return canonical_sha256(detached)


def _manifest_units(manifest, label):
    units = {}
    for index, raw in enumerate(require_list(manifest.get("units"), f"{label}.units")):
        unit = require_object(raw, f"{label}.units[{index}]")
        unit_id = require_text(unit.get("unit_id"), f"{label}.units[{index}].unit_id")
        if unit_id in units:
            raise ValueError(f"{label} contains duplicate unit_id {unit_id}")
        units[unit_id] = unit
    return units


def _snapshot_files(snapshot_root, manifest, label):
    root = Path(snapshot_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} snapshot root must be a regular directory")
    observed = {}
    for relative, expected_sha in manifest.get("file_sha256", {}).items():
        path = resolve_evidence_path(root, relative, f"{label} source file")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} source file is missing: {relative}")
        content = path.read_bytes()
        if bytes_sha256(content) != require_sha256(expected_sha, f"{label} file SHA"):
            raise ValueError(f"{label} source SHA mismatch: {relative}")
        observed[relative] = content
    return observed


def _normalized_inputs(stable_manifest, calibration_manifest):
    stable = require_object(stable_manifest, "stable source manifest")
    calibration = require_object(calibration_manifest, "calibration source manifest")
    for value, label in ((stable, "stable"), (calibration, "calibration")):
        if value.get("manifest_fingerprint") != _fingerprint_without(
            value, "manifest_fingerprint"
        ):
            raise ValueError(f"{label} source manifest fingerprint mismatch")
    if stable.get("revision_role") != "stable_source":
        raise ValueError("marker-only calibration requires a stable_source manifest")
    if calibration.get("revision_role") != "calibration":
        raise ValueError("marker-only calibration requires a calibration manifest")
    if stable.get("base_revision") != calibration.get("base_revision"):
        raise ValueError("marker-only calibration base revisions differ")
    if stable.get("calibration_revision") != calibration.get("source_revision"):
        raise ValueError(
            "calibration manifest is not the declared calibration revision"
        )
    if stable.get("model_adapter") != calibration.get("model_adapter"):
        raise ValueError("marker-only calibration model adapter identity differs")
    if stable.get("block_templates") != calibration.get("block_templates"):
        raise ValueError("marker-only calibration block templates differ")
    stable_units = _manifest_units(stable, "stable")
    calibration_units = _manifest_units(calibration, "calibration")
    if set(stable_units) != set(calibration_units):
        raise ValueError("marker-only calibration unit sets differ")
    for unit_id in sorted(stable_units):
        stable_unit = stable_units[unit_id]
        calibration_unit = calibration_units[unit_id]
        for field in (
            "block_template_id",
            "source_symbol",
            "source_file",
            "syntax_tree_hash",
        ):
            if stable_unit.get(field) != calibration_unit.get(field):
                raise ValueError(
                    f"marker-only calibration unit {unit_id} {field} differs"
                )
    return stable, calibration, stable_units, calibration_units


def _build_file_proofs(stable_files, calibration_files, calibration_units):
    if set(stable_files) != set(calibration_files):
        raise ValueError("marker-only calibration source file sets differ")
    insertions_by_file = {name: [] for name in stable_files}
    for unit_id, unit in calibration_units.items():
        source_file = unit["source_file"]
        for marker_kind in ("begin", "end"):
            marker = require_object(
                unit.get("marker_operations", {}).get(marker_kind),
                f"calibration unit {unit_id} {marker_kind} marker",
            )
            if (
                "insertion_start_offset" not in marker
                or "insertion_end_offset" not in marker
            ):
                raise ValueError(
                    f"calibration unit {unit_id} {marker_kind} lacks insertion span"
                )
            start = marker["insertion_start_offset"]
            end = marker["insertion_end_offset"]
            content = calibration_files[source_file]
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not (0 <= start < end <= len(content))
            ):
                raise ValueError(
                    f"calibration unit {unit_id} {marker_kind} insertion span is invalid"
                )
            insertion_sha = bytes_sha256(content[start:end])
            if insertion_sha != marker.get("insertion_sha256"):
                raise ValueError(
                    f"calibration unit {unit_id} {marker_kind} insertion SHA mismatch"
                )
            insertions_by_file[source_file].append(
                {
                    "unit_id": unit_id,
                    "marker_kind": marker_kind,
                    "start_offset": start,
                    "end_offset": end,
                    "sha256": insertion_sha,
                }
            )
    proofs = []
    for source_file in sorted(stable_files):
        stable_content = stable_files[source_file]
        calibration_content = calibration_files[source_file]
        insertions = sorted(
            insertions_by_file[source_file], key=lambda item: item["start_offset"]
        )
        cursor = 0
        retained = []
        for insertion in insertions:
            if insertion["start_offset"] < cursor:
                raise ValueError(
                    f"marker-only calibration insertions overlap in {source_file}"
                )
            retained.append(calibration_content[cursor : insertion["start_offset"]])
            cursor = insertion["end_offset"]
        retained.append(calibration_content[cursor:])
        stripped = b"".join(retained)
        if stripped != stable_content:
            raise ValueError(
                f"calibration source does not reduce to stable source after marker removal: {source_file}"
            )
        proofs.append(
            {
                "source_file": source_file,
                "stable_sha256": bytes_sha256(stable_content),
                "calibration_sha256": bytes_sha256(calibration_content),
                "stripped_calibration_sha256": bytes_sha256(stripped),
                "insertions": insertions,
            }
        )
    return proofs


def build_marker_only_calibration(
    *,
    stable_manifest,
    calibration_manifest,
    stable_snapshot_root,
    calibration_snapshot_root,
):
    stable, calibration, stable_units, calibration_units = _normalized_inputs(
        stable_manifest, calibration_manifest
    )
    stable_files = _snapshot_files(stable_snapshot_root, stable, "stable")
    calibration_files = _snapshot_files(
        calibration_snapshot_root, calibration, "calibration"
    )
    result = {
        "schema_version": "1.0",
        "protocol": "marker_only_calibration_v1",
        "stable_source_revision": stable["source_revision"],
        "calibration_revision": calibration["source_revision"],
        "stable_source_manifest_fingerprint": stable["manifest_fingerprint"],
        "calibration_source_manifest_fingerprint": calibration["manifest_fingerprint"],
        "unit_ids": sorted(stable_units),
        "files": _build_file_proofs(stable_files, calibration_files, calibration_units),
    }
    result["bridge_fingerprint"] = canonical_sha256(result)
    return result


def validate_marker_only_calibration(
    value,
    *,
    stable_manifest,
    calibration_manifest,
    stable_snapshot_root,
    calibration_snapshot_root,
):
    value = require_object(value, "marker-only calibration bridge")
    if value.get("protocol") != "marker_only_calibration_v1":
        raise ValueError("marker-only calibration protocol is unsupported")
    if value.get("schema_version") != "1.0":
        raise ValueError("marker-only calibration schema version is unsupported")
    if value.get("bridge_fingerprint") != _fingerprint_without(
        value, "bridge_fingerprint"
    ):
        raise ValueError("marker-only calibration bridge fingerprint mismatch")
    rebuilt = build_marker_only_calibration(
        stable_manifest=stable_manifest,
        calibration_manifest=calibration_manifest,
        stable_snapshot_root=stable_snapshot_root,
        calibration_snapshot_root=calibration_snapshot_root,
    )
    if rebuilt != value:
        for expected, observed in zip(rebuilt.get("files", []), value.get("files", [])):
            if expected.get("stable_sha256") != observed.get("stable_sha256"):
                raise ValueError("marker-only calibration stable source SHA mismatch")
        raise ValueError("marker-only calibration bridge cannot be reproduced")
    return {"status": "exact", "unit_ids": copy.deepcopy(value["unit_ids"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stable-source-manifest", required=True)
    parser.add_argument("--calibration-source-manifest", required=True)
    parser.add_argument("--stable-source-snapshot-root", required=True)
    parser.add_argument("--calibration-source-snapshot-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = build_marker_only_calibration(
            stable_manifest=load_json(args.stable_source_manifest),
            calibration_manifest=load_json(args.calibration_source_manifest),
            stable_snapshot_root=args.stable_source_snapshot_root,
            calibration_snapshot_root=args.calibration_source_snapshot_root,
        )
        atomic_write_json(args.output, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
