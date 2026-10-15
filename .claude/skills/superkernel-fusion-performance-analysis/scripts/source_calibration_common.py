#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared fail-closed helpers for source calibration artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


SHA256_PATTERN = frozenset("0123456789abcdef")


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def require_text(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def require_integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def require_scalar_id(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label} must be a non-empty string or integer")
    return require_text(str(value), label)


def require_sha256(value, label):
    text = require_text(value, label)
    if len(text) != 64 or any(character not in SHA256_PATTERN for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def load_json(path, label="JSON artifact"):
    artifact = Path(path)
    try:
        return json.loads(
            artifact.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains invalid constant {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_relative_path(value, label):
    text = require_text(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must be a normalized relative path without '..'")
    return path


def resolve_evidence_path(root, value, label):
    root = Path(root).resolve()
    relative = safe_relative_path(value, label)
    candidate = root.joinpath(relative)
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(f"{label} escapes the artifact root")
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} resolves outside the artifact root")
    return resolved


def load_hashed_json(root, reference, label):
    reference = require_object(reference, label)
    path = resolve_evidence_path(
        root, reference.get("relative_path"), f"{label}.relative_path"
    )
    expected = require_sha256(reference.get("sha256"), f"{label}.sha256")
    if not path.is_file():
        raise ValueError(f"{label} artifact does not exist: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, load_json(path, label)


def artifact_reference(path, *, relative_to):
    path = Path(path).resolve()
    root = Path(relative_to).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact {path} is outside {root}") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    return {
        "relative_path": relative.as_posix(),
        "sha256": file_sha256(path),
    }


def normalized_source_revision(value):
    return require_text(value, "source revision")


def validate_byte_span(span, file_size, label):
    span = require_object(span, label)
    start = require_integer(span.get("start_offset"), f"{label}.start_offset")
    end = require_integer(span.get("end_offset"), f"{label}.end_offset", minimum=1)
    if end <= start or end > file_size:
        raise ValueError(f"{label} must be an increasing span inside the source file")
    return start, end


def validate_exact_keys(value, allowed, label, *, required=()):
    value = require_object(value, label)
    missing = sorted(set(required) - set(value))
    unknown = sorted(set(value) - set(allowed))
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")
    return value


def atomic_write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        write_json(temporary, value)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
