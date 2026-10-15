#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collection-side freshness contract for structural association artifacts.

The manifest records protocol and filesystem freshness evidence. It does not
claim that fingerprints cryptographically prove which process produced bytes.
"""

import argparse
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import tempfile
import time
from pathlib import Path


SCHEMA_VERSION = "1.0"
MAX_JSON_DEPTH = 256
MAX_JSON_LOCATION_LENGTH = 96
COLLECTION_MARKER = ".collection-session.json"
WORKLOAD_ALIASES = {
    "model": ("model", "model_name"),
    "input": ("input", "inputs", "input_shape", "sequence_length"),
    "batch": ("batch", "batch_size"),
    "rank": ("rank", "rank_size", "world_size"),
    "mode": ("mode", "inference_mode", "prefill_decode_mode"),
    "warmup": ("warmup", "warmup_runs", "warmup_iterations"),
    "iterations": ("iterations", "iteration_count", "measurement_runs"),
    "runtime": ("runtime", "runtime_parameters", "runtime_options"),
}
ROLE_FILES = {
    "baseline_profile": {"kernel_details", "task_time", "trace_view"},
    "candidate_compat": {
        "sk_graph_origin",
        "sk_graph_updated",
        "sk_fused_nodes",
        "sk_scope_split",
        "super_kernel",
    },
    "candidate_verify": {
        "sk_graph_origin",
        "sk_graph_updated",
        "sk_fused_nodes",
        "sk_scope_split",
        "super_kernel",
    },
    "candidate_profile": {
        "kernel_details",
        "task_time",
        "trace_view",
        "sk_graph_origin",
        "sk_graph_updated",
        "sk_fused_nodes",
        "sk_scope_split",
        "super_kernel",
    },
}
OPTIONAL_FILE_ROLES = {"sk_prof"}
ALL_FILE_ROLES = set().union(*ROLE_FILES.values(), OPTIONAL_FILE_ROLES)
ROUND_SUFFIX = re.compile(r"(?:AUTO|BASE|P[1-9][0-9]*|FINAL)\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_SESSION_KEYS = {
    "schema_version",
    "capture_role",
    "capture_root",
    "session_fingerprint",
    "started_ns",
    "root_was_empty",
}
_MANIFEST_KEYS = {
    "schema_version",
    "capture_role",
    "capture_root",
    "capture",
    "source_revision",
    "config",
    "config_fingerprint",
    "control",
    "control_fingerprint",
    "producer",
    "workload",
    "workload_fingerprint",
    "files",
    "manifest_fingerprint",
}
_CAPTURE_KEYS = {
    "session_fingerprint",
    "started_ns",
    "finalized_ns",
    "root_was_empty",
}
_PRODUCER_KEYS = {
    "native_pid",
    "started_ns",
    "ended_ns",
    "command_fingerprint",
}
_FILE_RECORD_KEYS = {"path", "sha256", "size", "ctime_ns", "mtime_ns"}
_INPUT_EVIDENCE_KEYS = {
    "path",
    "dev",
    "ino",
    "size",
    "ctime_ns",
    "mtime_ns",
    "sha256",
}


def _json_location(path):
    if len(path) <= MAX_JSON_LOCATION_LENGTH:
        return path
    edge = (MAX_JSON_LOCATION_LENGTH - 3) // 2
    return f"{path[:edge]}...{path[-edge:]}"


def _validate_standard_json(value, path="$"):
    active_containers = set()
    stack = [("visit", value, path, 0)]
    while stack:
        operation, item, item_path, parent_depth = stack.pop()
        if operation == "leave":
            active_containers.remove(item)
            continue
        if item is None or isinstance(item, (str, bool, int)):
            continue
        if isinstance(item, float):
            if math.isfinite(item):
                continue
            raise ValueError(
                f"{_json_location(item_path)} must contain only standard JSON values"
            )
        if not isinstance(item, (list, dict)):
            raise ValueError(
                f"{_json_location(item_path)} must contain only standard JSON values"
            )

        depth = parent_depth + 1
        if depth > MAX_JSON_DEPTH:
            raise ValueError(
                f"{_json_location(item_path)} exceeds maximum JSON depth "
                f"{MAX_JSON_DEPTH} "
                "(JSON nesting limit)"
            )
        identity = id(item)
        if identity in active_containers:
            raise ValueError(
                f"{_json_location(item_path)} contains a cyclic JSON value"
            )
        active_containers.add(identity)
        stack.append(("leave", identity, None, None))
        if isinstance(item, list):
            for index in range(len(item) - 1, -1, -1):
                stack.append(("visit", item[index], f"{item_path}[{index}]", depth))
            continue
        pairs = list(item.items())
        for key, child in reversed(pairs):
            if not isinstance(key, str):
                raise ValueError(
                    f"{_json_location(item_path)} must contain only standard JSON values"
                )
            stack.append(("visit", child, f"{item_path}.{key}", depth))


def canonical_json(value):
    """Return deterministic JSON after rejecting Python-only/non-finite values."""
    _validate_standard_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError("value must contain only standard JSON values") from error


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_detached_json(value):
    return json.loads(canonical_json(value))


def normalize_workload(value):
    if not isinstance(value, dict):
        raise ValueError("workload must be an object")
    _validate_standard_json(value, "workload")
    normalized = {}
    for concept, aliases in WORKLOAD_ALIASES.items():
        present = [(name, value[name]) for name in aliases if name in value]
        if not present:
            raise ValueError(f"workload.{concept} is required")
        canonical_values = {canonical_json(item) for _, item in present}
        if len(canonical_values) != 1:
            raise ValueError(f"workload.{concept} aliases conflict")
        normalized[concept] = present[0][1]
    return json.loads(canonical_json(normalized))


def round_belongs_to_candidate(candidate_name, round_id):
    if not isinstance(candidate_name, str) or not isinstance(round_id, str):
        return False
    if not candidate_name or candidate_name != candidate_name.strip():
        return False
    if not round_id or round_id != round_id.strip():
        return False
    prefix = f"{candidate_name}-"
    return (
        round_id.startswith(prefix)
        and ROUND_SUFFIX.fullmatch(round_id[len(prefix) :]) is not None
    )


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path, label, *, max_bytes=None):
    data, evidence = _read_stable_input_bytes(path, label, max_bytes=max_bytes)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except RecursionError as error:
        raise ValueError(f"{label} exceeds JSON nesting limit") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} JSON: {error.msg}") from error
    _validate_standard_json(value, label)
    return value, evidence


def _trimmed_string(value, label):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a trimmed non-empty string")
    return value


def _integer(value, label, *, positive=False):
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value > 0 if positive else value >= 0)
    if not valid:
        domain = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a {domain} integer")
    return value


def _exact_keys(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"{label} has unknown fields: {', '.join(extra)}")


def _input_path_parts(path, label):
    try:
        lexical = os.fspath(path)
    except TypeError as error:
        raise ValueError(f"{label} path must be text") from error
    if not isinstance(lexical, str) or not lexical:
        raise ValueError(f"{label} path must be non-empty text")
    absolute = os.path.isabs(lexical)
    components = [part for part in lexical.split(os.sep) if part]
    if not components or lexical.endswith(os.sep):
        raise ValueError(f"{label} must be a regular file")
    canonical = Path(os.path.abspath(lexical))
    return canonical, os.sep if absolute else ".", components


def _open_input_regular_file(path, label):
    canonical, start, components = _input_path_parts(path, label)
    directory_fd = os.open(
        start,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptor = None
    try:
        for component in components[:-1]:
            next_fd, _ = _open_directory_component(directory_fd, component, label)
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            descriptor = os.open(
                components[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(f"{label} path contains a symlink") from error
            raise ValueError(f"{label} does not exist or cannot be opened") from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        return descriptor, directory_fd, components[-1], canonical, before
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
        raise


def _validated_max_bytes(max_bytes):
    if max_bytes is None:
        return None
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer or None")
    return max_bytes


def _byte_limit_text(max_bytes):
    mib = 1024 * 1024
    if max_bytes and max_bytes % mib == 0:
        return f"{max_bytes // mib} MiB ({max_bytes} bytes)"
    suffix = "byte" if max_bytes == 1 else "bytes"
    return f"{max_bytes} {suffix}"


def _read_input_descriptor_bytes(descriptor, label, max_bytes=None):
    max_bytes = _validated_max_bytes(max_bytes)
    chunks = []
    total = 0
    while True:
        block = os.read(descriptor, 64 * 1024)
        if not block:
            break
        total += len(block)
        if max_bytes is not None and total > max_bytes:
            raise ValueError(
                f"{label} exceeds byte limit {_byte_limit_text(max_bytes)}"
            )
        chunks.append(block)
    return b"".join(chunks)


def _input_evidence(path, file_stat, digest):
    return {
        "path": str(path),
        "dev": file_stat.st_dev,
        "ino": file_stat.st_ino,
        "size": file_stat.st_size,
        "ctime_ns": file_stat.st_ctime_ns,
        "mtime_ns": file_stat.st_mtime_ns,
        "sha256": digest,
    }


def _input_binding_signature(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_ctime_ns,
        file_stat.st_mtime_ns,
    )


def _verify_input_leaf_binding(parent_fd, leaf, descriptor_stat, label):
    try:
        leaf_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} path binding changed while reading") from error
    if _input_binding_signature(leaf_stat) != _input_binding_signature(descriptor_stat):
        raise ValueError(f"{label} path binding changed while reading")


def _read_stable_input_bytes(path, label, *, max_bytes=None):
    max_bytes = _validated_max_bytes(max_bytes)
    descriptor, parent_fd, leaf, canonical, before = _open_input_regular_file(
        path, label
    )
    try:
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(
                f"{label} exceeds byte limit {_byte_limit_text(max_bytes)}"
            )
        if max_bytes is None:
            data = _read_input_descriptor_bytes(descriptor, label)
        else:
            data = _read_input_descriptor_bytes(descriptor, label, max_bytes)
        after = os.fstat(descriptor)
        digest = hashlib.sha256(data).hexdigest()
        _verify_input_leaf_binding(parent_fd, leaf, after, label)
        if _input_binding_signature(before) != _input_binding_signature(after):
            raise ValueError(f"{label} changed while reading")
        return data, _input_evidence(canonical, after, digest)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _input_identity(evidence):
    return evidence["dev"], evidence["ino"]


def _revalidate_input_evidence(evidence, label, *, max_bytes=None):
    if not isinstance(evidence, dict) or set(evidence) != _INPUT_EVIDENCE_KEYS:
        raise ValueError(f"{label} input evidence is invalid")
    try:
        _, current = _read_stable_input_bytes(
            evidence["path"], label, max_bytes=max_bytes
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} input changed: {error}") from error
    if current != evidence:
        raise ValueError(f"{label} input changed after it was read")


def _revalidate_input_evidences(evidences):
    for label, evidence in evidences:
        _revalidate_input_evidence(evidence, label)


def _path_is_within(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _publication_path(path):
    candidate = Path(path)
    return candidate.parent.resolve(strict=False) / candidate.name


def _publication_entry_exists(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        try:
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(directory_fd)


def _fsync_directory(directory_fd):
    try:
        os.fsync(directory_fd)
    except OSError as error:
        unsupported = {errno.EINVAL, errno.EROFS}
        if hasattr(errno, "ENOTSUP"):
            unsupported.add(errno.ENOTSUP)
        if error.errno not in unsupported:
            raise


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    directory_fd = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.link(
                temporary_path.name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ValueError(f"output already exists: {path}") from error
        _fsync_directory(directory_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _session_payload(role, root, started_ns, root_was_empty):
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_role": role,
        "capture_root": str(root),
        "started_ns": started_ns,
        "root_was_empty": root_was_empty,
    }


def _session_record(role, root, started_ns):
    payload = _session_payload(role, root, started_ns, True)
    record = dict(payload)
    record["session_fingerprint"] = canonical_sha256(payload)
    return record


def _write_descriptor(descriptor, data):
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("unable to write collection evidence")
        offset += written


def _create_collection_marker(root_fd, record):
    data = (canonical_json(record) + "\n").encode("utf-8")
    descriptor = None
    temporary_name = None
    for _ in range(100):
        candidate = f".{COLLECTION_MARKER}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=root_fd,
            )
        except FileExistsError:
            continue
        temporary_name = candidate
        break
    if descriptor is None:
        raise OSError("unable to allocate collection marker temporary file")
    try:
        _write_descriptor(descriptor, data)
        os.fsync(descriptor)
        try:
            os.link(
                temporary_name,
                COLLECTION_MARKER,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            _validate_collection_marker(root_fd, record)
        else:
            _fsync_directory(root_fd)
            _validate_collection_marker(root_fd, record)
    finally:
        os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass


def _validate_session_record(record):
    _exact_keys(record, _SESSION_KEYS, "session record")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported session schema_version")
    role = record["capture_role"]
    if role not in ROLE_FILES:
        raise ValueError(f"unknown capture role: {role}")
    root_text = _trimmed_string(record["capture_root"], "capture_root")
    root_path = Path(root_text)
    if not root_path.is_absolute():
        raise ValueError("session capture_root must be absolute")
    try:
        root = root_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("capture root no longer exists") from error
    if str(root) != root_text or not root.is_dir() or root_path.is_symlink():
        raise ValueError("session capture_root must be a resolved regular directory")
    started_ns = _integer(record["started_ns"], "capture.started_ns")
    if record["root_was_empty"] is not True:
        raise ValueError("capture root was not empty at session start")
    expected = canonical_sha256(_session_payload(role, root, started_ns, True))
    if record["session_fingerprint"] != expected:
        raise ValueError("session fingerprint mismatch")
    return role, root


def _load_recoverable_session(session_path, role, root):
    try:
        record, evidence = _load_json(session_path, "session record")
        existing_role, existing_root = _validate_session_record(record)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"existing session_out is not a recoverable session record: {error}"
        ) from error
    if existing_role != role or existing_root != root:
        raise ValueError("session record does not match requested collection")
    return record, evidence


def _load_published_session(session_path, expected_record, role, root):
    record, evidence = _load_json(session_path, "session record")
    existing_role, existing_root = _validate_session_record(record)
    if record != expected_record or existing_role != role or existing_root != root:
        raise ValueError("published session record does not match collection")
    return record, evidence


def begin_collection_session(*, role, root, session_out):
    """Publish an external session, then idempotently publish its root marker."""
    if role not in ROLE_FILES:
        raise ValueError(f"unknown capture role: {role}")
    root_input = Path(root)
    if root_input.exists() and root_input.is_symlink():
        raise ValueError("capture root must not be a symlink")
    root_input.mkdir(parents=True, exist_ok=True)
    root_resolved = root_input.resolve(strict=True)
    if not root_resolved.is_dir():
        raise ValueError("capture root must be a directory")

    session_path = _publication_path(session_out)
    if _path_is_within(session_path, root_resolved):
        raise ValueError("session record must be outside the capture root")

    root_fd = os.open(
        root_resolved,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        entries = set(os.listdir(root_fd))
        if entries not in (set(), {COLLECTION_MARKER}):
            raise ValueError("capture root must be empty")
        marker_exists = entries == {COLLECTION_MARKER}
        session_exists = _publication_entry_exists(session_path)
        if marker_exists and not session_exists:
            raise ValueError(
                "collection marker exists without a recoverable session record"
            )

        if session_exists:
            record, session_evidence = _load_recoverable_session(
                session_path, role, root_resolved
            )
        else:
            started_ns = _integer(time.time_ns(), "started_ns")
            expected_record = _session_record(role, root_resolved, started_ns)
            _atomic_write_json(session_path, expected_record)
            record, session_evidence = _load_published_session(
                session_path, expected_record, role, root_resolved
            )

        _revalidate_input_evidence(session_evidence, "session record")
        _create_collection_marker(root_fd, record)
        if set(os.listdir(root_fd)) != {COLLECTION_MARKER}:
            raise ValueError(
                "capture root changed before collection marker was established"
            )
        _validate_collection_marker(root_fd, record)
        _revalidate_input_evidence(session_evidence, "session record")
        return record
    finally:
        os.close(root_fd)


def _validated_session(session):
    record, session_evidence = _load_json(session, "session record")
    _, root = _validate_session_record(record)
    if _path_is_within(Path(session_evidence["path"]), root):
        raise ValueError("session record must be outside the capture root")
    root_fd = _open_capture_root(root)
    try:
        _validate_collection_marker(root_fd, record)
    finally:
        os.close(root_fd)
    return record, session_evidence, root


def _parse_file_specs(file_specs):
    parsed = []
    for spec in file_specs:
        if not isinstance(spec, str) or "=" not in spec:
            raise ValueError("file must use ROLE=RELATIVE_PATH")
        file_role, relative_text = spec.split("=", 1)
        _trimmed_string(file_role, "file role")
        _trimmed_string(relative_text, "artifact path")
        if file_role not in ALL_FILE_ROLES:
            raise ValueError(f"unknown artifact file role: {file_role}")
        if any(not 0x20 <= ord(character) <= 0x7E for character in relative_text):
            raise ValueError("artifact path must use printable ASCII POSIX characters")
        if "\\" in relative_text:
            raise ValueError("artifact path must use POSIX '/' separators")
        if relative_text.startswith("/"):
            raise ValueError("artifact path must be relative")
        if relative_text == COLLECTION_MARKER:
            raise ValueError("collection marker is not a role artifact")
        parts = relative_text.split("/")
        if ".." in parts:
            raise ValueError("artifact path must not contain traversal ('..')")
        if any(part in {"", "."} for part in parts):
            raise ValueError("artifact path must be a canonical relative path")
        parsed.append((file_role, relative_text))
    return parsed


def _descriptor_signature(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_ctime_ns,
        file_stat.st_mtime_ns,
    )


def _directory_identity(file_stat):
    return (file_stat.st_dev, file_stat.st_ino)


def _open_capture_root(root):
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as error:
        raise ValueError(
            "capture root must be an openable regular directory"
        ) from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("capture root must be a regular directory")
    return descriptor


def _open_directory_component(parent_fd, name, label):
    descriptor = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            try:
                component_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError:
                component_stat = None
            if component_stat is not None and stat.S_ISLNK(component_stat.st_mode):
                raise ValueError(f"{label} path contains a symlink") from error
        raise ValueError(
            f"{label} path contains a missing or non-directory component"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    if not stat.S_ISDIR(file_stat.st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} path contains a non-directory component")
    return descriptor, file_stat


def _open_relative_regular_file(root_fd, relative_text, label):
    parts = relative_text.split("/")
    parent_fd = os.dup(root_fd)
    directory_identities = []
    descriptor = None
    try:
        for component in parts[:-1]:
            next_fd, directory_stat = _open_directory_component(
                parent_fd, component, label
            )
            os.close(parent_fd)
            parent_fd = next_fd
            directory_identities.append(_directory_identity(directory_stat))
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(f"{label} path contains a symlink") from error
            raise ValueError(f"{label} does not exist or cannot be opened") from error
        try:
            file_stat = os.fstat(descriptor)
        except Exception:
            os.close(descriptor)
            descriptor = None
            raise
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(descriptor)
            descriptor = None
            raise ValueError(f"{label} must be a regular file")
        return descriptor, parent_fd, parts, directory_identities, file_stat
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        raise


def _verify_relative_binding(
    root_fd,
    parts,
    expected_directories,
    expected_file_stat,
    label,
):
    current_fd = os.dup(root_fd)
    try:
        for component, expected_identity in zip(parts[:-1], expected_directories):
            next_fd, current_stat = _open_directory_component(
                current_fd, component, label
            )
            os.close(current_fd)
            current_fd = next_fd
            if _directory_identity(current_stat) != expected_identity:
                raise ValueError(f"{label} path changed while hashing")
        try:
            current_file_stat = os.stat(
                parts[-1], dir_fd=current_fd, follow_symlinks=False
            )
        except OSError as error:
            raise ValueError(f"{label} path changed while hashing") from error
        if not stat.S_ISREG(current_file_stat.st_mode) or _directory_identity(
            current_file_stat
        ) != _directory_identity(expected_file_stat):
            raise ValueError(f"{label} path changed while hashing")
    finally:
        os.close(current_fd)


def _read_descriptor_sha256(descriptor, label):
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _read_descriptor_bytes(descriptor, label):
    chunks = []
    total = 0
    while True:
        block = os.read(descriptor, 64 * 1024)
        if not block:
            break
        total += len(block)
        if total > 1024 * 1024:
            raise ValueError(f"{label} is unexpectedly large")
        chunks.append(block)
    return b"".join(chunks)


def _read_stable_relative_file(root_fd, relative_text, label, reader):
    descriptor, parent_fd, parts, directories, before = _open_relative_regular_file(
        root_fd, relative_text, label
    )
    try:
        value = reader(descriptor, relative_text)
        after = os.fstat(descriptor)
        _verify_relative_binding(root_fd, parts, directories, before, label)
        if _descriptor_signature(before) != _descriptor_signature(after):
            raise ValueError(f"{label} changed while hashing: {relative_text}")
        return value, after
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _collect_file_record(root_fd, relative_text, capture_started_ns):
    digest, file_stat = _read_stable_relative_file(
        root_fd,
        relative_text,
        "artifact",
        _read_descriptor_sha256,
    )
    if file_stat.st_ctime_ns < capture_started_ns:
        raise ValueError(f"artifact predates capture session: {relative_text}")
    return {
        "path": relative_text,
        "sha256": digest,
        "size": file_stat.st_size,
        "ctime_ns": file_stat.st_ctime_ns,
        "mtime_ns": file_stat.st_mtime_ns,
    }, _directory_identity(file_stat)


def _validate_collection_marker(root_fd, expected_record):
    marker_bytes, _ = _read_stable_relative_file(
        root_fd,
        COLLECTION_MARKER,
        "collection marker",
        _read_descriptor_bytes,
    )
    expected = (canonical_json(expected_record) + "\n").encode("utf-8")
    if marker_bytes != expected:
        raise ValueError("collection marker does not match session record")


def _scan_capture_directory(directory_fd, relative_prefix, started_ns):
    with os.scandir(directory_fd) as entries:
        snapshot = list(entries)
    for entry in snapshot:
        relative = f"{relative_prefix}/{entry.name}" if relative_prefix else entry.name
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"capture entry changed during scan: {relative}"
            ) from error
        if relative == COLLECTION_MARKER:
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ValueError("collection marker must be a regular file")
            continue
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ValueError(f"capture root contains symlink: {relative}")
        if stat.S_ISDIR(entry_stat.st_mode):
            child_fd, opened_stat = _open_directory_component(
                directory_fd, entry.name, "capture"
            )
            try:
                if _directory_identity(opened_stat) != _directory_identity(entry_stat):
                    raise ValueError(f"capture entry changed during scan: {relative}")
                if opened_stat.st_ctime_ns < started_ns:
                    raise ValueError(
                        f"capture content predates capture session: {relative}"
                    )
                _scan_capture_directory(child_fd, relative, started_ns)
            finally:
                os.close(child_fd)
            try:
                rebound_stat = os.stat(
                    entry.name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as error:
                raise ValueError(
                    f"capture entry changed during scan: {relative}"
                ) from error
            if not stat.S_ISDIR(rebound_stat.st_mode) or _directory_identity(
                rebound_stat
            ) != _directory_identity(opened_stat):
                raise ValueError(f"capture entry changed during scan: {relative}")
            continue
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ValueError(f"capture root contains unsafe special file: {relative}")
        try:
            file_fd = os.open(
                entry.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_fd,
            )
        except OSError as error:
            raise ValueError(
                f"capture entry changed during scan: {relative}"
            ) from error
        try:
            opened_stat = os.fstat(file_fd)
            if not stat.S_ISREG(opened_stat.st_mode) or _directory_identity(
                opened_stat
            ) != _directory_identity(entry_stat):
                raise ValueError(f"capture entry changed during scan: {relative}")
            if opened_stat.st_ctime_ns < started_ns:
                raise ValueError(
                    f"capture content predates capture session: {relative}"
                )
            rebound_stat = os.stat(
                entry.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(rebound_stat.st_mode) or _directory_identity(
                rebound_stat
            ) != _directory_identity(opened_stat):
                raise ValueError(f"capture entry changed during scan: {relative}")
        except OSError as error:
            raise ValueError(
                f"capture entry changed during scan: {relative}"
            ) from error
        finally:
            os.close(file_fd)


def _scan_capture_root(root_fd, started_ns):
    _scan_capture_directory(root_fd, "", started_ns)


def _validate_producer(native_pid, started_ns, ended_ns, command_fingerprint):
    native_pid = _integer(native_pid, "native_pid", positive=True)
    started_ns = _integer(started_ns, "started_ns")
    ended_ns = _integer(ended_ns, "ended_ns")
    if ended_ns <= started_ns:
        raise ValueError("ended_ns must be greater than started_ns")
    command_fingerprint = _valid_sha256(command_fingerprint, "command_fingerprint")
    return {
        "native_pid": native_pid,
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "command_fingerprint": command_fingerprint,
    }


def finalize_collection_manifest(
    *,
    session,
    source_revision,
    native_pid,
    started_ns,
    ended_ns,
    command_fingerprint,
    workload,
    config,
    control,
    file_specs,
    out,
):
    """Finalize one capture manifest from already-collected files."""
    session_record, session_evidence, root = _validated_session(session)
    source_revision = _trimmed_string(source_revision, "source_revision")
    producer = _validate_producer(native_pid, started_ns, ended_ns, command_fingerprint)
    capture_started_ns = session_record["started_ns"]
    if producer["started_ns"] < capture_started_ns:
        raise ValueError("producer run started before capture session")

    workload_value, workload_evidence = _load_json(workload, "workload")
    config_value, config_evidence = _load_json(config, "config")
    control_value, control_evidence = _load_json(control, "control")
    input_evidences = [
        ("session record", session_evidence),
        ("workload", workload_evidence),
        ("config", config_evidence),
        ("control", control_evidence),
    ]
    normalized_workload = normalize_workload(workload_value)
    normalized_config = _canonical_detached_json(config_value)
    normalized_control = _canonical_detached_json(control_value)

    parsed_specs = _parse_file_specs(file_specs)
    present_roles = {file_role for file_role, _ in parsed_specs}
    missing = sorted(ROLE_FILES[session_record["capture_role"]] - present_roles)
    if missing:
        raise ValueError(
            f"{session_record['capture_role']} missing required files: "
            + ", ".join(missing)
        )

    out_path = _publication_path(out)
    input_paths = {Path(evidence["path"]) for _, evidence in input_evidences}
    input_identities = {_input_identity(evidence) for _, evidence in input_evidences}
    if len(input_paths) != 4 or len(input_identities) != 4:
        raise ValueError(
            "session, workload, config, and control inputs must be distinct"
        )

    files = {}
    artifact_identities = set()
    artifact_paths = set()
    root_fd = _open_capture_root(root)
    try:
        _validate_collection_marker(root_fd, session_record)
        _scan_capture_root(root_fd, capture_started_ns)
        for file_role, relative_text in parsed_specs:
            record, artifact_identity = _collect_file_record(
                root_fd, relative_text, capture_started_ns
            )
            if artifact_identity in artifact_identities:
                raise ValueError(f"artifact input is repeated: {relative_text}")
            if artifact_identity in input_identities:
                raise ValueError(f"artifact overlaps a JSON input: {relative_text}")
            artifact_identities.add(artifact_identity)
            artifact_paths.add(root.joinpath(*relative_text.split("/")))
            files.setdefault(file_role, []).append(record)
    finally:
        os.close(root_fd)
    if out_path in input_paths or out_path in artifact_paths:
        raise ValueError("output overlaps an input")
    if out_path.parent.resolve(strict=True) != root:
        raise ValueError("manifest output must be directly inside the capture root")

    finalized_ns = _integer(time.time_ns(), "capture.finalized_ns")
    if finalized_ns <= capture_started_ns:
        raise ValueError("finalize time must be after capture session start")
    if producer["ended_ns"] > finalized_ns:
        raise ValueError("producer run ended after manifest finalization")
    for records in files.values():
        for record in records:
            if record["ctime_ns"] > finalized_ns:
                raise ValueError(
                    f"artifact ctime is after finalization: {record['path']}"
                )
    for records in files.values():
        records.sort(key=lambda item: item["path"])
    files = {key: files[key] for key in sorted(files)}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "capture_role": session_record["capture_role"],
        "capture_root": ".",
        "capture": {
            "session_fingerprint": session_record["session_fingerprint"],
            "started_ns": capture_started_ns,
            "finalized_ns": finalized_ns,
            "root_was_empty": True,
        },
        "source_revision": source_revision,
        "config": normalized_config,
        "config_fingerprint": canonical_sha256(normalized_config),
        "control": normalized_control,
        "control_fingerprint": canonical_sha256(normalized_control),
        "producer": producer,
        "workload": normalized_workload,
        "workload_fingerprint": canonical_sha256(normalized_workload),
        "files": files,
    }
    manifest["manifest_fingerprint"] = canonical_sha256(manifest)
    _revalidate_input_evidences(input_evidences)
    _atomic_write_json(out_path, manifest)
    return manifest


def _valid_sha256(value, label):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _producer_identity(manifest):
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        return None
    identity = (
        producer.get("native_pid"),
        producer.get("started_ns"),
        producer.get("command_fingerprint"),
    )
    try:
        hash(identity)
    except TypeError:
        return None
    return identity


def _duplicate(values):
    values = list(values)
    try:
        return len(values) != len(set(values))
    except TypeError:
        return False


def _validate_file_record(record, root_fd, capture_started_ns, finalized_ns):
    _exact_keys(record, _FILE_RECORD_KEYS, "file record")
    relative_text = _trimmed_string(record["path"], "file.path")
    parsed = _parse_file_specs([f"kernel_details={relative_text}"])[0][1]
    actual_sha256, current = _read_stable_relative_file(
        root_fd,
        parsed,
        "artifact",
        _read_descriptor_sha256,
    )
    if record["sha256"] != actual_sha256:
        raise ValueError(f"artifact content fingerprint changed: {relative_text}")
    expected_stat = (
        _integer(record["size"], "file.size"),
        _integer(record["ctime_ns"], "file.ctime_ns"),
        _integer(record["mtime_ns"], "file.mtime_ns"),
    )
    current_stat = (current.st_size, current.st_ctime_ns, current.st_mtime_ns)
    if expected_stat != current_stat:
        raise ValueError(f"artifact stat fingerprint changed: {relative_text}")
    _valid_sha256(record["sha256"], "file.sha256")
    if not capture_started_ns <= current.st_ctime_ns <= finalized_ns:
        raise ValueError(f"artifact ctime is outside capture session: {relative_text}")
    return _directory_identity(current)


def _validate_manifest(manifest, expected_role, root):
    _exact_keys(manifest, _MANIFEST_KEYS, f"{expected_role} manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{expected_role} schema_version mismatch")
    if manifest["capture_role"] != expected_role:
        raise ValueError(f"{expected_role} capture_role mismatch")
    if manifest["capture_root"] != ".":
        raise ValueError(f"{expected_role} capture_root must be '.'")
    _trimmed_string(manifest["source_revision"], "source_revision")
    config = _canonical_detached_json(manifest["config"])
    control = _canonical_detached_json(manifest["control"])
    if config != manifest["config"]:
        raise ValueError("config is not canonical JSON data")
    if control != manifest["control"]:
        raise ValueError("control is not canonical JSON data")
    _valid_sha256(manifest["config_fingerprint"], "config_fingerprint")
    _valid_sha256(manifest["control_fingerprint"], "control_fingerprint")
    if manifest["config_fingerprint"] != canonical_sha256(manifest["config"]):
        raise ValueError("config fingerprint mismatch")
    if manifest["control_fingerprint"] != canonical_sha256(manifest["control"]):
        raise ValueError("control fingerprint mismatch")

    capture = manifest["capture"]
    _exact_keys(capture, _CAPTURE_KEYS, "capture")
    if capture["root_was_empty"] is not True:
        raise ValueError("capture.root_was_empty must be true")
    capture_started_ns = _integer(capture["started_ns"], "capture.started_ns")
    finalized_ns = _integer(capture["finalized_ns"], "capture.finalized_ns")
    if finalized_ns <= capture_started_ns:
        raise ValueError("capture.finalized_ns must be greater than capture.started_ns")
    expected_session = _session_record(expected_role, root, capture_started_ns)
    if capture["session_fingerprint"] != expected_session["session_fingerprint"]:
        raise ValueError(f"{expected_role} session fingerprint mismatch")

    producer = manifest["producer"]
    _exact_keys(producer, _PRODUCER_KEYS, "producer")
    normalized_producer = _validate_producer(
        producer["native_pid"],
        producer["started_ns"],
        producer["ended_ns"],
        producer["command_fingerprint"],
    )
    if normalized_producer != producer:
        raise ValueError("producer is not canonical")
    if producer["started_ns"] < capture_started_ns:
        raise ValueError("producer run started before capture session")
    if producer["ended_ns"] > finalized_ns:
        raise ValueError("producer run ended after manifest finalization")

    workload = normalize_workload(manifest["workload"])
    if workload != manifest["workload"]:
        raise ValueError("workload is not normalized")
    if manifest["workload_fingerprint"] != canonical_sha256(workload):
        raise ValueError("workload fingerprint mismatch")

    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("files must be an object")
    unknown = sorted(set(files) - ALL_FILE_ROLES)
    if unknown:
        raise ValueError(f"files has unknown roles: {', '.join(unknown)}")
    missing = sorted(ROLE_FILES[expected_role] - set(files))
    if missing:
        raise ValueError(
            f"{expected_role} missing required files: {', '.join(missing)}"
        )
    root_fd = _open_capture_root(root)
    try:
        _validate_collection_marker(root_fd, expected_session)
        _scan_capture_root(root_fd, capture_started_ns)
        seen_identities = set()
        for file_role in sorted(files):
            records = files[file_role]
            if not isinstance(records, list) or not records:
                raise ValueError(f"files.{file_role} must be a non-empty array")
            paths = []
            for record in records:
                identity = _validate_file_record(
                    record, root_fd, capture_started_ns, finalized_ns
                )
                if identity in seen_identities:
                    raise ValueError(
                        f"artifact is repeated across file roles: {record['path']}"
                    )
                seen_identities.add(identity)
                paths.append(record["path"])
            if paths != sorted(paths) or len(paths) != len(set(paths)):
                raise ValueError(f"files.{file_role} is not canonical")
    finally:
        os.close(root_fd)

    payload = dict(manifest)
    fingerprint = payload.pop("manifest_fingerprint")
    _valid_sha256(fingerprint, "manifest_fingerprint")
    if fingerprint != canonical_sha256(payload):
        raise ValueError(f"{expected_role} manifest fingerprint mismatch")
    return manifest


def _require_matching(manifests, roles, key, label):
    values = [manifests[role][key] for role in roles]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{label} mismatch across collection manifests")
    return values[0]


def _validate_manifest_set_with_evidence(manifests, expected_roles=None):
    roles = tuple(ROLE_FILES) if expected_roles is None else tuple(expected_roles)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("expected_roles must contain unique collection roles")
    unknown_roles = sorted(set(roles) - set(ROLE_FILES))
    if unknown_roles:
        raise ValueError(f"unknown collection roles: {', '.join(unknown_roles)}")
    if not isinstance(manifests, dict) or set(manifests) != set(roles):
        role_list = ", ".join(roles)
        raise ValueError(f"expected exactly one manifest for roles: {role_list}")

    roots = {}
    loaded = {}
    evidences = {}
    for role in roles:
        value, evidence = _load_json(manifests[role], f"{role} manifest")
        roots[role] = Path(evidence["path"]).parent
        loaded[role] = value
        evidences[role] = evidence
    if _duplicate(_input_identity(value) for value in evidences.values()):
        raise ValueError("collection manifest inputs must be distinct")
    if _duplicate(roots.values()):
        raise ValueError("capture roots must be pairwise distinct")

    session_fingerprints = [
        value.get("capture", {}).get("session_fingerprint")
        if isinstance(value.get("capture"), dict)
        else None
        for value in loaded.values()
    ]
    if None not in session_fingerprints and _duplicate(session_fingerprints):
        raise ValueError("capture session fingerprints must be pairwise distinct")
    producer_identities = [_producer_identity(value) for value in loaded.values()]
    if None not in producer_identities and _duplicate(producer_identities):
        raise ValueError("producer identities must be pairwise distinct")

    for role in roles:
        _validate_manifest(loaded[role], role, roots[role])

    source_revision = _require_matching(
        loaded, roles, "source_revision", "source_revision"
    )
    _require_matching(loaded, roles, "config", "config")
    _require_matching(loaded, roles, "control", "control")
    workload_fingerprint = _require_matching(
        loaded, roles, "workload_fingerprint", "workload"
    )
    _require_matching(loaded, roles, "workload", "workload")
    config_fingerprint = _require_matching(
        loaded, roles, "config_fingerprint", "config"
    )
    control_fingerprint = _require_matching(
        loaded, roles, "control_fingerprint", "control"
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": source_revision,
        "workload_fingerprint": workload_fingerprint,
        "config_fingerprint": config_fingerprint,
        "control_fingerprint": control_fingerprint,
        "roles": {
            role: {
                "manifest_fingerprint": loaded[role]["manifest_fingerprint"],
                "session_fingerprint": loaded[role]["capture"]["session_fingerprint"],
            }
            for role in roles
        },
    }
    summary["set_fingerprint"] = canonical_sha256(summary)
    labeled_evidences = [(f"{role} manifest", evidences[role]) for role in roles]
    _revalidate_input_evidences(labeled_evidences)
    return summary, labeled_evidences


def validate_manifest_set(manifests, expected_roles=None):
    """Validate manifests while they remain bound to the files that were read.

    A caller can mutate files after this function returns; no filesystem API can
    extend this check beyond the function boundary.
    """
    summary, _ = _validate_manifest_set_with_evidence(
        manifests, expected_roles=expected_roles
    )
    return summary


def _parse_manifest_specs(specs):
    manifests = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("manifest must use ROLE=PATH")
        role, path = spec.split("=", 1)
        _trimmed_string(role, "manifest role")
        _trimmed_string(path, "manifest path")
        if role in manifests:
            raise ValueError(f"duplicate manifest role: {role}")
        manifests[role] = Path(path)
    return manifests


def _cli_expected_roles(manifests):
    provided = frozenset(manifests)
    profile_roles = frozenset(("baseline_profile", "candidate_profile"))
    full_roles = frozenset(ROLE_FILES)
    if provided not in (profile_roles, full_roles):
        raise ValueError(
            "validate-set requires exactly baseline_profile and candidate_profile "
            "manifests, or the complete four-role audit set"
        )
    return tuple(role for role in ROLE_FILES if role in provided)


def _validate_summary_output_path(path, manifest_evidences):
    output = _publication_path(path)
    for _, evidence in manifest_evidences:
        manifest_path = Path(evidence["path"])
        capture_root = manifest_path.parent
        if output == manifest_path or _path_is_within(output, capture_root):
            raise ValueError("validation output overlaps capture input")
    return output


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Create and validate structural-association collection manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    begin = subparsers.add_parser("begin")
    begin.add_argument("--role", required=True, choices=tuple(ROLE_FILES))
    begin.add_argument("--root", required=True, type=Path)
    begin.add_argument("--session-out", required=True, type=Path)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--session", required=True, type=Path)
    finalize.add_argument("--source-revision", required=True)
    finalize.add_argument("--native-pid", required=True, type=int)
    finalize.add_argument("--started-ns", required=True, type=int)
    finalize.add_argument("--ended-ns", required=True, type=int)
    finalize.add_argument("--command-fingerprint", required=True)
    finalize.add_argument("--workload", required=True, type=Path)
    finalize.add_argument("--config", required=True, type=Path)
    finalize.add_argument("--control", required=True, type=Path)
    finalize.add_argument("--file", required=True, action="append", dest="file_specs")
    finalize.add_argument("--out", required=True, type=Path)

    validate = subparsers.add_parser("validate-set")
    validate.add_argument("--manifest", required=True, action="append")
    validate.add_argument("--json-out", type=Path)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "begin":
            result = begin_collection_session(
                role=args.role, root=args.root, session_out=args.session_out
            )
            print(canonical_json(result))
            return 0
        if args.command == "finalize":
            result = finalize_collection_manifest(
                session=args.session,
                source_revision=args.source_revision,
                native_pid=args.native_pid,
                started_ns=args.started_ns,
                ended_ns=args.ended_ns,
                command_fingerprint=args.command_fingerprint,
                workload=args.workload,
                config=args.config,
                control=args.control,
                file_specs=args.file_specs,
                out=args.out,
            )
            print(canonical_json(result))
            return 0
        manifests = _parse_manifest_specs(args.manifest)
        result, manifest_evidences = _validate_manifest_set_with_evidence(
            manifests,
            expected_roles=_cli_expected_roles(manifests),
        )
        if args.json_out is None:
            print(canonical_json(result))
        else:
            output = _validate_summary_output_path(args.json_out, manifest_evidences)
            _revalidate_input_evidences(manifest_evidences)
            _atomic_write_json(output, result)
        return 0
    except (OSError, ValueError) as error:
        print(f"artifact_contract: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
