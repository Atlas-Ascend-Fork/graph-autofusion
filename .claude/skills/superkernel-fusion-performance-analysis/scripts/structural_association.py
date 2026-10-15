#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Canonical structural graphs for SuperKernel and profiler artifacts."""

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

from artifact_contract import (
    _load_json,
    _read_stable_input_bytes,
    _reject_json_constant,
    _revalidate_input_evidence,
    _unique_object,
    _validate_manifest,
    _validate_standard_json,
    canonical_json,
)


_CHILD_PREFIX = re.compile(r"^(?:\s*\[\s*\d+\s*/\s*\d+\s*\]\s*[-:]?\s*)+")
_STATIC_KERNEL = re.compile(r"(?:^|_)static_kernel_([A-Za-z][A-Za-z0-9]*)", re.ASCII)
_GENERATED_HASH = re.compile(r"_(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{64})(?=_|$)")
_OPAQUE_GENERATED_TAIL = re.compile(r"_(?:[0-9A-Za-z]{32}|[0-9A-Za-z]{64})$")
_ADDRESS = re.compile(r"(?:^|_)0x[0-9a-fA-F]+(?=_|$)")
_COMPILE_TAIL = re.compile(r"(?:_\d+_d\d+)?(?:__kernel\d+)?$")
_HIGH_PERFORMANCE = re.compile(r"_high_performance(?:_+\d+)?$")
_NUMERIC_TAIL = re.compile(r"_\d+$")
_PROFILER_SPECIALIZATION = re.compile(
    r"_(?:NDNZ|ND|NZ|FRACTAL_NZ|FP\d+|BF16|INT\d+|UINT\d+|BOOL|"
    r"normal|all|c\d+|double|false|true)(?:_|$)",
    re.IGNORECASE,
)
_COMMUNICATION = re.compile(
    r"(?:^|_)(?:all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|communication)(?:_|$)",
    re.IGNORECASE,
)
_SK_ENTRY_DUMP_WRAPPER = re.compile(
    r"^sk_entry_[A-Za-z0-9_]+_dump_profiling$", re.IGNORECASE
)
_MISSING_NAMES = {"", "N/A", "NA", "NONE", "NULL", "UNKNOWN"}
_MAX_STREAM_CANONICAL_LABELINGS = 1_000_000
_PROFILE_EVIDENCE_SEAL = object()
_PROFILE_MODEL_ID = re.compile(r"^(?P<model>[0-9]+)(?:_1)?$")
_FUSED_MODEL_COMPONENT = re.compile(r"^model_(?P<model>[0-9]+)(?:_1)?$")
_FUSED_DEVICE_COMPONENT = re.compile(r"^device_(?P<device>[0-9]+)$")
_FUSED_HEADER = re.compile(
    r"SK Function: (?P<name>[^,\r\n]+), scope id: (?P<scope>[0-9]+), "
    r"Node Count: (?P<count>[0-9]+)$"
)
_FUSED_LOCAL_ORIGIN = re.compile(
    r"\[(?P<local>[0-9]+)\]\s+\[nodeId:(?P<origin>[0-9]+),"
)
_FUSED_CANN_LOG_LINE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} "
    r"\[[0-9]+:[0-9]+\] \[INFO\] "
    r"\[sk_optimizer\.cpp:[0-9]+\] \[PrintSKNodesDetail\] +"
    r"(?P<payload>[^\r\n]+)$"
)
_FUSED_KERNEL_TYPES = frozenset(("AIC_ONLY", "AIV_ONLY", "MIX_1_1", "MIX_1_2"))

_MIB = 1024 * 1024
_CANDIDATE_MANIFEST_MAX_BYTES = 16 * _MIB
_ORIGIN_GRAPH_MAX_BYTES = 512 * _MIB
_KERNEL_DETAILS_MAX_BYTES = 512 * _MIB
_FUSED_GROUPS_MAX_BYTES = 256 * _MIB
_SUPER_KERNEL_LOG_MAX_BYTES = 512 * _MIB
_CANDIDATE_KERNEL_MAX_ROWS = 5_000_000
_CANDIDATE_TIMING_MAX_DIGITS = 64
_CANDIDATE_TIMING_MIN_EXPONENT = -30
_CANDIDATE_TIMING_MAX_ADJUSTED = 30
_CANDIDATE_TIMING_INTERNAL_MAX_DIGITS = _CANDIDATE_TIMING_MAX_DIGITS + 2
_CANDIDATE_TIMING_INTERNAL_MIN_EXPONENT = _CANDIDATE_TIMING_MIN_EXPONENT - 2
_CANDIDATE_ID_MAX_DIGITS = 19
_CANDIDATE_ID_MAX = (1 << 63) - 1
_CANDIDATE_SK_NAME_MAX_CHARS = 65_536
_ORIGIN_GRAPH_MAX_NODES = 1_000_000
_ORIGIN_GRAPH_MAX_STREAMS = 100_000
_FUSED_GROUPS_MAX_LINES = 5_000_000
_FUSED_METADATA_MAX_GROUPS = 100_000
_CANDIDATE_METADATA_MAX_NODES = 1_000_000
_FUSED_METADATA_MAX_FIELDS = 128
_FUSED_METADATA_MAX_LINE_CHARS = 65_536
_OCCURRENCE_MAX_DIGITS = 64
_OCCURRENCE_MIN_EXPONENT = -30
_OCCURRENCE_MAX_ADJUSTED = 30


_KERNEL_HEADERS = {
    "Step Id",
    "Device_id",
    "Model ID",
    "Task ID",
    "Stream ID",
    "Name",
    "Type",
    "Accelerator Core",
    "Start Time(us)",
    "Duration(us)",
    "Input Shapes",
    "Input Data Types",
    "Output Shapes",
    "Output Data Types",
}
_TASK_HEADERS = {
    "Device_id",
    "kernel_name",
    "kernel_type",
    "stream_id",
    "task_id",
    "task_time(us)",
    "task_start(us)",
    "task_stop(us)",
}
_CANDIDATE_KERNEL_HEADERS = {
    "Step Id",
    "Device_id",
    "Model ID",
    "Task ID",
    "Stream ID",
    "Name",
    "Type",
    "Accelerator Core",
    "Start Time(us)",
    "Duration(us)",
    "Block Num",
    "Mix Block Num",
}
_SHAPE_FIELDS = (
    ("Input Shapes", "input_shapes"),
    ("Input Data Types", "input_dtypes"),
    ("Output Shapes", "output_shapes"),
    ("Output Data Types", "output_dtypes"),
)


@dataclass(frozen=True)
class CanonicalNode:
    key: str
    model_role: str
    task_kind: str
    canonical_op: str
    core_family: str
    shape_dtype: tuple[str, ...]
    stream_key: str
    stream_ordinal: int
    topology_only: bool
    provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CanonicalEdge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class CanonicalGraph:
    model_role: str
    nodes: tuple[CanonicalNode, ...]
    edges: tuple[CanonicalEdge, ...]
    streams: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class FrozenJsonObject(Mapping):
    """An immutable, recursively detached JSON object."""

    entries: tuple[tuple[str, object], ...]

    def __getitem__(self, key):
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self.entries)

    def __len__(self):
        return len(self.entries)


@dataclass(frozen=True)
class InputEvidence:
    path: str
    dev: int
    ino: int
    size: int
    ctime_ns: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    relative_path: str
    sha256: str
    size: int
    ctime_ns: int
    mtime_ns: int
    device_id: int | None = None


@dataclass(frozen=True)
class ProducerIdentity:
    native_pid: int
    started_ns: int
    ended_ns: int
    command_fingerprint: str


@dataclass(frozen=True)
class ProfileBindingIdentity:
    manifest_path: str
    capture_root: str
    manifest_fingerprint: str
    session_fingerprint: str
    producer: ProducerIdentity


@dataclass(frozen=True)
class EvidenceBoundSource:
    record: ArtifactRecord
    evidence: InputEvidence


@dataclass(frozen=True)
class LoadedCandidateProfileManifest:
    capture_role: str
    capture_root: str
    manifest_fingerprint: str
    session_fingerprint: str
    producer: ProducerIdentity
    files: tuple[tuple[str, tuple[ArtifactRecord, ...]], ...]
    manifest_value: FrozenJsonObject
    evidence: InputEvidence
    _seal: object = field(repr=False, compare=False)

    def records_for(self, role):
        for candidate, records in self.files:
            if candidate == role:
                return records
        return ()

    @property
    def binding(self):
        return ProfileBindingIdentity(
            self.evidence.path,
            self.capture_root,
            self.manifest_fingerprint,
            self.session_fingerprint,
            self.producer,
        )


@dataclass(frozen=True)
class EvidenceBoundGraph:
    device_id: int
    model_id: str
    graph: CanonicalGraph
    source: EvidenceBoundSource


@dataclass(frozen=True)
class ProfileGraphSet:
    binding: ProfileBindingIdentity
    graphs: tuple[EvidenceBoundGraph, ...]
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class CandidateBoundary:
    source_scope: str
    start_function: str
    end_function: str
    start_op: str
    end_op: str


@dataclass(frozen=True)
class CandidateKernelRow:
    step_id: int
    device_id: int
    model_id: int
    sk_id: int
    task_id: int
    stream_id: int
    name: str
    boundary: CandidateBoundary
    raw_core: str
    core_family: str
    block_num: int
    mix_block_num: int
    start_us: Decimal
    duration_us: Decimal
    source_relative_path: str
    source_row: int


@dataclass(frozen=True)
class CandidateKernelRows:
    binding: ProfileBindingIdentity
    rows: tuple[CandidateKernelRow, ...]
    sources: tuple[EvidenceBoundSource, ...]
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class EvidenceBoundText:
    source: EvidenceBoundSource
    text: str

    @property
    def record(self):
        return self.source.record

    @property
    def evidence(self):
        return self.source.evidence


@dataclass(frozen=True)
class FusedGroups:
    binding: ProfileBindingIdentity
    sources: tuple[EvidenceBoundText, ...]
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _CandidateFusedMetadataNode:
    local_index: int
    origin_node_id: int
    node_key: str
    canonical_op: str
    expected_core_families: tuple[str, ...]
    num_blocks: int | None
    cube_num: int | None
    vec_num: int | None
    topology_only: bool
    task_kind: str


@dataclass(frozen=True, slots=True)
class _CandidateFusedMetadataGroup:
    device_id: int | None
    model_id: int | None
    sk_id: int | None
    name: str
    boundary: CandidateBoundary | None
    nodes: tuple[_CandidateFusedMetadataNode, ...]
    kernel_nodes: tuple[_CandidateFusedMetadataNode, ...]
    graph_fingerprint: str | None
    fused_fingerprint: str
    blockers: tuple[str, ...]


def canonical_op(value):
    """Remove generated launch identity while preserving the operator type."""
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    if text.upper() in _MISSING_NAMES:
        return "UNKNOWN"
    text = _CHILD_PREFIX.sub("", text).strip()
    if _SK_ENTRY_DUMP_WRAPPER.fullmatch(text):
        return "UNKNOWN"
    generated_match = _GENERATED_HASH.search(text)
    if generated_match:
        text = text[: generated_match.start()]
    text = _ADDRESS.sub("", text)
    while True:
        previous = text
        text = _COMPILE_TAIL.sub("", text)
        text = _HIGH_PERFORMANCE.sub("", text)
        text = _NUMERIC_TAIL.sub("", text)
        text = _OPAQUE_GENERATED_TAIL.sub("", text)
        text = text.rstrip("_")
        if text == previous:
            break
    if _COMMUNICATION.search(text):
        return text
    static_match = _STATIC_KERNEL.search(text)
    if static_match:
        return static_match.group(1)
    specialization = _PROFILER_SPECIALIZATION.search(text)
    if specialization:
        text = text[: specialization.start()]
    return text.strip("_ ") or "UNKNOWN"


def canonical_core_family(value):
    """Map native core and control labels into stable structural families."""
    if value is None:
        return "UNKNOWN"
    text = str(value).strip()
    upper = text.upper()
    if upper in {"AIV_ONLY", "AI_VECTOR_CORE", "MIX_AIV"}:
        return "AI_VECTOR_CORE"
    if upper in {"AIC_ONLY", "AI_CORE", "MIX_AIC", "MIX_1_1", "MIX_1_2"}:
        return "AI_CORE"
    if upper in {"WAIT", "NOTIFY", "RESET"} or upper.startswith(("EVENT", "MODEL")):
        return "CONTROL"
    if upper == "COMMUNICATION" or _COMMUNICATION.search(text):
        return "COMMUNICATION"
    return "UNKNOWN"


def _communication_kind(value):
    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    for kind in ("reducescatter", "allreduce", "allgather", "alltoallv", "alltoall"):
        if kind in compact:
            return kind
    return None


def _ops_agree(preferred, observed):
    if preferred == observed:
        return True
    preferred_collective = _communication_kind(preferred)
    return (
        preferred_collective is not None
        and preferred_collective == _communication_kind(observed)
    )


def _observed_op(value):
    if value is None:
        return None
    op = canonical_op(value)
    return None if op == "UNKNOWN" else op


def _model_role(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("model_role must be a trimmed non-empty string")
    return value


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _identity(value, label):
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        raise ValueError(f"{label} must be a scalar identity")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty scalar identity")
    return text


def _optional_identity(value, label):
    if value is None or str(value).strip() == "":
        return None
    return _identity(value, label)


def _node_reference(value, label):
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return _identity(value, label)


def _provenance(**values):
    result = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            text = canonical_json(value)
        elif isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        result.append((str(key), text))
    return tuple(sorted(result))


def _raw_parameter_evidence(value):
    evidence = {}
    stack = [("kernelParams", value)]
    while stack:
        path, item = stack.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                lowered = key.lower()
                if any(
                    token in lowered
                    for token in ("addr", "address", "handle", "ptr", "hash")
                ):
                    evidence[child_path] = child
                if isinstance(child, (dict, list)):
                    stack.append((child_path, child))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if isinstance(child, (dict, list)):
                    stack.append((f"{path}[{index}]", child))
    return evidence


def _natural_identity(value):
    text = str(value)
    if text.isdecimal():
        return (0, int(text), text)
    return (1, text)


def graph_from_sk_origin(path, *, model_role):
    """Parse an sk_graph_origin JSON artifact into an immutable graph."""
    value, input_evidence = _load_json(
        path, "sk_graph_origin", max_bytes=_ORIGIN_GRAPH_MAX_BYTES
    )
    graph = _graph_from_sk_origin_value(value, model_role=model_role)
    _revalidate_input_evidence(
        input_evidence, "sk_graph_origin", max_bytes=_ORIGIN_GRAPH_MAX_BYTES
    )
    return graph


def _graph_from_sk_origin_value(value, *, model_role):
    """Build a canonical graph from one already-read origin JSON value."""
    role = _model_role(model_role)
    document = _mapping(value, "sk_graph_origin")
    for field_name in (
        "version",
        "modelId",
        "deviceId",
        "totalStreams",
        "totalNodes",
        "streams",
    ):
        if field_name not in document:
            raise ValueError(f"sk_graph_origin missing field: {field_name}")
    model_id = _identity(document["modelId"], "modelId")
    device_id = _identity(document["deviceId"], "deviceId")
    total_streams = _integer(document["totalStreams"], "totalStreams")
    total_nodes = _integer(document["totalNodes"], "totalNodes")
    if total_streams > _ORIGIN_GRAPH_MAX_STREAMS:
        raise ValueError(
            f"sk_graph_origin exceeds stream limit {_ORIGIN_GRAPH_MAX_STREAMS}"
        )
    if total_nodes > _ORIGIN_GRAPH_MAX_NODES:
        raise ValueError(
            f"sk_graph_origin exceeds node limit {_ORIGIN_GRAPH_MAX_NODES}"
        )
    stream_values = _list(document["streams"], "streams")
    if len(stream_values) > _ORIGIN_GRAPH_MAX_STREAMS:
        raise ValueError(
            f"sk_graph_origin exceeds actual stream limit {_ORIGIN_GRAPH_MAX_STREAMS}"
        )
    if len(stream_values) != total_streams:
        raise ValueError("totalStreams does not match streams")

    stream_containers = []
    stream_ids = set()
    observed_nodes = 0
    for stream_index, stream_value in enumerate(stream_values):
        stream = _mapping(stream_value, f"streams[{stream_index}]")
        for field_name in ("streamIdxInGraph", "nodeCount", "nodes"):
            if field_name not in stream:
                raise ValueError(f"streams[{stream_index}] missing field: {field_name}")
        stream_id = _identity(stream["streamIdxInGraph"], "streamIdxInGraph")
        if stream_id in stream_ids:
            raise ValueError(f"duplicate streamIdxInGraph: {stream_id}")
        stream_ids.add(stream_id)
        declared_count = _integer(stream["nodeCount"], "nodeCount")
        node_values = _list(stream["nodes"], f"streams[{stream_index}].nodes")
        if len(node_values) != declared_count:
            raise ValueError(f"nodeCount mismatch for stream {stream_id}")
        observed_nodes += len(node_values)
        if observed_nodes > _ORIGIN_GRAPH_MAX_NODES:
            raise ValueError(
                f"sk_graph_origin exceeds actual node limit {_ORIGIN_GRAPH_MAX_NODES}"
            )
        stream_containers.append((stream_index, stream_id, declared_count, node_values))

    if observed_nodes != total_nodes:
        raise ValueError("totalNodes does not match stream node counts")

    parsed_streams = []
    node_ids = set()
    event_by_key = {}
    for stream_index, stream_id, declared_count, node_values in stream_containers:
        parsed_nodes = []
        stream_dependencies = []
        ordinals = set()
        for node_index, node_value in enumerate(node_values):
            label = f"streams[{stream_index}].nodes[{node_index}]"
            node = _mapping(node_value, label)
            required = {
                "nodeId",
                "streamIdxInGraph",
                "nodeIdxInStream",
                "nodeType",
                "preNodeId",
                "nextNodeId",
                "taskInfo",
            }
            missing = sorted(required - set(node))
            if missing:
                raise ValueError(f"{label} missing fields: {', '.join(missing)}")
            node_id = _identity(node["nodeId"], "nodeId")
            if node_id in node_ids:
                raise ValueError(f"duplicate nodeId: {node_id}")
            node_ids.add(node_id)
            node_stream = _identity(node["streamIdxInGraph"], "node streamIdxInGraph")
            if node_stream != stream_id:
                raise ValueError(
                    f"{label} streamIdxInGraph conflicts with parent stream"
                )
            ordinal = _integer(node["nodeIdxInStream"], "node ordinal")
            if ordinal in ordinals:
                raise ValueError(f"duplicate ordinal {ordinal} in stream {stream_id}")
            ordinals.add(ordinal)
            task_kind = _identity(node["nodeType"], "nodeType").upper()
            previous_node = _node_reference(node["preNodeId"], "preNodeId")
            next_node = _node_reference(node["nextNodeId"], "nextNodeId")
            task_infos = _list(node["taskInfo"], f"{label}.taskInfo")
            if not task_infos:
                raise ValueError(f"{label}.taskInfo must not be empty")
            physical_stream_ids = []
            for info_index, info_value in enumerate(task_infos):
                info = _mapping(info_value, f"{label}.taskInfo[{info_index}]")
                info_kind = _identity(info.get("taskType"), "task type").upper()
                if info_kind != task_kind:
                    raise ValueError(f"{label} task type conflicts with nodeType")
                physical_stream_ids.append(
                    _identity(info.get("streamId"), "taskInfo physical streamId")
                )
            if len(set(physical_stream_ids)) != 1:
                raise ValueError(f"{label} taskInfo physical streamId values conflict")

            raw_func = None
            raw_core = None
            raw_parameters = None
            event_id = None
            if task_kind == "KERNEL":
                kernel_params = []
                for info in task_infos:
                    params = _mapping(info.get("kernelParams"), f"{label}.kernelParams")
                    for field_name in ("funcName", "kernelType", "numBlocks"):
                        if field_name not in params:
                            raise ValueError(
                                f"{label}.kernelParams missing field: {field_name}"
                            )
                    _integer(params["numBlocks"], "numBlocks")
                    kernel_params.append(params)
                func_names = {
                    _identity(params["funcName"], "funcName")
                    for params in kernel_params
                }
                core_names = {
                    _identity(params["kernelType"], "kernelType")
                    for params in kernel_params
                }
                if len(func_names) != 1 or len(core_names) != 1:
                    raise ValueError(f"{label} kernel taskInfo entries conflict")
                raw_func = next(iter(func_names))
                raw_core = next(iter(core_names))
                raw_parameters = _raw_parameter_evidence(kernel_params[0])
                op = canonical_op(raw_func)
                core = canonical_core_family(raw_core)
                if canonical_core_family(op) == "COMMUNICATION":
                    core = "COMMUNICATION"
                topology_only = False
            else:
                event_ids = set()
                for info in task_infos:
                    params = info.get("eventParams")
                    if params is None:
                        continue
                    params = _mapping(params, f"{label}.eventParams")
                    if "eventId" in params:
                        event_ids.add(_identity(params["eventId"], "eventId"))
                if len(event_ids) > 1:
                    raise ValueError(f"{label} eventId values conflict")
                event_id = next(iter(event_ids), None)
                op = canonical_op(task_kind)
                core = canonical_core_family(task_kind)
                topology_only = True

            key = f"sk:node:{node_id}"
            stream_key = f"sk:stream:{stream_id}"
            task_ids = [
                _optional_identity(info.get("taskId"), "taskId") for info in task_infos
            ]
            provenance = _provenance(
                source="sk_graph_origin",
                model_id=model_id,
                device_id=device_id,
                logical_stream_id=stream_id,
                physical_stream_ids=physical_stream_ids,
                node_id=node_id,
                task_ids=task_ids,
                event_id=event_id,
                func_name=raw_func,
                kernel_type=raw_core,
                raw_parameters=raw_parameters,
                pre_node_id=node["preNodeId"],
                next_node_id=node["nextNodeId"],
            )
            parsed_nodes.append(
                CanonicalNode(
                    key=key,
                    model_role=role,
                    task_kind=task_kind,
                    canonical_op=op,
                    core_family=core,
                    shape_dtype=(),
                    stream_key=stream_key,
                    stream_ordinal=ordinal,
                    topology_only=topology_only,
                    provenance=provenance,
                )
            )
            event_by_key[key] = event_id
            stream_dependencies.append((ordinal, node_id, previous_node, next_node))
        if sorted(ordinals) != list(range(declared_count)):
            raise ValueError(f"node ordinal sequence is invalid for stream {stream_id}")
        stream_dependencies.sort(key=lambda item: item[0])
        for index, (_, node_id, previous_node, next_node) in enumerate(
            stream_dependencies
        ):
            expected_previous = stream_dependencies[index - 1][1] if index else None
            expected_next = (
                stream_dependencies[index + 1][1]
                if index + 1 < len(stream_dependencies)
                else None
            )
            if previous_node != expected_previous:
                raise ValueError(
                    f"node {node_id} preNodeId conflicts with same-stream adjacency"
                )
            if next_node != expected_next:
                raise ValueError(
                    f"node {node_id} nextNodeId conflicts with same-stream adjacency"
                )
        parsed_nodes.sort(key=lambda item: item.stream_ordinal)
        parsed_streams.append((stream_id, tuple(parsed_nodes)))

    parsed_streams.sort(key=lambda item: _natural_identity(item[0]))
    opaque_streams = []
    opaque_events = {}
    next_node_index = 0
    for stream_index, (_, stream_nodes) in enumerate(parsed_streams):
        stream_key = f"sk:s{stream_index:06d}"
        opaque_nodes = []
        for node in stream_nodes:
            key = f"sk:n{next_node_index:06d}"
            next_node_index += 1
            opaque_node = replace(node, key=key, stream_key=stream_key)
            opaque_nodes.append(opaque_node)
            opaque_events[key] = event_by_key[node.key]
        opaque_streams.append((stream_key, tuple(opaque_nodes)))
    parsed_streams = opaque_streams
    event_by_key = opaque_events
    nodes = tuple(node for _, stream_nodes in parsed_streams for node in stream_nodes)
    streams = tuple(
        (stream_key, tuple(node.key for node in stream_nodes))
        for stream_key, stream_nodes in parsed_streams
    )
    edges = []
    for _, stream_nodes in parsed_streams:
        edges.extend(
            CanonicalEdge(left.key, right.key, "stream_order")
            for left, right in zip(stream_nodes, stream_nodes[1:])
        )
    producers = defaultdict(list)
    waits = defaultdict(list)
    for node in nodes:
        event_id = event_by_key[node.key]
        if event_id is None:
            continue
        if node.task_kind == "NOTIFY":
            producers[event_id].append(node.key)
        elif node.task_kind == "WAIT":
            waits[event_id].append(node.key)
    for event_id in sorted(producers):
        if len(producers[event_id]) > 1:
            raise ValueError(f"event {event_id} has multiple NOTIFY producers")
    for event_id in sorted(set(producers) & set(waits)):
        source = producers[event_id][0]
        for target in sorted(waits[event_id]):
            edges.append(CanonicalEdge(source, target, "event"))
    return CanonicalGraph(role, nodes, tuple(edges), streams)


def _csv_rows_from_bytes(data, label, required_headers, *, max_rows=None):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 CSV") from error
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(
                f"{label} missing headers: {', '.join(sorted(required_headers))}"
            )
        if len(headers) != len(set(headers)):
            raise ValueError(f"{label} has duplicate headers")
        missing = sorted(required_headers - set(headers))
        if missing:
            raise ValueError(f"{label} missing headers: {', '.join(missing)}")
        rows = []
        for index, row in enumerate(reader, start=2):
            if max_rows is not None and index - 1 > max_rows:
                raise ValueError(f"{label} exceeds row limit {max_rows}")
            if None in row:
                raise ValueError(f"{label} row {index} has extra columns")
            rows.append(row)
        return rows
    except csv.Error as error:
        raise ValueError(f"invalid {label} CSV: {error}") from error


def _load_csv(path, label, required_headers):
    data, evidence = _read_stable_input_bytes(path, label)
    return _csv_rows_from_bytes(data, label, required_headers), evidence


def _decimal(value, label):
    text = "" if value is None else str(value).strip()
    try:
        result = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{label} must be finite timing") from error
    if not result.is_finite():
        raise ValueError(f"{label} must be finite timing")
    return result


def _decimal_text(value):
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result or "0"


def _freeze_json(value, *, label="JSON", depth=0):
    if depth > 256:
        raise ValueError(f"{label} exceeds maximum JSON depth 256")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{label} contains non-finite timing")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains non-finite number")
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(item, label=label, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{label} object keys must be strings")
        return FrozenJsonObject(
            tuple(
                (key, _freeze_json(item, label=label, depth=depth + 1))
                for key, item in sorted(value.items())
            )
        )
    raise ValueError(f"{label} contains a non-JSON value")


def _standard_json_from_bytes(data, label):
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
    return value


def _input_evidence_value(value):
    return InputEvidence(
        path=value["path"],
        dev=value["dev"],
        ino=value["ino"],
        size=value["size"],
        ctime_ns=value["ctime_ns"],
        mtime_ns=value["mtime_ns"],
        sha256=value["sha256"],
    )


def _input_evidence_dict(value):
    return {
        "path": value.path,
        "dev": value.dev,
        "ino": value.ino,
        "size": value.size,
        "ctime_ns": value.ctime_ns,
        "mtime_ns": value.mtime_ns,
        "sha256": value.sha256,
    }


def _record_device_id(role, relative_path):
    return None


def _manifest_wrapper(value, evidence):
    files = []
    for role in sorted(value["files"]):
        records = tuple(
            ArtifactRecord(
                role=role,
                relative_path=record["path"],
                sha256=record["sha256"],
                size=record["size"],
                ctime_ns=record["ctime_ns"],
                mtime_ns=record["mtime_ns"],
                device_id=_record_device_id(role, record["path"]),
            )
            for record in value["files"][role]
        )
        files.append((role, records))
    producer = value["producer"]
    return LoadedCandidateProfileManifest(
        capture_role=value["capture_role"],
        capture_root=str(Path(evidence["path"]).parent),
        manifest_fingerprint=value["manifest_fingerprint"],
        session_fingerprint=value["capture"]["session_fingerprint"],
        producer=ProducerIdentity(
            producer["native_pid"],
            producer["started_ns"],
            producer["ended_ns"],
            producer["command_fingerprint"],
        ),
        files=tuple(files),
        manifest_value=_freeze_json(value, label="candidate_profile manifest"),
        evidence=_input_evidence_value(evidence),
        _seal=_PROFILE_EVIDENCE_SEAL,
    )


def load_candidate_profile_manifest(path):
    """Load and validate one candidate-profile manifest at its capture root."""
    value, evidence = _load_json(
        path,
        "candidate_profile manifest",
        max_bytes=_CANDIDATE_MANIFEST_MAX_BYTES,
    )
    root = Path(evidence["path"]).parent
    _validate_manifest(value, "candidate_profile", root)
    loaded = _manifest_wrapper(value, evidence)
    _revalidate_loaded_manifest(loaded)
    return loaded


def _revalidate_loaded_manifest(manifest):
    if not isinstance(manifest, LoadedCandidateProfileManifest):
        raise TypeError("expected a validated candidate profile manifest")
    if getattr(manifest, "_seal", None) is not _PROFILE_EVIDENCE_SEAL:
        raise TypeError("expected a validated candidate profile manifest")
    _revalidate_input_evidence(
        _input_evidence_dict(manifest.evidence),
        "candidate_profile manifest",
        max_bytes=_CANDIDATE_MANIFEST_MAX_BYTES,
    )
    value, evidence = _load_json(
        manifest.evidence.path,
        "candidate_profile manifest",
        max_bytes=_CANDIDATE_MANIFEST_MAX_BYTES,
    )
    _validate_manifest(value, "candidate_profile", Path(evidence["path"]).parent)
    current = _manifest_wrapper(value, evidence)
    if current != manifest:
        raise ValueError("candidate profile manifest wrapper changed after validation")


def _require_loaded_manifest(manifest):
    _revalidate_loaded_manifest(manifest)
    return manifest


def _artifact_path(manifest, record):
    relative = PurePosixPath(record.relative_path)
    return Path(manifest.capture_root).joinpath(*relative.parts)


def _candidate_artifact_max_bytes(role):
    limits = {
        "sk_graph_origin": _ORIGIN_GRAPH_MAX_BYTES,
        "sk_graph_updated": _ORIGIN_GRAPH_MAX_BYTES,
        "kernel_details": _KERNEL_DETAILS_MAX_BYTES,
        "sk_fused_nodes": _FUSED_GROUPS_MAX_BYTES,
        "super_kernel": _SUPER_KERNEL_LOG_MAX_BYTES,
    }
    try:
        return limits[role]
    except KeyError as error:
        raise ValueError(
            f"unsupported candidate profile artifact role: {role}"
        ) from error


def _read_manifest_artifact(manifest, record, label):
    data, evidence_dict = _read_stable_input_bytes(
        _artifact_path(manifest, record),
        label,
        max_bytes=_candidate_artifact_max_bytes(record.role),
    )
    evidence = _input_evidence_value(evidence_dict)
    expected = (
        record.sha256,
        record.size,
        record.ctime_ns,
        record.mtime_ns,
    )
    observed = (
        evidence.sha256,
        evidence.size,
        evidence.ctime_ns,
        evidence.mtime_ns,
    )
    if expected != observed:
        raise ValueError(
            f"{label} no longer matches its candidate profile manifest record"
        )
    return data, EvidenceBoundSource(record, evidence)


def _wrapper_sources(value):
    if isinstance(value, ProfileGraphSet):
        return tuple(item.source for item in value.graphs)
    if isinstance(value, CandidateKernelRows):
        return value.sources
    if isinstance(value, FusedGroups):
        return tuple(item.source for item in value.sources)
    raise TypeError(f"unsupported candidate profile evidence: {type(value).__name__}")


def _revalidate_profile_input_sources(manifest, inputs):
    sources = []
    for value in inputs:
        if getattr(value, "_seal", None) is not _PROFILE_EVIDENCE_SEAL:
            raise TypeError(
                "candidate profile input was not created by a public loader"
            )
        if value.binding != manifest.binding:
            raise ValueError(
                "candidate profile inputs have different producer/session evidence"
            )
        sources.extend(_wrapper_sources(value))
    manifest_records = {
        (record.role, record.relative_path): record
        for _, records in manifest.files
        for record in records
    }
    for source in sources:
        key = (source.record.role, source.record.relative_path)
        if manifest_records.get(key) != source.record:
            raise ValueError(
                "candidate profile artifact wrapper changed after validation"
            )
        _revalidate_input_evidence(
            _input_evidence_dict(source.evidence),
            source.record.role,
            max_bytes=_candidate_artifact_max_bytes(source.record.role),
        )


def _strict_payload_equal(left, right):
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(
            _strict_payload_equal(
                getattr(left, definition.name), getattr(right, definition.name)
            )
            for definition in fields(left)
            if definition.compare
        )
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _strict_payload_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _reload_profile_input(manifest, value):
    loaders = {
        ProfileGraphSet: _load_profile_graphs_core,
        CandidateKernelRows: _load_candidate_kernel_rows_core,
        FusedGroups: _load_profile_fused_groups_core,
    }
    loader = loaders.get(type(value))
    if loader is None:
        raise TypeError(
            f"unsupported candidate profile evidence: {type(value).__name__}"
        )
    return loader(manifest)


def _finish_profile_input_load(manifest, value):
    _revalidate_profile_input_sources(manifest, (value,))
    _revalidate_input_evidence(
        _input_evidence_dict(manifest.evidence),
        "candidate_profile manifest",
        max_bytes=_CANDIDATE_MANIFEST_MAX_BYTES,
    )
    return value


def revalidate_profile_inputs(manifest, *inputs):
    """Reparse and verify that wrappers still match their manifest artifacts."""
    manifest = _require_loaded_manifest(manifest)
    _revalidate_profile_input_sources(manifest, inputs)
    reloaded = []
    for value in inputs:
        current = _reload_profile_input(manifest, value)
        if not _strict_payload_equal(current, value):
            raise ValueError(
                "candidate profile semantic payload does not match manifest sources"
            )
        reloaded.append(current)
    _revalidate_profile_input_sources(manifest, reloaded)
    _revalidate_input_evidence(
        _input_evidence_dict(manifest.evidence),
        "candidate_profile manifest",
        max_bytes=_CANDIDATE_MANIFEST_MAX_BYTES,
    )


def _load_profile_graphs_core(manifest):
    graphs = []
    identities = set()
    for record in manifest.records_for("sk_graph_origin"):
        data, source = _read_manifest_artifact(manifest, record, "sk_graph_origin")
        value = _standard_json_from_bytes(data, "sk_graph_origin")
        graph = _graph_from_sk_origin_value(value, model_role="candidate")
        device_id = int(_identity(value["deviceId"], "deviceId"))
        model_id = _identity(value["modelId"], "modelId")
        identity = (device_id, model_id)
        if identity in identities:
            raise ValueError(f"duplicate profile origin graph identity: {identity}")
        identities.add(identity)
        graphs.append(EvidenceBoundGraph(device_id, model_id, graph, source))
    graphs.sort(
        key=lambda item: (
            item.device_id,
            _natural_identity(item.model_id),
            item.source.record.relative_path,
        )
    )
    return ProfileGraphSet(manifest.binding, tuple(graphs), _PROFILE_EVIDENCE_SEAL)


def load_profile_graphs(manifest):
    """Load every profile-owned origin graph from one validated manifest."""
    manifest = _require_loaded_manifest(manifest)
    return _finish_profile_input_load(manifest, _load_profile_graphs_core(manifest))


def _nonnegative_csv_integer(value, label):
    text = _identity(value, label)
    if not text.isdecimal():
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(text)


def _candidate_bounded_nonnegative_integer(value, label, *, positive=False):
    if not value or not value.isdecimal() or len(value) > _CANDIDATE_ID_MAX_DIGITS:
        raise ValueError(f"{label} must be a bounded nonnegative integer")
    result = int(value)
    if result > _CANDIDATE_ID_MAX or (positive and result == 0):
        raise ValueError(f"{label} must be a bounded nonnegative integer")
    return result


def _candidate_sk_name_parts(name, label):
    invalid = f"{label} has invalid SuperKernel boundary name"
    if (
        not isinstance(name, str)
        or len(name) > _CANDIDATE_SK_NAME_MAX_CHARS
        or not name.startswith("sk_")
    ):
        raise ValueError(invalid)
    id_end = name.find("_", 3)
    if id_end <= 3:
        raise ValueError(invalid)
    try:
        sk_id = _candidate_bounded_nonnegative_integer(
            name[3:id_end], f"{label} SuperKernel id"
        )
    except ValueError:
        raise ValueError(invalid) from None
    body_start = id_end + 1
    start_marker = name.find("_start_", body_start)
    end_marker = name.rfind("_end_")
    if start_marker < body_start or end_marker <= start_marker + len("_start_"):
        raise ValueError(invalid)
    scope = name[body_start:start_marker].strip()
    start = name[start_marker + len("_start_") : end_marker]
    end = name[end_marker + len("_end_") :]
    if not scope or not start or not end:
        raise ValueError(invalid)
    return sk_id, scope, start, end


def _candidate_boundary(name, label):
    sk_id, scope, start, end = _candidate_sk_name_parts(name, label)
    return sk_id, CandidateBoundary(
        scope,
        start,
        end,
        canonical_op(start),
        canonical_op(end),
    )


def _candidate_rows_from_bytes(data, record):
    label = f"kernel_details {record.relative_path}"
    rows = _csv_rows_from_bytes(
        data,
        label,
        _CANDIDATE_KERNEL_HEADERS,
        max_rows=_CANDIDATE_KERNEL_MAX_ROWS,
    )
    parsed = []
    for index, row in enumerate(rows, start=2):
        row_label = f"{label} row {index}"
        name = _identity(row.get("Name"), f"{row_label}.Name")
        row_type = _identity(row.get("Type"), f"{row_label}.Type")
        if row_type != "SuperKernel" and not name.startswith("sk_"):
            continue
        start = _decimal(row.get("Start Time(us)"), f"{row_label} start")
        duration = _decimal(row.get("Duration(us)"), f"{row_label} duration")
        if start < 0 or duration <= 0:
            raise ValueError(f"{row_label} timing must be positive finite")
        sk_id, boundary = _candidate_boundary(name, row_label)
        raw_core = _identity(
            row.get("Accelerator Core"), f"{row_label}.Accelerator Core"
        )
        parsed.append(
            CandidateKernelRow(
                step_id=_nonnegative_csv_integer(
                    row.get("Step Id"), f"{row_label}.Step Id"
                ),
                device_id=_nonnegative_csv_integer(
                    row.get("Device_id"), f"{row_label}.Device_id"
                ),
                model_id=_nonnegative_csv_integer(
                    row.get("Model ID"), f"{row_label}.Model ID"
                ),
                sk_id=sk_id,
                task_id=_nonnegative_csv_integer(
                    row.get("Task ID"), f"{row_label}.Task ID"
                ),
                stream_id=_nonnegative_csv_integer(
                    row.get("Stream ID"), f"{row_label}.Stream ID"
                ),
                name=name,
                boundary=boundary,
                raw_core=raw_core,
                core_family=canonical_core_family(raw_core),
                block_num=_nonnegative_csv_integer(
                    row.get("Block Num"), f"{row_label}.Block Num"
                ),
                mix_block_num=_nonnegative_csv_integer(
                    row.get("Mix Block Num"), f"{row_label}.Mix Block Num"
                ),
                start_us=start,
                duration_us=duration,
                source_relative_path=record.relative_path,
                source_row=index,
            )
        )
    return parsed


def _load_candidate_kernel_rows_core(manifest):
    parsed = []
    sources = []
    seen = set()
    for record in manifest.records_for("kernel_details"):
        data, source = _read_manifest_artifact(manifest, record, "kernel_details")
        sources.append(source)
        for row in _candidate_rows_from_bytes(data, record):
            identity = (
                row.step_id,
                row.device_id,
                row.model_id,
                row.sk_id,
                row.start_us,
                row.duration_us,
                row.name,
            )
            if identity in seen:
                raise ValueError(
                    "duplicate candidate kernel parent across profile files"
                )
            seen.add(identity)
            parsed.append(row)
    parsed.sort(
        key=lambda row: (
            row.device_id,
            row.model_id,
            row.sk_id,
            row.step_id,
            row.start_us,
            row.source_relative_path,
            row.source_row,
        )
    )
    return CandidateKernelRows(
        manifest.binding,
        tuple(parsed),
        tuple(sources),
        _PROFILE_EVIDENCE_SEAL,
    )


def load_candidate_kernel_rows(manifest):
    """Load all candidate SuperKernel parent rows without float conversion."""
    manifest = _require_loaded_manifest(manifest)
    return _finish_profile_input_load(
        manifest, _load_candidate_kernel_rows_core(manifest)
    )


def _load_profile_fused_groups_core(manifest):
    sources = []
    for record in manifest.records_for("sk_fused_nodes"):
        data, source = _read_manifest_artifact(manifest, record, "sk_fused_nodes")
        line_count = data.count(b"\n") + bool(data and not data.endswith(b"\n"))
        if line_count > _FUSED_GROUPS_MAX_LINES:
            raise ValueError(
                f"sk_fused_nodes exceeds line limit {_FUSED_GROUPS_MAX_LINES}"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("sk_fused_nodes must be UTF-8 text") from error
        sources.append(EvidenceBoundText(source, text))
    return FusedGroups(manifest.binding, tuple(sources), _PROFILE_EVIDENCE_SEAL)


def load_profile_fused_groups(manifest):
    """Load all profile-owned fused logs as immutable UTF-8 evidence."""
    manifest = _require_loaded_manifest(manifest)
    return _finish_profile_input_load(
        manifest, _load_profile_fused_groups_core(manifest)
    )


def _candidate_fused_integer(value, label, *, positive=False):
    return _candidate_bounded_nonnegative_integer(value, label, positive=positive)


def _candidate_fused_fields(value, label):
    fields = []
    start = 0
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(value):
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                raise ValueError(f"{label} has unbalanced delimiters")
        elif character == "," and not stack:
            fields.append(value[start:index].strip())
            start = index + 1
            if len(fields) >= _FUSED_METADATA_MAX_FIELDS:
                raise ValueError(
                    f"{label} exceeds field limit {_FUSED_METADATA_MAX_FIELDS}"
                )
    if stack:
        raise ValueError(f"{label} has unbalanced delimiters")
    fields.append(value[start:].strip())
    if not fields[-1] or any(not item for item in fields):
        raise ValueError(f"{label} has an empty field")
    if len(fields) > _FUSED_METADATA_MAX_FIELDS:
        raise ValueError(f"{label} exceeds field limit {_FUSED_METADATA_MAX_FIELDS}")
    return tuple(fields)


def _candidate_fused_field_map(value, label):
    result = {}
    for item in _candidate_fused_fields(value, label):
        if ":" not in item:
            raise ValueError(f"{label} field must use name:value")
        name, field_value = (part.strip() for part in item.split(":", 1))
        if not name or not field_value:
            raise ValueError(f"{label} field must use name:value")
        if name in result:
            raise ValueError(f"{label} has duplicate field: {name}")
        result[name] = field_value
    return result


def _candidate_profile_graph_index(graphs):
    index = defaultdict(list)
    for bound_graph in graphs.graphs:
        match = _PROFILE_MODEL_ID.fullmatch(bound_graph.model_id)
        if match is not None:
            model_id = _candidate_bounded_nonnegative_integer(
                match.group("model"), "profile graph model"
            )
            index[model_id].append(bound_graph)
    return {model_id: tuple(items) for model_id, items in index.items()}


def _candidate_fused_path_context(relative_path, graph_index):
    model_ids = []
    device_ids = []
    blockers = set()
    for component in PurePosixPath(relative_path).parts:
        model = _FUSED_MODEL_COMPONENT.fullmatch(component)
        if model is not None:
            try:
                model_ids.append(
                    _candidate_fused_integer(model.group("model"), "metadata model")
                )
            except ValueError:
                blockers.add("metadata_model_path_invalid")
        device = _FUSED_DEVICE_COMPONENT.fullmatch(component)
        if device is not None:
            try:
                device_ids.append(
                    _candidate_fused_integer(device.group("device"), "metadata device")
                )
            except ValueError:
                blockers.add("metadata_device_path_invalid")
    if len(model_ids) != 1:
        blockers.add("metadata_model_path_invalid")
        model_id = None
    else:
        model_id = model_ids[0]
    matching_graphs = list(graph_index.get(model_id, ()))
    if device_ids:
        if len(device_ids) != 1:
            blockers.add("metadata_device_path_invalid")
            device_id = None
        else:
            device_id = device_ids[0]
            matching_graphs = [
                item for item in matching_graphs if item.device_id == device_id
            ]
    else:
        devices = {item.device_id for item in matching_graphs}
        if len(devices) != 1:
            blockers.add("metadata_device_ambiguous")
            device_id = None
        else:
            device_id = next(iter(devices))
            matching_graphs = [
                item for item in matching_graphs if item.device_id == device_id
            ]
    if len(matching_graphs) == 0:
        blockers.add("metadata_graph_missing")
        bound_graph = None
    elif len(matching_graphs) > 1:
        blockers.add("metadata_graph_ambiguous")
        bound_graph = None
    else:
        bound_graph = matching_graphs[0]
    return device_id, model_id, bound_graph, tuple(sorted(blockers))


def _candidate_origin_index(graph):
    index = defaultdict(list)
    blockers = set()
    for node in graph.nodes:
        counts = Counter(key for key, _ in node.provenance)
        for key, count in counts.items():
            if count <= 1:
                continue
            if key == "node_id":
                blockers.add("metadata_origin_node_id_provenance_duplicate")
            elif key == "kernel_type":
                blockers.add("metadata_origin_kernel_type_duplicate")
            else:
                blockers.add("metadata_origin_provenance_duplicate")
        node_ids = tuple(value for key, value in node.provenance if key == "node_id")
        if not node_ids:
            blockers.add("metadata_origin_node_id_provenance_missing")
        elif len(node_ids) == 1:
            index[node_ids[0]].append(node)
        if node.task_kind == "KERNEL":
            kernel_types = tuple(
                value for key, value in node.provenance if key == "kernel_type"
            )
            if not kernel_types:
                blockers.add("metadata_origin_kernel_type_missing")
    return (
        {node_id: tuple(nodes) for node_id, nodes in index.items()},
        tuple(sorted(blockers)),
    )


def _candidate_origin_node(origin_index, origin_node_id):
    matches = origin_index.get(str(origin_node_id), ())
    if not matches:
        raise ValueError(f"origin nodeId {origin_node_id} is missing")
    if len(matches) > 1:
        raise ValueError(f"origin nodeId {origin_node_id} is duplicated")
    return matches[0]


def _candidate_kernel_families(kernel_type):
    families = {
        "AIC_ONLY": ("AI_CORE",),
        "AIV_ONLY": ("AI_VECTOR_CORE",),
        "MIX_1_1": ("AI_CORE", "AI_VECTOR_CORE"),
        "MIX_1_2": ("AI_CORE", "AI_VECTOR_CORE"),
    }
    try:
        return families[kernel_type]
    except KeyError as error:
        raise ValueError(f"unsupported fused kernelType: {kernel_type}") from error


def _candidate_origin_kernel_type(origin_node):
    kernel_types = tuple(
        value for key, value in origin_node.provenance if key == "kernel_type"
    )
    if not kernel_types:
        raise ValueError("origin CanonicalNode kernel_type provenance is missing")
    if len(kernel_types) > 1:
        raise ValueError("origin CanonicalNode kernel_type provenance is duplicated")
    kernel_type = kernel_types[0]
    if kernel_type not in _FUSED_KERNEL_TYPES:
        raise ValueError(f"unsupported origin CanonicalNode kernel_type: {kernel_type}")
    return kernel_type


def _candidate_fused_local_origin(line, origin_index, label):
    match = _FUSED_LOCAL_ORIGIN.match(line)
    if match is None:
        raise ValueError(f"{label} must contain local [i] and origin nodeId")
    local_index = _candidate_fused_integer(match.group("local"), f"{label} local")
    origin_node_id = _candidate_fused_integer(
        match.group("origin"), f"{label} origin nodeId"
    )
    return (
        match,
        local_index,
        origin_node_id,
        _candidate_origin_node(origin_index, origin_node_id),
    )


def _parse_candidate_fused_kernel_line(line, origin_index, label):
    match, local_index, origin_node_id, origin_node = _candidate_fused_local_origin(
        line, origin_index, label
    )
    marker = "] - KernelInfos{"
    marker_index = line.find(marker, match.end())
    if marker_index < 0 or not line.endswith("}"):
        raise ValueError(f"{label} has a truncated KernelInfos record")
    origin_start = line.find("[nodeId:", match.start()) + 1
    origin_fields = _candidate_fused_field_map(
        line[origin_start:marker_index], f"{label} origin fields"
    )
    if origin_fields.get("nodeId") != str(origin_node_id):
        raise ValueError(f"{label} origin nodeId conflicts")
    kernel_fields = _candidate_fused_field_map(
        line[marker_index + len(marker) : -1], f"{label} KernelInfos"
    )
    required = {"funcName", "kernelType", "numBlocks", "cubeNum", "vecNum"}
    missing = sorted(required - set(kernel_fields))
    if missing:
        raise ValueError(f"{label} KernelInfos missing fields: {', '.join(missing)}")
    canonical_name = canonical_op(kernel_fields["funcName"])
    if canonical_name != origin_node.canonical_op:
        raise ValueError(f"{label} funcName conflicts with origin CanonicalNode")
    metadata_kernel_type = kernel_fields["kernelType"]
    families = _candidate_kernel_families(metadata_kernel_type)
    origin_kernel_type = _candidate_origin_kernel_type(origin_node)
    if metadata_kernel_type != origin_kernel_type:
        raise ValueError(
            f"{label} kernelType conflicts with origin kernel_type provenance"
        )
    if (
        origin_node.core_family != "COMMUNICATION"
        and origin_node.core_family not in families
    ):
        raise ValueError(f"{label} kernelType conflicts with origin core family")
    num_blocks = _candidate_fused_integer(
        kernel_fields["numBlocks"], f"{label} numBlocks"
    )
    cube_num = _candidate_fused_integer(kernel_fields["cubeNum"], f"{label} cubeNum")
    vec_num = _candidate_fused_integer(kernel_fields["vecNum"], f"{label} vecNum")
    expected_geometry = {
        "AIC_ONLY": (num_blocks, 0),
        "AIV_ONLY": (0, num_blocks),
        "MIX_1_1": (num_blocks, num_blocks),
        "MIX_1_2": (num_blocks, num_blocks * 2),
    }[kernel_fields["kernelType"]]
    if num_blocks == 0 or (cube_num, vec_num) != expected_geometry:
        raise ValueError(f"{label} kernel geometry is inconsistent")
    return _CandidateFusedMetadataNode(
        local_index,
        origin_node_id,
        origin_node.key,
        canonical_name,
        families,
        num_blocks,
        cube_num,
        vec_num,
        False,
        "KERNEL",
    )


def _parse_candidate_fused_control_line(line, origin_index, label):
    match, local_index, origin_node_id, origin_node = _candidate_fused_local_origin(
        line, origin_index, label
    )
    if not line.endswith("]"):
        raise ValueError(f"{label} has a truncated control record")
    origin_start = line.find("[nodeId:", match.start()) + 1
    items = _candidate_fused_fields(line[origin_start:-1], f"{label} control")
    control_names = {
        "EventWait": "WAIT",
        "EventNotify": "NOTIFY",
        "EventReset": "RESET",
    }
    control_item = items[-1]
    control_name = next(
        (
            name
            for name in control_names
            if control_item.startswith(name + "(") and control_item.endswith(")")
        ),
        None,
    )
    if control_name is None:
        raise ValueError(f"{label} has unsupported fused control entry")
    origin_fields = _candidate_fused_field_map(
        ",".join(items[:-1]), f"{label} origin fields"
    )
    if origin_fields.get("nodeId") != str(origin_node_id):
        raise ValueError(f"{label} origin nodeId conflicts")
    task_kind = control_names[control_name]
    if not origin_node.topology_only or origin_node.task_kind != task_kind:
        raise ValueError(f"{label} control conflicts with origin CanonicalNode")
    return _CandidateFusedMetadataNode(
        local_index,
        origin_node_id,
        origin_node.key,
        origin_node.canonical_op,
        (),
        None,
        None,
        None,
        True,
        task_kind,
    )


def _parse_candidate_fused_node_line(line, origin_index, label):
    if "] - KernelInfos{" in line:
        return _parse_candidate_fused_kernel_line(line, origin_index, label)
    return _parse_candidate_fused_control_line(line, origin_index, label)


def _candidate_fused_node_blocker(error):
    message = str(error)
    if "origin nodeId" in message and "is missing" in message:
        return "metadata_origin_node_missing"
    if "origin nodeId" in message and "is duplicated" in message:
        return "metadata_origin_node_duplicate"
    if "funcName conflicts" in message:
        return "metadata_op_mismatch"
    if "unsupported fused kernelType" in message:
        return "metadata_kernel_type_unsupported"
    if "kernel_type provenance is missing" in message:
        return "metadata_origin_kernel_type_missing"
    if "kernel_type provenance is duplicated" in message:
        return "metadata_origin_kernel_type_duplicate"
    if "unsupported origin CanonicalNode kernel_type" in message:
        return "metadata_origin_kernel_type_unsupported"
    if "origin kernel_type provenance" in message:
        return "metadata_origin_kernel_type_mismatch"
    if "kernelType conflicts" in message:
        return "metadata_origin_core_mismatch"
    if "KernelInfos missing fields" in message:
        return "metadata_kernel_fields_missing"
    if "kernel geometry" in message:
        return "metadata_kernel_geometry_invalid"
    if "control conflicts" in message:
        return "metadata_control_mismatch"
    if "bounded nonnegative integer" in message:
        return "metadata_integer_invalid"
    return "metadata_node_line_invalid"


def _candidate_fused_draft(path_blockers):
    return {
        "sk_id": None,
        "name": "",
        "boundary": None,
        "node_count": None,
        "line_count": 0,
        "node_line_count": 0,
        "local_indices": [],
        "origin_node_ids": [],
        "nodes": [],
        "blockers": set(path_blockers),
    }


def _candidate_fused_payload(line):
    if line.startswith("SK Function:") or _FUSED_LOCAL_ORIGIN.match(line):
        return line
    logger_match = _FUSED_CANN_LOG_LINE.fullmatch(line)
    if logger_match is None:
        raise ValueError("sk_fused_nodes line has an unsupported adapter prefix")
    payload = logger_match.group("payload")
    if payload.startswith("SK Function:") or _FUSED_LOCAL_ORIGIN.match(payload):
        return payload
    raise ValueError("sk_fused_nodes logger payload is not a fused metadata line")


def _parse_candidate_fused_source(
    source, graph_index, origin_indexes, graph_fingerprints, group_limit
):
    device_id, model_id, bound_graph, path_blockers = _candidate_fused_path_context(
        source.record.relative_path, graph_index
    )
    origin_index = None
    current_graph_fingerprint = None
    if bound_graph is not None:
        graph_key = id(bound_graph.graph)
        if graph_key not in origin_indexes:
            origin_indexes[graph_key] = _candidate_origin_index(bound_graph.graph)
        if graph_key not in graph_fingerprints:
            graph_fingerprints[graph_key] = graph_fingerprint(bound_graph.graph)
        origin_index, origin_blockers = origin_indexes[graph_key]
        path_blockers = tuple(sorted(set(path_blockers) | set(origin_blockers)))
        current_graph_fingerprint = graph_fingerprints[graph_key]
    lines = source.text.splitlines()
    groups = []
    current = None
    source_line_failure = False
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if len(line) > _FUSED_METADATA_MAX_LINE_CHARS:
            if current is None:
                current = _candidate_fused_draft(path_blockers)
            current["line_count"] += 1
            current["blockers"].add("metadata_line_limit_exceeded")
            source_line_failure = True
            continue
        try:
            line = _candidate_fused_payload(line)
        except ValueError:
            if current is None:
                current = _candidate_fused_draft(path_blockers)
            current["line_count"] += 1
            current["blockers"].add("metadata_line_adapter_invalid")
            source_line_failure = True
            continue
        if line.startswith("SK Function:"):
            if current is not None:
                groups.append(current)
                if len(groups) >= group_limit:
                    groups[-1]["blockers"].add("metadata_group_limit_exceeded")
                    source_line_failure = True
                    current = None
                    break
            current = _candidate_fused_draft(path_blockers)
            header = _FUSED_HEADER.fullmatch(line)
            if header is None:
                current["blockers"].add("metadata_header_invalid")
                continue
            current["name"] = header.group("name")
            try:
                sk_id, boundary = _candidate_boundary(
                    header.group("name"), f"sk_fused_nodes line {line_number}"
                )
                current["sk_id"] = sk_id
                current["boundary"] = boundary
            except ValueError:
                current["blockers"].add("metadata_header_name_invalid")
            try:
                scope_id = _candidate_fused_integer(
                    header.group("scope"), f"sk_fused_nodes line {line_number} scope"
                )
                node_count = _candidate_fused_integer(
                    header.group("count"), f"sk_fused_nodes line {line_number} count"
                )
                current["node_count"] = node_count
            except ValueError:
                current["blockers"].add("metadata_header_integer_invalid")
                continue
            if node_count > _CANDIDATE_METADATA_MAX_NODES:
                current["blockers"].add("metadata_node_count_limit_exceeded")
            if current["sk_id"] is not None and current["sk_id"] != scope_id:
                current["blockers"].add("metadata_scope_id_mismatch")
            continue
        if current is None:
            current = _candidate_fused_draft(path_blockers)
            current["blockers"].add("metadata_header_missing")
        if current["node_line_count"] >= _CANDIDATE_METADATA_MAX_NODES:
            current["blockers"].add("metadata_node_count_limit_exceeded")
            source_line_failure = True
            break
        current["node_line_count"] += 1
        current["line_count"] += 1
        local_match = _FUSED_LOCAL_ORIGIN.match(line)
        if local_match is not None:
            try:
                current["local_indices"].append(
                    _candidate_fused_integer(
                        local_match.group("local"),
                        f"sk_fused_nodes line {line_number} local",
                    )
                )
                current["origin_node_ids"].append(
                    _candidate_fused_integer(
                        local_match.group("origin"),
                        f"sk_fused_nodes line {line_number} origin nodeId",
                    )
                )
            except ValueError:
                current["blockers"].add("metadata_integer_invalid")
        else:
            current["blockers"].add("metadata_node_line_invalid")
        if origin_index is None:
            continue
        try:
            current["nodes"].append(
                _parse_candidate_fused_node_line(
                    line, origin_index, f"sk_fused_nodes line {line_number}"
                )
            )
        except ValueError as error:
            current["blockers"].add(_candidate_fused_node_blocker(error))
    if current is not None:
        groups.append(current)
    if not groups:
        current = _candidate_fused_draft(path_blockers)
        current["blockers"].add("metadata_header_missing")
        groups.append(current)
    if len(groups) > group_limit:
        groups = groups[:group_limit]
        groups[-1]["blockers"].add("metadata_group_limit_exceeded")
    result = []
    source_has_unattributed_failure = (
        source_line_failure
        or device_id is None
        or model_id is None
        or any(current["sk_id"] is None for current in groups)
    )
    for current in groups:
        nodes = tuple(sorted(current["nodes"], key=lambda node: node.local_index))
        blockers = set(current["blockers"])
        node_count = current["node_count"]
        local_indices = current["local_indices"]
        if node_count is None or current["line_count"] != node_count:
            blockers.add("metadata_node_count_mismatch")
        if len(local_indices) != len(set(local_indices)):
            blockers.add("metadata_local_duplicate")
        origin_node_ids = current["origin_node_ids"]
        if len(origin_node_ids) != len(set(origin_node_ids)):
            blockers.add("metadata_origin_node_duplicate")
        unique_local = set(local_indices)
        if (
            node_count is None
            or len(unique_local) != node_count
            or (
                unique_local
                and (min(unique_local) != 0 or max(unique_local) != node_count - 1)
            )
        ):
            blockers.add("metadata_local_sequence_invalid")
        kernel_nodes = tuple(node for node in nodes if not node.topology_only)
        if not kernel_nodes:
            blockers.add("metadata_kernel_nodes_missing")
        elif current["boundary"] is not None and (
            current["boundary"].start_op != kernel_nodes[0].canonical_op
            or current["boundary"].end_op != kernel_nodes[-1].canonical_op
        ):
            blockers.add("metadata_boundary_mismatch")
        result.append(
            _CandidateFusedMetadataGroup(
                device_id,
                model_id,
                current["sk_id"],
                current["name"],
                current["boundary"],
                nodes,
                kernel_nodes,
                current_graph_fingerprint,
                source.evidence.sha256,
                tuple(sorted(blockers)),
            )
        )
    if source_has_unattributed_failure:
        result = [
            replace(
                group,
                blockers=tuple(
                    sorted(
                        set(group.blockers) | {"metadata_source_unattributed_failure"}
                    )
                ),
            )
            for group in result
        ]
    return tuple(result)


def _parse_candidate_fused_metadata_core(graphs, fused):
    groups = []
    graph_index = _candidate_profile_graph_index(graphs)
    origin_indexes = {}
    graph_fingerprints = {}
    sources = sorted(fused.sources, key=lambda item: item.record.relative_path)
    for source_index, source in enumerate(sources):
        remaining_groups = _FUSED_METADATA_MAX_GROUPS - len(groups)
        if remaining_groups <= 0:
            groups = [
                replace(
                    group,
                    blockers=tuple(
                        sorted(
                            set(group.blockers)
                            | {
                                "metadata_group_limit_exceeded",
                                "metadata_source_unattributed_failure",
                            }
                        )
                    ),
                )
                for group in groups
            ]
            break
        groups.extend(
            _parse_candidate_fused_source(
                source,
                graph_index,
                origin_indexes,
                graph_fingerprints,
                remaining_groups,
            )
        )
        if len(groups) >= _FUSED_METADATA_MAX_GROUPS and source_index + 1 < len(
            sources
        ):
            groups = [
                replace(
                    group,
                    blockers=tuple(
                        sorted(
                            set(group.blockers)
                            | {
                                "metadata_group_limit_exceeded",
                                "metadata_source_unattributed_failure",
                            }
                        )
                    ),
                )
                for group in groups
            ]
            break
    if any(
        "metadata_source_unattributed_failure" in group.blockers for group in groups
    ):
        groups = [
            replace(
                group,
                blockers=tuple(
                    sorted(
                        set(group.blockers) | {"metadata_source_unattributed_failure"}
                    )
                ),
            )
            for group in groups
        ]
    identities = Counter(
        (group.device_id, group.model_id, group.sk_id)
        for group in groups
        if None not in (group.device_id, group.model_id, group.sk_id)
    )
    groups = [
        replace(
            group,
            blockers=tuple(sorted(set(group.blockers) | {"metadata_group_duplicate"})),
        )
        if identities[(group.device_id, group.model_id, group.sk_id)] > 1
        else group
        for group in groups
    ]
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                group.device_id is None,
                -1 if group.device_id is None else group.device_id,
                group.model_id is None,
                -1 if group.model_id is None else group.model_id,
                group.sk_id is None,
                -1 if group.sk_id is None else group.sk_id,
                group.name,
                group.fused_fingerprint,
            ),
        )
    )


def _node_label(node):
    return [
        node.task_kind,
        node.canonical_op,
        node.core_family,
        list(node.shape_dtype),
        node.topology_only,
    ]


def _partition_colors(signatures):
    serialized = [canonical_json(signature) for signature in signatures]
    palette = {
        signature: color for color, signature in enumerate(sorted(set(serialized)))
    }
    return tuple(palette[signature] for signature in serialized), serialized


def _refined_node_colors(initial_signatures, incoming, outgoing):
    colors, _ = _partition_colors(initial_signatures)
    class_count = len(set(colors))
    while True:
        signatures = []
        for index, color in enumerate(colors):
            incoming_colors = sorted(
                (
                    [kind, same_stream, colors[source]]
                    for kind, same_stream, source in incoming[index]
                ),
                key=canonical_json,
            )
            outgoing_colors = sorted(
                (
                    [kind, same_stream, colors[target]]
                    for kind, same_stream, target in outgoing[index]
                ),
                key=canonical_json,
            )
            signatures.append([color, incoming_colors, outgoing_colors])
        refined, serialized = _partition_colors(signatures)
        refined_class_count = len(set(refined))
        colors = refined
        if refined_class_count == class_count:
            return tuple(
                hashlib.sha256(
                    canonical_json(
                        [initial_signatures[index], signatures[index]]
                    ).encode("utf-8")
                ).hexdigest()
                for index in range(len(serialized))
            )
        class_count = refined_class_count


def _initial_stream_partition(stream_labels):
    grouped = defaultdict(list)
    for stream, labels in enumerate(stream_labels):
        grouped[canonical_json(list(labels))].append(stream)
    return tuple(tuple(grouped[label]) for label in sorted(grouped))


def _refine_stream_partition(partition, incoming, outgoing):
    while True:
        cell_by_stream = {}
        for cell_index, cell in enumerate(partition):
            for stream in cell:
                cell_by_stream[stream] = cell_index

        refined = []
        for cell in partition:
            groups = defaultdict(list)
            for stream in cell:
                incoming_by_cell = [[] for _ in partition]
                outgoing_by_cell = [[] for _ in partition]
                for other, source_position, target_position, kind in incoming[stream]:
                    incoming_by_cell[cell_by_stream[other]].append(
                        (source_position, target_position, kind)
                    )
                for other, source_position, target_position, kind in outgoing[stream]:
                    outgoing_by_cell[cell_by_stream[other]].append(
                        (source_position, target_position, kind)
                    )
                signature = (
                    tuple(tuple(sorted(edges)) for edges in incoming_by_cell),
                    tuple(tuple(sorted(edges)) for edges in outgoing_by_cell),
                )
                groups[signature].append(stream)
            refined.extend(tuple(groups[signature]) for signature in sorted(groups))
        refined = tuple(refined)
        if refined == partition:
            return refined
        partition = refined


def _stream_rank(value):
    return f"s{value:06d}"


def _stream_serialization(partition, stream_labels, edge_records, *, lower_bound=False):
    rank_by_stream = {}
    ordered_labels = []
    rank = 0
    for cell in partition:
        cell_rank = rank
        for stream in cell:
            rank_by_stream[stream] = cell_rank if lower_bound else rank
            ordered_labels.append(list(stream_labels[stream]))
            rank += 1
    edges = sorted(
        (
            [
                _stream_rank(rank_by_stream[source]),
                source_position,
                _stream_rank(rank_by_stream[target]),
                target_position,
                kind,
            ]
            for source, source_position, target, target_position, kind in edge_records
        ),
        key=canonical_json,
    )
    return canonical_json({"streams": ordered_labels, "edges": edges})


def _transposition_is_automorphism(first, second, stream_labels, edge_counter):
    if stream_labels[first] != stream_labels[second]:
        return False

    def swap(stream):
        if stream == first:
            return second
        if stream == second:
            return first
        return stream

    transformed = Counter(
        (swap(source), source_position, swap(target), target_position, kind)
        for source, source_position, target, target_position, kind in edge_counter.elements()
    )
    return transformed == edge_counter


def _canonical_stream_serialization(stream_labels, edge_records):
    stream_count = len(stream_labels)
    incoming = [[] for _ in range(stream_count)]
    outgoing = [[] for _ in range(stream_count)]
    for source, source_position, target, target_position, kind in edge_records:
        outgoing[source].append((target, source_position, target_position, kind))
        incoming[target].append((source, source_position, target_position, kind))

    initial = _refine_stream_partition(
        _initial_stream_partition(stream_labels), incoming, outgoing
    )
    labeling_bound = math.prod(math.factorial(len(cell)) for cell in initial)
    if labeling_bound > _MAX_STREAM_CANONICAL_LABELINGS:
        raise ValueError(
            "stream topology exceeds deterministic canonical-labeling safety bound"
        )

    edge_counter = Counter(edge_records)
    seen_states = set()
    seen_serializations = set()
    best = None

    def search(partition):
        nonlocal best
        partition = _refine_stream_partition(partition, incoming, outgoing)
        if partition in seen_states:
            return
        seen_states.add(partition)

        ambiguous = next((cell for cell in partition if len(cell) > 1), None)
        if ambiguous is None:
            serialization = _stream_serialization(
                partition, stream_labels, edge_records
            )
            if serialization not in seen_serializations:
                seen_serializations.add(serialization)
                if best is None or serialization < best:
                    best = serialization
            return

        lower_bound = _stream_serialization(
            partition, stream_labels, edge_records, lower_bound=True
        )
        if best is not None and lower_bound >= best:
            return

        cell_index = partition.index(ambiguous)
        representatives = []
        children = []
        for stream in ambiguous:
            if any(
                _transposition_is_automorphism(
                    stream, representative, stream_labels, edge_counter
                )
                for representative in representatives
            ):
                continue
            representatives.append(stream)
            remainder = tuple(item for item in ambiguous if item != stream)
            individualized = (
                partition[:cell_index]
                + ((stream,), remainder)
                + partition[cell_index + 1 :]
            )
            child = _refine_stream_partition(individualized, incoming, outgoing)
            children.append(
                (
                    _stream_serialization(
                        child, stream_labels, edge_records, lower_bound=True
                    ),
                    child,
                )
            )
        for _, child in sorted(children, key=lambda item: item[0]):
            search(child)

    search(initial)
    if best is None:
        raise ValueError("stream topology could not be canonically labeled")
    return best


def graph_fingerprint(graph):
    """Hash a key-independent refinement of directed structural topology."""
    if not isinstance(graph, CanonicalGraph):
        raise ValueError("graph must be a CanonicalGraph")
    role = _model_role(graph.model_role)
    node_by_key = {}
    for node in graph.nodes:
        if node.key in node_by_key:
            raise ValueError(f"duplicate canonical node key: {node.key}")
        if node.model_role != role:
            raise ValueError(f"node model_role conflicts with graph: {node.key}")
        node_by_key[node.key] = node
    membership = {}
    initial_signatures = {}
    stream_keys_seen = set()
    for stream_index, (stream_key, keys) in enumerate(graph.streams):
        if stream_key in stream_keys_seen:
            raise ValueError(f"duplicate graph stream key: {stream_key}")
        stream_keys_seen.add(stream_key)
        labels = []
        for position, key in enumerate(keys):
            if key not in node_by_key:
                raise ValueError(f"stream references unknown node: {key}")
            if key in membership:
                raise ValueError(f"node appears in multiple stream positions: {key}")
            node = node_by_key[key]
            if node.stream_key != stream_key:
                raise ValueError(f"node stream_key conflicts with owning stream: {key}")
            if node.stream_ordinal != position:
                raise ValueError(f"node stream_ordinal conflicts with position: {key}")
            labels.append(_node_label(node))
            membership[key] = (stream_index, position)
        stream_label = hashlib.sha256(
            canonical_json(labels).encode("utf-8")
        ).hexdigest()
        for position, key in enumerate(keys):
            initial_signatures[key] = [
                _node_label(node_by_key[key]),
                stream_label,
                position,
            ]
    if set(membership) != set(node_by_key):
        raise ValueError("graph streams do not contain every node exactly once")

    keys = tuple(node_by_key)
    index_by_key = {key: index for index, key in enumerate(keys)}
    incoming = [[] for _ in keys]
    outgoing = [[] for _ in keys]
    edge_records = []
    for edge in graph.edges:
        if edge.source not in membership or edge.target not in membership:
            raise ValueError("edge references unknown node")
        source = index_by_key[edge.source]
        target = index_by_key[edge.target]
        same_stream = membership[edge.source][0] == membership[edge.target][0]
        outgoing[source].append((edge.kind, same_stream, target))
        incoming[target].append((edge.kind, same_stream, source))
        edge_records.append((source, target, edge.kind, same_stream))
    colors = _refined_node_colors(
        [initial_signatures[key] for key in keys], incoming, outgoing
    )

    stream_labels = tuple(
        tuple(colors[index_by_key[key]] for key in stream_keys)
        for _, stream_keys in graph.streams
    )
    structural_edges = tuple(
        (
            membership[keys[source]][0],
            membership[keys[source]][1],
            membership[keys[target]][0],
            membership[keys[target]][1],
            kind,
        )
        for source, target, kind, _ in edge_records
    )
    canonical_graph = _canonical_stream_serialization(stream_labels, structural_edges)
    payload = {"model_role": role, "graph": canonical_graph}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
