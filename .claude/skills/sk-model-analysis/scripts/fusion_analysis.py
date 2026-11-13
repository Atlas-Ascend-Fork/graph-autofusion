"""Fusion-boundary evidence extraction for model-analysis reports.

The module deliberately keeps implementation-type inference separate from kernel
name normalization.  A name can be normalized for aggregation without providing
enough evidence to identify the backend or kernel mode.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from pathlib import Path
from typing import Any


_SCOPE_BATCH_BEGIN = re.compile(
    r"Scope split results begin:\s*pass=([^,]+),\s*totalScopes=(\d+)"
)
_SCOPE_BATCH_END = re.compile(r"Scope split results end:\s*pass=([^=]+?)\s*=*")
_SCOPE_SUMMARY = re.compile(
    r"\bScope\s+(\d+)(?:\s+\(scopeId=(\d+)\))?:\s+(\d+)\s+nodes,\s+(\d+)\s+streams"
)
_SCOPE_NODE = re.compile(r"nodeId:(\d+)")
_SCOPE_BREAK = re.compile(
    r"BreakInfo:\s*breakReason=(?P<reason>[^,]+),\s*"
    r"triggerNode=(?P<trigger>\d+),\s*triggerStream=(?P<stream>-?\d+)"
    r"(?:,\s*parentScope=(?P<parent>\d+))?"
    r"(?:,\s*syncAllNodes=\[(?P<sync>[^\]]*)\])?,\s*"
    r"detail=(?:\"(?P<quoted>[^\"]*)\"|(?P<bare>.*?))\s*$"
)

_FAILURE_NODE = re.compile(
    r"Node\s+\[nodeId:(?P<node_id>\d+),\s*streamId:(?P<stream_id>-?\d+),\s*"
    r"streamIdxInGraph:(?P<stream_idx>\d+),\s*nodeIdxInStream:(?P<node_idx>\d+)"
    r"[^\]]*\]\s*-\s*KernelInfos\{funcName:(?P<func>[^,}]+).*?\}:\s*"
    r"reason:\s*(?P<reason>[^\[]+?)\s*\[(?P<code>[^\]]+)\],\s*"
    r"reasonDetail:\s*(?P<detail>.*)$"
)
_FUNC_NAME = re.compile(r"funcName:([^,}]+)")
_KERNEL_TYPE = re.compile(r"kernelType:([^,}]+)")
_NODE_DETAIL = re.compile(
    r"nodeId:(?P<node_id>\d+),\s*streamId:(?P<stream_id>-?\d+),\s*"
    r"streamIdxInGraph:(?P<stream_idx>\d+),\s*nodeIdxInStream:(?P<node_idx>\d+)"
)
_FUSED_SCOPE_ID = re.compile(r"\bscopeId=(\d+)")
_FUSED_NODE_IDS = re.compile(r"fusedNodeIds=\[([^\]]*)\]")
_FUSED_STATUS = re.compile(r"scopeStatus=([A-Z_]+)")
_FUSED_BREAK = re.compile(
    r"breakReason=\[breakReason=(?P<reason>[^,\]]+),\s*"
    r"triggerNode=(?P<trigger>\d+),\s*triggerStream=(?P<stream>-?\d+)"
    r"(?:,\s*parentScope=(?P<parent>\d+))?"
    r"(?:,\s*syncAllNodes=\[(?P<sync>[^\]]*)\])?,\s*"
    r"detail=\"(?P<detail>[^\"]*)\"\],\s*"
    r"breakReasonDetail=(?P<reason_detail>.*?),\s*scopeName=\[(?P<scope_name>.*)\]"
)

_STATIC_HASH = re.compile(
    r"^static_kernel_(?P<op>.+?)_(?P<hash>[0-9a-fA-F]{16,})"
    r"(?P<suffix>(?:_.*)?)$",
    re.IGNORECASE,
)
_DYNAMIC_HASH = re.compile(
    r"^dynamic_kernel_(?P<op>.+?)_(?P<hash>[0-9a-fA-F]{16,})"
    r"(?P<suffix>(?:_.*)?)$",
    re.IGNORECASE,
)
_GENERIC_HASH = re.compile(
    r"^(?P<op>.+?)_(?P<hash>[0-9a-fA-F]{16,})(?P<suffix>(?:_.*)?)$"
)
_TRITON = re.compile(r"^triton(?:[_:])(?P<op>.+)$", re.IGNORECASE)
_STRUCTURAL_PREFIXES = (
    "sk_scope_kernel_begin",
    "sk_scope_kernel_end",
    "sk_placeholder_kernel",
)
_CONTROL_TYPES = {
    "EVENT_NOTIFY": "NOTIFY",
    "EVENT_WAIT": "WAIT",
    "EVENT_RESET": "RESET",
    "MEMORY_WRITE": "VALUE_WRITE",
    "MEMORY_WAIT": "VALUE_WAIT",
    "NOTIFY": "NOTIFY",
    "WAIT": "WAIT",
    "RESET": "RESET",
}


def _iter_lines(path: Path, line_range: tuple[int, int] | None = None):
    if not path.is_file():
        return
    begin = max(1, int(line_range[0])) if line_range else 1
    end = int(line_range[1]) if line_range else None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_no, line in enumerate(stream, start=1):
                if line_no < begin:
                    continue
                if end is not None and line_no > end:
                    break
                yield line_no, line.rstrip("\n")
    except OSError:
        return


def _source_status(
    path: Path,
    *,
    status: str,
    line_count: int = 0,
    record_count: int = 0,
    error_count: int = 0,
    format_error_count: int = 0,
    unknown_line_count: int = 0,
    samples: list[str] | None = None,
    line_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "status": status,
        "line_range": (
            {"begin_line": line_range[0], "end_line": line_range[1]}
            if line_range
            else None
        ),
        "line_count": line_count,
        "record_count": record_count,
        "error_count": error_count,
        "format_error_count": format_error_count,
        "unknown_line_count": unknown_line_count,
        "sample_lines": list(samples or [])[:10],
    }


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item) for item in re.findall(r"\d+", value)]


def _scope_id(scope: dict[str, Any]) -> int | None:
    value = scope.get("scope_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_scope_split(
    path: Path, line_range: tuple[int, int] | None = None
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "batches": [],
            "final_batch": None,
            "breaks": [],
            "scopes": {},
            "source": _source_status(path, status="unavailable", line_range=line_range),
        }

    batches: list[dict[str, Any]] = []
    current_batch: dict[str, Any] | None = None
    current_scope: dict[str, Any] | None = None
    line_count = 0
    error_count = 0
    unknown_count = 0
    samples: list[str] = []

    def finish_batch(*, complete: bool, end_line: int | None = None) -> None:
        nonlocal current_batch, current_scope
        if current_batch is None:
            return
        current_batch["complete"] = complete
        if end_line is not None:
            current_batch["end_line"] = end_line
        batches.append(current_batch)
        current_batch = None
        current_scope = None

    for line_no, line in _iter_lines(path, line_range) or ():
        line_count += 1
        begin_match = _SCOPE_BATCH_BEGIN.search(line)
        if begin_match:
            finish_batch(complete=False, end_line=line_no - 1)
            current_batch = {
                "pass": begin_match.group(1).strip(),
                "declared_scope_count": int(begin_match.group(2)),
                "begin_line": line_no,
                "end_line": None,
                "complete": False,
                "scopes": [],
                "breaks": [],
            }
            current_scope = None
            continue

        end_match = _SCOPE_BATCH_END.search(line)
        if end_match and current_batch is not None:
            finish_batch(complete=True, end_line=line_no)
            continue

        if current_batch is None:
            if "BreakInfo:" in line or (
                "Scope split results" in line
                and not (
                    _SCOPE_BATCH_BEGIN.search(line) or _SCOPE_BATCH_END.search(line)
                )
            ):
                unknown_count += 1
                if len(samples) < 10:
                    samples.append(line.strip())
            continue

        summary_match = _SCOPE_SUMMARY.search(line)
        if summary_match:
            display_id = int(summary_match.group(1))
            canonical_id = int(summary_match.group(2) or display_id)
            current_scope = {
                "display_scope_id": display_id,
                "scope_id": canonical_id,
                "node_count": int(summary_match.group(3)),
                "stream_count": int(summary_match.group(4)),
                "node_ids": [],
                "line": line_no,
            }
            current_batch["scopes"].append(current_scope)
            continue

        break_match = _SCOPE_BREAK.search(line)
        if break_match:
            record = {
                "scope_id": (current_scope.get("scope_id") if current_scope else None),
                "display_scope_id": (
                    current_scope.get("display_scope_id") if current_scope else None
                ),
                "break_reason": break_match.group("reason").strip(),
                "trigger_node_id": int(break_match.group("trigger")),
                "trigger_stream_idx": int(break_match.group("stream")),
                "parent_scope_id": (
                    int(break_match.group("parent"))
                    if break_match.group("parent")
                    else None
                ),
                "sync_all_node_ids": _parse_int_list(break_match.group("sync")),
                "detail": (
                    break_match.group("quoted")
                    if break_match.group("quoted") is not None
                    else (break_match.group("bare") or "").strip()
                ),
                "line": line_no,
                "pass": current_batch["pass"],
            }
            current_batch["breaks"].append(record)
            continue

        if current_scope is not None and "nodeId:" in line:
            node_match = _SCOPE_NODE.search(line)
            if node_match:
                current_scope["node_ids"].append(int(node_match.group(1)))
                continue

        if "BreakInfo:" in line or (
            "Scope split results" in line
            and not (_SCOPE_BATCH_BEGIN.search(line) or _SCOPE_BATCH_END.search(line))
        ):
            unknown_count += 1
            if len(samples) < 10:
                samples.append(line.strip())

    finish_batch(complete=False)
    completed = [batch for batch in batches if batch.get("complete")]
    final_batch = completed[-1] if completed else (batches[-1] if batches else None)
    final_scopes = {
        int(scope["scope_id"]): scope
        for scope in (final_batch or {}).get("scopes", [])
        if scope.get("scope_id") is not None
    }
    final_breaks = list((final_batch or {}).get("breaks", []))
    declared = (final_batch or {}).get("declared_scope_count", 0)
    recognized = len(final_scopes)
    if not batches or final_batch is None:
        status = "partial"
        error_count += 1
    elif declared and recognized == 0:
        status = "partial"
        error_count += 1
    elif not final_batch.get("complete") or unknown_count or error_count:
        status = "partial"
    else:
        status = "ready"
    return {
        "batches": batches,
        "final_batch": final_batch,
        "breaks": final_breaks,
        "scopes": final_scopes,
        "source": _source_status(
            path,
            status=status,
            line_count=line_count,
            record_count=sum(len(batch.get("breaks", [])) for batch in batches),
            error_count=error_count,
            unknown_line_count=unknown_count,
            samples=samples,
            line_range=line_range,
        ),
    }


def _parse_fusion_failures(
    path: Path, line_range: tuple[int, int] | None = None
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "failures": [],
            "fused_scopes": {},
            "source": _source_status(path, status="unavailable", line_range=line_range),
        }

    failures: list[dict[str, Any]] = []
    fused_scopes: dict[int, dict[str, Any]] = {}
    line_count = 0
    error_count = 0
    format_error_count = 0
    unknown_count = 0
    samples: list[str] = []
    in_fused_scopes = False
    current_fused: dict[str, Any] | None = None

    for line_no, line in _iter_lines(path, line_range) or ():
        line_count += 1
        if "[PrintOriginalScopes]" in line:
            in_fused_scopes = False
            current_fused = None
        if "[PrintFusedScopes]" in line and "scopes after fusion" in line:
            in_fused_scopes = True
            current_fused = None

        failure_match = _FAILURE_NODE.search(line)
        if failure_match:
            failures.append(
                {
                    "node_id": int(failure_match.group("node_id")),
                    "stream_id": int(failure_match.group("stream_id")),
                    "stream_idx_in_graph": int(failure_match.group("stream_idx")),
                    "node_idx_in_stream": int(failure_match.group("node_idx")),
                    "raw_name": failure_match.group("func").strip(),
                    "reason": failure_match.group("reason").strip(),
                    "reason_code": failure_match.group("code").strip(),
                    "detail": failure_match.group("detail").strip(),
                    "line": line_no,
                }
            )
            continue

        if in_fused_scopes and (
            "[PrintFusedScopes]" in line or "[PrintFusedNodeIds]" in line
        ):
            scope_match = _FUSED_SCOPE_ID.search(line)
            if scope_match:
                scope_id = int(scope_match.group(1))
                current_fused = fused_scopes.setdefault(
                    scope_id,
                    {
                        "scope_id": scope_id,
                        "fused_node_ids": [],
                        "status": None,
                        "break": None,
                        "line": line_no,
                    },
                )
                continue
            if current_fused is None:
                continue
            node_ids_match = _FUSED_NODE_IDS.search(line)
            if node_ids_match:
                current_fused["fused_node_ids"] = _parse_int_list(
                    node_ids_match.group(1)
                )
                continue
            status_match = _FUSED_STATUS.search(line)
            if status_match:
                current_fused["status"] = status_match.group(1)
                continue
            break_match = _FUSED_BREAK.search(line)
            if break_match:
                current_fused["break"] = {
                    "break_reason": break_match.group("reason").strip(),
                    "trigger_node_id": int(break_match.group("trigger")),
                    "trigger_stream_idx": int(break_match.group("stream")),
                    "parent_scope_id": (
                        int(break_match.group("parent"))
                        if break_match.group("parent")
                        else None
                    ),
                    "sync_all_node_ids": _parse_int_list(break_match.group("sync")),
                    "detail": break_match.group("detail"),
                    "reason_detail": break_match.group("reason_detail").strip(),
                    "scope_name": break_match.group("scope_name").strip(),
                    "line": line_no,
                }
                continue

        # Keep unrelated diagnostic lines ignored, but flag lines that look
        # like records from a known section and no longer match our grammar.
        looks_like_failure = "[DumpFusionFailReasons]" in line and "Node [" in line
        looks_like_fused_node_ids = (
            "[PrintFusedNodeIds]" in line and "fusedNodeIds" in line
        )
        looks_like_fused_status = "[PrintFusedScopes]" in line and (
            "scopeStatus=" in line or "breakReason=[breakReason=" in line
        )
        if looks_like_failure or looks_like_fused_node_ids or looks_like_fused_status:
            unknown_count += 1
            if len(samples) < 10:
                samples.append(line.strip())
            continue

        if "[FORMAT_ERROR]" in line:
            error_count += 1
            format_error_count += 1
            if len(samples) < 10:
                samples.append(line.strip())

    if not failures and not fused_scopes:
        status = "partial"
        error_count += 1
    elif unknown_count or (error_count - format_error_count):
        status = "partial"
    else:
        status = "ready"
    return {
        "failures": failures,
        "fused_scopes": fused_scopes,
        "source": _source_status(
            path,
            status=status,
            line_count=line_count,
            record_count=len(failures) + len(fused_scopes),
            error_count=error_count,
            format_error_count=format_error_count,
            unknown_line_count=unknown_count,
            samples=samples,
            line_range=line_range,
        ),
    }


def _parse_node_detail(path: Path) -> dict[int, dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    for line_no, line in _iter_lines(path) or ():
        match = _NODE_DETAIL.search(line)
        if not match:
            continue
        node_id = int(match.group("node_id"))
        func_match = _FUNC_NAME.search(line)
        kernel_type_match = _KERNEL_TYPE.search(line)
        if "EventWait" in line or "type=WAIT" in line:
            node_type = "EVENT_WAIT"
        elif "EventReset" in line or "type=RESET" in line:
            node_type = "EVENT_RESET"
        elif "EventNotify" in line or "type=NOTIFY" in line:
            node_type = "EVENT_NOTIFY"
        elif "MemoryWrite" in line or "type=MEMORY_WRITE" in line:
            node_type = "MEMORY_WRITE"
        elif "MemoryWait" in line or "type=MEMORY_WAIT" in line:
            node_type = "MEMORY_WAIT"
        elif "KernelInfos{" in line:
            node_type = "KERNEL"
        else:
            node_type = "UNKNOWN"
        nodes[node_id] = {
            "node_id": node_id,
            "stream_id": int(match.group("stream_id")),
            "stream_idx_in_graph": int(match.group("stream_idx")),
            "node_idx_in_stream": int(match.group("node_idx")),
            "node_type": node_type,
            "func_name": func_match.group(1).strip() if func_match else "",
            "kernel_type": (
                kernel_type_match.group(1).strip() if kernel_type_match else ""
            ),
            "line": line_no,
        }
    return nodes


def _node_map(node_library: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for node in (node_library or {}).get("nodes", []):
        if not isinstance(node, dict):
            continue
        try:
            node_id = int(node.get("node_id"))
        except (TypeError, ValueError):
            continue
        result[node_id] = dict(node)
    return result


def _metadata_from_node(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    metadata: dict[str, Any] = {}
    implementation = node.get("implementation")
    if isinstance(implementation, dict):
        metadata.update(implementation)
    for key in ("backend", "implementation_backend", "mode", "kernel_mode"):
        if node.get(key) is not None:
            target = "backend" if "backend" in key else "mode"
            metadata[target] = node[key]
    return metadata or None


def _canonical_backend(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"ascendc", "ascend_c"}:
        return "ascendc"
    if normalized in {"triton"}:
        return "triton"
    if normalized in {"custom", "custom_operator", "custom_op"}:
        return "custom"
    return "unknown"


def _canonical_mode(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"static", "static_kernel"}:
        return "static"
    if normalized in {"dynamic", "dynamic_kernel"}:
        return "dynamic"
    return None


def _metadata_implementation(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict) or not metadata:
        return None
    implementation = metadata.get("implementation")
    source = implementation if isinstance(implementation, dict) else metadata
    backend_value = source.get("backend") or source.get("implementation_backend")
    mode_value = source.get("mode") or source.get("kernel_mode")
    if backend_value is None and mode_value is None:
        return None
    backend = _canonical_backend(backend_value)
    mode = _canonical_mode(mode_value)
    return {
        "backend": backend,
        "mode": mode,
        "confidence": "high",
        "source": "metadata",
    }


def normalize_operator_identity(
    raw_name: str | None, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a stable display identity without discarding the raw name."""

    raw = str(raw_name or "").strip()
    explicit = _metadata_implementation(metadata)
    implementation = explicit or {
        "backend": "unknown",
        "mode": None,
        "confidence": "none",
        "source": "unknown",
    }
    if not raw:
        return {
            "raw_name": "",
            "normalized_name": "未命名节点",
            "implementation": implementation,
            "name_pattern": {"kind": "missing", "hash_tokens": [], "suffix": ""},
        }

    static_match = _STATIC_HASH.match(raw)
    if static_match:
        if explicit is None:
            implementation = {
                "backend": "ascendc",
                "mode": "static",
                "confidence": "medium",
                "source": "name_pattern",
            }
        return {
            "raw_name": raw,
            "normalized_name": static_match.group("op"),
            "implementation": implementation,
            "name_pattern": {
                "kind": "static_kernel_hash",
                "hash_tokens": [static_match.group("hash")],
                "suffix": static_match.group("suffix") or "",
            },
        }

    if raw.lower().startswith("static_kernel_"):
        if explicit is None:
            implementation = {
                "backend": "ascendc",
                "mode": "static",
                "confidence": "medium",
                "source": "name_pattern",
            }
        return {
            "raw_name": raw,
            "normalized_name": raw[len("static_kernel_") :],
            "implementation": implementation,
            "name_pattern": {
                "kind": "static_kernel_prefix",
                "hash_tokens": [],
                "suffix": "",
            },
        }

    dynamic_match = _DYNAMIC_HASH.match(raw)
    if dynamic_match:
        if explicit is None:
            implementation = {
                "backend": "ascendc",
                "mode": "dynamic",
                "confidence": "medium",
                "source": "name_pattern",
            }
        return {
            "raw_name": raw,
            "normalized_name": dynamic_match.group("op"),
            "implementation": implementation,
            "name_pattern": {
                "kind": "dynamic_kernel_hash",
                "hash_tokens": [dynamic_match.group("hash")],
                "suffix": dynamic_match.group("suffix") or "",
            },
        }

    if raw.lower().startswith("dynamic_kernel_"):
        if explicit is None:
            implementation = {
                "backend": "ascendc",
                "mode": "dynamic",
                "confidence": "medium",
                "source": "name_pattern",
            }
        return {
            "raw_name": raw,
            "normalized_name": raw[len("dynamic_kernel_") :],
            "implementation": implementation,
            "name_pattern": {
                "kind": "dynamic_kernel_prefix",
                "hash_tokens": [],
                "suffix": "",
            },
        }

    triton_match = _TRITON.match(raw)
    if triton_match:
        if explicit is None:
            implementation = {
                "backend": "triton",
                "mode": None,
                "confidence": "medium",
                "source": "name_pattern",
            }
        return {
            "raw_name": raw,
            "normalized_name": triton_match.group("op"),
            "implementation": implementation,
            "name_pattern": {
                "kind": "triton_name",
                "hash_tokens": [],
                "suffix": "",
            },
        }

    generic_match = _GENERIC_HASH.match(raw)
    if generic_match:
        return {
            "raw_name": raw,
            "normalized_name": generic_match.group("op"),
            "implementation": implementation,
            "name_pattern": {
                "kind": "hex_suffix",
                "hash_tokens": [generic_match.group("hash")],
                "suffix": generic_match.group("suffix") or "",
            },
        }

    if raw.startswith(_STRUCTURAL_PREFIXES):
        return {
            "raw_name": raw,
            "normalized_name": raw,
            "implementation": implementation,
            "name_pattern": {"kind": "structural", "hash_tokens": [], "suffix": ""},
        }
    return {
        "raw_name": raw,
        "normalized_name": raw,
        "implementation": implementation,
        "name_pattern": {"kind": "plain", "hash_tokens": [], "suffix": ""},
    }


def _control_name(node: dict[str, Any] | None) -> str | None:
    if not isinstance(node, dict):
        return None
    node_type = str(node.get("node_type") or node.get("type") or "").upper()
    return _CONTROL_TYPES.get(node_type)


def _is_structural_node(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return False
    raw_name = str(node.get("func_name") or "")
    return raw_name.startswith(_STRUCTURAL_PREFIXES)


def _identity_for_node(
    node_id: int,
    nodes: dict[int, dict[str, Any]],
    failures_by_node: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    node = nodes.get(node_id) or {}
    failure = failures_by_node.get(node_id) or {}
    control = _control_name(node)
    if control:
        return normalize_operator_identity(control), "control"
    raw_name = str(node.get("func_name") or failure.get("raw_name") or "").strip()
    if not raw_name:
        return normalize_operator_identity("未命名节点"), "unknown"
    identity = normalize_operator_identity(raw_name, _metadata_from_node(node))
    name = identity.get("normalized_name", raw_name)
    if name.startswith(_STRUCTURAL_PREFIXES):
        return identity, "structural"
    return identity, "operator"


def _scope_nodes_from_library(
    scope_library: dict[str, Any] | None,
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for scope in (scope_library or {}).get("scopes", []):
        if not isinstance(scope, dict):
            continue
        scope_id = _scope_id(scope)
        if scope_id is None:
            continue
        for node in scope.get("nodes", []) or []:
            if isinstance(node, dict):
                value = node.get("node_id")
            else:
                value = node
            try:
                result[scope_id].add(int(value))
            except (TypeError, ValueError):
                continue
    return result


def _merge_scope_nodes(
    parsed_scopes: dict[int, dict[str, Any]], library_nodes: dict[int, set[int]]
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for scope_id, scope in parsed_scopes.items():
        result[scope_id].update(int(node_id) for node_id in scope.get("node_ids", []))
    for scope_id, node_ids in library_nodes.items():
        result[scope_id].update(node_ids)
    return result


def _expected_scope_ids(scope_library: dict[str, Any] | None) -> set[int]:
    result: set[int] = set()
    for scope in (scope_library or {}).get("scopes", []) or []:
        if not isinstance(scope, dict):
            continue
        scope_id = _scope_id(scope)
        if scope_id is not None:
            result.add(scope_id)
    return result


def _select_scope_instance(
    scope_data: dict[str, Any], scope_library: dict[str, Any] | None
) -> None:
    """Select the scope-split batch already assigned to this model instance.

    Model-instance ranges are derived from ``super_kernel.log`` and are not
    guaranteed to be line-compatible with ``sk_scope_split.log``.  The scoped
    library therefore acts as the identity anchor when multiple instances are
    present.
    """

    expected = _expected_scope_ids(scope_library)
    if not expected:
        return
    candidates: list[tuple[int, int, bool, dict[str, Any]]] = []
    for index, batch in enumerate(scope_data.get("batches", [])):
        ids = {
            int(scope["scope_id"])
            for scope in batch.get("scopes", [])
            if scope.get("scope_id") is not None
        }
        overlap = len(ids & expected)
        if overlap:
            candidates.append((overlap, index, ids == expected, batch))
    if not candidates:
        return
    _overlap, _index, _exact, selected = max(
        candidates, key=lambda item: (item[2], item[0], item[1])
    )
    scope_data["final_batch"] = selected
    scope_data["scopes"] = {
        int(scope["scope_id"]): scope
        for scope in selected.get("scopes", [])
        if scope.get("scope_id") is not None and int(scope["scope_id"]) in expected
    }
    scope_data["breaks"] = [
        record
        for record in selected.get("breaks", [])
        if record.get("scope_id") in scope_data["scopes"]
    ]


def _filter_failure_instance(
    failure_data: dict[str, Any],
    scope_library: dict[str, Any] | None,
    boundary_records: list[dict[str, Any]],
) -> None:
    expected_scope_ids = _expected_scope_ids(scope_library)
    if not expected_scope_ids:
        return
    expected_nodes = set()
    for node_ids in _scope_nodes_from_library(scope_library).values():
        expected_nodes.update(node_ids)
    trigger_nodes = {
        int(record["trigger_node_id"])
        for record in boundary_records
        if record.get("trigger_node_id") is not None
    }
    if expected_nodes:
        failure_data["failures"] = [
            failure
            for failure in failure_data.get("failures", [])
            if int(failure.get("node_id", -1)) in expected_nodes
            or int(failure.get("node_id", -1)) in trigger_nodes
        ]
    failure_data["fused_scopes"] = {
        int(scope_id): fused
        for scope_id, fused in failure_data.get("fused_scopes", {}).items()
        if int(scope_id) in expected_scope_ids
    }


def _failure_identity(
    failure: dict[str, Any], nodes: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    node = nodes.get(int(failure["node_id"]), {})
    return normalize_operator_identity(
        failure.get("raw_name"), _metadata_from_node(node)
    )


def _failure_explanation(
    failure: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any] | None:
    if failure.get("reason_code") == "BINDMAP_INIT_EMPTY" and not str(
        identity.get("raw_name") or ""
    ).lower().startswith("static_kernel_"):
        return {
            "rule_id": "bindmap_init_empty_without_static_prefix",
            "summary": "算子不支持静态编译，或静态编译失败",
            "confidence": "high",
        }
    return None


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _implementation_signature(implementation: Any) -> tuple[str, str | None]:
    if not isinstance(implementation, dict):
        return ("unknown", None)
    return (
        _canonical_backend(implementation.get("backend")),
        _canonical_mode(implementation.get("mode")),
    )


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _source_available(source: dict[str, Any]) -> bool:
    return source.get("status") in {"ready", "partial"}


def build_fusion_analysis(
    model_dir: str | Path,
    *,
    node_library: dict[str, Any] | None = None,
    scope_library: dict[str, Any] | None = None,
    fused_library: dict[str, Any] | None = None,
    model_instance_range: tuple[int, int] | None = None,
    model_instance_id: str | None = None,
) -> dict[str, Any]:
    """Build optional fusion evidence for one model instance.

    The returned object is self-contained and safe to attach to an existing
    report.  Missing or malformed optional sources result in an unavailable or
    partial envelope instead of an exception.
    """

    model_path = Path(model_dir)
    scope_path = model_path / "sk_scope_split.log"
    failure_path = model_path / "sk_fusion_fail_reasons.log"
    has_scoped_library = bool((scope_library or {}).get("scopes"))
    source_line_range = None if has_scoped_library else model_instance_range
    scope_data = _parse_scope_split(scope_path, source_line_range)
    _select_scope_instance(scope_data, scope_library)
    failure_data = _parse_fusion_failures(failure_path, source_line_range)
    if model_instance_range is not None:
        _filter_failure_instance(
            failure_data, scope_library, scope_data.get("breaks", [])
        )

    nodes = _node_map(node_library)
    if not nodes:
        nodes = _parse_node_detail(model_path / "sk_node_detail.log")
    failures = list(failure_data["failures"])
    failures_by_node: dict[int, dict[str, Any]] = {}
    failure_identities: dict[int, dict[str, Any]] = {}
    for failure in failures:
        node_id = int(failure["node_id"])
        failures_by_node.setdefault(node_id, failure)
        failure_identities[node_id] = _failure_identity(failure, nodes)

    parsed_scope_nodes = _merge_scope_nodes(
        scope_data["scopes"], _scope_nodes_from_library(scope_library)
    )
    fused_scopes = dict(failure_data["fused_scopes"])
    for function in (fused_library or {}).get("functions", []) or []:
        if not isinstance(function, dict):
            continue
        try:
            scope_id = int(function.get("scope_id"))
        except (TypeError, ValueError):
            continue
        fused = fused_scopes.setdefault(
            scope_id,
            {
                "scope_id": scope_id,
                "fused_node_ids": [],
                "status": "SUCCESS",
                "break": None,
            },
        )
        fused["fused_node_ids"] = list(
            dict.fromkeys(
                list(fused.get("fused_node_ids", []))
                + [int(node_id) for node_id in function.get("node_ids", [])]
            )
        )
        fused["status"] = fused.get("status") or "SUCCESS"

    final_batch = scope_data.get("final_batch") or {}
    final_pass = str(final_batch.get("pass") or "")
    final_scope_order: list[int] = []
    selected_scope_ids = set(scope_data.get("scopes", {}))
    for scope in final_batch.get("scopes", []) or []:
        scope_id = _scope_id(scope)
        if scope_id is None:
            continue
        if selected_scope_ids and scope_id not in selected_scope_ids:
            continue
        if scope_id not in final_scope_order:
            final_scope_order.append(scope_id)
    if not final_scope_order:
        # A changed or missing scope-split format can still leave the scoped
        # library usable.  Preserve its order for adjacency in the fallback
        # fused-scope evidence path.
        for scope in (scope_library or {}).get("scopes", []) or []:
            scope_id = _scope_id(scope)
            if scope_id is not None and scope_id not in final_scope_order:
                final_scope_order.append(scope_id)
    final_scope_positions = {
        scope_id: index for index, scope_id in enumerate(final_scope_order)
    }

    def scope_fusion_status(scope_id: int | None) -> str:
        if scope_id is None:
            return "UNKNOWN"
        status = (fused_scopes.get(int(scope_id)) or {}).get("status")
        return str(status or "UNKNOWN").upper()

    boundaries: list[dict[str, Any]] = []
    impacts: dict[str, dict[str, Any]] = {}

    def get_impact(identity: dict[str, Any], source_kind: str) -> dict[str, Any]:
        key = str(
            identity.get("normalized_name") or identity.get("raw_name") or "未命名节点"
        )
        if key not in impacts:
            impacts[key] = {
                "operator": dict(identity),
                "source_kind": source_kind,
                "direct_breaks": {
                    "total": 0,
                    "by_reason": {},
                    "boundary_ids": [],
                    "trigger_node_ids": [],
                },
                "observed_effect": {
                    "affected_region_count": 0,
                    "affected_scope_ids": [],
                    "adjacent_scope_count": 0,
                    "adjacent_scope_ids": [],
                    "successful_adjacent_scope_count": 0,
                    "successful_adjacent_scope_ids": [],
                    "same_region_failure_kernel_count": 0,
                    "same_region_failure_node_ids": [],
                    "observed_unfused_kernel_count": 0,
                    "observed_unfused_kernel_node_ids": [],
                },
                "evidence": {
                    "failure_count": 0,
                    "failure_reason_counts": {},
                    "failure_node_ids": [],
                    "raw_examples": [],
                    "explanations": [],
                },
                "targets": [],
                "implementation_variants": [],
            }
        item = impacts[key]
        variant = item["implementation_variants"]
        implementation = identity.get("implementation")
        if implementation not in variant:
            variant.append(implementation)
        return item

    for failure in failures:
        identity = failure_identities[int(failure["node_id"])]
        source_kind = (
            "structural"
            if identity["name_pattern"]["kind"] == "structural"
            else "operator"
        )
        item = get_impact(identity, source_kind)
        evidence = item["evidence"]
        evidence["failure_count"] += 1
        reason_key = (
            f"{failure.get('reason', '')}|{failure.get('reason_code', '')}".strip("|")
        )
        counts = Counter(evidence.get("failure_reason_counts", {}))
        counts[reason_key] += 1
        evidence["failure_reason_counts"] = _counter_dict(counts)
        _append_unique(evidence["failure_node_ids"], int(failure["node_id"]))
        if (
            failure.get("raw_name")
            and failure["raw_name"] not in evidence["raw_examples"]
        ):
            if len(evidence["raw_examples"]) < 5:
                evidence["raw_examples"].append(failure["raw_name"])
        explanation = _failure_explanation(failure, identity)
        if explanation and explanation not in evidence["explanations"]:
            evidence["explanations"].append(explanation)

    boundary_records = list(scope_data.get("breaks", []))
    boundary_keys = {
        (
            int(boundary.get("scope_id"))
            if boundary.get("scope_id") is not None
            else None,
            int(boundary.get("trigger_node_id"))
            if boundary.get("trigger_node_id") is not None
            else None,
            str(boundary.get("break_reason") or "UNKNOWN"),
        )
        for boundary in boundary_records
    }
    # sk_fusion_fail_reasons.log carries the same causal break in its fused
    # scope dump.  Use it to recover a boundary when sk_scope_split.log is
    # missing or its boundary line format changes, while deduplicating the
    # normal two-source case.
    for scope_id, fused in fused_scopes.items():
        fused_break = fused.get("break") if isinstance(fused, dict) else None
        if (
            not isinstance(fused_break, dict)
            or fused_break.get("trigger_node_id") is None
        ):
            continue
        if selected_scope_ids and int(scope_id) not in selected_scope_ids:
            continue
        fallback_key = (
            int(scope_id),
            int(fused_break.get("trigger_node_id"))
            if fused_break.get("trigger_node_id") is not None
            else None,
            str(fused_break.get("break_reason") or "UNKNOWN"),
        )
        if fallback_key in boundary_keys:
            continue
        boundary_records.append(
            {
                "scope_id": int(scope_id),
                "display_scope_id": None,
                "break_reason": fused_break.get("break_reason"),
                "trigger_node_id": fused_break.get("trigger_node_id"),
                "trigger_stream_idx": fused_break.get("trigger_stream_idx"),
                "parent_scope_id": fused_break.get("parent_scope_id"),
                "sync_all_node_ids": list(fused_break.get("sync_all_node_ids", [])),
                "detail": fused_break.get("detail", ""),
                "line": fused_break.get("line"),
                "pass": final_pass or "FusedScopeLog",
                "source_kind": "fused_scope_log",
            }
        )
        boundary_keys.add(fallback_key)

    for ordinal, boundary in enumerate(boundary_records):
        trigger_id = int(boundary["trigger_node_id"])
        identity, source_kind = _identity_for_node(trigger_id, nodes, failures_by_node)
        scope_id = boundary.get("scope_id")
        boundary_id = ":".join(
            str(value)
            for value in (
                model_instance_id or "instance",
                final_pass or "final",
                scope_id if scope_id is not None else "unassigned",
                boundary.get("break_reason") or "unknown",
                trigger_id,
                ordinal,
            )
        )
        region_node_ids = (
            set(parsed_scope_nodes.get(int(scope_id), set()))
            if scope_id is not None
            else set()
        )
        failure_node_ids = {
            node_id for node_id in region_node_ids if node_id in failures_by_node
        }
        fused_node_ids = set(
            fused_scopes.get(int(scope_id), {}).get("fused_node_ids", [])
            if scope_id is not None
            else []
        )
        unfused_kernel_ids = {
            node_id
            for node_id in region_node_ids - fused_node_ids
            if (
                nodes.get(node_id)
                and not _control_name(nodes.get(node_id))
                and not _is_structural_node(nodes.get(node_id))
                and (
                    str(nodes[node_id].get("node_type") or "").upper() == "KERNEL"
                    or bool(nodes[node_id].get("func_name"))
                )
            )
        }
        if source_kind == "operator":
            # A break trigger is outside the owning scope in the raw logs, but
            # it is still the kernel that directly caused the boundary.
            unfused_kernel_ids.add(trigger_id)

        scope_position = (
            final_scope_positions.get(int(scope_id)) if scope_id is not None else None
        )
        previous_scope_id = (
            final_scope_order[scope_position - 1]
            if scope_position is not None and scope_position > 0
            else None
        )
        following_scope_id = (
            final_scope_order[scope_position + 1]
            if scope_position is not None
            and scope_position + 1 < len(final_scope_order)
            else None
        )
        adjacent_scope_ids = [
            scope_id
            for scope_id in (previous_scope_id, following_scope_id)
            if scope_id is not None
        ]
        successful_adjacent_scope_ids = [
            adjacent_id
            for adjacent_id in adjacent_scope_ids
            if scope_fusion_status(adjacent_id) == "SUCCESS"
        ]
        target_refs = []
        if scope_id is not None:
            target_refs.append(
                {"kind": "scope", "stage": "before", "scope_id": str(scope_id)}
            )
        target_refs.append(
            {"kind": "node", "stage": "before", "node_id": str(trigger_id)}
        )
        if previous_scope_id is not None:
            target_refs.append(
                {
                    "kind": "scope",
                    "stage": "context",
                    "relation": "previous",
                    "scope_id": str(previous_scope_id),
                }
            )
        if following_scope_id is not None:
            target_refs.append(
                {
                    "kind": "scope",
                    "stage": "context",
                    "relation": "following",
                    "scope_id": str(following_scope_id),
                }
            )
        record = {
            "id": boundary_id,
            "pass": boundary.get("pass") or final_pass,
            "scope_id": scope_id,
            "parent_scope_id": boundary.get("parent_scope_id")
            or (fused_scopes.get(int(scope_id), {}).get("break") or {}).get(
                "parent_scope_id"
            ),
            "break_reason": boundary.get("break_reason"),
            "detail": boundary.get("detail", ""),
            "break_reason_detail": (
                (fused_scopes.get(int(scope_id), {}).get("break") or {}).get(
                    "reason_detail"
                )
                if scope_id is not None
                else None
            ),
            "scope_fusion_status": scope_fusion_status(scope_id),
            "previous_scope_id": previous_scope_id,
            "following_scope_id": following_scope_id,
            "adjacent_scope_ids": adjacent_scope_ids,
            "successful_adjacent_scope_ids": successful_adjacent_scope_ids,
            "trigger": {
                "node_id": trigger_id,
                "stream_idx_in_graph": boundary.get("trigger_stream_idx"),
                "operator": identity,
                "source_kind": source_kind,
            },
            "sync_all_node_ids": list(boundary.get("sync_all_node_ids", [])),
            "same_region_failure_node_ids": sorted(failure_node_ids),
            "observed_unfused_kernel_node_ids": sorted(unfused_kernel_ids),
            "targets": target_refs,
            "source": {
                "file": str(
                    failure_path
                    if boundary.get("source_kind") == "fused_scope_log"
                    else scope_path
                ),
                "line": boundary.get("line"),
            },
        }
        boundaries.append(record)
        item = get_impact(identity, source_kind)
        direct = item["direct_breaks"]
        direct["total"] += 1
        reason_counts = Counter(direct.get("by_reason", {}))
        reason_counts[str(boundary.get("break_reason") or "UNKNOWN")] += 1
        direct["by_reason"] = _counter_dict(reason_counts)
        _append_unique(direct["boundary_ids"], boundary_id)
        _append_unique(direct["trigger_node_ids"], trigger_id)
        observed = item["observed_effect"]
        if scope_id is not None:
            _append_unique(observed["affected_scope_ids"], str(scope_id))
        for adjacent_scope_id in adjacent_scope_ids:
            _append_unique(observed["adjacent_scope_ids"], str(adjacent_scope_id))
        for adjacent_scope_id in successful_adjacent_scope_ids:
            _append_unique(
                observed["successful_adjacent_scope_ids"], str(adjacent_scope_id)
            )
        for node_id in sorted(failure_node_ids):
            _append_unique(observed["same_region_failure_node_ids"], node_id)
        for node_id in sorted(unfused_kernel_ids):
            _append_unique(observed["observed_unfused_kernel_node_ids"], node_id)
        item["targets"].extend(target_refs)

    for item in impacts.values():
        observed = item["observed_effect"]
        observed["affected_scope_ids"] = sorted(set(observed["affected_scope_ids"]))
        observed["affected_region_count"] = len(observed["affected_scope_ids"])
        observed["adjacent_scope_ids"] = sorted(
            set(observed["adjacent_scope_ids"]), key=lambda value: int(value)
        )
        observed["adjacent_scope_count"] = len(observed["adjacent_scope_ids"])
        observed["successful_adjacent_scope_ids"] = sorted(
            set(observed["successful_adjacent_scope_ids"]), key=lambda value: int(value)
        )
        observed["successful_adjacent_scope_count"] = len(
            observed["successful_adjacent_scope_ids"]
        )
        observed["same_region_failure_node_ids"] = sorted(
            set(observed["same_region_failure_node_ids"])
        )
        observed["same_region_failure_kernel_count"] = len(
            observed["same_region_failure_node_ids"]
        )
        observed["observed_unfused_kernel_node_ids"] = sorted(
            set(observed["observed_unfused_kernel_node_ids"])
        )
        observed["observed_unfused_kernel_count"] = len(
            observed["observed_unfused_kernel_node_ids"]
        )
        item["targets"] = [
            target
            for index, target in enumerate(item["targets"])
            if target not in item["targets"][:index]
        ]
        item["implementation_variants"] = [
            variant
            for index, variant in enumerate(item["implementation_variants"])
            if variant not in item["implementation_variants"][:index]
        ]
        if (
            len(
                {
                    _implementation_signature(variant)
                    for variant in item["implementation_variants"]
                }
            )
            > 1
        ):
            # The same normalized name can represent dynamic, static, Triton,
            # or custom implementations.  Do not present the first observed
            # variant as a fact when the evidence is mixed.
            item["operator"]["implementation"] = {
                "backend": "unknown",
                "mode": None,
                "confidence": "none",
                "source": "mixed_variants",
            }
        item["impact_status"] = (
            "direct" if item["direct_breaks"]["total"] else "evidence_only"
        )

    fusion_scope_ids = selected_scope_ids or set(fused_scopes)
    successful_sk_count = sum(
        1
        for scope_id, fused in fused_scopes.items()
        if int(scope_id) in fusion_scope_ids
        and str(fused.get("status") or "").upper() == "SUCCESS"
    )
    control_count = sum(
        1
        for boundary in boundaries
        if boundary["trigger"].get("source_kind") == "control"
    )
    operator_count = sum(
        1
        for boundary in boundaries
        if boundary["trigger"].get("source_kind") == "operator"
    )
    unclassified_count = len(boundaries) - control_count - operator_count
    source_statuses = [scope_data["source"], failure_data["source"]]
    available_count = sum(_source_available(source) for source in source_statuses)
    if available_count == 0:
        status = "unavailable"
    elif all(source.get("status") == "ready" for source in source_statuses):
        status = "ready"
    else:
        status = "partial"

    # Keep the output compact enough for an offline report while retaining all
    # node IDs needed for later Graph View highlighting.
    return {
        "schema_version": "1",
        "status": status,
        "summary": {
            "final_pass": final_pass or None,
            "final_region_count": len(scope_data.get("scopes", {}))
            or len((scope_library or {}).get("scopes", []))
            or len(fusion_scope_ids),
            "successful_sk_count": successful_sk_count,
            "boundary_count": len(boundaries),
            "operator_triggered_boundary_count": operator_count,
            "control_boundary_count": control_count,
            "unclassified_boundary_count": unclassified_count,
            "failure_record_count": len(failures),
            "unique_failure_node_count": len(failures_by_node),
        },
        "operator_impacts": sorted(
            impacts.values(),
            key=lambda item: (
                -int(item["direct_breaks"]["total"]),
                str(item["operator"].get("normalized_name") or ""),
            ),
        ),
        "boundaries": boundaries,
        "node_failures": [
            {
                **failure,
                "operator": failure_identities[int(failure["node_id"])],
                "source": {"file": str(failure_path), "line": failure.get("line")},
            }
            for failure in failures
        ],
        "sources": {
            "sk_scope_split.log": scope_data["source"],
            "sk_fusion_fail_reasons.log": failure_data["source"],
        },
        "coverage": {
            "final_boundary_records": len(boundaries),
            "boundary_trigger_node_matched_count": sum(
                1
                for boundary in boundaries
                if boundary["trigger"]["source_kind"] != "unknown"
            ),
            "boundary_trigger_node_unmatched_count": sum(
                1
                for boundary in boundaries
                if boundary["trigger"]["source_kind"] == "unknown"
            ),
            "historical_pass_count": max(0, len(scope_data.get("batches", [])) - 1),
        },
    }


__all__ = ["build_fusion_analysis", "normalize_operator_identity"]
