#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Canonical, JSON-serialisable schema used by the ``evidence`` command.

The legacy ATT parser returns a dataclass intended for human-oriented CSV
output.  This module is deliberately small and has no dependency on the CLI
or parser internals so that the skill can consume the generated JSONL later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Mapping, Optional


SCHEMA_VERSION = "att-evidence/v1"


_PIPE_ALIASES = {"M2": "AIV_MTE2", "M3": "AIV_MTE3", "V": "AIV_VEC"}


def normalize_pipe_estimates(pipe: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    values = pipe or {}
    normalized = {
        name: values.get(name, values.get(alias))
        for alias, name in _PIPE_ALIASES.items()
    }
    normalized.update(
        (name, value) for name, value in values.items() if name not in _PIPE_ALIASES
    )
    return normalized


@dataclass
class FinalTilingRecord:
    """A normalized final ATT tiling observation."""

    schema: Any = None
    source: Optional[str] = None
    selection_mode: Optional[str] = None
    op: Optional[str] = None
    graph: Optional[int] = None
    result: Optional[int] = None
    group: Optional[int] = None
    case: Optional[int] = None
    tiling_key: Optional[int] = None
    score: Optional[int] = None
    sub_case_tag: Optional[str] = None
    template_name: Optional[str] = None
    pipe_est: Optional[Mapping[str, Any]] = None
    tiling_repr: Optional[Any] = None
    repr_kind: Optional[str] = None
    repr_hash: Optional[str] = None
    parse_status: str = "ok"
    source_path: str = ""
    source_line: Optional[int] = None
    output_index: Optional[int] = None


@dataclass
class FinalTilingSummaryRecord:
    """Result-level mapping of groups to their final template identities."""

    schema: Any = None
    source: Optional[str] = None
    selection_mode: Optional[str] = None
    op: Optional[str] = None
    graph: Optional[int] = None
    result: Optional[int] = None
    groups: Optional[Mapping[str, Any]] = None
    parse_status: str = "ok"
    source_path: str = ""
    source_line: Optional[int] = None


def _json_value(value: Any) -> Any:
    """Convert parser values to values accepted by :mod:`json`.

    In particular, dictionaries may contain values represented as numpy-like
    scalar objects in downstream integrations.  ``item`` is used when
    available while preserving ordinary Python values unchanged.
    """

    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def make_record(
    *,
    operator: str,
    graph_id: Optional[int],
    result_id: Optional[int],
    group_id: Optional[int],
    case_id: Optional[int],
    tiling_values: Optional[Mapping[str, Any]] = None,
    objective: Any = None,
    source_path: str,
    source_line: Optional[int],
    parse_status: str = "ok",
    result_performance: Any = None,
    source_lines: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Build one canonical ATT evidence record.

    The required fields are intentionally explicit rather than copying an
    ``OperatorSummary`` wholesale.  Additional fields are additive and keep
    provenance useful to consumers without changing the required schema.
    """

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "operator": operator,
        "graph_id": graph_id,
        "result_id": result_id,
        "group_id": group_id,
        "case_id": case_id,
        "tiling_values": dict(tiling_values or {}),
        "objective": objective,
        "source_path": source_path,
        "source_line": source_line,
        "parse_status": parse_status,
    }
    if result_performance is not None:
        record["result_performance"] = result_performance
    if source_lines:
        record["source_lines"] = dict(source_lines)
    return _json_value(record)


def record_from_summary(
    summary: Any, source_path: str, source_line: Optional[int]
) -> Dict[str, Any]:
    """Adapt a legacy ``OperatorSummary`` to the canonical record shape."""

    return make_record(
        operator=summary.operator_name,
        graph_id=summary.graph,
        result_id=summary.result,
        group_id=summary.group,
        case_id=summary.case,
        tiling_values=summary.tiling_values,
        objective=summary.objective_value,
        result_performance=summary.result_performance,
        source_path=source_path,
        source_line=source_line,
        parse_status=summary.parse_status,
    )
