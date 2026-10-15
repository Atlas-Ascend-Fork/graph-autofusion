# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Parser for the ``op_summary.csv`` profiling format used by MiniDAG."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

from ..common.log import logger
from ..minidag.types import _f32


class _Profile:
    __slots__ = ("op_name", "execution_time", "cube_block_num", "vec_block_num")

    def __init__(
        self,
        op_name: str,
        execution_time: float,
        cube_block_num: int,
        vec_block_num: int,
    ) -> None:
        self.op_name = op_name
        self.execution_time = execution_time
        self.cube_block_num = cube_block_num
        self.vec_block_num = vec_block_num


_REQUIRED_COLUMNS = (
    "Op Name",
    "Task Type",
    "Task Duration(us)",
    "Block Num",
    "Mix Block Num",
)
_UINT32_MAX = (1 << 32) - 1


def _calculate_core_counts(
    task_type: str, block_num: int, mix_block_num: int
) -> "tuple[int, int]":
    cube = vec = 0
    if task_type == "AI_CORE":
        cube = block_num
    elif task_type == "AI_VECTOR_CORE":
        vec = block_num
    elif task_type == "MIX_AIC":
        cube, vec = block_num, mix_block_num
    elif task_type == "MIX_AIV":
        cube, vec = mix_block_num, block_num
    return cube, vec


def _parse_row(row: Mapping[str, str]) -> Optional[_Profile]:
    # Checks run in the order of ProfilingParser::ParseRow (columns, duration,
    # block counts) and log the same warnings, so a malformed row still tells
    # which field was wrong.
    columns = [key for key in row if key is not None]
    for column in _REQUIRED_COLUMNS:
        if column not in row:
            logger.warning("Required column not found: %s", column)
            return None
        # ``csv.DictReader`` uses ``None`` for a short row.  The C++ parser
        # rejects such rows when a required column index falls outside
        # ``fields``; mirror that instead of building a partial profile.
        if not isinstance(row[column], str):
            field_size = sum(row[key] is not None for key in columns) + len(
                row.get(None) or ()
            )
            logger.warning(
                "Invalid column index %d for field size %d",
                columns.index(column),
                field_size,
            )
            return None
    duration_text = row["Task Duration(us)"]
    try:
        # ``ProfilingData::execution_time`` is a C++ ``float``.  Round at the
        # parser boundary so an out-of-range value is handled as a malformed
        # row (the reference ``std::stof`` path does the same) rather than
        # surfacing as an ``OverflowError`` halfway through allocation.
        duration = _f32(float(duration_text))
    except (OverflowError, ValueError):
        logger.warning("Failed to parse Task Duration: %s", duration_text)
        return None
    # ``std::stof`` can produce infinities/NaNs for textual special values.
    # Neither is a meaningful execution cost and allowing one through would
    # poison weighted-score comparisons.  Treat it as a malformed row.
    if not math.isfinite(duration):
        logger.warning("Failed to parse Task Duration: %s", duration_text)
        return None
    # A zero or negative sample is not a usable cost, and writing 0.0 into an
    # effective cost would turn the sample into the exclusion marker the GE
    # adapter uses.  Skip the row instead.
    if duration <= 0.0:
        logger.warning("Skip non-positive Task Duration: %s", duration_text)
        return None
    try:
        block_num = int(row["Block Num"])
        mix_block_num = int(row["Mix Block Num"])
    except ValueError:
        logger.warning("Failed to parse Block Num or Mix Block Num")
        return None
    # The reference parser reads these fields into uint32_t.  Python integers
    # are unbounded, so enforce the source range before storing them; this also
    # avoids silently accepting negative counts or values that would wrap in
    # the C++ implementation.
    if not (0 <= block_num <= _UINT32_MAX and 0 <= mix_block_num <= _UINT32_MAX):
        logger.warning("Failed to parse Block Num or Mix Block Num")
        return None
    cube, vec = _calculate_core_counts(row["Task Type"], block_num, mix_block_num)
    return _Profile(row["Op Name"], duration, cube, vec)


def parse(csv_path: str) -> Dict[str, _Profile]:
    """Parse *csv_path* and return profiles keyed by operation name.

    Missing/malformed rows are ignored like the C++ parser.  A missing file or
    required header is an input error and raises ``ValueError`` so the outer
    allocation transaction can fail explicitly.
    """

    if not isinstance(csv_path, (str, os.PathLike)):
        raise TypeError("profiling path must be a string or PathLike")
    try:
        path_value = os.fspath(csv_path)
    except TypeError as exc:
        raise TypeError("profiling path must be a string or PathLike") from exc
    if isinstance(path_value, bytes):
        raise TypeError("profiling path must be a text path")
    path = Path(path_value)
    # File-level errors log the ProfilingParser::Parse wording where they are
    # found; the ValueError then fails the allocation.
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames
            if headers is None:
                logger.error("Failed to read header line from: %s", csv_path)
                raise ValueError(f"cannot read profiling header: {csv_path}")
            missing = [column for column in _REQUIRED_COLUMNS if column not in headers]
            if missing:
                for column in missing:
                    logger.error("Required column not found: %s", column)
                raise ValueError(
                    "profiling CSV missing required columns: " + ", ".join(missing)
                )
            profiles: Dict[str, _Profile] = {}
            for row in reader:
                profile = _parse_row(row)
                if profile is not None:
                    profiles[profile.op_name] = profile
    except FileNotFoundError as exc:
        logger.error("Failed to get real path for profiling file: %s", csv_path)
        raise ValueError(f"cannot open profiling file: {csv_path}") from exc
    except OSError as exc:
        logger.error("Failed to open profiling file: %s", csv_path)
        raise ValueError(f"cannot open profiling file: {csv_path}") from exc
    except UnicodeError as exc:
        raise ValueError(f"cannot decode profiling file: {csv_path}") from exc
    except csv.Error as exc:
        raise ValueError(f"invalid profiling CSV: {csv_path}") from exc
    logger.info("Parsed %d profiling records from %s", len(profiles), csv_path)
    return profiles


def _without_scope(name: str) -> str:
    pos = name.rfind("/")
    return name[pos + 1 :] if pos >= 0 and pos + 1 < len(name) else name


def match_profile(
    profiles: Mapping[str, _Profile], node_name: str
) -> Optional[_Profile]:
    """Match by full operation name, then by the final scope component."""

    profile = profiles.get(node_name)
    if profile is not None:
        return profile
    return profiles.get(_without_scope(node_name))


__all__ = ["_Profile", "parse", "match_profile"]
