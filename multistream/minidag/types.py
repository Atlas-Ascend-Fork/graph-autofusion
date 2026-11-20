# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Public and private value types used by :mod:`multistream`.

This module intentionally has no imports from other multistream modules.  Keeping
the constants and the small value objects here makes importing the public
package cheap for hosts which only need to inspect a strategy.
"""

from __future__ import annotations

from enum import Enum
import math
import numbers
import struct
from typing import Any, Sequence


INVALID_STREAM_ID = -1
MAX_STREAM_NUM = 64


class CoreType(Enum):
    """Execution engine a node's kernel runs on.

    A node attribute set through ``DAGNode.set_core_type``: it describes the
    kernel itself, not its cost, so it is not a ``NodeCost`` field.  The values
    repeat the member names: unlike ``MergeStrategy`` this type is only passed
    between Python layers and never appears in a GE option or in a serialized
    artefact, so there is nothing to align the spelling with.
    """

    __slots__ = ()

    AIC = "AIC"
    MIX_AIC = "MIX_AIC"
    AIV = "AIV"
    MIX_AIV = "MIX_AIV"
    UNKNOWN = "UNKNOWN"


class MergeStrategy(Enum):
    """Strategy used to merge logical paths onto physical streams."""

    __slots__ = ()

    # Keep the serialized spellings used by GE's option parser.  The enum
    # member names provide the Python-facing constant names.
    LOAD_BALANCE = "LoadBalance"
    MAIN_STREAM = "MainStream"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "loadbalance": cls.LOAD_BALANCE,
                "load_balance": cls.LOAD_BALANCE,
                "mainstream": cls.MAIN_STREAM,
                "main_stream": cls.MAIN_STREAM,
            }
            return aliases.get(key)
        return None


class NodeCost:
    """Immutable per-node cost values consumed by stream planning.

    ``NodeCost`` deliberately accepts keyword arguments only.  The field
    names form the public contract, while the values themselves are copied
    into a node and into each plan's effective-cost array.  Keeping the
    object immutable prevents profiling (or another plan) from accidentally
    changing the graph's source cost.

    The core counts come in two pairs with different sources.
    ``cube_block_num`` / ``vec_block_num`` are the cores the kernel actually
    occupies, parsed from profiling (or given by a cost model).
    ``cube_core_num_limit`` / ``vec_core_num_limit`` are the upper bounds the
    user configured; 0 means no limit.  Profiling only ever writes the first
    pair.
    """

    __slots__ = (
        "execution_time",
        "memory_usage",
        "bandwidth_usage",
        "cube_block_num",
        "vec_block_num",
        "cube_core_num_limit",
        "vec_core_num_limit",
    )

    def __init__(
        self,
        *,
        execution_time: float = -1.0,
        memory_usage: int = 0,
        bandwidth_usage: int = 0,
        cube_block_num: int = 0,
        vec_block_num: int = 0,
        cube_core_num_limit: int = 0,
        vec_core_num_limit: int = 0,
    ) -> None:
        # Do not call ``self.__setattr__`` here: all writes after
        # construction are intentionally rejected by the immutable value
        # object below.
        rounded_execution_time = _validate_execution_time(execution_time)
        normalized_values = {
            "memory_usage": _validate_nonnegative_int("memory_usage", memory_usage),
            "bandwidth_usage": _validate_nonnegative_int(
                "bandwidth_usage", bandwidth_usage
            ),
            "cube_block_num": _validate_nonnegative_int(
                "cube_block_num", cube_block_num
            ),
            "vec_block_num": _validate_nonnegative_int("vec_block_num", vec_block_num),
            "cube_core_num_limit": _validate_nonnegative_int(
                "cube_core_num_limit", cube_core_num_limit
            ),
            "vec_core_num_limit": _validate_nonnegative_int(
                "vec_core_num_limit", vec_core_num_limit
            ),
        }
        object.__setattr__(self, "execution_time", rounded_execution_time)
        for field_name, value in normalized_values.items():
            object.__setattr__(self, field_name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("NodeCost is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("NodeCost is immutable")

    def __repr__(self) -> str:
        values = ", ".join(
            f"{field}={getattr(self, field)!r}" for field in self.__slots__
        )
        return f"NodeCost({values})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeCost):
            return NotImplemented
        return all(
            getattr(self, field) == getattr(other, field) for field in self.__slots__
        )

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, field) for field in self.__slots__))


def _validate_execution_time(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise TypeError("execution_time must be a number")
    try:
        value_as_float = float(value)
    except TypeError as exc:
        raise TypeError("execution_time must be a real number") from exc
    except (OverflowError, ValueError) as exc:
        raise ValueError("execution_time is outside the finite float32 range") from exc
    if not math.isfinite(value_as_float):
        raise ValueError("execution_time must be finite")
    try:
        rounded = _f32(value_as_float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("execution_time is outside the finite float32 range") from exc
    if not math.isfinite(rounded):
        raise ValueError("execution_time must be finite")
    return rounded


def _validate_nonnegative_int(field_name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{field_name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _f32(value: float) -> float:
    """Round *value* to the IEEE-754 binary32 representation used by GE."""

    # ``struct`` also gives us the same overflow behaviour as a C++ float
    # conversion (OverflowError is intentionally allowed to propagate).
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _cost_of(node: Any, costs: Sequence[NodeCost]) -> NodeCost:
    """Return the cost corresponding to a node's dense insertion index."""

    index = getattr(node, "_index", -1)
    if not isinstance(index, int) or index < 0 or index >= len(costs):
        if not costs:
            raise IndexError(
                f"node cost index {index!r} is out of range (costs is empty)"
            )
        raise IndexError(f"node cost index {index!r} is out of range [0, {len(costs)})")
    return costs[index]


__all__ = [
    "NodeCost",
    "CoreType",
    "MergeStrategy",
    "INVALID_STREAM_ID",
    "MAX_STREAM_NUM",
]
