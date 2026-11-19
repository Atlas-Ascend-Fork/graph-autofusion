# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Read-only result of stream planning."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .node import DAGNode


_KERNEL_TYPES = ("AIC", "AIV")


class StreamPlan:
    __slots__ = (
        "_stream_ids",
        "_required_stream_count",
        "_node_refs",
        "_core_num_suggestions",
    )

    def __init__(
        self,
        stream_ids: Sequence[int],
        required_stream_count: int,
        node_refs: Optional[Iterable[DAGNode]] = None,
        core_num_suggestions: Optional[Mapping[Tuple[int, str], int]] = None,
    ):
        self._stream_ids: List[int] = [int(value) for value in stream_ids]
        self._required_stream_count = int(required_stream_count)
        self._node_refs = tuple(node_refs) if node_refs is not None else tuple()
        # A sparse table: only nodes with an actual suggestion appear here, so
        # a graph without any suggestion costs nothing.  The plan owns the
        # result and never recomputes it from the node costs, which the caller
        # may keep changing after planning returned.
        self._core_num_suggestions = (
            dict(core_num_suggestions) if core_num_suggestions else {}
        )

    def _index_of(self, node: DAGNode) -> int:
        if not isinstance(node, DAGNode):
            raise TypeError("node must be a DAGNode")
        index = node._index
        if self._node_refs and (
            index < 0
            or index >= len(self._node_refs)
            or self._node_refs[index] is not node
        ):
            raise ValueError(
                "node does not belong to the graph used to create this plan"
            )
        if index < 0 or index >= len(self._stream_ids):
            if not self._stream_ids:
                raise ValueError(f"node index {index} is out of range for empty plan")
            raise ValueError(
                f"node index {index} is out of range [0, {len(self._stream_ids)}) "
                "for this plan"
            )
        return index

    def get_stream_id(self, node: DAGNode) -> int:
        return self._stream_ids[self._index_of(node)]

    def get_required_stream_count(self) -> int:
        return self._required_stream_count

    def get_core_num(self, node: DAGNode, kernel_type: str) -> int:
        """Return the suggested core count for *node* on *kernel_type*.

        ``kernel_type`` is the queried resource, either ``"AIC"`` or
        ``"AIV"``.  ``MIX_AIC``/``MIX_AIV`` describe the node itself and are
        not query keys: a mixed node is asked about twice, once per engine.

        A return value of 0 means there is no suggestion and the caller keeps
        its current configuration.  This release never produces a positive
        suggestion.
        """

        index = self._index_of(node)
        if not isinstance(kernel_type, str):
            raise TypeError("kernel_type must be a string")
        if kernel_type not in _KERNEL_TYPES:
            raise ValueError('kernel_type must be "AIC" or "AIV"')
        return self._core_num_suggestions.get((index, kernel_type), 0)


__all__ = ["StreamPlan"]
