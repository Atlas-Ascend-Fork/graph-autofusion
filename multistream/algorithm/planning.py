# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Public transaction boundary for stream planning.

The package plans resources: it decides which stream each node should run
on and returns that decision as a ``StreamPlan``.  It never creates a stream
or reserves any device resource; the host does that from the plan.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..common.errors import _check_internal
from ..common.log import logger
from ..common.runtime_platform import _query_core_limits
from ..minidag.graph import DAGGraph
from ..minidag.plan import StreamPlan
from ..minidag.types import MAX_STREAM_NUM, INVALID_STREAM_ID, MergeStrategy
from .stream_allocator import _StreamAllocConfig, _is_dag, by_path_cover


def _normalize_strategy(strategy) -> MergeStrategy:
    if isinstance(strategy, MergeStrategy):
        return strategy
    if isinstance(strategy, str):
        normalized = strategy.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "loadbalance": MergeStrategy.LOAD_BALANCE,
            "load_balance": MergeStrategy.LOAD_BALANCE,
            "mainstream": MergeStrategy.MAIN_STREAM,
            "main_stream": MergeStrategy.MAIN_STREAM,
        }
        if normalized in aliases:
            return aliases[normalized]
    raise ValueError("merge_strategy must be a MergeStrategy value")


def _validate_inputs(
    graph, max_stream_num, base_stream_id, merge_strategy
) -> Tuple[int, int, MergeStrategy]:
    if not isinstance(graph, DAGGraph):
        raise TypeError("graph must be a DAGGraph")
    if max_stream_num is None:
        max_num = 8
    else:
        if isinstance(max_stream_num, bool) or not isinstance(max_stream_num, int):
            raise TypeError("max_stream_num must be an integer or None")
        max_num = max_stream_num
    if max_num < 1 or max_num > MAX_STREAM_NUM:
        raise ValueError(f"max_stream_num must be in [1, {MAX_STREAM_NUM}]")
    if isinstance(base_stream_id, bool) or not isinstance(base_stream_id, int):
        raise TypeError("base_stream_id must be an integer")
    if base_stream_id < 0:
        raise ValueError("base_stream_id must be non-negative")
    strategy = _normalize_strategy(merge_strategy)
    return max_num, base_stream_id, strategy


def plan(
    graph: DAGGraph,
    *,
    max_stream_num: Optional[int] = None,
    base_stream_id: int = 0,
    merge_strategy: MergeStrategy = MergeStrategy.LOAD_BALANCE,
) -> StreamPlan:
    """Plan which stream each graph node runs on and return an immutable plan.

    All work is performed before constructing the result, so an exception never
    exposes a partially assigned plan or mutates the input graph.
    """

    try:
        max_num, base_id, requested = _validate_inputs(
            graph, max_stream_num, base_stream_id, merge_strategy
        )
        nodes = graph.get_nodes()
        if not _is_dag(graph):
            logger.error("graph contains a cycle; stream allocation requires a DAG")
            raise RuntimeError(
                "graph contains a cycle; stream allocation requires a DAG"
            )
        # The hardware limits belong to this plan only.  Query them
        # before any cost work so a Runtime that is not ready fails the whole
        # call instead of half of it.
        core_limits = _query_core_limits()
        costs = [node.get_node_cost() for node in nodes]
        config = _StreamAllocConfig(
            max_stream_id=max_num - 1,
            required_streams=0,
            base_stream_id=base_id,
            merge_strategy=requested,
            core_limits=core_limits,
        )
        stream_ids = by_path_cover(graph, costs, config)
        try:
            stream_id_count = len(stream_ids)
        except (TypeError, AttributeError):
            _check_internal(False, "allocator returned a non-sized stream-id result")
            stream_id_count = -1  # unreachable; keeps type checkers satisfied
        if stream_id_count != len(nodes):
            _check_internal(False, "allocator returned wrong stream-id count")

        # Completeness and stream-limit checks are deliberately performed only
        # at this final boundary.  Zero-cost non-serial nodes are excluded by
        # contract and retain INVALID_STREAM_ID; serial nodes are always given
        # an appended stream even when their cost is zero.  Treat the allocator
        # as an untrusted internal component here: a malformed mapping must not
        # become a seemingly successful plan with a stream id that the GE
        # context cannot represent.
        nonserial_streams = set()
        for node, stream_id, cost in zip(nodes, stream_ids, costs):
            if isinstance(stream_id, bool) or not isinstance(stream_id, int):
                _check_internal(
                    False,
                    f"node {node._name} has a non-integer stream id",
                )
            if node.get_stream_label() or abs(float(cost.execution_time)) > 1e-6:
                if stream_id == INVALID_STREAM_ID:
                    _check_internal(
                        False,
                        f"node {node._name} has no stream id",
                    )
                if node.get_stream_label():
                    if stream_id < base_id:
                        _check_internal(
                            False,
                            f"node {node._name} has an out-of-range stream id",
                        )
                else:
                    # Non-serial ids are the compact physical mapping and must
                    # stay within the requested [base, base + max_num) range.
                    if stream_id < base_id or stream_id >= base_id + max_num:
                        _check_internal(
                            False,
                            f"node {node._name} has an out-of-range stream id",
                        )
                    nonserial_streams.add(stream_id - base_id)
            elif stream_id != INVALID_STREAM_ID:
                _check_internal(
                    False,
                    f"excluded node {node._name} unexpectedly has a stream id",
                )
        if len(nonserial_streams) > max_num:
            _check_internal(False, "physical stream limit exceeded")
        _check_internal(
            isinstance(config.required_streams, int)
            and not isinstance(config.required_streams, bool),
            "required stream count is not an integer",
        )
        _check_internal(config.required_streams >= 0, "negative required stream count")
        # Core-count suggestions arrive with the second-phase algorithm.  Say
        # so once per plan so a caller can tell "not implemented yet"
        # from "the algorithm found no suggestion".
        logger.info(
            "core number suggestion is not supported in this release; "
            "StreamPlan.get_core_num always returns 0."
        )
        return StreamPlan(stream_ids, config.required_streams, nodes)
    except Exception:
        logger.error("stream allocation failed", exc_info=True)
        raise


__all__ = ["plan"]
