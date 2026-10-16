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

import os
from typing import Optional, Tuple

from ..common.errors import _check_internal
from ..common.log import logger
from ..common.runtime_platform import _query_core_limits
from ..minidag.graph import DAGGraph
from ..minidag.plan import StreamPlan
from ..minidag.types import MAX_STREAM_NUM, INVALID_STREAM_ID, MergeStrategy, NodeCost
from .profiling import match_profile, parse
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
    graph, max_stream_num, base_stream_id, merge_strategy, profiling_path
) -> Tuple[int, int, MergeStrategy, Optional[str]]:
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
    if profiling_path is not None and not isinstance(
        profiling_path, (str, os.PathLike)
    ):
        raise TypeError("profiling_path must be a string, PathLike, or None")
    if profiling_path is None:
        path = None
    else:
        try:
            path_value = os.fspath(profiling_path)
        except TypeError as exc:
            raise TypeError(
                "profiling_path must be a string, PathLike, or None"
            ) from exc
        if isinstance(path_value, bytes):
            raise TypeError("profiling_path must be a text path")
        path = None if path_value == "" else path_value
    return max_num, base_stream_id, strategy, path


def _build_effective_costs(
    graph: DAGGraph, profiling_path: Optional[str] = None, *, return_match: bool = False
):
    """Build a private cost array and report which nodes a sample matched.

    The returned index set only records the name matches.  Whether those
    matches select the weighted merger is decided later by the allocator,
    which is the first component that knows which nodes take part in
    ordinary path merging.
    """

    costs = [node.get_node_cost() for node in graph.get_nodes()]
    if not profiling_path:
        return (costs, frozenset()) if return_match else costs
    profiles = parse(profiling_path)
    matched_indices = set()
    for node, cost in zip(graph.get_nodes(), costs):
        profile = match_profile(profiles, node._name)
        if profile is None:
            continue
        matched_indices.add(node._index)
        logger.info(
            "Use profiling cost for %s: time=%f, cube=%d, vec=%d",
            node._name,
            profile.execution_time,
            profile.cube_block_num,
            profile.vec_block_num,
        )
        # 0.0 is an explicit exclusion marker set by the GE adapter.  It must
        # win over a same-name profiling record *for the execution time*.
        # The C++ adapter still copies the profiling block counts before it
        # applies that marker, and the weighted merger consumes those counts
        # independently.  Preserve that ordering here: only the duration
        # assignment is conditional, while cube/vector counts are always
        # taken from the matched profile.
        # NodeCost is immutable.  Profiling applies to the effective copy
        # used by this plan only; the source graph keeps its original cost so
        # repeated plans cannot leak profiling state.  The user's core-count
        # limits are configuration, not measurements, and a sample never
        # overwrites them.
        costs[node._index] = NodeCost(
            execution_time=(
                cost.execution_time
                if float(cost.execution_time) == 0.0
                else profile.execution_time
            ),
            memory_usage=cost.memory_usage,
            bandwidth_usage=cost.bandwidth_usage,
            cube_block_num=profile.cube_block_num,
            vec_block_num=profile.vec_block_num,
            cube_core_num_limit=cost.cube_core_num_limit,
            vec_core_num_limit=cost.vec_core_num_limit,
        )
    return (costs, frozenset(matched_indices)) if return_match else costs


def _log_routing(
    graph_name: str, requested: MergeStrategy, config: _StreamAllocConfig
) -> None:
    if config.use_profiling_weights:
        implementation = "WeightedLoadBalance"
        reason = "profiling samples matched participating nodes"
    else:
        implementation = requested.value
        if not config.profiling_hit_indices:
            reason = "no profiling sample matched a participating node"
        else:
            reason = "profiling samples matched only excluded or serial nodes"
    logger.info(
        "graph %s requested strategy=%s, used implementation=%s, reason=%s.",
        graph_name,
        requested.value,
        implementation,
        reason,
    )


def plan(
    graph: DAGGraph,
    *,
    max_stream_num: Optional[int] = None,
    base_stream_id: int = 0,
    merge_strategy: MergeStrategy = MergeStrategy.LOAD_BALANCE,
    profiling_path: Optional[str] = None,
) -> StreamPlan:
    """Plan which stream each graph node runs on and return an immutable plan.

    Valid profiling samples on nodes participating in ordinary path merging
    select weighted load balancing, overriding either requested base strategy.
    Without such samples, the requested strategy is used.

    All work is performed before constructing the result, so an exception never
    exposes a partially assigned plan or mutates the input graph.
    """

    try:
        max_num, base_id, requested, profile_path = _validate_inputs(
            graph, max_stream_num, base_stream_id, merge_strategy, profiling_path
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
        costs, matched_indices = _build_effective_costs(
            graph, profile_path, return_match=True
        )
        config = _StreamAllocConfig(
            max_stream_id=max_num - 1,
            required_streams=0,
            base_stream_id=base_id,
            merge_strategy=requested,
            profiling_hit_indices=matched_indices,
            core_limits=core_limits,
        )
        stream_ids = by_path_cover(graph, costs, config)
        _log_routing(graph.get_name(), requested, config)
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
