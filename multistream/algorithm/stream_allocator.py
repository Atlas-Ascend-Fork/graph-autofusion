# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Path-cover stream allocator and serial-label splitting."""

from __future__ import annotations

import heapq
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..common.log import logger
from ..minidag.node import DAGNode
from ..minidag.types import INVALID_STREAM_ID, MergeStrategy, NodeCost, _cost_of
from .merger import StreamMergeOptions, _Merger
from .path_cover import HopcroftKarp


DEFAULT_MAX_PHYSICAL_STREAMS = 8


class _StreamAllocConfig:
    """Internal counterpart of ``StreamAllocConfig``.

    ``max_stream_id`` follows the C++ convention (inclusive).  The public
    ``plan(max_stream_num=...)`` converts a count to this field before
    calling the allocator.
    """

    __slots__ = (
        "max_stream_id",
        "required_streams",
        "base_stream_id",
        "merge_strategy",
        "core_limits",
    )

    def __init__(
        self,
        max_stream_id: int = -1,
        required_streams: int = 0,
        base_stream_id: int = 0,
        merge_strategy: MergeStrategy = MergeStrategy.LOAD_BALANCE,
        core_limits: Optional[object] = None,
    ) -> None:
        self.max_stream_id = max_stream_id
        self.required_streams = required_streams
        self.base_stream_id = base_stream_id
        self.merge_strategy = merge_strategy
        self.core_limits = core_limits


class _SerialGraphSplit:
    __slots__ = (
        "residual_graph",
        "serial_groups",
        "serial_label_order",
        "residual_to_original",
        "excluded_nodes",
    )

    def __init__(
        self,
        residual_graph: object,
        serial_groups: Dict[str, List[DAGNode]],
        serial_label_order: List[str],
        residual_to_original: List[DAGNode],
        excluded_nodes: List[DAGNode],
    ) -> None:
        self.residual_graph = residual_graph
        self.serial_groups = serial_groups
        self.serial_label_order = serial_label_order
        self.residual_to_original = residual_to_original
        self.excluded_nodes = excluded_nodes


def _is_serial_node(node: DAGNode) -> bool:
    return bool(node.get_stream_label())


def _is_zero_cost_node(node: DAGNode, cost: NodeCost) -> bool:
    # Keep the literal and comparison from the C++ implementation.  The
    # execution time has already been rounded to f32 by the boundary layer.
    return abs(float(cost.execution_time)) <= 1e-6


def _is_dag(graph) -> bool:
    """Return whether all graph nodes participate in an acyclic graph."""

    nodes = graph.get_nodes()
    index_by_identity = {id(node): idx for idx, node in enumerate(nodes)}
    indegree = [0] * len(nodes)
    succs = [[] for _ in nodes]
    for src in nodes:
        src_idx = index_by_identity[id(src)]
        for edge in src._get_output_edges():
            dst_idx = index_by_identity.get(id(edge._dst_node))
            if dst_idx is None:
                # An edge crossing graph ownership is malformed.  Treat it as
                # a failed DAG rather than silently dropping it.
                return False
            succs[src_idx].append(dst_idx)
            indegree[dst_idx] += 1
    ready = list(idx for idx, degree in enumerate(indegree) if degree == 0)
    heapq.heapify(ready)
    visited = 0
    while ready:
        idx = heapq.heappop(ready)
        visited += 1
        for dst in succs[idx]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                heapq.heappush(ready, dst)
    return visited == len(nodes)


def _sort_by_topo_id(nodes: List[DAGNode]) -> None:
    nodes.sort(key=lambda node: node._index)


def _build_residual_edges(
    original_graph, split: _SerialGraphSplit, zero_by_identity: Mapping[int, bool]
) -> None:
    residual_graph = split.residual_graph
    added_edges: "set[Tuple[str, str]]" = set()

    def add_residual_edge(src_node: DAGNode, dst_node: DAGNode) -> None:
        if src_node._name == dst_node._name:
            return
        key = (src_node._name, dst_node._name)
        if key in added_edges:
            return
        added_edges.add(key)
        residual_src = residual_graph.find_node(src_node._name)
        residual_dst = residual_graph.find_node(dst_node._name)
        if residual_src is None:
            logger.error(
                "Add residual edge failed: %s -> %s src not found in residual graph.",
                src_node._name,
                dst_node._name,
            )
            raise RuntimeError("residual edge endpoint is missing")
        if residual_dst is None:
            logger.error(
                "Add residual edge failed: %s -> %s dst not found in residual graph.",
                src_node._name,
                dst_node._name,
            )
            raise RuntimeError("residual edge endpoint is missing")
        residual_graph.add_edge(residual_src, -1, residual_dst, -1)

    def add_edges_from_node(src_node: DAGNode) -> None:
        pending: List[DAGNode] = []
        visited_serial: "set[str]" = set()

        def handle(dst_node: DAGNode) -> None:
            if _is_serial_node(dst_node) or zero_by_identity.get(id(dst_node), False):
                if dst_node._name not in visited_serial:
                    visited_serial.add(dst_node._name)
                    pending.append(dst_node)
            else:
                add_residual_edge(src_node, dst_node)

        for edge in src_node._get_output_edges():
            handle(edge._dst_node)
        cursor = 0
        while cursor < len(pending):
            serial_node = pending[cursor]
            cursor += 1
            for edge in serial_node._get_output_edges():
                handle(edge._dst_node)

    for node in original_graph.get_nodes():
        if _is_serial_node(node) or zero_by_identity.get(id(node), False):
            continue
        add_edges_from_node(node)


def _split_serial_graph(graph, costs: Sequence[NodeCost]) -> _SerialGraphSplit:
    from ..minidag.graph import DAGGraph

    residual_graph = DAGGraph(graph.get_name() + "_residual")
    serial_groups: Dict[str, List[DAGNode]] = {}
    serial_label_order: List[str] = []
    seen_labels = set()
    residual_to_original: List[DAGNode] = []
    excluded_nodes: List[DAGNode] = []
    zero_by_identity = {
        id(node): _is_zero_cost_node(node, _cost_of(node, costs))
        for node in graph.get_nodes()
    }

    nodes = graph.get_nodes()
    _sort_by_topo_id(nodes)
    for node in nodes:
        if _is_serial_node(node):
            label = node.get_stream_label()
            serial_groups.setdefault(label, []).append(node)
            if label not in seen_labels:
                seen_labels.add(label)
                serial_label_order.append(label)
            continue
        if zero_by_identity[id(node)]:
            excluded_nodes.append(node)
            continue
        residual_node = residual_graph.add_node(node._name, node._type)
        if residual_node is None:
            logger.error(
                "Add residual node failed: %s, type: %s.", node._name, node._type
            )
            raise RuntimeError("duplicate residual node")
        residual_node.set_node_cost(_cost_of(node, costs))
        residual_node.set_core_type(node.get_core_type())
        residual_to_original.append(node)
    logger.info(
        "serial node num: %d, non-weight node num: %d",
        sum(len(group) for group in serial_groups.values()),
        len(excluded_nodes),
    )

    for group in serial_groups.values():
        _sort_by_topo_id(group)
    _build_residual_edges(
        graph,
        _SerialGraphSplit(
            residual_graph,
            serial_groups,
            serial_label_order,
            residual_to_original,
            excluded_nodes,
        ),
        zero_by_identity,
    )
    return _SerialGraphSplit(
        residual_graph,
        serial_groups,
        serial_label_order,
        residual_to_original,
        excluded_nodes,
    )


def _assign_serial_stream_ids(
    split: _SerialGraphSplit, first_stream_id: int, stream_ids: Dict[int, int]
) -> None:
    for label_index, label in enumerate(split.serial_label_order):
        stream_id = first_stream_id + label_index
        group = split.serial_groups.get(label, ())
        for node in group:
            stream_ids[node._index] = stream_id
        logger.debug(
            "serial label %s set as stream %d, nodes: %s",
            label,
            stream_id,
            "".join(f"{node._index} " for node in group),
        )


def _path_cover_routes(graph) -> List[List[int]]:
    nodes = graph.get_nodes()
    node_count = len(nodes)
    if node_count == 0:
        return []
    hk = HopcroftKarp(node_count)
    # Caller edge iteration is insertion order, the deterministic replacement
    # for the old shared_ptr-address ordered map.
    for src_index, src in enumerate(nodes):
        for edge in src._get_output_edges():
            dst = edge._dst_node
            dst_index = dst._index
            if 0 <= dst_index < node_count and nodes[dst_index] is dst:
                hk._add_edge(src_index + 1, dst_index + 1 + node_count, 1)
    hk.min_stream_cover()
    return [[node_id - 1 for node_id in route] for route in hk.get_routes()]


def _by_path_cover_core(
    graph,
    config: _StreamAllocConfig,
    residual_to_original: Optional[Sequence[DAGNode]] = None,
) -> Tuple[List[int], int, List[DAGNode]]:
    logger.info("rest dag node num: %d", len(graph.get_nodes()))
    routes = _path_cover_routes(graph)
    if not routes:
        logger.info("Logical stream num:%d, merged physical stream num:%d", 0, 0)
        return [], 0, []
    physical_limit = (
        config.max_stream_id + 1
        if config.max_stream_id >= 0
        else DEFAULT_MAX_PHYSICAL_STREAMS
    )
    residual_nodes = graph.get_nodes()
    strategy = config.merge_strategy
    strategy_value = getattr(strategy, "value", strategy)
    strategy_text = str(strategy_value).strip().lower().replace("-", "_")
    if strategy_text in ("loadbalance", "load_balance", "0"):
        strategy_value = "load_balance"
    else:
        strategy_value = "main_stream"
    options = StreamMergeOptions(
        strategy=strategy_value, physical_stream_limit=physical_limit
    )
    mapping = _Merger(options).merge(graph, routes)
    if len(mapping) != len(routes):
        raise RuntimeError(
            "multistream internal error: merger returned an invalid mapping"
        )
    ids_by_residual_index: Dict[int, int] = {}
    for route_index, route in enumerate(routes):
        stream_id = config.base_stream_id + mapping[route_index]
        for residual_index in route:
            if residual_index < 0 or residual_index >= len(residual_nodes):
                raise RuntimeError(
                    "multistream internal error: path route index is invalid"
                )
            ids_by_residual_index[residual_index] = stream_id
        # The residual graph renumbers its nodes densely; print original-graph
        # indices so the node list matches the C++ topo ids.
        route_nodes = (
            [residual_to_original[index] for index in route]
            if residual_to_original is not None
            else [residual_nodes[index] for index in route]
        )
        logger.debug(
            "logical stream %d merged to physical stream %d, set as %d, nodes: %s",
            route_index,
            mapping[route_index],
            stream_id,
            "".join(f"{node._index} " for node in route_nodes),
        )
    required = (max(mapping) + 1) if mapping else 0
    logger.info(
        "Logical stream num:%d, merged physical stream num:%d", len(routes), required
    )
    return (
        [
            ids_by_residual_index.get(index, INVALID_STREAM_ID)
            for index, _ in enumerate(residual_nodes)
        ],
        required,
        residual_nodes,
    )


def by_path_cover(
    graph, costs: Sequence[NodeCost], config: _StreamAllocConfig
) -> List[int]:
    """Allocate streams and return an ID list aligned to ``graph`` nodes."""

    logger.info(
        "ByPathCover start: graph=%s, nodes=%d, max_stream_id=%d, base_stream_id=%d",
        graph.get_name(),
        len(graph.get_nodes()),
        config.max_stream_id,
        config.base_stream_id,
    )
    if not _is_dag(graph):
        logger.error("graph contains a cycle; stream allocation requires a DAG")
        raise RuntimeError("graph contains a cycle; stream allocation requires a DAG")
    split = _split_serial_graph(graph, costs)
    residual_ids, required, _ = _by_path_cover_core(
        split.residual_graph, config, split.residual_to_original
    )

    stream_ids: Dict[int, int] = {}
    # residual_ids are aligned to residual graph nodes; map them back to the
    # original graph's dense indices.
    for residual_index, node in enumerate(split.residual_to_original):
        if residual_index < len(residual_ids):
            stream_ids[node._index] = residual_ids[residual_index]
    _assign_serial_stream_ids(split, config.base_stream_id + required, stream_ids)
    config.required_streams = required + len(split.serial_label_order)
    logger.info(
        "ByPathCover with residual graph done: residual_nodes=%d, serial_labels=%d, required_streams=%d",
        len(split.residual_graph.get_nodes()),
        len(split.serial_label_order),
        config.required_streams,
    )

    # Return IDs in original insertion order; excluded zero-cost nodes remain
    # INVALID_STREAM_ID by design.
    return [
        stream_ids.get(node._index, INVALID_STREAM_ID) for node in graph.get_nodes()
    ]


__all__ = ["_StreamAllocConfig", "by_path_cover", "_is_dag", "_split_serial_graph"]
