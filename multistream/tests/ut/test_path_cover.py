# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Minimum path-cover (Hopcroft--Karp) tests, plan §4.4."""

import pytest

import multistream
from multistream.algorithm.path_cover import HopcroftKarp
from multistream.algorithm.stream_allocator import _path_cover_routes


def test_path_cover_count_and_routes_cover_each_node_once(diamond_graph):
    graph = diamond_graph
    routes = _path_cover_routes(graph)
    flattened = [node for route in routes for node in route]
    assert len(routes) == 2
    assert sorted(flattened) == list(range(len(graph.get_nodes())))
    assert len(flattened) == len(set(flattened))
    positions = {
        node: (route_id, pos)
        for route_id, route in enumerate(routes)
        for pos, node in enumerate(route)
    }
    for edge in graph._get_edges():
        # A route edge must preserve graph direction; cross-route edges are
        # allowed and are intentionally not asserted as adjacent.
        src_route, src_pos = positions[edge._src_node._index]
        dst_route, dst_pos = positions[edge._dst_node._index]
        if src_route == dst_route:
            assert src_pos < dst_pos


def test_single_node_and_empty_graph():
    graph = multistream.DAGGraph("single")
    graph.add_node("only", "Op")
    assert _path_cover_routes(graph) == [[0]]
    assert _path_cover_routes(multistream.DAGGraph("empty")) == []


@pytest.mark.parametrize("node_count", [-1, True, 1.0, "1"])
def test_hopcroft_karp_rejects_invalid_node_count(node_count):
    with pytest.raises(ValueError):
        HopcroftKarp(node_count)


def test_hopcroft_karp_rejects_out_of_range_edges():
    matcher = HopcroftKarp(2)
    for edge in [(-1, 1), (0, 100), (100, 0)]:
        with pytest.raises(ValueError):
            matcher._add_edge(*edge, 1)


def test_routes_are_deterministic_across_repeated_runs(diamond_graph):
    expected = _path_cover_routes(diamond_graph)
    assert all(_path_cover_routes(diamond_graph) == expected for _ in range(5))


def test_path_cover_route_order_follows_node_insertion_order():
    graph = multistream.DAGGraph("order")
    nodes = [graph.add_node(name, "Op") for name in ("z", "a", "m")]
    graph.add_edge(nodes[0], 0, nodes[1], 0)
    graph.add_edge(nodes[1], 0, nodes[2], 0)
    assert _path_cover_routes(graph) == [[0, 1, 2]]
