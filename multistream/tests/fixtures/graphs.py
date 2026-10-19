# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Public-API graph fixtures for multistream tests."""

import random

import pytest

import multistream
from utils.graphs import _chain


@pytest.fixture
def linear_graph():
    return lambda n=4: _chain(n)


@pytest.fixture
def diamond_graph():
    graph = multistream.DAGGraph("diamond")
    src = graph.add_node("src", "Op")
    a1 = graph.add_node("a1", "Op")
    a2 = graph.add_node("a2", "Op")
    b1 = graph.add_node("b1", "Op")
    b2 = graph.add_node("b2", "Op")
    join = graph.add_node("join", "Op")
    graph.add_edge(src, 0, a1, 0)
    graph.add_edge(a1, 0, a2, 0)
    graph.add_edge(src, 1, b1, 0)
    graph.add_edge(b1, 0, b2, 0)
    graph.add_edge(a2, 0, join, 0)
    graph.add_edge(b2, 0, join, 1)
    return graph


@pytest.fixture
def multi_branch_graph():
    def build(width=3, depth=3):
        graph = multistream.DAGGraph(f"branches_{width}x{depth}")
        root = graph.add_node("root", "Op")
        branches = []
        for branch in range(width):
            previous = root
            branch_nodes = []
            for level in range(depth):
                node = graph.add_node(f"b{branch}_{level}", "Op")
                graph.add_edge(previous, 0, node, 0)
                previous = node
                branch_nodes.append(node)
            branches.append(branch_nodes)
        return graph

    return build


@pytest.fixture
def deep_chain_graph():
    return lambda n=100: _chain(n, name=f"deep_{n}")


@pytest.fixture
def large_graph():
    def build(n=100):
        graph = multistream.DAGGraph(f"large_{n}")
        nodes = [graph.add_node(f"n{i}", "Op") for i in range(n)]
        rng = random.Random(17)
        for dst in range(1, n):
            # A bounded number of deterministic predecessors gives a useful
            # mix of independent and dependent units without cycles.
            candidates = list(range(dst))
            rng.shuffle(candidates)
            for src in candidates[: min(2, len(candidates))]:
                graph.add_edge(nodes[src], 0, nodes[dst], 0)
        return graph

    return build


@pytest.fixture
def serial_labeled_graph():
    graph = multistream.DAGGraph("serial")
    a = graph.add_node("a", "Op")
    s1 = graph.add_node("s1", "Op")
    b = graph.add_node("b", "Op")
    s2 = graph.add_node("s2", "Op")
    c = graph.add_node("c", "Op")
    s1.set_stream_label("serial-a")
    s2.set_stream_label("serial-b")
    graph.add_edge(a, 0, s1, 0)
    graph.add_edge(s1, 0, b, 0)
    graph.add_edge(b, 0, s2, 0)
    graph.add_edge(s2, 0, c, 0)
    return graph
