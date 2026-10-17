# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Graph builders shared by multistream tests."""

from __future__ import annotations

import random

import multistream


def _chain(n: int, *, name: str = "chain") -> multistream.DAGGraph:
    graph = multistream.DAGGraph(name)
    nodes = [graph.add_node(f"n{i}", "Op") for i in range(n)]
    for src, dst in zip(nodes, nodes[1:]):
        graph.add_edge(src, 0, dst, 0)
    return graph


def build_linear(n: int = 4):
    """Return ``(graph, nodes)`` for direct algorithm tests."""
    graph = multistream.DAGGraph(f"linear_{n}")
    nodes = [graph.add_node(f"n{i}", "Op") for i in range(n)]
    for src, dst in zip(nodes, nodes[1:]):
        graph.add_edge(src, 0, dst, 0)
    return graph, nodes


def build_diamond():
    graph = multistream.DAGGraph("diamond")
    names = ("src", "a1", "a2", "b1", "b2", "join")
    nodes = [graph.add_node(name, "Op") for name in names]
    for src, dst, src_port, dst_port in (
        (0, 1, 0, 0),
        (1, 2, 0, 0),
        (0, 3, 1, 0),
        (3, 4, 0, 0),
        (2, 5, 0, 0),
        (4, 5, 0, 1),
    ):
        graph.add_edge(nodes[src], src_port, nodes[dst], dst_port)
    return graph, nodes


def build_multi_branch(width: int = 3, depth: int = 2):
    graph = multistream.DAGGraph(f"branches_{width}x{depth}")
    root = graph.add_node("root", "Op")
    nodes = [root]
    for branch in range(width):
        previous = root
        for level in range(depth):
            node = graph.add_node(f"b{branch}_{level}", "Op")
            graph.add_edge(previous, 0, node, 0)
            previous = node
            nodes.append(node)
    return graph, nodes


def build_large_random(node_count: int = 100, seed: int = 17):
    """Return a reproducible sparse DAG for scale/determinism checks."""
    graph = multistream.DAGGraph(f"large_{node_count}_{seed}")
    nodes = [graph.add_node(f"n{i}", "Op") for i in range(node_count)]
    rng = random.Random(seed)
    for dst in range(1, node_count):
        candidates = list(range(dst))
        rng.shuffle(candidates)
        for src in candidates[: min(2, len(candidates))]:
            graph.add_edge(nodes[src], 0, nodes[dst], 0)
    return graph, nodes


def build_serial_labeled():
    graph = multistream.DAGGraph("serial")
    nodes = [graph.add_node(name, "Op") for name in ("a", "s1", "b", "s2", "c")]
    nodes[1].set_stream_label("serial-a")
    nodes[3].set_stream_label("serial-b")
    for src, dst in zip(nodes, nodes[1:]):
        graph.add_edge(src, 0, dst, 0)
    return graph, nodes
