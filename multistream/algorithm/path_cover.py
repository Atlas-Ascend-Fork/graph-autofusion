# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Minimum path cover helper (Hopcroft--Karp), translated from GE."""

from __future__ import annotations

from collections import deque
from typing import List


INF = 0x3F3F3F3F
MAX_NODE = 1_000_000
SOURCE_NODE_ID = 0
NODE_SPLIT_FACTOR = 2
SINK_OFFSET = 1
EDGE_PAIR_SIZE = 2
FORWARD_EDGE_OFFSET = 2
BACKWARD_EDGE_OFFSET = 1


class _Edge:
    __slots__ = ("frm", "to", "cap", "flow")

    def __init__(self, frm: int, to: int, cap: int, flow: int = 0) -> None:
        self.frm = frm
        self.to = to
        self.cap = cap
        self.flow = flow


class HopcroftKarp:
    """The C++ class is a unit-capacity max-flow implementation.

    Despite its historical name, it is not the textbook bipartite matching
    class; preserving its graph numbering and recursive DFS is important for
    route determinism and for parity with the old allocator.
    """

    __slots__ = (
        "node_num_",
        "total_node_count_",
        "edge_num_",
        "source_node_",
        "sink_node_",
        "edges_",
        "adjacency_list_",
        "visited_",
        "distance_",
        "current_edge_idx_",
        "routes_",
    )

    def __init__(self, node_count: int) -> None:
        if (
            isinstance(node_count, bool)
            or not isinstance(node_count, int)
            or node_count < 0
        ):
            raise ValueError("node count must be a non-negative integer")
        self.node_num_ = node_count
        self.total_node_count_ = node_count * NODE_SPLIT_FACTOR + EDGE_PAIR_SIZE
        self.source_node_ = SOURCE_NODE_ID
        self.sink_node_ = node_count * NODE_SPLIT_FACTOR + SINK_OFFSET
        self.edge_num_ = 0
        self.edges_: List[_Edge] = []
        self.adjacency_list_: List[List[int]] = [
            [] for _ in range(self.total_node_count_ + 1)
        ]
        self.visited_ = [False] * (self.total_node_count_ + 1)
        self.distance_ = [0] * (self.total_node_count_ + 1)
        self.current_edge_idx_ = [0] * (self.total_node_count_ + 1)
        self.routes_: List[List[int]] = []
        for i in range(1, node_count + 1):
            self._add_edge(self.source_node_, i, 1)
            self._add_edge(i + node_count, self.sink_node_, 1)

    def _add_edge(self, frm: int, to: int, cap: int) -> None:
        if not (0 <= frm < len(self.adjacency_list_)) or not (
            0 <= to < len(self.adjacency_list_)
        ):
            raise ValueError("Hopcroft-Karp edge endpoint out of range")
        edge_index = len(self.edges_)
        self.edges_.append(_Edge(frm, to, cap, 0))
        self.edges_.append(_Edge(to, frm, 0, 0))
        self.edge_num_ = len(self.edges_)
        self.adjacency_list_[frm].append(edge_index)
        self.adjacency_list_[to].append(edge_index + 1)

    def _bfs(self) -> bool:
        self.visited_[:] = [False] * len(self.visited_)
        queue = deque([self.source_node_])
        self.visited_[self.source_node_] = True
        self.distance_[self.source_node_] = 0
        while queue:
            x = queue.popleft()
            for edge_index in self.adjacency_list_[x]:
                edge = self.edges_[edge_index]
                if not self.visited_[edge.to] and edge.cap > edge.flow:
                    self.visited_[edge.to] = True
                    self.distance_[edge.to] = self.distance_[x] + 1
                    queue.append(edge.to)
        return self.visited_[self.sink_node_]

    def _dfs(self, current_node: int, remaining_flow: int) -> int:
        if current_node == self.sink_node_ or remaining_flow == 0:
            return remaining_flow
        flow = 0
        i = self.current_edge_idx_[current_node]
        while i < len(self.adjacency_list_[current_node]):
            self.current_edge_idx_[current_node] = i
            edge_index = self.adjacency_list_[current_node][i]
            edge = self.edges_[edge_index]
            if self.distance_[current_node] + 1 == self.distance_[edge.to]:
                pushed = self._dfs(edge.to, min(remaining_flow, edge.cap - edge.flow))
                if pushed > 0:
                    edge.flow += pushed
                    self.edges_[edge_index ^ 1].flow -= pushed
                    flow += pushed
                    remaining_flow -= pushed
                    if remaining_flow == 0:
                        break
            i += 1
            self.current_edge_idx_[current_node] = i
        return flow

    def min_stream_cover(self) -> int:
        flow = 0
        while self._bfs():
            self.current_edge_idx_[:] = [0] * len(self.current_edge_idx_)
            flow += self._dfs(self.source_node_, INF)

        # ``to`` and ``from`` hold the matched left->right links.  Edges are
        # inspected in insertion order, exactly as in the C++ implementation.
        to = [0] * (self.total_node_count_ + 1)
        from_ = [0] * (self.total_node_count_ + 1)
        for i in range(1, self.node_num_ + 1):
            for edge_index in self.adjacency_list_[i]:
                edge = self.edges_[edge_index]
                if edge.flow == 1:
                    nxt = edge.to - self.node_num_
                    if 1 <= nxt <= self.node_num_:
                        to[i] = nxt
                        from_[nxt] = i
                    break

        self.routes_ = []
        visited = [False] * (self.total_node_count_ + 1)
        for i in range(1, self.node_num_ + 1):
            if from_[i] == 0 and not visited[i]:
                x = i
                route: List[int] = []
                while x:
                    if visited[x]:
                        break
                    route.append(x)
                    visited[x] = True
                    x = to[x]
                self.routes_.append(route)
        for i in range(1, self.node_num_ + 1):
            if not visited[i]:
                self.routes_.append([i])
                visited[i] = True
        return self.node_num_ - flow

    def get_routes(self) -> List[List[int]]:
        return [list(route) for route in self.routes_]


__all__ = ["HopcroftKarp"]
