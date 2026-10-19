# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Logical-stream merger tests, plan §4.6."""

import logging

import pytest

import multistream
from multistream.algorithm.merger import (
    K_MERGE_IMPROVE_EPS,
    StreamMergeOptions,
    _ConcurrentScore,
    _LiteScore,
    _Merger,
    _StreamMergeSolver,
    _compare_double,
    _concurrent_less,
    _lite_less,
    merge,
)


def _graph_from_edges(edge_pairs, node_count=None):
    if node_count is None:
        node_count = max((max(pair) for pair in edge_pairs), default=-1) + 1
    graph = multistream.DAGGraph("merge")
    nodes = [graph.add_node(f"n{i}", "Op") for i in range(node_count)]
    for src, dst in edge_pairs:
        graph.add_edge(nodes[src], 0, nodes[dst], 0)
    return graph


def _chain(n=4):
    return _graph_from_edges([(i, i + 1) for i in range(n - 1)], n)


def _is_valid_mapping(mapping, routes, limit):
    return len(mapping) == len(routes) and all(0 <= value < limit for value in mapping)


def test_load_balance_spreads_diamond_while_main_stream_collapses(caplog):
    graph = _graph_from_edges([(0, 1), (1, 2), (0, 3), (3, 4), (2, 5), (4, 5)], 6)
    routes = [[0, 1, 2], [3, 4, 5]]
    caplog.set_level(logging.INFO, logger="multistream")
    load = merge(
        graph,
        routes,
        StreamMergeOptions(strategy="LoadBalance", physical_stream_limit=2),
    )
    main = merge(
        graph,
        routes,
        StreamMergeOptions(strategy="MainStream", physical_stream_limit=2),
    )
    assert load == [0, 1]
    assert main == [0, 0]
    assert any("kLoadBalance" in record.getMessage() for record in caplog.records)
    assert any("kMainStream" in record.getMessage() for record in caplog.records)


def test_empty_single_and_repeated_routes():
    graph = _chain(4)
    assert merge(graph, []) == []
    assert merge(graph, [[0, 1, 2, 3]]) == [0]
    assert merge(graph, [[0, 1], [2, 3]]) == [0, 0]
    # Independent units at the same level are deliberately spread by
    # LoadBalance to expose parallelism.
    independent = _graph_from_edges([], 4)
    assert merge(independent, [[0, 1], [2, 3]]) == [0, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"physical_stream_limit": 0},
        {"window_width": 0},
        {"candidate_limit": 0},
        {"repair_moves": -1},
        {"light_stream_limit": 0, "strategy": "load_balance"},
        {"low_conflict_limit": 0, "strategy": "main_stream"},
        {"main_stream": -1, "strategy": "main_stream"},
    ],
)
def test_invalid_options_raise_before_route_work(kwargs):
    graph = _chain(2)
    routes = [[0, 1]]
    with pytest.raises(ValueError):
        _Merger(StreamMergeOptions(**kwargs)).merge(graph, routes)


def test_mapping_respects_physical_limit_and_candidate_limit():
    graph = _graph_from_edges([], 5)
    routes = [[i] for i in range(5)]
    mapping = merge(
        graph,
        routes,
        StreamMergeOptions(
            strategy="load_balance", physical_stream_limit=2, candidate_limit=1
        ),
    )
    assert _is_valid_mapping(mapping, routes, 2)


def test_malformed_routes_and_cycles_are_rejected_without_mapping():
    graph = _chain(3)
    for routes in (
        [[0, 1], [1, 2]],
        [[0, 1], []],
        [[0, 1], [3]],
        [[0, 1]],
        [[0, 1, 2], [2]],
    ):
        with pytest.raises(ValueError):
            merge(graph, routes)
    cycle = _chain(3)
    nodes = cycle.get_nodes()
    cycle.add_edge(nodes[2], 0, nodes[0], 0)
    with pytest.raises(RuntimeError):
        merge(cycle, [[0, 1, 2]])


def test_merge_failures_log_the_cpp_wording_where_found(caplog, monkeypatch):
    caplog.set_level(logging.ERROR, logger="multistream")

    def first_error(action, exc_type):
        caplog.clear()
        with pytest.raises(exc_type):
            action()
        errors = [
            (record.filename, record.getMessage())
            for record in caplog.records
            if record.levelno == logging.ERROR
        ]
        return errors[0]

    graph = _chain(3)
    for routes, message in (
        ([[0, 1], [1, 2]], "Node n1 appears in more than one logical stream."),
        ([[0, 1], []], "Logical stream 1 should not be empty."),
        ([[0, 1], [3]], "Node index 3 is out of range [0, 3)."),
        ([[0, 1]], "Node n2 is not assigned to any logical stream."),
    ):
        assert first_error(lambda: merge(graph, routes), ValueError) == (
            "merger.py",
            message,
        )
    cycle = _chain(3)
    nodes = cycle.get_nodes()
    cycle.add_edge(nodes[2], 0, nodes[0], 0)
    assert first_error(lambda: merge(cycle, [[0, 1, 2]]), RuntimeError) == (
        "merger.py",
        "Merge only supports DAG graphs.",
    )

    for strategy, name, value, message in (
        (
            "main_stream",
            "physical_stream_limit",
            0,
            "Merge physical stream limit must be greater than 0.",
        ),
        (
            "main_stream",
            "window_width",
            0,
            "Merge window width must be greater than 0.",
        ),
        (
            "main_stream",
            "candidate_limit",
            0,
            "Merge candidate limit must be greater than 0.",
        ),
        (
            "main_stream",
            "repair_moves",
            -1,
            "Merge repair moves must be greater than or equal to 0.",
        ),
        (
            "load_balance",
            "light_stream_limit",
            0,
            "LoadBalance light stream limit must be greater than 0.",
        ),
        (
            "main_stream",
            "low_conflict_limit",
            0,
            "MainStream low conflict limit must be greater than 0.",
        ),
        (
            "main_stream",
            "main_stream",
            -1,
            "MainStream main stream must be greater than or equal to 0.",
        ),
    ):
        options = StreamMergeOptions(strategy=strategy)
        setattr(options, name, value)
        action = lambda: _Merger(options).merge(graph, [[0, 1, 2]])  # noqa: E731
        assert first_error(action, ValueError) == ("merger.py", message)

    solver = _StreamMergeSolver(graph, [[0, 1, 2]], StreamMergeOptions())
    unassigned = ("merger.py", "Try to remove unassigned unit 0.")
    assert (
        first_error(lambda: solver._remove_load(0, [-1], [], []), RuntimeError)
        == unassigned
    )
    assert (
        first_error(lambda: solver._remove_main(0, [-1], []), RuntimeError)
        == unassigned
    )
    assert first_error(lambda: _StreamMergeSolver._compact([-1]), RuntimeError) == (
        "merger.py",
        "There is still an unassigned unit when compacting assignment.",
    )
    for strategy, finder in (
        ("load_balance", "_load_candidates"),
        ("main_stream", "_main_candidates"),
    ):
        monkeypatch.setattr(_StreamMergeSolver, finder, lambda self, *args: [])
        options = StreamMergeOptions(strategy=strategy)
        action = lambda: _Merger(options).merge(graph, [[0, 1, 2]])  # noqa: E731
        assert first_error(action, RuntimeError) == (
            "merger.py",
            "No candidate physical stream for unit 0.",
        )


def test_solver_ordering_and_dependency_metadata():
    graph = _graph_from_edges([(0, 2), (1, 2)], 3)
    solver = _StreamMergeSolver(graph, [[0], [1], [2]], StreamMergeOptions())
    solver._build_node_order()
    solver._build_edges()
    solver._build_levels()
    solver._build_units()
    solver._build_unit_dependency_graph()
    solver._build_units_by_earliest_level()
    assert solver.levels == [0, 0, 1]
    assert solver.unit_pred_edges[2] == {0: 1, 1: 1}
    assert solver.unit_succ_edges[0] == {2: 1}
    assert solver.units_by_earliest_level[0] == [0, 1]


def test_event_local_and_candidate_helpers_account_for_cross_stream_edges():
    graph = _graph_from_edges([(0, 1), (1, 2)], 3)
    solver = _StreamMergeSolver(graph, [[0], [1], [2]], StreamMergeOptions())
    solver._build_node_order()
    solver._build_edges()
    solver._build_levels()
    solver._build_units()
    solver._build_unit_dependency_graph()
    solver._build_units_by_earliest_level()
    assert solver._calc_event_local(1, 0, [0, -1, -1]) == 0
    assert solver._calc_event_local(1, 1, [0, -1, -1]) == 1


def test_repair_and_compact_helpers_are_safe():
    assert _StreamMergeSolver._compact([]) == []
    assert _StreamMergeSolver._compact([4, 2, 4]) == [1, 0, 1]
    with pytest.raises(RuntimeError):
        _StreamMergeSolver._compact([0, -1])
    graph = _chain(3)
    result = merge(graph, [[0, 1, 2]], StreamMergeOptions(repair_moves=2))
    assert result == [0]


def test_score_tie_breakers_and_epsilon():
    assert _compare_double(1.0, 1.0 + K_MERGE_IMPROVE_EPS / 2) == 0
    lhs = _LiteScore(
        total=1, event_local=0, window_balance=2, total_load=3, candidate_stream=1
    )
    rhs = _LiteScore(
        total=1, event_local=0, window_balance=2, total_load=3, candidate_stream=2
    )
    assert _lite_less(lhs, rhs)
    assert not _lite_less(rhs, lhs)
    c1 = _ConcurrentScore(
        total=1,
        event_local=0,
        parallel_conflict=0,
        new_flow_penalty=0,
        main_stream_bonus=1,
    )
    c2 = _ConcurrentScore(
        total=1,
        event_local=0,
        parallel_conflict=0,
        new_flow_penalty=0,
        main_stream_bonus=2,
    )
    assert _concurrent_less(c1, c2)


def test_large_and_deep_graphs_remain_valid(large_graph, deep_chain_graph):
    graph = large_graph(150)
    routes = [[i] for i in range(len(graph.get_nodes()))]
    mapping = merge(graph, routes, StreamMergeOptions(physical_stream_limit=8))
    assert _is_valid_mapping(mapping, routes, 8)
    graph = deep_chain_graph(200)
    assert merge(graph, [list(range(200))]) == [0]
