# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Weighted logical-stream merger tests, plan §4.7.

The weighted solver is intentionally tested through both its public merger
entry point and a small set of private helpers.  The latter cover simulation,
candidate ordering and resource-profile branches that are otherwise difficult
to reach with a single end-to-end graph.
"""

import pytest

import multistream
from multistream.algorithm.weighted_merger import (
    K_IMPROVE_EPS,
    WeightedStreamMergeOptions,
    _LiteScore,
    _RunningNode,
    _SimulationContext,
    _WeightedMerger,
    _WeightedSolver,
    _compare_double,
    merge,
)


def _graph_from_edges(edge_pairs, node_count=None):
    if node_count is None:
        node_count = max((max(pair) for pair in edge_pairs), default=-1) + 1
    graph = multistream.DAGGraph("weighted")
    nodes = [graph.add_node(f"n{i}", "Op") for i in range(node_count)]
    for src, dst in edge_pairs:
        graph.add_edge(nodes[src], 0, nodes[dst], 0)
    return graph


def _chain(n=4):
    return _graph_from_edges([(i, i + 1) for i in range(n - 1)], n)


def _costs(graph, duration=1.0):
    return [
        multistream.NodeCost(execution_time=duration, cube_block_num=1, vec_block_num=1)
        for _ in graph.get_nodes()
    ]


def _solver(graph, routes, costs=None, **kwargs):
    return _WeightedSolver(
        graph, routes, WeightedStreamMergeOptions(**kwargs), costs or _costs(graph)
    )


def test_empty_single_and_limited_merges():
    graph = _chain(3)
    assert merge(graph, []) == []
    assert merge(graph, [[0, 1, 2]], costs=_costs(graph)) == [0]
    routes = [[0], [1], [2]]
    mapping = merge(
        graph,
        routes,
        WeightedStreamMergeOptions(physical_stream_limit=1),
        _costs(graph),
    )
    assert mapping == [0, 0, 0]
    assert len(mapping) == len(routes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"physical_stream_limit": 0},
        {"window_width": 0},
        {"candidate_limit": 0},
        {"light_stream_limit": 0},
        {"repair_moves": -1},
        {"resim_candidate_limit": -1},
    ],
)
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _WeightedMerger(WeightedStreamMergeOptions(**kwargs)).merge(_chain(1), [[0]])


def test_invalid_route_sets_and_cycles_are_rejected():
    graph = _chain(3)
    for routes in ([[0, 1], [1, 2]], [[0, 1], []], [[0], [1]], [[0, 1], [3]]):
        with pytest.raises(ValueError):
            merge(graph, routes, costs=_costs(graph))
    cycle = _chain(3)
    nodes = cycle.get_nodes()
    cycle.add_edge(nodes[2], 0, nodes[0], 0)
    with pytest.raises(RuntimeError):
        merge(cycle, [[0, 1, 2]], costs=_costs(cycle))


def test_weighted_failures_log_the_cpp_wording_where_found(caplog, monkeypatch):
    caplog.set_level("ERROR", logger="multistream")

    def errors():
        return [
            (record.filename, record.getMessage())
            for record in caplog.records
            if record.levelname == "ERROR"
        ]

    def first_error(action, exc_type):
        caplog.clear()
        with pytest.raises(exc_type):
            action()
        return errors()[0]

    graph = _chain(3)
    for routes, message in (
        ([[0, 1], [1, 2]], "Node n1 appears in more than one logical stream."),
        ([[0, 1], []], "Weighted logical stream 1 should not be empty."),
        ([[0], [1]], "Node n2 is not assigned to any logical stream."),
        ([[0, 1], [3]], "Node index 3 is out of range [0, 3)."),
    ):
        action = lambda: merge(graph, routes, costs=_costs(graph))  # noqa: E731
        assert first_error(action, ValueError) == ("weighted_merger.py", message)
    cycle = _chain(3)
    nodes = cycle.get_nodes()
    cycle.add_edge(nodes[2], 0, nodes[0], 0)
    action = lambda: merge(cycle, [[0, 1, 2]], costs=_costs(cycle))  # noqa: E731
    assert first_error(action, RuntimeError) == (
        "weighted_merger.py",
        "Weighted merge only supports DAG graphs.",
    )

    for kwargs, message in (
        (
            {"physical_stream_limit": 0},
            "Weighted stream merge physical stream limit must be greater than 0.",
        ),
        (
            {"window_width": 0},
            "Weighted stream merge window width must be greater than 0.",
        ),
        (
            {"candidate_limit": 0},
            "Weighted stream merge candidate limit must be greater than 0.",
        ),
        (
            {"light_stream_limit": 0},
            "Weighted stream merge light_stream_limit must be greater than 0.",
        ),
        (
            {"repair_moves": -1},
            "Weighted stream merge repair_moves must be greater than or equal to 0.",
        ),
        (
            {"resim_candidate_limit": -1},
            "Weighted stream merge resim_candidate_limit must be greater than or equal to 0.",
        ),
    ):
        merger = _WeightedMerger(WeightedStreamMergeOptions(**kwargs))
        action = lambda: merger.merge(_chain(1), [[0]])  # noqa: E731
        assert first_error(action, ValueError) == ("weighted_merger.py", message)

    # These three checks return a failure value instead of raising, like C++.
    solver = _solver(graph, [[0, 1, 2]])
    solver._build_unit_dependency_graph()
    caplog.clear()
    assert solver._remove_unit_from_stream(0, solver._new_streaming_state()) == -1
    assert errors() == [
        ("weighted_merger.py", "Try to remove unassigned weighted unit 0.")
    ]
    caplog.clear()
    assert solver._build_simulation_node_streams([-1], True, [-1] * 3, [0] * 3) == -1
    assert errors() == [
        (
            "weighted_merger.py",
            "There is still an unassigned weighted unit in final simulation.",
        )
    ]
    context = _SimulationContext()
    context.node_streams = [0]
    caplog.clear()
    assert solver._start_simulation_ready_head([0], context) is False
    assert errors() == [
        (
            "weighted_merger.py",
            "Weighted stream simulation queue state is inconsistent.",
        )
    ]

    assert first_error(lambda: solver._compact_assignment([-1]), RuntimeError) == (
        "weighted_merger.py",
        "There is still an unassigned weighted unit when compacting assignment.",
    )
    monkeypatch.setattr(_WeightedSolver, "_candidate_flows", lambda self, *args: [])
    chain = _chain(2)
    action = lambda: merge(chain, [[0], [1]], costs=_costs(chain))  # noqa: E731
    assert first_error(action, RuntimeError) == (
        "weighted_merger.py",
        "No candidate physical stream for weighted unit 0.",
    )


def test_resource_profile_loading_clamps_missing_and_large_values():
    graph = _chain(3)
    costs = [
        multistream.NodeCost(execution_time=-1.0, cube_block_num=2**40),
        multistream.NodeCost(execution_time=2.5, vec_block_num=3),
        multistream.NodeCost(execution_time=0.0),
    ]
    solver = _solver(graph, [[0, 1, 2]], costs)
    assert solver.node_durations == [0.0, pytest.approx(2.5), 0.0]
    assert solver.node_aic_cores[0] == 2**31 - 1
    assert solver.unit_total_duration == [pytest.approx(2.5)]
    assert solver.unit_active_levels[0] == [0, 1]


def test_node_cost_summary_is_logged_like_cpp(caplog):
    graph = _chain(3)
    caplog.set_level("INFO", logger="multistream")
    costs = [
        multistream.NodeCost(execution_time=-1.0),
        multistream.NodeCost(execution_time=2.5),
        multistream.NodeCost(execution_time=0.0),
    ]
    _solver(graph, [[0, 1, 2]], costs)
    records = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert (
        "WARNING",
        "Weighted stream NodeCost summary: valid=2, missing=1.",
    ) in records

    caplog.clear()
    _solver(graph, [[0, 1, 2]])
    records = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert ("INFO", "Weighted stream NodeCost summary: valid=3, missing=0.") in records


def test_resource_profiles_and_units_are_sorted_by_duration_resource_and_id():
    graph = _graph_from_edges([], 4)
    costs = [
        multistream.NodeCost(execution_time=value, cube_block_num=1)
        for value in (1, 3, 2, 3)
    ]
    solver = _solver(graph, [[0], [1], [2], [3]], costs)
    assert solver.unit_total_duration == [1.0, 3.0, 2.0, 3.0]
    solver._build_unit_dependency_graph()
    solver._build_units_by_earliest_level()
    assert solver.units_by_earliest_level[0][:2] == [1, 3]
    assert solver.units_by_earliest_level[0][-1] == 0


def test_origin_hint_and_candidate_ordering_helpers():
    graph = _graph_from_edges([(0, 1)], 2)
    solver = _solver(graph, [[0], [1]])
    solver._build_unit_dependency_graph()
    state = solver._new_streaming_state()
    solver._apply_unit_to_stream(0, 0, state)
    state.stream_total_durations[1] = 2.0
    state.origin_stream_to_flow_counts[1] = {1: 2, 0: 1}
    assert solver._origin_hint_flows(1, state) == [1, 0]
    assert solver._adjacent_candidate_flows(1, state) == [0]
    assert solver._candidate_flows(1, 1, state, True)[0] == 0
    assert solver._fallback_candidate_flows(state, True) == [1]


def test_light_and_unique_candidate_limits():
    graph = _graph_from_edges([], 2)
    solver = _solver(graph, [[0], [1]], light_stream_limit=1, candidate_limit=2)
    state = solver._new_streaming_state()
    solver._apply_unit_to_stream(0, 0, state)
    solver._apply_unit_to_stream(1, 1, state)
    assert solver._light_candidate_flows(0, state) == [0]
    assert solver._unique_limited_candidates([1, 1, 0]) == [1, 0]
    solver.options.candidate_limit = 1
    assert solver._unique_limited_candidates([1, 0]) == [1]


def test_window_helpers_and_zero_duration_active_levels():
    graph = _chain(4)
    solver = _solver(graph, [[0, 1, 2, 3]], window_width=2)
    assert solver._window_end(0) == 1
    assert solver._window_mass([1.0, 2.0, 3.0, 4.0], 0) == 3.0
    assert solver._window_active_levels(0, 3) == [3]
    zero_solver = _solver(
        graph,
        [[0], [1, 2, 3]],
        [multistream.NodeCost(execution_time=0.0) for _ in graph.get_nodes()],
    )
    assert zero_solver.unit_active_levels == [[], []]


def test_apply_remove_update_stream_state_and_origin_counters():
    graph = _chain(2)
    solver = _solver(graph, [[0], [1]])
    state = solver._new_streaming_state()
    assert solver._apply_unit_to_stream(0, 0, state) is True
    assert state.assignment == [0, -1]
    assert state.active_stream_count == 1
    assert state.stream_total_durations[0] == 1.0
    assert solver._remove_unit_from_stream(0, state) == 0
    assert state.assignment == [-1, -1]
    assert solver._remove_unit_from_stream(0, state) == -1
    with pytest.raises(RuntimeError):
        solver._apply_unit_to_stream(0, 3, state)


def test_running_heap_order_and_score_tie_breakers():
    values = [_RunningNode(3.0, 2, 2), _RunningNode(1.0, 4, 3), _RunningNode(1.0, 2, 4)]
    heap = []
    for value in values:
        solver = _WeightedSolver
        solver._heap_push_running(heap, value)
    assert solver._heap_peek_min(heap).node_index == 4
    assert [solver._heap_pop_running(heap).node_index for _ in range(3)] == [4, 3, 2]
    with pytest.raises(RuntimeError):
        solver._heap_pop_running(heap)
    lhs = _LiteScore(
        total=1, time_conflict=1, event_local=0, time_load=1, candidate_stream=0
    )
    rhs = _LiteScore(
        total=1, time_conflict=2, event_local=0, time_load=1, candidate_stream=0
    )
    assert solver._lite_score_less(lhs, rhs)
    assert _compare_double(1.0, 1.0 + K_IMPROVE_EPS / 2) == 0


def test_simulation_helpers_report_feasible_and_infeasible_assignments():
    graph = _chain(3)
    solver = _solver(graph, [[0], [1], [2]])
    context = solver._new_simulation_context([0, 1, 1], [1, 1, 1])
    assert solver._simulation_ready_heads(context) == [0]
    assert solver._run_simulation(3, context).feasible is True
    assert context.executed_count == 3
    assert solver._build_and_run_simulation([-1, -1, -1], False).makespan == 0.0
    assert solver._build_and_run_simulation([-1, 0, 0], True).feasible is False


def test_simulation_comparison_and_same_time_finish():
    graph = _graph_from_edges([], 2)
    solver = _solver(graph, [[0], [1]], _costs(graph, 1.0))
    solver.node_bottom_ranks = [2.0, 1.0]
    solver.node_durations = [1.0, 1.0]
    solver.position = [0, 1]
    assert solver._simulation_node_compare(0, 1) < 0
    context = _SimulationContext()
    context.node_streams = [0, 1]
    context.running_in_stream = [1, 1]
    context.finished = [False, False]
    context.running_heap = [_RunningNode(2.0, 0, 0), _RunningNode(2.0, 1, 1)]
    solver._finish_next_simulation_nodes(context)
    assert context.executed_count == 2 and context.current_time == 2.0


def test_select_best_stream_uses_resimulation_and_falls_back_when_infeasible():
    graph = _graph_from_edges([], 2)
    solver = _solver(graph, [[0], [1]], resim_candidate_limit=2)
    solver._build_unit_dependency_graph()
    state = solver._new_streaming_state()
    solver._apply_unit_to_stream(0, 0, state)
    assert solver._select_best_stream(1, 0, [0, 1], state) in (0, 1)
    solver.options.resim_candidate_limit = 0
    assert solver._select_best_stream(1, 0, [0, 1], state) in (0, 1)


def test_compact_solve_and_repair_contract():
    solver = _solver(_chain(3), [[0], [1], [2]], repair_moves=2)
    assert solver.solve() == [0, 0, 0]
    assert solver._compact_assignment([]) == []
    assert solver._compact_assignment([3, 1, 3]) == [1, 0, 1]
    with pytest.raises(RuntimeError):
        solver._compact_assignment([0, -1])


def test_cross_stream_dependency_changes_event_local_score():
    graph = _graph_from_edges([(0, 1)], 2)
    solver = _solver(graph, [[0], [1]])
    solver._build_unit_dependency_graph()
    assert solver._calc_event_local(1, 0, [0, -1]) == 0
    assert solver._calc_event_local(1, 1, [0, -1]) == 1


def test_large_independent_graph_merges_quickly(large_graph):
    graph = large_graph(120)
    routes = [[idx] for idx in range(len(graph.get_nodes()))]
    mapping = merge(
        graph,
        routes,
        WeightedStreamMergeOptions(physical_stream_limit=8),
        _costs(graph),
    )
    assert len(mapping) == len(routes)
    assert all(0 <= stream < 8 for stream in mapping)
