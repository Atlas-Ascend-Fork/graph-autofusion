# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import logging

import pytest

import multistream


def test_public_surface_and_linear_graph():
    assert set(multistream.__all__) == {
        "DAGGraph",
        "DAGNode",
        "NodeCost",
        "CoreType",
        "StreamPlan",
        "MergeStrategy",
        "plan",
    }
    graph = multistream.DAGGraph("linear")
    first = graph.add_node("first", "Op")
    second = graph.add_node("second", "Op")
    graph.add_edge(first, 0, second, 0)
    plan = multistream.plan(graph)
    assert plan.get_required_stream_count() == 1
    assert plan.get_stream_id(first) == plan.get_stream_id(second) == 0


def test_serial_and_zero_cost_nodes():
    graph = multistream.DAGGraph("labels")
    regular = graph.add_node("regular", "Op")
    serial = graph.add_node("serial", "Op")
    excluded = graph.add_node("excluded", "Op")
    serial.set_stream_label("label")
    excluded.set_node_cost(multistream.NodeCost(execution_time=0.0))
    graph.add_edge(regular, 0, serial, 0)
    graph.add_edge(serial, 0, excluded, 0)
    plan = multistream.plan(graph)
    assert plan.get_stream_id(regular) >= 0
    assert plan.get_stream_id(serial) >= 0
    assert plan.get_stream_id(excluded) == multistream.INVALID_STREAM_ID


@pytest.mark.parametrize("value", [0, 65, True, "8"])
def test_stream_limit_validation(value):
    graph = multistream.DAGGraph("g")
    graph.add_node("n", "Op")
    with pytest.raises((TypeError, ValueError)):
        multistream.plan(graph, max_stream_num=value)


def test_graph_edge_validation_is_atomic_and_checks_ownership():
    graph = multistream.DAGGraph("g")
    src = graph.add_node("src", "Op")
    dst = graph.add_node("dst", "Op")
    foreign_graph = multistream.DAGGraph("foreign")
    foreign = foreign_graph.add_node("foreign", "Op")

    with pytest.raises(ValueError):
        graph.add_edge(src, 0, foreign, 0)
    with pytest.raises(ValueError):
        graph.add_edge(src, -2, dst, 0)
    # Edge access is deliberately private (``_Edge`` is not public API), so
    # the structural assertions below go through the internal accessors.
    assert len(graph._get_edges()) == 0

    graph.add_edge(src, -1, dst, -1)
    assert len(graph._get_edges()) == 1
    assert len(src._get_output_edges()) == 1
    assert len(dst._get_input_edges()) == 1


def test_stream_plan_rejects_foreign_nodes_and_new_attributes():
    graph = multistream.DAGGraph("g")
    node = graph.add_node("node", "Op")
    plan = multistream.plan(graph)

    with pytest.raises(AttributeError):
        plan.stream_ids = []
    foreign_graph = multistream.DAGGraph("foreign")
    foreign = foreign_graph.add_node("node", "Op")
    with pytest.raises(ValueError):
        plan.get_stream_id(foreign)
    assert plan.get_stream_id(node) == 0


def test_profiling_scope_match_preserves_explicit_zero_marker(tmp_path):
    # The adapter uses execution_time == 0.0 to mark a node without a device
    # task.  A same-name profiling record must not overwrite that marker.
    # The hit is still recorded, but the node is excluded from merging and so
    # never reaches the intersection that selects the weighted merger.
    profiling = tmp_path / "op_summary.csv"
    profiling.write_text(
        "Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num\n"
        "excluded,AI_CORE,42.5,8,0\n"
        "active,AI_VECTOR_CORE,7.25,4,0\n",
        encoding="utf-8",
    )
    graph = multistream.DAGGraph("profile")
    excluded = graph.add_node("scope/excluded", "Op")
    active = graph.add_node("scope/active", "Op")
    excluded.set_node_cost(multistream.NodeCost(execution_time=0.0))
    active.set_node_cost(multistream.NodeCost(execution_time=-1.0))

    from multistream.algorithm.planning import _build_effective_costs

    costs, matched = _build_effective_costs(graph, str(profiling), return_match=True)
    assert matched == frozenset({excluded._index, active._index})
    assert costs[excluded._index].execution_time == 0.0
    # The adapter's zero marker wins only for duration; profiling resource
    # counts are still copied and remain available to the weighted merger.
    assert costs[excluded._index].cube_block_num == 8
    assert costs[active._index].execution_time == pytest.approx(7.25, rel=1e-6)
    assert costs[active._index].vec_block_num == 4


def test_real_profiling_changes_stream_grouping_and_logs(tmp_path, caplog):
    """A real DAGGraph must switch to weighted merging when profiling matches."""

    graph = multistream.DAGGraph("real_profiling_split")
    nodes = [graph.add_node(name, "AI_CORE") for name in ("a", "b", "c", "d")]
    graph.add_edge(nodes[0], 0, nodes[1], 0)

    caplog.set_level(logging.INFO, logger="multistream")
    plain = multistream.plan(
        graph,
        max_stream_num=2,
        merge_strategy=multistream.MergeStrategy.LOAD_BALANCE,
    )
    plain_ids = tuple(plain.get_stream_id(node) for node in nodes)
    plain_logs = [record.getMessage() for record in caplog.records]
    assert plain_ids == (0, 0, 1, 1)
    assert plain.get_required_stream_count() == 2
    assert "Use kLoadBalance strategy." in plain_logs
    assert (
        "graph real_profiling_split requested strategy=LoadBalance, "
        "used implementation=LoadBalance, "
        "reason=no profiling sample matched a participating node."
    ) in plain_logs

    profiling = tmp_path / "op_summary.csv"
    profiling.write_text(
        "Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num\n"
        "a,AI_CORE,1,1,0\n"
        "b,AI_CORE,1,1,0\n"
        "c,AI_CORE,1,1,0\n"
        "d,AI_CORE,2,1,0\n",
        encoding="utf-8",
    )
    caplog.clear()
    profiled = multistream.plan(
        graph,
        max_stream_num=2,
        merge_strategy=multistream.MergeStrategy.LOAD_BALANCE,
        profiling_path=profiling,
    )
    profiled_ids = tuple(profiled.get_stream_id(node) for node in nodes)
    profile_logs = [record.getMessage() for record in caplog.records]

    assert profiled_ids == (0, 0, 0, 1)
    assert profiled.get_required_stream_count() == 2
    assert plain_ids[2] == plain_ids[3]
    assert profiled_ids[0] == profiled_ids[1] == profiled_ids[2]
    assert profiled_ids[2] != profiled_ids[3]
    assert any(
        message.startswith("Parsed 4 profiling records from ")
        for message in profile_logs
    )
    assert (
        "graph real_profiling_split requested strategy=LoadBalance, "
        "used implementation=WeightedLoadBalance, "
        "reason=profiling samples matched participating nodes."
    ) in profile_logs
    assert not any("Use kLoadBalance strategy." in message for message in profile_logs)
    # The profiling overlay is private to this allocation; the source graph is unchanged.
    assert all(node.get_node_cost().execution_time == -1.0 for node in nodes)


def test_profiling_parser_skips_nonfinite_and_out_of_range_rows(tmp_path):
    profiling = tmp_path / "invalid_costs.csv"
    profiling.write_text(
        "Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num\n"
        "nan_cost,AI_CORE,nan,1,0\n"
        "inf_cost,AI_CORE,inf,1,0\n"
        "overflow_cost,AI_CORE,1e1000,1,0\n"
        "block_overflow,AI_CORE,1,4294967296,0\n"
        "negative_block,AI_CORE,1,-1,0\n"
        "valid,AI_CORE,1.25,2,0\n",
        encoding="utf-8",
    )

    from multistream.algorithm.profiling import parse

    profiles = parse(profiling)
    assert set(profiles) == {"valid"}
    assert profiles["valid"].execution_time == pytest.approx(1.25, rel=1e-6)
    assert profiles["valid"].cube_block_num == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1e100])
def test_execution_time_rejects_nonfinite_or_float32_overflow(value):
    with pytest.raises((TypeError, ValueError)):
        multistream.NodeCost(execution_time=value)


def test_plan_accepts_pathlike_and_spaced_strategy(tmp_path):
    profiling = tmp_path / "costs.csv"
    profiling.write_text(
        "Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num\n"
        "node,AI_CORE,3.5,1,0\n",
        encoding="utf-8",
    )
    graph = multistream.DAGGraph("pathlike")
    node = graph.add_node("node", "Op")
    plan = multistream.plan(
        graph,
        merge_strategy="load balance",
        profiling_path=profiling,
    )
    assert plan.get_stream_id(node) >= 0


@pytest.mark.parametrize(
    "strategy",
    [
        "LoadBalance",
        "load_balance",
        multistream.MergeStrategy.LOAD_BALANCE,
    ],
)
def test_internal_merger_accepts_load_balance_aliases(strategy):
    graph = multistream.DAGGraph("aliases")
    first = graph.add_node("first", "Op")
    second = graph.add_node("second", "Op")
    graph.add_edge(first, 0, second, 0)

    from multistream.algorithm.merger import StreamMergeOptions, _Merger
    from multistream.algorithm.stream_allocator import _path_cover_routes

    routes = _path_cover_routes(graph)
    mapping = _Merger(StreamMergeOptions(strategy=strategy, repair_moves=2)).merge(
        graph, routes
    )
    assert mapping == [0]
