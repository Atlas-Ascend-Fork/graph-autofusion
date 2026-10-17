# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Public planning transaction and consistency tests, plan §4.9."""

import logging

import pytest

import multistream
from multistream.algorithm.planning import (
    _build_effective_costs,
    _normalize_strategy,
)


def _graph(n=4):
    graph = multistream.DAGGraph("plan")
    nodes = [graph.add_node(f"n{i}", "Op") for i in range(n)]
    for src, dst in zip(nodes, nodes[1:]):
        graph.add_edge(src, 0, dst, 0)
    return graph


def test_plan_argument_validation():
    graph = _graph(1)
    with pytest.raises(TypeError):
        multistream.plan(object())
    for value in (0, 65):
        with pytest.raises(ValueError):
            multistream.plan(graph, max_stream_num=value)
    for value in (True, "8"):
        with pytest.raises(TypeError):
            multistream.plan(graph, max_stream_num=value)
    assert multistream.plan(graph, max_stream_num=None).get_required_stream_count() == 1
    for value in (True, -1, "0"):
        with pytest.raises((TypeError, ValueError)):
            multistream.plan(graph, base_stream_id=value)
    for value in (b"x", 1):
        with pytest.raises(TypeError):
            multistream.plan(graph, profiling_path=value)


@pytest.mark.parametrize(
    "value",
    [
        multistream.MergeStrategy.LOAD_BALANCE,
        "LoadBalance",
        "load_balance",
        "MainStream",
        "main_stream",
    ],
)
def test_strategy_aliases_are_accepted(value):
    assert isinstance(_normalize_strategy(value), multistream.MergeStrategy)


@pytest.mark.parametrize(
    "value", ["weighted load balance", "WeightedLoadBalance", "weighted_load_balance"]
)
def test_weighted_strategy_strings_are_rejected(value):
    # The weighted merger is selected by profiling data alone.  Keeping a
    # string alias would let a caller route into it from the outside.
    with pytest.raises(ValueError):
        _normalize_strategy(value)


def test_invalid_strategy_and_profiling_path_fail_without_partial_plan(tmp_path):
    graph = _graph(2)
    with pytest.raises(ValueError):
        multistream.plan(graph, merge_strategy="not-a-strategy")
    with pytest.raises(ValueError):
        multistream.plan(graph, profiling_path=tmp_path / "missing.csv")
    assert all(
        node.get_node_cost() == multistream.NodeCost() for node in graph.get_nodes()
    )


def test_cycle_is_rejected_and_graph_is_not_mutated():
    graph = _graph(2)
    nodes = graph.get_nodes()
    graph.add_edge(nodes[-1], 1, nodes[0], 1)
    before = list(graph._get_edges())
    with pytest.raises(RuntimeError):
        multistream.plan(graph)
    assert graph._get_edges() == before


def test_cycle_is_logged_where_found_before_the_boundary_log(caplog):
    graph = _graph(2)
    nodes = graph.get_nodes()
    graph.add_edge(nodes[-1], 1, nodes[0], 1)
    caplog.set_level(logging.ERROR, logger="multistream")
    with pytest.raises(RuntimeError):
        multistream.plan(graph)
    errors = [
        (record.filename, record.getMessage())
        for record in caplog.records
        if record.levelno == logging.ERROR
    ]
    assert errors == [
        ("planning.py", "graph contains a cycle; stream allocation requires a DAG"),
        ("planning.py", "stream allocation failed"),
    ]


def test_stream_plan_is_read_only_and_checks_node_identity():
    graph = _graph(2)
    plan = multistream.plan(graph)
    with pytest.raises(AttributeError):
        plan.extra = 1
    with pytest.raises(TypeError):
        plan.get_stream_id(object())
    foreign = multistream.DAGGraph("foreign").add_node("n0", "Op")
    with pytest.raises(ValueError):
        plan.get_stream_id(foreign)


def test_required_count_and_base_offset_are_self_consistent(diamond_graph):
    graph = diamond_graph
    plan0 = multistream.plan(graph, max_stream_num=2, base_stream_id=0)
    plan7 = multistream.plan(graph, max_stream_num=2, base_stream_id=7)
    for node in graph.get_nodes():
        assert plan7.get_stream_id(node) == plan0.get_stream_id(node) + 7
    used = {plan0.get_stream_id(node) for node in graph.get_nodes()}
    assert plan0.get_required_stream_count() == len(used)


def test_zero_cost_exclusion_and_serial_override():
    graph = multistream.DAGGraph("cost")
    normal = graph.add_node("normal", "Op")
    zero = graph.add_node("zero", "Op")
    serial = graph.add_node("serial", "Op")
    zero.set_node_cost(multistream.NodeCost(execution_time=0.0))
    serial.set_stream_label("serial")
    serial.set_node_cost(multistream.NodeCost(execution_time=0.0))
    plan = multistream.plan(graph)
    assert plan.get_stream_id(normal) >= 0
    assert plan.get_stream_id(zero) == multistream.INVALID_STREAM_ID
    assert plan.get_stream_id(serial) >= 0


def test_effective_cost_overlay_is_transaction_local(op_summary_csv):
    graph = _graph(1)
    path = op_summary_csv([["n0", "AI_CORE", "3", "8", "0"]])
    costs, matched = _build_effective_costs(graph, path, return_match=True)
    assert matched == frozenset({0})
    assert costs[0].execution_time == pytest.approx(3)
    assert graph.get_nodes()[0].get_node_cost() == multistream.NodeCost()
    # A name match only records the hit; the allocator turns it into the
    # routing decision once it knows which nodes participate.
    assert _build_effective_costs(graph, None, return_match=True)[1] == frozenset()


def test_plan_is_deterministic_and_failure_is_logged(caplog, deep_chain_graph):
    graph = deep_chain_graph(20)
    plans = [
        multistream.plan(graph, merge_strategy=strategy)
        for strategy in multistream.MergeStrategy
    ]
    for plan in plans:
        assert [plan.get_stream_id(node) for node in graph.get_nodes()] == [
            plan.get_stream_id(node) for node in graph.get_nodes()
        ]
    caplog.set_level(logging.ERROR, logger="multistream")
    bad = _graph(1)
    with pytest.raises(ValueError):
        multistream.plan(bad, max_stream_num=0)
    assert any(
        "stream allocation failed" in record.getMessage() for record in caplog.records
    )
