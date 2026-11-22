# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Engine affinity fields, core-count queries and profiling routing."""

from __future__ import annotations

import logging

import pytest

import multistream
from multistream.algorithm.planning import _build_effective_costs


def _op_summary(tmp_path, rows, name="op_summary.csv"):
    path = tmp_path / name
    lines = ["Op Name,Task Type,Task Duration(us),Block Num,Mix Block Num"]
    lines.extend(",".join(str(field) for field in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The new cost fields


def test_new_fields_survive_every_cost_copy():
    graph = multistream.DAGGraph("fields")
    node = graph.add_node("n0", "Op")
    node.set_node_cost(
        multistream.NodeCost(
            execution_time=2.0,
            cube_core_num_limit=4,
            vec_core_num_limit=6,
        )
    )
    node.set_core_type(multistream.CoreType.MIX_AIV)
    # Through the node's own copy, ...
    stored = node.get_node_cost()
    assert node.get_core_type() is multistream.CoreType.MIX_AIV
    assert (stored.cube_core_num_limit, stored.vec_core_num_limit) == (4, 6)
    # ... through the allocation's effective costs, ...
    costs = _build_effective_costs(graph)
    assert (costs[0].cube_core_num_limit, costs[0].vec_core_num_limit) == (4, 6)
    # ... and through the residual graph the allocator builds.
    from multistream.algorithm.stream_allocator import _split_serial_graph

    split = _split_serial_graph(graph, costs)
    residual_node = split.residual_graph.get_nodes()[0]
    assert residual_node.get_core_type() is multistream.CoreType.MIX_AIV
    residual = residual_node.get_node_cost()
    assert (residual.cube_core_num_limit, residual.vec_core_num_limit) == (4, 6)


def test_profiling_never_overwrites_the_engine_or_the_user_limit(tmp_path):
    graph = multistream.DAGGraph("overlay")
    node = graph.add_node("n0", "Op")
    node.set_node_cost(
        multistream.NodeCost(
            execution_time=1.0,
            memory_usage=11,
            bandwidth_usage=22,
            cube_core_num_limit=4,
            vec_core_num_limit=6,
        )
    )
    node.set_core_type(multistream.CoreType.AIV)
    # The sample says AI_CORE; the engine still comes from the adapter.
    path = _op_summary(tmp_path, [["n0", "AI_CORE", 5.0, 8, 0]])
    costs, matched = _build_effective_costs(graph, str(path), return_match=True)
    assert matched == frozenset({0})
    assert costs[0].execution_time == pytest.approx(5.0)
    assert costs[0].cube_block_num == 8
    assert node.get_core_type() is multistream.CoreType.AIV
    assert (costs[0].cube_core_num_limit, costs[0].vec_core_num_limit) == (4, 6)
    assert node.get_node_cost().cube_block_num == 0
    assert costs[0].memory_usage == 11
    assert costs[0].bandwidth_usage == 22


# --------------------------------------------------------------------------
# Core-count suggestion queries


def _plan_with_node():
    graph = multistream.DAGGraph("plan")
    node = graph.add_node("n0", "Op")
    return multistream.plan(graph), node


@pytest.mark.parametrize("kernel_type", ["AIC", "AIV"])
def test_core_num_is_zero_for_every_legal_query(kernel_type):
    plan, node = _plan_with_node()
    assert plan.get_core_num(node, kernel_type) == 0


@pytest.mark.parametrize(
    "kernel_type", ["aic", "aiv", "MIX_AIC", "MIX_AIV", "", "AI_CORE"]
)
def test_illegal_kernel_type_strings_are_rejected(kernel_type):
    # MIX_AIC and MIX_AIV describe the node, not a queryable resource: a
    # mixed node is asked about once per engine instead.
    plan, node = _plan_with_node()
    with pytest.raises(ValueError):
        plan.get_core_num(node, kernel_type)


@pytest.mark.parametrize("kernel_type", [None, 0, b"AIC", multistream.CoreType.AIC])
def test_non_string_kernel_type_is_a_type_error(kernel_type):
    plan, node = _plan_with_node()
    with pytest.raises(TypeError):
        plan.get_core_num(node, kernel_type)


def test_core_num_checks_node_ownership_like_get_stream_id():
    plan, _node = _plan_with_node()
    with pytest.raises(TypeError):
        plan.get_core_num(object(), "AIC")
    foreign = multistream.DAGGraph("foreign").add_node("n0", "Op")
    with pytest.raises(ValueError):
        plan.get_core_num(foreign, "AIC")


def test_repeated_plans_stay_separate(diamond_graph):
    graph = diamond_graph
    first = multistream.plan(graph, max_stream_num=2)
    second = multistream.plan(graph, max_stream_num=2)
    assert first is not second
    for node in graph.get_nodes():
        assert first.get_core_num(node, "AIC") == 0
        assert second.get_core_num(node, "AIV") == 0
        assert first.get_stream_id(node) == second.get_stream_id(node)


def test_plan_states_that_suggestions_are_not_implemented(caplog):
    caplog.set_level(logging.INFO, logger="multistream")
    _plan_with_node()
    assert any(
        "core number suggestion is not supported in this release" in message
        for message in (record.getMessage() for record in caplog.records)
    )


# --------------------------------------------------------------------------
# Profiling routing


def _routing_graph():
    """Two independent branches whose merging depends on the weights."""

    graph = multistream.DAGGraph("routing")
    nodes = [graph.add_node(name, "Op") for name in ("a", "b", "c", "d")]
    graph.add_edge(nodes[0], 0, nodes[1], 0)
    return graph, nodes


def _implementation(caplog):
    for record in caplog.records:
        message = record.getMessage()
        if "used implementation=" in message:
            return message.split("used implementation=")[1].split(",")[0]
    raise AssertionError("no routing log was emitted")


@pytest.mark.parametrize(
    "strategy",
    list(multistream.MergeStrategy),
)
def test_matched_profiling_overrides_both_base_strategies(tmp_path, caplog, strategy):
    graph, nodes = _routing_graph()
    path = _op_summary(
        tmp_path,
        [
            ["a", "AI_CORE", 1.0, 1, 0],
            ["b", "AI_CORE", 1.0, 1, 0],
            ["c", "AI_CORE", 1.0, 1, 0],
            ["d", "AI_CORE", 2.0, 1, 0],
        ],
    )
    caplog.set_level(logging.INFO, logger="multistream")
    plan = multistream.plan(
        graph, max_stream_num=2, merge_strategy=strategy, profiling_path=path
    )
    assert _implementation(caplog) == "WeightedLoadBalance"
    assert tuple(plan.get_stream_id(node) for node in nodes) == (0, 0, 0, 1)
    assert plan.get_required_stream_count() == 2
    assert (
        f"graph routing requested strategy={strategy.value}, "
        "used implementation=WeightedLoadBalance, "
        "reason=profiling samples matched participating nodes."
    ) in [record.getMessage() for record in caplog.records]
    assert all(node.get_node_cost().execution_time == -1.0 for node in nodes)
    # A later allocation without profiling must return to the requested
    # base strategy and must not reuse the previous routing decision.
    caplog.clear()
    multistream.plan(graph, max_stream_num=2, merge_strategy=strategy)
    assert _implementation(caplog) == strategy.value


@pytest.mark.parametrize(
    "strategy",
    [multistream.MergeStrategy.LOAD_BALANCE, multistream.MergeStrategy.MAIN_STREAM],
)
def test_without_a_match_both_strategies_keep_the_basic_algorithm(
    tmp_path, caplog, strategy
):
    graph, _nodes = _routing_graph()
    path = _op_summary(tmp_path, [["other", "AI_CORE", 1.0, 1, 0]])
    caplog.set_level(logging.INFO, logger="multistream")
    multistream.plan(
        graph, max_stream_num=2, merge_strategy=strategy, profiling_path=path
    )
    assert _implementation(caplog) == strategy.value


def test_hand_written_costs_alone_do_not_select_the_weighted_merger(caplog):
    graph, nodes = _routing_graph()
    for node in nodes:
        node.set_node_cost(multistream.NodeCost(execution_time=3.0))
    caplog.set_level(logging.INFO, logger="multistream")
    multistream.plan(graph, max_stream_num=2)
    assert _implementation(caplog) == "LoadBalance"


@pytest.mark.parametrize("strategy", list(multistream.MergeStrategy))
def test_a_hit_on_excluded_nodes_alone_does_not_weight_the_merge(
    tmp_path, caplog, strategy
):
    graph, nodes = _routing_graph()
    # The adapter's exclusion marker keeps these nodes out of merging, so
    # their samples cannot select the weighted merger either.
    for node in (nodes[2], nodes[3]):
        node.set_node_cost(multistream.NodeCost(execution_time=0.0))
    path = _op_summary(
        tmp_path, [["c", "AI_CORE", 1.0, 1, 0], ["d", "AI_CORE", 2.0, 1, 0]]
    )
    caplog.set_level(logging.INFO, logger="multistream")
    multistream.plan(
        graph, max_stream_num=2, merge_strategy=strategy, profiling_path=path
    )
    assert _implementation(caplog) == strategy.value
    assert "excluded or serial nodes" in "".join(
        record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize("strategy", list(multistream.MergeStrategy))
def test_a_hit_on_serial_label_groups_alone_does_not_weight_the_merge(
    tmp_path, caplog, strategy
):
    graph, nodes = _routing_graph()
    for index, node in enumerate(nodes):
        node.set_stream_label(f"serial-{index}")
    path = _op_summary(tmp_path, [["a", "AI_CORE", 1.0, 1, 0]])
    caplog.set_level(logging.INFO, logger="multistream")
    multistream.plan(
        graph, max_stream_num=4, merge_strategy=strategy, profiling_path=path
    )
    assert _implementation(caplog) == strategy.value
    assert "profiling samples matched only excluded or serial nodes" in caplog.text


@pytest.mark.parametrize("strategy", list(multistream.MergeStrategy))
def test_a_partial_match_on_a_participating_node_is_enough(tmp_path, caplog, strategy):
    graph, nodes = _routing_graph()
    nodes[3].set_node_cost(multistream.NodeCost(execution_time=0.0))
    path = _op_summary(
        tmp_path, [["c", "AI_CORE", 1.0, 1, 0], ["d", "AI_CORE", 2.0, 1, 0]]
    )
    caplog.set_level(logging.INFO, logger="multistream")
    multistream.plan(
        graph, max_stream_num=2, merge_strategy=strategy, profiling_path=path
    )
    assert _implementation(caplog) == "WeightedLoadBalance"


@pytest.mark.parametrize("strategy", list(multistream.MergeStrategy))
@pytest.mark.parametrize("duration", ["0", "0.0", "-1", "-0.5", "nan", "inf", "1e-50"])
def test_invalid_samples_are_not_valid_costs(tmp_path, caplog, duration, strategy):
    from multistream.algorithm.profiling import parse

    path = _op_summary(tmp_path, [["a", "AI_CORE", duration, 4, 0]])
    assert parse(path) == {}

    graph, _nodes = _routing_graph()
    caplog.set_level(logging.INFO, logger="multistream")
    multistream.plan(
        graph, max_stream_num=2, merge_strategy=strategy, profiling_path=path
    )
    # A zero sample must not become the exclusion marker, and it must not
    # select the weighted merger either.
    assert _implementation(caplog) == strategy.value
    costs = _build_effective_costs(graph, str(path))
    assert costs[0].execution_time == -1.0
    assert costs[0].cube_block_num == 0


@pytest.mark.parametrize("strategy", list(multistream.MergeStrategy))
def test_an_empty_but_valid_profiling_file_falls_back_to_the_basic_algorithm(
    tmp_path, caplog, strategy
):
    graph, _nodes = _routing_graph()
    path = _op_summary(tmp_path, [])
    caplog.set_level(logging.INFO, logger="multistream")
    plan = multistream.plan(
        graph, max_stream_num=2, merge_strategy=strategy, profiling_path=path
    )
    assert _implementation(caplog) == strategy.value
    assert plan.get_required_stream_count() >= 1


def test_the_original_exclusion_marker_survives_a_matched_sample(tmp_path):
    graph, nodes = _routing_graph()
    nodes[0].set_node_cost(multistream.NodeCost(execution_time=0.0))
    path = _op_summary(tmp_path, [["a", "AI_CORE", 9.0, 4, 0]])
    plan = multistream.plan(graph, max_stream_num=2, profiling_path=path)
    assert plan.get_stream_id(nodes[0]) == multistream.INVALID_STREAM_ID
    assert nodes[0].get_node_cost().execution_time == 0.0
