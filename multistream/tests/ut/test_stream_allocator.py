# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Path-cover allocator and serial-label split tests (plan §4.5)."""

import pytest

import multistream
from multistream.algorithm.stream_allocator import (
    DEFAULT_MAX_PHYSICAL_STREAMS,
    _StreamAllocConfig,
    _is_dag,
    _split_serial_graph,
    by_path_cover,
)


def _costs(graph, value=1.0):
    if value != 1.0:
        return [multistream.NodeCost(execution_time=value) for _ in graph.get_nodes()]
    return [node.get_node_cost() for node in graph.get_nodes()]


def _run(
    graph,
    *,
    max_stream_id=7,
    base=0,
    strategy=multistream.MergeStrategy.LOAD_BALANCE,
    weighted=False,
):
    # ``weighted`` is what ``plan`` derives from the profiling hits; the
    # allocator tests set it directly to exercise the weighted merger.
    config = _StreamAllocConfig(
        max_stream_id=max_stream_id,
        base_stream_id=base,
        merge_strategy=strategy,
        use_profiling_weights=weighted,
    )
    ids = by_path_cover(graph, _costs(graph), config)
    return ids, config.required_streams


def test_linear_and_branch_graphs_have_expected_stream_shape(
    linear_graph, multi_branch_graph
):
    ids, required = _run(linear_graph(8))
    assert required == 1
    assert len(set(ids)) == 1
    graph = multi_branch_graph(width=3, depth=2)
    ids, required = _run(graph, max_stream_id=7)
    assert required >= 1
    assert all(stream >= 0 for stream in ids)


def test_empty_graph_and_default_limit():
    graph = multistream.DAGGraph("empty")
    ids, required = _run(graph, max_stream_id=-1)
    assert ids == []
    assert required == 0
    assert DEFAULT_MAX_PHYSICAL_STREAMS == 8


def test_nonserial_ids_respect_base_and_physical_limit(multi_branch_graph):
    graph = multi_branch_graph(width=4, depth=2)
    ids, required = _run(graph, max_stream_id=1, base=7)
    assert required <= 2
    assert all(7 <= stream < 9 for stream in ids)


def test_data_and_netoutput_types_are_not_filtered():
    graph = multistream.DAGGraph("types")
    data = graph.add_node("data", "Data")
    out = graph.add_node("out", "NetOutput")
    graph.add_edge(data, 0, out, 0)
    ids, required = _run(graph)
    assert ids == [ids[0], ids[0]]
    assert required == 1


def test_serial_labels_share_streams_and_are_appended(serial_labeled_graph):
    ids, required = _run(serial_labeled_graph, max_stream_id=1, base=4)
    graph = serial_labeled_graph
    by_name = {node._name: ids[node._index] for node in graph.get_nodes()}
    assert by_name["s1"] != by_name["s2"]
    assert by_name["s1"] >= 4 and by_name["s2"] >= 4
    assert required == 3


def test_all_serial_graph_and_zero_cost_serial_node():
    graph = multistream.DAGGraph("all_serial")
    a = graph.add_node("a", "Op")
    b = graph.add_node("b", "Op")
    a.set_stream_label("x")
    b.set_stream_label("y")
    b.set_node_cost(multistream.NodeCost(execution_time=0.0))
    ids, required = _run(graph, max_stream_id=1)
    assert ids[0] != ids[1]
    assert required == 2


def test_path_cover_summaries_keep_cpp_wording(caplog):
    graph = multistream.DAGGraph("summary")
    a = graph.add_node("a", "Op")
    b = graph.add_node("b", "Op")
    serial = graph.add_node("serial", "Op")
    zero = graph.add_node("zero", "Op")
    graph.add_edge(a, 0, b, 0)
    serial.set_stream_label("x")
    zero.set_node_cost(multistream.NodeCost(execution_time=0.0))
    caplog.set_level("INFO", logger="multistream")
    _ids, required = _run(graph, max_stream_id=7, base=3)
    messages = [record.getMessage() for record in caplog.records]
    for message in (
        "ByPathCover start: graph=summary, nodes=4, max_stream_id=7, base_stream_id=3",
        "serial node num: 1, non-weight node num: 1",
        "rest dag node num: 2",
        "Logical stream num:1, merged physical stream num:1",
        "ByPathCover with residual graph done: residual_nodes=2, serial_labels=1, required_streams=2",
    ):
        assert message in messages
    assert required == 2


def test_route_and_serial_debug_logs_use_original_node_indices(caplog):
    graph = multistream.DAGGraph("debug")
    zero = graph.add_node("zero", "Op")
    a = graph.add_node("a", "Op")
    b = graph.add_node("b", "Op")
    serial = graph.add_node("serial", "Op")
    graph.add_edge(a, 0, b, 0)
    zero.set_node_cost(multistream.NodeCost(execution_time=0.0))
    serial.set_stream_label("x")
    caplog.set_level("DEBUG", logger="multistream")
    _run(graph, max_stream_id=7, base=3)
    messages = [record.getMessage() for record in caplog.records]
    # "zero" is left out of the residual graph, so residual index 0 is original index 1.
    assert (
        "logical stream 0 merged to physical stream 0, set as 3, nodes: 1 2 "
        in messages
    )
    assert "serial label x set as stream 4, nodes: 3 " in messages


def test_serial_stream_ids_are_appended_even_when_physical_limit_is_one():
    graph = multistream.DAGGraph("serial_limit")
    regular = graph.add_node("regular", "Op")
    serial_nodes = [graph.add_node(f"s{i}", "Op") for i in range(3)]
    for index, node in enumerate(serial_nodes):
        node.set_stream_label(f"label-{index}")
    ids, required = _run(graph, max_stream_id=0, base=4)
    assert ids[regular._index] == 4
    assert [ids[node._index] for node in serial_nodes] == [5, 6, 7]
    assert required == 4


def test_zero_cost_nonserial_node_is_excluded():
    graph = multistream.DAGGraph("zero")
    active = graph.add_node("active", "Op")
    excluded = graph.add_node("excluded", "Op")
    excluded.set_node_cost(multistream.NodeCost(execution_time=0.0))
    ids, _ = _run(graph)
    assert ids[active._index] >= 0
    assert ids[excluded._index] == multistream.INVALID_STREAM_ID


@pytest.mark.parametrize("execution_time", [0.0, -0.0, 1e-7, -1e-7])
def test_costs_inside_zero_epsilon_are_excluded(execution_time):
    graph = multistream.DAGGraph("epsilon")
    node = graph.add_node("node", "Op")
    node.set_node_cost(multistream.NodeCost(execution_time=execution_time))
    ids, required = _run(graph)
    assert ids == [multistream.INVALID_STREAM_ID]
    assert required == 0


@pytest.mark.parametrize("execution_time", [-1.0, 1e-5, 2.0])
def test_costs_outside_zero_epsilon_are_allocated(execution_time):
    graph = multistream.DAGGraph("epsilon")
    node = graph.add_node("node", "Op")
    node.set_node_cost(multistream.NodeCost(execution_time=execution_time))
    ids, required = _run(graph)
    assert ids == [0]
    assert required == 1


def test_weighted_strategy_reacts_to_node_cost_changes():
    graph = multistream.DAGGraph("weighted_costs")
    nodes = [graph.add_node(str(index), "Op") for index in range(4)]
    graph.add_edge(nodes[0], 0, nodes[2], 0)
    graph.add_edge(nodes[1], 0, nodes[3], 0)
    for node in nodes:
        node.set_node_cost(
            multistream.NodeCost(execution_time=1.0, cube_block_num=1, vec_block_num=1)
        )
    first, _ = _run(graph, max_stream_id=1, weighted=True)
    nodes[1].set_node_cost(
        multistream.NodeCost(execution_time=10.0, cube_block_num=1, vec_block_num=1)
    )
    nodes[3].set_node_cost(
        multistream.NodeCost(execution_time=10.0, cube_block_num=1, vec_block_num=1)
    )
    second, _ = _run(graph, max_stream_id=1, weighted=True)
    assert first != second


def test_cycle_and_foreign_edge_abort_before_ids_are_produced(caplog):
    graph = multistream.DAGGraph("cycle")
    a = graph.add_node("a", "Op")
    b = graph.add_node("b", "Op")
    graph.add_edge(a, 0, b, 0)
    graph.add_edge(b, 0, a, 0)
    assert not _is_dag(graph)
    caplog.set_level("ERROR", logger="multistream")
    with pytest.raises(RuntimeError):
        _run(graph)
    errors = [
        (record.filename, record.getMessage())
        for record in caplog.records
        if record.levelname == "ERROR"
    ]
    assert errors == [
        (
            "stream_allocator.py",
            "graph contains a cycle; stream allocation requires a DAG",
        )
    ]

    owner = multistream.DAGGraph("owner")
    src = owner.add_node("src", "Op")
    foreign = multistream.DAGGraph("foreign").add_node("foreign", "Op")
    # Deliberately bypass add_edge's ownership check to exercise the boundary.
    from multistream.minidag.node import _Edge

    edge = _Edge(src, 0, foreign, 0)
    src._add_output_edge(edge)
    assert not _is_dag(owner)
    with pytest.raises(RuntimeError):
        _run(owner)


def test_weighted_strategy_uses_explicit_node_cost_without_profiling(
    multi_branch_graph,
):
    graph = multi_branch_graph(width=2, depth=2)
    for node in graph.get_nodes():
        node.set_node_cost(
            multistream.NodeCost(
                execution_time=10.0 if node._name.endswith("_0") else 1.0
            )
        )
    ids, required = _run(graph, weighted=True)
    assert len(ids) == len(graph.get_nodes())
    assert 1 <= required <= 8


def test_residual_edges_bridge_excluded_and_serial_nodes():
    graph = multistream.DAGGraph("residual")
    src = graph.add_node("src", "Op")
    zero = graph.add_node("zero", "Op")
    serial = graph.add_node("serial", "Op")
    dst = graph.add_node("dst", "Op")
    zero.set_node_cost(multistream.NodeCost(execution_time=0.0))
    serial.set_stream_label("s")
    graph.add_edge(src, 0, zero, 0)
    graph.add_edge(zero, 0, serial, 0)
    graph.add_edge(serial, 0, dst, 0)
    graph.add_edge(src, 1, dst, 1)
    split = _split_serial_graph(graph, _costs(graph))
    edges = {
        (edge._src_node._name, edge._dst_node._name)
        for edge in split.residual_graph._get_edges()
    }
    assert edges == {("src", "dst")}
    assert all(src_name != dst_name for src_name, dst_name in edges)


def test_residual_nodes_and_serial_groups_follow_insertion_order():
    graph = multistream.DAGGraph("order")
    graph.add_node("first", "Op")
    serial_a = graph.add_node("sa", "Op")
    serial_b = graph.add_node("sb", "Op")
    serial_a.set_stream_label("same")
    serial_b.set_stream_label("same")
    graph.add_node("second", "Op")
    split = _split_serial_graph(graph, _costs(graph))
    assert [node._name for node in split.residual_graph.get_nodes()] == [
        "first",
        "second",
    ]
    assert [node._name for node in split.serial_groups["same"]] == ["sa", "sb"]
