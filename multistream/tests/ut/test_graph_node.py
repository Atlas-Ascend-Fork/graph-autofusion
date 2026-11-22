# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DAGGraph / DAGNode tests (test plan §4.3, 15 baseline cases in 10 groups)."""

import pytest

import multistream


def test_add_and_find_nodes_preserve_dense_insertion_order():
    graph = multistream.DAGGraph("g")
    first = graph.add_node("first", "Relu")
    graph.add_node("second", "Conv2D")
    assert isinstance(first, multistream.DAGNode)
    assert (first._name, first._type, first._index) == (
        "first",
        "Relu",
        0,
    )
    assert graph.find_node("first") is first
    assert graph.find_node("missing") is None
    assert [node._name for node in graph.get_nodes()] == ["first", "second"]
    assert len(graph.get_nodes()) == 2


def test_duplicate_or_invalid_nodes_are_rejected_without_mutation():
    graph = multistream.DAGGraph("g")
    graph.add_node("n", "Op")
    assert graph.add_node("n", "Other") is None
    assert len(graph.get_nodes()) == 1
    for name, node_type in ((1, "Op"), ("n2", 2), (None, "Op")):
        with pytest.raises(TypeError):
            graph.add_node(name, node_type)
    with pytest.raises(TypeError):
        graph.find_node(1)


def test_edges_are_recorded_in_graph_and_node_views():
    graph = multistream.DAGGraph("g")
    src = graph.add_node("src", "Op")
    dst = graph.add_node("dst", "Op")
    graph.add_edge(src, 0, dst, 1)
    graph.add_edge(src, 1, dst, 2)
    assert len(graph._get_edges()) == 2
    assert len(src._get_output_edges()) == 2
    assert len(dst._get_input_edges()) == 2
    assert [edge._src_port for edge in dst._get_input_edges()] == [0, 1]
    assert src._get_output_edges()[0]._dst_node is dst
    assert dst._get_input_nodes() == [src, src]
    # Accessors return copies; mutating one cannot remove an edge.
    edges = graph._get_edges()
    edges.clear()
    assert len(graph._get_edges()) == 2


def test_control_edges_use_minus_one_and_mixed_edges_keep_order():
    graph = multistream.DAGGraph("g")
    src = graph.add_node("src", "Op")
    dst = graph.add_node("dst", "Op")
    graph.add_edge(src, -1, dst, -1)
    edge = graph._get_edges()[0]
    assert edge._src_port == edge._dst_port == -1
    graph.add_edge(src, 0, dst, 0)
    assert [edge._src_port for edge in src._get_output_edges()] == [-1, 0]


@pytest.mark.parametrize("src_port,dst_port", [(-2, 0), (0, -2), (-2, -2)])
def test_invalid_ports_are_atomic(src_port, dst_port):
    graph = multistream.DAGGraph("g")
    src = graph.add_node("src", "Op")
    dst = graph.add_node("dst", "Op")
    with pytest.raises(ValueError):
        graph.add_edge(src, src_port, dst, dst_port)
    assert graph._get_edges() == []


@pytest.mark.parametrize("port", [True, False, 1.0, "0", None])
def test_non_integer_ports_are_rejected(port):
    graph = multistream.DAGGraph("g")
    src = graph.add_node("src", "Op")
    dst = graph.add_node("dst", "Op")
    with pytest.raises(TypeError):
        graph.add_edge(src, port, dst, 0)
    assert graph._get_edges() == []


@pytest.mark.parametrize("endpoint", [None, object(), "node"])
def test_edge_endpoints_must_be_nodes(endpoint):
    graph = multistream.DAGGraph("g")
    node = graph.add_node("node", "Op")
    with pytest.raises(TypeError):
        graph.add_edge(endpoint, 0, node, 0)
    with pytest.raises(TypeError):
        graph.add_edge(node, 0, endpoint, 0)
    assert graph._get_edges() == []


def test_cross_graph_endpoint_is_rejected_atomically():
    graph = multistream.DAGGraph("g")
    src = graph.add_node("src", "Op")
    other = multistream.DAGGraph("other").add_node("dst", "Op")
    with pytest.raises(ValueError):
        graph.add_edge(src, 0, other, 0)
    assert graph._get_edges() == []


def test_stream_labels_and_costs_are_independent_copies():
    graph = multistream.DAGGraph("g")
    first = graph.add_node("first", "Op")
    second = graph.add_node("second", "Op")
    assert first.get_stream_label() == ""
    first.set_stream_label("serial")
    assert first.get_stream_label() == "serial"
    with pytest.raises(TypeError):
        first.set_stream_label(1)
    cost = multistream.NodeCost(execution_time=3.5, cube_block_num=8)
    first.set_node_cost(cost)
    assert first.get_node_cost() == cost
    assert first.get_node_cost() is not cost
    assert second.get_node_cost() == multistream.NodeCost()
    with pytest.raises(TypeError):
        first.set_node_cost(object())


def test_graph_name_and_ownership_contract():
    assert multistream.DAGGraph("name").get_name() == "name"
    with pytest.raises(TypeError):
        multistream.DAGGraph(None)
    graph = multistream.DAGGraph("g")
    node = graph.add_node("n", "Op")
    assert graph._owns_node(node)
    assert not graph._owns_node(object())
    assert not graph._owns_node(multistream.DAGGraph("other").add_node("n", "Op"))
