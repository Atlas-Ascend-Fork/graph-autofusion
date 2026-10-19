# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Public API and value-object contract tests, plan §4.10."""

import pathlib
import sys

import pytest

import multistream
from multistream.minidag.types import _cost_of


def test_public_exports_and_constants():
    assert set(multistream.__all__) == {
        "DAGGraph",
        "DAGNode",
        "NodeCost",
        "CoreType",
        "StreamPlan",
        "MergeStrategy",
        "plan",
    }
    assert multistream.INVALID_STREAM_ID == -1
    assert multistream.MAX_STREAM_NUM == 64


def test_object_method_surface_is_pinned():
    # Every setter has a matching getter so a caller can print what it built.
    # Edges stay private: neighbours are read as nodes.  Pin the surface so a
    # new public method is a deliberate decision rather than a side effect.
    def public_methods(obj):
        return {name for name in dir(obj) if not name.startswith("_")}

    assert public_methods(multistream.DAGGraph) == {
        "add_node",
        "add_edge",
        "find_node",
        "get_name",
        "get_nodes",
        "get_node_count",
    }
    assert public_methods(multistream.DAGNode) == {
        "get_name",
        "get_type",
        "get_index",
        "set_stream_label",
        "get_stream_label",
        "set_node_cost",
        "get_node_cost",
        "set_core_type",
        "get_core_type",
    }
    assert public_methods(multistream.StreamPlan) == {
        "get_stream_id",
        "get_required_stream_count",
        "get_core_num",
    }


def test_merge_strategy_serialization_and_aliases():
    assert [member.value for member in multistream.MergeStrategy] == [
        "LoadBalance",
        "MainStream",
    ]
    assert (
        multistream.MergeStrategy("LoadBalance")
        is multistream.MergeStrategy.LOAD_BALANCE
    )
    assert (
        multistream.MergeStrategy("main stream")
        is multistream.MergeStrategy.MAIN_STREAM
    )
    # The weighted merger is reached through profiling data only, so no
    # public spelling may select it.
    for value in ("WeightedLoadBalance", "weighted load balance"):
        with pytest.raises(ValueError):
            multistream.MergeStrategy(value)
    with pytest.raises(ValueError):
        multistream.MergeStrategy("bad")
    with pytest.raises(ValueError):
        multistream.MergeStrategy(1)


def test_core_type_members_and_values():
    assert [member.name for member in multistream.CoreType] == [
        "AIC",
        "MIX_AIC",
        "AIV",
        "MIX_AIV",
        "UNKNOWN",
    ]
    # Unlike MergeStrategy this type never reaches a GE option, so the value
    # simply repeats the member name.
    assert all(member.value == member.name for member in multistream.CoreType)


def test_node_cost_is_immutable_value_object():
    fields = dict(
        execution_time=1.25,
        memory_usage=3,
        bandwidth_usage=4,
        cube_block_num=5,
        vec_block_num=6,
        cube_core_num_limit=7,
        vec_core_num_limit=8,
    )
    cost = multistream.NodeCost(**fields)
    assert tuple(getattr(cost, field) for field in cost.__slots__) == (
        1.25,
        3,
        4,
        5,
        6,
        7,
        8,
    )
    assert cost == multistream.NodeCost(**fields)
    assert hash(cost) == hash(multistream.NodeCost(**fields))
    assert "execution_time" in repr(cost)
    for operation in (
        lambda: setattr(cost, "execution_time", 2),
        lambda: delattr(cost, "execution_time"),
    ):
        with pytest.raises(AttributeError):
            operation()
    with pytest.raises(TypeError):
        multistream.NodeCost(1)
    with pytest.raises(TypeError):
        multistream.NodeCost(unknown=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1e100])
def test_node_cost_rejects_nonfinite_or_float32_overflow(value):
    with pytest.raises((TypeError, ValueError)):
        multistream.NodeCost(execution_time=value)


def test_node_cost_defaults_cover_the_new_fields():
    cost = multistream.NodeCost()
    # 0 means the user set no limit.
    assert cost.cube_core_num_limit == 0
    assert cost.vec_core_num_limit == 0
    # Removed fields must be rejected, not silently ignored.
    for removed in ("compute_cycles", "core_type", "is_core_num_limited"):
        assert not hasattr(cost, removed)
        with pytest.raises(TypeError):
            multistream.NodeCost(**{removed: 1})


@pytest.mark.parametrize("value", ["AIC", 0, None, multistream.CoreType])
def test_core_type_accepts_only_enum_members(value):
    # Converting a string from GE or Inductor metadata is the adapter's job.
    node = multistream.DAGGraph("g").add_node("n", "Op")
    with pytest.raises(TypeError):
        node.set_core_type(value)
    assert node.get_core_type() is multistream.CoreType.UNKNOWN


@pytest.mark.parametrize(
    "field",
    [
        "memory_usage",
        "bandwidth_usage",
        "cube_block_num",
        "vec_block_num",
        "cube_core_num_limit",
        "vec_core_num_limit",
    ],
)
def test_node_cost_integer_fields_reject_negative_bool_and_non_int(field):
    for value in (-1, True, 1.5, "1"):
        with pytest.raises((TypeError, ValueError)):
            multistream.NodeCost(**{field: value})


def test_execution_time_is_rounded_to_float32():
    value = 1.0 / 3.0
    cost = multistream.NodeCost(execution_time=value)
    assert cost.execution_time != value
    assert cost.execution_time == pytest.approx(value, rel=1e-7)


def test_internal_package_has_no_bare_asserts_or_third_party_dependencies():
    package_dir = pathlib.Path(multistream.__file__).parent
    # Glob the sub-packages one by one.  In a source checkout the package root
    # also holds the tests and the CI scripts, which a recursive glob would
    # pull in.
    sources = [package_dir / "__init__.py"]
    for sub_package in ("common", "minidag", "algorithm"):
        sources.extend(sorted((package_dir / sub_package).glob("*.py")))
    assert len(sources) > 12
    assert not any(
        "assert " in line and line.lstrip().startswith("assert ")
        for path in sources
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    assert not any(name in sys.modules for name in ("numpy", "scipy"))


def test_cost_of_rejects_out_of_range_index():
    graph = multistream.DAGGraph("g")
    node = graph.add_node("n", "Op")
    with pytest.raises(IndexError):
        _cost_of(node, [])
