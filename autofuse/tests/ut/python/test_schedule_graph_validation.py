# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions, Schedule


@pytest.fixture(autouse=True)
def schedule_platform(tmp_path, monkeypatch):
    # 调度和报错不需要真实设备；隔离图 dump 等测试产物。
    monkeypatch.chdir(tmp_path)
    ascir.utils.set_platform("2201", 1, 245760)


def make_reduce_graph(name, output_shape=(4, 1), empty_schedule=False):
    """构造沿第二轴归约的完整图；合法输出保留两条轴，长度应为 [4, 1]。"""
    graph = ascir.HintGraph(name)
    a = graph.create_axis("a", 4)
    r = graph.create_axis("r", 8)
    axes = [a, r]
    data = ascir.ops.Data("data", graph)
    data.attr.ir_attr.index = 0
    data.y.dtype = ascir.dtypes.float16
    data.y.axis = axes
    data.y.size = [4, 8]
    data.y.strides = [8, 1]
    load = ascir.ops.Load("load")
    load.x = data.y
    load.attr.sched.axis = axes
    load.y.dtype = ascir.dtypes.float16
    load.y.axis = axes
    load.y.size = [4, 8]
    load.y.strides = [8, 1]
    reduce = ascir.ops.Max("reduce")
    reduce.x = load.y
    reduce.attr.sched.axis = [] if empty_schedule else axes
    reduce.y.dtype = ascir.dtypes.float16
    reduce.y.axis = axes
    reduce.y.size = list(output_shape)
    reduce.y.strides = [1, 0]
    store = ascir.ops.Store("store")
    store.x = reduce.y
    store.attr.sched.axis = axes
    store.y.dtype = ascir.dtypes.float16
    store.y.axis = axes
    store.y.size = list(output_shape)
    store.y.strides = [1, 0]
    output = ascir.ops.Output("output", graph)
    output.attr.ir_attr.index = 0
    output.x = store.y
    return graph


def entrypoint(name):
    if name == "schedule_v1":
        return Schedule(AutofuserOptions()).schedule
    fuser = Autofuser(AutofuserOptions())
    return getattr(fuser, name)


@pytest.mark.parametrize("entry", ["schedule", "autofuse_backend", "schedule_v1"])
@pytest.mark.parametrize("empty_schedule", [False, True])
def test_invalid_reduce_reports_shape_error_before_codegen(entry, empty_schedule):
    run = entrypoint(entry)
    # 同一入口连续失败两次，第二次必须报告本次根因，不能残留第一次的诊断。
    cases = [((5, 1), "non-reduced axis length changed"), ((4, 2), "expected=1")]
    for shape, detail in cases:
        graph = make_reduce_graph("invalid_reduce_shape", shape, empty_schedule)
        with pytest.raises(RuntimeError) as error:
            run(graph)
        message = str(error.value)
        assert "[AF][SCHEDULE GRAPH VALIDATION ERROR][REDUCE_AXIS]" in message
        assert "stage=schedule" in message
        assert "node=reduce" in message
        assert detail in message


@pytest.mark.parametrize("entry", ["schedule", "schedule_v1"])
def test_valid_reduce_still_schedules_after_failure(entry):
    run = entrypoint(entry)
    with pytest.raises(
        RuntimeError, match=r"\[AF\]\[SCHEDULE GRAPH VALIDATION ERROR\]\[REDUCE_AXIS\]"
    ):
        run(make_reduce_graph("invalid_before_valid", (4, 2)))
    result = run(make_reduce_graph("valid_reduce_shape"))
    assert result is not None
    assert result.get_input_num() == 1
    assert result.get_output_num() == 1
