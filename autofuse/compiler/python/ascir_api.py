#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Dict, Union, List, Tuple, Callable
from autofuse.pyautofuse import ascir

# 全局状态管理
_graph_metadata = {}  # 格式: {graph_id: GraphMetadata}


class GraphMetadata:
    __slots__ = ["op_counters", "data_indices", "output_indices", "ops"]

    def __init__(self):
        # 算子类型计数器：{"op_type": count}
        self.op_counters: Dict[str, int] = {}
        self.data_indices = 0
        self.output_indices = 0
        self.ops: List[ascir.Operator] = []

    def get_counter(self, op_type: str) -> int:
        """获取并递增指定算子类型的计数器"""
        cnt = self.op_counters.get(op_type, 0)
        self.op_counters[op_type] = cnt + 1
        return cnt


def _get_metadata(graph: ascir.HintGraph) -> GraphMetadata:
    graph_id = graph.name
    if graph_id not in _graph_metadata:
        _graph_metadata[graph_id] = GraphMetadata()
    return _graph_metadata[graph_id]


def _derive_strides(size: List[ascir.SizeExpr]) -> List[ascir.SizeExpr]:
    """根据 size 推导连续内存的 strides"""
    stride = 1
    derived_strides = []
    for dim in reversed(size):
        derived_strides.insert(0, stride)
        stride *= dim
    return derived_strides


def _derive_sizes_and_strides(
    axis: List[ascir.Axis],
) -> (List[ascir.SizeExpr], List[ascir.SizeExpr]):
    """根据 axis 推导连续内存的 sizes and strides"""
    tmp_repeats = []
    tmp_strides = []
    for tmp_axis in reversed(axis):
        if not tmp_strides:
            tmp_strides.append(1)
        else:
            tmp_strides.append(tmp_repeats[-1] * tmp_strides[-1])
        tmp_repeats.append(tmp_axis.size)
    return list(reversed(tmp_repeats)), list(reversed(tmp_strides))


def _infer_or_set_view(view_holder: ascir.OpsOperatorOutput, axis, size, stride):
    view_holder.axis = axis

    if size is not None:
        if len(size) != len(axis):
            raise ValueError("size len should be same with axis len")
        view_holder.size = size

    if stride is not None:
        if len(stride) != len(axis):
            raise ValueError("stride should be same with axis len")
        view_holder.strides = stride

    if size is None and stride is None:
        view_holder.size, view_holder.strides = _derive_sizes_and_strides(axis)
    elif size is not None and stride is None:
        view_holder.strides = _derive_strides(size)
    elif size is None and stride is not None:
        raise ValueError("when stride is given, size must also be given")


def _generate_op_name(graph: ascir.HintGraph, op_type: str) -> str:
    """生成唯一算子名称（如 data_0, load_1）"""
    meta = _get_metadata(graph)
    cnt = meta.get_counter(op_type)
    return f"{op_type}_{cnt}"


def _common_in_1_out_1_normal_op(
    op_type: str,
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    # 实例化并保存
    name = _generate_op_name(owner_graph, op_type.lower())
    op_class = getattr(ascir.ops, op_type)  # 动态类型推导, 返回例如ascir.ops.Abs
    op_instance_1_in_2_out = op_class(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance_1_in_2_out)
    op_instance_1_in_2_out.attr.sched.axis = axis
    op_instance_1_in_2_out.x = x
    _infer_or_set_view(op_instance_1_in_2_out.y, axis, size, stride)
    op_instance_1_in_2_out.infer_dtype()
    return op_instance_1_in_2_out.y


def _common_dynamic_in_1_out_1_normal_op(
    op_type: str,
    owner_graph: ascir.HintGraph,
    x: List[ascir.OpsOperatorOutput],
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, op_type.lower())
    op_class = getattr(ascir.ops, op_type)  # 动态类型推导, 返回例如ascir.ops.Concat
    op_instance = op_class(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.x = x
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def _common_in_2_out_1_normal_op(
    op_type: str,
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, op_type.lower())
    op_class = getattr(ascir.ops, op_type)  # 动态类型推导, 返回例如ascir.ops.Add
    op_instance_2_in_1_out = op_class(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance_2_in_1_out)
    op_instance_2_in_1_out.attr.sched.axis = axis
    op_instance_2_in_1_out.x1 = x1
    op_instance_2_in_1_out.x2 = x2
    _infer_or_set_view(op_instance_2_in_1_out.y, axis, size, stride)
    op_instance_2_in_1_out.infer_dtype()
    return op_instance_2_in_1_out.y


def _common_in_3_out_1_normal_op(
    op_type: str,
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    x3: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, op_type.lower())
    op_class = getattr(ascir.ops, op_type)  # 动态类型推导, 返回例如ascir.ops.Select
    op_instance_3_in_1_out = op_class(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance_3_in_1_out)
    op_instance_3_in_1_out.attr.sched.axis = axis
    op_instance_3_in_1_out.x1 = x1
    op_instance_3_in_1_out.x2 = x2
    op_instance_3_in_1_out.x3 = x3
    _infer_or_set_view(op_instance_3_in_1_out.y, axis, size, stride)
    op_instance_3_in_1_out.infer_dtype()
    return op_instance_3_in_1_out.y


def Data(
    owner_graph: ascir.HintGraph, *, dtype: ascir.dtypes
) -> ascir.OpsOperatorOutput:
    meta = _get_metadata(owner_graph)

    name = _generate_op_name(owner_graph, "data")

    data_op = ascir.ops.Data(name, owner_graph)
    meta.ops.append(data_op)
    index = meta.data_indices
    meta.data_indices += 1
    data_op.attr.ir_attr.index = index

    data_op.y.dtype = dtype
    data_op.infer_dtype()
    return data_op.y


def Scalar(
    owner_graph: ascir.HintGraph, *, dtype: ascir.dtypes, value: str
) -> ascir.OpsOperatorOutput:
    meta = _get_metadata(owner_graph)
    name = _generate_op_name(owner_graph, "scalar")
    op = ascir.ops.Scalar(name, owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.value = value
    op.y.dtype = dtype
    op.infer_dtype()
    return op.y


def ScalarData(
    owner_graph: ascir.HintGraph, *, dtype: ascir.dtypes, value: str
) -> ascir.OpsOperatorOutput:
    meta = _get_metadata(owner_graph)
    name = _generate_op_name(owner_graph, "scalardata")
    op = ascir.ops.ScalarData(name, owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.value = value
    op.y.dtype = dtype
    op.infer_dtype()
    return op.y


def Workspace(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Workspace", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Output(
    owner_graph: ascir.HintGraph,
    x: Union[ascir.OpsOperatorOutput, List[ascir.OpsOperatorOutput]],
    *,
    dtype=None,
):
    meta = _get_metadata(owner_graph)
    name = _generate_op_name(owner_graph, "output")
    op = ascir.ops.Output(name)
    meta.ops.append(op)
    op.x = x
    index = meta.output_indices
    meta.output_indices += 1
    op.attr.ir_attr.index = index
    if dtype is not None:
        op.y.dtype = dtype
    op.infer_dtype()
    return op.y


def IndexExpr(
    owner_graph: ascir.HintGraph,
    *,
    dtype: ascir.dtypes,
    expr: ascir.SizeExpr,
) -> ascir.OpsOperatorOutput:
    if not isinstance(expr, ascir.SizeExpr):
        raise TypeError("IndexExpr expr must be SizeExpr")
    meta = _get_metadata(owner_graph)
    name = _generate_op_name(owner_graph, "indexexpr")
    op = ascir.ops.IndexExpr(name, owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.expr = expr
    op.y.dtype = dtype
    op.infer_dtype()
    return op.y


def arange(
    owner_graph: ascir.HintGraph,
    *,
    dtype: ascir.dtypes,
    base: ascir.SizeExpr,
    step: ascir.SizeExpr,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    if not isinstance(base, ascir.SizeExpr) or not isinstance(step, ascir.SizeExpr):
        raise TypeError("Arange base and step must be SizeExpr")
    if not axis:
        raise ValueError("Arange axis must not be empty")
    if stride is not None and len(stride) != len(axis):
        raise ValueError("stride should be same with axis len")
    if stride is not None:
        last_nonzero = next(
            (
                (index, value)
                for index, value in reversed(list(enumerate(stride)))
                if not (value == ascir.SizeExpr(0))
            ),
            None,
        )
        if last_nonzero is not None and not (last_nonzero[1] == ascir.SizeExpr(1)):
            raise ValueError(
                "Arange requires unit physical stride on the vectorized axis"
            )
        effective_size = size if size is not None else [item.size for item in axis]
        if any(
            value == ascir.SizeExpr(0) and not (dimension == ascir.SizeExpr(1))
            for dimension, value in zip(effective_size, stride)
        ):
            raise ValueError(
                "Arange does not support non-singleton zero strides; "
                "use 1D Arange followed by explicit Broadcast"
            )
    meta = _get_metadata(owner_graph)
    name = _generate_op_name(owner_graph, "arange")
    op = ascir.ops.Arange(name, owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.base = base
    op.attr.ir_attr.step = step
    op.y.dtype = dtype
    _infer_or_set_view(op.y, axis, size, stride)
    op.attr.sched.axis = axis
    op.infer_dtype()
    return op.y


def Load(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    offset: Optional[ascir.SizeExpr] = None,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "load")
    load_op = ascir.ops.Load(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(load_op)

    if offset is not None:
        load_op.attr.ir_attr.offset = offset
    load_op.attr.sched.axis = axis
    load_op.x = x
    _infer_or_set_view(load_op.y, axis, size, stride)
    load_op.infer_dtype()
    return load_op.y


def Broadcast(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Broadcast", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Store(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    offset: Optional[ascir.SizeExpr] = None,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "store")
    op = ascir.ops.Store(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)

    if offset is not None:
        op.attr.ir_attr.offset = offset
    op.attr.sched.axis = axis
    op.x = x
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return op.y


def Cast(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    dtype: ascir.dtypes,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "cast")
    op = ascir.ops.Cast(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)
    op.attr.sched.axis = axis
    op.x = x
    op.y.dtype = dtype
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return op.y


def Abs(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Abs", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Exp(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Exp", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Exp2(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Exp2", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Floor(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Floor", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Fma(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    x3: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_3_out_1_normal_op(
        "Fma", owner_graph, x1, x2, x3, axis=axis, size=size, stride=stride
    )


def BitwiseNot(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "BitwiseNot", owner_graph, x, axis=axis, size=size, stride=stride
    )


def BitwiseOr(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "BitwiseOr", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def BitwiseXor(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "BitwiseXor", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Ceil(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Ceil", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Ceil2Int(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Ceil2Int", owner_graph, x, axis=axis, size=size, stride=stride
    )


def FloorToInt(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "FloorToInt", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Lgamma(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Lgamma", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Cos(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Cos", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Acos(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Acos", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Cosh(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Cosh", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Atan2(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Atan2", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def CopySign(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "CopySign", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Sqrt(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Sqrt", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Rsqrt(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Rsqrt", owner_graph, x, axis=axis, size=size, stride=stride
    )


def RemovePad(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "RemovePad", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Pad(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Pad", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Round(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Round", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Neg(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Neg", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Relu(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Relu", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Reciprocal(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Reciprocal", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Erf(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Erf", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Erfcx(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Erfcx", owner_graph, x, axis=axis, size=size, stride=stride
    )


def ModifiedBesselI0(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "ModifiedBesselI0", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Remainder(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Remainder", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def ModifiedBesselI1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "ModifiedBesselI1", owner_graph, x, axis=axis, size=size, stride=stride
    )


def ModifiedBesselK0(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "ModifiedBesselK0", owner_graph, x, axis=axis, size=size, stride=stride
    )


def ModifiedBesselK1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "ModifiedBesselK1", owner_graph, x, axis=axis, size=size, stride=stride
    )


def BesselJ0(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "BesselJ0", owner_graph, x, axis=axis, size=size, stride=stride
    )


def BesselJ1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "BesselJ1", owner_graph, x, axis=axis, size=size, stride=stride
    )


def BesselY0(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "BesselY0", owner_graph, x, axis=axis, size=size, stride=stride
    )


def BesselY1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "BesselY1", owner_graph, x, axis=axis, size=size, stride=stride
    )


def ScaledModifiedBesselK0(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "ScaledModifiedBesselK0", owner_graph, x, axis=axis, size=size, stride=stride
    )


def ScaledModifiedBesselK1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "ScaledModifiedBesselK1", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Igamma(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Igamma", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Igammac(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Igammac", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Zeta(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Zeta", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Ndtr(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Ndtr", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Ndtri(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Ndtri", owner_graph, x, axis=axis, size=size, stride=stride
    )


def SphericalBesselJ0(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "SphericalBesselJ0", owner_graph, x, axis=axis, size=size, stride=stride
    )


def LogNdtr(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "LogNdtr", owner_graph, x, axis=axis, size=size, stride=stride
    )


def _reject_bucketize_sorter(sorter, sorter_indices) -> None:
    if sorter is not None or sorter_indices is not None:
        raise ValueError(
            "Bucketize AscIR does not support sorter; boundaries must already be sorted"
        )


def _is_int32_indexing_dtype(indexing_dtype) -> bool:
    if indexing_dtype is None:
        return True
    if indexing_dtype in ("int32", "torch.int32", "DT_INT32"):
        return True
    name = getattr(indexing_dtype, "name", None) or getattr(
        indexing_dtype, "__name__", None
    )
    text = str(indexing_dtype)
    return name in ("int32", "DT_INT32") or text in ("int32", "torch.int32", "DT_INT32")


def Bucketize(
    owner_graph: ascir.HintGraph,
    values: ascir.OpsOperatorOutput,
    boundaries: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
    right: bool = False,
    indexing_dtype=None,
    boundary_indices: Optional[ascir.OpsOperatorOutput] = None,
    sorter=None,
    sorter_indices: Optional[ascir.OpsOperatorOutput] = None,
) -> ascir.OpsOperatorOutput:
    """Convert inductor bucketize into an AscIR node.

    values and boundaries are graph tensors. boundaries keeps its own shape and
    is binary-searched, not broadcast onto values. The returned tensor is the
    int32 bucket index. boundary_indices is accepted for signature compatibility
    and is not an extra input: the kernel writes the indices itself.
    """
    del boundary_indices
    _reject_bucketize_sorter(sorter, sorter_indices)
    if not _is_int32_indexing_dtype(indexing_dtype):
        raise ValueError("Bucketize AscIR only supports indexing_dtype int32")
    name = _generate_op_name(owner_graph, "bucketize")
    op_instance = ascir.ops.Bucketize(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.ir_attr.right = bool(right)
    op_instance.attr.sched.axis = axis
    op_instance.x1 = values
    op_instance.x2 = boundaries
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def NextAfter(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "NextAfter", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def PolyGamma(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "PolyGamma", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def SignBit(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "SignBit", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Frexp(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> Tuple[ascir.OpsOperatorOutput, ascir.OpsOperatorOutput]:
    name = _generate_op_name(owner_graph, "frexp")
    op_instance = ascir.ops.Frexp(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.x = x
    _infer_or_set_view(op_instance.mantissa, axis, size, stride)
    _infer_or_set_view(op_instance.exponent, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.mantissa, op_instance.exponent


def _shifted_chebyshev_polynomial_op(
    op_type: str,
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(
        owner_graph,
        op_type[0].lower()
        + "".join(["_" + c.lower() if c.isupper() else c for c in op_type[1:]]),
    )
    op_instance = getattr(ascir.ops, op_type)(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.attr.ir_attr.n = n
    op_instance.x = x
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def ShiftedChebyshevPolynomialT(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ShiftedChebyshevPolynomialT",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def ShiftedChebyshevPolynomialU(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ShiftedChebyshevPolynomialU",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def ShiftedChebyshevPolynomialV(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ShiftedChebyshevPolynomialV",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def ShiftedChebyshevPolynomialW(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ShiftedChebyshevPolynomialW",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def ChebyshevPolynomialT(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ChebyshevPolynomialT",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def ChebyshevPolynomialU(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ChebyshevPolynomialU",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def ChebyshevPolynomialV(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ChebyshevPolynomialV",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def ChebyshevPolynomialW(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: int,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _shifted_chebyshev_polynomial_op(
        "ChebyshevPolynomialW",
        owner_graph,
        x,
        n,
        axis=axis,
        size=size,
        stride=stride,
    )


def HermitePolynomialH(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "hermite_polynomial_h")
    op_instance = ascir.ops.HermitePolynomialH(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.x = x
    op_instance.n = n
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def HermitePolynomialHe(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "hermite_polynomial_he")
    op_instance = ascir.ops.HermitePolynomialHe(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.x = x
    op_instance.n = n
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def LaguerrePolynomialL(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "laguerre_polynomial_l")
    op_instance = ascir.ops.LaguerrePolynomialL(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.x = x
    op_instance.n = n
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def LegendrePolynomialP(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    n: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "legendre_polynomial_p")
    op_instance = ascir.ops.LegendrePolynomialP(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.x = x
    op_instance.n = n
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def AiryAi(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "AiryAi", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Erfinv(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Erfinv", owner_graph, x, axis=axis, size=size, stride=stride
    )


def FloorDiv(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "FloorDiv", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Sign(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Sign", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Tanh(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Tanh", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Sin(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Sin", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Asin(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Asin", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Asinh(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Asinh", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Atan(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Atan", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Atanh(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Atanh", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Digamma(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Digamma", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Erfc(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Erfc", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Acosh(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Acosh", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Isnan(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Isnan", owner_graph, x, axis=axis, size=size, stride=stride
    )


def IsInf(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "IsInf", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Max(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Max", owner_graph, x, axis=axis, size=size, stride=stride
    )


def ArgMax(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "ArgMax", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Softmax(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Softmax", owner_graph, x, axis=axis, size=size, stride=stride
    )


def ArgMaxMultiRPhase1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> tuple:
    """ArgMax R轴分核 Phase1: 1输入2输出（value, index）"""
    name = _generate_op_name(owner_graph, "argmaxmultirphase1")
    op_class = getattr(ascir.ops, "ArgMaxMultiRPhase1")
    op_instance = op_class(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.x = x
    _infer_or_set_view(op_instance.value, axis, size, stride)
    _infer_or_set_view(op_instance.index, axis, size, stride)
    op_instance.infer_dtype()
    return (op_instance.value, op_instance.index)


def ArgMaxMultiRPhase2(
    owner_graph: ascir.HintGraph,
    value: ascir.OpsOperatorOutput,
    index: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    """ArgMax R轴分核 Phase2: 2输入1输出"""
    name = _generate_op_name(owner_graph, "argmaxmultirphase2")
    op_class = getattr(ascir.ops, "ArgMaxMultiRPhase2")
    op_instance = op_class(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op_instance)
    op_instance.attr.sched.axis = axis
    op_instance.value = value
    op_instance.index = index
    _infer_or_set_view(op_instance.y, axis, size, stride)
    op_instance.infer_dtype()
    return op_instance.y


def Any(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Any", owner_graph, x, axis=axis, size=size, stride=stride
    )


def All(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "All", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Sum(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Sum", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Min(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Min", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Mean(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Mean", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Prod(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Prod", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Ge(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Ge", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Ne(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Ne", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Eq(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Eq", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Gt(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Gt", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def RShift(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "RShift", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Le(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Le", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Add(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Add", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Sub(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Sub", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Div(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Div", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Mul(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Mul", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def TrueDiv(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "TrueDiv", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Minimum(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Minimum", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Maximum(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Maximum", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def LogicalOr(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "LogicalOr", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def LogicalNot(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "LogicalNot", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def LogicalAnd(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "LogicalAnd", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def LogicalXor(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "LogicalXor", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Fmod(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Fmod", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Hypot(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Hypot", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Select(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    x3: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_3_out_1_normal_op(
        "Select", owner_graph, x1, x2, x3, axis=axis, size=size, stride=stride
    )


def MaskedFill(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    mask: ascir.OpsOperatorOutput,
    value: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_3_out_1_normal_op(
        "MaskedFill", owner_graph, x, mask, value, axis=axis, size=size, stride=stride
    )


def Sigmoid(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Sigmoid", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Concat(
    owner_graph: ascir.HintGraph,
    x: List[ascir.OpsOperatorOutput],
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_dynamic_in_1_out_1_normal_op(
        "Concat", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Where(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    x3: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_3_out_1_normal_op(
        "Where", owner_graph, x1, x2, x3, axis=axis, size=size, stride=stride
    )


def Gather(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
    negative_index_support: bool = False,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "Gather".lower())
    op = ascir.ops.Gather(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)

    op.attr.ir_attr.negative_index_support = negative_index_support
    op.attr.sched.axis = axis
    op.x1 = x1
    op.x2 = x2
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return op.y


def IndirectLoad(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: int,
    sched_axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "indirectload")
    op = ascir.ops.IndirectLoad(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)

    op.attr.ir_attr.axis = axis
    op.attr.sched.axis = sched_axis
    op.x1 = x1
    op.x2 = x2
    _infer_or_set_view(op.y, sched_axis, size, stride)
    op.infer_dtype()
    return op.y


def BitwiseAnd(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "BitwiseAnd", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Ln(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Ln", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Expm1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Expm1", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Log2(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Log2", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Log10(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Log10", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Log1p(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Log1p", owner_graph, x, axis=axis, size=size, stride=stride
    )


def LShift(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "LShift", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Mod(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Mod", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Lt(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Lt", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Pow(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "Pow", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def ClipByValue(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    x3: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_3_out_1_normal_op(
        "ClipByValue", owner_graph, x1, x2, x3, axis=axis, size=size, stride=stride
    )


def LeakyRelu(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    negative_slope: float,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "LeakyRelu".lower())
    op = ascir.ops.LeakyRelu(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)

    op.attr.ir_attr.negative_slope = negative_slope
    op.attr.sched.axis = axis
    op.x = x
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return op.y


def Nop(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Nop", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Transpose(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Transpose", owner_graph, x, axis=axis, size=size, stride=stride
    )


def IsFinite(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "IsFinite", owner_graph, x, axis=axis, size=size, stride=stride
    )


def I0(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "I0", owner_graph, x, axis=axis, size=size, stride=stride
    )


def I0e(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "I0e", owner_graph, x, axis=axis, size=size, stride=stride
    )


def I1e(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "I1e", owner_graph, x, axis=axis, size=size, stride=stride
    )


def I1(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "I1", owner_graph, x, axis=axis, size=size, stride=stride
    )


def Rand(
    owner_graph: ascir.HintGraph,
    *,
    dtype: ascir.dtypes,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    """
    Rand - generate uniform random numbers using PhiloxRandom algorithm.
    Uses fixed philoxKey = {0, 0} and philoxCounter = {0, 0, 0, 0} for continuous mode.
    Supports float data type.
    """
    meta = _get_metadata(owner_graph)
    name = _generate_op_name(owner_graph, "rand")
    rand_op = ascir.ops.Rand(name, owner_graph)
    meta.ops.append(rand_op)

    # Set dtype and view, then infer
    rand_op.y.dtype = dtype
    _infer_or_set_view(rand_op.y, axis, size, stride)
    rand_op.infer_dtype()

    return rand_op.y


def Randn(
    owner_graph: ascir.HintGraph,
    *,
    dtype: ascir.dtypes,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    """
    Randn - generate normal random numbers using PhiloxRandom algorithm.
    Uses fixed philoxKey = {0, 0} and philoxCounter = {0, 0, 0, 0} for continuous mode.
    Supports uint32_t and int32_t data types.
    """
    meta = _get_metadata(owner_graph)
    name = _generate_op_name(owner_graph, "randn")
    randn_op = ascir.ops.Randn(name, owner_graph)
    meta.ops.append(randn_op)

    # Set dtype and view, then infer
    randn_op.y.dtype = dtype
    _infer_or_set_view(randn_op.y, axis, size, stride)
    randn_op.infer_dtype()

    return randn_op.y


def Identity(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_1_out_1_normal_op(
        "Identity", owner_graph, x, axis=axis, size=size, stride=stride
    )


def DeviceAssertAsync(
    owner_graph: ascir.HintGraph,
    cond: ascir.OpsOperatorOutput,
    msg: str,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "device_assert_async")
    op = ascir.ops.DeviceAssertAsync(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.msg = str(msg)
    op.attr.sched.axis = axis
    op.x = cond
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return op.y


def InlineAsmElementwise(
    owner_graph: ascir.HintGraph,
    *inputs: ascir.OpsOperatorOutput,
    asm: str,
    axis: List[ascir.Axis],
    containts: Optional[str] = None,
    dtype=ascir.dtypes.float32,
    ispure: bool = True,
    packet: int = 1,
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    if not inputs:
        raise ValueError("InlineAsmElementwise requires at least one input")
    name = _generate_op_name(owner_graph, "inline_asm_elementwise")
    op = ascir.ops.InlineAsmElementwise(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.asm_str = asm
    op.attr.ir_attr.containts = "" if containts is None else containts
    op.attr.ir_attr.ispure = bool(ispure)
    op.attr.ir_attr.packet = int(packet)
    op.attr.sched.axis = axis
    op.x = list(inputs)
    op.y.dtype = dtype
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return op.y


def Masked(
    owner_graph: ascir.HintGraph,
    mask: ascir.OpsOperatorOutput,
    body: Union[ascir.OpsOperatorOutput, Callable[[], ascir.OpsOperatorOutput]],
    other: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    body_value = body() if callable(body) else body
    return _common_in_3_out_1_normal_op(
        "Masked",
        owner_graph,
        mask,
        body_value,
        other,
        axis=axis,
        size=size,
        stride=stride,
    )


def Randint64(
    owner_graph: ascir.HintGraph,
    seed: ascir.OpsOperatorOutput,
    offset: ascir.OpsOperatorOutput,
    low: ascir.OpsOperatorOutput,
    high: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    name = _generate_op_name(owner_graph, "randint64")
    op = ascir.ops.Randint64(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)
    op.attr.sched.axis = axis
    op.seed = seed
    op.offset = offset
    op.low = low
    op.high = high
    op.y.dtype = ascir.dtypes.int64
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return op.y


def XorSum(
    owner_graph: ascir.HintGraph,
    x1: ascir.OpsOperatorOutput,
    x2: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> ascir.OpsOperatorOutput:
    return _common_in_2_out_1_normal_op(
        "XorSum", owner_graph, x1, x2, axis=axis, size=size, stride=stride
    )


def Welford(
    owner_graph: ascir.HintGraph,
    x: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> Tuple[ascir.OpsOperatorOutput, ascir.OpsOperatorOutput]:
    name = _generate_op_name(owner_graph, "welford")
    op = ascir.ops.Welford(name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)
    op.attr.sched.axis = axis
    op.x = x
    _infer_or_set_view(op.mean, axis, size, stride)
    _infer_or_set_view(op.m2, axis, size, stride)
    op.infer_dtype()
    return op.mean, op.m2


def Scan(
    owner_graph: ascir.HintGraph,
    dtypes: Tuple[int, ...],
    combine_fn: Callable,
    *values: ascir.OpsOperatorOutput,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> Tuple[ascir.OpsOperatorOutput, ...]:
    if not dtypes:
        raise ValueError("Scan dtypes must be non-empty")
    name = _generate_op_name(owner_graph, "scan")
    op = ascir.ops.Scan(name, len(dtypes))
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.dtypes = ",".join(str(item) for item in dtypes)
    op.attr.ir_attr.combine_fn = getattr(combine_fn, "__name__", repr(combine_fn))
    op.attr.sched.axis = axis
    op.x = list(values)
    outputs = []
    for index, item in enumerate(dtypes):
        op.y[index].dtype = item
        _infer_or_set_view(op.y[index], axis, size, stride)
        outputs.append(op.y[index])
    op.infer_dtype()
    return tuple(outputs)


def StoreReduction(
    owner_graph: ascir.HintGraph,
    name: str,
    index: ascir.SizeExpr,
    values: ascir.OpsOperatorOutput,
    *,
    axis: List[ascir.Axis],
    size: Optional[List[ascir.SizeExpr]] = None,
    stride: Optional[List[ascir.SizeExpr]] = None,
) -> None:
    op_name = _generate_op_name(owner_graph, "store_reduction")
    op = ascir.ops.StoreReduction(op_name)
    meta = _get_metadata(owner_graph)
    meta.ops.append(op)
    op.attr.ir_attr.value_name = name
    op.attr.ir_attr.offset = index
    op.attr.sched.axis = axis
    op.x = values
    _infer_or_set_view(op.y, axis, size, stride)
    op.infer_dtype()
    return None
