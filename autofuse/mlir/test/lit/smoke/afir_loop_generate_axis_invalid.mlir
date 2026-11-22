// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `AxisAttr.from` 的来源轴引用必须指向本图里存在的轴。
//
// from 引用 split / merge 的来源轴，由 LoopGenerate 校验其存在性。
// 查询后必须取得有效 AxisAttr，避免对空属性读取 getAxisType()。
// 编进来取决于构建选项，不能当成门禁）。校验因此放在 `collectGraphAxes`，与「轴 id 非负」
// 「轴 id 不重复」同一处。

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --split-input-file --verify-diagnostics %s

// expected-error @+1 {{AFIR axis 1 derives from unknown axis id 7}}
func.func private @unknown_merge_source(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
  afir.function_role = #afir.function_role<kernel_body>,
  afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
  afir.asc_graph = #afir.asc_graph<
    axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
            <id = 1, name = "m", axis_type = Merged, size = #afir.expr<"s0">, align = #afir.expr<"1">, from = [0, 7]>],
    type = Compute, size_var = [#afir.expr<"s0">]>
} {
  %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>} : memref<16xf16>
  %loaded = afir.load %input outs(%tmp) {
    afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
    afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
    afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
  return
}

// -----

// 负的来源轴 id 同样在这里挡下，而不是留给 assert。
// expected-error @+1 {{AFIR axis 1 derives from unknown axis id -1}}
func.func private @negative_merge_source(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
  afir.function_role = #afir.function_role<kernel_body>,
  afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
  afir.asc_graph = #afir.asc_graph<
    axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>,
            <id = 1, name = "m", axis_type = Merged, size = #afir.expr<"s0">, align = #afir.expr<"1">, from = [0, -1]>],
    type = Compute, size_var = [#afir.expr<"s0">]>
} {
  %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>} : memref<16xf16>
  %loaded = afir.load %input outs(%tmp) {
    afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
    afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
    afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
  } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
  return
}
