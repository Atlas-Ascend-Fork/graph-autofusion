// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-loop-generate --afir-memory-allocate --verify-diagnostics %s

// 签名是手写的：`%ws` 是一块 workspace GM 张量，但形参表里没有与它 `buffer_id` 相配的
// `role = workspace_offset`。这时必须报错而不是把偏移当 0——当 0 会让这块 workspace 落在
// workspace 区的起点，与另一块重叠着写，是静默的数据损坏。
module attributes {afir.workspace_tensor_ids = array<i64: 2>} {
  func.func private @workspace_without_offset(
      %input: memref<16xf16, 22 : i32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
      %tiling: !emitasc.py_struct<"TilingData", [ui32], ["s0"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    // expected-error @below {{InvalidBinding: workspace 2 has no workspace_offset kernel argument, so its offset would never be assigned}}
    %ws = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>} : memref<16xf16, #afir.space<gm>>
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16, #afir.space<ub>>
    %loaded = afir.load %input outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16, 22 : i32>, memref<16xf16, #afir.space<ub>> -> memref<16xf16, #afir.space<ub>>
    afir.store %loaded outs(%ws) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16, #afir.space<ub>>, memref<16xf16, #afir.space<gm>>
    return
  }
}
