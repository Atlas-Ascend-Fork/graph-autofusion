// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s
// RUN: af-opt %s | af-opt | FileCheck %s

module {
  func.func @descriptor_host(%input: tensor<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = #afir.space<gm>,>}) -> tensor<4xf16> attributes {
    afir.asc_graph = #afir.asc_graph<axes = [
      <id = 0, name = "tile", axis_type = TileInner, size = #afir.expr<"4">>
    ], type = Compute>,
    afir.arg_descs = [#afir.tensor_desc<role = #afir.role<input>>],
    afir.workspace_plan = [#afir.buffer_desc<space = #afir.space<gm>, size_bytes_expr = #afir.expr<"256">,
      alignment_bytes = 16, life_time_axis_id = -1>] } {
    %result = afir.load %input {
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"4">],
        strides = [#afir.expr<"1">], role = #afir.role<input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], vectorized_axis = [0],
        vectorized_strides = [#afir.expr<"1">], role = #afir.role<output>>],
      afir.buffer_desc = #afir.buffer_desc<space = #afir.space<ub>, buffer_id = 7, reuse_id = 1,
        slot_count = 2, queue_depth = 3, size_bytes_expr = #afir.expr<"128">, alignment_bytes = 16,
        life_time_axis_id = 0>
    } : tensor<4xf16> -> tensor<4xf16>
    return %result : tensor<4xf16>
  }
}

// 值的规划结果在值的定义点：kernel 参数在 arg attr 上，tensor 形态 Op 的结果在 Op 上。
// CHECK: func.func @descriptor_host(%arg0: tensor<4xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>,>}) -> tensor<4xf16> attributes {
// CHECK-SAME: afir.arg_descs = [#afir.tensor_desc<role = <input>>]
// CHECK-SAME: afir.asc_graph = #afir.asc_graph<axes = [<id = 0
// CHECK-SAME: afir.workspace_plan = [#afir.buffer_desc<space = <gm>, size_bytes_expr = <"256">, alignment_bytes = 16, life_time_axis_id = -1>]
// CHECK: %{{.*}} = afir.load %arg0
// CHECK-SAME: afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 7, reuse_id = 1, slot_count = 2, queue_depth = 3, size_bytes_expr = <"128">, alignment_bytes = 16, life_time_axis_id = 0>
// CHECK-SAME: afir.operand_descs = [#afir.tensor_desc<axis = [0],repeats = [<"4">],strides = [<"1">],role = <input>>]
// CHECK-SAME: afir.result_descs = [#afir.tensor_desc<axis = [0],vectorized_axis = [0],vectorized_strides = [<"1">],role = <output>>]
