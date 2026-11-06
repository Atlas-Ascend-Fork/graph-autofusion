// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Compact output [8, bt] uses ActualSize(bt) as its row stride on the last tile.
// Legacy: GetVectorizedStrideActualSizeFlag + BuildTransposeLoopParams.
// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc %s \
// RUN:   | FileCheck %s

#gm = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"37">, #afir.expr<"8">],
  strides = [#afir.expr<"8">, #afir.expr<"1">], role = <input>>
#src = #afir.tensor_desc<axis = [4, 5, 3, 1],
  repeats = [#afir.expr<"(37 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">, #afir.expr<"8">],
  strides = [#afir.expr<"(8 * bTb_size * bt_size)">, #afir.expr<"(8 * bt_size)">, #afir.expr<"8">, #afir.expr<"1">],
  vectorized_axis = [3, 1], vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [4, 5, 1, 3],
  repeats = [#afir.expr<"(37 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"8">, #afir.expr<"bt_size">],
  strides = [#afir.expr<"(8 * bTb_size * bt_size)">, #afir.expr<"(8 * bt_size)">, #afir.expr<"bt_size">, #afir.expr<"1">],
  vectorized_axis = [1, 3], vectorized_strides = [#afir.expr<"bt_size">, #afir.expr<"1">], role = <compute>>

func.func private @compact_tail(
  %input: memref<37x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
    afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"1184">, life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm],
  afir.asc_graph = #afir.asc_graph<tiling_key = 0,
    axes = [<id = 0, name = "b", axis_type = Original, size = #afir.expr<"37">, split_pair_other_id = 0>,
      <id = 1, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 1>,
      <id = 3, name = "bt", axis_type = TileInner, size = #afir.expr<"bt_size">, from = [0], split_pair_other_id = 6>,
      <id = 4, name = "bTB", axis_type = BlockOuter,
        size = #afir.expr<"Ceiling((Ceiling((37 / (bt_size))) / (bTb_size)))">, from = [6], split_pair_other_id = 5>,
      <id = 5, name = "bTb", axis_type = BlockInner, size = #afir.expr<"bTb_size">, from = [6], split_pair_other_id = 4>,
      <id = 6, name = "bT", axis_type = TileOuter, size = #afir.expr<"Ceiling((37 / (bt_size)))">, from = [0], split_pair_other_id = 3>],
    type = Compute, size_var = [#afir.expr<"bt_size">, #afir.expr<"bTb_size">]>,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(37 / (__arg0 * __arg1))">} : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
    afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, size_from_resource = true,
      life_time_axis_id = -1>} : memref<?x?x?x8xf32>
  %dst = memref.alloc(%outer, %block, %tile) {
    afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, size_from_resource = true,
      life_time_axis_id = -1>} : memref<?x?x8x?xf32>
  %tmp = memref.alloc(%tile) {
    afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, reuse_id = 2, size_from_resource = true,
      life_time_axis_id = -1>} : memref<?xui8>
  %result = afir.transpose %src outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>]
  } : memref<?x?x?x8xf32>, memref<?x?x8x?xf32>, memref<?xui8> -> memref<?x?x8x?xf32>
  return
}

// CHECK-LABEL: func.func private @compact_tail
// CHECK: %[[ACTUAL:[a-zA-Z0-9_]+]] = arith.select {{.*}}axis = 3, role = actual_size
// CHECK: ascendc.transpose_extend_reg_base {{.*}}, %[[ACTUAL]], {{.*}}, %[[ACTUAL]], {{.*}}innerDim = 2
