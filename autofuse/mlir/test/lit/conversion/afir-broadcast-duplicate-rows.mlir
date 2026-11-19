// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s

// The ui8 wrapper fills GetSize(), not the requested count; never use it for an individual row.
// RUN: sed 's/f32/ui8/g' %s > %t.ui8.mlir
// RUN: not af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %t.ui8.mlir 2>&1 | FileCheck %s --check-prefix=UI8
// UI8: 'afir.broadcast' op has no supported Ascend implementation

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"1">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1024">], strides = [#afir.expr<"1">], role = <output>>
// Both vector axes read the same scalar. Write eight elements every sixteen, even on the last tile.
#src_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">],
                              strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"0">, #afir.expr<"0">], role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                         #afir.expr<"zt_size">, #afir.expr<"8">],
                              strides = [#afir.expr<"(8 * zTb_size * zt_size)">, #afir.expr<"(8 * zt_size)">,
                                         #afir.expr<"8">, #afir.expr<"1">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">], role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<1xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                             afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4">,
                                                                  life_time_axis_id = -1>},
      %output: memref<1024xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                 afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">,
                                                                      life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#gm_in, #gm_out],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
              <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
               from = [0], split_pair_other_id = 2>,
              <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
               split_pair_other_id = 1>,
              <id = 3, name = "zTB", axis_type = BlockOuter,
               size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [1],
               split_pair_other_id = 4>,
              <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [1],
               split_pair_other_id = 3>,
              <id = 5, name = "z1", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 5>],
      type = Compute,
      size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tmp_bytes = afir.tiling_param "b0_size" : index
    %tile = afir.tiling_param "zt_size" : index
    %block = afir.tiling_param "zTb_size" : index
    %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
        : (index, index) -> index
    %src = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                                                slot_count = 1, queue_depth = 2,
                                                                size_from_resource = true,
                                                                life_time_axis_id = -1>} : memref<1x1x1x1xf32>
    %loaded = afir.load %input outs(%src) {
      afir.destination_descs = [#src_tile],
      afir.operand_descs = [#gm_in],
      afir.result_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
    } : memref<1xf32>, memref<1x1x1x1xf32> -> memref<1x1x1x1xf32>
    %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?x8xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %broadcast = afir.broadcast %loaded outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>
    } : memref<1x1x1x1xf32>, memref<?x?x?x8xf32>, memref<?xui8> -> memref<?x?x?x8xf32>
    afir.store %broadcast outs(%output) {
      afir.destination_descs = [#gm_out],
      afir.operand_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d4)>]
    } : memref<?x?x?x8xf32>, memref<1024xf32>
    return
  }
}

// CHECK: %[[ACTUAL:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}
// CHECK: ascendc.wait_flag {{.*}}mte2_s
// CHECK: %[[SCALAR:.*]] = ascendc.local_tensor.get_value
// CHECK: scf.for %[[ROW:.*]] = {{.*}} to %[[ACTUAL]] step
// CHECK: %[[PITCH:.*]] = arith.constant 16 : index
// CHECK: %[[OFFSET:.*]] = arith.muli %[[ROW]], %[[PITCH]]
// CHECK: %[[DST:.*]] = ascendc.local_tensor.subindex %{{.*}}[%[[OFFSET]]]
// CHECK: %[[LEN:.*]] = arith.constant 8 : index
// CHECK: ascendc.duplicate_l2 %[[DST]], %[[SCALAR]], %[[LEN]]
