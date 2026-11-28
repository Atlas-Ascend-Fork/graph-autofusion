// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RegBased candidate lowering: descriptors and actual extents survive resource
// lowering. All calls use continuous APIs inside stride-aware loops, including
// reversed Sub/Div/TrueDiv inputs; equal shapes keep their ordinary implementation.
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend950PR_9599 %s | FileCheck %s --implicit-check-not=ascendc.binary_brc_inline

#gm_in0 = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"s1">],
                            strides = [#afir.expr<"0">, #afir.expr<"1">], role = <input>>
#gm_in1 = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"s0">, #afir.expr<"s1">],
                            strides = [#afir.expr<"s1">, #afir.expr<"1">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"s0">, #afir.expr<"s1">],
                            strides = [#afir.expr<"s1">, #afir.expr<"1">], role = <output>>
#brc_tile = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"s1">],
                              strides = [#afir.expr<"0">, #afir.expr<"1">],
                              vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">],
                              role = <compute>>
#other_tile = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"s0">, #afir.expr<"s1">],
                                strides = [#afir.expr<"s1">, #afir.expr<"1">],
                                vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"(8 * Ceiling((Rational(1 , 8) * s1)))">, #afir.expr<"1">],
                                role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input0: memref<1x128xf32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">, life_time_axis_id = -1>,
                                  afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>},
      %input1: memref<128x128xf32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"65536">, life_time_axis_id = -1>,
                                    afir.kernel_arg = #afir.kernel_arg<role = input, index = 1>},
      %output: memref<128x128xf32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"65536">, life_time_axis_id = -1>,
                                    afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>}) attributes {
    afir.arg_descs = [#gm_in0, #gm_in1, #gm_out],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [<id = 0, name = "z0", axis_type = Original, size = #afir.expr<"s0">>,
              <id = 1, name = "z1", axis_type = Original, size = #afir.expr<"s1">>],
      type = Compute,
      size_var = [#afir.expr<"s0">, #afir.expr<"s1">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %brc_src = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2,
                                                                    queue_depth = 2, size_bytes_expr = <"512">,
                                                                    alignment_bytes = 32, life_time_axis_id = -1>}
        : memref<1x128xf32>
    %other_src = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, slot_count = 2,
                                                                      queue_depth = 2, size_bytes_expr = <"65536">,
                                                                      alignment_bytes = 32, life_time_axis_id = -1>}
        : memref<128x128xf32>
    %loaded0 = afir.load %input0 outs(%brc_src) {
      afir.destination_descs = [#brc_tile], afir.operand_descs = [#gm_in0], afir.result_descs = [#brc_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]
    } : memref<1x128xf32>, memref<1x128xf32> -> memref<1x128xf32>
    %loaded1 = afir.load %input1 outs(%other_src) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#gm_in1], afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]
    } : memref<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32>
    %dst = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2,
                                                                queue_depth = 2, size_bytes_expr = <"65536">,
                                                                alignment_bytes = 32, life_time_axis_id = -1>}
        : memref<128x128xf32>
    %computed = afir.add %loaded0, %loaded1 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#brc_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<1x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Minimum must use the same row-broadcast implementation in either direction.
    %minimum_lhs = afir.minimum %loaded0, %loaded1 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#brc_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<1x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    %minimum_rhs = afir.minimum %loaded1, %loaded0 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #brc_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<1x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Equal shapes must keep selecting the ordinary tensor-tensor implementation.
    %minimum_same = afir.minimum %loaded1, %minimum_rhs outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Maximum must use the same row-broadcast implementation in either direction.
    %maximum_lhs = afir.maximum %loaded0, %loaded1 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#brc_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<1x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    %maximum_rhs = afir.maximum %loaded1, %loaded0 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #brc_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<1x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Equal shapes must keep selecting the ordinary tensor-tensor implementation.
    %maximum_same = afir.maximum %loaded1, %maximum_rhs outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Sub must use the same row-broadcast implementation in either direction.
    %sub_lhs = afir.sub %loaded0, %loaded1 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#brc_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<1x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    %sub_rhs = afir.sub %loaded1, %loaded0 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #brc_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<1x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Equal shapes must keep selecting the ordinary tensor-tensor implementation.
    %sub_same = afir.sub %loaded1, %sub_rhs outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Div must use the same row-broadcast implementation in either direction.
    %div_lhs = afir.div %loaded0, %loaded1 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#brc_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<1x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    %div_rhs = afir.div %loaded1, %loaded0 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #brc_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<1x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    // Equal shapes must keep selecting the ordinary tensor-tensor implementation.
    %div_same = afir.div %loaded1, %div_rhs outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    %truediv_lhs = afir.truediv %loaded0, %loaded1 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#brc_tile, #other_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<1x128xf32>, memref<128x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    %truediv_rhs = afir.truediv %loaded1, %loaded0 outs(%dst) {
      afir.destination_descs = [#other_tile], afir.operand_descs = [#other_tile, #brc_tile],
      afir.result_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
    } : (memref<128x128xf32>, memref<1x128xf32>), memref<128x128xf32> -> memref<128x128xf32>
    afir.store %div_same outs(%output) {
      afir.destination_descs = [#gm_out], afir.operand_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]
    } : memref<128x128xf32>, memref<128x128xf32>
    return
  }
}

// CHECK: ascendc.add_l2
// CHECK: ascendc.min_l2
// CHECK: ascendc.min_l2
// CHECK: ascendc.min_l2
// CHECK: ascendc.max_l2
// CHECK: ascendc.max_l2
// CHECK: ascendc.max_l2
// CHECK: ascendc.sub_extend_reg_base
// CHECK: ascendc.sub_extend_reg_base
// CHECK: ascendc.sub_extend_reg_base
// CHECK: ascendc.div_extend_reg_base
// CHECK: ascendc.div_extend_reg_base
// CHECK: ascendc.div_extend_reg_base
// CHECK: ascendc.div_extend_reg_base
// CHECK: ascendc.div_extend_reg_base
