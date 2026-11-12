// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 内联广播（brc_inline，第三批）接通 binary elementwise 的覆盖：capability 判据选中它、
// pattern 建出 `ascendc.binary_brc_inline`。设计见
// docs/development/2026-11-10-afir-broadcast-inline-design.md 功能需求 4。
//
// 形态是 `(1, s1) + (s0, s1)`：两根向量化轴 `[0, 1]`，axis 0 上广播侧 repeat 恒为 1、
// 步长为 0；axis 1 两侧同宽。与
// `test/ut/conversion/ElementwiseTransferPlanTest.cpp` 用同一套描述子（判据本身已经在
// 那边被单独钉住），这里钉的是下一层——选中 implementation 之后真的建出了带正确实参的
// 目标 op，不是答 supported 却建错 op 或者建到旧的 tensor-tensor pattern 上。
//
// 两侧同形时判据返回不支持、回落 `ascendc.add_l2` 的反例在
// `ElementwiseTransferPlanTest.RejectsSameShapedOperands`（判据层）钉住；这里不重复
// 用 `sed` 改一份近似图——descriptor 里 repeats/strides 与 vectorized_axis/strides
// 要同时改，裸替换容易改出一张四不像的图，不如直接在 UT 里手写。
//
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s

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
                                vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"s1">, #afir.expr<"1">],
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
    afir.store %computed outs(%output) {
      afir.destination_descs = [#gm_out], afir.operand_descs = [#other_tile],
      afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]
    } : memref<128x128xf32>, memref<128x128xf32>
    return
  }
}

// CHECK: ascendc.binary_brc_inline
// CHECK-SAME: wrappedApiName = "Add"
// CHECK-NOT: ascendc.add_l2
