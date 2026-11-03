// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 逐元素算子的输入与目的地的**局部有效视图必须可证明相同**，不相同时显式拒绝。
//
// 一次一维调用只覆盖一段从基址算起的连续地址，长度由 `materializeScheduledLocalSpan`
// 按输入的有效视图折算。目的地若是另一套 extent/stride，这个长度对它就是错的：写出去的
// 元素落在目的地布局的哪几行，取决于两套 stride 的差——不崩、不越界、数值悄悄是错的。
//
// 夹具与 `afir-two-dim-compute.mlir` 完全相同，只把 `afir.add` 的目的地换成一项有效视图的
// 描述符。`destination_descs` 与 `result_descs` 要一起换：verifier 先查这两者是否一致
// （`InvalidBinding: tied result and destination descriptors differ`），只换一处会在到达
// 能力查询之前就失败，钉住的就不是这条判据了。

// RUN: not af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s
// RUN: not af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 | FileCheck %s

#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0 * s1)">, life_time_axis_id = -1>
#ub_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * z0t_size * z1t_size)">, alignment_bytes = 32, life_time_axis_id = -1>
// 出队列要单独一个 buffer_id：同一个队列资源不能既是 TransferIn 又是 TransferOut。
#ub_out_queue = #afir.buffer_desc<space = <ub>, buffer_id = 1, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * z0t_size * z1t_size)">, alignment_bytes = 32, life_time_axis_id = -1>

#gm_desc = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">],
  strides = [#afir.expr<"(s1 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * z0t_size)">, #afir.expr<"s1">,
             #afir.expr<"z1t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7], vectorized_strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
  role = <input>>
#gm_out_desc = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">],
  strides = [#afir.expr<"(s1 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * z0t_size)">, #afir.expr<"s1">,
             #afir.expr<"z1t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7], vectorized_strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
  role = <output>>
#ub_desc = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">],
  strides = [#afir.expr<"(s1 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * z0t_size)">, #afir.expr<"s1">,
             #afir.expr<"z1t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7], vectorized_strides = [#afir.expr<"z1t_size">, #afir.expr<"1">],
  role = <compute>>

#ub_narrow_desc = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">],
  strides = [#afir.expr<"(s1 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * z0t_size)">, #afir.expr<"s1">,
             #afir.expr<"z1t_size">, #afir.expr<"1">],
  vectorized_axis = [7], vectorized_strides = [#afir.expr<"1">],
  role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<2x4x8x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer},
      %output: memref<2x4x8x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer}) attributes {
    afir.arg_descs = [#gm_desc, #gm_out_desc],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [
        <id = 0, name = "z0", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "z0T", axis_type = TileOuter, size = #afir.expr<"Ceiling((s0 / (z0t_size)))">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
        <id = 2, name = "z0t", axis_type = TileInner, size = #afir.expr<"z0t_size">,
          align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>,
        <id = 3, name = "z0TB", axis_type = BlockOuter, size = #afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 4>,
        <id = 4, name = "z0Tb", axis_type = BlockInner, size = #afir.expr<"z0Tb_size">,
          align = #afir.expr<"1">, from = [1], split_pair_other_id = 3>,
        <id = 5, name = "z1", axis_type = Original, size = #afir.expr<"s1">>,
        <id = 6, name = "z1T", axis_type = TileOuter, size = #afir.expr<"Ceiling((s1 / (z1t_size)))">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 7>,
        <id = 7, name = "z1t", axis_type = TileInner, size = #afir.expr<"z1t_size">,
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 6>],
      type = Compute,
      size_var = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"z0t_size">, #afir.expr<"z0Tb_size">,
                  #afir.expr<"z1t_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue} : memref<2x4x8x4x16xf16>
    %result = memref.alloc() {afir.buffer_desc = #ub_out_queue} : memref<2x4x8x4x16xf16>
    %loaded = afir.load %input outs(%tile) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#gm_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16> -> memref<2x4x8x4x16xf16>
    // CHECK: 'afir.add' op has no supported Ascend implementation
    %computed = afir.add %loaded, %loaded outs(%result) {
      afir.destination_descs = [#ub_narrow_desc],
      afir.operand_descs = [#ub_desc, #ub_desc],
      afir.result_descs = [#ub_narrow_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>
    } : (memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16>), memref<2x4x8x4x16xf16> -> memref<2x4x8x4x16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#gm_out_desc],
      afir.operand_descs = [#ub_narrow_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16>
    return
  }
}
