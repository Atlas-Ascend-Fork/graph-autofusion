// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 二维 tile 上的逐元素计算按行下降：两个 actual extent 决定尾块的行数与每行长度，
// nominal local stride 决定 UB 行距。行间空隙不能当成连续元素参与 Add。

// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc --target=Ascend910B1 %s | FileCheck %s --check-prefixes=IR,MEM
// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc --target=Ascend950DT_950x %s | FileCheck %s --check-prefixes=IR,REG

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
    %computed = afir.add %loaded, %loaded outs(%result) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#ub_desc, #ub_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>
    } : (memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16>), memref<2x4x8x4x16xf16> -> memref<2x4x8x4x16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#gm_out_desc],
      afir.operand_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16>
    return
  }
}

// IR: %[[Z1T:.*]] = arith.index_cast

// IR: %[[BLOCKCOUNT:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 2, role = actual_size>}
// IR: %[[BLOCKLEN:.*]] = arith.select {{.*}} {afir.axis_binding = #afir.axis_binding<axis = 7, role = actual_size>}
// MEM: scf.for %[[ROW:.*]] = %{{.*}} to %[[BLOCKCOUNT]] step
// MEM: %[[SRC_OFF:.*]] = arith.muli %[[ROW]], %{{.*}} : index
// MEM: ascendc.local_tensor.subindex %{{.*}}[%[[SRC_OFF]]]
// MEM: ascendc.add_l2 %{{.*}}, %{{.*}}, %{{.*}}, %[[BLOCKLEN]]

// REG: %[[COUNT:.*]] = arith.muli %[[BLOCKCOUNT]], %[[BLOCKLEN]] : index
// REG: ascendc.add_l2 %{{.*}}, %{{.*}}, %{{.*}}, %[[COUNT]]
