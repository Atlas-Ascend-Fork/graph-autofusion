// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `cache_block_split_fused_brc_axis`：循环轴与切分轴**同源**时，谓词是「本循环首趟」或
// 「广播复用周期起点」。
//
// 与 `afir-exec-condition-two-dim-cross-origin.mlir` 是同一张二维图，只把 `loop_axis` 换成
// z0 的块内轴（`z0Tb`，id 4）——缓存值变化的那一维与循环跑的那一维重合，于是要算相位：
// 本维上的全局 tile 序号是「分核序号 × 块内长度 + 块内序号」，对 `z0T` 的循环次数取模归零
// 就是一个广播复用周期的起点。
//
// 块轴这里取的仍然是与切分轴同源的那一对（`z0TB` / `z0Tb`），在这张图上它恰好与 `loop_axis`
// 重合；一维图上的同一形态见 `afir-exec-condition-cache-guard.mlir`。
//
// **这一格是回归守卫，不是变异见证。** 把块轴解析改回「与 `loop_axis` 配对」之后它照样通过：
// 这张图的 `loop_axis` 就是那一对里的块内轴，两种解析取到同一根轴。它钉的是另一件事——
// 在有两根原始轴的轴表上，相位形态不能被「不同源就只算首趟」那一支抢走。会因本次改动变红的
// 那一格是 `afir-exec-condition-two-dim-cross-origin.mlir`。

// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate \
// RUN:   --afir-memory-allocate --afir-select-implementation --afir-pipeline-sync \
// RUN:   --afir-exec-condition --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate \
// RUN:   --afir-memory-allocate --afir-select-implementation --afir-pipeline-sync \
// RUN:   --afir-exec-condition --target=Ascend910B1 %s | FileCheck %s --check-prefix=IFCNT
// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate \
// RUN:   --afir-memory-allocate --afir-select-implementation --afir-pipeline-sync \
// RUN:   --afir-exec-condition --target=Ascend910B1 %s | FileCheck %s --check-prefix=ORI

#gm_buffer = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0 * s1)">, life_time_axis_id = -1>
#ub_queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 1, queue_depth = 2, size_bytes_expr = <"(2 * z0t_size * z1t_size)">, alignment_bytes = 32, life_time_axis_id = -1>
// 出队列要单独一个 buffer_id：同一个队列资源不能既是 TransferIn 又是 TransferOut。
#ub_out_queue = #afir.buffer_desc<space = <ub>, buffer_id = 1, slot_count = 1, queue_depth = 2, size_bytes_expr = <"(2 * z0t_size * z1t_size)">, alignment_bytes = 32, life_time_axis_id = -1>

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
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16> -> memref<2x4x8x4x16xf16>
    %computed = afir.add %loaded, %loaded outs(%result) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#ub_desc, #ub_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 4, exec_condition = cache_block_split_fused_brc_axis>
    } : (memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16>), memref<2x4x8x4x16xf16> -> memref<2x4x8x4x16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#gm_out_desc],
      afir.operand_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16>
    return
  }
}


// IR-LABEL: func.func private @stage0_template0
// 分核序号先落到块轴的坐标上。
// IR: %[[BLOCKIDX:.*]] = afir.block_id
// IR: %[[BLOCKPOS:.*]] = arith.remsi %[[BLOCKIDX]], %{{.*}} {afir.axis_binding = #afir.axis_binding<axis = 3, role = axis_index>}
// IR: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step
// IR: %[[FIRST:.*]] = arith.cmpi eq, %[[IV]], %[[ZERO:.*]] : index
// IR: %[[SCALED:.*]] = arith.muli %[[BLOCKPOS]], %{{.*}} : index
// IR: %[[LINEAR:.*]] = arith.addi %[[SCALED]], %[[IV]] : index
// IR: %[[PHASE:.*]] = arith.remsi %[[LINEAR]], %{{.*}} : index
// IR: %[[START:.*]] = arith.cmpi eq, %[[PHASE]], %[[ZERO]] : index
// IR: %[[GUARD:.*]] = arith.ori %[[FIRST]], %[[START]] : i1
// IR: scf.if %[[GUARD]] {
// IR:   afir.add
// IR: }

// 整个 kernel body 里恰好一个守卫。
// IFCNT: scf.if
// IFCNT-NOT: scf.if

// 恰好一份相位谓词：`arith.ori` 出现一次，之后不再出现（第一条正向断言匹配的就是首次出现，
// 所以这两行合起来等于「全文恰好一条」）。
// ORI: arith.ori
// ORI-NOT: arith.ori
