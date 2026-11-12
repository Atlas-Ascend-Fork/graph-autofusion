// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `cache_block_split_fused_brc_axis`：循环轴与切分轴**不同源**时，谓词只有「本循环首趟」
// 一项。
//
// 这张图是二维调度（`afir-two-dim-compute.mlir` 的轴表与 descriptor），分块切分在 z0 上，
// 而 `loop_axis` 是最内那一维的 TileOuter（`z1T`，id 6）。缓存值沿 z0 的 tile 序号变化，
// 而 z0 的 tile 序号在 `z1T` 这层循环里不变，所以整个循环内缓存值不变：外层循环每推进一步，
// 本循环重新从 0 开始，首趟那一项正好重算一次。相位在这里无从谈起——归纳变量走的是另一维。
//
// 这一格此前整类回落（诊断 `cannot build the cache guard`）：块轴的解析要求 `loop_axis` 是
// 与某个 BlockOuter 配对的 BlockInner，一维调度恰好成立，多维调度不成立。块轴现在按「与切
// 分轴同源」解析，与循环跑在哪根轴上无关。
//
// 同源那一支的谓词形态见 `afir-exec-condition-two-dim-same-origin.mlir` 与
// `afir-exec-condition-cache-guard.mlir`。

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
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]
    } : memref<2x4x8x4x16xf16>, memref<2x4x8x4x16xf16> -> memref<2x4x8x4x16xf16>
    %computed = afir.add %loaded, %loaded outs(%result) {
      afir.destination_descs = [#ub_desc],
      afir.operand_descs = [#ub_desc, #ub_desc],
      afir.result_descs = [#ub_desc],
      afir.sched = #afir.sched<axis = [3, 4, 6], loop_axis = 6, exec_condition = cache_block_split_fused_brc_axis>
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


// IR-LABEL: func.func private @stage0_template0
// 外层循环跑 z0 的块内轴，内层循环跑 z1T；守卫发在内层，用的是内层的归纳变量。
// IR: scf.for %[[OUTER:.*]] = %{{.*}} to %{{.*}} step
// IR: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step
// IR: %[[GUARD:.*]] = arith.cmpi eq, %[[IV]], %{{.*}} : index
// IR: scf.if %[[GUARD]] {
// IR:   afir.add
// IR: }

// 整个 kernel body 里恰好一个守卫。
// IFCNT: scf.if
// IFCNT-NOT: scf.if

// **一条 `arith.ori` 都不该有**：它是相位那一支独有的（首趟 || 周期起点）。这一格的循环变量
// 走的是另一维，发相位就是把另一维的序号当成了本维的。
// ORI-NOT: arith.ori
