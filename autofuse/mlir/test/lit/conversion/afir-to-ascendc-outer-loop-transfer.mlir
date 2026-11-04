// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 有效视图多于两项的搬运：最内两项进 `DataCopyPadExtend`，其余每项在 DMA 外面发一层
// `scf.for`。`afir-to-ascendc-two-dim-transfer.mlir` 钉的是恰好两项（不发循环）的形态，
// 本用例钉三项（一层）与四项（两层）。
//
// **嵌套层数是 N − 2，不是「有循环就行」**。少一层的表现是整块没搬——文本上仍然是一条
// 合法的 `data_copy_pad_extend`，所以这里必须按层数钉死，两个 section 各自数 `scf.for`。
//
// 两个 section 都没有计算算子：二维及以上的 tile 进 elementwise 是本期明确拒绝的，所以
// 只能 Load 直接接 Store。
//
// V2201 与 V3510 各发一遍：外层循环不含任何目标相关的分支，两个目标的产物必须同形。

// RUN: af-opt --split-input-file --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc --target=Ascend910B1 %s | FileCheck %s
// RUN: af-opt --split-input-file --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc --target=Ascend950DT_950x %s | FileCheck %s

//===----------------------------------------------------------------------===//
// 三项：一层外层循环。
//
// 三根原始轴 z0 / z1 / z2 各切一刀 tile，z0 的 TileOuter 再 block 切一刀。GM 是行主序的
// `s0 × s1 × s2`，tile 是 `z0t × z1t × z2t`，在 UB 里连续铺开。
//
// 有效视图（外 -> 内），三项都合并不掉——每一项内侧紧邻的都是 TileInner，规则 4 不许跨
// 这条边界合并：
//
//   [0] z0t: extent=z0t_size  gm=(s1 * s2)  ub=(z1t_size * z2t_size)   << 外层循环
//   [1] z1t: extent=z1t_size  gm=s2         ub=z2t_size                << blockCount
//   [2] z2t: extent=z2t_size  gm=1          ub=1                       << blockLen
//===----------------------------------------------------------------------===//

#gm_buffer3 = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0 * s1 * s2)">, life_time_axis_id = -1>
#ub_queue3 = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * z0t_size * z1t_size * z2t_size)">, alignment_bytes = 32, life_time_axis_id = -1>

#gm_desc3 = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7, 9, 10],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">,
             #afir.expr<"Ceiling((s2 / (z2t_size)))">, #afir.expr<"z2t_size">],
  strides = [#afir.expr<"(s1 * s2 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * s2 * z0t_size)">,
             #afir.expr<"(s1 * s2)">, #afir.expr<"(s2 * z1t_size)">, #afir.expr<"s2">,
             #afir.expr<"z2t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7, 10],
  vectorized_strides = [#afir.expr<"(z1t_size * z2t_size)">, #afir.expr<"z2t_size">, #afir.expr<"1">],
  role = <input>>
#gm_out_desc3 = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7, 9, 10],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">,
             #afir.expr<"Ceiling((s2 / (z2t_size)))">, #afir.expr<"z2t_size">],
  strides = [#afir.expr<"(s1 * s2 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * s2 * z0t_size)">,
             #afir.expr<"(s1 * s2)">, #afir.expr<"(s2 * z1t_size)">, #afir.expr<"s2">,
             #afir.expr<"z2t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7, 10],
  vectorized_strides = [#afir.expr<"(z1t_size * z2t_size)">, #afir.expr<"z2t_size">, #afir.expr<"1">],
  role = <output>>
#ub_desc3 = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7, 9, 10],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">,
             #afir.expr<"Ceiling((s2 / (z2t_size)))">, #afir.expr<"z2t_size">],
  strides = [#afir.expr<"(s1 * s2 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * s2 * z0t_size)">,
             #afir.expr<"(s1 * s2)">, #afir.expr<"(s2 * z1t_size)">, #afir.expr<"s2">,
             #afir.expr<"z2t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7, 10],
  vectorized_strides = [#afir.expr<"(z1t_size * z2t_size)">, #afir.expr<"z2t_size">, #afir.expr<"1">],
  role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<2x4x8x4x16x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer3},
      %output: memref<2x4x8x4x16x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer3}) attributes {
    afir.arg_descs = [#gm_desc3, #gm_out_desc3],
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
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 6>,
        <id = 8, name = "z2", axis_type = Original, size = #afir.expr<"s2">>,
        <id = 9, name = "z2T", axis_type = TileOuter, size = #afir.expr<"Ceiling((s2 / (z2t_size)))">,
          align = #afir.expr<"1">, from = [8], split_pair_other_id = 10>,
        <id = 10, name = "z2t", axis_type = TileInner, size = #afir.expr<"z2t_size">,
          align = #afir.expr<"1">, from = [8], split_pair_other_id = 9>],
      type = Compute,
      size_var = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">, #afir.expr<"z0t_size">,
                  #afir.expr<"z0Tb_size">, #afir.expr<"z1t_size">, #afir.expr<"z2t_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue3} : memref<2x4x8x4x16x4x16xf16>
    %loaded = afir.load %input outs(%tile) {
      afir.destination_descs = [#ub_desc3],
      afir.operand_descs = [#gm_desc3],
      afir.result_descs = [#ub_desc3],
      afir.sched = #afir.sched<axis = [3, 4, 6, 9], loop_axis = 9>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>]
    } : memref<2x4x8x4x16x4x16xf16>, memref<2x4x8x4x16x4x16xf16> -> memref<2x4x8x4x16x4x16xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#gm_out_desc3],
      afir.operand_descs = [#ub_desc3],
      afir.sched = #afir.sched<axis = [3, 4, 6, 9], loop_axis = 9>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>]
    } : memref<2x4x8x4x16x4x16xf16>, memref<2x4x8x4x16x4x16xf16>
    return
  }
}

// CHECK-LABEL: func.func private @stage0_template0

// `LoopGenerate` 先发三层调度循环（z0Tb、z1T、z2T），并为每根配对 inner 发一份
// `actual_size`。搬运侧的三个量全部取自这些绑定，循环外物化一次：
// CHECK: %[[BOUND:.*]] = arith.select {{.*}}<axis = 2, role = actual_size>
// CHECK: %[[BLOCKCOUNT:.*]] = arith.select {{.*}}<axis = 7, role = actual_size>
// CHECK: %[[BLOCKLEN:.*]] = arith.select {{.*}}<axis = 10, role = actual_size>

// Load。
//
// 两侧缝隙、外层那一项的两侧步长，都在循环**外**算一次；循环体内只剩乘加。这不是风格
// ——把 stride 留在体内会让每一轮重算一遍同一个乘积。
// CHECK:      %[[LGMGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
// CHECK-NEXT: %[[LUBGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
// CHECK-NEXT: %[[LGSTRIDE:.*]] = index.mul
// CHECK-NEXT: %[[LUBSTRIDE:.*]] = index.mul
//
// 上界是 z0t 的 **actual_size**，不是标称尺寸：尾块那一轮少几行，按标称走会多搬。
// CHECK:      scf.for %[[LIV:.*]] = %{{.*}} to %[[BOUND]] step %{{.*}} {
// 行内偏移两侧各算各的：GM 侧叠在 tile 起点之上，UB 侧从 0 起算（tile 在 buffer 里的
// 起点恒为基址）。
// CHECK-NEXT:   %[[LGT:.*]] = arith.muli %[[LIV]], %[[LGSTRIDE]]
// CHECK-NEXT:   %[[LGOFF:.*]] = arith.addi %{{.*}}, %[[LGT]]
// CHECK-NEXT:   %[[LLOFF:.*]] = arith.muli %[[LIV]], %[[LUBSTRIDE]]
// CHECK-NEXT:   %{{.*}} = ascendc.global_tensor.subindex %{{.*}}[%[[LGOFF]]]
// CHECK-NEXT:   %{{.*}} = ascendc.local_tensor.subindex %{{.*}}[%[[LLOFF]]]
// CHECK-NEXT:   ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %[[BLOCKCOUNT]], %[[BLOCKLEN]], %[[LGMGAP]], %[[LUBGAP]]
// **只有一层**：DMA 之后立刻收口。少一层的表现是整块没搬，文本上仍是一条合法的 DMA。
// CHECK-NEXT: }

// Store。同一套形态，方向相反：两个子视图的先后换位，两个缝隙参数也随之**交换位置**
// （src 侧取 UB、dst 侧取 GM）。
// CHECK:      %[[SGMGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
// CHECK-NEXT: %[[SUBGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
// CHECK-NEXT: %[[SGSTRIDE:.*]] = index.mul
// CHECK-NEXT: %[[SUBSTRIDE:.*]] = index.mul
// CHECK:      scf.for %[[SIV:.*]] = %{{.*}} to %[[BOUND]] step %{{.*}} {
// CHECK-NEXT:   %[[SGT:.*]] = arith.muli %[[SIV]], %[[SGSTRIDE]]
// CHECK-NEXT:   %[[SGOFF:.*]] = arith.addi %{{.*}}, %[[SGT]]
// CHECK-NEXT:   %[[SLOFF:.*]] = arith.muli %[[SIV]], %[[SUBSTRIDE]]
// CHECK-NEXT:   %{{.*}} = ascendc.local_tensor.subindex %{{.*}}[%[[SLOFF]]]
// CHECK-NEXT:   %{{.*}} = ascendc.global_tensor.subindex %{{.*}}[%[[SGOFF]]]
// CHECK-NEXT:   ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %[[BLOCKCOUNT]], %[[BLOCKLEN]], %[[SUBGAP]], %[[SGMGAP]]
// CHECK-NEXT: }

// -----

//===----------------------------------------------------------------------===//
// 四项：两层外层循环。
//
// 同一套形态再加一根原始轴 z3。有效视图（外 -> 内）：
//
//   [0] z0t: extent=z0t_size  gm=(s1 * s2 * s3)  ub=(z1t_size * z2t_size * z3t_size)  << 外层
//   [1] z1t: extent=z1t_size  gm=(s2 * s3)       ub=(z2t_size * z3t_size)             << 外层
//   [2] z2t: extent=z2t_size  gm=s3              ub=z3t_size                          << blockCount
//   [3] z3t: extent=z3t_size  gm=1               ub=1                                 << blockLen
//===----------------------------------------------------------------------===//

#gm_buffer4 = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0 * s1 * s2 * s3)">, life_time_axis_id = -1>
#ub_queue4 = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * z0t_size * z1t_size * z2t_size * z3t_size)">, alignment_bytes = 32, life_time_axis_id = -1>

#gm_desc4 = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7, 9, 10, 12, 13],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">,
             #afir.expr<"Ceiling((s2 / (z2t_size)))">, #afir.expr<"z2t_size">,
             #afir.expr<"Ceiling((s3 / (z3t_size)))">, #afir.expr<"z3t_size">],
  strides = [#afir.expr<"(s1 * s2 * s3 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * s2 * s3 * z0t_size)">,
             #afir.expr<"(s1 * s2 * s3)">, #afir.expr<"(s2 * s3 * z1t_size)">, #afir.expr<"(s2 * s3)">,
             #afir.expr<"(s3 * z2t_size)">, #afir.expr<"s3">, #afir.expr<"z3t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7, 10, 13],
  vectorized_strides = [#afir.expr<"(z1t_size * z2t_size * z3t_size)">, #afir.expr<"(z2t_size * z3t_size)">,
                        #afir.expr<"z3t_size">, #afir.expr<"1">],
  role = <input>>
#gm_out_desc4 = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7, 9, 10, 12, 13],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">,
             #afir.expr<"Ceiling((s2 / (z2t_size)))">, #afir.expr<"z2t_size">,
             #afir.expr<"Ceiling((s3 / (z3t_size)))">, #afir.expr<"z3t_size">],
  strides = [#afir.expr<"(s1 * s2 * s3 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * s2 * s3 * z0t_size)">,
             #afir.expr<"(s1 * s2 * s3)">, #afir.expr<"(s2 * s3 * z1t_size)">, #afir.expr<"(s2 * s3)">,
             #afir.expr<"(s3 * z2t_size)">, #afir.expr<"s3">, #afir.expr<"z3t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7, 10, 13],
  vectorized_strides = [#afir.expr<"(z1t_size * z2t_size * z3t_size)">, #afir.expr<"(z2t_size * z3t_size)">,
                        #afir.expr<"z3t_size">, #afir.expr<"1">],
  role = <output>>
#ub_desc4 = #afir.tensor_desc<
  axis = [3, 4, 2, 6, 7, 9, 10, 12, 13],
  repeats = [#afir.expr<"Ceiling((Ceiling((s0 / (z0t_size))) / (z0Tb_size)))">, #afir.expr<"z0Tb_size">,
             #afir.expr<"z0t_size">, #afir.expr<"Ceiling((s1 / (z1t_size)))">, #afir.expr<"z1t_size">,
             #afir.expr<"Ceiling((s2 / (z2t_size)))">, #afir.expr<"z2t_size">,
             #afir.expr<"Ceiling((s3 / (z3t_size)))">, #afir.expr<"z3t_size">],
  strides = [#afir.expr<"(s1 * s2 * s3 * z0Tb_size * z0t_size)">, #afir.expr<"(s1 * s2 * s3 * z0t_size)">,
             #afir.expr<"(s1 * s2 * s3)">, #afir.expr<"(s2 * s3 * z1t_size)">, #afir.expr<"(s2 * s3)">,
             #afir.expr<"(s3 * z2t_size)">, #afir.expr<"s3">, #afir.expr<"z3t_size">, #afir.expr<"1">],
  vectorized_axis = [2, 7, 10, 13],
  vectorized_strides = [#afir.expr<"(z1t_size * z2t_size * z3t_size)">, #afir.expr<"(z2t_size * z3t_size)">,
                        #afir.expr<"z3t_size">, #afir.expr<"1">],
  role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<2x4x8x4x16x4x16x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.buffer_desc = #gm_buffer4},
      %output: memref<2x4x8x4x16x4x16x4x16xf16> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.buffer_desc = #gm_buffer4}) attributes {
    afir.arg_descs = [#gm_desc4, #gm_out_desc4],
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
          align = #afir.expr<"1">, from = [5], split_pair_other_id = 6>,
        <id = 8, name = "z2", axis_type = Original, size = #afir.expr<"s2">>,
        <id = 9, name = "z2T", axis_type = TileOuter, size = #afir.expr<"Ceiling((s2 / (z2t_size)))">,
          align = #afir.expr<"1">, from = [8], split_pair_other_id = 10>,
        <id = 10, name = "z2t", axis_type = TileInner, size = #afir.expr<"z2t_size">,
          align = #afir.expr<"1">, from = [8], split_pair_other_id = 9>,
        <id = 11, name = "z3", axis_type = Original, size = #afir.expr<"s3">>,
        <id = 12, name = "z3T", axis_type = TileOuter, size = #afir.expr<"Ceiling((s3 / (z3t_size)))">,
          align = #afir.expr<"1">, from = [11], split_pair_other_id = 13>,
        <id = 13, name = "z3t", axis_type = TileInner, size = #afir.expr<"z3t_size">,
          align = #afir.expr<"1">, from = [11], split_pair_other_id = 12>],
      type = Compute,
      size_var = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">, #afir.expr<"s3">,
                  #afir.expr<"z0t_size">, #afir.expr<"z0Tb_size">, #afir.expr<"z1t_size">,
                  #afir.expr<"z2t_size">, #afir.expr<"z3t_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tile = memref.alloc() {afir.buffer_desc = #ub_queue4} : memref<2x4x8x4x16x4x16x4x16xf16>
    %loaded = afir.load %input outs(%tile) {
      afir.destination_descs = [#ub_desc4],
      afir.operand_descs = [#gm_desc4],
      afir.result_descs = [#ub_desc4],
      afir.sched = #afir.sched<axis = [3, 4, 6, 9, 12], loop_axis = 12>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>]
    } : memref<2x4x8x4x16x4x16x4x16xf16>, memref<2x4x8x4x16x4x16x4x16xf16> -> memref<2x4x8x4x16x4x16x4x16xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#gm_out_desc4],
      afir.operand_descs = [#ub_desc4],
      afir.sched = #afir.sched<axis = [3, 4, 6, 9, 12], loop_axis = 12>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>]
    } : memref<2x4x8x4x16x4x16x4x16xf16>, memref<2x4x8x4x16x4x16x4x16xf16>
    return
  }
}

// CHECK-LABEL: func.func private @stage0_template0

// CHECK: %[[BOUND0:.*]] = arith.select {{.*}}<axis = 2, role = actual_size>
// CHECK: %[[BOUND1:.*]] = arith.select {{.*}}<axis = 7, role = actual_size>
// CHECK: %[[BLOCKCOUNT:.*]] = arith.select {{.*}}<axis = 10, role = actual_size>
// CHECK: %[[BLOCKLEN:.*]] = arith.select {{.*}}<axis = 13, role = actual_size>

// Load。
// CHECK:      %[[LGMGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
// CHECK-NEXT: %[[LUBGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
//
// 两层嵌套，外层走 z0t、内层走 z1t，各自的上界是各自的 actual_size。
// CHECK:      scf.for %[[LIV0:.*]] = %{{.*}} to %[[BOUND0]] step %{{.*}} {
// CHECK-NEXT:   scf.for %[[LIV1:.*]] = %{{.*}} to %[[BOUND1]] step %{{.*}} {
// 两层的行内偏移**分别**累加到两侧。漏掉任何一项的表现是每轮搬同一块，数值上整段重复。
// CHECK-NEXT:     %[[LG0:.*]] = arith.muli %[[LIV0]], %{{.*}}
// CHECK-NEXT:     %[[LGS:.*]] = arith.addi %{{.*}}, %[[LG0]]
// CHECK-NEXT:     %[[LL0:.*]] = arith.muli %[[LIV0]], %{{.*}}
// CHECK-NEXT:     %[[LG1:.*]] = arith.muli %[[LIV1]], %{{.*}}
// CHECK-NEXT:     %[[LGOFF:.*]] = arith.addi %[[LGS]], %[[LG1]]
// CHECK-NEXT:     %[[LL1:.*]] = arith.muli %[[LIV1]], %{{.*}}
// CHECK-NEXT:     %[[LLOFF:.*]] = arith.addi %[[LL0]], %[[LL1]]
// CHECK-NEXT:     %{{.*}} = ascendc.global_tensor.subindex %{{.*}}[%[[LGOFF]]]
// CHECK-NEXT:     %{{.*}} = ascendc.local_tensor.subindex %{{.*}}[%[[LLOFF]]]
// CHECK-NEXT:     ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %[[BLOCKCOUNT]], %[[BLOCKLEN]], %[[LGMGAP]], %[[LUBGAP]]
// 恰好两层：DMA 之后连着两个收口。
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Store。
// CHECK:      %[[SGMGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
// CHECK-NEXT: %[[SUBGAP:.*]] = arith.subi %{{.*}}, %[[BLOCKLEN]] : index
// CHECK:      scf.for %[[SIV0:.*]] = %{{.*}} to %[[BOUND0]] step %{{.*}} {
// CHECK-NEXT:   scf.for %[[SIV1:.*]] = %{{.*}} to %[[BOUND1]] step %{{.*}} {
// CHECK-NEXT:     %[[SG0:.*]] = arith.muli %[[SIV0]], %{{.*}}
// CHECK-NEXT:     %[[SGS:.*]] = arith.addi %{{.*}}, %[[SG0]]
// CHECK-NEXT:     %[[SL0:.*]] = arith.muli %[[SIV0]], %{{.*}}
// CHECK-NEXT:     %[[SG1:.*]] = arith.muli %[[SIV1]], %{{.*}}
// CHECK-NEXT:     %[[SGOFF:.*]] = arith.addi %[[SGS]], %[[SG1]]
// CHECK-NEXT:     %[[SL1:.*]] = arith.muli %[[SIV1]], %{{.*}}
// CHECK-NEXT:     %[[SLOFF:.*]] = arith.addi %[[SL0]], %[[SL1]]
// CHECK-NEXT:     %{{.*}} = ascendc.local_tensor.subindex %{{.*}}[%[[SLOFF]]]
// CHECK-NEXT:     %{{.*}} = ascendc.global_tensor.subindex %{{.*}}[%[[SGOFF]]]
// CHECK-NEXT:     ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %[[BLOCKCOUNT]], %[[BLOCKLEN]], %[[SUBGAP]], %[[SGMGAP]]
// CHECK-NEXT:   }
// CHECK-NEXT: }
