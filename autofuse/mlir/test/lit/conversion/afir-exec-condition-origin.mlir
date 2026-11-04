// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR

// `cache_block_split_origin_brc_axis`：TILE_OUT 及以外的原始轴全是广播轴，缓存值在整个
// 分块循环里不变，只在本核第一趟算。谓词因此只有 `i == 0` 那一项，没有相位。
//
// **今天没有哪张图走到这一格**：`NodeCacheMarker::MarkNodeCacheable` 无条件写的是 fused
// 那一个取值（注释说 origin 那支「由于引入精度问题，暂时关闭」）。留这一格是因为
// `exec_condition` 是上游的枚举，bridge 翻译的是整个枚举而不是其中一个值——少一格的表现
// 是把 origin 静默当成 fused，那会多算，也会在上游放开那一支时无声地错。

#gm_in2 = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], role = <input>>
#gm_out2 = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <output>>
#src_tile2 = #afir.tensor_desc<axis = [3, 4, 2],
                               repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">],
                               strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">],
                               vectorized_axis = [2], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#dst_tile2 = #afir.tensor_desc<axis = [3, 4, 2],
                               repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                          #afir.expr<"zt_size">],
                               strides = [#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">,
                                          #afir.expr<"1">],
                               vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>

module @plan1 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<1xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                             afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4">,
                                                                  life_time_axis_id = -1>},
      %output: memref<128xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"512">,
                                                                     life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#gm_in2, #gm_out2],
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
               split_pair_other_id = 3>],
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
                                                                life_time_axis_id = -1>} : memref<1x1x1xf32>
    %loaded = afir.load %input outs(%src) {
      afir.destination_descs = [#src_tile2],
      afir.operand_descs = [#gm_in2],
      afir.result_descs = [#src_tile2],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4,
                               exec_condition = cache_block_split_origin_brc_axis>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>]
    } : memref<1xf32>, memref<1x1x1xf32> -> memref<1x1x1xf32>
    %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %broadcast = afir.broadcast %loaded outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#dst_tile2],
      afir.operand_descs = [#src_tile2],
      afir.result_descs = [#dst_tile2],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4,
                               exec_condition = cache_block_split_origin_brc_axis>
    } : memref<1x1x1xf32>, memref<?x?x?xf32>, memref<?xui8> -> memref<?x?x?xf32>
    afir.store %broadcast outs(%output) {
      afir.destination_descs = [#gm_out2],
      afir.operand_descs = [#dst_tile2],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>]
    } : memref<?x?x?xf32>, memref<128xf32>
    return
  }
}

// IR-LABEL: func.func private @stage0_template0
// IR: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step
// IR: %[[GUARD:.*]] = arith.cmpi eq, %[[IV]], %{{.*}} : index
// IR: ascendc.que_bind.alloc_tensor
// IR: scf.if %[[GUARD]] {
// IR:   ascendc.data_copy_pad_extend
// IR: }
// IR: scf.if %[[GUARD]] {
// IR:   ascendc.duplicate_l2
// IR: }
// 相位那一段整个不该出现：origin 说的是「整个分块循环里都不变」，没有周期起点可言。
// IR-NOT: arith.ori
