// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 缓存条件只适用于**片上**结果：目的地在 GM 的算子报告能力不足。
//
// 守卫包住一个写 GM 的算子之后，输出只在周期起点写出去，其余 tile 根本没落盘——同样是在生
// 成的 kernel 里看不出的静默数值错误。上游不会这么标（`MarkNodesCacheableBottomUp` 走到
// Load 就截断），legacy 也不按算子类别区分（`enable_cache` 只看 `IsValidCacheCondition`），
// 所以这是一条未校验的输入约束。判据写在**存储空间**上而不是算子类型上：`afir.store`
// 之外还有别的写 GM 的算子。
//
// 这张图把条件从 `load` / `broadcast` 挪到了 `store` 上，所以能力不足只有这一条。

// RUN: not af-opt --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-exec-condition \
// RUN:   --target=Ascend910B1 %s 2>&1 | FileCheck %s

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <output>>
#src_tile = #afir.tensor_desc<axis = [3, 4, 2],
                              repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">],
                              strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">],
                              vectorized_axis = [2], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [3, 4, 2],
                              repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                         #afir.expr<"zt_size">],
                              strides = [#afir.expr<"(zTb_size * zt_size)">, #afir.expr<"zt_size">,
                                         #afir.expr<"1">],
                              vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>

module @plan0 attributes {
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
      afir.destination_descs = [#src_tile],
      afir.operand_descs = [#gm_in],
      afir.result_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
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
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>
    } : memref<1x1x1xf32>, memref<?x?x?xf32>, memref<?xui8> -> memref<?x?x?xf32>
    afir.store %broadcast outs(%output) {
      afir.destination_descs = [#gm_out],
      afir.operand_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4,
                               exec_condition = cache_block_split_fused_brc_axis>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>]
    } : memref<?x?x?xf32>, memref<128xf32>
    return
  }
}

// CHECK: UnsupportedOperation: exec_condition cache_block_split_fused_brc_axis caches an on-chip result
