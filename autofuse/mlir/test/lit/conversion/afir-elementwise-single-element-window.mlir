// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 计算 op 的向量化轴 extent 全为 1：本趟只有一个元素，元素数是常量 1，一层循环都不建。
//
// 真实来源是广播轴被切分、又与外轴合成一根块轴的那个模板（`exec_condition =
// cache_block_split_fused_brc_axis`，与 `afir-exec-condition-cache-guard.mlir` 同一张图）：
// 调度把 elementwise 排到 `afir.broadcast` 之前，于是它作用在广播前那块
// `repeats = [n, 1, 1], vectorized_axis = [zt]` 的 tile 上。`[8,1] -> [8,128]` 的 `B0Y1`
// 模板实测就是这个形状（2026-10-12，`test/probe/CapabilityBoundaryProbe.cpp` 的 T2 / T6）。
//
// 缺了这一条会怎样：`collapseVectorizedAxes` 在这个形状上返回零段，而
// `materializeVectorizedSegments` 的第一道闸是「段数不足窗口大小就报失败」，于是 pattern
// 失败而 capability 答支持——`failed to legalize operation 'afir.abs'`，整图硬失败而不是
// 回落。搬运侧（`materializeScheduledTileIndices`）早就收下了同一档形状、取常量 1，
// 这一条钉的是计算侧与它说同一句话。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=single_element %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP

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
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4,
                               exec_condition = cache_block_split_fused_brc_axis>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>]
    } : memref<1xf32>, memref<1x1x1xf32> -> memref<1x1x1xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %mid = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
                                                                size_from_resource = true,
                                                                life_time_axis_id = -1>} : memref<1x1x1xf32>
    // 广播之前的那个计算 op：向量化轴 `zt` 在这块 tile 上 repeat 是 1。
    %computed = afir.abs %loaded outs(%mid) tmp(%tmp) {
      afir.destination_descs = [#src_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4,
                               exec_condition = cache_block_split_fused_brc_axis>
    } : memref<1x1x1xf32>, memref<1x1x1xf32>, memref<?xui8> -> memref<1x1x1xf32>
    %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?xf32>
    %broadcast = afir.broadcast %computed outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4,
                               exec_condition = cache_block_split_fused_brc_axis>
    } : memref<1x1x1xf32>, memref<?x?x?xf32>, memref<?xui8> -> memref<?x?x?xf32>
    afir.store %broadcast outs(%output) {
      afir.destination_descs = [#gm_out],
      afir.operand_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>]
    } : memref<?x?x?xf32>, memref<128xf32>
    return
  }
}

// IR-LABEL: func.func private @stage0_template0
// tile 循环只有这一层。单元素窗口不建循环，多一层就说明零段被当成「有段」分解了。
// IR: scf.for
// IR-NOT: scf.for

// 两个 tile 的偏移都是常量 0，元素数是常量 1——不是 `local_tensor.get_size`（那是 identity
// 一档，这张图走的是调度那一支），也不是带尾块修正的那条 select 链（零段没有哪一项要换）。
//
// 锚在 vecin 的 DeQue 上：Load 自己那条链里也有一对偏移常量 0 的 subindex，不锚会先捕到它。
// IR: ascendc.que_bind.deque_tensor
// IR: ascendc.reinterpret_cast
// IR: %[[SRC:.*]] = ascendc.local_tensor.subindex %{{.*}}[%c0
// IR: %[[DST:.*]] = ascendc.local_tensor.subindex %{{.*}}[%c0
// IR: %[[COUNT:.*]] = arith.constant 1 : index
// IR-NEXT: ascendc.abs_extend %[[DST]], %[[SRC]], %{{.*}}, %[[COUNT]]

// kernel 文本侧的同一件事：传给算子的元素数就是 1。
// CPP: AbsExtend<float>({{.*}}, {{.*}}, {{.*}}, static_cast<uint32_t>(c1_idx));
