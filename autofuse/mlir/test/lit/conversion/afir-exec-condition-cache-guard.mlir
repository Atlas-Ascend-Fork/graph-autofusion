// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.sched` 的 `exec_condition` → 循环体里的 `scf.if`（缓存守卫）。
//
// 图取广播的常量源一档（与 `afir-broadcast-duplicate.mlir` 同一张图），因为**上游只给广播链
// 标这个条件**（`optimize/autoschedule/node_cache_marker.cpp`）。守卫本身与是哪个算子无关，
// 由 `afir.sched` 一处决定。
//
// 跑整条 lowering，不是只跑 `--afir-exec-condition`：要钉的事实有一半在守卫**之外**——
// `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` 与流水屏障必须每趟照发，只有算子本身进
// `if`。这与 legacy 的 `Loop::GenerateBody` 逐字一致，而它只在资源物化与同步都跑过之后
// 才看得见。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=CNT
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=exec_condition %t.mlir \
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
// 分核序号先落到块轴的坐标上：谓词里的全局序号用的是它，不是 `GetBlockIdx()` 本身。
// IR: %[[BLOCKIDX:.*]] = ascendc.get_block_idx
// IR: %[[BLOCKPOS:.*]] = arith.remsi %[[BLOCKIDX]], %{{.*}} {afir.axis_binding = #afir.axis_binding<axis = 3, role = axis_index>}
// IR: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step
// 谓词发在循环体开头，两个算子共用一份——legacy 那边也是每个循环一个 `bool`。
// IR: %[[FIRST:.*]] = arith.cmpi eq, %[[IV]], %[[ZERO:.*]] : index
// 分块轴源轴（`zT`）上的全局序号：分核序号 × 块内长度 + 块内序号。
// IR: %[[SCALED:.*]] = arith.muli %[[BLOCKPOS]], %{{.*}} : index
// IR: %[[LINEAR:.*]] = arith.addi %[[SCALED]], %[[IV]] : index
// 相位归零 = 一个广播复用周期的起点。模数是 `zT` 的循环次数。
// IR: %[[PHASE:.*]] = arith.remsi %[[LINEAR]], %{{.*}} : index
// IR: %[[START:.*]] = arith.cmpi eq, %[[PHASE]], %[[ZERO]] : index
// IR: %[[GUARD:.*]] = arith.ori %[[FIRST]], %[[START]] : i1
// 队列的 Alloc 在 `if` **外面**：被跳过的只有搬运本身，缓冲的 slot 每趟照旧轮转一遍。
// IR: ascendc.que_bind.alloc_tensor
// IR: scf.if %[[GUARD]] {
// IR:   ascendc.data_copy_pad_extend
// IR: }
// 同步边也在外面。
// IR: ascendc.pipe_barrier pipe_mte2
// IR: ascendc.que_bind.enque_tensor
// IR: ascendc.que_bind.deque_tensor
// IR: scf.if %[[GUARD]] {
// IR:   ascendc.duplicate_l2
// IR: }
// IR: ascendc.pipe_barrier pipe_v
// Store 没有这个条件，不该被包起来：它每趟都要写一片出去，`if` 里只会写第一趟。
// IR: ascendc.que_bind.deque_tensor
// IR: ascendc.data_copy_pad_extend
// IR-NOT: scf.if

// 整个 kernel body 里恰好两个守卫，一份谓词。少一个是漏包，多一个是谓词没有被复用
// （`GuardCache` 失效），两种都只在计数上看得出来。
// CNT-LABEL: func.func private @stage0_template0
// CNT-COUNT-2: scf.if
// CNT-NOT: scf.if

// CPP: AllocTensor<uint8_t>()
// CPP: if (
// CPP: DataCopyPadExtend<float>
// CPP: }
// CPP: AscendC::PipeBarrier<PIPE_MTE2>();
// CPP: if (
// CPP: AscendC::Duplicate(
// CPP: }
// CPP: AscendC::PipeBarrier<PIPE_V>();
