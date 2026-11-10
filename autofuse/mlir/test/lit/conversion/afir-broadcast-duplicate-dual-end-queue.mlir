// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.broadcast` 常量源一档（`Duplicate`）的**双端队列**形态。
//
// 队列资源按 `(space, queue, buffer_id)` 共享：广播源 tile 与直连 store 的 load tile 写
// 同一个 `buffer_id = 0`，就落在同一条 vecin 队列上。直连链（load → store，无矢量消费
// 者）把这条队列物化成 `TQueBind<VECIN, VECOUT>`，而广播的标量读走的是同一条队列的
// deque——`scalarReadEvent` 按队列类型答「源在哪条管线上被写」，双端队列的类型是
// `que_bind`，不带 `queue` 的 position。认不出类型就答不了 MTE2_S，capability 层整图
// 报不支持、回落 legacy。
//
// 这一条钉的是：双端队列的源侧（`getSrcPosition`）是 VECIN 时，标量读事件的答案与
// 单端 vecin 队列相同——MTE2_S。三个 RUN 分别钉 pass 产物（双端队列确实物化）、
// conversion 产物（事件对生成于标量读之前）、能力层选择（这一档选 duplicate 实现）。

// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync %s | FileCheck %s --check-prefix=SYNC
// RUN: af-opt --target=Ascend910B1 --afir-plan-candidate-lowering-pipeline %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"128">], strides = [#afir.expr<"1">], role = <output>>
#gm_scalar_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], role = <output>>
// 源 tile：向量化轴（2 号，TileInner）上 repeat 1、stride 0——「每个迭代读同一份」，
// 也就是这一档的判据。
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
                                                                     life_time_axis_id = -1>},
      %scalar_out: memref<1xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 1>,
                                  afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4">,
                                                                       life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#gm_in, #gm_out, #gm_scalar_out],
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
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>]
    } : memref<?x?x?xf32>, memref<128xf32>
    // 直连链：tile 与广播源 tile 同一个 buffer_id = 0，共享同一条 vecin 队列。load 的
    // 消费者只有 store（MTE2 → MTE3 直连），这条共享队列被物化为双端 TQueBind。
    %direct_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                                                         slot_count = 1, queue_depth = 2,
                                                                         size_from_resource = true,
                                                                         life_time_axis_id = -1>} : memref<1x1x1xf32>
    %direct_loaded = afir.load %input outs(%direct_tile) {
      afir.destination_descs = [#src_tile],
      afir.operand_descs = [#gm_in],
      afir.result_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>]
    } : memref<1xf32>, memref<1x1x1xf32> -> memref<1x1x1xf32>
    afir.store %direct_loaded outs(%scalar_out) {
      afir.destination_descs = [#gm_scalar_out],
      afir.operand_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>]
    } : memref<1x1x1xf32>, memref<1xf32>
    return
  }
}

// 直连链把共享队列（buffer_id = 0）物化成双端；广播目的地的队列（buffer_id = 1）仍是
// 单端 vecout——只有共享的那条换类型。
// SYNC: !ascendc.que_bind<vecin, vecout
// SYNC: !ascendc.queue<vecout

// 事件对在标量读之前：双端队列的源侧是 VECIN（MTE2 写入），标量读的同步答案与
// 单端 vecin 队列相同。
// IR: func.func private @stage0_template0
// IR: scf.for
// IR: ascendc.set_flag {{.*}}mte2_s
// IR: ascendc.wait_flag {{.*}}mte2_s
// IR: ascendc.local_tensor.get_value
// IR: ascendc.duplicate_l2
// IR-NOT: afir.broadcast

// SELECT: ascend.implementation = {id = "broadcast.tensor.duplicate", kind = "ascendc_api"}
