// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// **`Load → Store` 直连：`MTE2 → MTE3` 的流水线同步，以及 Store 的 GM 偏移。**
//
// 这一份是**真实 dump**：`[1,8] → [128,8]` 广播图的 `B0Y1` 模板。广播轴 `z0` 排到 tile 外，
// `RemoveRedundantBroadcastNode` 把 `brc` 整个删掉，图退化成 `Load → Store`——中间没有 V 计算。
// descriptor 一个字没改，只把 template 序号与 tiling_key 归 0。
//
// 钉两件事：
//
// 1. **同步**。`AFIRPipelineSync` 此前只认 `MTE2 → Vector`、`Vector → MTE3`、`Vector → Vector`，
//    这条直连交接报 `unsupported pipeline synchronization from MTE2 to MTE3`。顺序只能由队列的
//    enque / deque 给——`PipeBarrier` 只在同一条流水内排序，MTE2 与 MTE3 之间没有 barrier 可用。
//    所以这一档不做 barrier 回落：拿不到队列就报错，不静默放过。
//
// 2. **Store 的 GM 偏移**。Store 的**局部** descriptor 在 `z0` 上 stride 是 0（那是被删掉的广播
//    语义，每趟重读同一块 UB），而 GM 目的地 `strides = [8, 1]` 每趟前进 8。只看局部会漏掉
//    `z0` 那一项，输出只有第一行对。所以偏移的权威是 GM 侧 descriptor
//    （`materializeTensorOffset` 的 `OffsetContext::strideOf`，GM 优先）。
//
// 数值由 `test/e2e/cpu/cases/load_broadcast_segmented_store` 的 `_1x8` 覆盖，但**那条用例跑到的
// 是运行期 tiling 选中的模板**，选的是 `Y0`；`Y1` 这一份只有本 lit 钉得住。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=load_store_mte2_mte3 %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(%arg0: memref<1x8xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"8">]}, %arg1: memref<128x8xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"128">, #afir.expr<"8">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"8">],strides = [<"0">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"128">, <"8">],strides = [<"8">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"128">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"8">, align = <"-1">>, <id = 2, name = "z1T", axis_type = TileOuter, size = <"Ceiling((8 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 3>, <id = 3, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 2>, <id = 4, name = "z0z1T", axis_type = Merged, size = <"(128 * Ceiling((8 / (z1t_size))))">, align = <"1">, from = [0, 2], split_pair_other_id = 0>, <id = 5, name = "z0z1TB", axis_type = BlockOuter, size = <"Ceiling((128 * Ceiling((8 / (z1t_size))) / (z0z1Tb_size)))">, align = <"1">, from = [4], split_pair_other_id = 6>, <id = 6, name = "z0z1Tb", axis_type = BlockInner, size = <"z0z1Tb_size">, align = <"1">, from = [4], split_pair_other_id = 5>], type = Compute, size_var = [<"z1t_size">, <"z0z1Tb_size">]>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_broadcast_segmented_store_1x8_0_B0Y1_S0G0C1", afir.queue_ids = array<i64: 0>, afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "z1t_size" : index
    %1 = afir.symbolic_eval %0 {expr = #afir.expr<"(8 / (__arg0))">} : (index) -> index
    %alloc = memref.alloc(%1, %0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<1x?x?xf32, #afir.space<ub>>
    %2 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1">, <"(8 / (z1t_size))">, <"z1t_size">],strides = [<"0">, <"z1t_size">, <"1">],vectorized_axis = [3],vectorized_strides = [<"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"8">],strides = [<"0">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1">, <"(8 / (z1t_size))">, <"z1t_size">],strides = [<"0">, <"z1t_size">, <"1">],vectorized_axis = [3],vectorized_strides = [<"1">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 3], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"(8 / (z1t_size))">, #afir.expr<"z1t_size">], indexing_maps = [#map4, #map5]} : memref<1x8xf32, #afir.space<gm>>, memref<1x?x?xf32, #afir.space<ub>> -> memref<1x?x?xf32, #afir.space<ub>>
    afir.store %2 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"128">, <"8">],strides = [<"8">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1">, <"(8 / (z1t_size))">, <"z1t_size">],strides = [<"0">, <"z1t_size">, <"1">],vectorized_axis = [3],vectorized_strides = [<"1">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 3], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"(8 / (z1t_size))">, #afir.expr<"z1t_size">], indexing_maps = [#map6, #map7]} : memref<1x?x?xf32, #afir.space<ub>>, memref<128x8xf32, #afir.space<gm>>
    return
  }}

// IR-LABEL: func.func private @stage0_template0
// 队列的 enque / deque 就是这条交接的同步；`PIPE_MTE2` 的屏障在入队之前。
// IR: ascendc.pipe_barrier pipe_mte2
// IR: ascendc.que_bind.enque_tensor
// IR: ascendc.que_bind.deque_tensor
// 中间不出现 V 管线的屏障——这条路上没有 V 计算，插 V 屏障说明分派走错了档。
// IR-NOT: afir.pipe_barrier
// IR-NOT: ascendc.pipe_barrier pipe_v

// 两条搬运：进来一条、出去一条，中间只有队列那一对，没有计算。
// CPP: DataCopyPadExtend<float>(
// CPP: .EnQue(
// CPP: .DeQue<
// CPP: DataCopyPadExtend<float>(
// CPP-NOT: AscendC::Broadcast
// CPP-NOT: AscendC::Duplicate
