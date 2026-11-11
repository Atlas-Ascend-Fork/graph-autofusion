// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 向量化轴被剔空、但地址在非向量化轴上前进：偏移要逐轴算出来。
//
// 这一份是真实 dump：`[1024,1] → [1024,128]` 广播图的 `B0Y1` 模板（广播轴 `z1` 切出
// `z1T`/`z1t`，`z1T` 与 `z0` 合成 `z0z1T` 再分核，所以循环走的是合轴之后那根，广播轴排在
// tile 外）。descriptor 一个字没改，只把 template 序号与 tiling_key 归 0。
//
// 看 Load 的两个 descriptor：局部侧 `vectorized_axis = [3]`，该轴 `repeats = 1` /
// `vectorized_strides = 0` → `collapseVectorizedAxes` 按 extent 1 剔掉，一段都不剩；
// 而 `z0`（非向量化轴）上 `strides = 1`，循环每一趟该读下一个元素。
//
// 此前 `materializeScheduledTileIndices` 在「一段都不剩」这一档无条件返回偏移 0，理由写的是
// 「这块 tile 就是一个元素，地址也不随 tile 前进」。两句都不成立：局部 tile 是 `1024x1x1`
// （1024 个元素，只是只有一根轴参与向量化），地址在非向量化的循环轴上照样走。生成的
// kernel 每趟都读 GM 第 0 个元素，编得过、跑得完、输出每一行都等于 `input0[0]`
// （CPU 模拟器实测，见 `test/e2e/cpu/cases/load_broadcast_segmented_store`）。
//
// 现在偏移按 `Σ 坐标(a) × strides[a]` 算（`materializeTensorOffset`），坐标由
// `materializeAxisIndexFromLinear` 从线性序号按轴树拆出来。本份钉的就是那条分解：
// Load 的偏移必须是 `(线性序号 / size(z1T)) % 1024 × 1`，不是常量 0。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s

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
  func.func private @stage0_template0(%arg0: memref<1024x1xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"1024">, #afir.expr<"1">]}, %arg1: memref<1024x128xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"524288">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"1024">, #afir.expr<"128">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1024">, <"1">],strides = [<"1">, <"0">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"1024">, <"128">],strides = [<"128">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"1024">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"128">, align = <"-1">>, <id = 2, name = "z1T", axis_type = TileOuter, size = <"Ceiling((128 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 3>, <id = 3, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 2>, <id = 4, name = "z0z1T", axis_type = Merged, size = <"(1024 * Ceiling((128 / (z1t_size))))">, align = <"1">, from = [0, 2], split_pair_other_id = 0>, <id = 5, name = "z0z1TB", axis_type = BlockOuter, size = <"Ceiling((1024 * Ceiling((128 / (z1t_size))) / (z0z1Tb_size)))">, align = <"1">, from = [4], split_pair_other_id = 6>, <id = 6, name = "z0z1Tb", axis_type = BlockInner, size = <"z0z1Tb_size">, align = <"1">, from = [4], split_pair_other_id = 5>], type = Compute, size_var = [<"z1t_size">, <"z0z1Tb_size">]>, afir.buffer_ids = array<i64: 0>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_broadcast_segmented_store_1024x1_0_B0Y1_S0G0C1", afir.queue_ids = array<i64: 0, 1>, afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "z1t_size" : index
    %2 = afir.symbolic_eval %1 {expr = #afir.expr<"(128 / (__arg0))">} : (index) -> index
    %alloc = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<1024x1x1xf32, #afir.space<ub>>
    %3 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1024">, <"1">, <"1">],strides = [<"1">, <"0">, <"0">],vectorized_axis = [3],vectorized_strides = [<"0">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1024">, <"1">],strides = [<"1">, <"0">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1024">, <"1">, <"1">],strides = [<"1">, <"0">, <"0">],vectorized_axis = [3],vectorized_strides = [<"0">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 3], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"1024">, #afir.expr<"1">, #afir.expr<"1">], indexing_maps = [#map4, #map5]} : memref<1024x1xf32, #afir.space<gm>>, memref<1024x1x1xf32, #afir.space<ub>> -> memref<1024x1x1xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%2, %1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<1024x?x?xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %4 = afir.broadcast %3 outs(%alloc_0) tmp(%alloc_1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1024">, <"(128 / (z1t_size))">, <"z1t_size">],strides = [<"128">, <"z1t_size">, <"1">],vectorized_axis = [3],vectorized_strides = [<"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1024">, <"1">, <"1">],strides = [<"1">, <"0">, <"0">],vectorized_axis = [3],vectorized_strides = [<"0">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1024">, <"(128 / (z1t_size))">, <"z1t_size">],strides = [<"128">, <"z1t_size">, <"1">],vectorized_axis = [3],vectorized_strides = [<"1">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 3], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"1024">, #afir.expr<"(128 / (z1t_size))">, #afir.expr<"z1t_size">]} : memref<1024x1x1xf32, #afir.space<ub>>, memref<1024x?x?xf32, #afir.space<ub>>, memref<?xui8, #afir.space<ub>> -> memref<1024x?x?xf32, #afir.space<ub>>
    afir.store %4 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1024">, <"128">],strides = [<"128">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 2, 3],repeats = [<"1024">, <"(128 / (z1t_size))">, <"z1t_size">],strides = [<"128">, <"z1t_size">, <"1">],vectorized_axis = [3],vectorized_strides = [<"1">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 3], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"1024">, #afir.expr<"(128 / (z1t_size))">, #afir.expr<"z1t_size">], indexing_maps = [#map6, #map7]} : memref<1024x?x?xf32, #afir.space<ub>>, memref<1024x128xf32, #afir.space<gm>>
    return
  }}

// CHECK-LABEL: func.func private @stage0_template0
// CHECK: scf.for %[[IV:.*]] = %c0
// 线性序号：分核序号 × 块内长度 + 归纳变量。
// CHECK: %[[LINEAR:.*]] = arith.addi %{{.*}}, %[[IV]]
// 拆出 `z0`：先除以内侧那一支（`z1T`）的循环次数，再按 `z0` 自己的循环次数取模。
// CHECK: %[[Q:.*]] = arith.divsi %[[LINEAR]], %{{.*}}
// CHECK: %[[Z0:.*]] = arith.remsi %[[Q]], %c1024
// 源轴 stride 为 1，偏移直接取 z0 的序号。
// CHECK: %[[SRC:.*]] = ascendc.global_tensor.subindex %{{.*}}[%[[Z0]]]
// CHECK: ascendc.data_copy_pad_extend %{{.*}}, %[[SRC]]
