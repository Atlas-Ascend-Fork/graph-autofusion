// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 一次算完（非累加）的 `afir.mean`：`ReduceSum` 之后无条件乘一次 `1 / N`。
//
// 输入是 `afir-to-ascendc-reduce-allload-ar.mlir` 的同一份真实候选（AllLoad 模板，
// `f32[16, 32]` 归约末维），只把算子换成 `afir.mean`。AllLoad 的身份与那一份同样成立：
// 归约轴 `z1` 整根不切（axes 里没有 `z1T`/`z1t`），所以 `accumulationAxes` 为空。
//
// 本用例钉两件事：
//   1. 非累加形态**不发** `scf.if`——`Muls` 无条件执行一次。多包一层 region 什么也不多做，
//      却会让这条路径的产物与开放累加能力之前不一致。
//   2. 分母仍然取**完整逻辑归约域**（Original 轴 `z1` 的 size = 32）。这一档里它恰好等于
//      tile 的 actual（R 没被切，两者必然相等），所以这里**看不出**两种取法的差别；能看出
//      差别的是 `afir-to-ascendc-reduce-mean-accumulation.mlir`。两份一起才覆盖
//      「所有轮次用同一个 N」这条验收。
//
// `--implicit-check-not=ascendc.add_l2` 与那一份同理：一次算完不该有任何跨轮累加。

// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --implicit-check-not=ascendc.add_l2

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3)>
module @reduce_mean_allload_ar attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = true, afir.module_kind = "plan", afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>, afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>} {
  func.func private @reduce_mean_allload_ar_body(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<16x1xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"64">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"1">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"1">],strides = [<"1">, <"0">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 3, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 2>, <id = 4, name = "z0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((16 / (z0t_size))) / (z0Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 5>, <id = 5, name = "z0Tb", axis_type = BlockInner, size = <"z0Tb_size">, align = <"1">, from = [2], split_pair_other_id = 4>], type = Compute, size_var = [<"z0t_size">, <"z0Tb_size">]>, afir.buffer_ids = array<i64: 0, 1>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_allload_16x32_0_B0Y0_S2G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "z0t_size" : index
    %3 = afir.tiling_param "z0Tb_size" : index
    %4 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(16 / (__arg0 * __arg1))">} : (index, index) -> index
    %5 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(16 / (__arg0 * __arg1))">} : (index, index) -> index
    %alloc = memref.alloc(%4, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x32xf32, #afir.space<ub>>
    %6 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"32">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"1">],vectorized_axis = [3, 1],vectorized_strides = [<"32">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"32">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"1">],vectorized_axis = [3, 1],vectorized_strides = [<"32">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [4, 5, 3, 1], loop_axis = 5>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"32">], indexing_maps = [#map, #map1]} : memref<16x32xf32, #afir.space<gm>>, memref<?x?x?x32xf32, #afir.space<ub>> -> memref<?x?x?x32xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%5, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x1xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_2 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %7 = afir.mean %6 outs(%alloc_0) scratch(%alloc_1, %alloc_2) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">],vectorized_axis = [3, 1],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"32">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"1">],vectorized_axis = [3, 1],vectorized_strides = [<"32">, <"1">],role = <compute>>], afir.reduction_domain = #afir.reduction_domain<axes = [1], sizes = [<"32">]>, afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">],vectorized_axis = [3, 1],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [4, 5, 3, 1], loop_axis = 5>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">]} : memref<?x?x?x32xf32, #afir.space<ub>>, memref<?x?x?x1xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<?x?x?x1xf32, #afir.space<ub>>
    afir.store %7 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"1">],strides = [<"1">, <"0">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">],vectorized_axis = [3, 1],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [4, 5, 3, 1], loop_axis = 5>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">], indexing_maps = [#map2, #map3]} : memref<?x?x?x1xf32, #afir.space<ub>>, memref<16x1xf32, #afir.space<gm>>
    return
  }
}


// CHECK-LABEL: func.func private @reduce_mean_allload_ar_body

// 一次搬入：整块 `A_tile x 32`。R 整根在 tile 里，二维 DMA 折成一维是对的。
// CHECK: ascendc.data_copy_pad_extend

// AR 的物理内维是被归约的 R，32B 对齐落在它身上（f32 下 8 个元素一组）。
// CHECK: %[[RDIV:[a-zA-Z0-9_]+]] = arith.ceildivsi %{{[a-zA-Z0-9_]+}}, %[[C8:[a-zA-Z0-9_]+]] : index
// CHECK: %[[RALIGN:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = arith.mulsi_extended %[[RDIV]], %[[C8]] : index
// CHECK: cf.assert %{{[a-zA-Z0-9_]+}}, "reduce aligned extent must not overflow"

// AR 先把对齐引入的那一段填成单位元，再归约；`layout = 0` 是 AR。
// CHECK: ascendc.duplicate_l1
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK-SAME: layout = 0
// CHECK-SAME: srcInnerPad = true

// 分母：Original 轴 `z1` 的 size = 32，`index -> i64 -> f32`，中间不经 i32。
// CHECK: %[[N:.*]] = arith.constant 32 : index
// CHECK: %[[N64:.*]] = arith.index_cast %[[N]] : index to i64
// CHECK: %[[DEN:.*]] = arith.sitofp %[[N64]] : i64 to f32
// CHECK: %[[SCALE:.*]] = arith.divf %{{.*}}, %[[DEN]] : f32

// 非累加形态屏障与 `Muls` 之间没有 `scf.if`：无条件除一次。
// CHECK: ascendc.pipe_barrier pipe_v
// CHECK-NEXT: ascendc.muls_l2 %[[DST:.*]], %[[DST]], %[[SCALE]], %{{.*}}

// 结果搬出。
// CHECK: ascendc.data_copy_pad_extend
