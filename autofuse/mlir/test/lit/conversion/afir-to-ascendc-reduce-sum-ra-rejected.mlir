// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// R 组多于一根轴的 RA 必须被拒绝，而不是编译出一个会把垃圾行加进和里的 kernel。
//
// 输入与 `afir-to-ascendc-reduce-sum-ra.mlir` 是同一份真实调度，唯一的差别是向量化轴由
// `[5, 3]` 扩成 `[4, 5, 3]`：轴 4（z0T）在 descriptor 的完整 `strides` 里本来就是 `"0"`，
// 所以把它并进向量化轴是自洽的，不是硬凑——归约侧两根轴 stride 全 0，R 组因此有两根。
// 那一份能跑完整条流水线，所以这里失败只可能来自 R 组的轴数。
//
// 为什么必须拒绝：`srcShape` 的外维是一个扁平行数，API 按固定行距逐行走。多根外维轴
// 时，只有除最外一根之外每一根都「actual == nominal」，这些行才真正连成一片；否则中间隔着
// 尾块留下的空行，扁平行数会把空行也数成数据行。RA 的外维是 R，多数进去的那些行会被**加进
// 有效和里**——不崩、不越界，数值悄悄是错的。
//
// 要放开它，得先让「这根轴在这个 op 处会不会取尾块」在编译期可判定（今天
// `materializeScheduledAxisExtents` 只给 SSA `Value`，比不了）。见设计文档 3.3。

// RUN: not af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s 2>&1 \
// RUN:   | FileCheck %s

// 能力查询在 SelectImplementation 处就拒了，走不到下降。下降侧还有第二道
// （`isSupportedReduceLayout` 的同一个谓词），去掉这一道时它会接住并报 `failed to legalize`。
// CHECK: error: 'afir.sum' op has no supported Ascend implementation

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
module @reduce_sum_ra_rejected attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>, afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>} {
  func.func private @reduce_sum_ra_two_r_axes(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<1x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"128">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"32">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"32">],strides = [<"0">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z1T", axis_type = TileOuter, size = <"Ceiling((32 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 3>, <id = 3, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 2>, <id = 4, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 5>, <id = 5, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 4>, <id = 6, name = "z1TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((32 / (z1t_size))) / (z1Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 7>, <id = 7, name = "z1Tb", axis_type = BlockInner, size = <"z1Tb_size">, align = <"1">, from = [2], split_pair_other_id = 6>], type = Compute, size_var = [<"z1t_size">, <"z0t_size">, <"z1Tb_size">]>, afir.buffer_ids = array<i64: 0, 1>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_ra_probe_0_B0Y1R0_S0G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "z1t_size" : index
    %3 = afir.tiling_param "z1Tb_size" : index
    %4 = afir.tiling_param "z0t_size" : index
    %5 = afir.symbolic_eval %4 {expr = #afir.expr<"(16 / (__arg0))">} : (index) -> index
    %6 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(32 / (__arg0 * __arg1))">} : (index, index) -> index
    %7 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(32 / (__arg0 * __arg1))">} : (index, index) -> index
    %alloc = memref.alloc(%5, %4, %6, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x?x?xf32, #afir.space<ub>>
    %8 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(16 / (z0t_size))">, <"z0t_size">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"(32 * z0t_size)">, <"32">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [4, 5, 3],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)) * z0t_size)">, <"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(16 / (z0t_size))">, <"z0t_size">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"(32 * z0t_size)">, <"32">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [4, 5, 3],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)) * z0t_size)">, <"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(16 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"(32 / (z1Tb_size * z1t_size))">, #afir.expr<"z1Tb_size">, #afir.expr<"z1t_size">], indexing_maps = [#map, #map1]} : memref<16x32xf32, #afir.space<gm>>, memref<?x?x?x?x?xf32, #afir.space<ub>> -> memref<?x?x?x?x?xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%7, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<1x1x?x?x?xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_2 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %9 = afir.sum %8 outs(%alloc_0) scratch(%alloc_1, %alloc_2) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"0">, <"0">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [4, 5, 3],vectorized_strides = [<"0">, <"0">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(16 / (z0t_size))">, <"z0t_size">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"(32 * z0t_size)">, <"32">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [4, 5, 3],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)) * z0t_size)">, <"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"0">, <"0">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [4, 5, 3],vectorized_strides = [<"0">, <"0">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"(32 / (z1Tb_size * z1t_size))">, #afir.expr<"z1Tb_size">, #afir.expr<"z1t_size">]} : memref<?x?x?x?x?xf32, #afir.space<ub>>, memref<1x1x?x?x?xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<1x1x?x?x?xf32, #afir.space<ub>>
    afir.store %9 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"32">],strides = [<"0">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"0">, <"0">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [4, 5, 3],vectorized_strides = [<"0">, <"0">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"(32 / (z1Tb_size * z1t_size))">, #afir.expr<"z1Tb_size">, #afir.expr<"z1t_size">], indexing_maps = [#map2, #map3]} : memref<1x1x?x?x?xf32, #afir.space<ub>>, memref<1x32xf32, #afir.space<gm>>
    return
  }
}
