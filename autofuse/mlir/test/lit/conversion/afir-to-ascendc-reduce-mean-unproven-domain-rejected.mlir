// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 没有携带 `#afir.reduction_domain` 的 `afir.mean` 必须被拒绝。
//
// 这份输入与 `afir-to-ascendc-reduce-mean-accumulation.mlir` 逐字相同，只去掉了 mean 上的
// `afir.reduction_domain`。那一份能走完整条流水线，所以这里失败**只可能**来自那个属性缺席。
//
// 为什么缺席就必须拒绝：Mean 的分母必须是**完整**逻辑归约域，而
// `afir::analyzeMeanReductionDomain` 只能回答「这张图的轴图怎么说」。跨阶段改写会**合成**
// 自己的 Original 轴——RCore 第二阶段的图里 `Rm` 就是一根 Original 轴、`size` 齐全、`from`
// 为空，分析的每一条判据它都满足，可它数的是第一阶段留下了几份 partial，不是元素数。按它
// 当分母，结果偏大 `R_per_core` 倍，而两个阶段各自看都"合法"，没有一道检查会响。
//
// 「输入代表的是不是完整域」只在 bridge 产物上还看得见（`AFIRMemoryAllocate` 之后 reduce 的
// 输入与 `afir.load` 之间没有 SSA 边），所以由 `afir::annotateReductionDomains` 在那里证一次
// 写进 IR。**缺席即未证明**，这里只能拒绝——不得退回轴图的结论，那正是会给出错误分母的路。

// RUN: not af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s 2>&1 \
// RUN:   | FileCheck %s

// 能力查询在 SelectImplementation 处就拒了，走不到下降。下降侧还有第二道（同一个判据），
// 去掉这一道时它会接住并报 `failed to legalize`。
// CHECK: error: 'afir.mean' op has no supported Ascend implementation

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
module @reduce_mean_unproven_domain_rejected attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>, ascendc.target_aiv_num = 48 : i64, ascendc.target_ub_size = 196608 : i64} {
  func.func private @reduce_mean_later_stage_rejected_body(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 3, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 2>, <id = 4, name = "z1T", axis_type = TileOuter, size = <"Ceiling((32 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 5>, <id = 5, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 4>, <id = 6, name = "z0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((16 / (z0t_size))) / (z0Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 7>, <id = 7, name = "z0Tb", axis_type = BlockInner, size = <"z0Tb_size">, align = <"1">, from = [2], split_pair_other_id = 6>], type = Compute, size_var = [<"z0t_size">, <"z1t_size">, <"z0Tb_size">]>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_16x32_0_B0Y0R1_S0G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "z1t_size" : index
    %3 = afir.tiling_param "z0t_size" : index
    %4 = afir.tiling_param "z0Tb_size" : index
    %5 = afir.symbolic_eval %4, %3 {expr = #afir.expr<"(16 / (__arg0 * __arg1))">} : (index, index) -> index
    %6 = afir.symbolic_eval %2 {expr = #afir.expr<"(32 / (__arg0))">} : (index) -> index
    %7 = afir.symbolic_eval %4, %3 {expr = #afir.expr<"(16 / (__arg0 * __arg1))">} : (index, index) -> index
    %alloc = memref.alloc(%5, %4, %3, %6, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x?x?xf32, #afir.space<ub>>
    %8 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"(32 / (z1t_size))">, <"z1t_size">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"(32 / (z1t_size))">, <"z1t_size">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"(32 / (z1t_size))">, #afir.expr<"z1t_size">], indexing_maps = [#map, #map1]} : memref<16x32xf32, #afir.space<gm>>, memref<?x?x?x?x?xf32, #afir.space<ub>> -> memref<?x?x?x?x?xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_1 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_2 = memref.alloc(%7, %4, %3) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x1x1xf32, #afir.space<ub>>
    %9 = afir.mean %8 outs(%alloc_2) scratch(%alloc_0, %alloc_1) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"(32 / (z1t_size))">, <"z1t_size">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">]} : memref<?x?x?x?x?xf32, #afir.space<ub>>, memref<?x?x?x1x1xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<?x?x?x1x1xf32, #afir.space<ub>>
    afir.store %9 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">], indexing_maps = [#map2, #map3]} : memref<?x?x?x1x1xf32, #afir.space<ub>>, memref<16x32xf32, #afir.space<gm>>
    return
  }
}
