// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RCore（R 轴也参与分核）模板的下降。输入是 `load_sum_store_rcore_ar_8x1024` 的真实候选
// （`f32[8, 1024] -> [8, 1]`，`schedule plan 1`），两个 stage 都在这一份模块里。
//
// 这条用例补的是「调度模板 × 归约形态」矩阵里 RCore 两格——
// `2026-09-20-two-dim-transfer-addressing-design.md` 8.5 的六组合验收项此前只有四格有 lit。
// 一份输入同时钉住两格，是因为 RCore 的两格不是两张独立的图：
//
//   stage 0 = phase 1，`..._r_multicore_phase_1_graph_B1Y0R1_S1G0C0`，在 `[A, Rm_t]` 上做
//             AR，结果写 GM workspace；
//   stage 1 = phase 2，`..._r_multicore_phase_2_graph_B0Y1R0_S1G1C0`，从 workspace 的
//             `[Rm, A]` 上做 **RA**，结果写 kernel 输出。
//
// 「RA 是 RCore phase 2 的硬前置」（`2026-09-21-afir-reduce-sum-ra-lowering-design.md` 3.3）
// 在这里是字面的：同一个 kernel 里两种 layout 都要传对。
//
// 另外钉住三件只在 RCore 下成立的事：
//   - 两个 stage 共用 `afir.workspace_tensor_ids = array<i64: 2>` 的那一块 GM；
//   - `afir.var_relations` 把 stage 0 算出的 `Rm_org_size` 绑到 stage 1 的同名变量上，
//     这条跨 stage 的绑定必须活着穿过整条下降管线；
//   - stage 1 不生成 `ReduceInit`：RA 的被归约维在物理外侧，没有要填单位元的那一段，而
//     `reduce_init.h` 的填充按行主序写死，生成是往错地址写。这条用 `CHECK-NOT` 写在
//     stage 1 的 `CHECK-LABEL` 块内，所以不会被 stage 0 那一处 `ReduceInit` 满足。
//
// 两个目标各生成一遍：reduce 的下降没有任何目标相关分支，两份输出逐字节相同。950 无
// E2E，只能比对文本，这就是它在矩阵验收里的位置。

// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s

// Swapping scratch uses together with their roles must preserve the whole lowered module.
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s > %t.original
// RUN: sed 's/scratch(%alloc_2, %alloc_3)/scratch(%alloc_3, %alloc_2)/g; s/\["api_scratch", "round_result"\]/["round_result", "api_scratch"]/g' %s > %t.reordered.mlir
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %t.reordered.mlir > %t.reordered
// RUN: diff %t.original %t.reordered

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
module @reduce_rcore_subgraph0_plan0 attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>, afir.var_relations = [{dst_stage = 1 : i64, dst_var = "A_org_size", src_expr = #afir.expr<"8">, src_stage = 0 : i64}, {dst_stage = 1 : i64, dst_var = "Rm_org_size", src_expr = #afir.expr<"Ceiling((Ceiling((1024 / (z1t_size))) / (z1Tt_size)))">, src_stage = 0 : i64}], afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1, 1]>} {
  func.func private @stage0_template0(%arg0: memref<8x1024xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32768">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"8">, #afir.expr<"1024">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"8">, <"1024">],strides = [<"1024">, <"1">],role = <input>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"8">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"1024">, align = <"-1">>, <id = 2, name = "z0T", axis_type = TileOuter, size = <"Ceiling((8 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 3, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 2>, <id = 4, name = "z1T", axis_type = TileOuter, size = <"Ceiling((1024 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 5>, <id = 5, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 4>, <id = 6, name = "z1TT", axis_type = TileOuter, size = <"Ceiling((Ceiling((1024 / (z1t_size))) / (z1Tt_size)))">, align = <"1">, from = [4], split_pair_other_id = 7>, <id = 7, name = "z1Ttb", axis_type = BlockInner, size = <"z1Tt_size">, align = <"1">, from = [4], split_pair_other_id = 6>, <id = 8, name = "z0Tz1TTB", axis_type = BlockOuter, size = <"(Ceiling((8 / (z0t_size))) * Ceiling((Ceiling((1024 / (z1t_size))) / (z1Tt_size))))">, align = <"1">, from = [2, 6], split_pair_other_id = 0>], type = Compute, size_var = [<"z0t_size">, <"z1t_size">, <"z1Tt_size">]>, afir.buffer_ids = array<i64: 0, 1>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "reduce_rcore_0_r_multicore_phase_1_graph_B1Y0R1_S1G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"Max(0, (4 * Max(Max(8, (8 * Ceiling((Ceiling((1024 / (z1t_size))) / (z1Tt_size))))), z0t_size)))">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "z1t_size" : index
    %3 = afir.tiling_param "z1Tt_size" : index
    %4 = afir.tiling_param "z0t_size" : index
    %5 = afir.symbolic_eval %4 {expr = #afir.expr<"(8 / (__arg0))">} : (index) -> index
    %6 = afir.symbolic_eval %4 {expr = #afir.expr<"(8 / (__arg0))">} : (index) -> index
    %7 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">} : (index, index) -> index
    %8 = afir.symbolic_eval %4 {expr = #afir.expr<"(8 / (__arg0))">} : (index) -> index
    %alloc = memref.alloc(%5, %4) {afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(32 * Ceiling((Ceiling((1024 / (z1t_size))) / (z1Tt_size))))">, life_time_axis_id = -1>} : memref<?x?x1x1x1xf32, #afir.space<gm>>
    %alloc_0 = memref.alloc(%6, %4, %7, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x?x?xf32, #afir.space<ub>>
    %9 = afir.load %arg0 outs(%alloc_0) {afir.destination_descs = [#afir.tensor_desc<axis = [2, 3, 6, 7, 5],repeats = [<"(8 / (z0t_size))">, <"z0t_size">, <"(1024 / (z1Tt_size * z1t_size))">, <"z1Tt_size">, <"z1t_size">],strides = [<"(1024 * z0t_size)">, <"1024">, <"(z1Tt_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"8">, <"1024">],strides = [<"1024">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [2, 3, 6, 7, 5],repeats = [<"(8 / (z0t_size))">, <"z0t_size">, <"(1024 / (z1Tt_size * z1t_size))">, <"z1Tt_size">, <"z1t_size">],strides = [<"(1024 * z0t_size)">, <"1024">, <"(z1Tt_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [8, 7, 3, 5], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"(8 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"(1024 / (z1Tt_size * z1t_size))">, #afir.expr<"z1Tt_size">, #afir.expr<"z1t_size">], indexing_maps = [#map, #map1]} : memref<8x1024xf32, #afir.space<gm>>, memref<?x?x?x?x?xf32, #afir.space<ub>> -> memref<?x?x?x?x?xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%8, %4) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x1x1x1xf32, #afir.space<ub>>
    %alloc_2 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_3 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %10 = afir.sum %9 outs(%alloc_1) scratch(%alloc_2, %alloc_3) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [2, 3, 6, 7, 5],repeats = [<"(8 / (z0t_size))">, <"z0t_size">, <"1">, <"1">, <"1">],strides = [<"z0t_size">, <"1">, <"0">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [2, 3, 6, 7, 5],repeats = [<"(8 / (z0t_size))">, <"z0t_size">, <"(1024 / (z1Tt_size * z1t_size))">, <"z1Tt_size">, <"z1t_size">],strides = [<"(1024 * z0t_size)">, <"1024">, <"(z1Tt_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [2, 3, 6, 7, 5],repeats = [<"(8 / (z0t_size))">, <"z0t_size">, <"1">, <"1">, <"1">],strides = [<"z0t_size">, <"1">, <"0">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [8, 7, 3, 5], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"(8 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">]} : memref<?x?x?x?x?xf32, #afir.space<ub>>, memref<?x?x1x1x1xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<?x?x1x1x1xf32, #afir.space<ub>>
    afir.store %10 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [2, 3, 6, 7, 5],repeats = [<"(8 / (z0t_size))">, <"z0t_size">, <"1">, <"1">, <"1">],strides = [<"z0t_size">, <"1">, <"8">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [2, 3, 6, 7, 5],repeats = [<"(8 / (z0t_size))">, <"z0t_size">, <"1">, <"1">, <"1">],strides = [<"z0t_size">, <"1">, <"0">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [8, 7, 3, 5], loop_axis = 8>, afir.symbolic_shape = [#afir.expr<"(8 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">], indexing_maps = [#map2, #map2]} : memref<?x?x1x1x1xf32, #afir.space<ub>>, memref<?x?x1x1x1xf32, #afir.space<gm>>
    return
  }
  func.func private @stage1_template0(%arg0: memref<1x?xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(4 * A_org_size)">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"A_org_size">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"A_org_size">],strides = [<"0">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "Rm", axis_type = Original, size = <"Rm_org_size">, align = <"1">, split_pair_other_id = 0>, <id = 1, name = "A", axis_type = Original, size = <"A_org_size">, align = <"1">, split_pair_other_id = 0>, <id = 2, name = "AT", axis_type = TileOuter, size = <"Ceiling((A_org_size / (At_size)))">, align = <"1">, from = [1], split_pair_other_id = 3>, <id = 3, name = "At", axis_type = TileInner, size = <"At_size">, align = <"1">, from = [1], split_pair_other_id = 2>, <id = 4, name = "RmT", axis_type = TileOuter, size = <"Ceiling((Rm_org_size / (Rmt_size)))">, align = <"1">, from = [0], split_pair_other_id = 5>, <id = 5, name = "Rmt", axis_type = TileInner, size = <"Rmt_size">, align = <"1">, from = [0], split_pair_other_id = 4>, <id = 6, name = "ATB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((A_org_size / (At_size))) / (ATb_size)))">, align = <"1">, from = [2], split_pair_other_id = 7>, <id = 7, name = "ATb", axis_type = BlockInner, size = <"ATb_size">, align = <"1">, from = [2], split_pair_other_id = 6>], type = Compute, size_var = [<"A_org_size">, <"Rm_org_size">, <"At_size">, <"Rmt_size">, <"ATb_size">]>, afir.buffer_ids = array<i64: 0, 1>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "reduce_rcore_0_r_multicore_phase_2_graph_B0Y1R0_S1G1C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"Max(0, (4 * Max(Max(Max(Max(Max(1, At_size), A_org_size), (A_org_size * Rmt_size)), (A_org_size * Rm_org_size)), (ATb_size * At_size))))">, afplan.variant = #afplan.variant<stage = 1, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "At_size" : index
    %3 = afir.tiling_param "ATb_size" : index
    %4 = afir.tiling_param "Rmt_size" : index
    %5 = afir.tiling_param "A_org_size" : index
    %6 = afir.tiling_param "Rm_org_size" : index
    %7 = afir.symbolic_eval %6, %4 {expr = #afir.expr<"(__arg0 / (__arg1))">} : (index, index) -> index
    %8 = afir.symbolic_eval %3, %5, %2 {expr = #afir.expr<"(__arg1 / (__arg0 * __arg2))">} : (index, index, index) -> index
    %9 = afir.symbolic_eval %3, %5, %2 {expr = #afir.expr<"(__arg1 / (__arg0 * __arg2))">} : (index, index, index) -> index
    %alloc = memref.alloc(%6, %5) {afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>} : memref<?x?xf32, #afir.space<gm>>
    %alloc_0 = memref.alloc(%7, %4, %8, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x?x?xf32, #afir.space<ub>>
    %10 = afir.load %alloc outs(%alloc_0) {afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(Rm_org_size / (Rmt_size))">, <"Rmt_size">, <"(A_org_size / (ATb_size * At_size))">, <"ATb_size">, <"At_size">],strides = [<"(A_org_size * Rmt_size)">, <"A_org_size">, <"(ATb_size * At_size)">, <"At_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"(8 * Ceiling((At_size * Rational(1 , 8))))">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"Rm_org_size">, <"A_org_size">],strides = [<"A_org_size">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(Rm_org_size / (Rmt_size))">, <"Rmt_size">, <"(A_org_size / (ATb_size * At_size))">, <"ATb_size">, <"At_size">],strides = [<"(A_org_size * Rmt_size)">, <"A_org_size">, <"(ATb_size * At_size)">, <"At_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"(8 * Ceiling((At_size * Rational(1 , 8))))">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(Rm_org_size / (Rmt_size))">, #afir.expr<"Rmt_size">, #afir.expr<"(A_org_size / (ATb_size * At_size))">, #afir.expr<"ATb_size">, #afir.expr<"At_size">], indexing_maps = [#map, #map1]} : memref<?x?xf32, #afir.space<gm>>, memref<?x?x?x?x?xf32, #afir.space<ub>> -> memref<?x?x?x?x?xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%9, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<1x1x?x?x?xf32, #afir.space<ub>>
    %alloc_2 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_3 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %11 = afir.sum %10 outs(%alloc_1) scratch(%alloc_2, %alloc_3) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(A_org_size / (ATb_size * At_size))">, <"ATb_size">, <"At_size">],strides = [<"0">, <"0">, <"(ATb_size * At_size)">, <"At_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"0">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(Rm_org_size / (Rmt_size))">, <"Rmt_size">, <"(A_org_size / (ATb_size * At_size))">, <"ATb_size">, <"At_size">],strides = [<"(A_org_size * Rmt_size)">, <"A_org_size">, <"(ATb_size * At_size)">, <"At_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"(8 * Ceiling((At_size * Rational(1 , 8))))">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(A_org_size / (ATb_size * At_size))">, <"ATb_size">, <"At_size">],strides = [<"0">, <"0">, <"(ATb_size * At_size)">, <"At_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"0">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"(A_org_size / (ATb_size * At_size))">, #afir.expr<"ATb_size">, #afir.expr<"At_size">]} : memref<?x?x?x?x?xf32, #afir.space<ub>>, memref<1x1x?x?x?xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<1x1x?x?x?xf32, #afir.space<ub>>
    afir.store %11 outs(%arg0) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"A_org_size">],strides = [<"0">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(A_org_size / (ATb_size * At_size))">, <"ATb_size">, <"At_size">],strides = [<"0">, <"0">, <"(ATb_size * At_size)">, <"At_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"0">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"(A_org_size / (ATb_size * At_size))">, #afir.expr<"ATb_size">, #afir.expr<"At_size">], indexing_maps = [#map3, #map4]} : memref<1x1x?x?x?xf32, #afir.space<ub>>, memref<1x?xf32, #afir.space<gm>>
    return
  }
}


// 模块级：两个 stage、共用的 workspace 张量、跨 stage 的变量绑定。
// 顺序按属性字典序：`afir.var_relations` -> `afir.workspace_tensor_ids` -> `afplan.stages`。
// CHECK: afir.var_relations
// CHECK-SAME: dst_var = "Rm_org_size"
// CHECK-SAME: afir.workspace_tensor_ids = array<i64: 2>
// CHECK-SAME: afplan.stages = #afplan.stages<template_counts = [1, 1]>

// phase 1：AR。被归约维在物理内侧，对齐补出来的那几列要填单位元，所以必须生成 `ReduceInit`。
// CHECK-LABEL: func.func private @stage0_template0
// CHECK: ascendc.data_copy_pad_extend
// CHECK: ascendc.duplicate_l1
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK-SAME: layout = 0 : i32, srcInnerPad = true
// R 被切成多个 tile，所以有跨轮累加：首轮拷贝、其余轮次相加。
// CHECK: ascendc.data_copy_l2
// CHECK: ascendc.add_l2
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: ascendc.pipe_barrier pipe_v
// 落点是 GM workspace，不是 kernel 输出。
// CHECK: ascendc.data_copy_pad_extend

// phase 2：RA，且不生成 `ReduceInit`。
// CHECK-LABEL: func.func private @stage1_template0
// CHECK-NOT: ascendc.duplicate_l1
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK-SAME: layout = 1 : i32, srcInnerPad = true
// CHECK: ascendc.add_l2
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: ascendc.pipe_barrier pipe_v

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-scope-reduce-scratch %s | FileCheck %s --check-prefix=SCOPE
// SCOPE-LABEL: func.func private @stage0_template0
// SCOPE: scf.for
// SCOPE: memref.alloc{{.*}}life_time_axis_id = -1{{.*}}afir.scratch_scope = 7 : i64
// SCOPE: memref.alloc{{.*}}life_time_axis_id = -1{{.*}}afir.scratch_scope = 7 : i64
// SCOPE: afir.sum
// SCOPE-LABEL: func.func private @stage1_template0
// SCOPE: scf.for
// SCOPE: scf.for
// SCOPE: memref.alloc{{.*}}life_time_axis_id = -1{{.*}}afir.scratch_scope = 4 : i64
// SCOPE: memref.alloc{{.*}}life_time_axis_id = -1{{.*}}afir.scratch_scope = 4 : i64
// SCOPE: afir.sum

// Reduce completion must not depend on pipeline-sync/free_tensor barriers or on
// scope attributes surviving view canonicalization. Check lowering in isolation.
// RUN: af-opt --afir-kernel-body-signature --afir-materialize-tiling-params --afir-loop-generate --afir-scope-reduce-scratch --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc --target=Ascend910B1 %s | FileCheck %s --check-prefix=COMPLETE
// COMPLETE: ascendc.add_l2
// COMPLETE-NEXT: {{^ *}}}
// COMPLETE-NEXT: ascendc.pipe_barrier pipe_v
// COMPLETE: ascendc.add_l2
// COMPLETE-NEXT: {{^ *}}}
// COMPLETE-NEXT: ascendc.pipe_barrier pipe_v
