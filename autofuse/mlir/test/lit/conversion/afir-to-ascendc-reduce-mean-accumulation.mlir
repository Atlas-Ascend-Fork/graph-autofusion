// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 跨轮累加的 `afir.mean`：求和照 Sum 的样子逐轮累加，末轮再整体乘一次 `1 / N`。
//
// 输入与 `afir-to-ascendc-reduce-sum.mlir` 是同一份真实调度，唯一的差别是算子换成
// `afir.mean`。这份调度把归约轴切成了 tile（`afir.sched` 的 `loop_axis = 4`，z1T 是累加
// 轴），于是 `ReducePlan::isAccumulating()` 成立；本用例钉的就是这个形态下「除几次、除
// 多少」两件事。
//
// 本文件此前钉的是拒绝。拒绝的两条理由都已经不成立：
//   - 「legacy 多轮分支不生成均值缩放」是错的——legacy 在循环外生成
//     （`codegen/codegen_kernel_loop.cpp:891`），分母同样是原始轴尺寸之积；
//   - 「总 R 只能从调度前的原始轴尺寸算」是对的，而那正是现在的做法
//     （`afir::MeanReductionDomain`），不是障碍。
// 第二阶段会不会重复除一次也不是问题：RCore 的第二阶段由调度侧改写成 `Sum`
// （`optimize/task_generator/reduce_schedule_case_generator.cpp:165`）。

// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s

// 求和那一半与 Sum 逐字相同：首轮拷贝、其余轮次 `Add`。
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK: %[[FIRST:.*]] = arith.cmpi eq, %[[IV:.*]], %{{.*}} : index
// CHECK: scf.if %[[FIRST]] {
// CHECK:   ascendc.data_copy_l2
// CHECK: } else {
// CHECK:   ascendc.add_l2
// CHECK: }

// 只在末轮除一次。判据是剩余距离 `upperBound - iv <= step`，而不是 `iv + step >= upperBound`
// ——后者的加法会回绕，回绕之后末轮判成非末轮，整个 kernel 少除一次。
// CHECK: %[[REMAIN:.*]] = arith.subi %{{.*}}, %[[IV]] : index
// CHECK: %[[LAST:.*]] = arith.cmpi sle, %[[REMAIN]], %{{.*}} : index

// 分母是完整逻辑归约域：z1 这根 Original 轴的 size 是 32，与 z1t_size 切成几轮无关。
// 取某一轮的 actual 会得到「均值的均值」，取 nominal 会把 32B 对齐的填充算进元素数。
// `index -> i64 -> f32`，中间不经 i32。
// CHECK: %[[N:.*]] = arith.constant 32 : index
// CHECK: %[[N64:.*]] = arith.index_cast %[[N]] : index to i64
// CHECK: %[[DEN:.*]] = arith.sitofp %[[N64]] : i64 to f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[SCALE:.*]] = arith.divf %[[ONE]], %[[DEN]] : f32

// CHECK: scf.if %[[LAST]] {
// 屏障跟着 `Muls` 一起进 `scf.if`：它防的是本轮 `Add` 与这条 `Muls` 的依赖，非末轮没有
// `Muls` 也就不必付这次屏障。
// CHECK:   ascendc.pipe_barrier pipe_v
// CHECK:   ascendc.muls_l2 %[[DST:.*]], %[[DST]], %[[SCALE]], %{{.*}}
// CHECK: }

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
module @reduce_mean_accumulation attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>, ascendc.target_aiv_num = 48 : i64, ascendc.target_ub_size = 196608 : i64} {
  func.func private @reduce_mean_accumulation_body(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 3, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 2>, <id = 4, name = "z1T", axis_type = TileOuter, size = <"Ceiling((32 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 5>, <id = 5, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 4>, <id = 6, name = "z0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((16 / (z0t_size))) / (z0Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 7>, <id = 7, name = "z0Tb", axis_type = BlockInner, size = <"z0Tb_size">, align = <"1">, from = [2], split_pair_other_id = 6>], type = Compute, size_var = [<"z0t_size">, <"z1t_size">, <"z0Tb_size">]>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_16x32_0_B0Y0R1_S0G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
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
    %9 = afir.mean %8 outs(%alloc_2) scratch(%alloc_0, %alloc_1) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"(32 / (z1t_size))">, <"z1t_size">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"z1t_size">, <"1">],vectorized_axis = [3, 5],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.reduction_domain = #afir.reduction_domain<axes = [1], sizes = [<"32">]>, afir.result_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">]} : memref<?x?x?x?x?xf32, #afir.space<ub>>, memref<?x?x?x1x1xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<?x?x?x1x1xf32, #afir.space<ub>>
    afir.store %9 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [6, 7, 3, 4, 5],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">, <"0">],vectorized_axis = [3, 5],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 3, 5], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"1">], indexing_maps = [#map2, #map3]} : memref<?x?x?x1x1xf32, #afir.space<ub>>, memref<16x32xf32, #afir.space<gm>>
    return
  }
}
