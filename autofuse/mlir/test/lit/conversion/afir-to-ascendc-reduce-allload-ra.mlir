// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// AllLoad（全载）模板的 RA 下降。输入是 `load_sum_store_allload_ara_4x32x8` 的真实候选
// （`f32[4, 32, 8]` 归约中间维，`schedule plan 3` 的 stage 0 template 0）。
//
// 这条用例的存在有两个不可替代的理由：
//
// 1. AllLoad 下只有 ARA 能产出 RA 形态。 二维 RA（输出 `[1, N]`）根本不会生成 AllLoad
//    候选——`IsLegacyFullLoadReduce` 要求 `in.repeats[0] == out.repeats[0]`，
//    `IsTailReduceFullLoad` 要求轴数 ≥ 3 且归约末轴，二维 RA 两条都不满足。而 ARA 的
//    `vectorized_axis = [1, 4, 2]`（`[R, A_inner, A_outer]`）首项被归约 -> 判成 RA。
//
// 2. 它是 RA `srcShape` 公式里「A 组连乘」的第一个真实语料。
//    `2026-09-21-afir-reduce-sum-ra-lowering-design.md` 的验收表里注明过「连乘部分尚无用例
//    覆盖」——既有 RA 语料的 A 组都只有一根轴，乘与不乘看不出差别。这里 A 组有两根
//    （`z2` 内 + `z0t` 外），内维 = `align(A_inner) × Π A_outer`，那条
//    `reduce merged extent must not overflow` 守的就是这个连乘。
//
// 另外顺带钉住 AllLoad 的两条形态判据，判据都写成反向断言（在整份输出上生效）：
//   - RA 不生成 `ReduceInit`：被归约维在物理外侧，没有需要填单位元的那一段，而
//     `reduce_init.h` 的填充按行主序写死，RA 下生成是往错地址写。
//   - AllLoad 不生成 `add_l2`：`loop_axis` 之内已无归约维，一次算完，没有跨轮累加。

// 两个目标各生成一遍（950 无 E2E，只能比对文本）：reduce 的下降没有任何目标相关分支，
// 两份输出**逐字节相同**，所以共用同一组 CHECK。这是
// `2026-09-20-two-dim-transfer-addressing-design.md` 8.5 六组合验收项的 lit 一侧。
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --implicit-check-not=ascendc.duplicate_l1 --implicit-check-not=ascendc.add_l2
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s \
// RUN:   | FileCheck %s --implicit-check-not=ascendc.duplicate_l1 --implicit-check-not=ascendc.add_l2

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module @reduce_allload_ra attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = true, afir.module_kind = "plan", afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>, afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>} {
  func.func private @reduce_allload_ra_body(%arg0: memref<4x32x8xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"4">, #afir.expr<"32">, #afir.expr<"8">]}, %arg1: memref<4x1x8xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"128">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"4">, #afir.expr<"1">, #afir.expr<"8">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1, 2],repeats = [<"4">, <"32">, <"8">],strides = [<"256">, <"8">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1, 2],repeats = [<"4">, <"1">, <"8">],strides = [<"8">, <"0">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"4">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z2", axis_type = Original, size = <"8">, align = <"-1">>, <id = 3, name = "z0T", axis_type = TileOuter, size = <"Ceiling((4 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 4>, <id = 4, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 5, name = "z0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((4 / (z0t_size))) / (z0Tb_size)))">, align = <"1">, from = [3], split_pair_other_id = 6>, <id = 6, name = "z0Tb", axis_type = BlockInner, size = <"z0Tb_size">, align = <"1">, from = [3], split_pair_other_id = 5>], type = Compute, size_var = [<"z0t_size">, <"z0Tb_size">]>, afir.buffer_ids = array<i64: 0, 1>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_allload_ara_4x32x8_0_B0Y0_S3G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "z0t_size" : index
    %3 = afir.tiling_param "z0Tb_size" : index
    %4 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(4 / (__arg0 * __arg1))">} : (index, index) -> index
    %5 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(4 / (__arg0 * __arg1))">} : (index, index) -> index
    %alloc = memref.alloc(%4, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x32x8xf32, #afir.space<ub>>
    %6 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [5, 6, 4, 1, 2],repeats = [<"(4 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"32">, <"8">],strides = [<"(256 * z0Tb_size * z0t_size)">, <"(256 * z0t_size)">, <"256">, <"8">, <"1">],vectorized_axis = [1, 4, 2],vectorized_strides = [<"(8 * z0t_size)">, <"8">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2],repeats = [<"4">, <"32">, <"8">],strides = [<"256">, <"8">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [5, 6, 4, 1, 2],repeats = [<"(4 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"32">, <"8">],strides = [<"(256 * z0Tb_size * z0t_size)">, <"(256 * z0t_size)">, <"256">, <"8">, <"1">],vectorized_axis = [1, 4, 2],vectorized_strides = [<"(8 * z0t_size)">, <"8">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 1, 4, 2], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"(4 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"32">, #afir.expr<"8">], indexing_maps = [#map, #map1]} : memref<4x32x8xf32, #afir.space<gm>>, memref<?x?x?x32x8xf32, #afir.space<ub>> -> memref<?x?x?x32x8xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%5, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x1x8xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_2 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %7 = afir.sum %6 outs(%alloc_0) scratch(%alloc_1, %alloc_2) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [5, 6, 4, 1, 2],repeats = [<"(4 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"8">],strides = [<"(8 * z0Tb_size * z0t_size)">, <"(8 * z0t_size)">, <"8">, <"0">, <"1">],vectorized_axis = [1, 4, 2],vectorized_strides = [<"0">, <"8">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [5, 6, 4, 1, 2],repeats = [<"(4 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"32">, <"8">],strides = [<"(256 * z0Tb_size * z0t_size)">, <"(256 * z0t_size)">, <"256">, <"8">, <"1">],vectorized_axis = [1, 4, 2],vectorized_strides = [<"(8 * z0t_size)">, <"8">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [5, 6, 4, 1, 2],repeats = [<"(4 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"8">],strides = [<"(8 * z0Tb_size * z0t_size)">, <"(8 * z0t_size)">, <"8">, <"0">, <"1">],vectorized_axis = [1, 4, 2],vectorized_strides = [<"0">, <"8">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 1, 4, 2], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"(4 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"8">]} : memref<?x?x?x32x8xf32, #afir.space<ub>>, memref<?x?x?x1x8xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<?x?x?x1x8xf32, #afir.space<ub>>
    afir.store %7 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1, 2],repeats = [<"4">, <"1">, <"8">],strides = [<"8">, <"0">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [5, 6, 4, 1, 2],repeats = [<"(4 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">, <"8">],strides = [<"(8 * z0Tb_size * z0t_size)">, <"(8 * z0t_size)">, <"8">, <"0">, <"1">],vectorized_axis = [1, 4, 2],vectorized_strides = [<"0">, <"8">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [5, 6, 1, 4, 2], loop_axis = 6>, afir.symbolic_shape = [#afir.expr<"(4 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">, #afir.expr<"8">], indexing_maps = [#map2, #map3]} : memref<?x?x?x1x8xf32, #afir.space<ub>>, memref<4x1x8xf32, #afir.space<gm>>
    return
  }
}

// CHECK-LABEL: func.func private @reduce_allload_ra_body

// RA 的物理内维是保留维 A，所以 32B 对齐落在 A 的内层轴上，不落在 R 上。整份输出里
// 「reduce aligned extent」只有这一条，拿它当锚点。
// CHECK: cf.assert %{{[a-zA-Z0-9_]+}}, "reduce aligned extent must not overflow"

// 紧接着就是连乘：A 组有两根轴，内维 = 对齐后的内层轴 × 外层轴。此前的 RA 语料 A 组都只有
// 一根轴，乘与不乘看不出差别，这一项一直没有用例覆盖。
// CHECK-NEXT: %[[AMERGED:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = arith.mulsi_extended %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}} : index
// CHECK: cf.assert %{{[a-zA-Z0-9_]+}}, "reduce merged extent must not overflow"

// srcShape 按物理次序给：RA 的外维是 R（整根 32，常量），内维是上面那个 A 连乘。
// 盯住内维来自 `AMERGED` 是本用例的要点——公式写错时 AR 下看不出差别，RA 下就是静默错值。
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]] = arith.index_cast %{{[a-zA-Z0-9_]+}} : index to i32
// CHECK-NEXT: %[[INNER:[a-zA-Z0-9_]+]] = arith.index_cast %[[AMERGED]] : index to i32
// CHECK-NEXT: ascendc.reduce_sum_2d_l2 %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[OUTER]], %[[INNER]]
// CHECK-SAME: layout = 1
// CHECK-SAME: srcInnerPad = true

// 算完直接搬出去，没有 workspace 往返。
// CHECK: ascendc.data_copy_pad_extend
