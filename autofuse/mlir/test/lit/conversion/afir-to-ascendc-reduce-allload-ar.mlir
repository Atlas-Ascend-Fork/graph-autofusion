// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// AllLoad（全载）模板的 AR 下降。输入是 `load_sum_store_allload_16x32` 的真实候选
// （`f32[16, 32]` 归约末维，`schedule plan 2`），与 `afir-to-ascendc-reduce-sum.mlir`
// 的 RCommon 同一张图、不同模板。
//
// AllLoad 的签名是三件事同时成立，缺一不可：
//   1. 归约轴 `z1` 整根不切 —— `afir.asc_graph` 的 axes 里没有 `z1T`/`z1t`，只有 A 轴的
//      Tile/Block 四根。`vectorized_axis = [3, 1]` 里的 1 就是未切的原始 R 轴。
//   2. `is_reduce_mem_reuse = true` —— 调度器只在 AllLoad 置真（`optimize/optimize.cpp`）。
//      MLIR 侧不读它，复用决策在 `buf_que_allocator` 已经做完，这里只是模板的身份标记。
//   3. 不跨轮累加 —— `loop_axis = 5`（A 的 BlockInner）之内已经没有归约维，`ReducePlan`
//      的 `accumulationAxes` 为空，`isAccumulating()` 假。
//
// 第 3 条是本用例真正要钉的东西，判据写成 `--implicit-check-not=ascendc.add_l2`：RCommon 要
// 在「第一轮拷贝 / 后续轮累加」的 `scf.if` 里生成 `add_l2`，AllLoad 一次算完，一条都不该有。
// 这条反向断言比正向检查更结实——它在整份输出上生效，新插进来的累加也躲不过。
//
// 调度器仍然给了两块 scratch（AllLoad 用不到第二块），`ReducePlan` 只按 `apiScratchIndex`
// 取第一块给高阶 API，不因为「有两块」就去动第二块。

// 两个目标各生成一遍（950 无 E2E，只能比对文本）：reduce 的下降没有任何目标相关分支，
// 两份输出**逐字节相同**，所以共用同一组 CHECK。这是
// `2026-09-20-two-dim-transfer-addressing-design.md` 8.5 六组合验收项的 lit 一侧。
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --implicit-check-not=ascendc.add_l2
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s \
// RUN:   | FileCheck %s --implicit-check-not=ascendc.add_l2

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d3)>
module @reduce_allload_ar attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = true, afir.module_kind = "plan", afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>, afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>} {
  func.func private @reduce_allload_ar_body(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<16x1xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"64">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"1">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"1">],strides = [<"1">, <"0">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 3>, <id = 3, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 2>, <id = 4, name = "z0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((16 / (z0t_size))) / (z0Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 5>, <id = 5, name = "z0Tb", axis_type = BlockInner, size = <"z0Tb_size">, align = <"1">, from = [2], split_pair_other_id = 4>], type = Compute, size_var = [<"z0t_size">, <"z0Tb_size">]>, afir.buffer_ids = array<i64: 0, 1>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_allload_16x32_0_B0Y0_S2G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
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
    %7 = afir.sum %6 outs(%alloc_0) scratch(%alloc_1, %alloc_2) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">],vectorized_axis = [3, 1],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"32">],strides = [<"(32 * z0Tb_size * z0t_size)">, <"(32 * z0t_size)">, <"32">, <"1">],vectorized_axis = [3, 1],vectorized_strides = [<"32">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">],vectorized_axis = [3, 1],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [4, 5, 3, 1], loop_axis = 5>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">]} : memref<?x?x?x32xf32, #afir.space<ub>>, memref<?x?x?x1xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<?x?x?x1xf32, #afir.space<ub>>
    afir.store %7 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"1">],strides = [<"1">, <"0">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 3, 1],repeats = [<"(16 / (z0Tb_size * z0t_size))">, <"z0Tb_size">, <"z0t_size">, <"1">],strides = [<"(z0Tb_size * z0t_size)">, <"z0t_size">, <"1">, <"0">],vectorized_axis = [3, 1],vectorized_strides = [<"1">, <"0">],role = <compute>>], afir.sched = #afir.sched<axis = [4, 5, 3, 1], loop_axis = 5>, afir.symbolic_shape = [#afir.expr<"(16 / (z0Tb_size * z0t_size))">, #afir.expr<"z0Tb_size">, #afir.expr<"z0t_size">, #afir.expr<"1">], indexing_maps = [#map2, #map3]} : memref<?x?x?x1xf32, #afir.space<ub>>, memref<16x1xf32, #afir.space<gm>>
    return
  }
}


// CHECK-LABEL: func.func private @reduce_allload_ar_body

// 一次搬入：整块 `A_tile × 32`。R 整根在 tile 里，二维 DMA 折成一维是对的。
// CHECK: ascendc.data_copy_pad_extend

// AR 的物理内维是被归约的 R，32B 对齐落在它身上（f32 下 8 个元素一组）。
// CHECK: %[[RDIV:[a-zA-Z0-9_]+]] = arith.ceildivsi %{{[a-zA-Z0-9_]+}}, %[[C8:[a-zA-Z0-9_]+]] : index
// CHECK: %[[RALIGN:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = arith.mulsi_extended %[[RDIV]], %[[C8]] : index
// CHECK: cf.assert %{{[a-zA-Z0-9_]+}}, "reduce aligned extent must not overflow"

// AR 先把对齐引入的那一段填成单位元，再归约；`layout = 0` 是 AR，`srcInnerPad` 说明内维带填充。
// CHECK: ascendc.duplicate_l1
// CHECK: ascendc.reduce_sum_2d_l2
// CHECK-SAME: layout = 0
// CHECK-SAME: srcInnerPad = true

// 算完直接搬出去。没有段结果缓冲的二次读写，也没有 workspace 往返。
// CHECK: ascendc.data_copy_pad_extend
