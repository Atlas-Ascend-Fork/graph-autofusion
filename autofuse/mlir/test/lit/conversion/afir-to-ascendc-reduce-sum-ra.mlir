// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.sum` 的 RA 形态（物理布局 `[R, A]`，首维被归约）。
//
// 输入是真实调度产物，不是手改 stride 改出来的：`load_sum_store` 的 `[16, 32] -> [1, 32]`、
// `reduce_axes = {0}`，取 codegen 入口处 dump 的那份模块。这一点很要紧——此前那条 RA 负例
// 只翻转了 `vectorized_strides` 没翻 `strides`，同一根轴在两处一个说被归约、一个说保留，
// 拿它当正例会钉住一个永远不会出现的形状。（`afir.original_graph_name` 里的 `ra_probe`
// 是当时那条临时注册的名字，模块本身是调度器原样产出的。）
//
// 本用例钉的是「语义 A/R 到物理外/内的翻译」，这是 AR 下看不出来的那一半：
//   - 轴 5（z0t，从原轴 0 切出）是被归约维 R，物理外维；
//   - 轴 3（z1t，从原轴 1 切出）是保留维 A，物理内维。
// 两根轴的 actual 都带 `#afir.axis_binding`，所以下面可以按轴 id 而不是 SSA 编号来断言
// 「哪一个量进了哪一个参数」。
//
// `--implicit-check-not` 守的是 `ReduceInit`：AR 要它把对齐补出来的 R 位置填成单位元，
// RA 的被归约维在物理外侧、给的已经是 actual，天然没有那一段；而 `reduce_init.h` 的填充
// 按行主序写死，RA 下生成就是往错地址写。legacy 同样只在 AR 下生成
// （`codegen/api_call/reduce/reduce_api_call_base.cpp:547` 的 `if (x.isAr)`）。

// 两个目标各生成一遍（950 无 E2E，只能比对文本）：reduce 的下降没有任何目标相关分支，
// 两份输出**逐字节相同**，所以共用同一组 CHECK。这是
// `2026-09-20-two-dim-transfer-addressing-design.md` 8.5 六组合验收项的 lit 一侧。
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --implicit-check-not=ascendc.duplicate_l1
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s \
// RUN:   | FileCheck %s --implicit-check-not=ascendc.duplicate_l1

// CHECK-LABEL: func.func private @reduce_sum_ra_body

// 轴 3 = A、轴 5 = R 的本轮实际长度。尾块下二者都会缩水，所以是 `arith.select`。
// CHECK: %[[AACT:[a-zA-Z0-9_]+]] = arith.select %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}} {afir.axis_binding = #afir.axis_binding<axis = 3, role = actual_size>}
// CHECK: %[[RACT:[a-zA-Z0-9_]+]] = arith.select %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}} {afir.axis_binding = #afir.axis_binding<axis = 5, role = actual_size>}

// 32B 对齐落在物理内维上，RA 下那是 A；而且取的是 A 的标称长度——它是行距，
// 行距按标称分配，不随尾块变化。AR 下这次对齐落在 R 上，是本用例与 AR 用例的分水岭。
// CHECK: %[[ADIV:[a-zA-Z0-9_]+]] = arith.ceildivsi %{{[a-zA-Z0-9_]+}}, %[[C8:[a-zA-Z0-9_]+]]
// CHECK: %[[AALIGN:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = arith.mulsi_extended %[[ADIV]], %[[C8]]
// CHECK: cf.assert %{{[a-zA-Z0-9_]+}}, "reduce aligned extent must not overflow"

// `srcShape = {物理外维行数, 物理内维行距}`：外维取 R 的 **actual**（尾块下必须少走几行，
// 没有 ReduceInit 兜底），内维取 align 后的 A nominal。
// CHECK: %[[SEG:[a-zA-Z0-9_]+]] = ascendc.reinterpret_cast
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]] = arith.index_cast %[[RACT]] : index to i32
// CHECK: %[[INNER:[a-zA-Z0-9_]+]] = arith.index_cast %[[AALIGN]] : index to i32
// CHECK: ascendc.reduce_sum_2d_l2 %[[SEG]], %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[OUTER]], %[[INNER]]
// CHECK-SAME: layout = 1
// CHECK-SAME: srcInnerPad = true

// 跨 tile 累加按语义 A 的 actual 对齐后取长度，不是物理外维 R——按 R 合并就是该合
// A 个结果却合了 R 个。这一条 AR / RA 共用同一个来源。
// CHECK: %[[CDIV:[a-zA-Z0-9_]+]] = arith.ceildivsi %[[AACT]], %{{[a-zA-Z0-9_]+}}
// CHECK: %[[CALIGN:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = arith.mulsi_extended %[[CDIV]], %{{[a-zA-Z0-9_]+}}
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = arith.index_cast %[[CALIGN]] : index to i32
// CHECK: scf.if
// CHECK:   ascendc.data_copy_l2 %{{[a-zA-Z0-9_]+}}, %[[SEG]], %[[COUNT]]
// CHECK: } else {
// CHECK:   ascendc.add_l2 %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[SEG]], %[[COUNT]]
// CHECK: }

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
module @reduce_sum_ra attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>, afir.workspace_tensor_ids = array<i64: 2>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.stages = #afplan.stages<template_counts = [1]>} {
  func.func private @reduce_sum_ra_body(%arg0: memref<16x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"2048">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"16">, #afir.expr<"32">]}, %arg1: memref<1x32xf32, #afir.space<gm>> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"128">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"32">]}) attributes {afir.arg_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"32">],strides = [<"0">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"16">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"32">, align = <"-1">>, <id = 2, name = "z1T", axis_type = TileOuter, size = <"Ceiling((32 / (z1t_size)))">, align = <"1">, from = [1], split_pair_other_id = 3>, <id = 3, name = "z1t", axis_type = TileInner, size = <"z1t_size">, align = <"1">, from = [1], split_pair_other_id = 2>, <id = 4, name = "z0T", axis_type = TileOuter, size = <"Ceiling((16 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 5>, <id = 5, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 4>, <id = 6, name = "z1TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((32 / (z1t_size))) / (z1Tb_size)))">, align = <"1">, from = [2], split_pair_other_id = 7>, <id = 7, name = "z1Tb", axis_type = BlockInner, size = <"z1Tb_size">, align = <"1">, from = [2], split_pair_other_id = 6>], type = Compute, size_var = [<"z1t_size">, <"z0t_size">, <"z1Tb_size">]>, afir.buffer_ids = array<i64: 0, 1>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_sum_store_ra_probe_0_B0Y1R0_S0G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_plan = [#afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(4 * A_org_size * Rm_org_size)">, life_time_axis_id = -1>], afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>} {
    %0 = afir.tiling_param "b0_size" : index
    %1 = afir.tiling_param "b1_size" : index
    %2 = afir.tiling_param "z1t_size" : index
    %3 = afir.tiling_param "z1Tb_size" : index
    %4 = afir.tiling_param "z0t_size" : index
    %5 = afir.symbolic_eval %4 {expr = #afir.expr<"(16 / (__arg0))">} : (index) -> index
    %6 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(32 / (__arg0 * __arg1))">} : (index, index) -> index
    %7 = afir.symbolic_eval %3, %2 {expr = #afir.expr<"(32 / (__arg0 * __arg1))">} : (index, index) -> index
    %alloc = memref.alloc(%5, %4, %6, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x?x?xf32, #afir.space<ub>>
    %8 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(16 / (z0t_size))">, <"z0t_size">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"(32 * z0t_size)">, <"32">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"16">, <"32">],strides = [<"32">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(16 / (z0t_size))">, <"z0t_size">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"(32 * z0t_size)">, <"32">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"(16 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"(32 / (z1Tb_size * z1t_size))">, #afir.expr<"z1Tb_size">, #afir.expr<"z1t_size">], indexing_maps = [#map, #map1]} : memref<16x32xf32, #afir.space<gm>>, memref<?x?x?x?x?xf32, #afir.space<ub>> -> memref<?x?x?x?x?xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%7, %3, %2) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<1x1x?x?x?xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%1) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %alloc_2 = memref.alloc(%0) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %9 = afir.sum %8 outs(%alloc_0) scratch(%alloc_1, %alloc_2) {afir.scratch_roles = ["api_scratch", "round_result"], afir.destination_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"0">, <"0">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"0">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"(16 / (z0t_size))">, <"z0t_size">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"(32 * z0t_size)">, <"32">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"0">, <"0">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"0">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 4>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"(32 / (z1Tb_size * z1t_size))">, #afir.expr<"z1Tb_size">, #afir.expr<"z1t_size">]} : memref<?x?x?x?x?xf32, #afir.space<ub>>, memref<1x1x?x?x?xf32, #afir.space<ub>>, scratch(memref<?xui8, #afir.space<ub>>, memref<?xui8, #afir.space<ub>>) -> memref<1x1x?x?x?xf32, #afir.space<ub>>
    afir.store %9 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 1],repeats = [<"1">, <"32">],strides = [<"0">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [4, 5, 6, 7, 3],repeats = [<"1">, <"1">, <"(32 / (z1Tb_size * z1t_size))">, <"z1Tb_size">, <"z1t_size">],strides = [<"0">, <"0">, <"(z1Tb_size * z1t_size)">, <"z1t_size">, <"1">],vectorized_axis = [5, 3],vectorized_strides = [<"0">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [6, 7, 4, 5, 3], loop_axis = 7>, afir.symbolic_shape = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"(32 / (z1Tb_size * z1t_size))">, #afir.expr<"z1Tb_size">, #afir.expr<"z1t_size">], indexing_maps = [#map2, #map3]} : memref<1x1x?x?x?xf32, #afir.space<ub>>, memref<1x32xf32, #afir.space<gm>>
    return
  }
}
