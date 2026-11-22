// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// **跨段 load/store 的 block 参数断言**（设计文档功能需求 1 的合成 lit）。
//
// 来源是真实 dump：`load_transpose_store_2x37x23` 的保留转置模板 S0G0C0（mem 档），只删了
// 同 plan 的 template1、`template_counts` 2→1，并修正旧 dump 的 GM 输出视图。图的语义是 `(2,37,23)` 按
// `perm(0,2,1)` 转置后写回，GM 输出行主序 `(2,23,37)@{851,37,1}`——**z1（37 个）是输出侧
// 的最内轴**，这一点决定了 store 的两个段几何。
//
// 两端的段几何（合轴：GM/UB 两侧都无洞才能并）：
//
//   load：`[(z0t×37 行, GM 23, UB 16·Ceil(z2t/16)), (z2t, 1, 1)]`——z0t 与 z1 之间
//         GM `851==37×23`、UB `592C==37×16C` 双侧无洞，**并进同一个段**，所以
//         blockCount 是 `37 × ActualSize(z0t)`，且 load 不需要外层循环。
//   store：`[(z2t 列, GM 37, UB 48), (37, 1, 1)]` + z0t 外层循环——GM 侧 `37==37`
//         无洞可并，但 UB 侧 `48 != 37` 有补位缝，并不了；z0t 在 GM 侧 `851 != 37×z2t`
//         也有洞，于是第三根向量轴退化成 `scf.for`（界 = ActualSize(z0t)）。
//
// 参数算式（与 legacy `SetDmaParams` 同式，`Size`=对齐后、`ActualSize`=尾块修正后）：
//
//   load  = (blockCount=37·ActualSize(z0t), blockLen=ActualSize(z2t),
//            srcStride=23−len, dstStride=16·Ceil(z2t/16)−len)
//   store = (blockCount=ActualSize(z2t), blockLen=37,
//            srcStride=48−37, dstStride=37−37)
//   store 的 dstStride 是 0 不是错：GM 侧行距恰好等于行长（37 连续），`DataCopyPad`
//   的 Normal 模式本来就把补到 32B 边界的空隙一并覆盖。
//
// 方向映射只钉一处的事实：load 第一操作数是 local_tensor、store 是 global_tensor
// （`DataCopyPadExtend` 两个重载的单位换算相反，GM 乘 sizeof(T)、UB 除 align_num）。
// 数值正确性由 e2e `load_transpose_store_2x37x23` 钉（NPU bit 级）；本 lit 钉参数代数。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s -o %t.mlir
// RUN: af-opt --afir-aggregate-launch --afir-aggregate-launch-graph-name=load_store_strided_block %t.mlir \
// RUN:   | FileCheck %s --check-prefix=CPP

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d1, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d4, d6)>
module @mlir_load_transpose_store_2x37x23_subgraph0_plan0 attributes {afir.cube_type = -1 : i64, afir.enable_group_parallel = false, afir.is_reduce_mem_reuse = false, afir.module_kind = "plan", afir.target = #afir.target<soc = "Ascend910B1", aiv_num = 48, ub_size = 196608>, afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>, afplan.score_func = "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\0A  return 1;\0A}\0A", afplan.stages = #afplan.stages<template_counts = [1]>} {
  func.func private @stage0_template0(%arg0: memref<2x37x23xf32, 22 : i32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"6808">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>, afir.symbolic_shape = [#afir.expr<"2">, #afir.expr<"37">, #afir.expr<"23">]}, %arg1: memref<2x23x37xf32, 22 : i32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"6808">, life_time_axis_id = -1>, afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>, afir.symbolic_shape = [#afir.expr<"2">, #afir.expr<"23">, #afir.expr<"37">]}, %arg2: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}, %arg3: !emitasc.py_struct<"TilingData", [i64, i64, i64, i64, !emitasc.py_struct<"ConfusionTransposeTiling", [ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32, ui32], ["param0", "param1", "param2", "param3", "param4", "param5", "param6", "param7", "param8", "param9", "param10", "param11", "param12", "param13", "param14", "param15", "param16", "param17"]>, i64, i64, i64], ["block_dim", "z2t_size", "z0t_size", "z2Tz0Tb_size", "transpose_tilingData_0", "q0_size", "q1_size", "b0_size"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {afir.api_tiling_fields = [{field = "transpose_tilingData_0", fields = ["param0", "param1", "param2", "param3", "param4", "param5", "param6", "param7", "param8", "param9", "param10", "param11", "param12", "param13", "param14", "param15", "param16", "param17"], type = "ConfusionTransposeTiling"}], afir.arg_descs = [#afir.tensor_desc<axis = [0, 1, 2],repeats = [<"2">, <"37">, <"23">],strides = [<"851">, <"23">, <"1">],role = <input>>, #afir.tensor_desc<axis = [0, 1, 2],repeats = [<"2">, <"37">, <"23">],strides = [<"851">, <"23">, <"1">],role = <output>>], afir.asc_graph = #afir.asc_graph<tiling_key = 0,axes = [<id = 0, name = "z0", axis_type = Original, size = <"2">, align = <"-1">>, <id = 1, name = "z1", axis_type = Original, size = <"37">, align = <"-1">>, <id = 2, name = "z2", axis_type = Original, size = <"23">, align = <"-1">>, <id = 3, name = "z2T", axis_type = TileOuter, size = <"Ceiling((23 / (z2t_size)))">, align = <"1">, from = [2], split_pair_other_id = 4>, <id = 4, name = "z2t", axis_type = TileInner, size = <"z2t_size">, align = <"1">, from = [2], split_pair_other_id = 3>, <id = 5, name = "z0T", axis_type = TileOuter, size = <"Ceiling((2 / (z0t_size)))">, align = <"1">, from = [0], split_pair_other_id = 6>, <id = 6, name = "z0t", axis_type = TileInner, size = <"z0t_size">, align = <"1">, from = [0], split_pair_other_id = 5>, <id = 7, name = "z2Tz0T", axis_type = Merged, size = <"(Ceiling((2 / (z0t_size))) * Ceiling((23 / (z2t_size))))">, align = <"1">, from = [3, 5], split_pair_other_id = 0>, <id = 8, name = "z2Tz0TB", axis_type = BlockOuter, size = <"Ceiling((Ceiling((2 / (z0t_size))) * Ceiling((23 / (z2t_size))) / (z2Tz0Tb_size)))">, align = <"1">, from = [7], split_pair_other_id = 9>, <id = 9, name = "z2Tz0Tb", axis_type = BlockInner, size = <"z2Tz0Tb_size">, align = <"1">, from = [7], split_pair_other_id = 8>], type = Compute, size_var = [<"z2t_size">, <"z0t_size">, <"z2Tz0Tb_size">]>, afir.axis_bindings = [#afir.axis_binding<axis = 0, role = axis_size>, #afir.axis_binding<axis = 1, role = axis_size>, #afir.axis_binding<axis = 2, role = axis_size>, #afir.axis_binding<axis = 3, role = axis_index>, #afir.axis_binding<axis = 3, role = axis_size>, #afir.axis_binding<axis = 3, role = loop_size>, #afir.axis_binding<axis = 4, role = axis_size>, #afir.axis_binding<axis = 4, role = tail_size>, #afir.axis_binding<axis = 4, role = actual_size>, #afir.axis_binding<axis = 5, role = axis_index>, #afir.axis_binding<axis = 5, role = axis_size>, #afir.axis_binding<axis = 5, role = loop_size>, #afir.axis_binding<axis = 6, role = axis_size>, #afir.axis_binding<axis = 6, role = tail_size>, #afir.axis_binding<axis = 6, role = actual_size>, #afir.axis_binding<axis = 7, role = axis_index>, #afir.axis_binding<axis = 7, role = axis_size>, #afir.axis_binding<axis = 8, role = axis_index>, #afir.axis_binding<axis = 8, role = axis_size>, #afir.axis_binding<axis = 8, role = loop_size>, #afir.axis_binding<axis = 9, role = loop>, #afir.axis_binding<axis = 9, role = axis_size>, #afir.axis_binding<axis = 9, role = tail_size>, #afir.axis_binding<axis = 9, role = actual_size>], afir.buffer_ids = array<i64: 0>, afir.double_buffer = false, afir.function_role = #afir.function_role<kernel_body>, afir.original_graph_name = "mlir_load_transpose_store_2x37x23_0_B0X2Y0_S0G0C0", afir.queue_ids = array<i64: 0, 1>, afir.workspace_size = #afir.expr<"0">, afplan.variant = #afplan.variant<stage = 0, template = 0>, ascendc.aicore, ascendc.num_inputs = 2 : i32} {
    %0 = afir.tiling_param "z2Tz0Tb_size" {afir.axis_binding = #afir.axis_binding<axis = 9, role = axis_size>} : index
    %1 = afir.tiling_param "q0_size" : index
    %2 = afir.tiling_param "q1_size" : index
    %3 = afir.tiling_param "b0_size" : index
    %4 = afir.tiling_param "z2t_size" {afir.axis_binding = #afir.axis_binding<axis = 4, role = axis_size>} : index
    %5 = afir.tiling_param "z0t_size" {afir.axis_binding = #afir.axis_binding<axis = 6, role = axis_size>} : index
    %6 = afir.symbolic_eval %5 {expr = #afir.expr<"(2 / (__arg0))">} : (index) -> index
    %7 = afir.symbolic_eval %4 {expr = #afir.expr<"(23 / (__arg0))">} : (index) -> index
    %8 = afir.symbolic_eval %5 {expr = #afir.expr<"(2 / (__arg0))">} : (index) -> index
    %9 = afir.symbolic_eval %4 {expr = #afir.expr<"(23 / (__arg0))">} : (index) -> index
    %alloc = memref.alloc(%6, %5, %7, %4) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x37x?x?xf32, #afir.space<ub>>
    %alloc_0 = memref.alloc(%8, %5, %9, %4) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 2, queue_depth = 2, size_from_resource = true, life_time_axis_id = -1>} : memref<?x?x?x?x37xf32, #afir.space<ub>>
    %alloc_1 = memref.alloc(%3) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xui8, #afir.space<ub>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %10 = afir.symbolic_eval  {afir.axis_binding = #afir.axis_binding<axis = 0, role = axis_size>, expr = #afir.expr<"2">} : () -> index
    %11 = afir.symbolic_eval  {afir.axis_binding = #afir.axis_binding<axis = 1, role = axis_size>, expr = #afir.expr<"37">} : () -> index
    %12 = afir.symbolic_eval  {afir.axis_binding = #afir.axis_binding<axis = 2, role = axis_size>, expr = #afir.expr<"23">} : () -> index
    %13 = arith.remsi %12, %4 {afir.axis_binding = #afir.axis_binding<axis = 4, role = tail_size>} : index
    %14 = arith.divsi %12, %4 {afir.axis_binding = #afir.axis_binding<axis = 3, role = axis_size>} : index
    %15 = arith.cmpi ne, %13, %c0 : index
    %16 = arith.select %15, %c1, %c0 : index
    %17 = arith.addi %14, %16 {afir.axis_binding = #afir.axis_binding<axis = 3, role = loop_size>} : index
    %18 = arith.remsi %10, %5 {afir.axis_binding = #afir.axis_binding<axis = 6, role = tail_size>} : index
    %19 = arith.divsi %10, %5 {afir.axis_binding = #afir.axis_binding<axis = 5, role = axis_size>} : index
    %20 = arith.cmpi ne, %18, %c0 : index
    %21 = arith.select %20, %c1, %c0 : index
    %22 = arith.addi %19, %21 {afir.axis_binding = #afir.axis_binding<axis = 5, role = loop_size>} : index
    %23 = arith.muli %c1, %17 : index
    %24 = arith.muli %23, %22 {afir.axis_binding = #afir.axis_binding<axis = 7, role = axis_size>} : index
    %25 = arith.remsi %24, %0 {afir.axis_binding = #afir.axis_binding<axis = 9, role = tail_size>} : index
    %26 = arith.divsi %24, %0 {afir.axis_binding = #afir.axis_binding<axis = 8, role = axis_size>} : index
    %27 = arith.cmpi ne, %25, %c0 : index
    %28 = arith.select %27, %c1, %c0 : index
    %29 = arith.addi %26, %28 {afir.axis_binding = #afir.axis_binding<axis = 8, role = loop_size>} : index
    %30 = arith.muli %22, %c1 : index
    %31 = afir.block_id {dimension = 0 : i64} : index
    %32 = arith.remsi %31, %29 {afir.axis_binding = #afir.axis_binding<axis = 8, role = axis_index>} : index
    %33 = arith.cmpi slt, %32, %26 : index
    %34 = arith.select %33, %0, %25 {afir.axis_binding = #afir.axis_binding<axis = 9, role = actual_size>} : index
    %35 = arith.muli %32, %0 : index
    scf.for %arg4 = %c0 to %34 step %c1 {
      %36 = arith.addi %35, %arg4 {afir.axis_binding = #afir.axis_binding<axis = 7, role = axis_index>} : index
      %37 = arith.divsi %36, %30 : index
      %38 = arith.remsi %37, %17 {afir.axis_binding = #afir.axis_binding<axis = 3, role = axis_index>} : index
      %39 = arith.cmpi slt, %38, %14 : index
      %40 = arith.select %39, %4, %13 {afir.axis_binding = #afir.axis_binding<axis = 4, role = actual_size>} : index
      %41 = arith.remsi %36, %22 {afir.axis_binding = #afir.axis_binding<axis = 5, role = axis_index>} : index
      %42 = arith.cmpi slt, %41, %19 : index
      %43 = arith.select %42, %5, %18 {afir.axis_binding = #afir.axis_binding<axis = 6, role = actual_size>} : index
      %44 = afir.load %arg0 outs(%alloc) {afir.destination_descs = [#afir.tensor_desc<axis = [5, 6, 1, 3, 4],repeats = [<"(2 / (z0t_size))">, <"z0t_size">, <"37">, <"(23 / (z2t_size))">, <"z2t_size">],strides = [<"(851 * z0t_size)">, <"851">, <"23">, <"z2t_size">, <"1">],vectorized_axis = [6, 1, 4],vectorized_strides = [<"(592 * Ceiling((Rational(1 , 16) * z2t_size)))">, <"(16 * Ceiling((Rational(1 , 16) * z2t_size)))">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2],repeats = [<"2">, <"37">, <"23">],strides = [<"851">, <"23">, <"1">],role = <input>>], afir.result_descs = [#afir.tensor_desc<axis = [5, 6, 1, 3, 4],repeats = [<"(2 / (z0t_size))">, <"z0t_size">, <"37">, <"(23 / (z2t_size))">, <"z2t_size">],strides = [<"(851 * z0t_size)">, <"851">, <"23">, <"z2t_size">, <"1">],vectorized_axis = [6, 1, 4],vectorized_strides = [<"(592 * Ceiling((Rational(1 , 16) * z2t_size)))">, <"(16 * Ceiling((Rational(1 , 16) * z2t_size)))">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [8, 9, 6, 4, 1], loop_axis = 9>, afir.symbolic_shape = [#afir.expr<"(2 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"37">, #afir.expr<"(23 / (z2t_size))">, #afir.expr<"z2t_size">], indexing_maps = [#map, #map1]} : memref<2x37x23xf32, 22 : i32>, memref<?x?x37x?x?xf32, #afir.space<ub>> -> memref<?x?x37x?x?xf32, #afir.space<ub>>
      %45 = afir.transpose %44 outs(%alloc_0) tmp(%alloc_1) {afir.destination_descs = [#afir.tensor_desc<axis = [5, 6, 3, 4, 1],repeats = [<"(2 / (z0t_size))">, <"z0t_size">, <"(23 / (z2t_size))">, <"z2t_size">, <"37">],strides = [<"(851 * z0t_size)">, <"851">, <"(37 * z2t_size)">, <"37">, <"1">],vectorized_axis = [6, 4, 1],vectorized_strides = [<"(48 * z2t_size)">, <"48">, <"1">],role = <compute>>], afir.operand_descs = [#afir.tensor_desc<axis = [5, 6, 1, 3, 4],repeats = [<"(2 / (z0t_size))">, <"z0t_size">, <"37">, <"(23 / (z2t_size))">, <"z2t_size">],strides = [<"(851 * z0t_size)">, <"851">, <"23">, <"z2t_size">, <"1">],vectorized_axis = [6, 1, 4],vectorized_strides = [<"(592 * Ceiling((Rational(1 , 16) * z2t_size)))">, <"(16 * Ceiling((Rational(1 , 16) * z2t_size)))">, <"1">],role = <compute>>], afir.result_descs = [#afir.tensor_desc<axis = [5, 6, 3, 4, 1],repeats = [<"(2 / (z0t_size))">, <"z0t_size">, <"(23 / (z2t_size))">, <"z2t_size">, <"37">],strides = [<"(851 * z0t_size)">, <"851">, <"(37 * z2t_size)">, <"37">, <"1">],vectorized_axis = [6, 4, 1],vectorized_strides = [<"(48 * z2t_size)">, <"48">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [8, 9, 6, 4, 1], loop_axis = 9>, afir.symbolic_shape = [#afir.expr<"(2 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"(23 / (z2t_size))">, #afir.expr<"z2t_size">, #afir.expr<"37">], api_tiling_field = "transpose_tilingData_0", indexing_maps = [#map2, #map3]} : memref<?x?x37x?x?xf32, #afir.space<ub>>, memref<?x?x?x?x37xf32, #afir.space<ub>>, memref<?xui8, #afir.space<ub>> -> memref<?x?x?x?x37xf32, #afir.space<ub>>
      afir.store %45 outs(%arg1) {afir.destination_descs = [#afir.tensor_desc<axis = [0, 2, 1],repeats = [<"2">, <"23">, <"37">],strides = [<"851">, <"37">, <"1">],role = <output>>], afir.operand_descs = [#afir.tensor_desc<axis = [5, 6, 3, 4, 1],repeats = [<"(2 / (z0t_size))">, <"z0t_size">, <"(23 / (z2t_size))">, <"z2t_size">, <"37">],strides = [<"(851 * z0t_size)">, <"851">, <"(37 * z2t_size)">, <"37">, <"1">],vectorized_axis = [6, 4, 1],vectorized_strides = [<"(48 * z2t_size)">, <"48">, <"1">],role = <compute>>], afir.sched = #afir.sched<axis = [8, 9, 6, 4, 1], loop_axis = 9>, afir.symbolic_shape = [#afir.expr<"(2 / (z0t_size))">, #afir.expr<"z0t_size">, #afir.expr<"(23 / (z2t_size))">, #afir.expr<"z2t_size">, #afir.expr<"37">], indexing_maps = [#map4, #map5]} : memref<?x?x?x?x37xf32, #afir.space<ub>>, memref<2x23x37xf32, 22 : i32>
    } {afir.axis_binding = #afir.axis_binding<axis = 9, role = loop>}
    return
  }
}

// 不走 ND-DMA 那条路：这是普通 Load/Store 的跨段下降，两条路不该同时命中。
// IR-NOT: data_copy_nddma

// IR-LABEL: func.func private @stage0_template0
// blockLen 与 blockCount 的尾块修正：z2t/z0t 是切分轴，ActualSize 是运行期 select。
// IR: %[[LEN:.*]] = arith.select %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}} {afir.axis_binding = #afir.axis_binding<axis = 4, role = actual_size>} : index
// IR: %[[Z0T:.*]] = arith.select %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}} {afir.axis_binding = #afir.axis_binding<axis = 6, role = actual_size>} : index
// load 的 z0t 并进段内（count=37×z0t），这里不该再出现循环。
// IR-NOT: scf.for
// IR: %[[CNT:.*]] = index.mul %c37{{[^,]*}}, %[[Z0T]]
// IR: %[[CEIL:.*]] = index.ceildivu %{{[^,]*}}, %c16{{[^,]*}}
// IR: %[[UBP:.*]] = index.mul %c16{{[^,]*}}, %[[CEIL]]
// IR: %[[SRC:.*]] = arith.subi %c23{{[^,]*}}, %[[LEN]] : index
// IR: %[[DST:.*]] = arith.subi %[[UBP]], %[[LEN]] : index
// GM 侧带 tile 起点子视图；UB 侧下标恒 0（队列按 slot 轮转）。
// IR: %{{.*}} = ascendc.global_tensor.subindex %{{.*}}[%{{.*}}] : !ascendc.global_tensor<2x37x23xf32>, index, !ascendc.global_tensor
// IR: %{{.*}} = ascendc.local_tensor.subindex %{{.*}}[%c0{{.*}}] : !ascendc.local_tensor
// IR: ascendc.data_copy_pad_extend %{{[^,]*}}, %{{[^,]*}}, %[[CNT]], %[[LEN]], %[[SRC]], %[[DST]] : !ascendc.local_tensor<{{[^>]*}}>, !ascendc.global_tensor<{{[^>]*}}>, index, index, index, index

// store：GM 侧行距 37 == 行长 37（dstStride 两个常量相减得 0）；UB 侧行距 48（补位缝）。
// IR: %[[S_DST:.*]] = arith.subi %c37{{[^,]*}}, %c37{{[^,]*}} : index
// IR: %[[S_SRC:.*]] = arith.subi %c48{{[^,]*}}, %c37{{[^,]*}} : index
// 第三根向量轴（z0t）退化成外层循环，界是 ActualSize(z0t)。
// IR: scf.for %{{[^=]*}} = %c0{{[^ ]*}} to %[[Z0T]] step %c1{{[^ ]*}} {
// store 的 blockCount 复用 load 的 blockLen 那个 ActualSize(z2t)（`%[[LEN]]`），
// blockLen 是常量 37（z1 不切分、无尾块修正）；第一操作数换成 global_tensor。
// IR: ascendc.data_copy_pad_extend %{{[^,]*}}, %{{[^,]*}}, %[[LEN]], %c37{{[^,]*}}, %[[S_SRC]], %[[S_DST]] : !ascendc.global_tensor<{{[^>]*}}>, !ascendc.local_tensor<{{[^>]*}}>, index, index, index, index

// kernel 文本侧：与 IR 同一套代数。load 是 count=c37×z0t_actual、两个行距减 blockLen；
// store 的两个 stride 是常量差（0 与 11），且整条调用包在 z0t 的 for 里。
// CPP: uint32_t v{{[0-9]+}} = c37_idx * v{{[0-9]+}};
// CPP: uint32_t v{{[0-9]+}} = c23_idx - v{{[0-9]+}};
// CPP: DataCopyPadExtend<float>(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(v{{[0-9]+}}));
// CPP: uint32_t v{{[0-9]+}} = c37_idx - c37_idx;
// CPP: uint32_t v{{[0-9]+}} = c48_idx - c37_idx;
// CPP: for (uint32_t v{{[0-9]+}} = c0_idx; v{{[0-9]+}} < v{{[0-9]+}}; v{{[0-9]+}} += c1_idx) {
// CPP: DataCopyPadExtend<float>(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(c37_idx), static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(v{{[0-9]+}}));
