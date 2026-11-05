// Copyright (c) 2026 Huawei Technologies Co, Ltd.
// This program is free software and you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// **>4 维外抛**：五根向量化轴，最内四根进 `TransposeExtend<3, 4>`，最外那根（切分轴 `bt`）
// 由 `materializeVectorizedSegments` 折成外层 `scf.for`（上界含尾块修正）——legacy 的
// 「前 size-4 根轴进 `outer_loop_axes`」同一语义（`reg_transpose_api_call.cpp:156-169`）。
//
// 同一张图在 910B 上**就是表外 perm 的反例**：mem 侧的 21 项表只到 4 维，5 维 perm
// （`[0, 1, 4, 3, 2]`）查不到 → `unsupported`、整图回落 legacy。切分轴保持在最外、
// perm 全落在内层轴——这是**合法调度形态**里的表外 perm；把切分轴搬离最外的那些形态
// 在 reg 档就先挂了（`ScheduledContiguous`），轮不到 perm 判据说话。

// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc %s \
// RUN:   | FileCheck %s --check-prefix=IR
// RUN: not af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE

// 逻辑形状 `[1024, 3, 5, 16, 8]`（5D），`b` 被 `bt` 切块；perm 落在内层
// `(c2, z, w) → (w, z, c2)`。
#axes = #afir.asc_graph<
  tiling_key = 0,
  axes = [<id = 0, name = "b", axis_type = Original, size = #afir.expr<"1024">, split_pair_other_id = 0>,
          <id = 1, name = "c1", axis_type = Original, size = #afir.expr<"3">, split_pair_other_id = 1>,
          <id = 2, name = "c2", axis_type = Original, size = #afir.expr<"5">, split_pair_other_id = 2>,
          <id = 3, name = "z", axis_type = Original, size = #afir.expr<"16">, split_pair_other_id = 3>,
          <id = 4, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 4>,
          <id = 5, name = "bt", axis_type = TileInner, size = #afir.expr<"bt_size">, from = [0],
           split_pair_other_id = 8>,
          <id = 6, name = "bTB", axis_type = BlockOuter,
           size = #afir.expr<"Ceiling((Ceiling((1024 / (bt_size))) / (bTb_size)))">, from = [8],
           split_pair_other_id = 7>,
          <id = 7, name = "bTb", axis_type = BlockInner, size = #afir.expr<"bTb_size">, from = [8],
           split_pair_other_id = 6>,
          <id = 8, name = "bT", axis_type = TileOuter, size = #afir.expr<"Ceiling((1024 / (bt_size)))">,
           from = [0], split_pair_other_id = 5>],
  type = Compute,
  size_var = [#afir.expr<"bt_size">, #afir.expr<"bTb_size">]>

#src_tile = #afir.tensor_desc<axis = [6, 7, 5, 1, 2, 3, 4],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"3">, #afir.expr<"5">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"(1920 * bTb_size * bt_size)">, #afir.expr<"(1920 * bt_size)">, #afir.expr<"1920">,
               #afir.expr<"640">, #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    vectorized_axis = [5, 1, 2, 3, 4],
    vectorized_strides = [#afir.expr<"1920">, #afir.expr<"640">, #afir.expr<"128">, #afir.expr<"8">,
                           #afir.expr<"1">],
    role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [6, 7, 5, 1, 4, 3, 2],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"3">, #afir.expr<"8">, #afir.expr<"16">, #afir.expr<"5">],
    strides = [#afir.expr<"(1200 * bTb_size * bt_size)">, #afir.expr<"(1200 * bt_size)">, #afir.expr<"1200">,
               #afir.expr<"400">, #afir.expr<"80">, #afir.expr<"5">, #afir.expr<"1">],
    vectorized_axis = [5, 1, 4, 3, 2],
    vectorized_strides = [#afir.expr<"1200">, #afir.expr<"400">, #afir.expr<"80">, #afir.expr<"5">,
                           #afir.expr<"1">],
    role = <compute>>

#gm5d = #afir.tensor_desc<axis = [0, 1, 2, 3, 4],
    repeats = [#afir.expr<"1024">, #afir.expr<"3">, #afir.expr<"5">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"1920">, #afir.expr<"640">, #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    role = <input>>

func.func private @outer_loop_transpose(
    %input: memref<1024x3x5x16x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                      afir.buffer_desc = #afir.buffer_desc<space = <gm>,
                                                                           size_bytes_expr = <"7864320">,
                                                                           life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm5d],
  afir.asc_graph = #axes,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?x?x?x3x5x16x8xf32>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x3x8x16x5xf32>
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %loaded = afir.load %input outs(%src) {
    afir.destination_descs = [#src_tile],
    afir.operand_descs = [#gm5d],
    afir.result_descs = [#src_tile],
    afir.sched = #afir.sched<axis = [6, 7, 5, 1, 2, 3, 4], loop_axis = 7>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7, d1, d2, d3, d4)>]
  } : memref<1024x3x5x16x8xf32>, memref<?x?x?x3x5x16x8xf32> -> memref<?x?x?x3x5x16x8xf32>
  %result = afir.transpose %loaded outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src_tile],
    afir.destination_descs = [#dst_tile],
    afir.result_descs = [#dst_tile],
    afir.sched = #afir.sched<axis = [6, 7, 5, 1, 2, 3, 4], loop_axis = 7>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>,
                     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d6, d5, d4)>]
  } : memref<?x?x?x3x5x16x8xf32>, memref<?x?x?x3x8x16x5xf32>, memref<?xui8> -> memref<?x?x?x3x8x16x5xf32>
  return
}

// SELECT: ascend.implementation = {id = "transpose.ub_ub_reg_base", kind = "ascendc_api"}

// 五根向量化轴 > 包装上限 4：最外一根折成外层 `scf.for`（切分轴 `bt` 那一段，上界带尾块
// 修正），循环体内是最内四段的 `TransposeExtend<3, 4>`。**没有 `scf.for` 就是外抛没做**，
// 发出 5 段的调用（包装只收 `dim ≤ 4`，设备编译都过不了）。
// IR-LABEL: func.func private @outer_loop_transpose
// IR: scf.for
// IR: ascendc.transpose_extend_reg_base
// IR-SAME: innerDim = 3 : i8
// IR-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 4, 4, 4>
// IR-SAME: totalDim = 4 : i8
// IR-NOT: ascendc.confusion_transpose

// mem 侧的 21 项表只到 4 维：5 维 perm 在表外，`unsupported` 回落 legacy（legacy 在这里
// 是 `TRANSPOSE_INVALID` 编译失败，MLIR 报不支持严格不劣于它）。
// WRONGSIDE: no supported
