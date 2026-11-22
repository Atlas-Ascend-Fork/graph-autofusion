// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.transpose`（带 host tiling）→ mem 侧的
// `codegen::ConfusionTranspose<float>(dst, src, tmp, AutoFuseTransposeType::X, tiling)`。
//
// IR 与 `afir-transpose-reg-base.mlir` 那一格**只差两个属性**：func 级
// `afir.api_tiling_fields`（声明嵌套 struct 的字段布局）和 op 级 `api_tiling_field`
// （本节点取哪个字段）。同一张图在两个家族下各走各的下降，这就是矩阵 tier D2 的
// 「API 名 / 扩展头 / tmp / host tiling」四项差集在一处的样子。
//
// **函数壳是 canonical kernel ABI 的**（`afir-nddma-reg-base.mlir` 同款）：GM 入参带
// `afir.kernel_arg` / GM `buffer_desc`，func 挂 `afir.arg_descs`；`afir.load` 把 tile 搬进
// UB，transpose 消费它的结果。没有 store——nddma 那份先例同样止于一个消费 op，本文件钉
// 的是 transpose 的下降，不是整图。
//
// perm `[0, 2, 1]`（`(z, w) → (w, z)`）折成 `TRANSPOSE_ND2ND_021`——21 项表里 3 维的
// 那一行。**tiling struct 必须在相位 A 就进 canonical 实参**：conversion（相位 C）按字段
// 名在实参上建 `emitasc.member` 访问；这个 RUN 链就是那道链路的回归——字段没并进去的话，
// mem 档报「TilingData is missing the transpose field」，`afir.transpose` 留在 IR 里不降。
//
// **maps 是 bridge 逐字会发的**（`AfirBuilder.cpp` 的 `makeTransferIndexingMapsAttr`）：
// domain = 两侧 desc 的 axis 并集（源在前），每侧结果 = 自己 rank 位置上的 axis 对应的
// 迭代维；不在并集里的位置发常数 0。load 与 transpose 各自按这条规则生成，本文件不手编。

// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate %s | FileCheck %s --check-prefix=ORDER
// ORDER-LABEL: func.func private @kept_transpose_tile
// ORDER-NOT: ascendc.pipe.init_buffer
// ORDER: ascendc.pipe.init_queue
// ORDER: ascendc.pipe.init_buffer

// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc %s \
// RUN:   | FileCheck %s --check-prefix=IR
// **本文件不做 950 的 run**：2D 那一格 reg 档收不了（切分轴不在最外，`ScheduledContiguous`
// 不成立），同一文件里三个 func 在 950 上必挂。「同一张图两个家族各走各的」那一半故事由
// `afir-transpose-reg-base.mlir`（950 正例）与本文件的 3D/4D 形状共同讲。

// --- 3D `[0, 2, 1]`：`TRANSPOSE_ND2ND_021`（f16 主格）--------------------------
#axes = #afir.asc_graph<
  tiling_key = 0,
  axes = [<id = 0, name = "b", axis_type = Original, size = #afir.expr<"1024">, split_pair_other_id = 0>,
          <id = 1, name = "z", axis_type = Original, size = #afir.expr<"16">, split_pair_other_id = 1>,
          <id = 2, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 2>,
          <id = 3, name = "bt", axis_type = TileInner, size = #afir.expr<"bt_size">, from = [0],
           split_pair_other_id = 6>,
          <id = 4, name = "bTB", axis_type = BlockOuter,
           size = #afir.expr<"Ceiling((Ceiling((1024 / (bt_size))) / (bTb_size)))">, from = [6],
           split_pair_other_id = 5>,
          <id = 5, name = "bTb", axis_type = BlockInner, size = #afir.expr<"bTb_size">, from = [6],
           split_pair_other_id = 4>,
          <id = 6, name = "bT", axis_type = TileOuter, size = #afir.expr<"Ceiling((1024 / (bt_size)))">,
           from = [0], split_pair_other_id = 3>],
  type = Compute,
  size_var = [#afir.expr<"bt_size">, #afir.expr<"bTb_size">]>

#gm_in = #afir.tensor_desc<axis = [0, 1, 2],
    repeats = [#afir.expr<"1024">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">], role = <input>>
#src_tile = #afir.tensor_desc<axis = [4, 5, 3, 1, 2],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"(128 * bTb_size * bt_size)">, #afir.expr<"(128 * bt_size)">, #afir.expr<"128">,
               #afir.expr<"8">, #afir.expr<"1">],
    vectorized_axis = [3, 1, 2],
    vectorized_strides = [#afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [4, 5, 3, 2, 1],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"8">, #afir.expr<"16">],
    strides = [#afir.expr<"(128 * bTb_size * bt_size)">, #afir.expr<"(128 * bt_size)">, #afir.expr<"128">,
               #afir.expr<"16">, #afir.expr<"1">],
    vectorized_axis = [3, 2, 1],
    vectorized_strides = [#afir.expr<"128">, #afir.expr<"16">, #afir.expr<"1">],
    role = <compute>>

func.func private @kept_transpose_tile(
    %input: memref<1024x16x8xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                   afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"262144">,
                                                                        life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_in],
  afir.asc_graph = #axes,
  // host tiling 字段：bridge 的 `<nodeName>_tilingData_<tiling_case_id>` 形态（legacy 期望件
  // `tests/st/codegen/e2e/load_transpose_store_expect_code` 里的 `transpose_tilingData_0`
  // 同名）；成员名抄 `ConfusionTransposeTiling` 的 12 个 param（`tests/st/att/.../tiling_api.h`）。
  afir.api_tiling_fields = [{field = "transpose_tilingData_0", type = "ConfusionTransposeTiling",
                             fields = ["param0", "param1", "param2", "param3", "param4", "param5",
                                       "param6", "param7", "param8", "param9", "param10", "param11"]}],
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?x?x?x16x8xf16>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16xf16>
  // tmp 是 ui8 字节 scratch（`ConfusionTranspose` 的形参是 `LocalTensor<uint8_t>`）。
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  // load 与 transpose 同一 sched（同一 tile 空间）；maps 按 axis 并集规则机械生成
  // （domain [0,1,2 | 4,5,3]，GM 侧 (d0,d1,d2)，tile 侧 (d3,d4,d5,d1,d2)）。
  %loaded = afir.load %input outs(%src) {
    afir.destination_descs = [#src_tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#src_tile],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1, 2], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d1, d2)>]
  } : memref<1024x16x8xf16>, memref<?x?x?x16x8xf16> -> memref<?x?x?x16x8xf16>
  %result = afir.transpose %loaded outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src_tile],
    afir.destination_descs = [#dst_tile],
    afir.result_descs = [#dst_tile],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1, 2], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>],
    api_tiling_field = "transpose_tilingData_0"
  } : memref<?x?x?x16x8xf16>, memref<?x?x?x8x16xf16>, memref<?xui8> -> memref<?x?x?x8x16xf16>
  return
}

// SELECT-LABEL: func.func private @kept_transpose_tile
// SELECT: ascend.implementation = {id = "transpose.ub_ub_extend", kind = "ascendc_api"}

// mem 档三件套：按字段名取整个 `ConfusionTransposeTiling`（`emitasc.member` 是一份局部
// 拷贝，非 const 引用绑上去，legacy 的 `auto apiTilingData = t-><field>;` 同一形态）、
// perm 折好的枚举、**不发 reg 侧那条**。偏移侧的 subindex（本地恒 0）不在此钉。
// IR-LABEL: func.func private @kept_transpose_tile
// IR: emitasc.member {{.*}} "transpose_tilingData_0"
// IR: ascendc.confusion_transpose
// 枚举按底层 i32 打印（`AutoFuseTransposeType` 的声明序）：ONLY=0、021=4、0321=6，
// 与 `transpose_base_type.h` 的取值一致；名字在 C++ 发射侧还原。
// IR-SAME: transposeType = 4 : i32
// IR-NOT: ascendc.transpose_extend_reg_base
// IR-NOT: afir.transpose

// --- 2D `[1, 0]`：`TRANSPOSE_ND2ND_ONLY` ---------------------------------------
//
// 逻辑形状 `[1024, 8]`，perm 把两根轴全换（`(b, w) → (w, b)`）——切分轴不再在最外。
// **reg 档收不了这种形态**（`ScheduledContiguous` 要求输出侧 `vectorized_axis.front()` 是
// 切分轴且配对 `TileOuter`，这里最外是 `w`），mem 档没有这条约束（偏移恒 0、不建循环，
// 尺寸全在 host tiling 里）——这一格同时是「两档能力面的差」的证据。
#axes2d = #afir.asc_graph<
  tiling_key = 0,
  axes = [<id = 0, name = "b", axis_type = Original, size = #afir.expr<"1024">, split_pair_other_id = 0>,
          <id = 1, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 1>,
          <id = 2, name = "bt", axis_type = TileInner, size = #afir.expr<"bt_size">, from = [0],
           split_pair_other_id = 5>,
          <id = 3, name = "bTB", axis_type = BlockOuter,
           size = #afir.expr<"Ceiling((Ceiling((1024 / (bt_size))) / (bTb_size)))">, from = [5],
           split_pair_other_id = 4>,
          <id = 4, name = "bTb", axis_type = BlockInner, size = #afir.expr<"bTb_size">, from = [5],
           split_pair_other_id = 3>,
          <id = 5, name = "bT", axis_type = TileOuter, size = #afir.expr<"Ceiling((1024 / (bt_size)))">,
           from = [0], split_pair_other_id = 2>],
  type = Compute,
  size_var = [#afir.expr<"bt_size">, #afir.expr<"bTb_size">]>

#gm2d = #afir.tensor_desc<axis = [0, 1],
    repeats = [#afir.expr<"1024">, #afir.expr<"8">],
    strides = [#afir.expr<"8">, #afir.expr<"1">], role = <input>>
#src2d = #afir.tensor_desc<axis = [3, 4, 2, 1],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"8">],
    strides = [#afir.expr<"(8 * bTb_size * bt_size)">, #afir.expr<"(8 * bt_size)">, #afir.expr<"8">,
               #afir.expr<"1">],
    vectorized_axis = [2, 1],
    vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">],
    role = <compute>>
#dst2d = #afir.tensor_desc<axis = [3, 4, 1, 2],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"8">,
               #afir.expr<"bt_size">],
    strides = [#afir.expr<"(8 * bTb_size * bt_size)">, #afir.expr<"(8 * bt_size)">, #afir.expr<"bt_size">,
               #afir.expr<"1">],
    vectorized_axis = [1, 2],
    vectorized_strides = [#afir.expr<"bt_size">, #afir.expr<"1">],
    role = <compute>>

func.func private @two_dim_only(
    %input: memref<1024x8xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"16384">,
                                                                     life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm2d],
  afir.asc_graph = #axes2d,
  afir.api_tiling_fields = [{field = "transpose_tilingData_0", type = "ConfusionTransposeTiling",
                             fields = ["param0", "param1", "param2", "param3", "param4", "param5",
                                       "param6", "param7", "param8", "param9", "param10", "param11"]}],
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?x?x?x8xf16>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x8x?xf16>
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %loaded = afir.load %input outs(%src) {
    afir.destination_descs = [#src2d],
    afir.operand_descs = [#gm2d],
    afir.result_descs = [#src2d],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4, d1)>]
  } : memref<1024x8xf16>, memref<?x?x?x8xf16> -> memref<?x?x?x8xf16>
  %result = afir.transpose %loaded outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src2d],
    afir.destination_descs = [#dst2d],
    afir.result_descs = [#dst2d],
    afir.sched = #afir.sched<axis = [3, 4, 2, 1], loop_axis = 4>,
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>],
    api_tiling_field = "transpose_tilingData_0"
  } : memref<?x?x?x8xf16>, memref<?x?x8x?xf16>, memref<?xui8> -> memref<?x?x8x?xf16>
  return
}

// SELECT-LABEL: func.func private @two_dim_only
// SELECT: ascend.implementation = {id = "transpose.ub_ub_extend", kind = "ascendc_api"}
// IR-LABEL: func.func private @two_dim_only
// IR: ascendc.confusion_transpose
// IR-SAME: transposeType = 0 : i32
// IR-NOT: ascendc.transpose_extend_reg_base

// --- 4D `[0, 3, 2, 1]`：`TRANSPOSE_ND2ND_0321` ---------------------------------
//
// 逻辑形状 `[1024, 3, 16, 8]`，perm `(c, z, w) → (w, z, c)`——4D 表里 `0321` 那一行。
// 切分轴仍在最外，reg 档也收（两侧能力面在这一格重叠，各自的断言在各自的文件里）。
#axes4d = #afir.asc_graph<
  tiling_key = 0,
  axes = [<id = 0, name = "b", axis_type = Original, size = #afir.expr<"1024">, split_pair_other_id = 0>,
          <id = 1, name = "c", axis_type = Original, size = #afir.expr<"3">, split_pair_other_id = 1>,
          <id = 2, name = "z", axis_type = Original, size = #afir.expr<"16">, split_pair_other_id = 2>,
          <id = 3, name = "w", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 3>,
          <id = 4, name = "bt", axis_type = TileInner, size = #afir.expr<"bt_size">, from = [0],
           split_pair_other_id = 7>,
          <id = 5, name = "bTB", axis_type = BlockOuter,
           size = #afir.expr<"Ceiling((Ceiling((1024 / (bt_size))) / (bTb_size)))">, from = [7],
           split_pair_other_id = 6>,
          <id = 6, name = "bTb", axis_type = BlockInner, size = #afir.expr<"bTb_size">, from = [7],
           split_pair_other_id = 5>,
          <id = 7, name = "bT", axis_type = TileOuter, size = #afir.expr<"Ceiling((1024 / (bt_size)))">,
           from = [0], split_pair_other_id = 4>],
  type = Compute,
  size_var = [#afir.expr<"bt_size">, #afir.expr<"bTb_size">]>

#gm4d = #afir.tensor_desc<axis = [0, 1, 2, 3],
    repeats = [#afir.expr<"1024">, #afir.expr<"3">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"384">, #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">], role = <input>>
#src4d = #afir.tensor_desc<axis = [5, 6, 4, 1, 2, 3],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"3">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"(384 * bTb_size * bt_size)">, #afir.expr<"(384 * bt_size)">, #afir.expr<"384">,
               #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    vectorized_axis = [4, 1, 2, 3],
    vectorized_strides = [#afir.expr<"384">, #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    role = <compute>>
#dst4d = #afir.tensor_desc<axis = [5, 6, 4, 3, 2, 1],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"8">, #afir.expr<"16">, #afir.expr<"3">],
    strides = [#afir.expr<"(384 * bTb_size * bt_size)">, #afir.expr<"(384 * bt_size)">, #afir.expr<"384">,
               #afir.expr<"48">, #afir.expr<"3">, #afir.expr<"1">],
    vectorized_axis = [4, 3, 2, 1],
    vectorized_strides = [#afir.expr<"384">, #afir.expr<"48">, #afir.expr<"3">, #afir.expr<"1">],
    role = <compute>>

func.func private @four_dim_0321(
    %input: memref<1024x3x16x8xf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                     afir.buffer_desc = #afir.buffer_desc<space = <gm>,
                                                                          size_bytes_expr = <"786432">,
                                                                          life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm4d],
  afir.asc_graph = #axes4d,
  afir.api_tiling_fields = [{field = "transpose_tilingData_0", type = "ConfusionTransposeTiling",
                             fields = ["param0", "param1", "param2", "param3", "param4", "param5",
                                       "param6", "param7", "param8", "param9", "param10", "param11"]}],
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?x?x?x3x16x8xf16>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16x3xf16>
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %loaded = afir.load %input outs(%src) {
    afir.destination_descs = [#src4d],
    afir.operand_descs = [#gm4d],
    afir.result_descs = [#src4d],
    afir.sched = #afir.sched<axis = [5, 6, 4, 1, 2, 3], loop_axis = 6>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d1, d2, d3)>]
  } : memref<1024x3x16x8xf16>, memref<?x?x?x3x16x8xf16> -> memref<?x?x?x3x16x8xf16>
  %result = afir.transpose %loaded outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src4d],
    afir.destination_descs = [#dst4d],
    afir.result_descs = [#dst4d],
    afir.sched = #afir.sched<axis = [5, 6, 4, 1, 2, 3], loop_axis = 6>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5, d4, d3)>],
    api_tiling_field = "transpose_tilingData_0"
  } : memref<?x?x?x3x16x8xf16>, memref<?x?x?x8x16x3xf16>, memref<?xui8> -> memref<?x?x?x8x16x3xf16>
  return
}

// SELECT-LABEL: func.func private @four_dim_0321
// SELECT: ascend.implementation = {id = "transpose.ub_ub_extend", kind = "ascendc_api"}
// IR-LABEL: func.func private @four_dim_0321
// IR: ascendc.confusion_transpose
// IR-SAME: transposeType = 6 : i32
