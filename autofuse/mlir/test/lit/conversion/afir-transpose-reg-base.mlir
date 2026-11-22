// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software and you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.transpose` → reg 侧的 `TransposeExtend<inner, total>(dst, src, tmp, {...}, {...}, {...})`。
//
// **保留转置的标准形态**：分块轴（`bt`）两侧都在最外，perm 落在内层轴（`(z, w) → (w, z)`）。
// 这正是调度器留下的形状——尾轴转置才有得分（score = 1），调度不会消掉它；perm 落在分块轴
// 上的那种早就被重排进 GM 搬运了。`dst` 侧 `vectorized_axis.front()` 是 `bt`（TileInner），
// `ScheduledContiguous` 因此成立：分块内层循环由循环生成接管，三根内层轴各成一段。
//
// perm 本身只活在 descriptor 里（两侧 `vectorized_axis` 的次序不同），两份 indexing map 都是
// 直投影（各用五个互不重复的维度）——下降侧从 descriptor 的次序差推导 perm
// （`analyzeTranspose`，对应 legacy 的 `ReorderInputStrideByOutputAxisOrder`），maps 只承担
// 「两端逻辑坐标同一批」的等量证明（`ExpectEq(1024/(bTb*bt) * bTb * bt * 16 * 8, ...)`，
// 逐字相同，`Prove` 直接过）。
//
// **这一层钉「三段 + inner/total 模板实参」**：`operandSegmentSizes` 后三组各 3、
// `innerDim = 3`、`totalDim = 3`。每一段的数值是 SSA 值，由
// `test/lit/target/ascendc-transpose-reg-base.mlir` 钉 C++ 形态、由 CPU E2E 钉数值。
//
// `bf16` 那一格是 dtype 覆盖：reg 收、mem 不收（`isSupportedTransposeElementType` 的家族
// 分支），在 950 上它与 f16 同路。

// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc %s \
// RUN:   | FileCheck %s --check-prefix=IR

// `b`（1024）被 `bt` 切块：`bt` 是 TileInner（配对 `bT`），`bTB`/`bTb` 是块外/块内。
// `z`(16)、`w`(8) 是不切的原始轴——perm 就落在这两根上。
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

// 输入 tile：`(bt, z, w)` 布局，行距 8（w 连续、z 步 8、bt 步 128 = 16×8）。
#src_tile = #afir.tensor_desc<axis = [4, 5, 3, 1, 2],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"(128 * bTb_size * bt_size)">, #afir.expr<"(128 * bt_size)">, #afir.expr<"128">,
               #afir.expr<"8">, #afir.expr<"1">],
    vectorized_axis = [3, 1, 2],
    vectorized_strides = [#afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">],
    role = <compute>>

// 输出 tile：`(bt, w, z)` 布局——同一批元素换成 z 连续（z 步 1、w 步 16、bt 步 128 = 8×16）。
// `vectorized_axis.front()` 仍是 `bt`，切分轴不动，perm 全在内层。
#dst_tile = #afir.tensor_desc<axis = [4, 5, 3, 2, 1],
    repeats = [#afir.expr<"(1024 / (bTb_size * bt_size))">, #afir.expr<"bTb_size">, #afir.expr<"bt_size">,
               #afir.expr<"8">, #afir.expr<"16">],
    strides = [#afir.expr<"(128 * bTb_size * bt_size)">, #afir.expr<"(128 * bt_size)">, #afir.expr<"128">,
               #afir.expr<"16">, #afir.expr<"1">],
    vectorized_axis = [3, 2, 1],
    vectorized_strides = [#afir.expr<"128">, #afir.expr<"16">, #afir.expr<"1">],
    role = <compute>>

// GM 侧入口（与 `afir-transpose-mem-extend.mlir` 同一张图的 f32 版）：`afir.load` 把 tile
// 搬进 UB，maps 按 axis 并集规则机械生成（domain [0,1,2 | 4,5,3]）。
#gm_in = #afir.tensor_desc<axis = [0, 1, 2],
    repeats = [#afir.expr<"1024">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">], role = <input>>

func.func private @kept_transpose_tile(
    %input: memref<1024x16x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                   afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"524288">,
                                                                        life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm_in],
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
      : memref<?x?x?x16x8xf32>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16xf32>
  // tmp 是全生命周期 TBuf（`life_time_axis_id = -1`、无队列字段），独占整块资源；元素类型
  // ui8——两个包装的形参都是 `LocalTensor<uint8_t>`（字节scratch，不是同 T 的数据缓冲）。
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %loaded = afir.load %input outs(%src) {
    afir.destination_descs = [#src_tile],
    afir.operand_descs = [#gm_in],
    afir.result_descs = [#src_tile],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1, 2], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d1, d2)>]
  } : memref<1024x16x8xf32>, memref<?x?x?x16x8xf32> -> memref<?x?x?x16x8xf32>
  %result = afir.transpose %loaded outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src_tile],
    afir.destination_descs = [#dst_tile],
    afir.result_descs = [#dst_tile],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1, 2], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>]
  } : memref<?x?x?x16x8xf32>, memref<?x?x?x8x16xf32>, memref<?xui8> -> memref<?x?x?x8x16xf32>
  return
}

// 同文件不再放 bf16 格：搬运族（`afir.load`/`afir.store`）的 ODS 元素类型表**不含
// bf16**（`Data.td` 的 `AFIR_DataMoveElementTypes`），GM 腿进不来 bf16——真实图里 bf16
// 靠 cast 换进来。dtype 那一面由 `afir-transpose-reg-base-bf16.mlir` 以简单形态单独钉
// （`afir.abs` 的 `afir-abs-extension-api.mlir` 同款，不进 kernel ABI 壳）。

// 两格都选中 reg 侧那一档。
// SELECT-LABEL: func.func private @kept_transpose_tile
// SELECT: ascend.implementation = {id = "transpose.ub_ub_reg_base", kind = "ascendc_api"}

// 三根内层轴各成一段（不合并——重排后的布局不连续），`dst/src/tmp` 各一，三组尺寸各 3；
// 模板实参 inner = 3（参与转置的内层数）、total = 3（总段数）。**不发 mem 侧那条**：
// 两条路不该同时命中。
// IR-LABEL: func.func private @kept_transpose_tile
// IR: ascendc.transpose_extend_reg_base
// 属性按字典序打印，SAME 按同一序给（`innerDim` < `operandSegmentSizes` < `totalDim`）。
// IR-SAME: innerDim = 3 : i8
// IR-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 3, 3, 3>
// IR-SAME: totalDim = 3 : i8
// IR-NOT: ascendc.confusion_transpose

// mem 侧选不出：这一格没有 `api_tiling_field`（reg 档不收 host tiling），mem 档缺它没有
// `ConfusionTransposeTiling` 可取，整图回落 legacy。
// RUN: not af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE
// WRONGSIDE: no supported

// --- 4D `[0, 3, 2, 1]`：四段、inner 3 ------------------------------------------------
//
// 逻辑形状 `[1024, 3, 16, 8]`，切分轴仍在最外（reg 档的 `ScheduledContiguous` 成立），
// perm `(c, z, w) → (w, z, c)` 落在内层轴。四根向量化轴全部进调用（`totalDim = 4` 是包装
// 的上限，不到外抛线），contiguous 的内三根折成 `innerDim = 3`。
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

#gm4d = #afir.tensor_desc<axis = [0, 1, 2, 3],
    repeats = [#afir.expr<"1024">, #afir.expr<"3">, #afir.expr<"16">, #afir.expr<"8">],
    strides = [#afir.expr<"384">, #afir.expr<"128">, #afir.expr<"8">, #afir.expr<"1">], role = <input>>

func.func private @four_dim_segments(
    %input: memref<1024x3x16x8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                     afir.buffer_desc = #afir.buffer_desc<space = <gm>,
                                                                          size_bytes_expr = <"1572864">,
                                                                          life_time_axis_id = -1>}) attributes {
  afir.arg_descs = [#gm4d],
  afir.asc_graph = #axes4d,
  afir.function_role = #afir.function_role<kernel_body>
} {
  %tile = afir.tiling_param "bt_size" : index
  %block = afir.tiling_param "bTb_size" : index
  %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(1024 / (__arg0 * __arg1))">}
      : (index, index) -> index
  %src = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?x?x?x3x16x8xf32>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16x3xf32>
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
  } : memref<1024x3x16x8xf32>, memref<?x?x?x3x16x8xf32> -> memref<?x?x?x3x16x8xf32>
  %result = afir.transpose %loaded outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src4d],
    afir.destination_descs = [#dst4d],
    afir.result_descs = [#dst4d],
    afir.sched = #afir.sched<axis = [5, 6, 4, 1, 2, 3], loop_axis = 6>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5, d4, d3)>]
  } : memref<?x?x?x3x16x8xf32>, memref<?x?x?x8x16x3xf32>, memref<?xui8> -> memref<?x?x?x8x16x3xf32>
  return
}

// SELECT-LABEL: func.func private @four_dim_segments
// SELECT: ascend.implementation = {id = "transpose.ub_ub_reg_base", kind = "ascendc_api"}
// IR-LABEL: func.func private @four_dim_segments
// IR: ascendc.transpose_extend_reg_base
// IR-SAME: innerDim = 3 : i8
// IR-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 4, 4, 4>
// IR-SAME: totalDim = 4 : i8
// IR-NOT: ascendc.confusion_transpose
