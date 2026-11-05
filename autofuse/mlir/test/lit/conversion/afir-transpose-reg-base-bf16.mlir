// Copyright (c) 2026 Huawei Technologies Co, Ltd.
// This program is free software and you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// **bf16 的 dtype 分界**：reg 收（`isSupportedTransposeElementType` 的家族分支）、mem 不收
// （`TransDataTo5HD` 通路没有 bf16 形态）。950 上与 f16 同路（`afir-transpose-reg-base.mlir`
// 的 f32/f16 主格），910B 上整图回落。
//
// **为什么单独一个文件、且 GM 入参没有 load**：搬运族（`afir.load`/`afir.store`）的 ODS
// 元素类型表不含 bf16（`Data.td` 的 `AFIR_DataMoveElementTypes`）——bf16 进不来 GM 搬运，
// 真实图靠 cast 换进来（f16/f32 GM → cast → bf16 计算）。cast 链路不属于本需求的测试面，
// 这里 GM 入参**声明但不消费**（kernel ABI 只对 arg_descs 与参数对账，不要求消费），钉的
// 是 transpose 自身的 dtype 能力面。`afir-transpose-unsupported.mlir` 里那格是同一分界的
// 简单形态（910B 单侧拒）；本文件给全式图的正反两halves。

// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s \
// RUN:   | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --target=Ascend950DT_950x --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation --convert-afir-to-ascendc %s \
// RUN:   | FileCheck %s --check-prefix=IR
// RUN: not af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-select-implementation %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE
// WRONGSIDE: no supported

// 与 `afir-transpose-reg-base.mlir` 同一张 3D 图（perm `[0, 2, 1]`），只换元素类型。
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

func.func private @kept_transpose_tile_bf16(
    %input: memref<1024x16x8xbf16> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                                   afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"262144">,
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
      : memref<?x?x?x16x8xbf16>
  %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>}
      : memref<?x?x?x8x16xbf16>
  %tmp = memref.alloc(%tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 0,
                                           size_from_resource = true, life_time_axis_id = -1>}
      : memref<?xui8>
  %result = afir.transpose %src outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#src_tile],
    afir.destination_descs = [#dst_tile],
    afir.result_descs = [#dst_tile],
    afir.sched = #afir.sched<axis = [4, 5, 3, 1, 2], loop_axis = 5>,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>]
  } : memref<?x?x?x16x8xbf16>, memref<?x?x?x8x16xbf16>, memref<?xui8> -> memref<?x?x?x8x16xbf16>
  return
}

// SELECT-LABEL: func.func private @kept_transpose_tile_bf16
// SELECT: ascend.implementation = {id = "transpose.ub_ub_reg_base", kind = "ascendc_api"}

// 与 f32 主格同构：三段、inner 3 / total 3，类型换成 bf16 的 local tensor。
// IR-LABEL: func.func private @kept_transpose_tile_bf16
// IR: ascendc.transpose_extend_reg_base
// IR-SAME: innerDim = 3 : i8
// IR-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 3, 3, 3>
// IR-SAME: totalDim = 3 : i8
// IR-SAME: !ascendc.local_tensor<?x?x?x8x16xbf16>
// IR-NOT: ascendc.confusion_transpose
