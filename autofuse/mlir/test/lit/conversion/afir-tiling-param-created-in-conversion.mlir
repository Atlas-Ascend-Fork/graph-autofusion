// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// conversion 阶段新建的 `afir.tiling_param` 也要完成 lowering。
//
// 尺寸序言那一步（`--afir-materialize-tiling-params`，流水线阶段 A）不是后面几步的前置：
// `getOrCreateTilingParam` 是 get-or-create，缺的时候消费者自己建。本用例就是不跑那一步的
// 那条 RUN 行——它钉住两件事，少任何一件都会静默出错而不是报错：
//
//   1. 新建那一条要用调用方的 `builder`（`ConversionPatternRewriter`）。自己开一个
//      `OpBuilder` 建出来的 Op 不进 dialect conversion 的建档，`applyFullConversion` 连它
//      非法都不知道，会把 `afir.tiling_param` 原样留在下降完的函数里，一条诊断都没有；
//   2. `LowerTilingParamOp` 要收没有 `ascend.implementation` 属性的那一条。实现选择
//      pass 跑在 conversion 之前，新建的这条拿不到那个属性；不收的话它没人 legalize，
//      整个 pattern 回滚、报在 `afir.store` 上，看起来像搬运的判据写错了。
//
// 这张图里 `step_size` 只出现在 `#dst_tile` 循环轴那一项的 stride 上（「每趟前进多少」），
// 不在任何 `afir.buffer_desc` 里。阶段 A 不跑时，第一个碰到它的就是 `afir.store` 的下降，
// 于是那一条由 conversion 就地建出来。阶段 A 跑的时候由谁收它、以及只出现在 descriptor 里
// 的符号那一类，是 `afir-tiling-param-descriptor-symbol.mlir` 的事。
//
// 图取自 `afir-broadcast-two-segment.mlir`：UB 缓冲整块覆盖一个 block 的形状才会让局部
// descriptor 的循环轴 stride 是个符号表达式，单 tile 的缓冲那一项是 0。广播本身与本用例
// 无关，只有 `#dst_tile` 的那一项 stride 从 `(8 * zt_size)` 换成了 `(8 * step_size)`。

// 阶段 A 不跑：`step_size` 由 conversion 自己建，并且要下降成 `tiling.step_size`。
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature \
// RUN:   --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync --afir-select-implementation \
// RUN:   --convert-afir-to-ascendc %s | FileCheck %s --implicit-check-not=afir.tiling_param
// 阶段 A 跑：同一个名字、同一处下降结果。两条 RUN 行的差别只有「谁建的那一条」。
// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s \
// RUN:   | FileCheck %s --check-prefix=PIPELINE --implicit-check-not=afir.tiling_param

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"8">], strides = [#afir.expr<"1">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1024">], strides = [#afir.expr<"1">], role = <output>>
// 两根向量化轴：zt 是广播轴（repeat 1 / stride 0），z1 不是（repeat 8 / stride 1）。
#src_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"8">],
                              strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                         #afir.expr<"zt_size">, #afir.expr<"8">],
                              strides = [#afir.expr<"(8 * zTb_size * zt_size)">, #afir.expr<"(8 * step_size)">,
                                         #afir.expr<"8">, #afir.expr<"1">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module @plan0 attributes {
  afir.module_kind = "plan",
  afplan.candidate = #afplan.candidate<subgraph = 0, plan = 0>,
  afplan.stages = #afplan.stages<template_counts = [1]>
} {
  func.func private @stage0_template0(
      %input: memref<8xf32> {afir.kernel_arg = #afir.kernel_arg<role = input, index = 0>,
                             afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">,
                                                                  life_time_axis_id = -1>},
      %output: memref<1024xf32> {afir.kernel_arg = #afir.kernel_arg<role = output, index = 0>,
                                 afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"4096">,
                                                                      life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#gm_in, #gm_out],
    afir.asc_graph = #afir.asc_graph<
      tiling_key = 0,
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">, split_pair_other_id = 0>,
              <id = 1, name = "zT", axis_type = TileOuter, size = #afir.expr<"Ceiling((128 / (zt_size)))">,
               from = [0], split_pair_other_id = 2>,
              <id = 2, name = "zt", axis_type = TileInner, size = #afir.expr<"zt_size">, from = [0],
               split_pair_other_id = 1>,
              <id = 3, name = "zTB", axis_type = BlockOuter,
               size = #afir.expr<"Ceiling((Ceiling((128 / (zt_size))) / (zTb_size)))">, from = [1],
               split_pair_other_id = 4>,
              <id = 4, name = "zTb", axis_type = BlockInner, size = #afir.expr<"zTb_size">, from = [1],
               split_pair_other_id = 3>,
              <id = 5, name = "z1", axis_type = Original, size = #afir.expr<"8">, split_pair_other_id = 5>],
      type = Compute,
      size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">, #afir.expr<"step_size">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %tmp_bytes = afir.tiling_param "b0_size" : index
    %tile = afir.tiling_param "zt_size" : index
    %block = afir.tiling_param "zTb_size" : index
    %outer = afir.symbolic_eval %block, %tile {expr = #afir.expr<"(128 / (__arg0 * __arg1))">}
        : (index, index) -> index
    %src = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, reuse_id = 0,
                                                                slot_count = 1, queue_depth = 2,
                                                                size_from_resource = true,
                                                                life_time_axis_id = -1>} : memref<1x1x1x8xf32>
    %loaded = afir.load %input outs(%src) {
      afir.destination_descs = [#src_tile],
      afir.operand_descs = [#gm_in],
      afir.result_descs = [#src_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>]
    } : memref<8xf32>, memref<1x1x1x8xf32> -> memref<1x1x1x8xf32>
    %dst = memref.alloc(%outer, %block, %tile) {
      afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, reuse_id = 1, slot_count = 1,
                                           queue_depth = 2, size_from_resource = true,
                                           life_time_axis_id = -1>} : memref<?x?x?x8xf32>
    %tmp = memref.alloc(%tmp_bytes) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                          size_from_resource = true,
                                                                          alignment_bytes = 32,
                                                                          life_time_axis_id = -1>} : memref<?xui8>
    %broadcast = afir.broadcast %loaded outs(%dst) tmp(%tmp) {
      afir.destination_descs = [#dst_tile],
      afir.operand_descs = [#src_tile],
      afir.result_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>
    } : memref<1x1x1x8xf32>, memref<?x?x?x8xf32>, memref<?xui8> -> memref<?x?x?x8xf32>
    afir.store %broadcast outs(%output) {
      afir.destination_descs = [#gm_out],
      afir.operand_descs = [#dst_tile],
      afir.sched = #afir.sched<axis = [3, 4, 2, 5], loop_axis = 4>,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d4)>]
    } : memref<?x?x?x8xf32>, memref<1024xf32>
    return
  }
}

// 断的是两件事：那个名字被取到了（`afir.tiling_param` 变成 `tiling.step_size`），以及它排在
// 用到它的搬运之前——它落在入口块的尺寸序言里，消费者都在它之后。`--implicit-check-not`
// 管的是反面：一条 `afir.tiling_param` 都不许留在下降完的 IR 里。
// CHECK-LABEL: func.func private @stage0_template0
// CHECK: emitasc.member %{{.*}} "step_size"
// CHECK: ascendc.data_copy_pad_extend

// PIPELINE-LABEL: func.func private @stage0_template0
// PIPELINE: emitasc.member %{{.*}} "step_size"
// PIPELINE: ascendc.data_copy_pad_extend
