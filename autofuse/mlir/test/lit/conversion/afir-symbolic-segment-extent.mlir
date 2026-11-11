// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 段内轴的 extent 是复合符号表达式时，conversion 阶段物化得出来。
//
// 与 `afir-broadcast-two-segment.mlir` 是同一张图，只有一处不同：z1 的长度从字面量 `8`
// 换成 `(2 * z1_size)`。这一处足以让 Load 的搬运长度、Broadcast 的 `Z`、Store 的每趟前进量
// 全部落到符号表达式上。
//
// 钉的是一条基础设施边界：descriptor 里非字面量的 stride / extent 由
// `afir::materializeSizeAt` 物化，它生成的是 `afir.symbolic_eval`。进 conversion 之前这个 op
// 已经被 `AFIRMemoryAllocatePass` 清空（它在末尾跑同一组下降 pattern），所以输入里没有；
// 但 conversion 的 pattern 自己会建。少了那一份 pattern，新建的 op 没人 legalize，
// dialect conversion 会把整个 pattern 回滚，报的是 `failed to legalize operation
// 'afir.load'` ——看起来像判据写错了，实际是新建的 op 落不了地。这个用例实测过：
// 把 `ConvertAFIRToAscendCPass` 里那一行 `populateSymbolicLoweringPatterns` 注释掉就变红。
//
// 字面量走常量、不走这条路，所以别把这里的 `(2 * z1_size)` 改回字面量——改回去这个用例
// 就什么都不钉了。

// RUN: af-opt --afir-plan-candidate-lowering-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR

#gm_in = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"8">], strides = [#afir.expr<"1">], role = <input>>
#gm_out = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1024">], strides = [#afir.expr<"1">], role = <output>>
// 两根向量化轴：zt 是广播轴（repeat 1 / stride 0），z1 不是（repeat 8 / stride 1）。
#src_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"1">, #afir.expr<"(2 * z1_size)">],
                              strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst_tile = #afir.tensor_desc<axis = [3, 4, 2, 5],
                              repeats = [#afir.expr<"(128 / (zTb_size * zt_size))">, #afir.expr<"zTb_size">,
                                         #afir.expr<"zt_size">, #afir.expr<"(2 * z1_size)">],
                              strides = [#afir.expr<"(2 * z1_size * zTb_size * zt_size)">, #afir.expr<"(2 * z1_size * zt_size)">,
                                         #afir.expr<"(2 * z1_size)">, #afir.expr<"1">],
                              vectorized_axis = [2, 5],
                              vectorized_strides = [#afir.expr<"(2 * z1_size)">, #afir.expr<"1">], role = <compute>>

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
              <id = 5, name = "z1", axis_type = Original, size = #afir.expr<"(2 * z1_size)">, split_pair_other_id = 5>],
      type = Compute,
      size_var = [#afir.expr<"zt_size">, #afir.expr<"zTb_size">, #afir.expr<"z1_size">]>,
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

// IR-LABEL: func.func private @stage0_template0
// 符号表达式落成了 index 运算，没有 afir.symbolic_eval 残留。
// IR-NOT: afir.symbolic_eval

// Load 的搬运长度是 `2 * z1_size`，不是字面量。
// IR: scf.for
// IR: %[[LOAD_LEN:.*]] = index.mul
// IR: ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %{{.*}}, %[[LOAD_LEN]], %

// Broadcast 的 Z（第四与第七实参）同样是它。
// IR: %[[BC_Z:.*]] = index.mul
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %{{.*}}, %[[BC_Z]], %{{.*}}, %{{.*}}, %[[BC_Z]], %{{.*}}, %

// Store 的每趟前进量与搬运长度也都带着它。
// IR: %[[STORE_Z:.*]] = index.mul
// IR: ascendc.data_copy_pad_extend %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %

// IR-NOT: afir.symbolic_eval
