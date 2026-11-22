// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 尺寸序言要收 tensor descriptor 里的符号。
//
// `AFIRMaterializeTilingParamsPass` 自述的是「把 kernel body 的**每一个**运行期尺寸符号提到
// 尺寸序言里」，而它此前只走 `size_var` 与 `afir.buffer_desc` 两处。descriptor 的 `repeats` /
// `strides` / `vectorized_strides` 同样会被物化成 `index` 值（`materializeExtentValue` /
// `materializeStrideValue` → `materializeSizeAt`，自由符号绑到 `getOrCreateTilingParam`），
// 漏收的后果不是报错而是换个地方建：下降时第一个碰到它的 pattern 就地建一条，那时实现
// 选择已经跑过，诊断也早该在这里一次性报完了。
//
// 本用例的符号刻意三处都不在：`stride_only` 不在 `size_var`、不在任何 `afir.buffer_desc`、
// 也没有哪条 `afir.tiling_param` 预先写在函数里。它只出现在 descriptor 上，所以这条 RUN 行
// 只有 descriptor 那一遍扫描能让它变绿。
//
// 三个字段各放一个符号：`repeat_only` 在 `repeats`、`stride_only` 在 `strides`、
// `vstride_only` 在 `vectorized_strides`。少扫哪一个字段就少一条。

// RUN: af-opt --afir-materialize-tiling-params %s | FileCheck %s

module {
  func.func private @descriptor_symbols(
      %input: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"256">,
                                                                 life_time_axis_id = -1>},
      %output: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"256">,
                                                                  life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"repeat_only">], strides = [#afir.expr<"1">],
                        role = <input>>,
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"repeat_only">], strides = [#afir.expr<"stride_only">],
                        role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"128">>],
      type = Compute>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    // 这块 tile 的动态尺寸刻意走常量：本用例要的是「函数里没有任何 `afir.tiling_param`
    // 预先写着」，尺寸序言里出现什么全部由 pass 自己收。
    %size = arith.constant 128 : index
    %tile = memref.alloc(%size) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0,
                                                                      slot_count = 1, queue_depth = 2,
                                                                      size_bytes_expr = <"256">,
                                                                      alignment_bytes = 32,
                                                                      life_time_axis_id = -1>} : memref<?xf16>
    %loaded = afir.load %input outs(%tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"repeat_only">],
                                                 strides = [#afir.expr<"1">], vectorized_axis = [0],
                                                 vectorized_strides = [#afir.expr<"vstride_only">],
                                                 role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"repeat_only">],
                                              strides = [#afir.expr<"1">], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"repeat_only">],
                                             strides = [#afir.expr<"1">], vectorized_axis = [0],
                                             vectorized_strides = [#afir.expr<"vstride_only">],
                                             role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<?xf16>, memref<?xf16> -> memref<?xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"repeat_only">],
                                                 strides = [#afir.expr<"stride_only">], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"repeat_only">],
                                              strides = [#afir.expr<"1">], vectorized_axis = [0],
                                              vectorized_strides = [#afir.expr<"vstride_only">],
                                              role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<?xf16>, memref<?xf16>
    return
  }
}

// 三个都落在尺寸序言里。顺序不断（`collectRequiredSymbols` 的收集次序不是契约）。
// CHECK-LABEL: func.func private @descriptor_symbols
// CHECK-DAG: afir.tiling_param "repeat_only"
// CHECK-DAG: afir.tiling_param "stride_only"
// CHECK-DAG: afir.tiling_param "vstride_only"
