// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 「一块运行期尺寸的缓冲」——尺寸 SSA 化的第三步（设计文档 3.3 / ADR-11）。
//
// 前两步让运行期尺寸成为一个 SSA 值；这一步让它成为 `memref` 类型的一部分：UB 上的
// 局部张量是 `memref<?xf16>`，动态维由入口块的**尺寸序言**给出——块首那一段连续的
// `afir.tiling_param` 与 `afir.symbolic_eval`。
//
// 本用例盯住的是一个不会自己变红的形态错误。`memref.alloc` 的动态操作数在 AscendC
// 物化路径上没有读点（队列字节数来自 `size_bytes_expr`，成员尺寸来自 tensor
// descriptor），所以写错了也照样生成正确的 kernel——`bridge/AfirBuilder` 此前就在这里
// 填 `arith.constant 1`。判据因此落在结构上，而不是落在生成结果上。

// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate %s | FileCheck %s --check-prefix=AFIR
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-memory-allocate %s | FileCheck %s --check-prefix=MATERIALIZED
// 运行期尺寸的缓冲不会让 C3 退步：循环生成之后函数体里仍然只有 AFIR 与它复用的标准方言。
// RUN: af-opt --target=Ascend910B1 --afir-kernel-body-signature --afir-materialize-tiling-params \
// RUN:   --afir-loop-generate --afir-verify-kernel-complete %s | FileCheck %s --check-prefix=COMPLETE

module {
  func.func private @dynamic_tile(%input: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"(2 * s0)">], strides = [#afir.expr<"1">], role = <input>>,
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"(2 * s0)">], strides = [#afir.expr<"1">], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    // 尺寸序言：符号取值，再由它算出这块缓冲的元素个数。复合表达式走
    // `afir.symbolic_eval`，单符号的（例如直接用 `%n`）就是那一份 `afir.tiling_param`。
    %n = afir.tiling_param "s0" : index
    %m = afir.symbolic_eval %n {expr = #afir.expr<"(2 * __arg0)">} : (index) -> index
    %tmp = memref.alloc(%m) {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<?xf16>
    %loaded = afir.load %input outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"(2 * s0)">], strides = [#afir.expr<"1">], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"(2 * s0)">], strides = [#afir.expr<"1">], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"(2 * s0)">], strides = [#afir.expr<"1">], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<?xf16>, memref<?xf16> -> memref<?xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"(2 * s0)">], strides = [#afir.expr<"1">], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"(2 * s0)">], strides = [#afir.expr<"1">], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<?xf16>, memref<?xf16>
    return
  }
}

// AFIR-LABEL: func.func private @dynamic_tile
// 序言整体留在块首，不进循环树：`afir.tiling_param` 与 `afir.symbolic_eval` 都没有
// `afir.sched` 可言，位置由「谁都要用它」决定。循环生成把这两类一起豁免（见
// dialect/AFIR/Transforms/LoopGenerate.cpp 的 isSizePrologueOperation）。
// AFIR: %[[S0:.*]] = afir.tiling_param "s0"
// AFIR-NEXT: %[[EXTENT:.*]] = afir.symbolic_eval %[[S0]]
// 缓冲是动态的，而且是循环不变的：范围由 tiling 参数决定，与循环变量无关。
// AFIR-NEXT: memref.alloc(%[[EXTENT]]) {{.*}} : memref<?xf16>
// AFIR: scf.for %{{.*}} to %[[S0]]

// MATERIALIZED-LABEL: func.func private @dynamic_tile
// MATERIALIZED: %[[S0:.*]] = afir.tiling_param "s0"
// 队列字节数仍然来自 `size_bytes_expr`，与 alloc 的动态维是两件事——前者是主机侧为这个
// 资源预留的字节（含对齐与复用打包），后者是这块张量的元素个数。
// MATERIALIZED: ascendc.pipe.init_queue
// MATERIALIZED: ascendc.que_bind.alloc_tensor
// MATERIALIZED: ascendc.reinterpret_cast
// 动态 alloc 被资源物化完整接管，没有残留。
// MATERIALIZED-NOT: memref.alloc

// COMPLETE-LABEL: func.func private @dynamic_tile
// COMPLETE: memref.alloc(%{{.*}}) {{.*}} : memref<?xf16>
