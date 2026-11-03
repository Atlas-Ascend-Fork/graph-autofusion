// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

// workspace 偏移链路的 body 一侧，外加「跨 stage 全核同步」这条判据。
//
// 形状照着 RCore 铺：两个 stage 共用 `buffer_id = 2` 这块 workspace，stage 0 写、stage 1 读。
// `%ws` 的基址不是某个形参，而是「workspace 形参 + 本块的字节偏移」；偏移由 entry 从 tiling
// 读出来后经 `role = workspace_offset` 形参传进来。
//
// stage 0 里留了一个普通 GM 入参 `%input` 作对照：它不是 workspace，仍旧直接绑形参，不绕
// `ptr_offset`。
module attributes {
  afir.workspace_tensor_ids = array<i64: 2>,
  afplan.stages = #afplan.stages<template_counts = [1, 1]>
} {
  func.func private @stage0(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 0, template = 0>
  } {
    %ws = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>} : memref<16xf16>
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    afir.store %loaded outs(%ws) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }

  func.func private @stage1(%output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>,
    afplan.variant = #afplan.variant<stage = 1, template = 0>
  } {
    %ws = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <gm>, buffer_id = 2, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>} : memref<16xf16>
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %ws outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    afir.store %loaded outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// 偏移形参夹在 workspace 与 tiling 之间，`index` 就是这块 workspace 的 buffer id。
// stage 0 不是最后一个 stage 且绑了 workspace，因此要在它跑完之后全核同步——判据与 legacy
// 的 `has_workspace_node && j != schedule_groups.size() - 1` 同源。
// CHECK-LABEL: func.func private @stage0
// CHECK-SAME: %[[INPUT:[A-Za-z0-9_]+]]: memref<16xf16, 22 : i32>
// CHECK-SAME: %[[WS:[A-Za-z0-9_]+]]: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}
// CHECK-SAME: %[[OFF:[A-Za-z0-9_]+]]: ui64 {afir.kernel_arg = #afir.kernel_arg<role = workspace_offset, index = 2>}
// CHECK-SAME: {afir.kernel_arg = #afir.kernel_arg<role = tiling>}
// CHECK-SAME: ascendc.need_sync_all

// 普通 GM 入参：直接绑形参。
// CHECK: ascendc.global_tensor.set_global_buffer %{{[0-9]+}}, %[[INPUT]]

// workspace：先按字节偏移挪，再按元素类型解释回去，最后才绑。
//
// 结果类型的地址空间钉成 `22 : i32`（AscendC 的 `AddressSpace`）而不是 `#afir.space<gm>`：
// 这条 memref 要被发成 C++ 变量声明，`CodeEmitter::emitBaseMemRefType` 按 `IntegerAttr`
// 取 memory space，给它方言属性会在发射期直接 assert 掉（MLIR 各 pass 全绿也查不出来）。
// CHECK: %[[IDX:[0-9]+]] = emitc.cast %[[OFF]] : ui64 to index
// CHECK: %[[BASE:[0-9]+]] = emitasc.ptr_offset %[[WS]][%[[IDX]]] : memref<ui8, 22 : i32>, memref<ui8, 22 : i32>
// CHECK: %[[TYPED:[0-9]+]] = emitasc.reinterpret_cast %[[BASE]] : memref<ui8, 22 : i32> to memref<16xf16, 22 : i32>
// CHECK: ascendc.global_tensor.set_global_buffer %{{[0-9]+}}, %[[TYPED]]

// stage 1 同样绑了 workspace，但它是最后一个 stage——后面没有读方，不发同步。
// CHECK-LABEL: func.func private @stage1
// CHECK-NOT: ascendc.need_sync_all
// CHECK: emitasc.ptr_offset
