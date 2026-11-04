// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Agreement Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// **`Load → Store` 直连（`MTE2 → MTE3`），缓冲不是队列：报错，不做 barrier 回落。**
//
// 顺序只能由队列的 enque / deque 给——`PipeBarrier` 只在同一条流水内排序，MTE2 与 MTE3
// 之间没有 barrier 可用。这里的缓冲没有 `slot_count` / `queue_depth`（是 TBuf 不是队列），
// 所以报错而不是静默放过：放过去就是 Store 可能读到还没搬完的 UB。
//
// 这条交接此前整档不支持（本文件原先钉的就是那句
// `unsupported pipeline synchronization from MTE2 to MTE3`）。队列那一档现在发得出来，
// 正例见 `conversion/afir-transfer-load-store-mte2-mte3.mlir`——那一份是广播图
// `[1,8] → [128,8]` 的 `B0Y1` 模板的真实 dump（`brc` 被删之后就是这个形状）。

// RUN: not af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate \
// RUN:   --afir-pipeline-sync %s 2>&1 | FileCheck %s

module {
  func.func private @load_store(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>,
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
        size_bytes_expr = <"(2 * s0)">,
        alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%tmp) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
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


// CHECK: MTE2 to MTE3 synchronization requires a queue-backed buffer
