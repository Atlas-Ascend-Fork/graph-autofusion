// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Version 2.0.
// RUN: not af-opt --mlir-print-ir-after-all --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s 2>&1 | FileCheck %s

module {
  func.func private @rollback_after_resource_creation(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp = memref.alloc() : memref<32xf16>
    %view = memref.subview %tmp[0] [16] [2] {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
      slot_count = 2, queue_depth = 2, size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>}
      : memref<32xf16> to memref<16xf16, strided<[2], offset: 0>>
    %loaded = afir.load %input outs(%view) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16, strided<[2], offset: 0>> -> memref<16xf16, strided<[2], offset: 0>>
    return
  }
}

// CHECK: UnsupportedOperation: canonical resource materialization currently requires a contiguous identity StorageView
// 回滚要回滚干净：失败时不能留下半物化的资源。判据是 IR 里有没有 `ascendc` 算子——
// 此前这里还有一行 `CHECK-NOT: afir.ascendc_materialized`，那个标记属性已随验收 8 删除，
// 留着是一条恒真的死断言（CHECK-NOT 一个不存在的名字永远通过）。
// CHECK-NOT: ascendc.pipe
// CHECK-NOT: ascendc.queue
