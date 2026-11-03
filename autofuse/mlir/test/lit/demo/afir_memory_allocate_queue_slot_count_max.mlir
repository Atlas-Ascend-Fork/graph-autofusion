// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms of
// the CANN Open Software License Version 2.0.
// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

// 一条队列上的两个张量各说自己要几格（`slot_count`），队列本身的格数是这些数的
// **最大值**——legacy 也是这么聚合的（`codegen_kernel.cpp` 对 `que_buf_num_value` 取
// max）。两个计算节点串联时这种情况就会出现：中间张量只要 1 格，搬入那块要 2 格。
//
// 此前这里要求两者相等，合法的调度会被判成 "conflicting descriptors"。判据落在
// `init_queue` 的格数上：取错成 min 的话它会变成 1，而 CPU 侧数值断言看不出区别。

module {
  func.func private @queue_slot_count_max(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    // 同一条队列（buffer_id 2）上的两块搬入，一块要 1 格、一块要 2 格。
    %tmp0 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 1, queue_depth = 2,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %tmp1 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, slot_count = 2, queue_depth = 2,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    // 计算结果落在普通 UB buffer 上，免得同一条队列既是搬入又是搬出。
    %tmp2 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
        size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %in0 = afir.load %input outs(%tmp0) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %in1 = afir.load %input outs(%tmp1) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %sum = afir.add %in0, %in1 outs(%tmp2) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %sum outs(%input) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// CHECK: %[[SLOTS:.*]] = arith.constant 2 : i32
// CHECK: ascendc.pipe.init_queue %{{.*}}, %{{.*}}, %[[SLOTS]]
