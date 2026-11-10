// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

module {
  func.func private @shared_queue_members(%input0: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">, life_time_axis_id = -1>}, %input1: memref<5xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"16">], strides = [#afir.expr<"1">],
        vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <input>>,
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"5">], strides = [#afir.expr<"1">],
        vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <input>>
    ],
    // `size_var` 里不写 `"16"`：size_var 是**符号**表，字面量不是变量。此前写在这里
    // 会让 TilingData 多出一个叫 `16` 的字段、并按名字去取它——生成的 C++ 是
    // `tiling.16`，而失败发生在离这里很远的地方。轴尺寸本来就是常量，物化时直接
    // 出常量，不需要经过 tiling 参数。
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"16">>],
      type = Compute>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp0 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, reuse_id = 0,
        slot_count = 1, queue_depth = 1, size_bytes_expr = <"64">, life_time_axis_id = -1>} : memref<16xf16>
    %tmp1 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, reuse_id = 0,
        slot_count = 1, queue_depth = 1, size_bytes_expr = <"64">, life_time_axis_id = -1>} : memref<5xf16>
    %loaded0 = afir.load %input0 outs(%tmp0) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"16">],
        strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">],
        role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"16">],
        strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">],
        role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"16">],
        strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">],
        role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %loaded1 = afir.load %input1 outs(%tmp1) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"5">],
        strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">],
        role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"5">],
        strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">],
        role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], repeats = [#afir.expr<"5">],
        strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">],
        role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<5xf16>, memref<5xf16> -> memref<5xf16>
    return
  }
}

// Members have different extents; each occupies its own 32-byte-aligned range.
// CHECK-LABEL: func.func private @shared_queue_members
// CHECK: %[[QUEUE:.*]] = ascendc.queue : <vecin, 1>
// CHECK: ascendc.pipe.init_queue %{{.*}}, %[[QUEUE]], %{{.*}}, %{{.*}} : !ascendc.queue<vecin, 1>, i32, index
// CHECK: %[[RAW:.*]] = ascendc.que_bind.alloc_tensor %[[QUEUE]]
// 第一个成员的偏移是 0，取块的 subindex 是恒等操作，被 LocalTensorSubIndexOp::fold
// 折掉，它的 reinterpret_cast 直接落在 %[[RAW]] 上；第二个成员偏移 32，subindex 留下。
// CHECK: ascendc.reinterpret_cast %[[RAW]]
// CHECK: ascendc.local_tensor.subindex %[[RAW]][%c32]
// CHECK: ascendc.reinterpret_cast {{.*}} to !ascendc.local_tensor<5xf16>
