// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

module {
  func.func private @shared_queue_vectorized_dynamic(%input0: memref<?x?x?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(4 * s0 * s1 * s2)">, life_time_axis_id = -1>}, %input1: memref<?x?x?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(4 * s0 * s1 * s2)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <input>>,
      #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <input>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [
        <id = 0, name = "s0_axis", axis_type = Original, size = #afir.expr<"s0">>,
        <id = 1, name = "s1_axis", axis_type = Original, size = #afir.expr<"s1">>,
        <id = 2, name = "s2_axis", axis_type = Original, size = #afir.expr<"s2">>
      ],
      type = Compute,
      size_var = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp0 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, reuse_id = 0,
        slot_count = 1, queue_depth = 1, size_from_resource = true, life_time_axis_id = -1>} : memref<16x32x5xf32>
    %tmp1 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, reuse_id = 0,
        slot_count = 1, queue_depth = 1, size_from_resource = true, life_time_axis_id = -1>} : memref<16x32x5xf32>
    %loaded0 = afir.load %input0 outs(%tmp0) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0, 1, 2],
        repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2],
        repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0, 1, 2],
        repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>],
      afir.sched = #afir.sched<axis = [0, 1, 2], loop_axis = 2>,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
    } : memref<?x?x?xf32>, memref<16x32x5xf32> -> memref<16x32x5xf32>
    %loaded1 = afir.load %input1 outs(%tmp1) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0, 1, 2],
        repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2],
        repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0, 1, 2],
        repeats = [#afir.expr<"s0">, #afir.expr<"s1">, #afir.expr<"s2">],
        strides = [#afir.expr<"(s1 * s2)">, #afir.expr<"s2">, #afir.expr<"1">],
        vectorized_axis = [2], vectorized_strides = [#afir.expr<"1">], role = <compute>>],
      afir.sched = #afir.sched<axis = [0, 1, 2], loop_axis = 2>,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
    } : memref<?x?x?xf32>, memref<16x32x5xf32> -> memref<16x32x5xf32>
    return
  }
}

// CHECK-LABEL: func.func private @shared_queue_vectorized_dynamic
// CHECK: arith.mului_extended
// CHECK: %[[RAW:.*]] = ascendc.que_bind.alloc_tensor
// 偏移 0 的 subindex 是恒等操作，被 LocalTensorSubIndexOp::fold 折掉；只有偏移非 0 的
// 第二个成员留下 subindex。
// CHECK: ascendc.reinterpret_cast %[[RAW]]
// CHECK: ascendc.local_tensor.subindex %[[RAW]][
