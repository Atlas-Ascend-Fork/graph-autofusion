// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s 2>&1 | FileCheck %s

// 放宽「一条队列上不能混 dtype」之后，真正不兼容的组合仍然要拒收。
//
// 这里两个成员差的是 `queue_depth`（1 与 2）：它说的是这条队列轮转几块，是队列自己的
// 属性，两个成员各说一个数就是矛盾，取 max 或取第一个都会让物化出来的队列与其中一个
// 成员的假设不符。它在 `afir::canonicalResourceFactsAgree` 的清单里，与
// `slot_count`（取 max）、元素类型（不比）正好分在两边。
//
// 与 afir_memory_allocate_shared_queue_members.mlir 成对：除了第二个成员的
// `queue_depth`，两份逐字相同——那一份两个成员共用一条队列、这一份被拒收。
//
// 「混 dtype 现在放得过去」不在这里钉：conversion 这一侧的判据只看 descriptor，而
// descriptor 上从来没有 dtype，所以这里写一份混 dtype 的正例是假的（改动前它也绿）。
// 那一条钉在 bridge 的 UT（`SharesOneQueueAcrossDtypes...`）与 E2E
// `load_mask_select_store` 上。

module {
  func.func private @shared_queue_conflict(%input0: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">, life_time_axis_id = -1>}, %input1: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"32">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"16">], strides = [#afir.expr<"1">],
        vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <input>>,
      #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"16">], strides = [#afir.expr<"1">],
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
        slot_count = 1, queue_depth = 2, size_bytes_expr = <"64">, life_time_axis_id = -1>} : memref<16xf16>
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
    return
  }
}

// CHECK: InvalidBinding: conflicting descriptors for one canonical resource
