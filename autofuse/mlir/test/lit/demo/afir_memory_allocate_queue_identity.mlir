// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 队列身份只由 descriptor 决定。
//
// `%tmp` 是 `afir.load` 的落点，但它的 descriptor 没有 `slot_count` / `queue_depth`，
// 所以它是一块普通 UB buffer：走 TBuf，尺寸字段是 `b2_size`。判据曾经还有一条
// 「被搬运族的 Op 读写就算队列」，那时同一段 IR 会得到 TQue 与 `q2_size`。
//
// 判据必须只看 descriptor，是因为它同时决定 `size_from_resource` 拼出来的字段名：
// 建 TilingData 字段的是 `collectCanonicalResourceIds`（看函数全局），取符号的是
// `isQueueBinding`（看绑定点），两边各带一套「谁在读写它」的启发就会各漂一份，漂了
// 之后 kernel 侧读的字段与 host 侧填的字段对不上。现在两者是同一个表达式。

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

module {
  func.func private @transfer_into_buffer(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
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
    %tmp = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_from_resource = true, alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
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

// TilingData 里是 `b2_size`——`AFIRKernelBodySignature` 把 buffer 2 归到了 bufferIds。
// CHECK-LABEL: func.func private @transfer_into_buffer
// CHECK-SAME: !emitasc.py_struct<"TilingData", [i64, i64, i64], ["block_dim", "s0", "b2_size"]>

// 资源侧同样：TBuf 而不是 TQue，且大小取的是同一个 `b2_size`。
// CHECK: %[[SIZE:.*]] = afir.tiling_param "b2_size"
// CHECK: %[[PIPE:.*]] = ascendc.pipe
// CHECK: %[[TBUF:.*]] = ascendc.tbuf : <veccalc>
// CHECK: ascendc.pipe.init_buffer %[[PIPE]], %[[TBUF]]
