// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate %s | FileCheck %s

// 一条队列上先后落下的若干块，复用同一次 AllocTensor，不是各要一格。legacy 的
// reuse_from 链就是这个：`codegen_kernel_loop.cpp::AllocOutputs` 里 VecCalc 输出
// （以及 `HandleVecOutAlloc` 的 VecOut 输出）只要前一个队列成员不是 VecOut，就只生成
// 一条 `PipeBarrier<PIPE_V>`，不再 `AllocBuf`；整条队列自始至终只有一个 `q<N>_buf`。
//
// 各要一格会在运行期炸：`init_queue` 的格数是各成员 `slot_count` 的最大值（见
// afir_memory_allocate_queue_slot_count_max.mlir），三个计算节点串在一条队列上就是
// 3 > 2，AscendC 报 "size is 3, which exceed limits 2"。
//
// 这里三个 add 串成一条链，输出都落在同一条队列（buffer_id 3）的三个 reuse 组上。
// 判据：这条队列只有一次 alloc_tensor、一次 free_tensor，三个 add 的 outs 都是它的
// 视图（于是第二、三个 add 是原地算）。

module {
  func.func private @queue_reuse_chain(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %tmp0 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, reuse_id = 0,
        slot_count = 2, queue_depth = 2, size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %tmp1 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3, reuse_id = 1,
        slot_count = 1, queue_depth = 2, size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %tmp2 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3, reuse_id = 2,
        slot_count = 1, queue_depth = 2, size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %tmp3 = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3, reuse_id = 3,
        slot_count = 2, queue_depth = 2, size_bytes_expr = #afir.expr<"32">, life_time_axis_id = -1>} : memref<16xf16>
    %in0 = afir.load %input outs(%tmp0) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %sum0 = afir.add %in0, %in0 outs(%tmp1) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    %sum1 = afir.add %sum0, %in0 outs(%tmp2) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    %sum2 = afir.add %sum1, %in0 outs(%tmp3) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>, #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %sum2 outs(%input) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// CHECK-LABEL: func.func private @queue_reuse_chain
// CHECK: %[[VECIN:.*]] = ascendc.queue : <vecin, 2>
// CHECK: %[[VECOUT:.*]] = ascendc.queue : <vecout, 2>
// CHECK: scf.for
// 整条链一次 alloc：三个 add 的 outs 都从这一块上切视图。三个视图的字节偏移都是 0，
// 取块的 subindex 是恒等操作，被 LocalTensorSubIndexOp::fold 折掉，reinterpret_cast
// 直接落在 %[[RAW]] 上。
// CHECK: %[[RAW:.*]] = ascendc.que_bind.alloc_tensor %[[VECOUT]]
// CHECK: %[[T0:.*]] = ascendc.reinterpret_cast %[[RAW]]
// CHECK: afir.add %{{.*}} outs(%[[T0]])
// CHECK: %[[T1:.*]] = ascendc.reinterpret_cast %[[RAW]]
// CHECK: afir.add %[[T0]], %{{.*}} outs(%[[T1]])
// CHECK: %[[T2:.*]] = ascendc.reinterpret_cast %[[RAW]]
// CHECK: afir.add %[[T1]], %{{.*}} outs(%[[T2]])
// 归还一次，在整条链的末次使用之后。
// CHECK: afir.store %[[T2]]
// CHECK: ascendc.que_bind.free_tensor %[[VECOUT]], %[[RAW]]
// 这条队列上不再有第二次 alloc / free。
// CHECK-NOT: ascendc.que_bind.alloc_tensor %[[VECOUT]]
// CHECK-NOT: ascendc.que_bind.free_tensor %[[VECOUT]]
