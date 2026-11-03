// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// load 直连 store：数据不经矢量计算直接搬出（MTE2 → MTE3），物化为双端队列
// TQueBind<VECIN, VECOUT>，由队列隐式同步，不加显式事件。

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync %s | FileCheck %s
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=FINAL

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
        slot_count = 2, queue_depth = 2, size_bytes_expr = <"(2 * s0)">,
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

// 双端队列：声明类型从单端 TQue 换成 TQueBind<VECIN, VECOUT>，成员操作不变。
// CHECK: ascendc.que_bind : <vecin, vecout, 1>
// CHECK: afir.load
// CHECK: ascendc.pipe_barrier pipe_mte2
// CHECK: ascendc.que_bind.enque_tensor %{{.*}}, %{{.*}} : !ascendc.que_bind<vecin, vecout, 1>, !ascendc.local_tensor<?xui8>
// CHECK: %[[DEQUEUED:.*]] = ascendc.que_bind.deque_tensor %{{.*}} : !ascendc.que_bind<vecin, vecout, 1>, !ascendc.local_tensor<?xui8>
// CHECK: %[[TYPED:.*]] = ascendc.reinterpret_cast %[[DEQUEUED]]
// CHECK: afir.store %[[TYPED]]
// CHECK: ascendc.pipe_barrier pipe_mte3
// CHECK: ascendc.que_bind.free_tensor %{{.*}}, %{{.*}} : !ascendc.que_bind<vecin, vecout, 1>, !ascendc.local_tensor<?xui8>
// 队列已含同步，不得叠加显式事件。
// CHECK-NOT: afir.event_record
// CHECK-NOT: afir.event_wait

// FINAL-LABEL: func.func private @load_store
// FINAL: ascendc.que_bind : <vecin, vecout, 1>
// FINAL: ascendc.data_copy_pad_extend
// FINAL: ascendc.data_copy_pad_extend
// FINAL-NOT: afir.load
// FINAL-NOT: afir.store
// FINAL-NOT: ascendc.set_flag
// FINAL-NOT: ascendc.wait_flag
