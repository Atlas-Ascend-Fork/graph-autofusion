// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 同一 load 结果既供矢量计算、又供 store 直连消费。矢量侧走 MTE2→Vector 队列同步；
// store 侧插入 transfer_in→transfer_out 显式同步对，队列保持单端 VECIN。

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync %s | FileCheck %s
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=FINAL

module {
  func.func private @load_add_store_passthrough(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %passthrough: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %passthrough2: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
    afir.arg_descs = [
      #afir.tensor_desc<axis = [0], role = <input>>,
      #afir.tensor_desc<axis = [0], role = <output>>,
      #afir.tensor_desc<axis = [0], role = <output>>,
      #afir.tensor_desc<axis = [0], role = <output>>
    ],
    afir.asc_graph = #afir.asc_graph<
      axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"s0">>],
      type = Compute,
      size_var = [#afir.expr<"s0">]>,
    afir.function_role = #afir.function_role<kernel_body>
  } {
    %load_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %compute_tile = memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 3,
                                                            slot_count = 2, queue_depth = 2,
                                                            size_bytes_expr = <"(2 * s0)">,
                                                            alignment_bytes = 32, life_time_axis_id = -1>} : memref<16xf16>
    %loaded = afir.load %input outs(%load_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <input>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16> -> memref<16xf16>
    %computed = afir.add %loaded, %loaded outs(%compute_tile) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>,
                            #afir.tensor_desc<axis = [0], role = <compute>>],
      afir.result_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>
    } : (memref<16xf16>, memref<16xf16>), memref<16xf16> -> memref<16xf16>
    afir.store %computed outs(%output) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    afir.store %loaded outs(%passthrough) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    afir.store %loaded outs(%passthrough2) {
      afir.destination_descs = [#afir.tensor_desc<axis = [0], role = <output>>],
      afir.operand_descs = [#afir.tensor_desc<axis = [0], role = <compute>>],
      afir.sched = #afir.sched<axis = [0], loop_axis = 0>,
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]
    } : memref<16xf16>, memref<16xf16>
    return
  }
}

// 矢量侧：既有 MTE2→Vector 队列同步，队列保持单端。
// CHECK: ascendc.queue : <vecin, 2>
// CHECK: ascendc.queue : <vecout, 2>
// CHECK: afir.load
// 直连 store 侧：load 之后记录 transfer_in→transfer_out 事件。
// CHECK: %[[TOKEN:.*]] = afir.event_record : <from = transfer_in, to = transfer_out>
// CHECK: ascendc.que_bind.enque_tensor %{{.*}}, %{{.*}} : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<?xui8>
// CHECK: %[[DEQUEUED:.*]] = ascendc.que_bind.deque_tensor %{{.*}} : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<?xui8>
// CHECK: %[[TYPED:.*]] = ascendc.reinterpret_cast %[[DEQUEUED]]
// CHECK: afir.add %[[TYPED]], %[[TYPED]]
// 计算结果走既有 VECOUT 队列。
// CHECK: afir.store %{{.*}}
// CHECK: ascendc.pipe_barrier pipe_mte3
// CHECK: ascendc.que_bind.free_tensor %{{.*}} : !ascendc.queue<vecout, 2>, !ascendc.local_tensor<?xui8>
// 直连 store：等待事件后读取原始视图，不经队列出队。
// CHECK: afir.event_wait %[[TOKEN]] : <from = transfer_in, to = transfer_out>
// CHECK: afir.store %{{.*}}
// 第二个直连 store：MTE3 同 pipe 程序序被首 store 的 wait 覆盖，不再有自己的事件对。
// CHECK-NOT: afir.event_
// CHECK: afir.store %{{.*}}
// 最后一个直连 Store 完成后，才能让下一轮 MTE2 复用输入 slot。
// CHECK-NEXT: %[[RELEASE:.*]] = afir.event_record : <from = transfer_out, to = transfer_in>
// CHECK-NEXT: afir.event_wait %[[RELEASE]] : <from = transfer_out, to = transfer_in>
// 释放屏障覆盖矢量与搬出两类读者。
// CHECK: ascendc.pipe_barrier pipe_mte3
// CHECK: ascendc.pipe_barrier pipe_v
// CHECK: ascendc.que_bind.free_tensor %{{.*}} : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<?xui8>

// FINAL-LABEL: func.func private @load_add_store_passthrough
// FINAL: ascendc.data_copy_pad_extend
// FINAL: %{{.*}} = ascendc.pipe.alloc_event_id %{{.*}}, mte2_mte3 : i8
// FINAL: ascendc.set_flag mte2_mte3, %{{.*}} : i8
// FINAL: ascendc.add_l2
// FINAL: ascendc.data_copy_pad_extend
// FINAL: ascendc.wait_flag mte2_mte3, %{{.*}} : i8
// FINAL: ascendc.pipe.release_event_id
// FINAL: ascendc.data_copy_pad_extend
// FINAL: ascendc.data_copy_pad_extend
// FINAL: %[[RELEASE_ID:.*]] = ascendc.pipe.alloc_event_id %{{.*}}, mte3_mte2 : i8
// FINAL-NEXT: ascendc.set_flag mte3_mte2, %[[RELEASE_ID]] : i8
// FINAL-NEXT: ascendc.wait_flag mte3_mte2, %[[RELEASE_ID]] : i8
// FINAL-NEXT: ascendc.pipe.release_event_id %{{.*}}, %[[RELEASE_ID]] {event = {{[0-9]+}} : i32} : !ascendc.pipe, i8
// FINAL: ascendc.que_bind.free_tensor %{{.*}} : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<?xui8>
