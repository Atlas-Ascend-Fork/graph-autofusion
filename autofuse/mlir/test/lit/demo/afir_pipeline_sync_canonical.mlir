// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-kernel-body-signature --afir-loop-generate --afir-memory-allocate --afir-pipeline-sync %s | FileCheck %s
// RUN: af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=FINAL

module {
  func.func private @load_add_store(%input: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}, %output: memref<16xf16> {afir.buffer_desc = #afir.buffer_desc<space = <gm>, size_bytes_expr = <"(2 * s0)">, life_time_axis_id = -1>}) attributes {
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
    return
  }
}

// CHECK: ascendc.queue : <vecin, 2>
// CHECK: ascendc.queue : <vecout, 2>
// CHECK: afir.load
// CHECK: ascendc.que_bind.enque_tensor %{{.*}}, %{{.*}} : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<?xui8>
// CHECK: %[[DEQUEUED:.*]] = ascendc.que_bind.deque_tensor %{{.*}} : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<?xui8>
// 取块偏移是 0，subindex 是恒等操作，被 LocalTensorSubIndexOp::fold 折掉。
// CHECK: %[[DEQUEUED_TYPED:.*]] = ascendc.reinterpret_cast %[[DEQUEUED]]
// CHECK-NOT: ascendc.que_bind.deque_tensor_in_place
// CHECK: afir.add %[[DEQUEUED_TYPED]], %[[DEQUEUED_TYPED]]
// CHECK: ascendc.pipe_barrier pipe_v
// CHECK: ascendc.que_bind.free_tensor %{{.*}} : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<?xui8>
// CHECK: ascendc.que_bind.enque_tensor %{{.*}}, %{{.*}} : !ascendc.queue<vecout, 2>, !ascendc.local_tensor<?xui8>
// CHECK: %[[STORE_DEQUEUED:.*]] = ascendc.que_bind.deque_tensor %{{.*}} : !ascendc.queue<vecout, 2>, !ascendc.local_tensor<?xui8>
// CHECK: %[[STORE_TYPED:.*]] = ascendc.reinterpret_cast %[[STORE_DEQUEUED]]
// CHECK: afir.store %[[STORE_TYPED]]
// CHECK: ascendc.pipe_barrier pipe_mte3
// CHECK: ascendc.que_bind.free_tensor %{{.*}} : !ascendc.queue<vecout, 2>, !ascendc.local_tensor<?xui8>

// FINAL-LABEL: func.func private @load_add_store
// FINAL-SAME: (%{{[A-Za-z0-9_]+}}: memref<16xf16, 22 : i32>
// FINAL-SAME: %{{[A-Za-z0-9_]+}}: memref<16xf16, 22 : i32>
// FINAL-NOT: afir.load
// FINAL-NOT: afir.add
// FINAL-NOT: afir.store
// Load 与 Store 都下降成 `data_copy_pad_extend`（`data_copy_l2` 那个 op 还在方言里、
// `ascendc_vector_target.mlir` 仍在钉它，只是这条路不再产出它）：裸 `AscendC::DataCopy`
// 要求 `count * sizeof(T)` 32B 对齐，而 tile 的元素个数是运行期值，尾块对不齐时 NPU 上
// 不报错、按 32B 向下取整。
// FINAL: ascendc.data_copy_pad_extend
// FINAL: ascendc.add_l2
// FINAL: ascendc.data_copy_pad_extend
