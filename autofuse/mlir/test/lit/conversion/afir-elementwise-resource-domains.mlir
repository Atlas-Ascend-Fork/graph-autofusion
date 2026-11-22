// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: sed 's/slot_count = 2, queue_depth = 2, //g' %s > %t.buffers.mlir
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.buffers.mlir 2>&1 | FileCheck %s --check-prefix=ALIAS
// RUN: sed 's/buffer_id = 0, life_time_axis_id/buffer_id = 0, slot_count = 2, queue_depth = 2, life_time_axis_id/g' %s > %t.queues.mlir
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.queues.mlir 2>&1 | FileCheck %s --check-prefix=ALIAS
// ALIAS: 'afir.add' op has no supported Ascend implementation

// IDs are local to each resource domain. Different pitches are legal across q0/b0,
// but remain unsupported when the descriptors identify the same queue or buffer.
#queue = #afir.buffer_desc<space = <ub>, buffer_id = 0, slot_count = 2, queue_depth = 2, life_time_axis_id = -1>
#buffer = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#dense = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"8">],
  strides = [<"8">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"8">, <"1">], role = <compute>>
#pitched = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"8">],
  strides = [<"8">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"16">, <"1">], role = <compute>>

module {
  func.func @queue_to_buffer(%src: !ascendc.local_tensor<64xf32> {afir.buffer_desc = #queue},
      %rhs: !ascendc.local_tensor<64xf32>,
      %dst: !ascendc.local_tensor<64xf32> {afir.buffer_desc = #buffer}) {
    %r = afir.add %src, %rhs outs(%dst) {
      afir.operand_descs = [#pitched, #dense], afir.destination_descs = [#dense], afir.result_descs = [#dense]
    } : (!ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>), !ascendc.local_tensor<64xf32> -> !ascendc.local_tensor<64xf32>
    return
  }
  func.func @buffer_to_queue(%src: !ascendc.local_tensor<64xf32> {afir.buffer_desc = #buffer},
      %rhs: !ascendc.local_tensor<64xf32>,
      %dst: !ascendc.local_tensor<64xf32> {afir.buffer_desc = #queue}) {
    %r = afir.add %src, %rhs outs(%dst) {
      afir.operand_descs = [#pitched, #dense], afir.destination_descs = [#dense], afir.result_descs = [#dense]
    } : (!ascendc.local_tensor<64xf32>, !ascendc.local_tensor<64xf32>), !ascendc.local_tensor<64xf32> -> !ascendc.local_tensor<64xf32>
    return
  }
}
// CHECK-LABEL: func.func @queue_to_buffer
// CHECK: scf.for
// CHECK: ascendc.add_l2
// CHECK-NOT: afir.add
// CHECK-LABEL: func.func @buffer_to_queue
// CHECK: scf.for
// CHECK: ascendc.add_l2
// CHECK-NOT: afir.add
