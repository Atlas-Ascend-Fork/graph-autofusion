// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: sed '/%neg = afir.neg/d' %s > %t.unwritten.mlir
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.unwritten.mlir 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// UNKNOWN: 'afir.minimum' op has no supported Ascend implementation
#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"32">], strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>

func.func @minimum_computed_queue(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %seed: !ascendc.local_tensor<1xf32>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %neg = afir.neg %seed outs(%one) {afir.operand_descs = [#one], afir.destination_descs = [#one], afir.result_descs = [#one]} : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  %r = afir.minimum %input, %one outs(%dst) {afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @minimum_computed_queue
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.wait_flag
// CHECK: ascendc.local_tensor.get_value
// CHECK: ascendc.mins
func.func @maximum_computed_queue(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %seed: !ascendc.local_tensor<1xf32>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %neg = afir.neg %seed outs(%one) {afir.operand_descs = [#one], afir.destination_descs = [#one], afir.result_descs = [#one]} : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  %r = afir.maximum %input, %one outs(%dst) {afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @maximum_computed_queue
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.wait_flag
// CHECK: ascendc.local_tensor.get_value
// CHECK: ascendc.maxs
func.func @where_then(%mask: !ascendc.local_tensor<32xui8>, %input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %r = afir.where %mask, %one, %input outs(%dst) tmp(%aux) {afir.operand_descs = [#full, #one, #full], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xui8>, !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<32xf32>), !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @where_then
// CHECK: ascendc.pipe.fetch_event_id {{.*}}mte2_s
// CHECK: ascendc.local_tensor.get_value
// CHECK: ascendc.where
func.func @where_else(%mask: !ascendc.local_tensor<32xui8>, %input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %r = afir.where %mask, %input, %one outs(%dst) tmp(%aux) {afir.operand_descs = [#full, #full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xui8>, !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @where_else
// CHECK: ascendc.pipe.fetch_event_id {{.*}}mte2_s
// CHECK: ascendc.local_tensor.get_value
// CHECK: ascendc.where
func.func @where_both(%mask: !ascendc.local_tensor<32xui8>, %input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %r = afir.where %mask, %one, %one outs(%dst) tmp(%aux) {afir.operand_descs = [#full, #one, #one], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xui8>, !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @where_both
// CHECK: ascendc.pipe.fetch_event_id {{.*}}mte2_s
// CHECK: ascendc.local_tensor.get_value
// CHECK: ascendc.where
