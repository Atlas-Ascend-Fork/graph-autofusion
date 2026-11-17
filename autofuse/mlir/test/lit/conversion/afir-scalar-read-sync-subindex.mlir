// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-scalar-read-sync %s | FileCheck %s --check-prefix=ATTR

#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"32">], strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>

func.func @mul_dequeued_scalar_subview(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %seed: !ascendc.local_tensor<1xf32>, %guard: i1) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %offset = arith.constant 32 : index
  %dequeued = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<64xui8>
  %read_view = ascendc.local_tensor.subindex %dequeued[%offset] : !ascendc.local_tensor<64xui8>, index, !ascendc.local_tensor<64xui8>
  %read_scalar = ascendc.reinterpret_cast %read_view : !ascendc.local_tensor<64xui8> to !ascendc.local_tensor<1xf32>
  scf.if %guard {
    %r = afir.mul %input, %read_scalar outs(%dst) {afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>
  }
  return
}
// IR-LABEL: func.func @mul_dequeued_scalar_subview
// IR: ascendc.pipe.fetch_event_id {{.*}}mte2_s
// IR: ascendc.wait_flag
// IR: ascendc.local_tensor.get_value
// IR: ascendc.muls
// ATTR-LABEL: func.func @mul_dequeued_scalar_subview
// ATTR: afir.mul
// ATTR-SAME: afir.scalar_read_sync = {{\[}}[], [#afir<pipe transfer_in>]]

func.func @mul_computed_scalar_subview(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %seed: !ascendc.local_tensor<1xf32>, %guard: i1) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %offset = arith.constant 32 : index
  %allocated = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<64xui8>
  %write_view = ascendc.local_tensor.subindex %allocated[%offset] : !ascendc.local_tensor<64xui8>, index, !ascendc.local_tensor<64xui8>
  %write_scalar = ascendc.reinterpret_cast %write_view : !ascendc.local_tensor<64xui8> to !ascendc.local_tensor<1xf32>
  scf.if %guard {
    %neg = afir.neg %seed outs(%write_scalar) {afir.operand_descs = [#one], afir.destination_descs = [#one], afir.result_descs = [#one]} : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  }
  ascendc.que_bind.enque_tensor %queue, %allocated : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<64xui8>
  %dequeued = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<64xui8>
  %read_view = ascendc.local_tensor.subindex %dequeued[%offset] : !ascendc.local_tensor<64xui8>, index, !ascendc.local_tensor<64xui8>
  %read_scalar = ascendc.reinterpret_cast %read_view : !ascendc.local_tensor<64xui8> to !ascendc.local_tensor<1xf32>
  scf.if %guard {
    %r = afir.mul %input, %read_scalar outs(%dst) {afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>
  }
  return
}
// IR-LABEL: func.func @mul_computed_scalar_subview
// IR: ascendc.pipe.fetch_event_id {{.*}}v_s
// IR: ascendc.wait_flag
// IR: ascendc.local_tensor.get_value
// IR: ascendc.muls
// ATTR-LABEL: func.func @mul_computed_scalar_subview
// ATTR: afir.mul
// ATTR-SAME: afir.scalar_read_sync = {{\[}}[], [#afir<pipe vector>]]
