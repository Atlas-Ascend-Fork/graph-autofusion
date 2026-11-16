// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s 2>&1 | FileCheck %s
// CHECK: 'afir.mul' op has no supported Ascend implementation

#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"32">], strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>

// Merely allocating an offset member does not establish a scalar producer.
func.func @mul_unwritten_subview(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %offset = arith.constant 32 : index
  %allocated = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<64xui8>
  %view = ascendc.local_tensor.subindex %allocated[%offset] : !ascendc.local_tensor<64xui8>, index, !ascendc.local_tensor<64xui8>
  %scalar = ascendc.reinterpret_cast %view : !ascendc.local_tensor<64xui8> to !ascendc.local_tensor<1xf32>
  %r = afir.mul %input, %scalar outs(%dst) {afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]} : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>
  return
}
