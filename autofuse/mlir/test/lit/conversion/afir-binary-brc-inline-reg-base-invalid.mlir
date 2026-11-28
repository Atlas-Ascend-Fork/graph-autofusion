// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --split-input-file --verify-diagnostics --afir-select-implementation --target=Ascend950PR_9599 %s

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @add_i8(%lhs: !ascendc.local_tensor<1x128xi8>, %rhs: !ascendc.local_tensor<3x128xi8>, %dst: !ascendc.local_tensor<3x128xi8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.add %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi8>, !ascendc.local_tensor<3x128xi8>), !ascendc.local_tensor<3x128xi8> -> !ascendc.local_tensor<3x128xi8>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @sub_i64(%lhs: !ascendc.local_tensor<1x128xi64>, %rhs: !ascendc.local_tensor<3x128xi64>, %dst: !ascendc.local_tensor<3x128xi64>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi64>, !ascendc.local_tensor<3x128xi64>), !ascendc.local_tensor<3x128xi64> -> !ascendc.local_tensor<3x128xi64>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @mul_ui16(%lhs: !ascendc.local_tensor<1x128xui16>, %rhs: !ascendc.local_tensor<3x128xui16>, %dst: !ascendc.local_tensor<3x128xui16>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.mul %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xui16>, !ascendc.local_tensor<3x128xui16>), !ascendc.local_tensor<3x128xui16> -> !ascendc.local_tensor<3x128xui16>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_i32(%lhs: !ascendc.local_tensor<1x128xi32>, %rhs: !ascendc.local_tensor<3x128xi32>, %dst: !ascendc.local_tensor<3x128xi32>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.div %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi32>, !ascendc.local_tensor<3x128xi32>), !ascendc.local_tensor<3x128xi32> -> !ascendc.local_tensor<3x128xi32>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_bf16(%lhs: !ascendc.local_tensor<1x128xbf16>, %rhs: !ascendc.local_tensor<3x128xbf16>, %dst: !ascendc.local_tensor<3x128xbf16>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.div %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xbf16>, !ascendc.local_tensor<3x128xbf16>), !ascendc.local_tensor<3x128xbf16> -> !ascendc.local_tensor<3x128xbf16>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @truediv_i32(%lhs: !ascendc.local_tensor<1x128xi32>, %rhs: !ascendc.local_tensor<3x128xi32>, %dst: !ascendc.local_tensor<3x128xi32>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.truediv %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi32>, !ascendc.local_tensor<3x128xi32>), !ascendc.local_tensor<3x128xi32> -> !ascendc.local_tensor<3x128xi32>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"1">], strides = [#afir.expr<"8">, #afir.expr<"0">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"8">, #afir.expr<"0">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @tail_broadcast(%lhs: !ascendc.local_tensor<3x1xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x1xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"127">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"127">], strides = [#afir.expr<"127">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"127">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"127">], strides = [#afir.expr<"127">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"127">, #afir.expr<"1">], role = <compute>>
func.func @unaligned_rows(%lhs: !ascendc.local_tensor<1x127xf32>, %rhs: !ascendc.local_tensor<3x127xf32>, %dst: !ascendc.local_tensor<3x127xf32>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x127xf32>, !ascendc.local_tensor<3x127xf32>), !ascendc.local_tensor<3x127xf32> -> !ascendc.local_tensor<3x127xf32>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @nonzero_broadcast_stride(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}


// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @broadcast_alias(%rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %lhs = ascendc.reinterpret_cast %dst : !ascendc.local_tensor<3x128xf32> to !ascendc.local_tensor<1x128xf32>
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
