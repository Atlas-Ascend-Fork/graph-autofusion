// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s

// Div supports floating-point rows only; Sub and Maximum retain integer support.
#brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_f16(%lhs: !ascendc.local_tensor<1x128xf16>, %rhs: !ascendc.local_tensor<128x128xf16>,
                         %dst: !ascendc.local_tensor<128x128xf16>, %tmp: !ascendc.local_tensor<32xui8>) {
  %result = afir.div %lhs, %rhs outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#brc, #full], afir.destination_descs = [#full], afir.result_descs = [#full],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf16>, !ascendc.local_tensor<128x128xf16>),
      !ascendc.local_tensor<128x128xf16>, !ascendc.local_tensor<32xui8> -> !ascendc.local_tensor<128x128xf16>
  return
}

// -----

#brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @sub_i32(%lhs: !ascendc.local_tensor<1x128xi32>, %rhs: !ascendc.local_tensor<128x128xi32>,
                         %dst: !ascendc.local_tensor<128x128xi32>, %tmp: !ascendc.local_tensor<32xui8>) {
  %result = afir.sub %lhs, %rhs outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#brc, #full], afir.destination_descs = [#full], afir.result_descs = [#full],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi32>, !ascendc.local_tensor<128x128xi32>),
      !ascendc.local_tensor<128x128xi32>, !ascendc.local_tensor<32xui8> -> !ascendc.local_tensor<128x128xi32>
  return
}

// -----

#brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @maximum_i32(%lhs: !ascendc.local_tensor<1x128xi32>, %rhs: !ascendc.local_tensor<128x128xi32>,
                         %dst: !ascendc.local_tensor<128x128xi32>) {
  %result = afir.maximum %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#brc, #full], afir.destination_descs = [#full], afir.result_descs = [#full],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi32>, !ascendc.local_tensor<128x128xi32>),
      !ascendc.local_tensor<128x128xi32> -> !ascendc.local_tensor<128x128xi32>
  return
}

// -----

#brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_i32(%lhs: !ascendc.local_tensor<1x128xi32>, %rhs: !ascendc.local_tensor<128x128xi32>,
                         %dst: !ascendc.local_tensor<128x128xi32>, %tmp: !ascendc.local_tensor<32xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.div %lhs, %rhs outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#brc, #full], afir.destination_descs = [#full], afir.result_descs = [#full],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi32>, !ascendc.local_tensor<128x128xi32>),
      !ascendc.local_tensor<128x128xi32>, !ascendc.local_tensor<32xui8> -> !ascendc.local_tensor<128x128xi32>
  return
}

// -----

#brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_i16(%lhs: !ascendc.local_tensor<1x128xi16>, %rhs: !ascendc.local_tensor<128x128xi16>,
                         %dst: !ascendc.local_tensor<128x128xi16>, %tmp: !ascendc.local_tensor<32xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.div %lhs, %rhs outs(%dst) tmp(%tmp) {
    afir.operand_descs = [#brc, #full], afir.destination_descs = [#full], afir.result_descs = [#full],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi16>, !ascendc.local_tensor<128x128xi16>),
      !ascendc.local_tensor<128x128xi16>, !ascendc.local_tensor<32xui8> -> !ascendc.local_tensor<128x128xi16>
  return
}

// -----

#brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_missing_tmp(%lhs: !ascendc.local_tensor<1x128xf16>, %rhs: !ascendc.local_tensor<128x128xf16>,
                         %dst: !ascendc.local_tensor<128x128xf16>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.div %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#brc, #full], afir.destination_descs = [#full], afir.result_descs = [#full],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf16>, !ascendc.local_tensor<128x128xf16>),
      !ascendc.local_tensor<128x128xf16> -> !ascendc.local_tensor<128x128xf16>
  return
}


// -----

#brc = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#full = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"128">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @sub_missing_tmp(%lhs: !ascendc.local_tensor<1x128xi32>, %rhs: !ascendc.local_tensor<128x128xi32>,
                         %dst: !ascendc.local_tensor<128x128xi32>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#brc, #full], afir.destination_descs = [#full], afir.result_descs = [#full],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi32>, !ascendc.local_tensor<128x128xi32>),
      !ascendc.local_tensor<128x128xi32> -> !ascendc.local_tensor<128x128xi32>
  return
}
