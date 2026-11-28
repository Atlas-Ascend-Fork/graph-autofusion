// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --split-input-file --afir-vector-func-to-reg-base --verify-diagnostics %s

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"1">], strides = [#afir.expr<"8">, #afir.expr<"0">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"8">, #afir.expr<"0">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_tail_broadcast(%lhs: !ascendc.local_tensor<3x1xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  // expected-error@+1 {{UnsupportedOperation: incompatible vector function local views}}
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.sub %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x1xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"127">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"127">], strides = [#afir.expr<"127">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"127">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"127">], strides = [#afir.expr<"127">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"127">, #afir.expr<"1">], role = <compute>>
func.func @vf_unaligned_rows(%lhs: !ascendc.local_tensor<1x127xf32>, %rhs: !ascendc.local_tensor<3x127xf32>, %dst: !ascendc.local_tensor<3x127xf32>) {
  // expected-error@+1 {{UnsupportedOperation: incompatible vector function local views}}
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.sub %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x127xf32>, !ascendc.local_tensor<3x127xf32>, !ascendc.local_tensor<3x127xf32>) -> !ascendc.local_tensor<3x127xf32>
  return
}

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_nonzero_broadcast_stride(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  // expected-error@+1 {{UnsupportedOperation: incompatible vector function local views}}
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.sub %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
