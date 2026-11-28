// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --split-input-file --afir-vector-func-to-reg-base %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --split-input-file --afir-vector-func-to-reg-base --cse %s | ascir-translate --split-input-file -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_add_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.add %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_add_lhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.add_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_add_lhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Add(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_add_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.add %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_add_rhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.add_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_add_rhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Add(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_mul_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.mul %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_mul_lhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.mul_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_mul_lhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Mul(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_mul_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.mul %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_mul_rhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.mul_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_mul_rhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Mul(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_minimum_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.minimum %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_minimum_lhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.min_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_minimum_lhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Min(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_minimum_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.minimum %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_minimum_rhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.min_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_minimum_rhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Min(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_maximum_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.maximum %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_maximum_lhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.max_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_maximum_lhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Max(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_maximum_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.maximum %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_maximum_rhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.max_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_maximum_rhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Max(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_sub_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
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
// IR-LABEL: func.func @vf_sub_lhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.sub_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_sub_lhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Sub(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_sub_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.sub %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_sub_rhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.sub_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_sub_rhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Sub(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_div_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.div %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_div_lhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.div_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_div_lhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Div(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_div_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.div %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_div_rhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.div_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_div_rhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Div(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_truediv_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.truediv %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_truediv_lhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.div_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_truediv_lhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Div(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_truediv_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.truediv %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_truediv_rhs
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.div_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_truediv_rhs
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Div(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [#afir.expr<"2">, #afir.expr<"1">, #afir.expr<"4">, #afir.expr<"128">], strides = [#afir.expr<"512">, #afir.expr<"0">, #afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [#afir.expr<"512">, #afir.expr<"0">, #afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [#afir.expr<"1">, #afir.expr<"3">, #afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"128">, #afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [#afir.expr<"0">, #afir.expr<"128">, #afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [#afir.expr<"2">, #afir.expr<"3">, #afir.expr<"4">, #afir.expr<"128">], strides = [#afir.expr<"1536">, #afir.expr<"512">, #afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [#afir.expr<"1536">, #afir.expr<"512">, #afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_alternating_sub(%lhs: !ascendc.local_tensor<2x1x4x128xf32>, %rhs: !ascendc.local_tensor<1x3x1x128xf32>, %dst: !ascendc.local_tensor<2x3x4x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 3>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.sub %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<2x1x4x128xf32>, !ascendc.local_tensor<1x3x1x128xf32>, !ascendc.local_tensor<2x3x4x128xf32>) -> !ascendc.local_tensor<2x3x4x128xf32>
  return
}
// IR-LABEL: func.func @vf_alternating_sub
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.sub_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_alternating_sub
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Sub(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"121">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"121">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"121">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @vf_padded_tail(%lhs: !ascendc.local_tensor<1x121xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.vector_func srcs(%lhs, %rhs) dsts(%dst) attributes {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } {
  ^bb0(%l: tensor<?xf32>, %r: tensor<?xf32>):
    %v = afir.sub %l, %r : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    afir.vector_func_yield %v : tensor<?xf32>
  } : (!ascendc.local_tensor<1x121xf32>, !ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>) -> !ascendc.local_tensor<3x128xf32>
  return
}
// IR-LABEL: func.func @vf_padded_tail
// IR: scf.for
// IR: emitasc.vec_scope
// IR: ascendc.sub_reg
// IR: ascendc.store_align_reg
// CPP-LABEL: void vf_padded_tail
// CPP: for (
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::Sub(
