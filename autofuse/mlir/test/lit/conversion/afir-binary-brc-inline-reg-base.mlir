// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --split-input-file --afir-select-implementation --target=Ascend950PR_9599 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --split-input-file --afir-to-ascendc-pipeline --target=Ascend950PR_9599 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --split-input-file --afir-to-ascendc-pipeline --target=Ascend950PR_9599 %s --cse | ascir-translate --split-input-file -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @add_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.add %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @add_lhs
// SELECT: id = "add.broadcast_inline.reg_base"
// IR-LABEL: func.func @add_lhs
// IR: scf.for
// IR: ascendc.add_l2
// CPP-LABEL: void add_lhs
// CPP: for (
// CPP: AscendC::Add

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @add_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.add %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @add_rhs
// SELECT: id = "add.broadcast_inline.reg_base"
// IR-LABEL: func.func @add_rhs
// IR: scf.for
// IR: ascendc.add_l2
// CPP-LABEL: void add_rhs
// CPP: for (
// CPP: AscendC::Add

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @mul_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.mul %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @mul_lhs
// SELECT: id = "mul.broadcast_inline.reg_base"
// IR-LABEL: func.func @mul_lhs
// IR: scf.for
// IR: ascendc.mul_l2
// CPP-LABEL: void mul_lhs
// CPP: for (
// CPP: AscendC::Mul

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @mul_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.mul %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @mul_rhs
// SELECT: id = "mul.broadcast_inline.reg_base"
// IR-LABEL: func.func @mul_rhs
// IR: scf.for
// IR: ascendc.mul_l2
// CPP-LABEL: void mul_rhs
// CPP: for (
// CPP: AscendC::Mul

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @minimum_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.minimum %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @minimum_lhs
// SELECT: id = "minimum.broadcast_inline.reg_base"
// IR-LABEL: func.func @minimum_lhs
// IR: scf.for
// IR: ascendc.min_l2
// CPP-LABEL: void minimum_lhs
// CPP: for (
// CPP: AscendC::Min

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @minimum_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.minimum %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @minimum_rhs
// SELECT: id = "minimum.broadcast_inline.reg_base"
// IR-LABEL: func.func @minimum_rhs
// IR: scf.for
// IR: ascendc.min_l2
// CPP-LABEL: void minimum_rhs
// CPP: for (
// CPP: AscendC::Min

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @maximum_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.maximum %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @maximum_lhs
// SELECT: id = "maximum.broadcast_inline.reg_base"
// IR-LABEL: func.func @maximum_lhs
// IR: scf.for
// IR: ascendc.max_l2
// CPP-LABEL: void maximum_lhs
// CPP: for (
// CPP: AscendC::Max

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @maximum_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.maximum %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @maximum_rhs
// SELECT: id = "maximum.broadcast_inline.reg_base"
// IR-LABEL: func.func @maximum_rhs
// IR: scf.for
// IR: ascendc.max_l2
// CPP-LABEL: void maximum_rhs
// CPP: for (
// CPP: AscendC::Max

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @sub_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @sub_lhs
// SELECT: id = "sub.broadcast_inline.reg_base"
// IR-LABEL: func.func @sub_lhs
// IR: scf.for
// IR: %[[SUB_LHS_L:.*]] = ascendc.local_tensor.subindex %arg0[
// IR: %[[SUB_LHS_R:.*]] = ascendc.local_tensor.subindex %arg1[
// IR: %[[SUB_LHS_D:.*]] = ascendc.local_tensor.subindex %arg2[
// IR: ascendc.sub_extend_reg_base %[[SUB_LHS_D]], %[[SUB_LHS_L]], %[[SUB_LHS_R]],
// CPP-LABEL: void sub_lhs
// CPP: for (
// CPP: SubExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @sub_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @sub_rhs
// SELECT: id = "sub.broadcast_inline.reg_base"
// IR-LABEL: func.func @sub_rhs
// IR: scf.for
// IR: %[[SUB_RHS_L:.*]] = ascendc.local_tensor.subindex %arg0[
// IR: %[[SUB_RHS_R:.*]] = ascendc.local_tensor.subindex %arg1[
// IR: %[[SUB_RHS_D:.*]] = ascendc.local_tensor.subindex %arg2[
// IR: ascendc.sub_extend_reg_base %[[SUB_RHS_D]], %[[SUB_RHS_L]], %[[SUB_RHS_R]],
// CPP-LABEL: void sub_rhs
// CPP: for (
// CPP: SubExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.div %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @div_lhs
// SELECT: id = "div.broadcast_inline.reg_base"
// IR-LABEL: func.func @div_lhs
// IR: scf.for
// IR: %[[DIV_LHS_L:.*]] = ascendc.local_tensor.subindex %arg0[
// IR: %[[DIV_LHS_R:.*]] = ascendc.local_tensor.subindex %arg1[
// IR: %[[DIV_LHS_D:.*]] = ascendc.local_tensor.subindex %arg2[
// IR: ascendc.div_extend_reg_base %[[DIV_LHS_D]], %[[DIV_LHS_L]], %[[DIV_LHS_R]],
// CPP-LABEL: void div_lhs
// CPP: for (
// CPP: DivExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.div %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @div_rhs
// SELECT: id = "div.broadcast_inline.reg_base"
// IR-LABEL: func.func @div_rhs
// IR: scf.for
// IR: %[[DIV_RHS_L:.*]] = ascendc.local_tensor.subindex %arg0[
// IR: %[[DIV_RHS_R:.*]] = ascendc.local_tensor.subindex %arg1[
// IR: %[[DIV_RHS_D:.*]] = ascendc.local_tensor.subindex %arg2[
// IR: ascendc.div_extend_reg_base %[[DIV_RHS_D]], %[[DIV_RHS_L]], %[[DIV_RHS_R]],
// CPP-LABEL: void div_rhs
// CPP: for (
// CPP: DivExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @truediv_lhs(%lhs: !ascendc.local_tensor<1x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.truediv %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @truediv_lhs
// SELECT: id = "truediv.broadcast_inline.reg_base"
// IR-LABEL: func.func @truediv_lhs
// IR: scf.for
// IR: %[[TRUEDIV_LHS_L:.*]] = ascendc.local_tensor.subindex %arg0[
// IR: %[[TRUEDIV_LHS_R:.*]] = ascendc.local_tensor.subindex %arg1[
// IR: %[[TRUEDIV_LHS_D:.*]] = ascendc.local_tensor.subindex %arg2[
// IR: ascendc.div_extend_reg_base %[[TRUEDIV_LHS_D]], %[[TRUEDIV_LHS_L]], %[[TRUEDIV_LHS_R]],
// CPP-LABEL: void truediv_lhs
// CPP: for (
// CPP: DivExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @truediv_rhs(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<1x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.truediv %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<1x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @truediv_rhs
// SELECT: id = "truediv.broadcast_inline.reg_base"
// IR-LABEL: func.func @truediv_rhs
// IR: scf.for
// IR: %[[TRUEDIV_RHS_L:.*]] = ascendc.local_tensor.subindex %arg0[
// IR: %[[TRUEDIV_RHS_R:.*]] = ascendc.local_tensor.subindex %arg1[
// IR: %[[TRUEDIV_RHS_D:.*]] = ascendc.local_tensor.subindex %arg2[
// IR: ascendc.div_extend_reg_base %[[TRUEDIV_RHS_D]], %[[TRUEDIV_RHS_L]], %[[TRUEDIV_RHS_R]],
// CPP-LABEL: void truediv_rhs
// CPP: for (
// CPP: DivExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [#afir.expr<"2">, #afir.expr<"1">, #afir.expr<"4">, #afir.expr<"128">], strides = [#afir.expr<"512">, #afir.expr<"0">, #afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [#afir.expr<"512">, #afir.expr<"0">, #afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [#afir.expr<"1">, #afir.expr<"3">, #afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"128">, #afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [#afir.expr<"0">, #afir.expr<"128">, #afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [#afir.expr<"2">, #afir.expr<"3">, #afir.expr<"4">, #afir.expr<"128">], strides = [#afir.expr<"1536">, #afir.expr<"512">, #afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [#afir.expr<"1536">, #afir.expr<"512">, #afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @alternating_sub(%lhs: !ascendc.local_tensor<2x1x4x128xf32>, %rhs: !ascendc.local_tensor<1x3x1x128xf32>, %dst: !ascendc.local_tensor<2x3x4x128xf32>) {
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 3>
  } : (!ascendc.local_tensor<2x1x4x128xf32>, !ascendc.local_tensor<1x3x1x128xf32>), !ascendc.local_tensor<2x3x4x128xf32> -> !ascendc.local_tensor<2x3x4x128xf32>
  return
}
// SELECT-LABEL: func.func @alternating_sub
// SELECT: id = "sub.broadcast_inline.reg_base"
// IR-LABEL: func.func @alternating_sub
// IR: scf.for
// IR: scf.for
// IR: scf.for
// IR: ascendc.sub_extend_reg_base
// CPP-LABEL: void alternating_sub
// CPP: SubExtend(

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @add_ui16(%lhs: !ascendc.local_tensor<1x128xui16>, %rhs: !ascendc.local_tensor<3x128xui16>, %dst: !ascendc.local_tensor<3x128xui16>) {
  %result = afir.add %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xui16>, !ascendc.local_tensor<3x128xui16>), !ascendc.local_tensor<3x128xui16> -> !ascendc.local_tensor<3x128xui16>
  return
}
// SELECT-LABEL: func.func @add_ui16
// SELECT: id = "add.broadcast_inline.reg_base"
// IR-LABEL: func.func @add_ui16
// IR: ascendc.add_l2
// CPP-LABEL: void add_ui16
// CPP: AscendC::Add

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @mul_ui32(%lhs: !ascendc.local_tensor<1x128xui32>, %rhs: !ascendc.local_tensor<3x128xui32>, %dst: !ascendc.local_tensor<3x128xui32>) {
  %result = afir.mul %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xui32>, !ascendc.local_tensor<3x128xui32>), !ascendc.local_tensor<3x128xui32> -> !ascendc.local_tensor<3x128xui32>
  return
}
// SELECT-LABEL: func.func @mul_ui32
// SELECT: id = "mul.broadcast_inline.reg_base"
// IR-LABEL: func.func @mul_ui32
// IR: ascendc.mul_l2
// CPP-LABEL: void mul_ui32
// CPP: AscendC::Mul

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @minimum_i16(%lhs: !ascendc.local_tensor<1x128xi16>, %rhs: !ascendc.local_tensor<3x128xi16>, %dst: !ascendc.local_tensor<3x128xi16>) {
  %result = afir.minimum %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi16>, !ascendc.local_tensor<3x128xi16>), !ascendc.local_tensor<3x128xi16> -> !ascendc.local_tensor<3x128xi16>
  return
}
// SELECT-LABEL: func.func @minimum_i16
// SELECT: id = "minimum.broadcast_inline.reg_base"
// IR-LABEL: func.func @minimum_i16
// IR: ascendc.min_l2
// CPP-LABEL: void minimum_i16
// CPP: AscendC::Min

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @maximum_i32(%lhs: !ascendc.local_tensor<1x128xi32>, %rhs: !ascendc.local_tensor<3x128xi32>, %dst: !ascendc.local_tensor<3x128xi32>) {
  %result = afir.maximum %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xi32>, !ascendc.local_tensor<3x128xi32>), !ascendc.local_tensor<3x128xi32> -> !ascendc.local_tensor<3x128xi32>
  return
}
// SELECT-LABEL: func.func @maximum_i32
// SELECT: id = "maximum.broadcast_inline.reg_base"
// IR-LABEL: func.func @maximum_i32
// IR: ascendc.max_l2
// CPP-LABEL: void maximum_i32
// CPP: AscendC::Max

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @sub_bf16(%lhs: !ascendc.local_tensor<1x128xbf16>, %rhs: !ascendc.local_tensor<3x128xbf16>, %dst: !ascendc.local_tensor<3x128xbf16>) {
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xbf16>, !ascendc.local_tensor<3x128xbf16>), !ascendc.local_tensor<3x128xbf16> -> !ascendc.local_tensor<3x128xbf16>
  return
}
// SELECT-LABEL: func.func @sub_bf16
// SELECT: id = "sub.broadcast_inline.reg_base"
// IR-LABEL: func.func @sub_bf16
// IR: ascendc.sub_extend_reg_base
// CPP-LABEL: void sub_bf16
// CPP: SubExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_f16(%lhs: !ascendc.local_tensor<1x128xf16>, %rhs: !ascendc.local_tensor<3x128xf16>, %dst: !ascendc.local_tensor<3x128xf16>) {
  %result = afir.div %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xf16>, !ascendc.local_tensor<3x128xf16>), !ascendc.local_tensor<3x128xf16> -> !ascendc.local_tensor<3x128xf16>
  return
}
// SELECT-LABEL: func.func @div_f16
// SELECT: id = "div.broadcast_inline.reg_base"
// IR-LABEL: func.func @div_f16
// IR: ascendc.div_extend_reg_base
// CPP-LABEL: void div_f16
// CPP: DivExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"128">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @truediv_bf16(%lhs: !ascendc.local_tensor<1x128xbf16>, %rhs: !ascendc.local_tensor<3x128xbf16>, %dst: !ascendc.local_tensor<3x128xbf16>) {
  %result = afir.truediv %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x128xbf16>, !ascendc.local_tensor<3x128xbf16>), !ascendc.local_tensor<3x128xbf16> -> !ascendc.local_tensor<3x128xbf16>
  return
}
// SELECT-LABEL: func.func @truediv_bf16
// SELECT: id = "truediv.broadcast_inline.reg_base"
// IR-LABEL: func.func @truediv_bf16
// IR: ascendc.div_extend_reg_base
// CPP-LABEL: void truediv_bf16
// CPP: DivExtend

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @add_same(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.add %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @add_same
// SELECT: id = "add.tensor_tensor"
// IR-LABEL: func.func @add_same
// IR-NOT: scf.for
// IR: ascendc.add_l2
// CPP-LABEL: void add_same

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @div_same(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.div %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @div_same
// SELECT: id = "div.tensor_tensor.extend_reg_base"
// IR-LABEL: func.func @div_same
// IR-NOT: scf.for
// IR: ascendc.div_extend_reg_base
// CPP-LABEL: void div_same

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"128">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @truediv_same(%lhs: !ascendc.local_tensor<3x128xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.truediv %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<3x128xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @truediv_same
// SELECT: id = "truediv.tensor_tensor.extend_reg_base"
// IR-LABEL: func.func @truediv_same
// IR-NOT: scf.for
// IR: ascendc.div_extend_reg_base
// CPP-LABEL: void truediv_same

// -----

#lhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"121">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#rhs = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"121">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"121">], strides = [#afir.expr<"128">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"128">, #afir.expr<"1">], role = <compute>>
func.func @padded_tail(%lhs: !ascendc.local_tensor<1x121xf32>, %rhs: !ascendc.local_tensor<3x128xf32>, %dst: !ascendc.local_tensor<3x128xf32>) {
  %result = afir.sub %lhs, %rhs outs(%dst) {
    afir.operand_descs = [#lhs, #rhs], afir.destination_descs = [#dst], afir.result_descs = [#dst],
    afir.sched = #afir.sched<axis = [0, 1], loop_axis = 1>
  } : (!ascendc.local_tensor<1x121xf32>, !ascendc.local_tensor<3x128xf32>), !ascendc.local_tensor<3x128xf32> -> !ascendc.local_tensor<3x128xf32>
  return
}
// SELECT-LABEL: func.func @padded_tail
// SELECT: id = "sub.broadcast_inline.reg_base"
// IR-LABEL: func.func @padded_tail
// IR: scf.for
// IR: arith.constant 121 : index
// IR: ascendc.sub_extend_reg_base
// CPP-LABEL: void padded_tail
// CPP: SubExtend(
