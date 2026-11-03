// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

// 二元的 D1：Mul / Maximum / Minimum 的 tensor-tensor 形态两套家族同一个调用、都不要
// tmp buffer，所以和 afir-unary-d0-both-targets.mlir 一样是「同一份 IR、两个 --target、
// 同一组 CHECK」。对照表把它们列成 D1 是因为**标量变体**两侧共用一个扩展头，那条路在
// E3——这份文件不涉及。
//
// Maximum / Minimum 还顺带钉住一件事：对照表里它们的 API 名带着 `AscendC::` 前缀
// （E0-1 的 QA 发现）。MLIR 这边前缀由 emitter 加，op 的 APIName 是裸的 `Max` / `Min`，
// 所以 CPP 侧必须恰好是一个前缀，不能是 `AscendC::AscendC::Max`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @mul_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                     %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.mul %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }

  func.func @maximum_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                         %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.maximum %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xi32> -> !ascendc.local_tensor<?xi32>
    return
  }

  func.func @minimum_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                         %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.minimum %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @mul_f16
// IR: ascendc.mul_l2 %arg2, %arg0, %arg1
// IR-LABEL: func.func @maximum_i32
// IR: ascendc.max_l2 %arg2, %arg0, %arg1
// IR-LABEL: func.func @minimum_f32
// IR: ascendc.min_l2 %arg2, %arg0, %arg1
// IR-NOT: afir.

// CPP-LABEL: void mul_f16(
// CPP: AscendC::Mul(
// CPP-LABEL: void maximum_i32(
// CPP: {{^ *}}AscendC::Max(
// CPP-LABEL: void minimum_f32(
// CPP: {{^ *}}AscendC::Min(

// SELECT: ascend.implementation = {id = "mul.tensor_tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "maximum.tensor_tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "minimum.tensor_tensor", kind = "ascendc_api"}
