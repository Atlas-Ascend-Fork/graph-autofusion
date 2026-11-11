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

// Mul / Maximum / Minimum 的 tensor-tensor 形态在两类 API 家族中均使用同一无 scratch 调用，
// 因此两目标共用 IR 和检查。D1 分类来自共用扩展头的标量变体，本文件不覆盖该变体。
// E3——这份文件不涉及。
//
// Max / Min 的 APIName 不带命名空间，AscendC:: 由 emitter 添加。
// 检查 C++ 输出只含一个前缀，避免生成 AscendC::AscendC::Max。

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
