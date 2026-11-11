// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

// afir-logical-extension-api.mlir 的另一半。两侧的差别在这里看得最清楚，而且**三样一起
// 变**——这是二元一类里第一对三样都不同的 op：
//
//   - 调用名不同：mem 侧是裸名 `LogicalAnd`，reg 侧是 `LogicalAndExtend`；
//   - 扩展头不同：`logical.h` / `logical_reg_base.h`；
//   - tmp buffer：mem 侧要且真用，reg 侧一块都没有——所以这里的函数签名里没有 tmp
//     参数，IR 上的调用是四个实参。
//
// `bitwise_and` 那一对只差前两样（它的 mem 侧 tmp 是个没人用的形参），`floor_div` /
// `remainder` 两侧同名。到这一对，三样才第一次一起错开。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @logical_and_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                                      %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_and %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @logical_and_reg_base_i16(%lhs: !ascendc.local_tensor<?xi16>, %rhs: !ascendc.local_tensor<?xi16>,
                                      %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_and %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>),
        !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @logical_or_reg_base_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                                     %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_or %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }
}

// IR-LABEL: func.func @logical_and_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.logical_and_extend_reg_base %arg2, %arg0, %arg1, %[[COUNT]]
// IR-NOT: ascendc.logical_and_extend %
// IR-NOT: afir.logical_and

// IR-LABEL: func.func @logical_and_reg_base_i16
// IR: ascendc.logical_and_extend_reg_base

// IR-LABEL: func.func @logical_or_reg_base_f16
// IR: ascendc.logical_or_extend_reg_base
// IR-NOT: afir.logical_or

// 带 `Extend` 后缀、不带 `AscendC::` 限定——与 mem 侧那份的 `LogicalAnd(` 正好成对。
// CPP-LABEL: void logical_and_reg_base_f32(
// CPP: {{^ *}}LogicalAndExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-LABEL: void logical_or_reg_base_f16(
// CPP: {{^ *}}LogicalOrExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT-COUNT-2: ascend.implementation = {id = "logical_and.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "logical_or.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
