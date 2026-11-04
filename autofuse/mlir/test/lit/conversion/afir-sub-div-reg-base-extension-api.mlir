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

// afir-sub-div-stock-api.mlir 的另一半。两侧的差别在这里看得最清楚：
//
//   - mem 侧发标准的 `AscendC::Sub`，却挂着一块用不上的 tmp buffer；
//   - reg 侧发 fork 自己的包装 `SubExtend`（`sub_reg_base.h`），**一块 tmp 都没有**。
//
// 包装的函数体就是转发给同名的 `AscendC::Sub`，所以两侧生成的机器行为一样，分得开它们的
// 是函数名带不带 `AscendC::` 限定、以及 IR 上建的是哪个 op。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @sub_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                              %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.sub %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @sub_reg_base_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                              %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.sub %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xi32> -> !ascendc.local_tensor<?xi32>
    return
  }

  func.func @div_reg_base_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                              %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.div %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @sub_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.sub_extend_reg_base %arg2, %arg0, %arg1, %[[COUNT]]
// IR-NOT: ascendc.sub_l2
// IR-NOT: afir.sub

// IR-LABEL: func.func @sub_reg_base_i32
// IR: ascendc.sub_extend_reg_base

// IR-LABEL: func.func @div_reg_base_f16
// IR: ascendc.div_extend_reg_base
// IR-NOT: afir.div

// 全局函数，不带 `AscendC::` 限定——与 mem 侧那份的 `AscendC::Sub(` 正好成对。
// CPP-LABEL: void sub_reg_base_f32(
// CPP: {{^ *}}SubExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-LABEL: void div_reg_base_f16(
// CPP: {{^ *}}DivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT-COUNT-2: ascend.implementation = {id = "sub.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "div.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
