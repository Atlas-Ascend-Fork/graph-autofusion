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
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// afir-logical-not-extension-api.mlir 的另一半。**这一侧与那一侧的差别不止扩展头**：
//
//   - 名字不同：这边叫 `LogicalNotExtend`，mem 侧是裸名 `LogicalNot`；
//   - **结果 dtype 不同**：包装的形参写死 `LocalTensor<uint8_t>`，所以结果恒为 ui8，
//     而 mem 侧两端同 dtype。一元里第一个两侧关系不同的 op（二元那边是 `truediv`）；
//   - 既不要 tmp buffer，也不要那块预先填好 1 的常量——mem 侧两样都要。
//
// `WRONGTARGET` 钉的是「结果 ui8、输入 f32、又没有 tmp buffer」这份 IR 在 mem 侧选不出
// 实现：那边只有 `Same` 且要 tmp 的那一个。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @logical_not_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                      %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_not %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }

  // reg 侧独有的那一档输入：`ui16 -> ui8`。mem 侧的对表里没有这一行。
  func.func @logical_not_reg_base_ui16(%input: !ascendc.local_tensor<?xui16>,
                                       %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_not %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xui16>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }
}

// IR-LABEL: func.func @logical_not_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.logical_not_extend_reg_base %arg1, %arg0, %[[COUNT]]
// IR-NOT: ascendc.duplicate_l2
// IR-NOT: afir.logical_not

// IR-LABEL: func.func @logical_not_reg_base_ui16
// IR: ascendc.logical_not_extend_reg_base

// CPP-LABEL: void logical_not_reg_base_f32(
// CPP: LogicalNotExtend<float>({{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-NOT: Duplicate

// SELECT-COUNT-2: ascend.implementation = {id = "logical_not.tensor.extend_reg_base", kind = "ascendc_api"}

// WRONGTARGET: no supported
