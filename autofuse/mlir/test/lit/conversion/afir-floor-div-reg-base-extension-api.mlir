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

// 检查 reg-based floor_div_reg_base.h 包装，不接收 tmp buffer，
// 因为 Div 和 Truncate 在寄存器中完成。整数支持仅在 reg 侧；mem 侧只注册 f16 / f32。
// ODS 为两侧并集，i32 应仅选择 reg 侧实现。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @floor_div_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                                    %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.floor_div %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  // 整数只有 reg 侧收：mem 侧 legacy 的 `REG_ASC_IR(FloorDiv)` 只注册了 f16 / f32。
  // 这一条是两个 implementation 落在不同元素类型族的现场——ODS 写的是并集，
  // 挑得动的只有 reg 这一侧。
  func.func @floor_div_reg_base_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                                    %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.floor_div %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xi32> -> !ascendc.local_tensor<?xi32>
    return
  }
}

// IR-LABEL: func.func @floor_div_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.floor_div_extend_reg_base %arg2, %arg0, %arg1, %[[COUNT]]
// IR-NOT: afir.floor_div

// IR-LABEL: func.func @floor_div_reg_base_i32
// IR: ascendc.floor_div_extend_reg_base

// 两侧包装函数名相同，差异在参数和扩展头。
// CPP-LABEL: void floor_div_reg_base_f32(
// CPP: {{^ *}}FloorDivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// 浮点和整数函数都应选择该实现，因此检查两次选择结果。
// SELECT-COUNT-2: ascend.implementation = {id = "floor_div.tensor.extend_reg_base", kind = "ascendc_api"}

// 此 IR 未提供 mem 侧必需的 tmp buffer，且 mem 侧不支持 i32，因此不能选择 mem 实现。
// WRONGTARGET: no supported
