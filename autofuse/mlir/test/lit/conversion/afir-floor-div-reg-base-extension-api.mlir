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

// afir-floor-div-extension-api.mlir 的另一半：同一个包装名，`floor_div_reg_base.h`，
// 不收 tmp buffer——reg 侧在寄存器里做完 `Div` + `Truncate`，没有中间商要落地。

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

// 与 mem 侧逐字相同的一行——两侧本来就是同一个函数名。
// CPP-LABEL: void floor_div_reg_base_f32(
// CPP: {{^ *}}FloorDivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// 两个函数都选中同一个 implementation，整数那个也是——所以是 COUNT-2 而不是一条。
// SELECT-COUNT-2: ascend.implementation = {id = "floor_div.tensor.extend_reg_base", kind = "ascendc_api"}

// mem 侧的包装要一块 tmp buffer，这段 IR 一块都没给；何况 i32 那一份 mem 侧根本不收。
// WRONGTARGET: no supported
