// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// `afir.floor_div` 是二元里的 `afir.neg`：**两张供给表给的是同一个函数名**
// （`FloorDivExtend`），只有扩展头不同。所以这两份用例里 CPP 那条 check 逐字一样，
// 分得开两侧的是 IR 里建的是哪个 op、以及收不收 tmp buffer。
//
// 这一份钉 mem 侧：`floor_div.h`，要 tmp（装 `Div` 的中间商再交给 `Floor`）。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @floor_div_extend_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                                  %destination: !ascendc.local_tensor<?xf32>,
                                  %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.floor_div %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @floor_div_extend_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.floor_div_extend %arg2, %arg0, %arg1, %arg3, %[[COUNT]]
// IR-NOT: ascendc.floor_div_extend_reg_base
// IR-NOT: afir.floor_div

// 全局函数，不带 `AscendC::` 限定，也没有显式模板实参。
// CPP-LABEL: void floor_div_extend_f32(
// CPP: {{^ *}}FloorDivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "floor_div.tensor.extend", kind = "ascendc_api"}

// reg 侧的包装不收 tmp buffer，所以同一段 IR 在那张供给表上选不出实现。
// WRONGTARGET: no supported
