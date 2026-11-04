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

// `afir.pow` 与 `afir.remainder` 同形：两侧同名、都要 tmp buffer，只有扩展头不同。
// 所以两份用例分得开的**只有 IR 里建的是哪个 op**，CPP 那条 check 两边逐字一样。
//
// 这一份钉 mem 侧（`pow.h`）。这里用 f32——mem 侧收 {i32, f16, f32}。
//
// 发出来的是裸名 `Pow(`，不是 `PowExtend(`：两侧 legacy 的 `GetApiName()` 写的就是
// `Pow`。**别把这条 check 写成 `AscendC::Pow`**——`AscendC::` 里没有这个名字，
// 标准调用叫 `Power`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @pow_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.pow %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @pow_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.pow %arg2, %arg0, %arg1, %arg3, %[[COUNT]]
// IR-NOT: ascendc.pow_reg_base
// IR-NOT: afir.pow

// CPP-LABEL: void pow_f32(
// CPP: {{^ *}}Pow({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "pow.tensor_tensor", kind = "ascendc_api"}
