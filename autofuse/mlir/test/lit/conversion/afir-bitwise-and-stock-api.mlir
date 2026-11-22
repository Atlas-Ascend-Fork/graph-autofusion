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

// 检查标准 AscendC::BitwiseAnd：不接收 tmp buffer，也不引入 Autofuse 扩展头。
// 声明通过 kernel_operator.h 引入的 CANN adv_api/math/bitwise_and.h 提供。
//
// 使用 ui8，与扩展用例的 i16 对照，验证位运算的无符号类型支持。
// 这与不接受 ui8 的 Add / Mul / Max / Min 类型集合不同。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @bitwise_and_stock_ui8(%lhs: !ascendc.local_tensor<?xui8>, %rhs: !ascendc.local_tensor<?xui8>,
                                   %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.bitwise_and %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>),
        !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }
}

// IR-LABEL: func.func @bitwise_and_stock_ui8
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.bitwise_and_l2 %arg2, %arg0, %arg1, %[[COUNT]]
// IR-NOT: ascendc.bitwise_and_extend
// IR-NOT: afir.bitwise_and

// CPP-LABEL: void bitwise_and_stock_ui8(
// CPP: AscendC::BitwiseAnd(
// CPP-NOT: BitwiseAndExtend

// SELECT: ascend.implementation = {id = "bitwise_and.tensor", kind = "ascendc_api"}

// mem 侧的包装要一块 tmp buffer，这段 IR 一块都没给。
// WRONGTARGET: no supported
