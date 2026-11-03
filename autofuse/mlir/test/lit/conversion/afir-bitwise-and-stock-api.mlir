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

// afir-bitwise-and-extension-api.mlir 的另一半：同一个 `afir.bitwise_and`，在有标准
// `AscendC::BitwiseAnd` 的那张供给表上。没有 tmp buffer，调用带 `AscendC::` 限定，也不
// 引任何扩展头——它声明在 CANN 的 `adv_api/math/bitwise_and.h`，与 `IsNan` 一族走同一条
// `kernel_operator.h` 链进来。
//
// 这里用 ui8 而不是上一份的 i16：位模式这一族的元素类型集合把无符号也算进来，而算术那
// 几个（add / mul / max / min）不认 ui8。两份用例合起来把这条差别的两端都走到。

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
