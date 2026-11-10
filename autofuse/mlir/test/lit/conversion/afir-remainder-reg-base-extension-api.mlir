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

// afir-remainder-extension-api.mlir 的另一半：`remainder_reg_base.h`，同样收一块 tmp
// buffer（尺寸 0，拿来挑 5 参重载的，见 OpVecBinary.td）。
//
// f16 只有这一侧收：mem 侧 legacy 的 `REG_ASC_IR(Remainder)` 只给了 f32 / i32。
// 所以整个 module 拿 f16 写，`WRONGTARGET` 那条同时钉住两件事——mem 侧选不出实现。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @remainder_reg_base_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                                    %destination: !ascendc.local_tensor<?xf16>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.remainder %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @remainder_reg_base_f16
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.remainder_extend_reg_base %arg2, %arg0, %arg1, %arg3, %[[COUNT]]
// IR-NOT: afir.remainder

// 与 mem 侧逐字相同的一行——两侧本来就是同一个函数名、同样的实参个数。
// CPP-LABEL: void remainder_reg_base_f16(
// CPP: {{^ *}}RemainderExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "remainder.tensor.extend_reg_base", kind = "ascendc_api"}

// mem 侧不收 f16。
// WRONGTARGET: no supported
