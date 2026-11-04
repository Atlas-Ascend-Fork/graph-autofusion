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

// afir-sign-extension-api.mlir 的另一半：同一个包装名 `SignExtend`、同样收 tmp buffer，
// 只有扩展头不同（`cast.h` + `sign_reg_base.h`），所以 CPP 那条 check 与 mem 侧逐字一样，
// 分得开两侧的是 IR 上建的是哪个 op。
//
// ui8 只有 reg 侧收：mem 侧 legacy 的 `REG_ASC_IR(Sign)` 只注册了 f16 / f32 / i32。
// 这一条是两个 implementation 落在不同元素类型集合的现场——ODS 写的是并集，
// 在 910B1 上挑不出实现，由下面的 WRONGTARGET 钉住。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @sign_reg_base_ui8(%input: !ascendc.local_tensor<?xui8>, %destination: !ascendc.local_tensor<?xui8>,
                               %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sign %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }
}

// IR-LABEL: func.func @sign_reg_base_ui8
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.sign_extend_reg_base %arg1, %arg0, %arg2, %[[COUNT]]
// IR-NOT: afir.sign

// 与 mem 侧逐字相同的一行——两侧本来就是同一个函数名。
// CPP-LABEL: void sign_reg_base_ui8(
// CPP: {{^ *}}SignExtend<uint8_t>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "sign.tensor.extend_reg_base", kind = "ascendc_api"}

// mem 侧不收 ui8,所以同一段 IR 在那张供给表上选不出实现。
// WRONGTARGET: no supported
