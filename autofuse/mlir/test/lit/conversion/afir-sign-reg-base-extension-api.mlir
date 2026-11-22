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

// SignExtend 两侧函数名和 scratch 参数相同，但 reg 侧使用 cast.h 和 sign_reg_base.h。
// C++ 调用检查相同，IR 检查用于区分所选 AscendC op。
//
// ui8 仅由 reg 侧支持；mem 侧 legacy 只注册 f16 / f32 / i32。
// ODS 为并集，WRONGTARGET 验证 mem 侧不能选择该输入。

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

// mem 侧不收 ui8,所以同一段 IR 在那张支持表上选不出实现。
// WRONGTARGET: no supported
