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

// 与 Abs 相同，Reciprocal 的 mem-based 包装需要 tmp buffer，标准调用不需要。
// 此处检查包装注册，并拒绝将带 scratch 的 IR 用于 reg-base 标准实现。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @reciprocal_extend_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                                   %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.reciprocal %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @reciprocal_extend_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.reciprocal_extend %arg1, %arg0, %arg2, %[[COUNT]]
// IR-NOT: afir.reciprocal

// CPP-LABEL: void reciprocal_extend_f32(
// CPP: {{^ *}}ReciprocalExtend<float>(
// CPP-NOT: AscendC::Reciprocal

// SELECT: ascend.implementation = {id = "reciprocal.tensor.extend", kind = "ascendc_api"}

// WRONGTARGET: no supported
