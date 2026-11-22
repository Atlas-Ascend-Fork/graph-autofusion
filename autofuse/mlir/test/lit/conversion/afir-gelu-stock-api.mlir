// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

// `afir.gelu`：两张支持表给的是**同一个** `AscendC::Gelu`，也不引扩展头，所以只有一个
// implementation、家族恒为 `Any`——两个 target 跑出来的 IR 与选中的 id 都一样，上面那两对
// RUN 共用同一组 check 就是在钉这件事。
//
// 但它要一块 tmp buffer，与 exp / ln / relu / sqrt 那一档不同。形状上它是 `sigmoid`
// 的 mem 侧那一半，只是不需要第二个 implementation。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @gelu_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.gelu %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @gelu_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.gelu %arg1, %arg0, %arg2, %[[COUNT]]
// IR-NOT: afir.gelu

// 标准调用，带 `AscendC::` 限定；tmp buffer 在 count 之前。
// CPP-LABEL: void gelu_f32(
// CPP: {{^ *}}AscendC::Gelu({{[^,()]*}}, {{[^,()]*}}, {{[^,()]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "gelu.tensor", kind = "ascendc_api"}
