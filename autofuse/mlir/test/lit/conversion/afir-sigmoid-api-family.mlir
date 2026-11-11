// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=MEM
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=REG
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=MEMCPP
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=REGCPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=MEMSELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=REGSELECT

// Sigmoid 两侧均需要 tmp buffer，但分别使用标准 Sigmoid 与 SigmoidExtend 包装。
// 同一份带 scratch 的 AFIR 应在两目标上选择不同的 AscendC op。
//
// 不能以 tmp buffer 的存在推断扩展包装；reg-base 标准调用也必须接收该缓冲。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @sigmoid_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>,
                         %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sigmoid %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }

  func.func @sigmoid_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                         %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sigmoid %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// MEM-LABEL: func.func @sigmoid_f16
// MEM: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// MEM-NEXT: ascendc.sigmoid_extend %arg1, %arg0, %arg2, %[[COUNT]]
// MEM-LABEL: func.func @sigmoid_f32
// MEM: ascendc.sigmoid_extend %arg1
// MEM-NOT: afir.sigmoid

// 标准调用同样收下了 tmp buffer——这块 UB 没有被丢掉。
// REG-LABEL: func.func @sigmoid_f16
// REG: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// REG-NEXT: ascendc.sigmoid %arg1, %arg0, %arg2, %[[COUNT]]
// REG-LABEL: func.func @sigmoid_f32
// 同上：`ascendc.sigmoid` 是 `ascendc.sigmoid_extend` 的前缀，不带操作数就挡不住。
// REG: ascendc.sigmoid %arg1
// REG-NOT: afir.sigmoid

// MEMCPP-LABEL: void sigmoid_f16(
// MEMCPP: {{^ *}}SigmoidExtend<half>(
// MEMCPP-NOT: AscendC::Sigmoid

// REGCPP-LABEL: void sigmoid_f16(
// REGCPP: AscendC::Sigmoid(
// REGCPP-NOT: SigmoidExtend

// MEMSELECT: ascend.implementation = {id = "sigmoid.tensor.extend", kind = "ascendc_api"}
// REGSELECT: ascend.implementation = {id = "sigmoid.tensor", kind = "ascendc_api"}
