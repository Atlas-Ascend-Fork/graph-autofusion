// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=MEM
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=REG
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=MEMSELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=REGSELECT

// Neg 的两类 API 均为无 tmp buffer 的 NegExtend 包装，AFIR 和调用文本相同，
// 但分别来自 neg.h 与 neg_reg_base.h，并对应不同 AscendC op。
// 可用实现类别与是否使用标准 API 是独立维度，不能合并为一个 feature。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @neg_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.neg %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }

  func.func @neg_i32(%input: !ascendc.local_tensor<?xi32>, %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.neg %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32> -> !ascendc.local_tensor<?xi32>
    return
  }
}

// MEM-LABEL: func.func @neg_f16
// MEM: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// MEM-NEXT: ascendc.neg_extend %arg1, %arg0, %[[COUNT]]
// MEM-LABEL: func.func @neg_i32
// 匹配时包含后续操作数，避免 neg_extend_reg_base 的前缀误命中 neg_extend。
// MEM: ascendc.neg_extend %arg1
// MEM-NOT: afir.neg

// REG-LABEL: func.func @neg_f16
// REG: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// REG-NEXT: ascendc.neg_extend_reg_base %arg1, %arg0, %[[COUNT]]
// REG-LABEL: func.func @neg_i32
// REG: ascendc.neg_extend_reg_base %arg1
// REG-NOT: afir.neg

// 两目标共用调用文本检查：NegExtend 为全局函数，元素类型通过模板实参指定。
// CPP-LABEL: void neg_f16(
// CPP: {{^ *}}NegExtend<half>(
// CPP-LABEL: void neg_i32(
// CPP: {{^ *}}NegExtend<int32_t>(
// CPP-NOT: AscendC::Neg

// stable id 说的是支持表，不是芯片。
// MEMSELECT: ascend.implementation = {id = "neg.tensor.extend", kind = "ascendc_api"}
// MEMSELECT-NOT: neg.tensor.extend_reg_base
// REGSELECT: ascend.implementation = {id = "neg.tensor.extend_reg_base", kind = "ascendc_api"}
