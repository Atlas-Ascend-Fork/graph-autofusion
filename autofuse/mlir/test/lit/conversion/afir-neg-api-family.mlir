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

// `afir.neg` 是「差异只在扩展 API 由哪张供给表提供」的那一档。两张表都只有
// `NegExtend` 包装，签名一样，两侧都不要 tmp buffer——所以 AFIR 完全相同，生成的调用
// 文本也完全相同（CPP 的检查两个 target 共用一份），唯一变的是包装出自
// `neg.h` 还是 `neg_reg_base.h`，体现在建出来的是哪个 ascendc op。
//
// 这正是「标准 API 在哪一侧」答不了的问题：`afir.abs` 的标准调用在 reg-base 一侧，
// 这里两侧都没有标准调用。把两者塞进同一个 feature 会在其中一个上答反。

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
// 带上后随的操作数：FileCheck 是子串匹配，光写 `ascendc.neg_extend` 会被
// `ascendc.neg_extend_reg_base` 的前缀吃掉，正好放过它要挡的那个错。
// MEM: ascendc.neg_extend %arg1
// MEM-NOT: afir.neg

// REG-LABEL: func.func @neg_f16
// REG: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// REG-NEXT: ascendc.neg_extend_reg_base %arg1, %arg0, %[[COUNT]]
// REG-LABEL: func.func @neg_i32
// REG: ascendc.neg_extend_reg_base %arg1
// REG-NOT: afir.neg

// 两个 target 上同一组 CHECK：包装是全局函数，没有 `AscendC::` 限定，元素类型是显式
// 模板实参——两张表给的是同一份文本。
// CPP-LABEL: void neg_f16(
// CPP: {{^ *}}NegExtend<half>(
// CPP-LABEL: void neg_i32(
// CPP: {{^ *}}NegExtend<int32_t>(
// CPP-NOT: AscendC::Neg

// stable id 说的是供给表，不是芯片。
// MEMSELECT: ascend.implementation = {id = "neg.tensor.extend", kind = "ascendc_api"}
// MEMSELECT-NOT: neg.tensor.extend_reg_base
// REGSELECT: ascend.implementation = {id = "neg.tensor.extend_reg_base", kind = "ascendc_api"}
