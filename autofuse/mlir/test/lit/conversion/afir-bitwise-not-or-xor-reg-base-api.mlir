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

// step 4 接通的三个：`bitwise_not` / `bitwise_or` / `bitwise_xor`。与第 7 批的
// `copy_sign` / `trunc_div` 同形——**全部只有 reg 侧，不带 tmp**——所以每个 op 只有一条
// implementation。
//
// 这一份要钉的是**挑错供给**那个坑：legacy 的 reg 侧发的是 fork 的包装
// `BitwiseNotExtend` / `BitwiseOrExtend` / `BitwiseXorExtend`
// （`api_regbase/bitwise_not.h` / `bitwise_or.h` / `bitwise_xor.h`），**不是**标准调用
// `AscendC::BitwiseNot` / `BitwiseOr` / `BitwiseXor`（`adv_api/math/` 下那三个同名函数
// 确实存在，对 i16/ui16/i32/ui8 这几位两者转发等价，但字面不同——挑标准调用会发一个
// legacy 从不发的调用，与 `afir.hypot` 那次同一种坑）。`CPP-NOT` 那几条盯的就是这个。
//
// 实参个数：一元三个（dst, src, count），二元四个（dst, src0, src1, count）。
// **一元带显式模板实参、二元不带**：`printApiExtensionUnary` 一贯打 `<T>`
// （`NegExtend<float>` 那一支），`printApiExtensionBinary` 的无 tmp 重载不打
// （`CopySignExtend` 那一支）——两条路径不对称，钉死各自该有的那一半。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @bitwise_not_i16(%input: !ascendc.local_tensor<?xi16>,
                             %destination: !ascendc.local_tensor<?xi16>) {
    %result = afir.bitwise_not %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>
        -> !ascendc.local_tensor<?xi16>
    return
  }

  func.func @bitwise_or_i16(%lhs: !ascendc.local_tensor<?xi16>,
                            %rhs: !ascendc.local_tensor<?xi16>,
                            %destination: !ascendc.local_tensor<?xi16>) {
    %result = afir.bitwise_or %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>),
        !ascendc.local_tensor<?xi16>
        -> !ascendc.local_tensor<?xi16>
    return
  }

  func.func @bitwise_xor_i16(%lhs: !ascendc.local_tensor<?xi16>,
                             %rhs: !ascendc.local_tensor<?xi16>,
                             %destination: !ascendc.local_tensor<?xi16>) {
    %result = afir.bitwise_xor %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>),
        !ascendc.local_tensor<?xi16>
        -> !ascendc.local_tensor<?xi16>
    return
  }
}

// IR-LABEL: func.func @bitwise_not_i16
// IR: ascendc.bitwise_not_extend_reg_base %arg1, %arg0
// IR-NOT: afir.bitwise_not

// IR-LABEL: func.func @bitwise_or_i16
// IR: ascendc.bitwise_or_extend_reg_base %arg2, %arg0, %arg1
// IR-NOT: afir.bitwise_or

// IR-LABEL: func.func @bitwise_xor_i16
// IR: ascendc.bitwise_xor_extend_reg_base %arg2, %arg0, %arg1
// IR-NOT: afir.bitwise_xor

// CPP-LABEL: void bitwise_not_i16(
// CPP: {{^ *}}BitwiseNotExtend<int16_t>({{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-NOT: AscendC::BitwiseNot

// CPP-LABEL: void bitwise_or_i16(
// CPP: {{^ *}}BitwiseOrExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-NOT: AscendC::BitwiseOr
// CPP-NOT: BitwiseOr<

// CPP-LABEL: void bitwise_xor_i16(
// CPP: {{^ *}}BitwiseXorExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-NOT: AscendC::BitwiseXor
// CPP-NOT: BitwiseXor<

// SELECT: ascend.implementation = {id = "bitwise_not.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "bitwise_or.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "bitwise_xor.tensor", kind = "ascendc_api"}

// 三个都只有 reg 侧的 implementation，mem 目标上一条都选不出来。
// WRONGTARGET: no supported
