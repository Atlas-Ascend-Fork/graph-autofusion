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

// afir-truediv-extension-api.mlir 的另一半，而这一半**几乎不是 `truediv` 自己的东西**：
// v2 的 `TrueDivAscIrCodegenImplV2` 写的 apiName 是 `DivExtend`、头是 `div_reg_base.h`，
// 与 `DivAscIrCodegenImplV2` 逐字相同。所以这里建的 op 就是 `afir.div` 的 reg 实现今天
// 已经在发的那个 `ascendc.div_extend_reg_base`，没有新建 def。
//
// 于是两侧差的不只是头，还有两样：
//
//   - **tmp buffer**。mem 侧 `TrueDivExtend` 收一块，这一侧的 `BinaryApiCall` 签名里
//     没有，所以整个 module 一块 tmp 都不挂——`WRONGTARGET` 那条就是被这一点拒掉的
//     （mem 侧的 implementation 要求有一块）。
//   - **i32**。`kTrueDivTypePairs` 在 v2 是 {f16, f32, bf16}，i32 那一行只有 mem 侧有，
//     bf16 那一行**现在建模了**（2026-10-20 进供给表），所以这个文件少一个 bf16 的函数
//     是**待补**，不是「表达不出来」——补它要先确认 `DivExtend` 的 reg 包装在 bf16 上
//     发什么。这里只有 f16 / f32 两个函数。
//
// 分得开两侧的只有 IR 上建的是哪个 op 与 SELECT 的 id：CPP 那一行与 `afir.div` 的
// reg 侧逐字相同。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @truediv_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                                  %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.truediv %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @truediv_reg_base_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                                  %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.truediv %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @truediv_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.div_extend_reg_base %arg2, %arg0, %arg1, %[[COUNT]]
// IR-NOT: ascendc.true_div_extend
// IR-NOT: afir.truediv

// IR-LABEL: func.func @truediv_reg_base_f16
// IR: ascendc.div_extend_reg_base

// 与 `afir.div` 的 reg 侧逐字相同的一行——两者本来就是同一个包装。
// CPP-LABEL: void truediv_reg_base_f32(
// CPP: {{^ *}}DivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// id 上仍然是 `truediv` 的：同一个 AscendC op 可以被两个 AFIR op 的 implementation 用。
// SELECT-COUNT-2: ascend.implementation = {id = "truediv.tensor_tensor.extend_reg_base", kind = "ascendc_api"}

// mem 侧那个 implementation 要一块 tmp buffer，这个 module 一块都没有。
// WRONGTARGET: no supported
