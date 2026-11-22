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

// 第 8 批接通的两个：`log10` 与 `hypot`。两个都是 reg-only，而且两个都走标准调用
// ——这是它们与第 7 批那些 `XxxExtendRegBaseOp` 的唯一差别，也是这一份要钉的东西。
//
// 两处坑各有一条反向 CHECK 盯着：
//
//   - `log10` 选错可用实现会变成 `MathLibraryOp`。那一类的 `isReuseSource` 是必填操作数，
//     emitter 把它当模板实参打，于是 IR 里会多出一个 `arith.constant false`、调用会变成
//     `AscendC::Log10<float, c0_i1>(...)`。`IR-NOT: arith.constant` 与 `CPP-NOT: Log10<`
//     就是这一条——它也是第 6 批那段注释里留下的空组合的落点。
//   - `hypot` 选错可用实现会去调 fork 的 `HypotExtend`（`api_regbase/hypot.h` 里确实有那个
//     函数，但 legacy 一处都不加载它）。`CPP-NOT: HypotExtend` 盯这一条。
//
// 实参个数仍然是调用形态的分界：`log10` 不要 tmp，三个；`hypot` 要一块，五个。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @log10_f32(%input: !ascendc.local_tensor<?xf32>,
                       %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.log10 %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @hypot_f32(%lhs: !ascendc.local_tensor<?xf32>,
                       %rhs: !ascendc.local_tensor<?xf32>,
                       %destination: !ascendc.local_tensor<?xf32>,
                       %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.hypot %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 一元标准调用、不要 tmp：三个实参，而且没有第四个 `isReuseSource`。
// IR-LABEL: func.func @log10_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.log10_l2 %arg1, %arg0, %[[COUNT]]
// IR-NOT: arith.constant
// IR-NOT: ascendc.log10_extend
// IR-NOT: afir.log10

// 二元标准调用、带 tmp：五个实参，tmp 排在两个源之后、计数之前。
// IR-LABEL: func.func @hypot_f32
// IR: %[[HCOUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.hypot_l2 %arg2, %arg0, %arg1, %arg3, %[[HCOUNT]]
// IR-NOT: ascendc.hypot_extend
// IR-NOT: afir.hypot

// CPP-LABEL: void log10_f32(
// CPP: AscendC::Log10(
// CPP-NOT: Log10<
// CPP-NOT: Log10Extend

// CPP-LABEL: void hypot_f32(
// CPP: AscendC::Hypot(
// CPP-NOT: HypotExtend

// SELECT: ascend.implementation = {id = "log10.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "hypot.tensor", kind = "ascendc_api"}

// 两个都只有 reg 侧的 implementation，mem 目标上一条都选不出来。
// WRONGTARGET: no supported
