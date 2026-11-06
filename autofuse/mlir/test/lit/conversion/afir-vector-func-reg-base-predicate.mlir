// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-vector-func-to-reg-base %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-vector-func-to-reg-base %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

// RUN: sed 's/afir.eq/afir.ge/g' %s | af-opt --afir-vector-func-to-reg-base | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=GE
// GE: AscendC::Reg::Compare<float, AscendC::CMPMODE::GE>(

// M3：谓词寄存器作为值（`afir.eq` -> `afir.where`）。
//
// 这一格钉的是**边界语义与产物形状**，数值那一半在 `load_eq_where_reg_base_store` 那格
// E2E 上（CPU 模拟器，32768 个元素逐个比）。要钉的三件事：
//
//   1. `afir.eq` 的结果声明成 `ascendc.mask_reg`，**不是** `ascendc.reg_tensor<ui8>`。
//      两者在 C++ 文本上是两种不同的声明，一位写错就是另一个东西。
//   2. `afir.where` 的 mask 实参**就是那条谓词**，不是 lane 活性掩码。拿后者当条件会在
//      每个条件为假的元素上算错（实测：短路成 lane 掩码后 `element 1` 起就错）。
//      **两路候选的先后（then / else）这一格钉不住**——换成 `Reg::Select(dst, src1, src0,
//      mask)` 在 IR 与 C++ 文本上都看不出不对，它的判据只能是数值（实测：接反后
//      `element 0` 就错）。那一条在 `load_eq_where_reg_base_store` 那格 E2E 上。
//   3. 谓词**一次都不物化成值**：`Reg::Duplicate` 一条都不发。
//
// 计数式而不是「存在一条」：四路取数少发一条也能编过（上一趟 lane 的寄存器还在）。

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @vector_func_predicate(%a: !ascendc.local_tensor<?xf32>, %b: !ascendc.local_tensor<?xf32>,
                                   %t: !ascendc.local_tensor<?xf32>, %e: !ascendc.local_tensor<?xf32>,
                                   %y: !ascendc.local_tensor<?xf32>) {
    %0 = afir.vector_func srcs(%a, %b, %t, %e) dsts(%y) attributes {
      afir.operand_descs = [#in, #in, #in, #in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xf32>, %lb: tensor<?xf32>, %lt: tensor<?xf32>, %le: tensor<?xf32>):
      %cond = afir.eq %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
          : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xui8>
      %res = afir.where %cond, %lt, %le {afir.operand_descs = [#out, #in, #in], afir.result_descs = [#out]}
          : (tensor<?xui8>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      afir.vector_func_yield %res : tensor<?xf32>
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>,
         !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>) -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @vector_func_predicate
// IR: emitasc.vec_scope

// 寄存器声明：四路取数各一个 `RegTensor`，`eq` 的结果一个 `MaskReg`，`where` 的结果再一个
// `RegTensor`。**计数式**：数成六个 `reg_tensor` 就说明谓词被当成一段值声明了。两条
// `IR-NOT` 管的是声明段与第一条搬运之间那一截——到那里为止声明必须正好这些。
// IR-COUNT-4: = ascendc.reg_tensor
// IR: %[[PRED:.*]] = ascendc.mask_reg
// IR: = ascendc.reg_tensor
// IR-NOT: = ascendc.reg_tensor
// IR-NOT: = ascendc.mask_reg
// IR-COUNT-4: ascendc.load_align_reg

// 比较写进那条谓词（`cmpMode = 2` 就是 `CMPMODE::EQ`，`Core/Attributes.td:109`；名字那一侧
// 由下面的 CPP 段钉），选择把它当 mask 用——**两处必须是同一个 SSA 值**，否则就是在拿
// lane 活性掩码当条件。那个 `%[[PRED]]` 绑定是这一格唯一能钉住的语义位（变异验证里它
// 确实红了），`src0` / `src1` 的先后它钉不住。
// IR: ascendc.compare_reg %[[PRED]],
// IR-SAME: cmpMode = 2 : i64
// IR: ascendc.select_reg %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[PRED]]
// IR-COUNT-1: ascendc.store_align_reg

// region 没有了，两个 AFIR 算子也没有剩下。
// IR-NOT: afir.vector_func
// IR-NOT: afir.eq
// IR-NOT: afir.where

// CPP-LABEL: void vector_func_predicate(
// CPP: __VEC_SCOPE__
// 谓词的声明：一条 `MaskReg`，不是 `RegTensor<uint8_t>`。
// CPP: AscendC::Reg::MaskReg
// 比较的元素类型取的是**源**（float），不是目的地——目的地是谓词，没有元素类型。
// CPP: AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(
// CPP: AscendC::Reg::Select(
// 谓词没有被物化回一段值。
// CPP-NOT: AscendC::Reg::Duplicate(
