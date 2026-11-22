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

// `afir.truediv` 的 mem 侧：`TrueDivExtend(dst, x1, x2, tmp, size)`，两个扩展头
// （`scalar_div.h` + `true_div.h`）加一个 `utils.h` 前导。
//
// 这份用例的重点是第二个函数：i32 输入、f32 目的地。 那是 `truediv` 唯一不同于
// `div` 的一行——包装里 `if constexpr (IsSameType<T, int32_t>)` 那一支先把两个 src
// `Cast` 成 f32 再 `Div`，`else` 分支字面上就是 `AscendC::Div`。把注册处的
// `ResultElementType::SameOrF32ForI32` 改回 `Same`、或者把 `kI32` 从集合里去掉，
// 第二个函数当场选不出实现。
//
// `WRONGTARGET` 钉的是「整个 module 是 mem 专有的」：reg 侧那个 implementation 建的是
// `DivExtend`，**既不收 tmp buffer、也不收 i32**（v2 的 `kTrueDivTypePairs` 是
// {f16, f32, bf16}），两条里任何一条都足以拒掉这里的两个函数。reg 侧长什么样见
// afir-truediv-reg-base-extension-api.mlir。
//
// 前导头这份用例测不出来：IR 建得出来、`ascir-translate` 也照打那行调用，
// `utils.h` 少了要到真编 kernel 才炸。拦它的是 `load_truediv_store`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @truediv_extend_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                                %destination: !ascendc.local_tensor<?xf32>,
                                %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.truediv %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 目的地的元素类型与两个输入的不同。这是 `truediv` 全部的存在理由。
  func.func @truediv_extend_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                                %destination: !ascendc.local_tensor<?xf32>,
                                %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.truediv %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @truediv_extend_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.true_div_extend %arg2, %arg0, %arg1, %arg3, %[[COUNT]]
// IR-NOT: afir.truediv

// i32 那一份建的是同一个 op，只是 dst 与 src 的元素类型不同——模板实参由实参推导，
// 生成侧一个都不写。
// IR-LABEL: func.func @truediv_extend_i32
// IR: %[[COUNT32:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.true_div_extend %arg2, %arg0, %arg1, %arg3, %[[COUNT32]]
// IR-NOT: ascendc.div_extend_reg_base
// IR-NOT: afir.truediv

// 裸名，不带 `AscendC::` 限定——`true_div.h` 里的包装是全局函数。五个实参按位置对：
// dst, src0, src1, tmp, count。
// CPP-LABEL: void truediv_extend_f32(
// CPP: {{^ *}}TrueDivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-LABEL: void truediv_extend_i32(
// CPP: {{^ *}}TrueDivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT-COUNT-2: ascend.implementation = {id = "truediv.tensor_tensor.extend", kind = "ascendc_api"}

// WRONGTARGET: no supported
