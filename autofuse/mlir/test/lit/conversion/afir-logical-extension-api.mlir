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

// `afir.logical_and` / `afir.logical_or` 的 mem 侧。两个 op 合在一份用例里，因为它们
// 只差名字：legacy 的两个 impl 类逐行同形，两侧的 `REG_ASC_IR` 也逐字同形。
//
// 这份用例钉的是二元一类里第一次出现的那条性质：
//
//   结果的 dtype 与两个输入无关，恒为 `ui8`。
//
// 每个函数的两个输入都是同一个非 `ui8` 的 dtype，目的地是 `ui8`。此前所有二元 op 的
// `AllElementTypesMatch` 都把三者绑在一起，这一对只绑前两个——ODS 那一层放开了，
// capability 那一层由 `ResultElementType::Uint8Logical` 接住，两层缺一层都会让
// `f32, f32 -> ui8` 判成 UnsupportedType。
//
// 另一条是 tmp buffer：mem 侧的包装真的用它（把两个输入 `Cast` 成 half 都落在
// 里面），所以是五个实参的调用。这与 reg 侧那份（四参、无 tmp）正好成对。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @logical_and_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                             %destination: !ascendc.local_tensor<?xui8>,
                             %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_and %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @logical_and_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                             %destination: !ascendc.local_tensor<?xui8>,
                             %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_and %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  // 输入也可以本来就是 `ui8`——那时三者同 dtype，但走的仍然是同一个 implementation。
  // 这一条挡的是「把结果判据写成 `destination == lhs`」那种写法：它对这个函数是对的，
  // 对上面两个是错的。
  func.func @logical_or_ui8(%lhs: !ascendc.local_tensor<?xui8>, %rhs: !ascendc.local_tensor<?xui8>,
                            %destination: !ascendc.local_tensor<?xui8>,
                            %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_or %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @logical_or_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                            %destination: !ascendc.local_tensor<?xui8>,
                            %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_or %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }
}

// tmp buffer 是第四个实参，在 count 之前——少一个就挑到别的重载去了。
// IR-LABEL: func.func @logical_and_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.logical_and_extend %arg2, %arg0, %arg1, %arg3, %[[COUNT]]
// IR-NOT: ascendc.logical_and_extend_reg_base
// IR-NOT: afir.logical_and

// IR-LABEL: func.func @logical_and_i32
// IR: ascendc.logical_and_extend

// IR-LABEL: func.func @logical_or_ui8
// IR: ascendc.logical_or_extend

// IR-LABEL: func.func @logical_or_f16
// IR: ascendc.logical_or_extend
// IR-NOT: afir.logical_or

// 裸名，不带 `AscendC::` 限定——`logical.h` 里的包装是全局函数。名字上没有
// `Extend` 后缀，这正是与 reg 侧分得开的地方。
// CPP-LABEL: void logical_and_f32(
// CPP: {{^ *}}LogicalAnd({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-LABEL: void logical_or_f16(
// CPP: {{^ *}}LogicalOr({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT-COUNT-2: ascend.implementation = {id = "logical_and.tensor_tensor", kind = "ascendc_api"}
// SELECT-COUNT-2: ascend.implementation = {id = "logical_or.tensor_tensor", kind = "ascendc_api"}
