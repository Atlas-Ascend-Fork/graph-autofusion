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

// dtype 第二波扩表（i8 / ui32 / ui64）的选择层验证。三个类型 legacy 只在 reg 侧供：
//   - `sub` 的 reg 行随之收宽，发出的仍是 `SubExtend` 包装；
//   - `eq` 走比较族的 reg 行；
//   - `add` 的 reg 侧集合在拆出的 `extend_reg_base` 行上（主行仍收两列交集）。
// floor_div 的 i8 与 i64 同理留在窄侧：包装分派没有 int8_t 分支，选不中是预期的
// 干净回落，这里钉住它——升表放大器效应的另一面（位有了，但那一行仍不许收）。

// RUN: sed 's/xui32/xi8/g' %s > %t.i8.mlir
// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %t.i8.mlir 2>&1 | FileCheck %s --check-prefix=NO-I8-FLOORDIV
// RUN: sed 's/afir.add /afir.mul /g' %s > %t.mul-i8.mlir
// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %t.mul-i8.mlir 2>&1 | FileCheck %s --check-prefix=NO-I8-MUL
// NO-I8-MUL: 'afir.mul' op has no supported Ascend implementation
// NO-I8-FLOORDIV: 'afir.floor_div' op has no supported Ascend implementation

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @sub_reg_base_ui32(%lhs: !ascendc.local_tensor<?xui32>, %rhs: !ascendc.local_tensor<?xui32>,
                               %destination: !ascendc.local_tensor<?xui32>) {
    %result = afir.sub %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui32>, !ascendc.local_tensor<?xui32>),
        !ascendc.local_tensor<?xui32> -> !ascendc.local_tensor<?xui32>
    return
  }

  func.func @sub_reg_base_ui64(%lhs: !ascendc.local_tensor<?xui64>, %rhs: !ascendc.local_tensor<?xui64>,
                               %destination: !ascendc.local_tensor<?xui64>) {
    %result = afir.sub %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui64>, !ascendc.local_tensor<?xui64>),
        !ascendc.local_tensor<?xui64> -> !ascendc.local_tensor<?xui64>
    return
  }

  func.func @eq_reg_base_i8(%lhs: !ascendc.local_tensor<?xi8>, %rhs: !ascendc.local_tensor<?xi8>,
                            %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.eq %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi8>, !ascendc.local_tensor<?xi8>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @add_reg_base_i8(%lhs: !ascendc.local_tensor<?xi8>, %rhs: !ascendc.local_tensor<?xi8>,
                             %destination: !ascendc.local_tensor<?xi8>) {
    %result = afir.add %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi8>, !ascendc.local_tensor<?xi8>),
        !ascendc.local_tensor<?xi8> -> !ascendc.local_tensor<?xi8>
    return
  }

  func.func @floor_div_reg_base_ui32(%lhs: !ascendc.local_tensor<?xui32>, %rhs: !ascendc.local_tensor<?xui32>,
                                     %destination: !ascendc.local_tensor<?xui32>) {
    %result = afir.floor_div %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui32>, !ascendc.local_tensor<?xui32>),
        !ascendc.local_tensor<?xui32> -> !ascendc.local_tensor<?xui32>
    return
  }
}

// IR-LABEL: func.func @sub_reg_base_ui32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.sub_extend_reg_base %arg2, %arg0, %arg1, %[[COUNT]]
// IR-NOT: afir.sub

// IR-LABEL: func.func @sub_reg_base_ui64
// IR: ascendc.sub_extend_reg_base

// IR-LABEL: func.func @eq_reg_base_i8
// IR: ascendc.compare_extend_contiguous_reg_base
// IR-NOT: afir.eq

// IR-LABEL: func.func @add_reg_base_i8
// IR: ascendc.add_l2
// IR-NOT: afir.add

// IR-LABEL: func.func @floor_div_reg_base_ui32
// IR: ascendc.floor_div_extend_reg_base

// reg 侧 sub 的包装与 mem 侧同名差异同 afir-sub-div-reg-base-extension-api.mlir：全局
// 函数、不带 `AscendC::` 限定。ui32 的 size 实参按 4 字节折算。
// CPP-LABEL: void sub_reg_base_ui32(
// CPP: {{^ *}}SubExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// ui64 的 sub 也走 SubExtend；i8 的 add 走 reg 独立注册行，仍调用 AscendC::Add。
// SELECT-COUNT-2: ascend.implementation = {id = "sub.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "eq.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "add.tensor_tensor.reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "floor_div.tensor.extend_reg_base", kind = "ascendc_api"}

// 910B 没有 reg 家族：sub 的 ui32/ui64、eq 的 i8、add 的 i8、floor_div 的 ui32 在 mem
// 列都没有，主行也收不进——五个函数全部干净回落。
// WRONGTARGET: no supported
