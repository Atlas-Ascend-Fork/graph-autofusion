// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s
// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s

// 谓词族关掉了 ODS 的 `AllElementTypesMatch`（它会把 `f32 -> ui8` 判死），于是
// `AFIR_BufferizableShapedType` 对 `!ascendc.local_tensor` 那一路**不再约束任何元素
// 类型**——下面这几份 IR 都能过 parser 和 verifier。唯一还拦得住它们的，是 capability
// 里 `areElementTypesSupported` 那几行。
//
// 没有这份文件的话：把那几行删成只查源、或者把「结果一个字节」漏掉，全套用例照样绿，
// 而错误要一路走到 device 编译器上，报在内联进 kernel 的头文本行号里。
//
// 三条判据在两张供给表上都成立，所以两个 `--target` 共用同一份 IR：能触发的只剩元素
// 类型和形状，家族与 tmp buffer 都对得上。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 结果必须是一个字节宽：f32 目的地不行。
func.func @isnan_float_destination(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                                   %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.isnan %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 输入必须是浮点：`isnan(int32)` 没有意义。只查「结果是 ui8」会把它放进来。
func.func @isnan_integer_input(%input: !ascendc.local_tensor<?xi32>, %destination: !ascendc.local_tensor<?xui8>,
                               %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.isnan %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 形状仍然要逐字相同。这一条以前由「整个类型相等」顺带管着，谓词族把它拆成只比形状
// 之后，得单独有条负例钉住——拆错的表现是把源和目的地长度不同的 op 也放行。
func.func @isnan_shape_mismatch(%input: !ascendc.local_tensor<4xf32>, %destination: !ascendc.local_tensor<8xui8>,
                                %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.isnan %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<4xf32>, !ascendc.local_tensor<8xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<8xui8>
  return
}
