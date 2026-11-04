// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

// 三元 elementwise 分两族，**只差「第一个操作数进不进类型绑定」这一条**：
//
//   选择族 select / where：第一个是 ui8 真值掩码，不进绑定
//     (`AFIR_BufferizableTernaryElementwiseOp`)
//   clip 族 clip_by_value：第一个是 x，与 min / max 和结果同 dtype，进绑定
//     (`AFIR_BufferizableTernaryHomoElementwiseOp`)
//
// 两个方向各一条负例、一条正例。三条缺一条，那两个 base 就可以被合并成一个而没有任何
// 用例变红——而合并的两种方向都有真实后果：都绑，选择族整条链只剩 ui8 能过；都不绑，
// `clip_by_value` 的 dtype 不一致要走到 device 编译器上才报，报的是模板推导冲突。

// 正例：选择族的条件是 ui8，另外两个是 f32。这一条钉的是「condition **不**在绑定里」。
func.func @select_mask_is_not_bound(%condition: tensor<16xui8>, %lhs: tensor<16xf32>, %rhs: tensor<16xf32>) {
  %result = afir.select %condition, %lhs, %rhs
      : (tensor<16xui8>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return
}

// -----

// 负例：选择族的后两个输入仍然要同 dtype。掩码不进绑定不等于整条绑定没了。
func.func @select_inputs_still_bound(%condition: tensor<16xui8>, %lhs: tensor<16xf16>, %rhs: tensor<16xf32>) {
  // expected-error @+1 {{failed to verify that all of {input1, input2, result} have same element type}}
  %result = afir.select %condition, %lhs, %rhs
      : (tensor<16xui8>, tensor<16xf16>, tensor<16xf32>) -> tensor<16xf32>
  return
}

// -----

// 正例：clip 的四个同 dtype。
func.func @clip_by_value_homogeneous(%x: tensor<16xf32>, %lower: tensor<16xf32>, %upper: tensor<16xf32>) {
  %result = afir.clip_by_value %x, %lower, %upper
      : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return
}

// -----

// 负例：clip 的第一个操作数是 x，不是掩码——它与另外两个不同 dtype 就该在这里就报。
func.func @clip_by_value_first_operand_is_bound(%x: tensor<16xf16>, %lower: tensor<16xf32>,
                                                %upper: tensor<16xf32>) {
  // expected-error @+1 {{failed to verify that all of {condition, input1, input2, result} have same element type}}
  %result = afir.clip_by_value %x, %lower, %upper
      : (tensor<16xf16>, tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return
}
