// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

func.func @wrong_binary_result(%lhs: tensor<2x4xf32>, %rhs: tensor<2x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{result shape must match the inferred broadcast shape}}
  %0 = afir.add %lhs, %rhs : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

func.func @incompatible_binary_operands(%lhs: tensor<2x4xf32>, %rhs: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{operands could not be broadcast to a compatible shape}}
  %0 = afir.add %lhs, %rhs : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

func.func @wrong_reduce_result(%input: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{result shape must match the inferred reduce shape}}
  %0 = afir.sum %input {axis = 0 : i32} : tensor<2x3x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @invalid_reduce_axis(%input: tensor<2x3xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{axis must be in the range [0, input rank)}}
  %0 = afir.sum %input {axis = 2 : i32} : tensor<2x3xf32> -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @invalid_add_element_type(%lhs: tensor<4xf64>, %rhs: tensor<4xf64>) -> tensor<4xf64> {
  // expected-error @+1 {{must be Tensor or MemRef of}}
  %0 = afir.add %lhs, %rhs : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

// 搬运这一族的类型域也会拒收域外的类型。**例子原先是 bf16，2026-10-20 换成 f64**：
// bf16 那天进了 `AFIR_DataMoveElementTypes`（legacy 的 Load / Store 两侧都供得起它），
// 于是这一格不再红——**换例子而不是放宽断言**，钉的仍然是「域外就是拒收」这件事。
// f64 两侧都没有，短期内也不会有。
func.func @invalid_load_element_type(%input: tensor<4xf64>) -> tensor<4xf64> {
  // expected-error @+1 {{must be Tensor or MemRef of}}
  %0 = afir.load %input : tensor<4xf64> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}
