// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt --verify-diagnostics --split-input-file %s

// `afir.scalar` 的结果有两种形态：tensor 形态（逻辑形态算子的标量操作数）与单个值
// （buffer 形态算子的标量操作数，bridge 为 ui16 / ui32 字面量建的就是这一种）。字面量的类型
// 必须等于结果的元素类型——对不上时下降出来的常量会带着另一种 dtype 被使用点读走。

// -----

// 正向：两种形态都能建出来。
func.func private @forms() {
  %value = afir.scalar {value = 3 : ui32} : ui32
  %tensor = afir.scalar {value = 3 : ui32} : tensor<4xui32>
  return
}

// -----

// 负向：单个值，字面量类型与结果不一致。
func.func private @value_mismatch() {
  // expected-error @+1 {{value attribute type must match the result element type}}
  %value = afir.scalar {value = 3 : ui32} : ui16
  return
}

// -----

// 负向：tensor 形态，字面量类型与元素类型不一致。
func.func private @tensor_mismatch() {
  // expected-error @+1 {{value attribute type must match the result element type}}
  %tensor = afir.scalar {value = 3 : ui32} : tensor<4xui16>
  return
}
