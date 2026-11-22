// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt %s | FileCheck %s

module attributes {
  afir.expr_constant = #afir.expr<"1">,
  afir.expr_symbol = #afir.expr<"s0">,
  afir.expr_add = #afir.expr<"(1 + s0)">,
  afir.expr_ceil = #afir.expr<"Ceiling((s0 / (tile_size)))">,
  afir.expr_max = #afir.expr<"Max(s0, s1)">,
  afir.expr_mod = #afir.expr<"Mod(s0, s1)">,
  afir.expr_pow = #afir.expr<"Pow(s0, -1)">,
  afir.expr_rational = #afir.expr<"Rational(1 , 2)">,
  afir.expr_relation = #afir.expr<"ExpectLt(s0, s1)">,
  afir.expr_logic = #afir.expr<"LogicAnd(ExpectLt(s0, s1), ExpectLt(s1, s2))">
} {}

// CHECK: afir.expr_add = #afir.expr<"(1 + s0)">
// CHECK: afir.expr_ceil = #afir.expr<"Ceiling((s0 / (tile_size)))">
// CHECK: afir.expr_constant = #afir.expr<"1">
// CHECK: afir.expr_logic = #afir.expr<"LogicAnd(ExpectLt(s0, s1), ExpectLt(s1, s2))">
// CHECK: afir.expr_max = #afir.expr<"Max(s0, s1)">
// CHECK: afir.expr_mod = #afir.expr<"Mod(s0, s1)">
// CHECK: afir.expr_pow = #afir.expr<"Pow(s0, -1)">
// CHECK: afir.expr_rational = #afir.expr<"Rational(1 , 2)">
// CHECK: afir.expr_relation = #afir.expr<"ExpectLt(s0, s1)">
// CHECK: afir.expr_symbol = #afir.expr<"s0">
