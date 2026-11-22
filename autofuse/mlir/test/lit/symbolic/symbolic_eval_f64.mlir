// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt %s --afir-verify-symbolic-safety --afir-lower-symbolic --afir-verify-no-symbolic-eval \
// RUN:   | FileCheck %s

module {
  func.func @perf_expression(%latency: f64, %exponent: f64) -> (f64, f64, f64) {
    %scaled = afir.symbolic_eval %latency {
      expr = #afir.expr<"(Rational(1 , 2) * __arg0)">
    } : (f64) -> f64
    %logged = afir.symbolic_eval %latency {
      expr = #afir.expr<"Log(__arg0)">,
      positive_args = array<i64: 0>
    } : (f64) -> f64
    %powered = afir.symbolic_eval %latency, %exponent {
      expr = #afir.expr<"Pow(__arg0, __arg1)">,
      positive_args = array<i64: 0>
    } : (f64, f64) -> f64
    return %scaled, %logged, %powered : f64, f64, f64
  }
}

// CHECK-LABEL: func.func @perf_expression
// CHECK: arith.constant 5.000000e-01 : f64
// CHECK: arith.mulf
// CHECK: math.log %arg0
// CHECK: math.powf %arg0, %arg1
// CHECK-NOT: afir.symbolic_eval
