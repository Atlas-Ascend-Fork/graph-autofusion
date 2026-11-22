// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt --split-input-file %s --afir-verify-symbolic-safety 2>&1 | FileCheck %s

module {
  func.func @unsafe_dynamic_divisor(%dim: index, %tile: index) -> index {
    %repeat = afir.symbolic_eval %dim, %tile {
      expr = #afir.expr<"Ceiling((__arg0 / (__arg1)))">
    } : (index, index) -> index
    return %repeat : index
  }
}

// CHECK: error: 'afir.symbolic_eval' op contains an unsafe or unsupported power

// -----

// Rational 只允许作为 Ceiling/Floor 的缩放因子；脱离 rounding 上下文时必须拒收。
module {
  func.func @rational_outside_rounding() -> index {
    %result = afir.symbolic_eval {
      expr = #afir.expr<"Rational(1 , 8)">
    } : () -> index
    return %result : index
  }
}

// CHECK: error: 'afir.symbolic_eval' op contains a rational that is not a ceiling/floor scaling factor

// -----

// 提供 overflow_checked 排除乘法溢出检查的影响，仅验证分子必须非负这一条件。
module {
  func.func @unsafe_rational_scaling(%value: index) -> index {
    %result = afir.symbolic_eval %value {
      expr = #afir.expr<"Ceiling((Rational(-1 , 8) * __arg0))">,
      overflow_checked
    } : (index) -> index
    return %result : index
  }
}

// CHECK: error: 'afir.symbolic_eval' op contains an unsafe rational scaling factor

// -----

// 指数不是可展开的静态非负整数；这里只测符号指数这一格。
module {
  func.func @symbolic_power(%base: index, %exponent: index) -> index {
    %result = afir.symbolic_eval %base, %exponent {
      expr = #afir.expr<"Pow(__arg0, __arg1)">
    } : (index, index) -> index
    return %result : index
  }
}

// CHECK: error: 'afir.symbolic_eval' op contains an unsupported negative or symbolic power
