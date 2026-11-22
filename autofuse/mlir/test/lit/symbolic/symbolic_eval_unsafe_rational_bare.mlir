// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// See the License for details.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt %s --afir-verify-symbolic-safety 2>&1 | FileCheck %s

// A rational coefficient in a BARE Mul has no exact integer lowering: p * x / q is not an
// integer in general, and truncating it with DivU would silently miscompute. Only the
// Ceiling/Floor-wrapped form is admitted (see symbolic_eval_lowering.mlir).
module {
  func.func @rational_bare_mul(%size: index) -> index {
    %half = afir.symbolic_eval %size {
      expr = #afir.expr<"(Rational(1 , 2) * __arg0)">
    } : (index) -> index
    return %half : index
  }
}

// CHECK: error: 'afir.symbolic_eval' op integer multiplication requires an explicit overflow_checked proof
