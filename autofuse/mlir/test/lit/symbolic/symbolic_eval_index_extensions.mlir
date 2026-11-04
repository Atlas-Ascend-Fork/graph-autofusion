// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You should not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --split-input-file %s --afir-verify-symbolic-safety --afir-lower-symbolic \
// RUN:   --afir-verify-no-symbolic-eval | FileCheck %s

module {
  func.func @index_pow(%base: index, %scale: index) -> index {
    %result = afir.symbolic_eval %base, %scale {
      expr = #afir.expr<"((__arg0 * __arg0) * __arg1)">,
      overflow_checked
    } : (index, index) -> index
    return %result : index
  }
}

// CHECK-LABEL: func.func @index_pow
// CHECK: %[[SQUARE:.*]] = index.mul %arg0, %arg0
// CHECK: %[[RESULT:.*]] = index.mul %[[SQUARE]], %arg1
// CHECK: return %[[RESULT]]
// CHECK-NOT: afir.symbolic_eval

// -----
module {
  func.func @index_rational_ceil(%value: index) -> index {
    %result = afir.symbolic_eval %value {
      expr = #afir.expr<"Ceiling((Rational(1 , 8) * __arg0))">,
      overflow_checked
    } : (index) -> index
    return %result : index
  }
}

// CHECK-LABEL: func.func @index_rational_ceil
// CHECK: %[[CONSTANT:.*]] = arith.constant 8
// CHECK: %[[CEIL:.*]] = index.ceildivu %arg0, %[[CONSTANT]]
// CHECK: return %[[CEIL]]
// CHECK-NOT: afir.symbolic_eval
