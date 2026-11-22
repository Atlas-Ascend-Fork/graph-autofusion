// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt %s --afir-verify-symbolic-safety 2>&1 | FileCheck %s

module {
  func.func @unchecked_overflow(%lhs: index, %rhs: index) -> index {
    %sum = afir.symbolic_eval %lhs, %rhs {
      expr = #afir.expr<"(__arg0 + __arg1)">
    } : (index, index) -> index
    return %sum : index
  }
}

// CHECK: error: 'afir.symbolic_eval' op integer addition requires an explicit overflow_checked proof
