// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// See LICENSE in the root of the software repository for the full text of the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt %s --afir-verify-symbolic-safety 2>&1 | FileCheck %s

// The rational allowance is positional: DIRECT factor of the Mul that is the DIRECT argument
// of the Ceiling/Floor. A Rational nested one level deeper -- here under an Add inside the
// wrapped Mul -- has no exact lowering and stays rejected.
module {
  func.func @rational_nested_add(%size: index, %bias: index) -> index {
    %value = afir.symbolic_eval %size, %bias {
      expr = #afir.expr<"Ceiling(((Rational(1 , 2) + __arg1) * __arg0))">,
      overflow_checked
    } : (index, index) -> index
    return %value : index
  }
}

// CHECK: error: 'afir.symbolic_eval' op contains a rational that is not a ceiling/floor scaling factor
