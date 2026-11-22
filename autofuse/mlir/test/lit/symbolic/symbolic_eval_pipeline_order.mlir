// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt %s --afir-lower-symbolic 2>&1 | FileCheck %s --check-prefix=ORDER
// RUN: not af-opt %s --afir-verify-no-symbolic-eval 2>&1 | FileCheck %s --check-prefix=RESIDUAL

module {
  func.func @pipeline_order(%dim: index) -> index {
    %extent = afir.symbolic_eval %dim {
      expr = #afir.expr<"Ceiling(__arg0)">
    } : (index) -> index
    return %extent : index
  }
}

// ORDER: error: 'afir.symbolic_eval' op must be processed by afir-verify-symbolic-safety before lowering
// RESIDUAL: error: 'afir.symbolic_eval' op remains before backend translation
