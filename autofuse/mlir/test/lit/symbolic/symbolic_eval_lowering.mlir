// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt %s --afir-simplify-symbolic --afir-verify-symbolic-safety --afir-lower-symbolic \
// RUN:   --afir-verify-no-symbolic-eval | FileCheck %s

module {
  func.func @host_tiling_symbolic_demo(%dim: index, %tile: index) -> (index, index, index, i1, i1, i1) {
    %repeat = afir.symbolic_eval %dim, %tile {
      expr = #afir.expr<"Ceiling((__arg0 / (__arg1)))">,
      positive_args = array<i64: 1>
    } : (index, index) -> index
    %extent = afir.symbolic_eval %dim, %tile {
      expr = #afir.expr<"Max(__arg0, __arg1)">
    } : (index, index) -> index
    %offset = afir.symbolic_eval %dim, %tile {
      expr = #afir.expr<"(__arg0 + __arg1)">,
      overflow_checked
    } : (index, index) -> index
    %guard = afir.symbolic_eval %dim, %tile {
      expr = #afir.expr<"ExpectLe(__arg1, __arg0)">
    } : (index, index) -> i1
    %true = afir.symbolic_eval {
      expr = #afir.expr<"True">
    } : () -> i1
    %false = afir.symbolic_eval {
      expr = #afir.expr<"False">
    } : () -> i1
    return %repeat, %extent, %offset, %guard, %true, %false : index, index, index, i1, i1, i1
  }

  func.func @aligned_extent(%extent: index) -> index {
    %aligned = afir.symbolic_eval %extent {
      expr = #afir.expr<"(8 * Ceiling((Rational(1 , 8) * __arg0)))">,
      overflow_checked
    } : (index) -> index
    return %aligned : index
  }
}

// CHECK-LABEL: func.func @host_tiling_symbolic_demo
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[REPEAT:.*]] = index.ceildivu %arg0, %arg1
// CHECK: %[[EXTENT:.*]] = index.maxu %arg0, %arg1
// CHECK: %[[OFFSET:.*]] = index.add %arg1, %arg0
// CHECK: %[[GUARD:.*]] = index.cmp ule(%arg1, %arg0)
// CHECK: return %[[REPEAT]], %[[EXTENT]], %[[OFFSET]], %[[GUARD]], %[[TRUE]], %[[FALSE]]
// CHECK-NOT: afir.symbolic_eval

// CHECK-LABEL: func.func @aligned_extent
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[DIV:.*]] = index.ceildivu %arg0, %[[C8]]
// CHECK: %[[ALIGNED:.*]] = index.mul %[[DIV]], %[[C8]]
// CHECK: return %[[ALIGNED]]
// CHECK-NOT: afir.symbolic_eval
