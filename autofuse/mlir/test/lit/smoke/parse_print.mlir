// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s
// RUN: af-opt --version | FileCheck %s --check-prefix=VERSION

module attributes {af.smoke = "parse-print"} {
  func.func private @bias(%arg0: i32) -> i32

  func.func @accumulate(%input: memref<4xi32>, %limit: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %sum = scf.for %i = %c0 to %limit step %c1 iter_args(%acc = %zero) -> (i32) {
      %value = memref.load %input[%i] : memref<4xi32>
      %biased = func.call @bias(%value) : (i32) -> i32
      %next = arith.addi %acc, %biased : i32
      scf.yield %next : i32
    }
    return %sum : i32
  }

  func.func @choose(%lhs: i32, %rhs: i32) -> i32 {
    %cond = arith.cmpi sgt, %lhs, %rhs : i32
    %result = arith.select %cond, %lhs, %rhs : i1, i32
    return %result : i32
  }
}

// CHECK-LABEL: module attributes {af.smoke = "parse-print"} {
// CHECK: func.func private @bias(i32) -> i32
// CHECK-LABEL: func.func @accumulate
// CHECK-SAME: memref<4xi32>
// CHECK-SAME: index
// CHECK: arith.constant 0 : index
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: memref.load
// CHECK: func.call @bias
// CHECK: arith.addi
// CHECK: scf.yield
// CHECK: return
// CHECK-LABEL: func.func @choose
// CHECK: arith.cmpi sgt
// CHECK: arith.select
// CHECK: return

// VERSION: LLVM version
// VERSION: Autofuse MLIR version af-mlir-phase1
