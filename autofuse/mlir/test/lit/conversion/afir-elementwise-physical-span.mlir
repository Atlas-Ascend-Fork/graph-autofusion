// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --cse --target=Ascend910B1 %s | FileCheck %s
// RUN: sed 's/16xf32/8xf32/g' %s > %t.small.mlir
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %t.small.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed '/^#dst = /,+1s/<"8">/<"16">/; s/16xf32/32xf32/g' %s > %t.unequal.mlir
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %t.unequal.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: has no supported Ascend implementation

#src = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">], strides = [<"1">, <"0">],
  vectorized_axis = [0, 1], vectorized_strides = [<"8">, <"0">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">], strides = [<"1">, <"0">],
  vectorized_axis = [0, 1], vectorized_strides = [<"8">, <"0">], role = <compute>>
module {
  func.func @neg_span(%src: !ascendc.local_tensor<16xf32>, %dst: !ascendc.local_tensor<16xf32>) {
    %r = afir.neg %src outs(%dst) {afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]}
      : !ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32> -> !ascendc.local_tensor<16xf32>
    return
  }
  func.func @muls_span(%src: !ascendc.local_tensor<16xf32>, %scalar: f32, %dst: !ascendc.local_tensor<16xf32>) {
    %r = afir.mul %src, %scalar outs(%dst) {afir.operand_descs = [#src, #afir.tensor_desc<role = <compute>>], afir.destination_descs = [#dst], afir.result_descs = [#dst]}
      : (!ascendc.local_tensor<16xf32>, f32), !ascendc.local_tensor<16xf32> -> !ascendc.local_tensor<16xf32>
    return
  }
  func.func @mul_span(%src: !ascendc.local_tensor<16xf32>, %rhs: !ascendc.local_tensor<16xf32>, %dst: !ascendc.local_tensor<16xf32>) {
    %r = afir.mul %src, %rhs outs(%dst) {afir.operand_descs = [#src, #src], afir.destination_descs = [#dst], afir.result_descs = [#dst]}
      : (!ascendc.local_tensor<16xf32>, !ascendc.local_tensor<16xf32>), !ascendc.local_tensor<16xf32> -> !ascendc.local_tensor<16xf32>
    return
  }
}
// CHECK-LABEL: func.func @neg_span
// CHECK-NOT: scf.for
// CHECK: arith.constant 2 : index
// CHECK: arith.constant 8 : index
// CHECK: arith.subi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.select
// CHECK: ascendc.neg_extend
// CHECK-NOT: afir.neg
// CHECK-LABEL: func.func @muls_span
// CHECK-NOT: scf.for
// CHECK: arith.constant 2 : index
// CHECK: arith.constant 8 : index
// CHECK: arith.subi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.select
// CHECK: ascendc.muls
// CHECK-NOT: afir.mul
// CHECK-LABEL: func.func @mul_span
// CHECK-NOT: scf.for
// CHECK: arith.constant 2 : index
// CHECK: arith.constant 8 : index
// CHECK: arith.subi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.select
// CHECK: ascendc.mul
// CHECK-NOT: afir.mul
