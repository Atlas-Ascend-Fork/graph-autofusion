// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=LOWER
// Only axis 1 broadcasts; the unit axis 0 has irrelevant unequal strides.
// The vector window [1, 32] is a row, not a scalar.
#src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"32">], strides = [#afir.expr<"1">, #afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 2], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"1">, #afir.expr<"2">, #afir.expr<"32">], strides = [#afir.expr<"64">, #afir.expr<"32">, #afir.expr<"1">], vectorized_axis = [0, 2], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>

func.func @mul_broadcast_lhs(%row: !ascendc.local_tensor<1x1x32xf32>, %full: !ascendc.local_tensor<1x2x32xf32>, %dst: !ascendc.local_tensor<1x2x32xf32>) {
  %r = afir.mul %row, %full outs(%dst) {afir.operand_descs = [#src, #dst], afir.destination_descs = [#dst], afir.result_descs = [#dst]} : (!ascendc.local_tensor<1x1x32xf32>, !ascendc.local_tensor<1x2x32xf32>), !ascendc.local_tensor<1x2x32xf32> -> !ascendc.local_tensor<1x2x32xf32>
  return
}
// CHECK-LABEL: func.func @mul_broadcast_lhs
// CHECK: id = "mul.tensor_tensor
// LOWER-LABEL: func.func @mul_broadcast_lhs
// LOWER: ascendc.mul
// LOWER-NOT: ascendc.muls
// LOWER: return

func.func @mul_broadcast_rhs(%row: !ascendc.local_tensor<1x1x32xf32>, %full: !ascendc.local_tensor<1x2x32xf32>, %dst: !ascendc.local_tensor<1x2x32xf32>) {
  %r = afir.mul %full, %row outs(%dst) {afir.operand_descs = [#dst, #src], afir.destination_descs = [#dst], afir.result_descs = [#dst]} : (!ascendc.local_tensor<1x2x32xf32>, !ascendc.local_tensor<1x1x32xf32>), !ascendc.local_tensor<1x2x32xf32> -> !ascendc.local_tensor<1x2x32xf32>
  return
}
// CHECK-LABEL: func.func @mul_broadcast_rhs
// CHECK: id = "mul.tensor_tensor
// LOWER-LABEL: func.func @mul_broadcast_rhs
// LOWER: ascendc.mul
// LOWER-NOT: ascendc.muls
// LOWER: return
