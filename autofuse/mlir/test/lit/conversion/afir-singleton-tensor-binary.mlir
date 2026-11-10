// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: sed s/f32/f16/g %s > %t.f16.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.f16.mlir | FileCheck %s
// RUN: sed 's/local_tensor<1xf32>/local_tensor<2xf32>/g; s/expr<"1">], strides = \[\#afir.expr<"0">\]/expr<"2">], strides = [#afir.expr<"0">]/g' %s > %t.multi.mlir
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.multi.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/%aux: !ascendc.local_tensor<?xui8>)/%aux: !ascendc.local_tensor<?xui8>, %unknown: !ascendc.local_tensor<1xf32>)/; s/afir.minimum %one,/afir.minimum %unknown,/' %s > %t.unknown.mlir
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.unknown.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: 'afir.minimum' op has no supported Ascend implementation
// A logical singleton stored in UB is read with V_S synchronization, then
// passed to the existing scalar API. Noncommutative operand order is retained.
#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"32">], strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>

func.func @minimum_lhs(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %buf = ascendc.tbuf : <veccalc>
  %one = ascendc.tbuf.get_tensor %buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<1xf32>
  %r = afir.minimum %one, %input outs(%dst) {
    afir.operand_descs = [#one, #full], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<1xf32>, !ascendc.local_tensor<32xf32>), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @minimum_lhs
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.set_flag
// CHECK: ascendc.wait_flag
// CHECK: %[[Sminimumlhs:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[Tminimumlhs:.*]] = ascendc.local_tensor.subindex %arg0[%c0{{.*}}]
// CHECK: %[[Dminimumlhs:.*]] = ascendc.local_tensor.subindex %arg1[%c0{{.*}}]
// CHECK: %[[Nminimumlhs:.*]] = arith.constant 32 : index
// CHECK: ascendc.mins_l2 %[[Dminimumlhs]], %[[Tminimumlhs]], %[[Sminimumlhs]], %[[Nminimumlhs]]

func.func @minimum_rhs(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %buf = ascendc.tbuf : <veccalc>
  %one = ascendc.tbuf.get_tensor %buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<1xf32>
  %r = afir.minimum %input, %one outs(%dst) {
    afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @minimum_rhs
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.set_flag
// CHECK: ascendc.wait_flag
// CHECK: %[[Sminimumrhs:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[Tminimumrhs:.*]] = ascendc.local_tensor.subindex %arg0[%c0{{.*}}]
// CHECK: %[[Dminimumrhs:.*]] = ascendc.local_tensor.subindex %arg1[%c0{{.*}}]
// CHECK: %[[Nminimumrhs:.*]] = arith.constant 32 : index
// CHECK: ascendc.mins_l2 %[[Dminimumrhs]], %[[Tminimumrhs]], %[[Sminimumrhs]], %[[Nminimumrhs]]

func.func @truediv_lhs(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %buf = ascendc.tbuf : <veccalc>
  %one = ascendc.tbuf.get_tensor %buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<1xf32>
  %r = afir.truediv %one, %input outs(%dst) tmp(%aux) {
    afir.operand_descs = [#one, #full], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<1xf32>, !ascendc.local_tensor<32xf32>), !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @truediv_lhs
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.set_flag
// CHECK: ascendc.wait_flag
// CHECK: %[[Struedivlhs:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[Ttruedivlhs:.*]] = ascendc.local_tensor.subindex %arg0[%c0{{.*}}]
// CHECK: %[[Dtruedivlhs:.*]] = ascendc.local_tensor.subindex %arg1[%c0{{.*}}]
// CHECK: %[[Ntruedivlhs:.*]] = arith.constant 32 : index
// CHECK: ascendc.true_div_extends %[[Dtruedivlhs]], %[[Ttruedivlhs]], %[[Struedivlhs]], %arg2, %[[Ntruedivlhs]]

func.func @truediv_rhs(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %buf = ascendc.tbuf : <veccalc>
  %one = ascendc.tbuf.get_tensor %buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<1xf32>
  %r = afir.truediv %input, %one outs(%dst) tmp(%aux) {
    afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @truediv_rhs
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.set_flag
// CHECK: ascendc.wait_flag
// CHECK: %[[Struedivrhs:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[Ttruedivrhs:.*]] = ascendc.local_tensor.subindex %arg0[%c0{{.*}}]
// CHECK: %[[Dtruedivrhs:.*]] = ascendc.local_tensor.subindex %arg1[%c0{{.*}}]
// CHECK: %[[Ntruedivrhs:.*]] = arith.constant 32 : index
// CHECK: ascendc.true_div_extends %[[Dtruedivrhs]], %[[Ttruedivrhs]], %[[Struedivrhs]], %arg2, %[[Ntruedivrhs]]

func.func @pow_lhs(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %buf = ascendc.tbuf : <veccalc>
  %one = ascendc.tbuf.get_tensor %buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<1xf32>
  %r = afir.pow %one, %input outs(%dst) tmp(%aux) {
    afir.operand_descs = [#one, #full], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<1xf32>, !ascendc.local_tensor<32xf32>), !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @pow_lhs
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.set_flag
// CHECK: ascendc.wait_flag
// CHECK: %[[Spowlhs:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[Tpowlhs:.*]] = ascendc.local_tensor.subindex %arg0[%c0{{.*}}]
// CHECK: %[[Dpowlhs:.*]] = ascendc.local_tensor.subindex %arg1[%c0{{.*}}]
// CHECK: %[[Npowlhs:.*]] = arith.constant 32 : index
// CHECK: ascendc.pow %[[Dpowlhs]], %[[Spowlhs]], %[[Tpowlhs]], %arg2, %[[Npowlhs]]

func.func @pow_rhs(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>, %aux: !ascendc.local_tensor<?xui8>) {
  %pipe = ascendc.pipe
  %buf = ascendc.tbuf : <veccalc>
  %one = ascendc.tbuf.get_tensor %buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<1xf32>
  %r = afir.pow %input, %one outs(%dst) tmp(%aux) {
    afir.operand_descs = [#full, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<32xf32>
  return
}
// CHECK-LABEL: func.func @pow_rhs
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.set_flag
// CHECK: ascendc.wait_flag
// CHECK: %[[Spowrhs:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[Tpowrhs:.*]] = ascendc.local_tensor.subindex %arg0[%c0{{.*}}]
// CHECK: %[[Dpowrhs:.*]] = ascendc.local_tensor.subindex %arg1[%c0{{.*}}]
// CHECK: %[[Npowrhs:.*]] = arith.constant 32 : index
// CHECK: ascendc.pow %[[Dpowrhs]], %[[Tpowrhs]], %[[Spowrhs]], %arg2, %[[Npowrhs]]
