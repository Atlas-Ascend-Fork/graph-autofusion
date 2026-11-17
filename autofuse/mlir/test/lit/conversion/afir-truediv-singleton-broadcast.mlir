// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// SELECT-COUNT-2: id = "truediv.singleton_scalar.duplicate"
// RUN: sed 's/repeats = \[\#afir.expr<"1">\]/repeats = [#afir.expr<"2">]/' %s > %t.invalid.mlir
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.invalid.mlir 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: no supported Ascend implementation

// Unknown writers cannot safely feed GetValue, even for a singleton.
// RUN: sed 's/%value, %divisor outs/%%seed, %divisor outs/' %s > %t.unsynchronized.mlir
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.unsynchronized.mlir 2>&1 | FileCheck %s --check-prefix=INVALID

#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">],
                         vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"257">], strides = [#afir.expr<"1">],
                          vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
#scalar = #afir.tensor_desc<role = <compute>>
func.func @singleton_truediv(%seed: !ascendc.local_tensor<1xf32>,
                              %dst: !ascendc.local_tensor<257xf32>,
                              %tmp: !ascendc.local_tensor<32xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %value = afir.neg %seed outs(%one) {afir.operand_descs = [#one], afir.destination_descs = [#one],
                                      afir.result_descs = [#one]}
      : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  %divisor = arith.constant 79092.0 : f32
  %result = afir.truediv %value, %divisor outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#one, #scalar], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (!ascendc.local_tensor<1xf32>, f32), !ascendc.local_tensor<257xf32>, !ascendc.local_tensor<32xui8>
      -> !ascendc.local_tensor<257xf32>
  return
}
// CHECK-LABEL: func.func @singleton_truediv
// CHECK: ascendc.pipe.fetch_event_id {{.*}}v_s
// CHECK: ascendc.set_flag
// CHECK: ascendc.wait_flag
// CHECK: %[[VALUE:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[QUOTIENT:.*]] = arith.divf %[[VALUE]], %{{.*}} : f32
// CHECK: ascendc.duplicate_l2 %{{.*}}, %[[QUOTIENT]],
// CHECK-NOT: ascendc.true_div_extends
// CHECK-NOT: afir.truediv
// CPP-LABEL: void singleton_truediv
// CPP: .GetValue(
// CPP: /
// CPP: AscendC::Duplicate(

func.func @scalar_div_singleton(%seed: !ascendc.local_tensor<1xf32>,
                              %dst: !ascendc.local_tensor<257xf32>,
                              %tmp: !ascendc.local_tensor<32xui8>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %value = afir.neg %seed outs(%one) {afir.operand_descs = [#one], afir.destination_descs = [#one],
                                      afir.result_descs = [#one]}
      : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  %divisor = arith.constant 79092.0 : f32
  %result = afir.truediv %divisor, %value outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#scalar, #one], afir.destination_descs = [#full], afir.result_descs = [#full]
  } : (f32, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<257xf32>, !ascendc.local_tensor<32xui8>
      -> !ascendc.local_tensor<257xf32>
  return
}
// CHECK-LABEL: func.func @scalar_div_singleton
// CHECK: %[[VALUE:.*]] = ascendc.local_tensor.get_value
// CHECK: %[[QUOTIENT:.*]] = arith.divf %{{.*}}, %[[VALUE]] : f32
// CHECK: ascendc.duplicate_l2 %{{.*}}, %[[QUOTIENT]],
// CPP-LABEL: void scalar_div_singleton
// CPP: .GetValue(
// CPP: /
// CPP: AscendC::Duplicate(
