// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s

#outside_src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"1">, #afir.expr<"3">, #afir.expr<"2">],
                                 strides = [#afir.expr<"0">, #afir.expr<"8">, #afir.expr<"1">],
                                 vectorized_axis = [0, 1, 2],
                                 vectorized_strides = [#afir.expr<"0">, #afir.expr<"8">, #afir.expr<"1">],
                                 role = <compute>>
#outside_dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"4">, #afir.expr<"3">, #afir.expr<"2">],
                                 strides = [#afir.expr<"24">, #afir.expr<"8">, #afir.expr<"1">],
                                 vectorized_axis = [0, 1, 2],
                                 vectorized_strides = [#afir.expr<"24">, #afir.expr<"8">, #afir.expr<"1">],
                                 role = <compute>>

module {
  func.func @broadcast_segment_outside_window(%src: !ascendc.local_tensor<?xf32>,
                                              %dst: !ascendc.local_tensor<?xf32>,
                                              %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#outside_src],
      afir.destination_descs = [#outside_dst],
      afir.result_descs = [#outside_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// Each plane has three 8-element physical rows, not six packed elements.
// CHECK: %[[N:.*]] = arith.constant 4 : index
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[PLANE:.*]] = arith.constant 24 : index
// CHECK: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[ONE]], %[[PLANE]], %[[ZERO]], %[[N]], %[[PLANE]], %[[ZERO]]
