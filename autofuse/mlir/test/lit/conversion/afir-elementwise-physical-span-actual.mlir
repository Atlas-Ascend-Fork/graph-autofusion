// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// The span uses the actual tail, not the nominal tile size or backing capacity.
#tile = #afir.tensor_desc<axis = [11], repeats = [<"n">], strides = [<"1">],
  vectorized_axis = [11], vectorized_strides = [<"8">], role = <compute>>
module {
  func.func @neg_span_actual(%src: !ascendc.local_tensor<?xf32>,
      %dst: !ascendc.local_tensor<?xf32>, %actual: index) attributes {afir.asc_graph =
    #afir.asc_graph<tiling_key = 0,
      axes = [<id = 10, name = "rows", axis_type = Original, size = <"100">>,
              <id = 11, name = "row_tile", axis_type = TileInner, size = <"n">, from = [10], split_pair_other_id = 12>,
              <id = 12, name = "row_outer", axis_type = TileOuter, size = <"Ceiling((100 / (n)))">, from = [10], split_pair_other_id = 11>], type = Compute>} {
    %zero = arith.constant 0 : index
    %rows = arith.addi %actual, %zero {afir.axis_binding = #afir.axis_binding<axis = 11, role = actual_size>} : index
    %r = afir.neg %src outs(%dst) {afir.operand_descs = [#tile], afir.destination_descs = [#tile], afir.result_descs = [#tile]}
      : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}
// CHECK-LABEL: func.func @neg_span_actual
// CHECK: %[[ROWS:.*]] = arith.addi
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[STRIDE:.*]] = arith.constant 8 : index
// CHECK: arith.cmpi sgt, %[[ROWS]], %[[ZERO]]
// CHECK: %[[LAST:.*]] = arith.subi %[[ROWS]], %[[ONE]]
// CHECK: %[[OFFSET:.*]] = arith.muli %[[LAST]], %[[STRIDE]]
// CHECK: %[[SPAN:.*]] = arith.addi %[[ONE]], %[[OFFSET]]
// CHECK: %[[COUNT:.*]] = arith.select {{.*}}, %[[SPAN]], %[[ZERO]]
// CHECK: ascendc.neg_extend {{.*}}, %[[COUNT]]
