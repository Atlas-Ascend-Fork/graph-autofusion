// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Local computation consumes ActualSize even without a GM transfer plan.
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: sed '/%rows = arith.addi/d' %s | not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 2>&1 | FileCheck %s --check-prefix=MISSING
// MISSING: failed to legalize operation 'afir.remove_pad'

#src = #afir.tensor_desc<axis = [11, 13], repeats = [<"n">, <"8">],
  strides = [<"16">, <"1">], vectorized_axis = [11, 13], vectorized_strides = [<"16">, <"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [11, 13], repeats = [<"n">, <"8">],
  strides = [<"8">, <"1">], vectorized_axis = [11, 13], vectorized_strides = [<"8">, <"1">], role = <compute>>
module {
  func.func @remove_pad_actual_rows(%src: !ascendc.local_tensor<?x8xf32>,
      %dst: !ascendc.local_tensor<?x8xf32>, %actual: index) attributes {afir.asc_graph =
    #afir.asc_graph<tiling_key = 0,
      axes = [<id = 10, name = "rows", axis_type = Original, size = <"100">>,
              <id = 11, name = "row_tile", axis_type = TileInner, size = <"n">, from = [10], split_pair_other_id = 12>,
              <id = 12, name = "row_outer", axis_type = TileOuter, size = <"Ceiling((100 / (n)))">, from = [10], split_pair_other_id = 11>,
              <id = 13, name = "cols", axis_type = Original, size = <"8">>], type = Compute>} {
    %zero = arith.constant 0 : index
    %rows = arith.addi %actual, %zero {afir.axis_binding = #afir.axis_binding<axis = 11, role = actual_size>} : index
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?x8xf32>, !ascendc.local_tensor<?x8xf32> -> !ascendc.local_tensor<?x8xf32>
    return
  }
}
// CHECK-LABEL: func.func @remove_pad_actual_rows
// CHECK: %[[ROWS:.*]] = arith.addi
// CHECK: %[[WIDTH:.*]] = arith.constant 8 : index
// CHECK: %[[PITCH:.*]] = arith.constant 16 : index
// CHECK: ascendc.remove_pad_extend {{.*}}, {{.*}}, %[[WIDTH]], %[[PITCH]], %[[ROWS]]
// CHECK-NOT: afir.remove_pad
