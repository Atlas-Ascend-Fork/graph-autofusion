// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Reject an inner A tail: nominal [2,3,8], actual [2,1,8] has valid rows
// at offsets 0 and 24. Flattening Π A_actual would read rows at 0 and 8.
// Multi-A layouts with full inner A extents are supported; this test protects
// the boundary where legacy's flattened row count cannot represent the view.

// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s 2>&1 | FileCheck %s

// 能力查询在 SelectImplementation 处就拒了，走不到下降。下降侧还有第二道
// （`isSupportedReduceLayout` 的同一个谓词），去掉这一道时它会接住并报 `failed to legalize`。
// CHECK: error: 'afir.sum' op has no supported Ascend implementation

module {
func.func private @ar_multi_a_inner_tail(
  %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"192">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
  %tiling: !emitasc.py_struct<"TilingData", [i64], ["t_size"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a0", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 1, name = "a1t", axis_type = TileInner, size = <"t_size">, align = <"1">, from = [3], split_pair_other_id = 4>,
 <id = 2, name = "r", axis_type = Original, size = <"8">, align = <"1">>,
 <id = 3, name = "a1", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 4, name = "a1T", axis_type = TileOuter, size = <"2">, align = <"1">, from = [3], split_pair_other_id = 1>
 ], type = Compute, size_var = [<"t_size">]>
} {
 %nominal = afir.tiling_param "t_size" : index
 %actual = "arith.constant"() {value = 1 : index, afir.axis_binding = #afir.axis_binding<axis = 1, role = actual_size>} : () -> index
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [4, 0, 1, 2], repeats = [<"2">, <"2">, <"t_size">, <"8">], strides = [<"(8 * t_size)">, <"32">, <"8">, <"1">], vectorized_axis = [0, 1, 2], vectorized_strides = [<"(8 * t_size)">, <"8">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [4, 0, 1, 2], repeats = [<"2">, <"2">, <"t_size">, <"1">], strides = [<"t_size">, <"4">, <"1">, <"0">], vectorized_axis = [0, 1, 2], vectorized_strides = [<"t_size">, <"1">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [4, 0, 1, 2], repeats = [<"2">, <"2">, <"t_size">, <"1">], strides = [<"t_size">, <"4">, <"1">, <"0">], vectorized_axis = [0, 1, 2], vectorized_strides = [<"t_size">, <"1">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [4, 0, 1, 2], loop_axis = 4>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}
