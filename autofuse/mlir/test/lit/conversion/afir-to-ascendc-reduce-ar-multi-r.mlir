// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Three/four R axes fold into one physical R span with 32-byte innermost padding.
// Outer R and innermost R tails are representable by the existing ReduceInit;
// the companion negative test rejects holes caused by a middle R tail.
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @ar_three_r_padded
// CHECK: ascendc.reduce_sum_2d_l2
module {
func.func private @ar_three_r_padded(
 %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
 %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
 %tiling: !emitasc.py_struct<"TilingData", [i64], ["t_size"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "axis0", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "axis1", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 2, name = "axis2", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 3, name = "axis3", axis_type = Original, size = <"2">, align = <"1">>
 ], type = Compute, size_var = [<"t_size">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"2">, <"2">, <"2">], strides = [<"32">, <"16">, <"8">, <"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"32">, <"16">, <"8">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"1">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"1">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// CHECK-LABEL: func.func private @ar_four_r_padded
// CHECK: ascendc.reduce_sum_2d_l2
module {
func.func private @ar_four_r_padded(
 %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
 %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
 %tiling: !emitasc.py_struct<"TilingData", [i64], ["t_size"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "axis0", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "axis1", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 2, name = "axis2", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 3, name = "axis3", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 4, name = "axis4", axis_type = Original, size = <"2">, align = <"1">>
 ], type = Compute, size_var = [<"t_size">]>
} {
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3, 4], repeats = [<"4">, <"2">, <"2">, <"2">, <"2">], strides = [<"64">, <"32">, <"16">, <"8">, <"1">], vectorized_axis = [0, 1, 2, 3, 4], vectorized_strides = [<"64">, <"32">, <"16">, <"8">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3, 4], repeats = [<"4">, <"1">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3, 4], vectorized_strides = [<"1">, <"0">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3, 4], repeats = [<"4">, <"1">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3, 4], vectorized_strides = [<"1">, <"0">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1, 2, 3, 4], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// CHECK-LABEL: func.func private @ar_three_r_outer_tail
// CHECK: ascendc.reduce_sum_2d_l2
module {
func.func private @ar_three_r_outer_tail(
 %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
 %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
 %tiling: !emitasc.py_struct<"TilingData", [i64], ["t_size"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "axis0", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "axis1", axis_type = Original, size = <"t_size">, align = <"1">>,
 <id = 2, name = "axis2", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 3, name = "axis3", axis_type = Original, size = <"2">, align = <"1">>
 ], type = Compute, size_var = [<"t_size">]>
} {
 %nominal = afir.tiling_param "t_size" : index
 %actual = "arith.constant"() {value = 1 : index, afir.axis_binding = #afir.axis_binding<axis = 1, role = actual_size>} : () -> index
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"t_size">, <"2">, <"2">], strides = [<"(16 * t_size)">, <"16">, <"8">, <"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"(16 * t_size)">, <"16">, <"8">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"1">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"1">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}

// -----

// CHECK-LABEL: func.func private @ar_three_r_innermost_tail
// CHECK: ascendc.reduce_sum_2d_l2
module {
func.func private @ar_three_r_innermost_tail(
 %src: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %dst: !ascendc.local_tensor<?xf32> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
 %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"2048">, life_time_axis_id = -1>},
 %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>},
 %tiling: !emitasc.py_struct<"TilingData", [i64], ["t_size"]> {afir.kernel_arg = #afir.kernel_arg<role = tiling>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "axis0", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "axis1", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 2, name = "axis2", axis_type = Original, size = <"2">, align = <"1">>,
 <id = 3, name = "axis3", axis_type = Original, size = <"t_size">, align = <"1">>
 ], type = Compute, size_var = [<"t_size">]>
} {
 %nominal = afir.tiling_param "t_size" : index
 %actual = "arith.constant"() {value = 1 : index, afir.axis_binding = #afir.axis_binding<axis = 3, role = actual_size>} : () -> index
 %result = afir.sum %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"2">, <"2">, <"t_size">], strides = [<"(32 * Ceiling((Rational(1 , 8) * t_size)))">, <"(16 * Ceiling((Rational(1 , 8) * t_size)))">, <"(8 * Ceiling((Rational(1 , 8) * t_size)))">, <"1">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"(32 * Ceiling((Rational(1 , 8) * t_size)))">, <"(16 * Ceiling((Rational(1 , 8) * t_size)))">, <"(8 * Ceiling((Rational(1 , 8) * t_size)))">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"1">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1, 2, 3], repeats = [<"4">, <"1">, <"1">, <"1">], strides = [<"1">, <"0">, <"0">, <"0">], vectorized_axis = [0, 1, 2, 3], vectorized_strides = [<"1">, <"0">, <"0">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1, 2, 3], loop_axis = 0>
 } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf32>
 return
}
}
