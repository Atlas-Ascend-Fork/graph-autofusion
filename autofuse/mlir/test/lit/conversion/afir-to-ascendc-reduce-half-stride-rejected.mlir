// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt %s --afir-to-ascendc-pipeline --target=Ascend910B1 2>&1 | FileCheck %s
// f16 R=17 requires pitch 32, not the f32 pitch 24.
// CHECK: error:
module {
func.func private @max_f16_wrong_pitch(
  %src: !ascendc.local_tensor<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 0, size_bytes_expr = <"384">, life_time_axis_id = -1>},
  %dst: !ascendc.local_tensor<?xf16> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 1, size_bytes_expr = <"32">, life_time_axis_id = -1>},
  %tmp: !ascendc.local_tensor<?xui8> {afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 2, size_bytes_expr = <"256">, life_time_axis_id = -1>},
  %workspace: memref<ui8, 22 : i32> {afir.kernel_arg = #afir.kernel_arg<role = workspace>}) attributes {
 afir.asc_graph = #afir.asc_graph<tiling_key = 0, axes = [
 <id = 0, name = "a", axis_type = Original, size = <"4">, align = <"1">>,
 <id = 1, name = "r", axis_type = Original, size = <"17">, align = <"1">>
 ], type = Compute, size_var = [<"unused">]>
} {
 %result = afir.max %src outs(%dst) scratch(%tmp) {afir.scratch_roles = ["api_scratch"],
 afir.operand_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"17">], strides = [<"24">, <"1">], vectorized_axis = [0, 1], vectorized_strides = [<"24">, <"1">], role = <compute>>],
 afir.destination_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.result_descs = [#afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">], strides = [<"1">, <"0">], vectorized_axis = [0, 1], vectorized_strides = [<"1">, <"0">], role = <compute>>],
 afir.sched = #afir.sched<axis = [0, 1], loop_axis = 0>
 } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, scratch(!ascendc.local_tensor<?xui8>) -> !ascendc.local_tensor<?xf16>
 return
}
}
