// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Equal UB views must remain compatible when GM strides prevent transfer-axis merging.
// Exercise both implementation selection and conversion, on both API families.
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefixes=CHECK,MEM
// RUN: sed '/^\/\//!s/ tmp(%%tmp)//; /^\/\//!s/, %%tmp: !ascendc.local_tensor<128xui8>//; /^\/\//!s/, !ascendc.local_tensor<128xui8>//' %s > %t.reg.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %t.reg.mlir | FileCheck %s --check-prefixes=CHECK,REG
// AbsExtend requires scratch on the mem-based family; AbsL2 takes none on the
// reg-based family. Keep the same GM/UB descriptors in both inputs.

#src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"32">, <"8">, <"1">], vectorized_axis = [1, 2], vectorized_strides = [<"8">, <"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"128">, <"32">, <"1">], vectorized_axis = [1, 2], vectorized_strides = [<"8">, <"1">], role = <compute>>
#graph = #afir.asc_graph<tiling_key = 0, axes = [
  <id = 0, name = "outer", axis_type = Original, size = <"2">>,
  <id = 1, name = "row", axis_type = Original, size = <"4">>,
  <id = 2, name = "col", axis_type = Original, size = <"8">>], type = Compute>

// Complete local views work with unbound GM strides and no schedule or graph metadata.
#dense = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"gm_outer">, <"gm_row">, <"gm_col">],
  vectorized_axis = [1, 2], vectorized_strides = [<"8">, <"1">], role = <compute>>
#pitched = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"gm_outer">, <"gm_row">, <"gm_col">],
  vectorized_axis = [1, 2], vectorized_strides = [<"16">, <"1">], role = <compute>>

module {
  func.func @abs_different_gm(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>, %tmp: !ascendc.local_tensor<128xui8>) attributes {afir.asc_graph = #graph} {
    %r = afir.abs %a outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst],
      afir.sched = #afir.sched<axis = [0, 1, 2], loop_axis = 0>
    } : !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<128xui8> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @add_different_gm(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>) attributes {afir.asc_graph = #graph} {
    %r = afir.add %a, %b outs(%dst) {
      afir.operand_descs = [#src, #dst], afir.destination_descs = [#dst], afir.result_descs = [#dst],
      afir.sched = #afir.sched<axis = [0, 1, 2], loop_axis = 0>
    } : (!ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>), !ascendc.local_tensor<2x4x8xf32> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @abs_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>, %tmp: !ascendc.local_tensor<128xui8>) {
    %r = afir.abs %a outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#pitched], afir.destination_descs = [#dense], afir.result_descs = [#dense]
    } : !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<128xui8> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @add_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>) {
    %r = afir.add %a, %b outs(%dst) {
      afir.operand_descs = [#pitched, #dense], afir.destination_descs = [#dense], afir.result_descs = [#dense]
    } : (!ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>), !ascendc.local_tensor<2x4x8xf32> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @clip_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %c: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>, %tmp: !ascendc.local_tensor<128xui8>) {
    %r = afir.clip_by_value %a, %b, %c outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#dense, #pitched, #dense], afir.destination_descs = [#dense], afir.result_descs = [#dense]
    } : (!ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>), !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<128xui8> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }

}
// CHECK-LABEL: func.func @abs_different_gm
// MEM: ascendc.abs_extend
// REG: ascendc.abs_l2
// CHECK-NOT: afir.abs
// CHECK-LABEL: func.func @add_different_gm
// CHECK: ascendc.
// CHECK-NOT: afir.add
// CHECK-LABEL: func.func @abs_different_ub
// CHECK-NOT: ascendc.local_tensor.get_size
// CHECK: scf.for
// CHECK: ascendc.local_tensor.subindex
// CHECK: ascendc.local_tensor.subindex
// MEM: ascendc.abs_extend
// REG: ascendc.abs_l2
// CHECK-LABEL: func.func @add_different_ub
// CHECK-NOT: ascendc.local_tensor.get_size
// CHECK: scf.for
// CHECK: ascendc.local_tensor.subindex
// CHECK: ascendc.local_tensor.subindex
// MEM: ascendc.add_l2
// REG: ascendc.add_l2
// CHECK-LABEL: func.func @clip_different_ub
// CHECK-NOT: ascendc.local_tensor.get_size
// CHECK: scf.for
// CHECK: ascendc.local_tensor.subindex
// CHECK: ascendc.local_tensor.subindex
// MEM: ascendc.clipbyvalue_contiguous
// REG: ascendc.clipbyvalue_contiguous_reg_base
