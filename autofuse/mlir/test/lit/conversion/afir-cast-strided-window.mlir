// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --cse --target=Ascend910B1 %s | FileCheck %s
// RUN: sed 's/ tmp([^)]*)//g; s/, !ascendc.local_tensor<?xui8>//g' %s > %t.reg.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --cse --target=Ascend950DT_950x %t.reg.mlir | FileCheck %s

#src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"1">],
  strides = [<"4">, <"1">, <"1">], vectorized_axis = [0, 1, 2],
  vectorized_strides = [<"128">, <"16">, <"0">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"1">],
  strides = [<"4">, <"1">, <"1">], vectorized_axis = [0, 1, 2],
  vectorized_strides = [<"64">, <"8">, <"0">], role = <compute>>
#single = #afir.tensor_desc<axis = [0], repeats = [<"1">], strides = [<"0">],
  vectorized_axis = [0], vectorized_strides = [<"0">], role = <compute>>

module {
  // Only the innermost effective axis becomes rows; outer padding keeps its loop.
  func.func @outer_loop(%input: !ascendc.local_tensor<?xf16>,
      %dst: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %r = afir.cast %input outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
    return
  }
  // No effective segment remains: do not attempt to construct a row window.
  func.func @singleton(%input: !ascendc.local_tensor<?xf16>,
      %dst: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %r = afir.cast %input outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#single], afir.destination_descs = [#single], afir.result_descs = [#single]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
    return
  }
}
// CHECK-LABEL: func.func @outer_loop
// CHECK: scf.for
// CHECK: ascendc.local_tensor.subindex
// CHECK: ascendc.local_tensor.subindex
// CHECK-NOT: scf.for
// CHECK: ascendc.cast_extend_strided{{(_reg_base)?}}
// CHECK-NOT: ascendc.cast_extend_contiguous
// CHECK-LABEL: func.func @singleton
// CHECK-NOT: scf.for
// CHECK: ascendc.cast_extend_contiguous{{(_reg_base)?}}
// CHECK-NOT: ascendc.cast_extend_strided
