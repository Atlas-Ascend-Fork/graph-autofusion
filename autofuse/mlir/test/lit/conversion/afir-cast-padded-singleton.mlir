// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: sed 's/ tmp([^)]*)//g; s/, !ascendc.local_tensor<?xui8>//g' %s > %t.reg.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %t.reg.mlir | FileCheck %s
// RUN: sed 's/<"16">/<"15">/g' %s > %t.unaligned.mlir
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %t.unaligned.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: 'afir.cast' op has no supported Ascend implementation

// Each logical element occupies its own 32-byte block. Removing the singleton
// axis must not turn the remaining padded axis into a contiguous API window.
#src = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"16">, <"0">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [<"4">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"8">, <"0">], role = <compute>>

module {
  func.func @cast_padded_f16(%input: !ascendc.local_tensor<?xf16>,
      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
  func.func @cast_padded_bf16(%input: !ascendc.local_tensor<?xbf16>,
      %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @cast_padded_f16
// CHECK: scf.for %[[I:.*]] =
// CHECK: %[[S:.*]] = arith.constant 16 : index
// CHECK: %[[SI:.*]] = arith.muli %[[I]], %[[S]]
// CHECK: %[[SRC:.*]] = ascendc.local_tensor.subindex %arg0[%[[SI]]]
// CHECK: %[[D:.*]] = arith.constant 8 : index
// CHECK: %[[DI:.*]] = arith.muli %[[I]], %[[D]]
// CHECK: %[[DST:.*]] = ascendc.local_tensor.subindex %arg1[%[[DI]]]
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: ascendc.cast_extend_contiguous{{(_reg_base)?}} %[[DST]], %[[SRC]], {{(%arg2, )?}}%[[ONE]]
// CHECK-NOT: afir.cast

// CHECK-LABEL: func.func @cast_padded_bf16
// CHECK: scf.for %[[I:.*]] =
// CHECK: %[[S:.*]] = arith.constant 16 : index
// CHECK: %[[SI:.*]] = arith.muli %[[I]], %[[S]]
// CHECK: %[[SRC:.*]] = ascendc.local_tensor.subindex %arg0[%[[SI]]]
// CHECK: %[[D:.*]] = arith.constant 8 : index
// CHECK: %[[DI:.*]] = arith.muli %[[I]], %[[D]]
// CHECK: %[[DST:.*]] = ascendc.local_tensor.subindex %arg1[%[[DI]]]
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: ascendc.cast_extend_contiguous{{(_reg_base)?}} %[[DST]], %[[SRC]], {{(%arg2, )?}}%[[ONE]]
// CHECK-NOT: afir.cast
