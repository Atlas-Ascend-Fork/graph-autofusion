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
// RUN: af-opt --afir-to-ascendc-pipeline --cse --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=MEMCPP
// RUN: af-opt --afir-to-ascendc-pipeline --cse --target=Ascend950DT_950x %t.reg.mlir | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=REGCPP
// RUN: sed 's/xf16/xi16/g; s/xbf16/xi16/g; s/xf32/xui16/g; s/<"8">/<"16">/g' %s > %t.fallback.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.fallback.mlir | FileCheck %s --check-prefix=FALLBACK
// RUN: sed 's/<"8">/<"16">/g' %s > %t.equal.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.equal.mlir | FileCheck %s --check-prefix=FALLBACK
// RUN: sed 's/xf16/xui8/g; s/xbf16/xui8/g; s/<"16">/<"32">/g' %s > %t.scratch.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.scratch.mlir | FileCheck %s --check-prefix=FALLBACK
// RUN: af-opt --afir-to-ascendc-pipeline --cse --target=Ascend910B1 %s | FileCheck %s --check-prefix=ARGS
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
// CHECK-NOT: scf.for
// CHECK: ascendc.cast_extend_strided{{(_reg_base)?}}
// CHECK-NOT: afir.cast
// CHECK-LABEL: func.func @cast_padded_bf16
// CHECK-NOT: scf.for
// CHECK: ascendc.cast_extend_strided{{(_reg_base)?}}
// CHECK-NOT: afir.cast
// MEMCPP-LABEL: void cast_padded_f16
// MEMCPP-NOT: for (
// MEMCPP: CastExtend({{.*}}, static_cast<uint32_t>({{.*}}), static_cast<uint32_t>({{.*}}), static_cast<uint32_t>({{.*}}), static_cast<uint32_t>({{.*}}), 4U);
// REGCPP-LABEL: void cast_padded_f16
// REGCPP-NOT: for (
// REGCPP: CastExtend({{.*}}, {static_cast<uint32_t>({{.*}}), static_cast<uint32_t>({{.*}})}, {static_cast<uint32_t>({{.*}}), 1U}, {static_cast<uint32_t>({{.*}}), 1U});
// FALLBACK-LABEL: func.func @cast_padded_f16
// FALLBACK: scf.for
// FALLBACK: ascendc.cast_extend_contiguous
// ARGS-LABEL: func.func @cast_padded_f16
// ARGS-DAG: %[[ROWS:.*]] = arith.constant 4 : index
// ARGS-DAG: %[[ONE:.*]] = arith.constant 1 : index
// ARGS-DAG: %[[IN:.*]] = arith.constant 16 : index
// ARGS-DAG: %[[OUT:.*]] = arith.constant 8 : index
// ARGS: ascendc.cast_extend_strided{{.*}}%[[ROWS]], %[[ONE]], %[[IN]], %[[OUT]]
