// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s
// RUN: sed 's/96xui8/64xui8/g' %s > %t.small.mlir
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %t.small.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/%raw_dst: !ascendc.local_tensor<96xui8>/%raw_dst: !ascendc.local_tensor<64xui8>/; s/%raw_dst\[%offset\] : !ascendc.local_tensor<96xui8>/%raw_dst[%offset] : !ascendc.local_tensor<64xui8>/' %s > %t.small-dst.mlir
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %t.small-dst.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/96xui8/?xui8/g' %s > %t.dynamic.mlir
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %t.dynamic.mlir | FileCheck %s
// REJECT: 'afir.cast' op has no supported Ascend implementation

// Two logical elements occupy different 32-byte blocks. The reinterpret shape
// is logical; capacity comes from the backing bytes minus the byte-view offset.
// Check both static and runtime backing sizes, and reject undersized sources
// and destinations (including the offset and element-width conversion).
#src = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"16">, <"0">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [<"2">, <"1">],
  strides = [<"1">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"8">, <"0">], role = <compute>>
module {
  func.func @cast_logical_f16(%raw_src: !ascendc.local_tensor<96xui8>,
      %raw_dst: !ascendc.local_tensor<96xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %offset = arith.constant 32 : index
    %src_bytes = ascendc.local_tensor.subindex %raw_src[%offset] : !ascendc.local_tensor<96xui8>, index, !ascendc.local_tensor<?xui8>
    %dst_bytes = ascendc.local_tensor.subindex %raw_dst[%offset] : !ascendc.local_tensor<96xui8>, index, !ascendc.local_tensor<?xui8>
    %input = ascendc.reinterpret_cast %src_bytes : !ascendc.local_tensor<?xui8> to !ascendc.local_tensor<2x1xf16>
    %destination = ascendc.reinterpret_cast %dst_bytes : !ascendc.local_tensor<?xui8> to !ascendc.local_tensor<2x1xf32>
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<2x1xf16>, !ascendc.local_tensor<2x1xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<2x1xf32>
    return
  }
  func.func @cast_logical_bf16(%raw_src: !ascendc.local_tensor<96xui8>,
      %raw_dst: !ascendc.local_tensor<96xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %offset = arith.constant 32 : index
    %src_bytes = ascendc.local_tensor.subindex %raw_src[%offset] : !ascendc.local_tensor<96xui8>, index, !ascendc.local_tensor<?xui8>
    %dst_bytes = ascendc.local_tensor.subindex %raw_dst[%offset] : !ascendc.local_tensor<96xui8>, index, !ascendc.local_tensor<?xui8>
    %input = ascendc.reinterpret_cast %src_bytes : !ascendc.local_tensor<?xui8> to !ascendc.local_tensor<2x1xbf16>
    %destination = ascendc.reinterpret_cast %dst_bytes : !ascendc.local_tensor<?xui8> to !ascendc.local_tensor<2x1xf32>
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#src], afir.destination_descs = [#dst], afir.result_descs = [#dst]
    } : !ascendc.local_tensor<2x1xbf16>, !ascendc.local_tensor<2x1xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<2x1xf32>
    return
  }
}
// CHECK-LABEL: func.func @cast_logical_f16
// CHECK-NOT: scf.for
// CHECK-DAG: arith.constant 16 : index
// CHECK-DAG: arith.constant 8 : index
// CHECK-DAG: arith.constant 1 : index
// CHECK: ascendc.cast_extend_strided
// CHECK-NOT: afir.cast
// CHECK-LABEL: func.func @cast_logical_bf16
// CHECK-NOT: scf.for
// CHECK-DAG: arith.constant 16 : index
// CHECK-DAG: arith.constant 8 : index
// CHECK-DAG: arith.constant 1 : index
// CHECK: ascendc.cast_extend_strided
// CHECK-NOT: afir.cast
