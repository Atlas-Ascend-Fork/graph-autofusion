// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: af-opt %s | FileCheck %s

#desc_2d = #afir.tensor_desc<axis = [0, 1], role = #afir.role<compute>>
#buffer = #afir.buffer_desc<space = #afir.space<ub>,>

func.func @logical_broadcast(%input: tensor<1x8xf16>) -> tensor<4x8xf16> {
  %result = afir.broadcast %input : tensor<1x8xf16> -> tensor<4x8xf16>
  return %result : tensor<4x8xf16>
}

// CHECK-LABEL: func.func @logical_broadcast
// CHECK: afir.broadcast %arg0 : tensor<1x8xf16> -> tensor<4x8xf16>

func.func @buffer_broadcast(%input: memref<1x8xf16> {afir.buffer_desc = #buffer}, %output: memref<4x8xf16> {afir.buffer_desc = #buffer}) attributes {
  afir.arg_descs = [#desc_2d, #desc_2d]
} {
  %result = afir.broadcast %input outs(%output) {
    afir.operand_descs = [#desc_2d],
    afir.result_descs = [#desc_2d],
    afir.destination_descs = [#desc_2d]
  } :
    memref<1x8xf16>, memref<4x8xf16> -> memref<4x8xf16>
  return
}

// CHECK-LABEL: func.func @buffer_broadcast
// CHECK: afir.broadcast %arg0 outs(%arg1)
// CHECK-SAME: : memref<1x8xf16>, memref<4x8xf16> -> memref<4x8xf16>
