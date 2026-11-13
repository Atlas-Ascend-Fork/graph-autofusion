// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s
// The logical tile is identical, but f16 and f32 rows have different 32B alignment.
// GM symbols are intentionally unbound: selection must only inspect local geometry.
#half = #afir.tensor_desc<axis = [0, 1], repeats = [<"rows">, <"n">],
  strides = [<"gm_row">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"(16 * Ceiling((Rational(1 , 16) * n)))">, <"1">], role = <compute>>
#float = #afir.tensor_desc<axis = [0, 1], repeats = [<"rows">, <"n">],
  strides = [<"gm_row">, <"1">], vectorized_axis = [0, 1],
  vectorized_strides = [<"(8 * Ceiling((Rational(1 , 8) * n)))">, <"1">], role = <compute>>
module {
  func.func @cast_symbolic_pitch(%src: !ascendc.local_tensor<?x?xf16>,
      %dst: !ascendc.local_tensor<?x?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
    %r = afir.cast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#half], afir.destination_descs = [#float], afir.result_descs = [#float]
    } : !ascendc.local_tensor<?x?xf16>, !ascendc.local_tensor<?x?xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?x?xf32>
    return
  }
}
// CHECK-LABEL: func.func @cast_symbolic_pitch
// CHECK: afir.cast
// CHECK-SAME: ascend.impl
