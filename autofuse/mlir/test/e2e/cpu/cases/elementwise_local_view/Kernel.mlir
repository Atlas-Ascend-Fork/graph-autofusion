// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#dense = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"gm_outer">, <"gm_row">, <"gm_col">],
  vectorized_axis = [1, 2], vectorized_strides = [<"8">, <"1">], role = <compute>>
#pitched = #afir.tensor_desc<axis = [0, 1, 2], repeats = [<"2">, <"4">, <"8">],
  strides = [<"gm_outer">, <"gm_row">, <"gm_col">],
  vectorized_axis = [1, 2], vectorized_strides = [<"16">, <"1">], role = <compute>>

module {
  func.func @abs_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>, %tmp: !ascendc.local_tensor<128xui8>) {
    %r = afir.abs %a outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#dense], afir.destination_descs = [#pitched], afir.result_descs = [#pitched]
    } : !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<128xui8> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @add_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>) {
    %r = afir.add %a, %b outs(%dst) {
      afir.operand_descs = [#dense, #pitched], afir.destination_descs = [#pitched], afir.result_descs = [#pitched]
    } : (!ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>), !ascendc.local_tensor<2x4x8xf32> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }
  func.func @clip_different_ub(%a: !ascendc.local_tensor<2x4x8xf32>, %b: !ascendc.local_tensor<2x4x8xf32>, %c: !ascendc.local_tensor<2x4x8xf32>, %dst: !ascendc.local_tensor<2x4x8xf32>, %tmp: !ascendc.local_tensor<128xui8>) {
    %r = afir.clip_by_value %a, %b, %c outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#dense, #pitched, #dense], afir.destination_descs = [#pitched], afir.result_descs = [#pitched]
    } : (!ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<2x4x8xf32>), !ascendc.local_tensor<2x4x8xf32>, !ascendc.local_tensor<128xui8> -> !ascendc.local_tensor<2x4x8xf32>
    return
  }

}
