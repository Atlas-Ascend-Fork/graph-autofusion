// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s

// `afir.broadcast` 的源这一位收裸标量（legacy `BroadcastScalar` 的 `in.is_constant`
// 一支）之后，哪些事实还被挡着。一格只违反一条判据，按谁拦住的分组：ODS、verifier 各有。
//
// 放开的只有「源的形状」这一项。元素类型、ODS 的类型表、logical 形态不收标量，三条都在。

#scalar_src = #afir.tensor_desc<role = <compute>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 一：标量的类型必须与结果的元素类型一致。广播不改 dtype，`verifyBroadcastShapeImpl`
// 对标量源照旧比这一条（`getElementTypeOrSelf` 对标量给的就是它自己）。
// 写错的表现是 `Duplicate(LocalTensor<half>, float, n)`——没有这条重载。
func.func @broadcast_scalar_element_type_mismatch(%destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant 2.000000e+00 : f32
  // expected-error @+1 {{broadcast input and result must have the same element type}}
  %result = afir.broadcast %scalar outs(%destination) {
    afir.operand_descs = [#scalar_src],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : f32, !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

// -----

#scalar_src = #afir.tensor_desc<role = <compute>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 二：ODS 收的是「元素类型在 `AFIR_BroadcastElementTypes` 表里的标量」，不是「任何标量」。
// `i64` 不在那张表里（表里有 `ui64`、没有 `i64`），所以这一条由 ODS 拦。
// 这一格钉的是放开的范围：把约束写成 `AnyType` 的话它会全绿。
func.func @broadcast_scalar_element_type_not_in_table(%destination: !ascendc.local_tensor<?xi64>) {
  %scalar = arith.constant 2 : i64
  // expected-error @+1 {{op operand #0 must be}}
  %result = afir.broadcast %scalar outs(%destination) {
    afir.operand_descs = [#scalar_src],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : i64, !ascendc.local_tensor<?xi64> -> !ascendc.local_tensor<?xi64>
  return
}

// -----

// 三：logical（tensor）形态照旧不收标量。bridge 对 ASCIR `Scalar` 节点也只在 buffer
// 形态下建值（`AfirBuilder.cpp` 的 `handleMarkerNode`），两边说的是同一件事。
func.func @broadcast_scalar_logical_form() -> tensor<128xf32> {
  %scalar = arith.constant 2.000000e+00 : f32
  // expected-error @+1 {{logical form expects tensor input and result}}
  %result = afir.broadcast %scalar : f32 -> tensor<128xf32>
  return %result : tensor<128xf32>
}
