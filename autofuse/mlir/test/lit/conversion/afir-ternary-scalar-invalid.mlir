// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s
// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s

// 三元族标量形态的边界：**第一个操作数不能是标量**。
//
// 选择族那一位是 mask（`where.h` 的三个标量重载都要 `LocalTensor<uint8_t>`），clip 族那一位
// 是 x（只有「三个全是标量」那个重载收标量的 `src_x`，而它没有 count / tmp 形参）。放开它
// 只会建出一个编不过的调用，所以 ODS 的 `scalarInputs` 只放宽后两位。
//
// 拦住这两条的是 **ODS 的类型约束**，不是 verifier、也不是 capability——`condition` 那一位
// 的类型仍然只有 `AFIR_BufferizableShapedType`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 一：select 的 mask 是标量。
func.func @select_scalar_mask(%then: !ascendc.local_tensor<?xf32>, %else: !ascendc.local_tensor<?xf32>,
                              %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %mask = arith.constant 1 : i8
  // expected-error @+1 {{operand #0 must be}}
  %result = afir.select %mask, %then, %else outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (i8, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 二：clip 的 x 是标量。三个输入都是标量的那个重载不在这条路上——它没有 count / tmp，
// 而且 verifier 的「至少一个输入有形状」也挡得住。
func.func @clip_scalar_x(%min: !ascendc.local_tensor<?xf32>, %max: !ascendc.local_tensor<?xf32>,
                         %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %x = arith.constant 2.000000e+00 : f32
  // expected-error @+1 {{operand #0 must be}}
  %result = afir.clip_by_value %x, %min, %max outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f32, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}
