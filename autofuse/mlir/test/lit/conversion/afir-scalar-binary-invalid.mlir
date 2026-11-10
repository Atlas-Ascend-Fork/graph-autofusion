// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s
// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s

// ODS 的 `scalarOperand` 说哪个操作数可以是标量（`"none"` / `"rhs"` / `"any"`），被指名
// 的位置约束从「一块 tile」变成「一块 tile 或者一个标量」。这份文件按谁拦住的分组：
// ODS、verifier、capability 各有。
//
// 分清楚是必要的：`afir-scalar-binary-api.mlir` 里那几个 op 的 `sameElementType` 恒开，
// 于是「标量类型与 tile 元素类型不符」这一条在它们身上到不了 capability——ODS 先
// 报了。真正驱动那条 capability 判据的是下面第三条：一个既不是 local tensor、也不是
// 标量的右操作数。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 一：标量的类型必须与 tile 的元素类型一致。`Adds(LocalTensor<half>, half)` 没有收
// float 的重载。拦住它的是 ODS 的 `AllElementTypesMatch`（它比的是
// `getElementTypeOrSelf`，标量的「元素类型」就是它自己），所以这是一条 **ODS 负例**，
// op 根本到不了 capability。
func.func @add_scalar_element_type_mismatch(%input: !ascendc.local_tensor<?xf16>,
                                            %destination: !ascendc.local_tensor<?xf16>) {
  %scalar = arith.constant 2.000000e+00 : f32
  // expected-error @+1 {{failed to verify that all of {lhs, rhs, result} have same element type}}
  %result = afir.add %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, f32), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 二：目的地的形状仍然要跟着 tile 走。标量不参与广播，于是这条判据全靠 tile 那一侧——
// 写错的表现是 `4` 对 `8` 也放行。
//
// 报这条的是 capability 不是 verifier：`verifyBufferizedElementwise` 的形状检查只对
// memref 目的地生效，`!ascendc.local_tensor` 那一路它整段跳过。
func.func @add_scalar_destination_shape_mismatch(%input: !ascendc.local_tensor<4xf16>,
                                                 %destination: !ascendc.local_tensor<8xf16>) {
  %scalar = arith.constant 2.000000e+00 : f16
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.add %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<4xf16>, f16), !ascendc.local_tensor<8xf16> -> !ascendc.local_tensor<8xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>
#ub_buffer = #afir.buffer_desc<space = <ub>, buffer_id = 0, life_time_axis_id = -1>
#gm_buffer = #afir.buffer_desc<space = <gm>, life_time_axis_id = -1>

// 三：右操作数既不是 local tensor、也不是标量。
//
// `getTensorTypes` 按「rhs 能不能 dyn_cast 成 LocalTensorType」二分，于是这种值会被归到
// 标量那一边，再由「标量类型 == tile 元素类型」拒掉——`!ascendc.global_tensor<?xf16>`
// 不等于 `f16`。这条就是那句判据唯一到得了的入口：上面第一条被 ODS 先拦了，而
// `sameElementType` 恒开的情况下真标量走不到这里。删掉那一行，这份 IR 就会被当成一个
// 合法的 `Adds`，把一块 GM 张量当成一个标量值传下去。
func.func @add_global_tensor_rhs(%lhs: !ascendc.local_tensor<?xf16> {afir.buffer_desc = #ub_buffer},
                                 %rhs: !ascendc.global_tensor<?xf16> {afir.buffer_desc = #gm_buffer},
                                 %destination: !ascendc.local_tensor<?xf16> {afir.buffer_desc = #ub_buffer}) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.add %lhs, %rhs outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, !ascendc.global_tensor<?xf16>), !ascendc.local_tensor<?xf16>
      -> !ascendc.local_tensor<?xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 四：两个操作数不能都是标量。`scalarOperand = "any"` 的 op 两个位置各自都可以是
// 标量，而 ODS 的类型约束是逐个操作数的，说不出「至多一个」——所以这一条是 **verifier
// 负例**，不是 ODS 负例。
//
// 挡住的不是一个假想形态：两个都是 `f16` 时上面那两条 ODS 约束全过
// （`AllElementTypesMatch` 只看元素类型，而标量的元素类型就是它自己），而
// `inferElementwiseShape` 会答「没有形状」，目的地的形状无从校验，生成的调用把两个
// 值当成两块 tile。legacy 那边这个形态是另一条路（`binary_api_call.cpp` 里
// `x1.IsAnyScalar() && x2.IsAnyScalar()` 生成的是 `Adds(y, (T)v1, (T)v2)`，连 tile 都没有），
// 这一层没有接。
func.func @add_both_operands_scalar(%destination: !ascendc.local_tensor<?xf16>) {
  %lhs = arith.constant 2.000000e+00 : f16
  %rhs = arith.constant 3.000000e+00 : f16
  // expected-error @+1 {{buffer form expects at least one shaped input}}
  %result = afir.add %lhs, %rhs outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f16, f16), !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 五：不可交换的那四个比较 op，标量仍然只能在右。 `lt` / `le` / `gt` / `ge` 的
// `scalarOperand` 是 `"rhs"`，交换过来 ODS 直接拒收——左操作数还钉在 shaped type 上。
//
// 这一条与 afir-scalar-binary-swapped-api.mlir 成对：那一份是 8 个可交换 op 的交换方向
// 都能下降，这一份是「可交换」这个前提没有被顺手放宽成全部。legacy 也只给 Eq / Ne
// 覆写了 `IsScalarInputSupportedIfExchangeInputs`，四个不等号一个都没有。
func.func @lt_scalar_first_operand(%input: !ascendc.local_tensor<?xf16>,
                                   %destination: !ascendc.local_tensor<?xui8>,
                                   %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f16
  // expected-error @+1 {{operand #0 must be Tensor or MemRef of}}
  %result = afir.lt %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f16, !ascendc.local_tensor<?xf16>),
      !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}
