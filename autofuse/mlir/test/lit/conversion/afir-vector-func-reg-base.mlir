// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-vector-func-to-reg-base %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-vector-func-to-reg-base %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

// `afir.vector_func` 的下降（M1）。它钉三件事：
//
//   1. 边界语义：region 内只有计算，UB <-> 寄存器的搬运是下降生成的，不是从子图
//      翻过来的。所以产物里 `LoadAlign` 恰好等于 src 数、`StoreAlign` 恰好等于 dst 数——
//      计数式而不是「存在一条」：少生成一条 Load 也能编过、也能算对（上一趟的寄存器还在），
//      要到别的形状上才暴露。
//   2. 产物形状：一个 `__VEC_SCOPE__`、里面恰好一层 lane 循环，寄存器声明在循环外。
//   3. 算子顺序：`add` 在 `abs` 之前。顺序错了结果就是另一个函数，而两种都编得过。
//
// 不在这里钉的：形状相关的判据（lane 一段多少个元素由目标的 `GetVecLen()` 定，这里是
// 符号尺寸）、以及能力不足的四条（它们要看诊断，在 `afir-vector-func-unsupported.mlir`）。

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @vector_func_add_abs(%a: !ascendc.local_tensor<?xf32>, %b: !ascendc.local_tensor<?xf32>,
                                 %y: !ascendc.local_tensor<?xf32>) {
    %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
      afir.operand_descs = [#in, #in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xf32>, %lb: tensor<?xf32>):
      %sum = afir.add %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
          : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      %res = afir.abs %sum {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xf32> -> tensor<?xf32>
      afir.vector_func_yield %res : tensor<?xf32>
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>)
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // `ln` / `sqrt` 两行，同 dtype 的正向：边界与中间结果同为 f32，不经过 Cast。跨 dtype 的
  // 那一格（bf16 进、f32 算、bf16 出）在 `afir-vector-func-reg-base-cast.mlir`。
  func.func @vector_func_ln_sqrt_f32(%a: !ascendc.local_tensor<?xf32>, %y: !ascendc.local_tensor<?xf32>) {
    %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
      afir.operand_descs = [#in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xf32>):
      %ln = afir.ln %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
          : tensor<?xf32> -> tensor<?xf32>
      %sqrt = afir.sqrt %ln {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xf32> -> tensor<?xf32>
      afir.vector_func_yield %sqrt : tensor<?xf32>
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>) -> !ascendc.local_tensor<?xf32>
    return
  }

  // 同上，half。寄存器形态的 `Ln` / `Sqrt` 收 half 与 float 两种，half 这一档的 lane 宽度
  // 与取存类型都要跟着边界走。
  func.func @vector_func_ln_sqrt_f16(%a: !ascendc.local_tensor<?xf16>, %y: !ascendc.local_tensor<?xf16>) {
    %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
      afir.operand_descs = [#in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xf16>):
      %ln = afir.ln %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
          : tensor<?xf16> -> tensor<?xf16>
      %sqrt = afir.sqrt %ln {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xf16> -> tensor<?xf16>
      afir.vector_func_yield %sqrt : tensor<?xf16>
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>) -> !ascendc.local_tensor<?xf16>
    return
  }

  // int8 的 VF 边界，同 dtype：边界与中间结果同为 i8，lane 按 1 字节算，取与存不带 dist。
  // 跨 dtype 的那一格（i8 进、half 算、i8 出）在 `afir-vector-func-reg-base-cast.mlir`。
  func.func @vector_func_abs_i8(%a: !ascendc.local_tensor<?xi8>, %y: !ascendc.local_tensor<?xi8>) {
    %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
      afir.operand_descs = [#in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xi8>):
      %res = afir.abs %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
          : tensor<?xi8> -> tensor<?xi8>
      afir.vector_func_yield %res : tensor<?xi8>
    } : (!ascendc.local_tensor<?xi8>, !ascendc.local_tensor<?xi8>) -> !ascendc.local_tensor<?xi8>
    return
  }
  // dtype 域的另一半：这些类型仍然放行。算子表里每一行只收一部分元素类型（见
  // `AFIRVectorFuncToRegBase.cpp` 的 `kRegOp*ElementTypes`），掩码写窄一格不会有任何东西报错，
  // 只是接管范围悄悄变小。所以把只收一部分类型的那几行，按还收的类型各走一格；拒收的那一侧在
  // `afir-vector-func-unsupported.mlir` 第八节。四格都是同 dtype、等宽，取与存不带 dist。

  // int16：`neg` -> `abs` -> `mul`，三行串成一格。
  func.func @vector_func_neg_abs_mul_i16(%a: !ascendc.local_tensor<?xi16>, %b: !ascendc.local_tensor<?xi16>,
                                         %y: !ascendc.local_tensor<?xi16>) {
    %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
      afir.operand_descs = [#in, #in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xi16>, %lb: tensor<?xi16>):
      %neg = afir.neg %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
          : tensor<?xi16> -> tensor<?xi16>
      %abs = afir.abs %neg {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xi16> -> tensor<?xi16>
      %res = afir.mul %abs, %lb {afir.operand_descs = [#out, #in], afir.result_descs = [#out]}
          : (tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
      afir.vector_func_yield %res : tensor<?xi16>
    } : (!ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>)
        -> !ascendc.local_tensor<?xi16>
    return
  }

  // bf16：`mul` 收，而 `abs` / `neg` 不收，所以 `mul` 单独一格。
  func.func @vector_func_mul_bf16(%a: !ascendc.local_tensor<?xbf16>, %b: !ascendc.local_tensor<?xbf16>,
                                  %y: !ascendc.local_tensor<?xbf16>) {
    %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
      afir.operand_descs = [#in, #in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xbf16>, %lb: tensor<?xbf16>):
      %res = afir.mul %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
          : (tensor<?xbf16>, tensor<?xbf16>) -> tensor<?xbf16>
      afir.vector_func_yield %res : tensor<?xbf16>
    } : (!ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>, !ascendc.local_tensor<?xbf16>)
        -> !ascendc.local_tensor<?xbf16>
    return
  }

  // int32：`relu` -> `abs` -> `neg`。
  func.func @vector_func_relu_abs_neg_i32(%a: !ascendc.local_tensor<?xi32>, %y: !ascendc.local_tensor<?xi32>) {
    %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
      afir.operand_descs = [#in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xi32>):
      %relu = afir.relu %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
          : tensor<?xi32> -> tensor<?xi32>
      %abs = afir.abs %relu {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xi32> -> tensor<?xi32>
      %res = afir.neg %abs {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xi32> -> tensor<?xi32>
      afir.vector_func_yield %res : tensor<?xi32>
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>) -> !ascendc.local_tensor<?xi32>
    return
  }

  // int8：`neg`（`abs` 的 int8 那一格在上面）。
  func.func @vector_func_neg_i8(%a: !ascendc.local_tensor<?xi8>, %y: !ascendc.local_tensor<?xi8>) {
    %0 = afir.vector_func srcs(%a) dsts(%y) attributes {
      afir.operand_descs = [#in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xi8>):
      %res = afir.neg %la {afir.operand_descs = [#in], afir.result_descs = [#out]}
          : tensor<?xi8> -> tensor<?xi8>
      afir.vector_func_yield %res : tensor<?xi8>
    } : (!ascendc.local_tensor<?xi8>, !ascendc.local_tensor<?xi8>) -> !ascendc.local_tensor<?xi8>
    return
  }
}

// 作用域外：三个 UB 首地址、一个元素数、一段 lane 的元素数、循环次数、以及 `UpdateMask`
// 要的那个可变计数器。
// IR-LABEL: func.func @vector_func_add_abs
// IR: ascendc.local_tensor.get_phy_addr_v2
// IR: ascendc.local_tensor.get_size
// IR: ascendc.get_vec_len
// IR: emitasc.variable

// 寄存器声明在作用域里、循环外：src 两个加中间结果两个。循环里重复声明在 C++ 上也合法，
// 但那样每趟都多一条，与 legacy 的产物对不起来。
// IR: emitasc.vec_scope
// IR-COUNT-4: ascendc.reg_tensor
// IR-NOT: ascendc.reg_tensor

// 恰好一层 lane 循环。两层意味着有人把 tile 循环也生成进了 region，而那样每个元素会被
// 算两遍——结果仍然对（幂等的那几个 op 上尤其），所以只能靠计数钉住。
// IR-COUNT-1: emitasc.vf_for
// IR-NOT: emitasc.vf_for

// 循环体：取两次、算两步、存一次，顺序固定。
// IR: ascendc.update_mask
// IR-COUNT-2: ascendc.load_align_reg
// IR-NOT: ascendc.load_align_reg
// IR: ascendc.add_reg
// IR: ascendc.abs_reg
// IR-COUNT-1: ascendc.store_align_reg
// IR-NOT: ascendc.store_align_reg

// region 没有了，GM <-> UB 的搬运一条也不该被生出来：子图边界的 `Load` / `Store` 表达的是
// UB <-> 寄存器，翻成 `afir.load` / `afir.store` 会多出两趟搬运，而那样也编得过、也算得对。
// IR-NOT: afir.vector_func
// IR-NOT: afir.load
// IR-NOT: afir.store

// 生成的是 `__VEC_SCOPE__` 加一层 `for`，调的是 `AscendC::Reg::` 那一类（legacy 写
// `MicroAPI::`，`namespace MicroAPI = Reg`，同一个函数）。
// CPP-LABEL: void vector_func_add_abs(
// CPP: __VEC_SCOPE__
// CPP: for (uint16_t
// CPP: AscendC::Reg::UpdateMask<float>(
// CPP-COUNT-2: AscendC::Reg::LoadAlign<float>(
// CPP: AscendC::Reg::Add(
// CPP: AscendC::Reg::Abs(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<float>(

// `ln` -> `sqrt`，f32：顺序固定，取一次、算两步、存一次，类型全程 f32。
// IR-LABEL: func.func @vector_func_ln_sqrt_f32
// IR-COUNT-1: emitasc.vf_for
// IR: ascendc.update_mask f32
// IR-COUNT-1: ascendc.load_align_reg
// IR: ascendc.ln_reg
// IR: ascendc.sqrt_reg
// IR-COUNT-1: ascendc.store_align_reg
// CPP-LABEL: void vector_func_ln_sqrt_f32(
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::UpdateMask<float>(
// CPP-COUNT-1: AscendC::Reg::LoadAlign<float>(
// CPP: AscendC::Reg::Ln(
// CPP: AscendC::Reg::Sqrt(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<float>(

// 同上，half：lane 宽度与取存类型都跟着 half 的边界走。
// IR-LABEL: func.func @vector_func_ln_sqrt_f16
// IR: ascendc.update_mask f16
// IR: ascendc.ln_reg
// IR: ascendc.sqrt_reg
// CPP-LABEL: void vector_func_ln_sqrt_f16(
// CPP: AscendC::Reg::UpdateMask<half>(
// CPP-COUNT-1: AscendC::Reg::LoadAlign<half>(
// CPP: AscendC::Reg::Ln(
// CPP: AscendC::Reg::Sqrt(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<half>(

// int8 同 dtype：lane 按 1 字节算，等宽的取与存不带 dist（缺席即 `DIST_NORM`）。
// IR-LABEL: func.func @vector_func_abs_i8
// IR: ascendc.update_mask i8
// IR: ascendc.load_align_reg
// IR-NOT: dist =
// IR: ascendc.abs_reg
// IR-COUNT-1: ascendc.store_align_reg
// CPP-LABEL: void vector_func_abs_i8(
// CPP: AscendC::Reg::UpdateMask<int8_t>(
// CPP-COUNT-1: AscendC::Reg::LoadAlign<int8_t>(
// CPP: AscendC::Reg::Abs(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<int8_t>(

// int16：三行都收 int16，类型全程 int16，等宽的取与存不带 dist。
// IR-LABEL: func.func @vector_func_neg_abs_mul_i16
// IR: ascendc.update_mask i16
// IR-COUNT-2: ascendc.load_align_reg
// IR-NOT: dist =
// IR: ascendc.neg_reg
// IR: ascendc.abs_reg
// IR: ascendc.mul_reg
// IR-COUNT-1: ascendc.store_align_reg
// CPP-LABEL: void vector_func_neg_abs_mul_i16(
// CPP: __VEC_SCOPE__
// CPP: AscendC::Reg::UpdateMask<int16_t>(
// CPP-COUNT-2: AscendC::Reg::LoadAlign<int16_t>(
// CPP: AscendC::Reg::Neg(
// CPP: AscendC::Reg::Abs(
// CPP: AscendC::Reg::Mul(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<int16_t>(

// bf16 的 `mul`：lane 宽度与取存类型都跟着 bf16 走，C++ 那一侧是 `bfloat16_t`。
// IR-LABEL: func.func @vector_func_mul_bf16
// IR: ascendc.update_mask bf16
// IR-COUNT-2: ascendc.load_align_reg
// IR-NOT: dist =
// IR: ascendc.mul_reg
// IR-COUNT-1: ascendc.store_align_reg
// CPP-LABEL: void vector_func_mul_bf16(
// CPP: AscendC::Reg::UpdateMask<bfloat16_t>(
// CPP-COUNT-2: AscendC::Reg::LoadAlign<bfloat16_t>(
// CPP: AscendC::Reg::Mul(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<bfloat16_t>(

// int32：`relu` 只收浮点与 32 / 64 位整数，int32 在其中。
// IR-LABEL: func.func @vector_func_relu_abs_neg_i32
// IR: ascendc.update_mask i32
// IR-COUNT-1: ascendc.load_align_reg
// IR-NOT: dist =
// IR: ascendc.relu_reg
// IR: ascendc.abs_reg
// IR: ascendc.neg_reg
// IR-COUNT-1: ascendc.store_align_reg
// CPP-LABEL: void vector_func_relu_abs_neg_i32(
// CPP: AscendC::Reg::UpdateMask<int32_t>(
// CPP-COUNT-1: AscendC::Reg::LoadAlign<int32_t>(
// CPP: AscendC::Reg::Relu(
// CPP: AscendC::Reg::Abs(
// CPP: AscendC::Reg::Neg(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<int32_t>(

// int8 的 `neg`：与上面 `abs` 的 int8 同一个 dtype 域（浮点与带符号整数）。
// IR-LABEL: func.func @vector_func_neg_i8
// IR: ascendc.update_mask i8
// IR-COUNT-1: ascendc.load_align_reg
// IR-NOT: dist =
// IR: ascendc.neg_reg
// IR-COUNT-1: ascendc.store_align_reg
// CPP-LABEL: void vector_func_neg_i8(
// CPP: AscendC::Reg::UpdateMask<int8_t>(
// CPP-COUNT-1: AscendC::Reg::LoadAlign<int8_t>(
// CPP: AscendC::Reg::Neg(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<int8_t>(
