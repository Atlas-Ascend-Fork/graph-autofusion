// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// 第 9 批接通的八个多项式族。**这一份真正要钉的是 `n` 落在哪个位置**——模板实参位，
// 不是实参位：
//
//     ChebyshevPolynomialTExtend<float, 3>(y, x, tmp, size)
//
// 挑错落点的两种失败形态各有一条 CHECK 盯着：
//
//   - 当成实参（照 `afir.leaky_relu` 的 slope 办）→ 会现建一个 `arith.constant`，
//     调用变成五个实参、模板位只剩 dtype。`IR-NOT: arith.constant` 与 CPP 那条
//     `<float, N>` 的字面量一起挡它。
//   - 把 n 丢掉 → 模板位只有 dtype，包装那边 `N` 推不出来。CPP 的 `, N>` 挡它。
//
// 八个各取一个**不同的 n**，而且刻意覆盖包装里那几条 `if constexpr` 的边界
// （`N < 0` / `N == 0` / `N == 1` / 其余走 `VF_CALL`）：抄同一个 n 八遍的话，
// 「n 是不是真的逐 op 读出来的」这件事钉不住——八条 CHECK 会互相替对方过。
// 其中 `shifted_chebyshev_polynomial_v` 取 -1，那是包装里 `Duplicate(dst, 0)` 那一支。
//
// dtype 只有 f32：八个的 legacy `REG_ASC_IR` 都是 `TensorType{DT_FLOAT}`，包装体里还有
// 一条 `static_assert(SupportType<T, float>())`。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @chebyshev_polynomial_t_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.chebyshev_polynomial_t %input outs(%destination) tmp(%tmp) {
      n = 3 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @chebyshev_polynomial_u_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.chebyshev_polynomial_u %input outs(%destination) tmp(%tmp) {
      n = 4 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @chebyshev_polynomial_v_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.chebyshev_polynomial_v %input outs(%destination) tmp(%tmp) {
      n = 0 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @chebyshev_polynomial_w_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.chebyshev_polynomial_w %input outs(%destination) tmp(%tmp) {
      n = 1 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @shifted_chebyshev_polynomial_t_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.shifted_chebyshev_polynomial_t %input outs(%destination) tmp(%tmp) {
      n = 2 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @shifted_chebyshev_polynomial_u_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.shifted_chebyshev_polynomial_u %input outs(%destination) tmp(%tmp) {
      n = 5 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @shifted_chebyshev_polynomial_v_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.shifted_chebyshev_polynomial_v %input outs(%destination) tmp(%tmp) {
      n = -1 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @shifted_chebyshev_polynomial_w_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.shifted_chebyshev_polynomial_w %input outs(%destination) tmp(%tmp) {
      n = 7 : i64,
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 建出来的 AscendC op 带四个实参加一个 `n` 属性；**没有** `arith.constant`。
// IR-LABEL: func.func @chebyshev_polynomial_t_f32
// IR: ascendc.chebyshev_polynomial_t_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = 3
// IR-NOT: afir.chebyshev_polynomial_t

// IR-LABEL: func.func @chebyshev_polynomial_u_f32
// IR: ascendc.chebyshev_polynomial_u_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = 4
// IR-NOT: afir.chebyshev_polynomial_u

// IR-LABEL: func.func @chebyshev_polynomial_v_f32
// IR: ascendc.chebyshev_polynomial_v_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = 0
// IR-NOT: afir.chebyshev_polynomial_v

// IR-LABEL: func.func @chebyshev_polynomial_w_f32
// IR: ascendc.chebyshev_polynomial_w_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = 1
// IR-NOT: afir.chebyshev_polynomial_w

// IR-LABEL: func.func @shifted_chebyshev_polynomial_t_f32
// IR: ascendc.shifted_chebyshev_polynomial_t_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = 2
// IR-NOT: afir.shifted_chebyshev_polynomial_t

// IR-LABEL: func.func @shifted_chebyshev_polynomial_u_f32
// IR: ascendc.shifted_chebyshev_polynomial_u_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = 5
// IR-NOT: afir.shifted_chebyshev_polynomial_u

// IR-LABEL: func.func @shifted_chebyshev_polynomial_v_f32
// IR: ascendc.shifted_chebyshev_polynomial_v_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = -1
// IR-NOT: afir.shifted_chebyshev_polynomial_v

// IR-LABEL: func.func @shifted_chebyshev_polynomial_w_f32
// IR: ascendc.shifted_chebyshev_polynomial_w_extend_reg_base %arg1, %arg0, %arg2
// IR-SAME: n = 7
// IR-NOT: afir.shifted_chebyshev_polynomial_w

// 整个模块里一个常量都不该有：`n` 是属性，不是现建的 SSA 值。
// IR-NOT: arith.constant

// CPP-LABEL: void chebyshev_polynomial_t_f32(
// CPP: {{^ *}}ChebyshevPolynomialTExtend<float, 3>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void chebyshev_polynomial_u_f32(
// CPP: {{^ *}}ChebyshevPolynomialUExtend<float, 4>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void chebyshev_polynomial_v_f32(
// CPP: {{^ *}}ChebyshevPolynomialVExtend<float, 0>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void chebyshev_polynomial_w_f32(
// CPP: {{^ *}}ChebyshevPolynomialWExtend<float, 1>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void shifted_chebyshev_polynomial_t_f32(
// CPP: {{^ *}}ShiftedChebyshevPolynomialTExtend<float, 2>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void shifted_chebyshev_polynomial_u_f32(
// CPP: {{^ *}}ShiftedChebyshevPolynomialUExtend<float, 5>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void shifted_chebyshev_polynomial_v_f32(
// CPP: {{^ *}}ShiftedChebyshevPolynomialVExtend<float, -1>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void shifted_chebyshev_polynomial_w_f32(
// CPP: {{^ *}}ShiftedChebyshevPolynomialWExtend<float, 7>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "chebyshev_polynomial_t.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "chebyshev_polynomial_u.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "chebyshev_polynomial_v.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "chebyshev_polynomial_w.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "shifted_chebyshev_polynomial_t.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "shifted_chebyshev_polynomial_u.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "shifted_chebyshev_polynomial_v.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "shifted_chebyshev_polynomial_w.tensor.extend_reg_base", kind = "ascendc_api"}

// 八个都只有 reg 侧的 implementation，mem 目标上一条都选不出来。
// WRONGTARGET: no supported
