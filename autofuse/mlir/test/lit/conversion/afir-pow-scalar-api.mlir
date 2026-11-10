// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950PR_9599 %s | FileCheck %s --check-prefix=SELECT-REG

// `afir.pow` 的标量形态：按 AFIR 原次序生成。
//
// 这是标量形态的第三种生成规则，前两种在 `afir-scalar-binary-api.mlir`（可交换的 8 个，
// tile 恒在第一个输入位）与 `afir-scalar-binary-direction-api.mlir`（`sub` / `div` /
// `truediv`，方向落在 bool 模板实参上）。`pow` 两者都不是：`pow.h` 与 `pow_reg_base.h`
// 各有四个同名重载，(t,t) / (t,s) / (s,t) / (s,s)，标量在哪个位置由 C++ 重载决议挑，
// 所以 legacy 的 `PowApiCall` 逐个位置原样打印、一次都不交换。
//
// 这份用例钉的就是「没有交换」：`pow_scalar_lhs` 那条的实参序必须是
// `Pow(dst, 标量, tile, tmp, count)`。要是照 `Subs` 那一类把 tile 挪到第一位，生成的是
// `tile ^ 标量`——两个都是合法的 C++、都编得过，只是算的不是同一个东西。
//
// 两侧共用一份输入：`pow` 两侧都要 tmp buffer（`CalcPowTmpSize` / `CalcPowTmpSizeV2`），
// 所以图一样。CPP 那一条也两侧通用——生成的调用文本逐字相同，差别只在内联进来的头
// （`pow.h` vs `pow_reg_base.h`），所以 mem / reg 由两条 SELECT 分。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @pow_scalar_rhs(%input: !ascendc.local_tensor<?xf32>,
                          %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f32
  %result = afir.pow %input, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, f32),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

func.func @pow_scalar_lhs(%input: !ascendc.local_tensor<?xf32>,
                          %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 3.000000e+00 : f32
  %result = afir.pow %scalar, %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f32, !ascendc.local_tensor<?xf32>),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// 标量在右：操作数序是 tile 在前。
// IR-LABEL: func.func @pow_scalar_rhs
// IR: %[[RHS_S:.*]] = arith.constant 2.000000e+00 : f32
// IR: ascendc.pow %arg1, %arg0, %[[RHS_S]], %arg2

// 标量在左：操作数序是标量在前。 这一条是本用例的全部理由——换成 `%arg0, %[[LHS_S]]`
// 就是把 `s ^ x` 算成了 `x ^ s`。
// IR-LABEL: func.func @pow_scalar_lhs
// IR: %[[LHS_S:.*]] = arith.constant 3.000000e+00 : f32
// IR: ascendc.pow %arg1, %[[LHS_S]], %arg0, %arg2

// 裸名 `Pow`、没有模板实参（`AscendC::` 里没有这个名字，标准调用叫 `Power`）。
// 字面量写全：emitter 打的是定长科学计数。
// CPP: {{^ *}}Pow({{v[0-9]+}}, {{v[0-9]+}}, c2_f32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}Pow({{v[0-9]+}}, c3_f32, {{v[0-9]+}}, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}));

// 两个方向同一个 id：位置不是第二个 implementation。
// SELECT: ascend.implementation = {id = "pow.tensor_scalar", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "pow.tensor_scalar", kind = "ascendc_api"}

// SELECT-REG: ascend.implementation = {id = "pow.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
// SELECT-REG: ascend.implementation = {id = "pow.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
