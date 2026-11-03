// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// 比较族的 mem 侧。六个 op 建的是**同一个** `ascendc.compare_extend_contiguous`，
// 只有 `cmpMode` 不同——这是这条路上第一次一个 AscendC def 服务六个 AFIR op，所以
// 这份用例的第一件事是钉住「那个属性真的跟着 AFIR op 走」：三个 op 各一个函数，
// IR 上的 `cmpMode` 必须是 0 / 2 / 4（LT / EQ / GE，见 `AscendC_CMPMODEAttr` 的编号）。
//
// 第二件事是**实参次序**：`CompareExtend(dst, s0, s1, CMPMODE::XX, size, tmp)` 里
// tmp 排在 size **之后**，与这条路上其余每一个包装都相反。次序写反了编译器不会报
// ——两个都是 `LocalTensor`——只有 CPP 那条 check 拦得住。
//
// 结果 dtype 恒为 `ui8`、与输入无关，这一条与 `logical_and` 同形，那边已经钉过，
// 这里只在函数签名上体现（输入 f32 / i32 / f16，目的地一律 ui8）。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @eq_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                    %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.eq %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  // i32 是 mem 侧收的第三个 dtype，reg 侧还多三个——`lt` 在 legacy 的注册里比其余五个
  // 少一个 i64，那一个 AFIR 这一层本来就建模不出来，所以六个 op 在这里收的是同一组。
  func.func @lt_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                    %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.lt %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  // 标量形态：右操作数是一个标量 SSA 值，建的是另一个 def（`CompareScalarExtend`）。
  func.func @ge_scalar_f16(%input: !ascendc.local_tensor<?xf16>,
                           %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %scalar = arith.constant 1.500000e+00 : f16
    %result = afir.ge %input, %scalar outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, f16),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }
}

// tmp 是最后一个实参，count 在它前面。
// IR-LABEL: func.func @eq_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.compare_extend_contiguous %arg2, %arg0, %arg1, %[[COUNT]], %arg3
// IR-SAME: cmpMode = 2 : i64
// IR-NOT: ascendc.compare_extend_contiguous_reg_base
// IR-NOT: afir.eq

// IR-LABEL: func.func @lt_i32
// IR: ascendc.compare_extend_contiguous
// IR-SAME: cmpMode = 0 : i64
// IR-NOT: afir.lt

// 标量不取块内下标——它在整条循环里就是同一个值。
// IR-LABEL: func.func @ge_scalar_f16
// IR: %[[SCALAR:.*]] = arith.constant 1.500000e+00 : f16
// IR: ascendc.compare_scalar_extend_contiguous %arg1, %arg0, %[[SCALAR]]
// IR-SAME: cmpMode = 4 : i64
// IR-NOT: afir.ge

// 裸名、不带 `AscendC::` 限定，不打模板实参（`InT` / `OutT` 两个都推得出来），
// `CMPMODE` 在第四个实参位，tmp 在 count 之后。
// CPP-LABEL: void eq_f32(
// CPP: {{^ *}}CompareExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, CMPMODE::EQ, static_cast<uint32_t>({{[^,]*}}), {{[^,]*}});
// CPP-LABEL: void lt_i32(
// CPP: {{^ *}}CompareExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, CMPMODE::LT, static_cast<uint32_t>({{[^,]*}}), {{[^,]*}});
// CPP-LABEL: void ge_scalar_f16(
// CPP: {{^ *}}CompareScalarExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, CMPMODE::GE, static_cast<uint32_t>({{[^,]*}}), {{[^,]*}});

// SELECT: ascend.implementation = {id = "eq.tensor_tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "lt.tensor_tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "ge.tensor_scalar", kind = "ascendc_api"}
