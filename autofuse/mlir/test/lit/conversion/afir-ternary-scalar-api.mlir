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

// 三元族的标量形态（mem 侧）：`select` / `where` 的 then / else，`clip_by_value` 的
// min / max，各自可以是标量，位置不换。
//
// 可用实现是同名重载——`where.h` 的场景 1/2/3、`clipbyvalue.h` 的场景 2/3/4——所以这一类
// 没有新的 AscendC def，只是把后两个操作数的类型约束放宽成「tile 或标量」。三种组合
// （只有 else、只有 then、两个都是）走的是同一个 implementation，因为生成的调用位置相同。
//
// 第一个操作数（mask / x）不能是标量，这一条在 ODS 那一层定死：`where.h` 的三个标量
// 重载的 `mask` 都是 `LocalTensor<uint8_t>`；`clipbyvalue.h` 只有「三个全是标量」那一个
// 重载收标量的 `src_x`，而它没有 count / tmp 形参，verifier 的「至少一个输入有形状」本来
// 就挡掉了全标量。负例在 afir-ternary-scalar-invalid.mlir。
//
// reg 侧在 afir-ternary-scalar-reg-base-api.mlir：那边不收 tmp buffer，图不一样。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @select_scalar_else(%mask: !ascendc.local_tensor<?xui8>, %then: !ascendc.local_tensor<?xf32>,
                              %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 2.000000e+00 : f32
  %result = afir.select %mask, %then, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xf32>, f32),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

func.func @select_scalar_then(%mask: !ascendc.local_tensor<?xui8>, %else: !ascendc.local_tensor<?xf32>,
                              %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 3.000000e+00 : f32
  %result = afir.select %mask, %scalar, %else outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xui8>, f32, !ascendc.local_tensor<?xf32>),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// 场景 1：两个数据输入都是标量，输出形状跟 mask。mask 是 tile，所以 verifier 的
// 「至少一个输入有形状」仍然满足。
func.func @select_both_scalar(%mask: !ascendc.local_tensor<?xui8>,
                              %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %on_true = arith.constant 4.000000e+00 : f32
  %on_false = arith.constant 5.000000e+00 : f32
  %result = afir.select %mask, %on_true, %on_false outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xui8>, f32, f32),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

func.func @where_scalar_else(%mask: !ascendc.local_tensor<?xui8>, %then: !ascendc.local_tensor<?xf16>,
                             %destination: !ascendc.local_tensor<?xf16>, %tmp: !ascendc.local_tensor<?xui8>) {
  %scalar = arith.constant 6.000000e+00 : f16
  %result = afir.where %mask, %then, %scalar outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xf16>, f16),
      !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf16>
  return
}

// clip 最常见的真实形态：上下界都是编译期常量，只有 x 是 tile。
func.func @clip_scalar_bounds(%x: !ascendc.local_tensor<?xf32>,
                              %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %min = arith.constant -1.000000e+00 : f32
  %max = arith.constant 1.000000e+00 : f32
  %result = afir.clip_by_value %x, %min, %max outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, f32, f32),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// 场景 3：min 是 tile、max 是标量。混着来的那一档。
func.func @clip_scalar_max(%x: !ascendc.local_tensor<?xf32>, %min: !ascendc.local_tensor<?xf32>,
                           %destination: !ascendc.local_tensor<?xf32>, %tmp: !ascendc.local_tensor<?xui8>) {
  %max = arith.constant 2.000000e+00 : f32
  %result = afir.clip_by_value %x, %min, %max outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, f32),
      !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// 操作数序原样保留：标量落在它在 AFIR 上的那个位置。
// IR-LABEL: func.func @select_scalar_else
// IR: %[[ELSE_S:.*]] = arith.constant 2.000000e+00 : f32
// IR: ascendc.where_contiguous %arg2, %arg0, %arg1, %[[ELSE_S]]

// IR-LABEL: func.func @select_scalar_then
// IR: %[[THEN_S:.*]] = arith.constant 3.000000e+00 : f32
// IR: ascendc.where_contiguous %arg2, %arg0, %[[THEN_S]], %arg1

// IR-LABEL: func.func @select_both_scalar
// IR: %[[T_S:.*]] = arith.constant 4.000000e+00 : f32
// IR: %[[F_S:.*]] = arith.constant 5.000000e+00 : f32
// IR: ascendc.where_contiguous %arg1, %arg0, %[[T_S]], %[[F_S]]

// IR-LABEL: func.func @clip_scalar_bounds
// IR: ascendc.clipbyvalue_contiguous %arg1, %arg0, %{{.*}}, %{{.*}}

// IR-LABEL: func.func @clip_scalar_max
// IR: ascendc.clipbyvalue_contiguous %arg2, %arg0, %arg1, %{{.*}}

// 裸名、没有模板实参；count 在 tmp buffer 前面（这一类的实参序与 `compare` 那一对
// 相反，见 `OpVecCmpsel.td` 上那段）。
// CPP: {{^ *}}Where({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, c2_f32, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});
// CPP: {{^ *}}Where({{v[0-9]+}}, {{v[0-9]+}}, c3_f32, {{v[0-9]+}}, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});
// CPP: {{^ *}}Where({{v[0-9]+}}, {{v[0-9]+}}, c4_f32, c5_f32, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});
// CPP: {{^ *}}Where({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, c6_f16, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});
// CPP: {{^ *}}ClipByValue({{v[0-9]+}}, {{v[0-9]+}}, cm1_f32, c1_f32, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});
// CPP: {{^ *}}ClipByValue({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, c2_f32, static_cast<uint32_t>({{v[0-9]+}}), {{v[0-9]+}});

// 三种组合同一个 id：标量落在哪个位置不是第二个 implementation。
// SELECT: ascend.implementation = {id = "select.tensor_scalar.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "select.tensor_scalar.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "select.tensor_scalar.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "where.tensor_scalar.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "clip_by_value.tensor_scalar.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "clip_by_value.tensor_scalar.extend", kind = "ascendc_api"}
