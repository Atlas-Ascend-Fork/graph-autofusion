// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950PR_9599 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950PR_9599 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950PR_9599 %s | FileCheck %s --check-prefix=SELECT

// reg 侧的标量形态**只有 `div` 一个**。`sub_reg_base.h` 里 `SubExtends` 也在、
// `true_div.h` 的 `TrueDivExtends` 也在，但 legacy v2 的 `IsScalarInputSupported` 对
// Sub / TrueDiv 返回 false（矩阵的 `reg_scalar` 列就是那个判据的摘要），调度产不出那种
// 图——接了是死代码，所以这条路上没有。
//
// **与 mem 侧那份是两个文件而不是两条 RUN**：`DivExtends` 不收 tmp buffer（`div_reg_base.h`
// 自己在寄存器里转存），而 `Divs` 收。图上带不带 tmp 操作数不一样，一份输入服务不了两侧
// ——`afir-logical-scalar-api.mlir` 与它的 `_reg_base` 那一份同一个理由。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @div_scalar_rhs(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
  %scalar = arith.constant 4.000000e+00 : f32
  %result = afir.div %input, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, f32), !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
  return
}

func.func @div_scalar_lhs(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
  %scalar = arith.constant 4.000000e+00 : f32
  %result = afir.div %scalar, %input outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (f32, !ascendc.local_tensor<?xf32>), !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
  return
}

// IR-LABEL: func.func @div_scalar_rhs
// IR: ascendc.div_extends_reg_base %arg1, %arg0,
// IR-NOT: isScalarLatter

// IR-LABEL: func.func @div_scalar_lhs
// IR: ascendc.div_extends_reg_base %arg1, %arg0, {{.*}}isScalarLatter = false

// **没有 tmp buffer 实参**：`count` 直接跟在标量后面。多一个的话这两条就红。
// CPP: {{^ *}}DivExtends<float, true>({{v[0-9]+}}, {{v[0-9]+}}, c4_f32, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}DivExtends<float, false>({{v[0-9]+}}, {{v[0-9]+}}, c4_f32, static_cast<uint32_t>({{v[0-9]+}}));

// SELECT: ascend.implementation = {id = "div.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "div.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
