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

// 三元族标量形态的 reg 侧。与 mem 侧那份（afir-ternary-scalar-api.mlir）是两个文件而不是
// 两条 RUN：`WhereExtend` / reg 侧的 `ClipByValue` 都不收 tmp buffer（自己开寄存器），
// 图上带不带 tmp 操作数不一样，一份输入服务不了两侧。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @where_scalar_else(%mask: !ascendc.local_tensor<?xui8>, %then: !ascendc.local_tensor<?xf32>,
                             %destination: !ascendc.local_tensor<?xf32>) {
  %scalar = arith.constant 2.000000e+00 : f32
  %result = afir.where %mask, %then, %scalar outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xf32>, f32),
      !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
  return
}

func.func @clip_scalar_bounds(%x: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
  %min = arith.constant -1.000000e+00 : f32
  %max = arith.constant 1.000000e+00 : f32
  %result = afir.clip_by_value %x, %min, %max outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf32>, f32, f32),
      !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
  return
}

// IR-LABEL: func.func @where_scalar_else
// IR: ascendc.where_contiguous_reg_base %arg2, %arg0, %arg1, %{{.*}}

// IR-LABEL: func.func @clip_scalar_bounds
// IR: ascendc.clipbyvalue_contiguous_reg_base %arg1, %arg0, %{{.*}}, %{{.*}}

// 没有 tmp buffer 实参：count 是最后一个。多一个的话这两条就红。
// CPP: {{^ *}}WhereExtend({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, c2_f32, static_cast<uint32_t>({{v[0-9]+}}));
// CPP: {{^ *}}ClipByValue({{v[0-9]+}}, {{v[0-9]+}}, cm1_f32, c1_f32, static_cast<uint32_t>({{v[0-9]+}}));

// SELECT: ascend.implementation = {id = "where.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "clip_by_value.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
