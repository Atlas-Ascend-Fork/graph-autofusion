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

// 选择族的 reg-base 一侧,与 afir-select-extension-api.mlir 互为对照:**同一个 AFIR op,
// 少一块 tmp buffer**。两个方向都钉住——这一份没有 tmp buffer,在 mem-based 一侧就是
// unsupported(见 WRONGTARGET),因为 `Where` 要调用方给那块 UB,而 AFIR 造不出来。
//
// 两个包装的头名也是两张供给表的判别式:`where.h` vs `where_reg_base.h`。注册期的
// `checkFamilyMatchesHeader` 逐项比对它们,写反了整张注册表建不起来。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @where_reg_base_f32(%condition: !ascendc.local_tensor<?xui8>,
                                %lhs: !ascendc.local_tensor<?xf32>,
                                %rhs: !ascendc.local_tensor<?xf32>,
                                %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.where %condition, %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @where_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg3
// IR-NEXT: ascendc.where_contiguous_reg_base %arg3, %arg0, %arg1, %arg2, %[[COUNT]]
// IR-NOT: afir.where

// CPP-LABEL: void where_reg_base_f32(
// CPP: {{^ *}}WhereExtend(
// CPP-NOT: AscendC::

// SELECT: ascend.implementation = {id = "where.tensor.extend_reg_base", kind = "ascendc_api"}

// WRONGTARGET: no supported
