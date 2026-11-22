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

// reg-base Where 不接收 tmp buffer，与 mem-based 用例对照。
// 同一 AFIR op 省略 scratch 后，mem 侧应拒绝，因为其包装要求调用方提供该缓冲。
//
// 两侧分别使用 where.h 与 where_reg_base.h。
// 注册时 checkFamilyMatchesHeader 校验家族与头名一致，写反会导致注册失败。

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
