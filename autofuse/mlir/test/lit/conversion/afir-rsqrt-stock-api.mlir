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

// 检查标准 AscendC::Rsqrt，与扩展用例对照。
// 此路径不引入扩展头、tmp buffer 或预填 1 的常量块。
//
// 反过来的负例（把不带 tmp 的这份 IR 喂给 mem 侧）由 `WRONGTARGET` 那一行钉：那边只有
// 一个要 tmp buffer 的 implementation，选不出来。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @rsqrt_stock_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.rsqrt %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @rsqrt_stock_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.rsqrt_l2 %arg1, %arg0, %[[COUNT]]
// IR-NOT: ascendc.rsqrt_extend
// IR-NOT: ascendc.duplicate_l2
// IR-NOT: afir.rsqrt

// CPP-LABEL: void rsqrt_stock_f32(
// CPP: AscendC::Rsqrt(
// CPP-NOT: RsqrtExtend
// CPP-NOT: Duplicate

// SELECT: ascend.implementation = {id = "rsqrt.tensor", kind = "ascendc_api"}

// WRONGTARGET: no supported
