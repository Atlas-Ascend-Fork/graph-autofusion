// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR \
// RUN:   --implicit-check-not=afir.is --implicit-check-not=_extend
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP --implicit-check-not=IsnanExtend \
// RUN:   --implicit-check-not=IsFiniteExtend --implicit-check-not=IsInfExtend
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// 标准 AscendC::IsNan 接收 LocalTensor<bool>，而 AFIR 结果为 ui8，
// 因此调用处需要 ReinterpretCast<bool>()。扩展包装直接接收 ui8，不需此转换；
// 标准调用也不需要 tmp buffer。
//
// 「三个包装一个都没出现」「没有 afir.is 残留」用 `--implicit-check-not` 写在 RUN 行
// 上，因为 `CHECK-NOT` 只在相邻两条正向 CHECK 之间生效。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @isnan_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.isnan %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @isfinite_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.isfinite %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @isinf_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.isinf %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }
}

// IR-LABEL: func.func @isnan_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.is_nan_l2 %arg1, %arg0, %[[COUNT]]
// IR-LABEL: func.func @isfinite_f16
// IR: ascendc.is_finite_l2 %arg1
// IR-LABEL: func.func @isinf_f32
// IR: ascendc.is_inf_l2 %arg1

// CPP-LABEL: void isnan_f32(
// CPP: AscendC::IsNan({{.*}}.template ReinterpretCast<bool>(),
// CPP-LABEL: void isfinite_f16(
// CPP: AscendC::IsFinite({{.*}}.template ReinterpretCast<bool>(),
// CPP-LABEL: void isinf_f32(
// CPP: AscendC::IsInf({{.*}}.template ReinterpretCast<bool>(),

// SELECT: ascend.implementation = {id = "isnan.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "isfinite.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "isinf.tensor", kind = "ascendc_api"}

// WRONGTARGET: no supported
