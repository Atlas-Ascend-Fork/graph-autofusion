// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR \
// RUN:   --implicit-check-not=afir.is --implicit-check-not=ascendc.is_nan_l2
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP --implicit-check-not=ReinterpretCast
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// 浮点谓词族的 mem-based 一侧：输入 f16/f32，结果 ui8。这一族是 elementwise 里第一个
// **结果 dtype 与输入不同**的，所以 ODS 的 `AllElementTypesMatch` 对它们是关掉的，
// capability 里那条「输入与目的地同元素类型」也换成了「输入浮点、结果一个字节」。
// 少掉这两处中的任何一处，下面这三个函数都建不出来。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @isnan_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xui8>,
                       %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.isnan %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @isfinite_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xui8>,
                          %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.isfinite %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @isinf_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xui8>,
                       %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.isinf %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }
}

// IR-LABEL: func.func @isnan_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.isnan_extend %arg1, %arg0, %arg2, %[[COUNT]]
// IR-LABEL: func.func @isfinite_f16
// IR: ascendc.isfinite_extend %arg1
// IR-LABEL: func.func @isinf_f32
// IR: ascendc.isinf_extend %arg1

// 包装收下的就是 ui8 目的地，不需要 ReinterpretCast——那一步只有标准调用那一侧要。
// 「没有 ReinterpretCast」「没有 afir.is 残留」「没有走成标准调用」这三条用
// `--implicit-check-not` 写在 RUN 行上：`CHECK-NOT` 只在相邻两条正向 CHECK 之间生效，
// 写在最后一条之后就只覆盖到末尾那一小段。
// CPP-LABEL: void isnan_f32(
// CPP: {{^ *}}IsnanExtend<float>(
// CPP-LABEL: void isfinite_f16(
// CPP: {{^ *}}IsFiniteExtend<half>(
// CPP-LABEL: void isinf_f32(
// CPP: {{^ *}}IsInfExtend<float>(

// SELECT: ascend.implementation = {id = "isnan.tensor.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "isfinite.tensor.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "isinf.tensor.extend", kind = "ascendc_api"}

// WRONGTARGET: no supported
