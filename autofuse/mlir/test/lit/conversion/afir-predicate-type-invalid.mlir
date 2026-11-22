// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s
// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s

// 谓词结果为 ui8，不能使用 AllElementTypesMatch 约束源和结果类型相同。
// local_tensor 在此 ODS 路径不限制具体元素类型，因此能力层必须分别校验浮点输入、
// 单字节结果和一致形状；这些负例确保检查不会延迟到设备编译。
// 两目标使用同一组 IR，API 家族与 tmp buffer 均匹配，失败应来自类型或形状。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 结果必须是一个字节宽：f32 目的地不行。
func.func @isnan_float_destination(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>,
                                   %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.isnan %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xf32>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 输入必须是浮点：`isnan(int32)` 没有意义。只查「结果是 ui8」会把它放进来。
func.func @isnan_integer_input(%input: !ascendc.local_tensor<?xi32>, %destination: !ascendc.local_tensor<?xui8>,
                               %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.isnan %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<?xui8>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 即使源和结果的元素类型不同，形状仍须一致；此负例拒绝长度不匹配。
func.func @isnan_shape_mismatch(%input: !ascendc.local_tensor<4xf32>, %destination: !ascendc.local_tensor<8xui8>,
                                %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.isnan %input outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : !ascendc.local_tensor<4xf32>, !ascendc.local_tensor<8xui8>, !ascendc.local_tensor<?xui8>
      -> !ascendc.local_tensor<8xui8>
  return
}
