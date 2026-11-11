// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt --afir-to-ascendc-pipeline %s 2>&1 | FileCheck %s --check-prefix=MISSING-TARGET
// RUN: not af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 --split-input-file %s 2>&1 | FileCheck %s --check-prefix=INVALID

// MISSING-TARGET: TargetSelectionConflict: one of --target or --target-profile is required

// 补全 canonical descriptors，使 f64 类型成为唯一拒绝原因。
// 否则能力查询会先返回 MissingInformation，无法验证元素类型检查是否生效。
#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @unsupported_add_type(%lhs: !ascendc.local_tensor<?xf64>, %rhs: !ascendc.local_tensor<?xf64>,
                                  %destination: !ascendc.local_tensor<?xf64>) {
    // INVALID: error: 'afir.add' op has no supported Ascend implementation
    %result = afir.add %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf64>, !ascendc.local_tensor<?xf64>),
        !ascendc.local_tensor<?xf64> -> !ascendc.local_tensor<?xf64>
    return
  }
}

// -----

// identity 路径的秩限制由 afir-elementwise-identity-rank-invalid.mlir 覆盖，
// 该文件统一检查一元、二元和三元算子，并提供完整 descriptor。

// lhs/rhs dtype 不一致由 ODS AllElementTypesMatch 在解析后校验，
// 不再进入 capability 查询。负例位于 smoke/afir_transfer_ops_invalid.mlir。

// -----

module {
  func.func @unknown_add_kind(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                              %destination: !ascendc.local_tensor<?xf16>) {
    // INVALID: error: 'afir.add' op has unknown Ascend implementation kind
    %result = afir.add %lhs, %rhs outs(%destination) {
      "ascend.implementation" = {kind = "unknown", id = "add.tensor_tensor"}
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
}

// -----

module {
  func.func @unknown_add_id(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                            %destination: !ascendc.local_tensor<?xf16>) {
    // INVALID: error: 'afir.add' op selected Ascend implementation is not registered for this operation
    %result = afir.add %lhs, %rhs outs(%destination) {
      "ascend.implementation" = {kind = "ascendc_api", id = "removed"}
    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }
}
