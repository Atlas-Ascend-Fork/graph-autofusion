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

// descriptor 三件写全，元素类型才是唯一能触发的那一条。**不写 descriptor 的话这份 IR
// 先被 `hasCanonicalDescriptors` 判成 MissingInformation**，报出来的错与 f64 无关——把
// 元素类型判据整条删掉它照样绿。原先这里就是这么写的。
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

// 这里原本还有一个 `unsupported_add_rank`：`2x8xf16`，断言未调度那条路只收秩为 1 的
// tile。它同样没写 descriptor，拦住它的其实是 MissingInformation 而不是秩——二元的秩
// 判据整条删掉它也照样绿。重做之后挪到了
// `afir-elementwise-identity-rank-invalid.mlir`，那份把一元 / 二元 / 三元三条一起钉住
// （一元原先根本没有这条判据）。

// 这里原本还有一个 `mismatched_add_types`：lhs f16 / rhs f32，断言 capability 查询
// 报「没有可用实现」。判据上移到了 ODS 的 `AllElementTypesMatch` 之后，这段 IR 连
// 解析都过不去，留在本文件里会让第一条 RUN（它不分 split）在解析期就失败。负例已挪到
// test/lit/smoke/afir_transfer_ops_invalid.mlir。

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
