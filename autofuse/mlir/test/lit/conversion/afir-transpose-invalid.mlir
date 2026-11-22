// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// **verifier 层的两格硬错**（与 `afir-transpose-unsupported.mlir` 的 capability 软拒相对）：
//
//   - `@scalar_input`：标量输入。转置的对象是 tile，ODS 的 `ShapedType` 约束在 verifier
//     就挡下，轮不到 capability 说话——legacy 的 `IsNodeValid` 同样拒标量
//     （`v2_ascir_codegen_impl.h:3750-3754`）。
//   - `@rank_mismatch`：两端 rank 不同。转置重排轴序，**不增删轴**——`verifyTransposeSameRank`
//     的判据（无秩的一端不比，memref 形态由 descriptor 与 maps 隐含 rank 相等）。
//
// `--split-input-file` 让两格各自独立报错；计数钉住「两格都报、不是一票全否」。

// RUN: not af-opt --split-input-file %s 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID-COUNT-2: error:
// INVALID: error: 'afir.transpose' op transpose must not add or remove axes: input and output ranks differ

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @scalar_input(%input: i32, %destination: !ascendc.local_tensor<?xi32>,
                          %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.transpose %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : i32, !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xi32>
    return
  }

  func.func @rank_mismatch(%input: !ascendc.local_tensor<?x?x8xi32>, %destination: !ascendc.local_tensor<?x8xi32>,
                           %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.transpose %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?x?x8xi32>, !ascendc.local_tensor<?x8xi32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?x8xi32>
    return
  }
}

// -----

// Logical tensor form must enforce the same rank contract as buffer form.
module {
  func.func @logical_rank_mismatch(%input: tensor<2x3xf32>) -> tensor<6xf32> {
    %result = afir.transpose %input : tensor<2x3xf32> -> tensor<6xf32>
    return %result : tensor<6xf32>
  }
}
