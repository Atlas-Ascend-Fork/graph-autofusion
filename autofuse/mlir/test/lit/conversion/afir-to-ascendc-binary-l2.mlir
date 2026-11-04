// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s -o %t.selected.mlir
// RUN: af-opt --convert-afir-to-ascendc --target=Ascend910B1 %t.selected.mlir | FileCheck %s --check-prefix=CONVERT

// 本文件的 IR 刻意**不带** `afir.buffer_desc`：`ascir-translate` 只注册 AscendC 方言，
// 而 `convert-afir-to-ascendc` 会把 kernel 参数上的 `afir.*` attr 原样留下，带上就解析
// 不了。值的规划结果在场时的下降由 afir-to-ascendc-binary-l2-buffer-desc.mlir 覆盖。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @add_l2_f16(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                        %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.add %lhs, %rhs outs(%destination) {
      "ascend.implementation" = {kind = "ascendc_api", id = "add.tensor_tensor"},
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } :
        (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }

  func.func @add_l2_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                        %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.add %lhs, %rhs outs(%destination) {
        afir.operand_descs = [#input_desc, #input_desc],
         afir.destination_descs = [#compute_desc],
         afir.result_descs = [#compute_desc]
        } :
        (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @add_l2_i16(%lhs: !ascendc.local_tensor<?xi16>, %rhs: !ascendc.local_tensor<?xi16>,
                        %destination: !ascendc.local_tensor<?xi16>) {
    %result = afir.add %lhs, %rhs outs(%destination) {
        afir.operand_descs = [#input_desc, #input_desc],
         afir.destination_descs = [#compute_desc],
         afir.result_descs = [#compute_desc]
        } :
        (!ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi16>),
        !ascendc.local_tensor<?xi16> -> !ascendc.local_tensor<?xi16>
    return
  }

  func.func @add_l2_i32(%lhs: !ascendc.local_tensor<?xi32>, %rhs: !ascendc.local_tensor<?xi32>,
                        %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.add %lhs, %rhs outs(%destination) {
        afir.operand_descs = [#input_desc, #input_desc],
         afir.destination_descs = [#compute_desc],
         afir.result_descs = [#compute_desc]
        } :
        (!ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32>),
        !ascendc.local_tensor<?xi32> -> !ascendc.local_tensor<?xi32>
    return
  }

  func.func @add_l2_inplace_lhs(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>) {
    %result = afir.add %lhs, %rhs outs(%lhs) {
        afir.operand_descs = [#input_desc, #input_desc],
         afir.destination_descs = [#compute_desc],
         afir.result_descs = [#compute_desc]
        } :
        (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>),
        !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }

}

// IR-LABEL: func.func @add_l2_f16
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.add_l2 %arg2, %arg0, %arg1, %[[COUNT]] {ascendc.execution_model = "SIMD", ascendc.instruction_form = "MemoryBased", ascendc.unit = "AiCore.Vector"}
// IR-LABEL: func.func @add_l2_f32
// IR: ascendc.add_l2
// IR-LABEL: func.func @add_l2_i16
// IR: ascendc.add_l2
// IR-LABEL: func.func @add_l2_i32
// IR: ascendc.add_l2
// IR-LABEL: func.func @add_l2_inplace_lhs
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg0
// IR-NEXT: ascendc.add_l2 %arg0, %arg0, %arg1, %[[COUNT]] {ascendc.execution_model = "SIMD", ascendc.instruction_form = "MemoryBased", ascendc.unit = "AiCore.Vector"}
// IR-NOT: afir.add

// CPP-LABEL: void add_l2_f16(
// CPP: AscendC::Add(
// CPP-LABEL: void add_l2_f32(
// CPP: AscendC::Add(
// CPP-LABEL: void add_l2_i16(
// CPP: AscendC::Add(
// CPP-LABEL: void add_l2_i32(
// CPP: AscendC::Add(
// CPP-LABEL: void add_l2_inplace_lhs(
// CPP-SAME: AscendC::LocalTensor<half> [[ADD_LHS:[A-Za-z0-9_]+]],
// CPP: int32_t [[ADD_COUNT:[A-Za-z0-9_]+]] = [[ADD_LHS]].GetSize();
// CPP-NEXT: AscendC::Add([[ADD_LHS]], [[ADD_LHS]], {{.*}}, [[ADD_COUNT]]);

// SELECT-LABEL: func.func @add_l2_f32
// SELECT: ascend.implementation = {id = "add.tensor_tensor", kind = "ascendc_api"}

// CONVERT-LABEL: func.func @add_l2_f16
// CONVERT: ascendc.add_l2
// CONVERT-NOT: afir.add
