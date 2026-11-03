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

// 第 6 批接通的九个，**全部只有 reg 侧**（legacy 的 mem 侧 `ascir_builtin_ops_v1.cpp`
// 不注册它们），所以没有「两侧选对了没有」这回事——每个 op 只有一条 implementation。
// 这一份钉的是三样：建出来的 AscendC op、发出去的调用形态、选中的 implementation id。
//
// 这批横跨四种调用形态，`IR` 那组的实参个数正是它们的分界：
//
//   - 一元无 tmp（`erfinv`）：三个实参；
//   - 一元带 tmp（`exp2`）：四个；
//   - 走标准数学库的（`log2` / `floor` / `lgamma`）：`MathLibraryOp` 带一个必填的
//     `isReuseSource`，所以它们比上面那条还多一个实参；
//   - 二元无 tmp（`logical_xor`）：四个；
//   - 二元带 tmp（`fmod` / `mod` / `next_after`）：五个。
//
// **`fmod` 与 `mod` 落到同一个 AscendC op**（legacy 两个调度节点的 apiName 都是
// `FmodExtend`），所以那两格分得开的只有 `SELECT` 那条——`IR` 和 `CPP` 两边逐字一样。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @erfinv_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                 %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.erfinv %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @exp2_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                               %destination: !ascendc.local_tensor<?xf32>,
                               %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.exp2 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @log2_f32(%input: !ascendc.local_tensor<?xf32>,
                      %destination: !ascendc.local_tensor<?xf32>,
                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.log2 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @floor_f16(%input: !ascendc.local_tensor<?xf16>,
                       %destination: !ascendc.local_tensor<?xf16>,
                       %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.floor %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }

  func.func @lgamma_f32(%input: !ascendc.local_tensor<?xf32>,
                        %destination: !ascendc.local_tensor<?xf32>,
                        %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.lgamma %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 结果恒 `ui8`，与 `logical_and` / `logical_or` 同一张表。
  func.func @logical_xor_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                                      %destination: !ascendc.local_tensor<?xui8>) {
    %result = afir.logical_xor %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @fmod_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                               %destination: !ascendc.local_tensor<?xf32>,
                               %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.fmod %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @mod_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                              %destination: !ascendc.local_tensor<?xf32>,
                              %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.mod %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @next_after_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.next_after %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @erfinv_reg_base_f32
// IR: ascendc.erfinv_extend_reg_base %arg1, %arg0
// IR-NOT: afir.erfinv

// 一元带 tmp：四个实参。
// IR-LABEL: func.func @exp2_reg_base_f32
// IR: ascendc.exp2_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.exp2

// 标准数学库那三个：没有 `_extend` 后缀，建出来的是 `Adv/Math.td` 里的那个 op。
// IR-LABEL: func.func @log2_f32
// IR: ascendc.log2 %arg1, %arg0, %arg2
// IR-NOT: afir.log2

// IR-LABEL: func.func @floor_f16
// IR: ascendc.floor %arg1, %arg0, %arg2
// IR-NOT: afir.floor

// IR-LABEL: func.func @lgamma_f32
// IR: ascendc.lgamma %arg1, %arg0, %arg2
// IR-NOT: afir.lgamma

// IR-LABEL: func.func @logical_xor_reg_base_f32
// IR: ascendc.logical_xor_extend_reg_base %arg2, %arg0, %arg1
// IR-NOT: afir.logical_xor

// 二元带 tmp：五个实参。
// IR-LABEL: func.func @fmod_reg_base_f32
// IR: ascendc.fmod_extend_reg_base %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.fmod

// `mod` 建出来的是**同一个** AscendC op，所以这条与上面那条逐字一样。
// IR-LABEL: func.func @mod_reg_base_f32
// IR: ascendc.fmod_extend_reg_base %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.mod

// IR-LABEL: func.func @next_after_reg_base_f32
// IR: ascendc.next_after_extend_reg_base %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.next_after

// 包装住在全局命名空间，靠 `AddKernelIncludes` 发的那条 using 兜着不带限定的名字。
// **一元发模板实参、二元不发**（`Erfinv<float>` vs `LogicalXorExtend`）——两个
// printer 各按各的来，与 `afir-remainder-extension-api.mlir` 里的写法一致。
// CPP-LABEL: void erfinv_reg_base_f32(
// CPP: {{^ *}}Erfinv<float>({{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void exp2_reg_base_f32(
// CPP: {{^ *}}Exp2<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void logical_xor_reg_base_f32(
// CPP: {{^ *}}LogicalXorExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void fmod_reg_base_f32(
// CPP: {{^ *}}FmodExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void next_after_reg_base_f32(
// CPP: {{^ *}}NextAfterExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "erfinv.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "exp2.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "log2.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "floor.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "lgamma.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "logical_xor.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "fmod.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "mod.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "next_after.tensor.extend_reg_base", kind = "ascendc_api"}
