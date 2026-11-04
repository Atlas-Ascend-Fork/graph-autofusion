// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

// D0：两套 API 家族给的是同一个调用，都不要 tmp buffer。**同一份 IR、两个 --target、
// 同一组 CHECK** 就是这一档的全部内容——这一点和 afir-abs-{stock,extension}-api.mlir
// 那对文件正好相反：那两个 D2 的 op 各自只在一个 target 上成立，所以必须拆成两份并
// 各带一条反向检查。这里如果哪天冒出平台分支，两条 RUN 中的一条就会红。
//
// 没有 tmp buffer 也是判据的一部分：capability 查询要求 tmp buffer 的有无与选中的 API 匹配，
// 而这四个 op 只注册了 stock 一条路。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @exp_f16(%input: !ascendc.local_tensor<?xf16>, %destination: !ascendc.local_tensor<?xf16>) {
    %result = afir.exp %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>
    return
  }

  func.func @ln_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.ln %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  // relu 是四个里唯一接受整型的。
  func.func @relu_i32(%input: !ascendc.local_tensor<?xi32>, %destination: !ascendc.local_tensor<?xi32>) {
    %result = afir.relu %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xi32>, !ascendc.local_tensor<?xi32> -> !ascendc.local_tensor<?xi32>
    return
  }

  func.func @sqrt_f32(%input: !ascendc.local_tensor<?xf32>, %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.sqrt %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @exp_f16
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.exp_l2 %arg1, %arg0, %[[COUNT]]
// IR-LABEL: func.func @ln_f32
// IR: ascendc.ln_l2
// IR-LABEL: func.func @relu_i32
// IR: ascendc.relu_l2
// IR-LABEL: func.func @sqrt_f32
// IR: ascendc.sqrt_l2
// 一条兜底的 NOT，放在最后一条正向 CHECK 之后——FileCheck 的 -NOT 只在相邻两条正向
// CHECK 之间生效，夹在每个函数中间的话只覆盖到那一小段。真正挡住残留的是
// `addIllegalDialect(afir)`，这条只是让失败信息更直白。
// IR-NOT: afir.

// 全部是 `AscendC::` 成员，没有任何扩展 wrapper。
// CPP-LABEL: void exp_f16(
// CPP: AscendC::Exp(
// CPP-LABEL: void ln_f32(
// CPP: AscendC::Ln(
// CPP-LABEL: void relu_i32(
// CPP: AscendC::Relu(
// CPP-LABEL: void sqrt_f32(
// CPP: AscendC::Sqrt(

// stable id 不含平台名，两个 target 上是同一个。
// SELECT: ascend.implementation = {id = "exp.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "ln.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "relu.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "sqrt.tensor", kind = "ascendc_api"}
