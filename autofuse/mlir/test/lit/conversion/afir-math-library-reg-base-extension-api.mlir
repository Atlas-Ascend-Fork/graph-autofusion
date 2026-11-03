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
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=NOFLAG --implicit-check-not="arith.constant false"

// afir-math-library-stock-api.mlir 的另一半。两侧分得开的东西比 `sign` / `remainder`
// 那几对多：**连函数名都不同**（`TanhExtend` vs `AscendC::Tanh`），而且 reg 侧这个包装
// 自己把 `isReuseSource` 定死在函数体里（`Tanh<T, false, tanh_config>`），所以这一侧
// 发的调用**不带模板实参**，也就不物化那个 `arith.constant false`。
//
// 下面 `NOFLAG` 那一条 RUN 就是钉这个的：把 `MaterializesReuseSource` 误加到这一侧的话，
// 多出来的那个常量既没人用、也进不了调用，只有这条能看出来。**它单开一条 RUN 而不是
// 写成 `IR-NOT`**：`CHECK-NOT` 只管到上一条匹配与下一条匹配之间，而那个常量是建在调用
// **前面**的，写成 `IR-NOT` 盖不住第一个函数那一段。`--implicit-check-not` 盖整份输出。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @tanh_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                      %destination: !ascendc.local_tensor<?xf32>,
                                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.tanh %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @erf_extend_reg_base_f16(%input: !ascendc.local_tensor<?xf16>,
                                     %destination: !ascendc.local_tensor<?xf16>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.erf %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }

  // `ndtr` 与上面两个同形，但它**只有这一侧**：mem 侧既没有 `AscendC::Ndtr` 也没有
  // `REG_ASC_IR(Ndtr)`，所以 `afir-math-library-stock-api.mlir` 那半边没有它的对应项。
  func.func @ndtr_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                      %destination: !ascendc.local_tensor<?xf32>,
                                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.ndtr %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 同族其余三个（第 2 批）。`airy_ai` 的包装要一个前导头
  // （`trigonometric_function_utils_reg_base.h`，`airy_ai.h` 调里面的 `AutofuseSin` /
  // `AutofuseCos`），**不是两个扩展头**——第 3 批把它从头列表改成了前导头那一格。真正声明
  // 两个扩展头的是第 4 批的 `i0e` / `i1e`（见本文件末尾）。头名与前导头由
  // `CodegenOrchestrationTest` 对着注册表查，这里只钉调用。
  func.func @ndtri_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                       %destination: !ascendc.local_tensor<?xf32>,
                                       %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.ndtri %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @log_ndtr_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                          %destination: !ascendc.local_tensor<?xf32>,
                                          %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.log_ndtr %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @i0_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                    %destination: !ascendc.local_tensor<?xf32>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.i0 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @airy_ai_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                         %destination: !ascendc.local_tensor<?xf32>,
                                         %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.airy_ai %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 第 3 批。两个的 utils 头都走前导头那一格，这里只钉调用；头名与前导头由
  // `CodegenOrchestrationTest` 对着供给表查。

  func.func @bessel_j0_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.bessel_j0 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @bessel_j1_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.bessel_j1 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 第 4 批。这两个的包装声明了**两个扩展头**（`modified_bessel_i0_reg_base.h` 要先内联），
  // 而这条管线不经 `AddKernelIncludes`，所以**内联顺序这里钉不住**，只钉调用和选型；顺序由
  // `ApiExtensionOrderTest` 和 200 格接管表管。

  func.func @i0e_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.i0e %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @i1e_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.i1e %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 第 5 批。`k0` / `k1` 也声明两个扩展头，依赖序与字典序一致；同上，顺序这里钉不住。

  func.func @modified_bessel_i0_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.modified_bessel_i0 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @modified_bessel_i1_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.modified_bessel_i1 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @modified_bessel_k0_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.modified_bessel_k0 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @modified_bessel_k1_extend_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                     %destination: !ascendc.local_tensor<?xf32>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.modified_bessel_k1 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @tanh_extend_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.tanh_extend_reg_base %arg1, %arg0, %arg2, %[[COUNT]]
// IR-NOT: ascendc.tanh %
// IR-NOT: afir.tanh

// IR-LABEL: func.func @erf_extend_reg_base_f16
// IR: ascendc.erf_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: ascendc.erf %
// IR-NOT: afir.erf

// IR-LABEL: func.func @ndtr_extend_reg_base_f32
// IR: ascendc.ndtr_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.ndtr

// IR-LABEL: func.func @ndtri_extend_reg_base_f32
// IR: ascendc.ndtri_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @log_ndtr_extend_reg_base_f32
// IR: ascendc.log_ndtr_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @i0_extend_reg_base_f32
// IR: ascendc.i0_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @airy_ai_extend_reg_base_f32
// IR: ascendc.airy_ai_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @bessel_j0_extend_reg_base_f32
// IR: ascendc.bessel_j0_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @bessel_j1_extend_reg_base_f32
// IR: ascendc.bessel_j1_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @i0e_extend_reg_base_f32
// IR: ascendc.i0e_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @i1e_extend_reg_base_f32
// IR: ascendc.i1e_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @modified_bessel_i0_extend_reg_base_f32
// IR: ascendc.modified_bessel_i0_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @modified_bessel_i1_extend_reg_base_f32
// IR: ascendc.modified_bessel_i1_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @modified_bessel_k0_extend_reg_base_f32
// IR: ascendc.modified_bessel_k0_extend_reg_base %arg1, %arg0, %arg2
// IR-LABEL: func.func @modified_bessel_k1_extend_reg_base_f32
// IR: ascendc.modified_bessel_k1_extend_reg_base %arg1, %arg0, %arg2

// 三个调用都在，而整份输出里没有那个 `isReuseSource` 常量——包装自己把它定死了。
// NOFLAG: ascendc.tanh_extend_reg_base
// NOFLAG: ascendc.erf_extend_reg_base
// NOFLAG: ascendc.ndtr_extend_reg_base
// NOFLAG: ascendc.ndtri_extend_reg_base
// NOFLAG: ascendc.log_ndtr_extend_reg_base
// NOFLAG: ascendc.i0_extend_reg_base
// NOFLAG: ascendc.airy_ai_extend_reg_base
// NOFLAG: ascendc.bessel_j0_extend_reg_base
// NOFLAG: ascendc.bessel_j1_extend_reg_base
// NOFLAG: ascendc.i0e_extend_reg_base
// NOFLAG: ascendc.i1e_extend_reg_base
// NOFLAG: ascendc.modified_bessel_i0_extend_reg_base
// NOFLAG: ascendc.modified_bessel_i1_extend_reg_base
// NOFLAG: ascendc.modified_bessel_k0_extend_reg_base
// NOFLAG: ascendc.modified_bessel_k1_extend_reg_base

// 包装住在全局命名空间，不带 `AscendC::`，也不带模板实参。
// CPP-LABEL: void tanh_extend_reg_base_f32(
// CPP: {{^ *}}TanhExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void erf_extend_reg_base_f16(
// CPP: {{^ *}}ErfExtend<half>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void ndtr_extend_reg_base_f32(
// CPP: {{^ *}}NdtrExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void ndtri_extend_reg_base_f32(
// CPP: {{^ *}}NdtriExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void log_ndtr_extend_reg_base_f32(
// CPP: {{^ *}}LogNdtrExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void i0_extend_reg_base_f32(
// CPP: {{^ *}}I0Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void airy_ai_extend_reg_base_f32(
// CPP: {{^ *}}AiryAiExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void bessel_j0_extend_reg_base_f32(
// CPP: {{^ *}}BesselJ0Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void bessel_j1_extend_reg_base_f32(
// CPP: {{^ *}}BesselJ1Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void i0e_extend_reg_base_f32(
// CPP: {{^ *}}I0eExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void i1e_extend_reg_base_f32(
// CPP: {{^ *}}I1eExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void modified_bessel_i0_extend_reg_base_f32(
// CPP: {{^ *}}ModifiedBesselI0Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void modified_bessel_i1_extend_reg_base_f32(
// CPP: {{^ *}}ModifiedBesselI1Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void modified_bessel_k0_extend_reg_base_f32(
// CPP: {{^ *}}ModifiedBesselK0Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void modified_bessel_k1_extend_reg_base_f32(
// CPP: {{^ *}}ModifiedBesselK1Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "tanh.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "erf.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "ndtr.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "ndtri.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "log_ndtr.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "i0.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "airy_ai.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "bessel_j0.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "bessel_j1.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "i0e.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "i1e.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "modified_bessel_i0.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "modified_bessel_i1.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "modified_bessel_k0.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "modified_bessel_k1.tensor.extend_reg_base", kind = "ascendc_api"}
