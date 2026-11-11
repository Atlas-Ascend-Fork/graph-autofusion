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

// 第 7 批接通的三十个，与第 6 批同款：全部只有 reg 侧，所以每个 op 只有一条
// implementation，没有「两侧选对了没有」这回事。这一份钉的是三样：建出来的 AscendC op、
// 生成的调用形态、选中的 implementation id。
//
// 这一批真正要钉的是同名歧义：`acos` / `ceil` / `cos` / `round` / `tan` 这些在
// `Adv/Math.td` 里都有一个同名的标准调用 `AscendC::Acos` 等，而 legacy 的 reg 侧走的是
// fork 的包装 `AcosExtend`。`IR` 那组里带 `_extend_reg_base` 后缀的就是包装那一支；
// 只有 `digamma` / `erfc` / `trunc` 三个落在不带后缀的标准调用上（legacy 一个包装头都
// 不给）。挑错一边在 lit 这一层就看得出来，不必等到接管表。
//
// `IR` 那组的实参个数仍然是调用形态的分界：
//
//   - 一元无 tmp（`expm1` / `log1p`）：三个实参；
//   - 一元带 tmp（19 个包装）：四个；
//   - 走标准数学库的（`digamma` / `erfc` / `trunc`）：`MathLibraryOp` 带一个必填的
//     `isReuseSource`，比上面那条还多一个；
//   - 二元无 tmp（`copy_sign` / `trunc_div`）：四个；
//   - 二元带 tmp（`atan2` / `zeta` / `igamma` / `igammac`）：五个。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @acos_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.acos %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @acosh_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.acosh %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @asin_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.asin %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @asinh_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.asinh %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @atan_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.atan %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @atanh_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.atanh %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @ceil_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.ceil %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @cos_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cos %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @cosh_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cosh %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @erfcx_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.erfcx %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @round_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.round %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @sin_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sin %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @sinh_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.sinh %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @tan_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.tan %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @bessel_y0_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.bessel_y0 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @bessel_y1_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.bessel_y1 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @spherical_bessel_j0_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.spherical_bessel_j0 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @scaled_modified_bessel_k0_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.scaled_modified_bessel_k0 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @scaled_modified_bessel_k1_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.scaled_modified_bessel_k1 %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @digamma_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.digamma %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @erfc_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.erfc %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @trunc_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.trunc %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @expm1_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.expm1 %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @log1p_f32(%input: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.log1p %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @copy_sign_f32(%lhs: !ascendc.local_tensor<?xf32>,
                     %rhs: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.copy_sign %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @trunc_div_f32(%lhs: !ascendc.local_tensor<?xf32>,
                     %rhs: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.trunc_div %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @atan2_f32(%lhs: !ascendc.local_tensor<?xf32>,
                     %rhs: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.atan2 %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @zeta_f32(%lhs: !ascendc.local_tensor<?xf32>,
                     %rhs: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.zeta %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @igamma_f32(%lhs: !ascendc.local_tensor<?xf32>,
                     %rhs: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.igamma %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @igammac_f32(%lhs: !ascendc.local_tensor<?xf32>,
                     %rhs: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.igammac %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 一元带 tmp 的 19 个包装：四个实参，op 名带 `_extend_reg_base`。
// IR-LABEL: func.func @acos_f32
// IR: ascendc.acos_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.acos

// IR-LABEL: func.func @acosh_f32
// IR: ascendc.acosh_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.acosh

// IR-LABEL: func.func @asin_f32
// IR: ascendc.asin_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.asin

// IR-LABEL: func.func @asinh_f32
// IR: ascendc.asinh_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.asinh

// IR-LABEL: func.func @atan_f32
// IR: ascendc.atan_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.atan

// IR-LABEL: func.func @atanh_f32
// IR: ascendc.atanh_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.atanh

// IR-LABEL: func.func @ceil_f32
// IR: ascendc.ceil_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.ceil

// IR-LABEL: func.func @cos_f32
// IR: ascendc.cos_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.cos

// IR-LABEL: func.func @cosh_f32
// IR: ascendc.cosh_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.cosh

// IR-LABEL: func.func @erfcx_f32
// IR: ascendc.erfcx_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.erfcx

// IR-LABEL: func.func @round_f32
// IR: ascendc.round_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.round

// IR-LABEL: func.func @sin_f32
// IR: ascendc.sin_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.sin

// IR-LABEL: func.func @sinh_f32
// IR: ascendc.sinh_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.sinh

// IR-LABEL: func.func @tan_f32
// IR: ascendc.tan_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.tan

// IR-LABEL: func.func @bessel_y0_f32
// IR: ascendc.bessel_y0_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.bessel_y0

// IR-LABEL: func.func @bessel_y1_f32
// IR: ascendc.bessel_y1_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.bessel_y1

// IR-LABEL: func.func @spherical_bessel_j0_f32
// IR: ascendc.spherical_bessel_j0_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.spherical_bessel_j0

// IR-LABEL: func.func @scaled_modified_bessel_k0_f32
// IR: ascendc.scaled_modified_bessel_k0_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.scaled_modified_bessel_k0

// IR-LABEL: func.func @scaled_modified_bessel_k1_f32
// IR: ascendc.scaled_modified_bessel_k1_extend_reg_base %arg1, %arg0, %arg2
// IR-NOT: afir.scaled_modified_bessel_k1

// 标准数学库那三个：没有 `_extend_reg_base` 后缀，建出来的是 `Adv/Math.td` 里的那个 op。
// IR-LABEL: func.func @digamma_f32
// IR: ascendc.digamma %arg1, %arg0, %arg2
// IR-NOT: afir.digamma

// IR-LABEL: func.func @erfc_f32
// IR: ascendc.erfc %arg1, %arg0, %arg2
// IR-NOT: afir.erfc

// IR-LABEL: func.func @trunc_f32
// IR: ascendc.trunc %arg1, %arg0, %arg2
// IR-NOT: afir.trunc

// 一元无 tmp：三个实参。
// IR-LABEL: func.func @expm1_f32
// IR: ascendc.expm1_extend_reg_base %arg1, %arg0
// IR-NOT: afir.expm1

// IR-LABEL: func.func @log1p_f32
// IR: ascendc.log1p_extend_reg_base %arg1, %arg0
// IR-NOT: afir.log1p

// 二元无 tmp：四个实参。
// IR-LABEL: func.func @copy_sign_f32
// IR: ascendc.copy_sign_extend_reg_base %arg2, %arg0, %arg1
// IR-NOT: afir.copy_sign

// IR-LABEL: func.func @trunc_div_f32
// IR: ascendc.trunc_div_extend_reg_base %arg2, %arg0, %arg1
// IR-NOT: afir.trunc_div

// 二元带 tmp：五个实参。
// IR-LABEL: func.func @atan2_f32
// IR: ascendc.atan2_extend_reg_base %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.atan2

// IR-LABEL: func.func @zeta_f32
// IR: ascendc.zeta_extend_reg_base %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.zeta

// IR-LABEL: func.func @igamma_f32
// IR: ascendc.igamma_extend_reg_base %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.igamma

// IR-LABEL: func.func @igammac_f32
// IR: ascendc.igammac_extend_reg_base %arg2, %arg0, %arg1, %arg3
// IR-NOT: afir.igammac

// 包装住在全局命名空间，靠 `AddKernelIncludes` 生成的那条 using 兜着不带限定的名字。
// 一元扩展调用生成模板实参、二元不生成（`AcosExtend<float>` vs `CopySignExtend`）——两个
// printer 各按各的来，与 `afir-reg-only-math-api.mlir` 里的写法一致。
// CPP-LABEL: void acos_f32(
// CPP: {{^ *}}AcosExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void acosh_f32(
// CPP: {{^ *}}AcoshExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void asin_f32(
// CPP: {{^ *}}AsinExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void asinh_f32(
// CPP: {{^ *}}AsinhExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void atan_f32(
// CPP: {{^ *}}AtanExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void atanh_f32(
// CPP: {{^ *}}AtanhExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void ceil_f32(
// CPP: {{^ *}}CeilExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void cos_f32(
// CPP: {{^ *}}CosExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void cosh_f32(
// CPP: {{^ *}}CoshExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void erfcx_f32(
// CPP: {{^ *}}ErfcxExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void round_f32(
// CPP: {{^ *}}RoundExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void sin_f32(
// CPP: {{^ *}}SinExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void sinh_f32(
// CPP: {{^ *}}SinhExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void tan_f32(
// CPP: {{^ *}}TanExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void bessel_y0_f32(
// CPP: {{^ *}}BesselY0Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void bessel_y1_f32(
// CPP: {{^ *}}BesselY1Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void spherical_bessel_j0_f32(
// CPP: {{^ *}}SphericalBesselJ0Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void scaled_modified_bessel_k0_f32(
// CPP: {{^ *}}ScaledModifiedBesselK0Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void scaled_modified_bessel_k1_f32(
// CPP: {{^ *}}ScaledModifiedBesselK1Extend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void expm1_f32(
// CPP: {{^ *}}Expm1Extend<float>({{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void log1p_f32(
// CPP: {{^ *}}Log1p<float>({{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void copy_sign_f32(
// CPP: {{^ *}}CopySignExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void trunc_div_f32(
// CPP: {{^ *}}TruncDivExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void atan2_f32(
// CPP: {{^ *}}Atan2Extend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void zeta_f32(
// CPP: {{^ *}}ZetaExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void igamma_f32(
// CPP: {{^ *}}IgammaExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// CPP-LABEL: void igammac_f32(
// CPP: {{^ *}}IgammacExtend({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(

// SELECT: ascend.implementation = {id = "acos.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "acosh.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "asin.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "asinh.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "atan.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "atanh.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "ceil.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "cos.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "cosh.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "erfcx.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "round.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "sin.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "sinh.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "tan.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "bessel_y0.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "bessel_y1.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "spherical_bessel_j0.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "scaled_modified_bessel_k0.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "scaled_modified_bessel_k1.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "digamma.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "erfc.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "trunc.tensor", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "expm1.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "log1p.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "copy_sign.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "trunc_div.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "atan2.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "zeta.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "igamma.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "igammac.tensor.extend_reg_base", kind = "ascendc_api"}
