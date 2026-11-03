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

// afir-compare-extension-api.mlir 的另一半。两侧的**调用名一样**（`CompareExtend`），
// 差的是别的三样，其中第一样这条路上此前没出现过：
//
//   - **`CMPMODE` 换了位置**：mem 侧是第四个实参，reg 侧是第三个模板实参。同一个
//     `cmpMode` 属性、两个 emitter，写在同一个 op 上分不开，所以是两个 def。
//   - **模板实参表**：reg 侧要打 `<InT, 1, CMPMODE::XX>`——`dim` 排在中间，写出它就得
//     把 `InT` 也写出来；`OutT` 排在最后，照旧由 dst 推。`1` 就是「连续」的全部含义。
//   - **tmp buffer**：`GetCompareSizeV2` 照样给这个节点分 32 字节，而 `CompareExtend`
//     的签名里没有那个形参——`TmpBufferUse::Unused`。所以函数签名里有 %tmp，调用里
//     没有它，与 `sub` / `div` 的 mem 侧同一档。
//
// 三个 brace-init 数组收的是 `const uint16_t (&)[1]`，**不是 `cast_reg_base.h` 的
// `uint32_t`**：照抄那一个写成 `{1U}` 是窄化初始化，device 编译器当场报错而 lit 全绿。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @ne_reg_base_f32(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                             %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.ne %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  // reg 侧比 mem 侧多收三个 dtype（i16 / ui8 / ui16）。ui8 这一行还顺带说明：输入与
  // 结果**可以**同 dtype，判据仍然是「结果恒为 ui8」而不是「结果等于输入」。
  func.func @gt_reg_base_ui8(%lhs: !ascendc.local_tensor<?xui8>, %rhs: !ascendc.local_tensor<?xui8>,
                             %destination: !ascendc.local_tensor<?xui8>, %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.gt %lhs, %rhs outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }

  func.func @le_scalar_reg_base_f32(%input: !ascendc.local_tensor<?xf32>,
                                    %destination: !ascendc.local_tensor<?xui8>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %scalar = arith.constant 2.500000e+00 : f32
    %result = afir.le %input, %scalar outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, f32),
        !ascendc.local_tensor<?xui8>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xui8>
    return
  }
}

// %arg3（tmp）不在调用里——这就是 `Unused` 与 mem 侧 `Passed` 的区别。
// IR-LABEL: func.func @ne_reg_base_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg2
// IR-NEXT: ascendc.compare_extend_contiguous_reg_base %arg2, %arg0, %arg1, %[[COUNT]]
// IR-SAME: cmpMode = 5 : i64
// IR-NOT: ascendc.compare_extend_contiguous %
// IR-NOT: afir.ne

// IR-LABEL: func.func @gt_reg_base_ui8
// IR: ascendc.compare_extend_contiguous_reg_base
// IR-SAME: cmpMode = 1 : i64

// IR-LABEL: func.func @le_scalar_reg_base_f32
// IR: ascendc.compare_scalar_extend_contiguous_reg_base
// IR-SAME: cmpMode = 3 : i64
// IR-NOT: afir.le

// CPP-LABEL: void ne_reg_base_f32(
// CPP: {{^ *}}CompareExtend<float, 1, CMPMODE::NE>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {static_cast<uint16_t>({{[^,]*}})}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)});
// CPP-LABEL: void gt_reg_base_ui8(
// CPP: {{^ *}}CompareExtend<uint8_t, 1, CMPMODE::GT>(
// CPP-LABEL: void le_scalar_reg_base_f32(
// CPP: {{^ *}}CompareScalarExtend<float, 1, CMPMODE::LE>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, {static_cast<uint16_t>({{[^,]*}})}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)});

// SELECT: ascend.implementation = {id = "ne.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "gt.tensor_tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "le.tensor_scalar.extend_reg_base", kind = "ascendc_api"}
