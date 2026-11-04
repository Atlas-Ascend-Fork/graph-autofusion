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

// `afir.cast` 的 mem 侧：`CastExtend(dst, src, tmp, size)`，扩展头 `cast.h`，要一块
// tmp buffer。
//
// **两侧不像前面那些 op 那样只差一个头——签名本身不同**（reg 侧收三个数组、不要 tmp），
// 所以这两份用例的 CPP check 也逐字不同，不像 `remainder` / `pow` 那样只能靠 IR 分辨。
//
// 取 f32 -> f16：两侧的对表里都有这一行。只有一侧有的那些由另一份钉。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @cast_f32_to_f16(%input: !ascendc.local_tensor<?xf32>,
                             %destination: !ascendc.local_tensor<?xf16>,
                             %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.cast %input outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf16>
    return
  }
}

// IR-LABEL: func.func @cast_f32_to_f16
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.cast_extend_contiguous %arg1, %arg0, %arg2, %[[COUNT]]
// IR-NOT: ascendc.cast_extend_contiguous_reg_base
// IR-NOT: afir.cast

// 包装，不是标准调用：没有 `AscendC::` 限定，也没有 `RoundMode` 实参——包装自己按
// (输入, 输出) 推。显式模板实参取的是**源**的元素类型，`OutT` 由 dst 推出来。
// CPP-LABEL: void cast_f32_to_f16(
// CPP: {{^ *}}CastExtend<float>({{[^,]*}}, {{[^,]*}}, {{[^,]*}}, static_cast<uint32_t>(
// CPP-NOT: AscendC::RoundMode

// SELECT: ascend.implementation = {id = "cast.tensor.extend", kind = "ascendc_api"}
