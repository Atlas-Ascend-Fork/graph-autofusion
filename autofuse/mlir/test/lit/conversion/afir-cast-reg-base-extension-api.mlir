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
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s --check-prefix=WRONGTARGET

// afir-cast-extension-api.mlir 的另一半：`cast_reg_base.h`，不要 tmp buffer，
// 三个实参是 brace-init 数组。
//
// 这是这条路上第一个 brace-init 数组形态的调用。 包装收的是三个
// `const uint32_t (&)[dim]`（输出各维长度、输出各维 stride、输入各维 stride），`dim` 由
// brace-init 的长度推出来——打一个元素就是在说连续形态。两个 stride 恒为 1，出自这个
// ascendc op 的助记符（`..._contiguous_...`）本身，不是从 IR 里读的。
//
// i16 -> i8 只有这一侧有：mem 侧 legacy 的 `kCastTypePairs` 里没有这一行（它的
// i8 一类只到 `f16 -> i8` / `ui8 -> i8`）。所以 `WRONGTARGET` 那条同时钉住 mem 侧
// 选不出实现。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @cast_i16_to_i8(%input: !ascendc.local_tensor<?xi16>,
                            %destination: !ascendc.local_tensor<?xi8>) {
    %result = afir.cast %input outs(%destination) {
      afir.operand_descs = [#input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : !ascendc.local_tensor<?xi16>, !ascendc.local_tensor<?xi8>
        -> !ascendc.local_tensor<?xi8>
    return
  }
}

// IR-LABEL: func.func @cast_i16_to_i8
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg1
// IR-NEXT: ascendc.cast_extend_contiguous_reg_base %arg1, %arg0, %[[COUNT]]
// IR-NOT: afir.cast

// 与 mem 侧同名而实参完全不同：三个数组，没有 tmp buffer，也没有显式模板实参
// （`dim` 只能从 brace-init 推，写出 `<InT, OutT>` 就得把它也写出来）。
// CPP-LABEL: void cast_i16_to_i8(
// CPP: {{^ *}}CastExtend({{[^,]*}}, {{[^,]*}}, {static_cast<uint32_t>({{[^)]*}})}, {1U}, {1U});

// SELECT: ascend.implementation = {id = "cast.tensor.extend_reg_base", kind = "ascendc_api"}

// mem 侧的对表里没有 i16 -> i8。
// WRONGTARGET: no supported
