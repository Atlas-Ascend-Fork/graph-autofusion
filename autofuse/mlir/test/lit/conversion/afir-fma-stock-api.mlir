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

// step 3 接通的 `afir.fma`。它与 `clip_by_value` 共用三元那条 pattern，而这一份要钉的
// 恰恰是两者不共用的那一截：操作数次序在两层上不一样。
//
//   IR   `ascendc.fma_l2 dst, s0, s1, s2, count, tmp`   <- pattern 建出来的次序
//   C++  `AscendC::Fma(dst, s0, s1, s2, tmp, count)`    <- `adv_api/math/fma.h` 的次序
//
// 这不是重复的两条 CHECK：ODS 的 `arguments` 顺序是我们自己定的（照 pattern 定，省一条
// 新 pattern），把两者对上是 emitter 的事。哪一层写反了只会在其中一条上红。
//
// 另两条反向 CHECK：
//
//   - `CPP-NOT: FmaExtend` —— 这一类里别的 op（`where` / `clip_by_value`）走的是 fork 的
//     包装、不带 `AscendC::` 限定。`Fma` 没有包装，按那几个选择可用实现会生成一个不存在的调用。
//   - `CPP-NOT: Fma<` —— 没有显式模板实参。`fma.h` 的第一个模板形参是 `const FmaConfig&`
//     而不是 `typename T`，打一个 `Fma<float>(...)` 会落在 `config` 上、编不过。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @fma_f32(%x1: !ascendc.local_tensor<?xf32>,
                     %x2: !ascendc.local_tensor<?xf32>,
                     %x3: !ascendc.local_tensor<?xf32>,
                     %destination: !ascendc.local_tensor<?xf32>,
                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.fma %x1, %x2, %x3 outs(%destination) tmp(%tmp) {
      afir.operand_descs = [#input_desc, #input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR 上 count 在 tmp 之前（与 `clip_by_value` 的那一条逐字同形）。
// IR-LABEL: func.func @fma_f32
// IR: %[[COUNT:.*]] = ascendc.local_tensor.get_size %arg3
// IR-NEXT: ascendc.fma_l2 %arg3, %arg0, %arg1, %arg2, %[[COUNT]], %arg4
// IR-NOT: afir.fma

// C++ 上 tmp 在 count 之前，正好相反。计数那一位捕下来比对，是为了钉住「最后一个实参
// 是计数」——只写一条 `AscendC::Fma(` 的话，tmp 与 count 调个头照样过。
// CPP-LABEL: void fma_f32(
// CPP: int32_t [[COUNT:v[0-9]+]] = [[DST:v[0-9]+]].GetSize();
// CPP-NEXT: AscendC::Fma([[DST]], {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, [[COUNT]]);
// CPP-NOT: FmaExtend
// CPP-NOT: Fma<

// SELECT: ascend.implementation = {id = "fma.tensor", kind = "ascendc_api"}

// 只有 reg 侧的 implementation，mem 目标上一条都选不出来。
// WRONGTARGET: no supported
