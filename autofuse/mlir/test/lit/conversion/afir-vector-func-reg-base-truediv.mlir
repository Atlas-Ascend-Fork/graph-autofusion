// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-vector-func-to-reg-base %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-vector-func-to-reg-base %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

// `afir.truediv` 进 VF 的 region（2026-10-26 加的那一行算子表）。
//
// 这一格钉的是「生成的是 `Div` 而不是别的」：`truediv` 在 reg 侧与 `div` 生成同一个调用
// ——legacy 的 `ascir_op_lowering_matrix.csv` 里 `TrueDiv` 的 `reg_api_name` 就是
// `DivExtend`（mem 侧才是 `TrueDivExtend` 加 tmp buffer，那是另一条路）。所以这一行与
// legacy 逐字一致，而不是我们这一侧挑了个近似；钉住 `Reg::Div` 就是钉住这件事。
//
// 为什么要这一格：`truediv_bf16_test` 这张 legacy 图此前报
// `'afir.truediv' has no register-form lowering`，整图回落。算子表加一行要带一格用例，
// 这是那一格。
//
// 形状与计数那几条已经由 `afir-vector-func-reg-base.mlir` 钉过（一个 vec_scope、一层 lane
// 循环、Load / Store 计数式），这里不重复，只钉本行新增的那一件事。

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @vector_func_truediv(%a: !ascendc.local_tensor<?xf32>, %b: !ascendc.local_tensor<?xf32>,
                                 %y: !ascendc.local_tensor<?xf32>) {
    %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
      afir.operand_descs = [#in, #in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xf32>, %lb: tensor<?xf32>):
      %res = afir.truediv %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
          : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      afir.vector_func_yield %res : tensor<?xf32>
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>)
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @vector_func_truediv
// IR: emitasc.vec_scope
// IR-COUNT-2: ascendc.load_align_reg
// IR: ascendc.div_reg
// IR-COUNT-1: ascendc.store_align_reg
// region 没有了，也没有多生出 GM <-> UB 的搬运。
// IR-NOT: afir.vector_func
// IR-NOT: afir.truediv

// 生成的是 `Reg::Div`，不是 `Reg::TrueDiv`（reg 侧没有那个函数）。
// CPP-LABEL: void vector_func_truediv(
// CPP: AscendC::Reg::Div(
// CPP-NOT: AscendC::Reg::TrueDiv(
