// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-vector-func-to-reg-base %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-vector-func-to-reg-base %s | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP

// `afir.vector_func` 的下降（M1）。它钉三件事：
//
//   1. **边界语义**：region 内只有计算，UB <-> 寄存器的搬运是下降**生成**的，不是从子图
//      翻过来的。所以产物里 `LoadAlign` 恰好等于 src 数、`StoreAlign` 恰好等于 dst 数——
//      **计数式而不是「存在一条」**：少发一条 Load 也能编过、也能算对（上一趟的寄存器还在），
//      要到别的形状上才暴露。
//   2. **产物形状**：一个 `__VEC_SCOPE__`、里面恰好一层 lane 循环，寄存器声明在循环外。
//   3. **算子顺序**：`add` 在 `abs` 之前。顺序错了结果就是另一个函数，而两种都编得过。
//
// 不在这里钉的：形状相关的判据（lane 一段多少个元素由目标的 `GetVecLen()` 定，这里是
// 符号尺寸）、以及能力不足的四条（它们要看诊断，在 `afir-vector-func-unsupported.mlir`）。

#in = #afir.tensor_desc<axis = [0], role = <input>>
#out = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @vector_func_add_abs(%a: !ascendc.local_tensor<?xf32>, %b: !ascendc.local_tensor<?xf32>,
                                 %y: !ascendc.local_tensor<?xf32>) {
    %0 = afir.vector_func srcs(%a, %b) dsts(%y) attributes {
      afir.operand_descs = [#in, #in],
      afir.destination_descs = [#out],
      afir.result_descs = [#out]
    } {
    ^bb0(%la: tensor<?xf32>, %lb: tensor<?xf32>):
      %sum = afir.add %la, %lb {afir.operand_descs = [#in, #in], afir.result_descs = [#out]}
          : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      %res = afir.abs %sum {afir.operand_descs = [#out], afir.result_descs = [#out]}
          : tensor<?xf32> -> tensor<?xf32>
      afir.vector_func_yield %res : tensor<?xf32>
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>)
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 作用域外：三个 UB 首地址、一个元素数、一段 lane 的元素数、循环次数、以及 `UpdateMask`
// 要的那个可变计数器。
// IR-LABEL: func.func @vector_func_add_abs
// IR: ascendc.local_tensor.get_phy_addr_v2
// IR: ascendc.local_tensor.get_size
// IR: ascendc.get_vec_len
// IR: emitasc.variable

// 寄存器声明在作用域里、循环外：src 两个加中间结果两个。循环里重复声明在 C++ 上也合法，
// 但那样每趟都多一条，与 legacy 的产物对不起来。
// IR: emitasc.vec_scope
// IR-COUNT-4: ascendc.reg_tensor
// IR-NOT: ascendc.reg_tensor

// 恰好一层 lane 循环。两层意味着有人把 tile 循环也生成进了 region，而那样每个元素会被
// 算两遍——**结果仍然对**（幂等的那几个 op 上尤其），所以只能靠计数钉住。
// IR-COUNT-1: emitasc.vf_for
// IR-NOT: emitasc.vf_for

// 循环体：取两次、算两步、存一次，顺序固定。
// IR: ascendc.update_mask
// IR-COUNT-2: ascendc.load_align_reg
// IR-NOT: ascendc.load_align_reg
// IR: ascendc.add_reg
// IR: ascendc.abs_reg
// IR-COUNT-1: ascendc.store_align_reg
// IR-NOT: ascendc.store_align_reg

// region 没有了，GM <-> UB 的搬运一条也不该被生出来：子图边界的 `Load` / `Store` 表达的是
// UB <-> 寄存器，翻成 `afir.load` / `afir.store` 会多出两趟搬运，而那样也编得过、也算得对。
// IR-NOT: afir.vector_func
// IR-NOT: afir.load
// IR-NOT: afir.store

// 发射出来是 `__VEC_SCOPE__` 加一层 `for`，调的是 `AscendC::Reg::` 那一族（legacy 写
// `MicroAPI::`，`namespace MicroAPI = Reg`，同一个函数）。
// CPP-LABEL: void vector_func_add_abs(
// CPP: __VEC_SCOPE__
// CPP: for (uint16_t
// CPP: AscendC::Reg::UpdateMask<float>(
// CPP-COUNT-2: AscendC::Reg::LoadAlign<float>(
// CPP: AscendC::Reg::Add(
// CPP: AscendC::Reg::Abs(
// CPP-COUNT-1: AscendC::Reg::StoreAlign<float>(
