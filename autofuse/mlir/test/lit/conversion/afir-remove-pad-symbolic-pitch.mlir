// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 32 B 整除判据的符号支：行距是 `af::sym::Align(z_size, 8)` 时判据必须成立。
//
// 调度算出来的行距就是这个形状（`base_alignment_strategy.cpp:705` 的
// `af::sym::Align(repeat, align_factor)`，而 `Align(x, a) = Mul(Ceiling(Div(x, a)), a)`），
// 规范形式是 `(8 * Ceiling((Rational(1 , 8) * z_size)))`。动态 shape 下这是常态，不是边角。
//
// 判据走的是表达式树的结构（`Mul` 里有一个字面量因子 8，其余因子取值为整数），不是算术
// 证明：后者要证 `P == 8 * Ceiling(P / 8)`，代进 `P = 8 * Ceiling(z_size / 8)` 之后左右差一层
// `Ceiling(Ceiling(...))`，而 SymEngine 的规范化只在实参是 `Number` 时求值，不折它。
// 这一条红了说明结构判据没命中，那时要改的是 `isProvablyMultipleOf` 而不是本用例。
//
// 只跑到 `--afir-select-implementation` 为止：符号行距物化出来是一条 `afir.tiling_param`，
// 而带它的函数必须有 workspace / TilingData 形参（`AscendCKernelABI` 的那条检查）——那要
// 一整张带 `afir.asc_graph` 的图，而本轮没有一条含 `RemovePad` 的图能端到端下降（理由见
// `afir-remove-pad.mlir` 的头注释）。行距物化本身由 `materializeStrideValue` 做，它与
// 广播整块复制那一趟共用同一条路，那一侧有覆盖。
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s

#sym_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"z_size">],
                             strides = [#afir.expr<"(8 * Ceiling((Rational(1 , 8) * z_size)))">,
                                        #afir.expr<"1">],
                             vectorized_axis = [0, 1],
                             vectorized_strides = [#afir.expr<"(8 * Ceiling((Rational(1 , 8) * z_size)))">,
                                                   #afir.expr<"1">], role = <compute>>
#sym_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"z_size">],
                             strides = [#afir.expr<"z_size">, #afir.expr<"1">], vectorized_axis = [0, 1],
                             vectorized_strides = [#afir.expr<"z_size">, #afir.expr<"1">], role = <compute>>

module {
  func.func @remove_pad_symbolic_pitch(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#sym_src],
      afir.destination_descs = [#sym_dst],
      afir.result_descs = [#sym_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// CHECK: ascend.implementation = {id = "remove_pad.tensor.extend", kind = "ascendc_api"}
