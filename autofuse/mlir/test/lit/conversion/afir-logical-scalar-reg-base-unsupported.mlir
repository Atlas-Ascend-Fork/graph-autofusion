// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 | FileCheck %s

// reg 侧的 `logical_or` 不收标量右操作数。 四个 impl 类里只有它没覆写
// `IsScalarInputSupported`，基类返回 `false`；矩阵上那一格写的就是 `no`。
//
// 这一条单独一份用例，因为它是整批里唯一一处「同一个 op、同一侧、tensor 形态能选中而
// 标量形态选不中」的不对称。把 `logical_or.tensor_scalar.extend_reg_base` 补注册上去
// （抄隔壁 `logical_and` 那行很容易顺手就抄了），这份用例立刻变绿——这是它存在的理由。
//
// 对照组在同一目录下的三份里：
//   - 同一侧、同一个 op、tensor 形态 —— afir-logical-reg-base-extension-api.mlir；
//   - 同一侧、`logical_and`、标量形态 —— afir-logical-scalar-reg-base-api.mlir；
//   - 另一侧、同一个 op、标量形态 —— afir-logical-scalar-api.mlir。
// 三份都是绿的，只有这一格是红的。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @logical_or_scalar_reg_base(%input: !ascendc.local_tensor<?xf16>,
                                        %destination: !ascendc.local_tensor<?xui8>) {
    %scalar = arith.constant 2.000000e+00 : f16
    %result = afir.logical_or %input, %scalar outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf16>, f16),
        !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xui8>
    return
  }
}

// CHECK: no supported
