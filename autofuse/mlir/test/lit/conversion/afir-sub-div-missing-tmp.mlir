// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 | FileCheck %s

// `TmpBufferUse::Unused` 与 `None` 的分界线，单独一份用例钉住。
//
// mem 侧的 `afir.sub` 发的是不带 tmp 的 `AscendC::Sub`，**但调度一定会给它一块**
// （legacy 的 `CalcBinaryApiTmpSize` 对 tensor / 标量两支都返回一块）。所以「没有 tmp
// 操作数」不是「这条路不需要它」，而是「这张图不是这条路算出来的」，要判不支持。
//
// 把 `BinaryElementwiseConversionPattern` 里 `schedulerGivesTmpBuffer(Tmp)` 那一支删掉，
// 这份用例立刻变绿——这是它存在的理由。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

module {
  func.func @sub_without_tmp(%lhs: !ascendc.local_tensor<?xf32>, %rhs: !ascendc.local_tensor<?xf32>,
                             %destination: !ascendc.local_tensor<?xf32>) {
    %result = afir.sub %lhs, %rhs outs(%destination) {
      afir.operand_descs = [#input_desc, #input_desc],
      afir.destination_descs = [#compute_desc],
      afir.result_descs = [#compute_desc]
    } : (!ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>),
        !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// CHECK: no supported
