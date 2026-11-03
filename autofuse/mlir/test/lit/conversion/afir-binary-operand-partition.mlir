// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend910B1 %s
// RUN: af-opt --verify-diagnostics --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s

// 二元路径改按 `afir::getOperandPartition()` 切操作数、并加上 tmp buffer 判据之后，这份
// 负例钉住的是：**多出来的那块 tmp buffer 不会被当成第二个输入**。
//
// 以前这里是位置下标 `{0, 1, 2}` 加一句 `getNumOperands() != 3`。那句话同时兼着两件
// 事——「操作数个数对」和「没有 tmp buffer」——于是把它换成三段划分之后，很容易只保留
// 前一件：`afir.add %a, %b outs(%d) tmp(%t)` 的三段划分是 `2 | 1 | 1`，输入数、目的地
// 数全都对，判据一条不剩，那块 UB 就被默默丢掉了。
//
// 四个 op 两张供给表上是同一个调用、都不要 tmp buffer，所以两个 `--target` 共用同一份
// IR、同一条 expected-error——能触发的只剩 tmp buffer 这一条。

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

func.func @add_with_unwanted_tmp_buffer(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>,
                                     %destination: !ascendc.local_tensor<?xf16>,
                                     %tmp: !ascendc.local_tensor<?xui8>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.add %lhs, %rhs outs(%destination) tmp(%tmp) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>), !ascendc.local_tensor<?xf16>,
      !ascendc.local_tensor<?xui8> -> !ascendc.local_tensor<?xf16>
  return
}

// -----

#input_desc = #afir.tensor_desc<axis = [0], role = <input>>
#compute_desc = #afir.tensor_desc<axis = [0], role = <compute>>

// 两个输入的形状仍然要一致。以前由「整个类型逐字相等」顺带管着，拆成元素类型与形状
// 两条之后得单独钉住。
func.func @mul_with_mismatched_input_shapes(%lhs: !ascendc.local_tensor<4xf16>,
                                            %rhs: !ascendc.local_tensor<8xf16>,
                                            %destination: !ascendc.local_tensor<4xf16>) {
  // expected-error @+1 {{has no supported Ascend implementation}}
  %result = afir.mul %lhs, %rhs outs(%destination) {
    afir.operand_descs = [#input_desc, #input_desc],
    afir.destination_descs = [#compute_desc],
    afir.result_descs = [#compute_desc]
  } : (!ascendc.local_tensor<4xf16>, !ascendc.local_tensor<8xf16>), !ascendc.local_tensor<4xf16>
      -> !ascendc.local_tensor<4xf16>
  return
}
