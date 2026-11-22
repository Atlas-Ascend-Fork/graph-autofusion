// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 同一存储同时存在 MTE2（afir.load）和 V（afir.neg）写入者，两条流水线都必须纳入同步集合。
// 属性中的顺序按 pipe 枚举值固定为 MTE2、V。
//
// 本用例只检查分析结果。手工构造的 afir.load 缺少完整下降所需的 asc_graph 与 indexing_maps。
// 此用例不覆盖多事件的完整下降；单事件的生成检查见 afir-scalar-read-sync-across-control-flow.mlir。

// RUN: af-opt --afir-scalar-read-sync %s | FileCheck %s --check-prefix=ATTR

#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">],
                         vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"32">], strides = [#afir.expr<"1">],
                          vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
#gm_one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], role = <input>>

func.func @minimum_scalar_written_by_two_pipelines(%gm: !ascendc.global_tensor<1xf32>,
                                                   %input: !ascendc.local_tensor<32xf32>,
                                                   %dst: !ascendc.local_tensor<32xf32>,
                                                   %seed: !ascendc.local_tensor<1xf32>) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : <vecin, 2>
  %one = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<vecin, 2>, !ascendc.local_tensor<1xf32>
  %loaded = afir.load %gm outs(%one) {afir.operand_descs = [#gm_one], afir.destination_descs = [#one],
                                      afir.result_descs = [#one]}
      : !ascendc.global_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  %neg = afir.neg %seed outs(%one) {afir.operand_descs = [#one], afir.destination_descs = [#one],
                                    afir.result_descs = [#one]}
      : !ascendc.local_tensor<1xf32>, !ascendc.local_tensor<1xf32> -> !ascendc.local_tensor<1xf32>
  %r = afir.minimum %input, %one outs(%dst) {afir.operand_descs = [#full, #one],
                                             afir.destination_descs = [#full], afir.result_descs = [#full]}
      : (!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>), !ascendc.local_tensor<32xf32>
        -> !ascendc.local_tensor<32xf32>
  return
}

// ATTR-LABEL: func.func @minimum_scalar_written_by_two_pipelines
// ATTR: afir.minimum
// ATTR-SAME: afir.scalar_read_sync = {{\[}}[], [#afir<pipe transfer_in>, #afir<pipe vector>]]
