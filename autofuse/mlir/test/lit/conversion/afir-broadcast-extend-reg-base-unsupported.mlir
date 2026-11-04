// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// **reg 侧收不了的形态必须报不支持，不能顺手发一条 shape 数组。**
//
// `--split-input-file` 的理由同 `afir-broadcast-unsupported.mlir`：pass 在第一个报错的 op 上
// 就停。两片各只违反一条判据。
// RUN: not af-opt --split-input-file --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 \
// RUN:   | FileCheck %s

// 第一片：**带填充，而且最内那一段是广播段**——这是 2026-10-11 放宽之后的新边界。
//
// 带填充本身现在收了（`afir-broadcast-extend-padded.mlir` 是那条正例）：最内那一位改发行距，
// 两侧发同一个数。但那条改法要求两侧的最内位是**同一位**——最内那一段是广播段时
// `src_shape` 那一位是 1、源自己的最内参与位在更外面，两侧要对上的是另一对数。那是一个
// 独立的形状，今天没有用例，按「宁可拒收」留着。
//
// 形状：目的地 `stride = [8, 1]`、段长 `[4, 5]`，行距 8 ≠ 行长 5（带填充）；而内段源 stride
// 是 0（广播段）、外段源 stride 是 1。
#inner_brc_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"1">],
                                   strides = [#afir.expr<"1">, #afir.expr<"0">], vectorized_axis = [0, 1],
                                   vectorized_strides = [#afir.expr<"1">, #afir.expr<"0">], role = <compute>>
#inner_brc_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"5">],
                                   strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                   vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module {
  func.func @broadcast_padded_inner_brc(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.broadcast %src outs(%dst) {
      afir.operand_descs = [#inner_brc_src],
      afir.destination_descs = [#inner_brc_dst],
      afir.result_descs = [#inner_brc_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 第二片：源在非广播段上不稠密。
//
// 三段 `[normal 3][brc 4][normal 8]`，目的地稠密（32 = 8 × 4、8 = 1 × 8），但源在最外那段上的
// stride 是 16 而不是 8——`src_shape = {3, 1, 8}` 描述的稠密张量上，那一维的跨度只能是 8。
// 目的地那一侧与正例逐字相同，所以这一片违反的只有源这一条。
//
// **这一片是 2026-10-11 那次放宽的回归闸门**：那次改动一度把「验稠密」与「发什么数」解耦，
// 源这一条就漏查了（相邻两个参与位之间那个等式没人查），这一格当场变绿。判据必须对着真要
// 发出去的那几个数查，见 `sideIsDense`。
#loose_src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"3">, #afir.expr<"1">, #afir.expr<"8">],
                               strides = [#afir.expr<"16">, #afir.expr<"0">, #afir.expr<"1">],
                               vectorized_axis = [0, 1, 2],
                               vectorized_strides = [#afir.expr<"16">, #afir.expr<"0">, #afir.expr<"1">],
                               role = <compute>>
#loose_dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"3">, #afir.expr<"4">, #afir.expr<"8">],
                               strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                               vectorized_axis = [0, 1, 2],
                               vectorized_strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                               role = <compute>>

module {
  func.func @broadcast_loose_source(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.broadcast %src outs(%dst) {
      afir.operand_descs = [#loose_src],
      afir.destination_descs = [#loose_dst],
      afir.result_descs = [#loose_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// CHECK-COUNT-2: no supported
