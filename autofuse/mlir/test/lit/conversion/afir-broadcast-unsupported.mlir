// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `--split-input-file`：pass 在第一个报错的 op 上就停，不分片的话后两格永远走不到，
// 「三条都要报出来」那条断言就成了空话。每一片自带 descriptor，别名不跨片可见。
// RUN: not af-opt --split-input-file --afir-select-implementation --target=Ascend910B1 %s 2>&1 \
// RUN:   | FileCheck %s

// 分段广播收不了的形态必须报不支持，不能顺手生成一条两段调用。
//
// 三片各只违反一条判据。一片里同时违反两条是查不出东西的：去掉其中一条判据，另一条
// 照样把它拒掉，变异验证不会变红——这份用例的第一版就栽在这上面。

// 第一片：广播段够不着最内两段窗口。
//
// 三段 `[brc 4][normal 3][normal 8]`。最内两段都是普通段、源在两段上都前进，而形态 A 要求
// 外段源 stride 为 0、形态 B 要求内段为 0，两条都过不去。
//
// 两根普通轴没有合并是这个用例要的：目的地在 axis1 上的 stride 是 16 而不是 8（内段元素
// 数），`stride[外] == stride[内] × extent[内]` 不成立、封段。源这一侧反倒是连续的
// （8 == 1 × 8）——逐对比较要所有 operand 都同意，一个否决就封段。
#outside_src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"1">, #afir.expr<"3">, #afir.expr<"8">],
                                 strides = [#afir.expr<"0">, #afir.expr<"8">, #afir.expr<"1">],
                                 vectorized_axis = [0, 1, 2],
                                 vectorized_strides = [#afir.expr<"0">, #afir.expr<"8">, #afir.expr<"1">],
                                 role = <compute>>
#outside_dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"4">, #afir.expr<"3">, #afir.expr<"8">],
                                 strides = [#afir.expr<"64">, #afir.expr<"16">, #afir.expr<"1">],
                                 vectorized_axis = [0, 1, 2],
                                 vectorized_strides = [#afir.expr<"64">, #afir.expr<"16">, #afir.expr<"1">],
                                 role = <compute>>

module {
  func.func @broadcast_segment_outside_window(%src: !ascendc.local_tensor<?xf32>,
                                              %dst: !ascendc.local_tensor<?xf32>,
                                              %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#outside_src],
      afir.destination_descs = [#outside_dst],
      afir.result_descs = [#outside_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 第二片：只违反「源的 stride 形态」。
//
// 目的地是稠密的（外段 stride 8 == 内段 extent 8），但源在内段上的 stride 是 2 而不是 1
// ——两轴 API 读的是一段连续的源，跨着读表达不了。
#gappy_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"8">],
                               strides = [#afir.expr<"0">, #afir.expr<"2">], vectorized_axis = [0, 1],
                               vectorized_strides = [#afir.expr<"0">, #afir.expr<"2">], role = <compute>>
#gappy_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                               strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                               vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module {
  func.func @broadcast_source_not_contiguous(%src: !ascendc.local_tensor<?xf32>,
                                             %dst: !ascendc.local_tensor<?xf32>,
                                             %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#gappy_src],
      afir.destination_descs = [#gappy_dst],
      afir.result_descs = [#gappy_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 两片都要报出来，少一条就说明对应那条判据没人挡着。
//
// 2026-10-26 从三片减到两片：原第二片是「形态 B 带填充」，它已经收了（正例在
// `afir-broadcast-tail-pitch-inner-brc.mlir`）。那一片当初的理由是「两侧要对上的是另一对
// 数，不是生成同一个行距能解决的」——前半句对、后半句不必要：形态 B 的源那一位恒是字面量 1，
// 只换目的地那一边就对，数值在 CPU 模拟器上验过。
// CHECK-COUNT-2: no supported
