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

// 第二个片段：验证形态 A 两侧内段步长必须相等的约束。
//
// 目的地是稠密的（外段 stride 8 == 内段 extent 8、内段 stride 1），而源在内段上的 stride 是 2。
// 两侧步长不等时需要重排，整行复制无法实现源步长为 2、目的地步长为 1 的映射。
// 两侧步长相等的填充布局已支持，正例见 `afir-broadcast-inner-pitch.mlir`。
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

// -----

// 第三个片段：验证形态 B 的窗口外段不能是广播段的约束。
//
// 三段 `[normal 4][brc 3][brc 2]`：最内那一段是广播段（形态 B），而窗口外段也是——两段之间
// 目的地有填充（`3 != 1 × 2`），无法合并。源在这两段上的 stride 均为 0；若将外段步长作为源
// 行距传给 `last_dim_stride`，会产生错误结果，因此根据段类型直接判为不支持。
//
// 本片段于 2026-11-15 新增：此前由形态 B 外段步长必须为字面量 1 的限制排除该布局，放宽为
// 任意源行距后，需要单独检查此约束。
#adjbrc_src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"4">, #afir.expr<"1">, #afir.expr<"1">],
                                strides = [#afir.expr<"1">, #afir.expr<"0">, #afir.expr<"0">],
                                vectorized_axis = [0, 1, 2],
                                vectorized_strides = [#afir.expr<"1">, #afir.expr<"0">, #afir.expr<"0">],
                                role = <compute>>
#adjbrc_dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"4">, #afir.expr<"3">, #afir.expr<"2">],
                                strides = [#afir.expr<"6">, #afir.expr<"2">, #afir.expr<"1">],
                                vectorized_axis = [0, 1, 2],
                                vectorized_strides = [#afir.expr<"12">, #afir.expr<"3">, #afir.expr<"1">],
                                role = <compute>>

module {
  func.func @broadcast_window_outer_is_broadcast(%src: !ascendc.local_tensor<?xf32>,
                                                 %dst: !ascendc.local_tensor<?xf32>,
                                                 %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#adjbrc_src],
      afir.destination_descs = [#adjbrc_dst],
      afir.result_descs = [#adjbrc_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 第四个片段：验证形态 B 的行内元素必须连续的约束。
//
// 形态 B `[normal 4][brc 2]`，源行距为 8（已支持），但目的地在广播那一位上的 stride 是
// 2 而不是 1——`AscendC::Broadcast<T, 2, 1>` 与 `BroadcastWithStride` 写出去的都是连续的一行，
// 不支持行内跨步写入。
#rowgap_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"1">],
                                strides = [#afir.expr<"1">, #afir.expr<"0">], vectorized_axis = [0, 1],
                                vectorized_strides = [#afir.expr<"8">, #afir.expr<"0">], role = <compute>>
#rowgap_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"2">],
                                strides = [#afir.expr<"2">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                vectorized_strides = [#afir.expr<"16">, #afir.expr<"2">], role = <compute>>

module {
  func.func @broadcast_row_not_contiguous(%src: !ascendc.local_tensor<?xf32>,
                                          %dst: !ascendc.local_tensor<?xf32>,
                                          %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#rowgap_src],
      afir.destination_descs = [#rowgap_dst],
      afir.result_descs = [#rowgap_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 第五个片段：验证内段步长大于 1 时必须取得物理行距的约束。
//
// 形态 A `[brc 4][normal 2]`，两侧内段 stride 均为 8（已支持），但目的地外段 stride
// 是 2，等于内段 extent，无法取得用于替换窗口参数的行距，调用将按 2 传参。包含填充的一行需要
// `2 × 8 = 16` 个元素的空间，而行距仅为 2，无法按整行复制正确实现，因此判为不支持。
#norow_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"2">],
                               strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1],
                               vectorized_strides = [#afir.expr<"0">, #afir.expr<"8">], role = <compute>>
#norow_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"2">],
                               strides = [#afir.expr<"2">, #afir.expr<"1">], vectorized_axis = [0, 1],
                               vectorized_strides = [#afir.expr<"2">, #afir.expr<"8">], role = <compute>>

module {
  func.func @broadcast_inner_pitch_without_row_pitch(%src: !ascendc.local_tensor<?xf32>,
                                                     %dst: !ascendc.local_tensor<?xf32>,
                                                     %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#norow_src],
      afir.destination_descs = [#norow_dst],
      afir.result_descs = [#norow_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 五个片段均应报告不支持，以确认各项约束均被检查。
//
// 2026-10-26 从三片减到两片：原第二片是「形态 B 带填充」，它已经收了（正例在
// `afir-broadcast-tail-pitch-inner-brc.mlir`）。那一片当初的理由是「两侧要对上的是另一对
// 数，不是生成同一个行距能解决的」——前半句对、后半句不必要：形态 B 的源那一位恒是字面量 1，
// 只换目的地那一边就对，数值在 CPU 模拟器上验过。
//
// 2026-11-15 扩展为五个片段：支持源行距和内段填充后（正例见 `afir-broadcast-source-pitch.mlir`
// 与 `afir-broadcast-inner-pitch.mlir`），为新增的三项约束分别补充负例。
// CHECK-COUNT-5: no supported
