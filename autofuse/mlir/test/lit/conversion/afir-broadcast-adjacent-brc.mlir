// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// mem 侧分段广播，两个广播段相邻：`[brc][brc][normal]`。
//
// 这是 `classifySegments` 删掉那条相邻判据所要的正例。此前它拒收这种形状、整图回落 legacy，
// 理由写着「推下来是对的，但没有用例证明」——这个文件就是那条用例。
//
// 为什么相邻的广播段并不上：合轴要求所有参与张量都满足
// `stride[外] == stride[内] × 段长[内]`。源在两个广播段上 stride 都是 0，`0 == 0 × 4` 成立、
// 不反对；挡住合并的是目的地在两段之间有填充——`stride[z0] = 64`，而
// `stride[z1] × 段长(z1) = 8 × 4 = 32`，64 ≠ 32，所以两段留着。设计文档里「相邻的广播段在
// 合轴那一步就并成一段了」那句话因此不完全成立。
//
// 为什么分解是对的：外层每一趟按本段在目的地上的 stride 整块复制
// （`emitBlockCopy` 取 `segment.strides[destination]`，不是段长之积），所以两段之间有填充也走
// 对地址；填充位一起复制，那些位置本来就是 don't care。
//
// 填充刻意只放在两个广播段之间，没放进最内那两段的窗口：窗口里 `dst_k` 既是行长也是行距
// （`BroadcastFirstDim` 的 `dst_inner_offset`），那里带填充是 mem 侧另一条判据的事
// （reg 侧的那一条见 `afir-broadcast-extend-padded.mlir`），不是本文件要钉的。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// 段形态 `[brc 2][brc 4][normal 8]`。目的地 stride `[64, 8, 1]`：最内两段稠密
// （8 == 1 × 8），而 `z0` 那一段带填充（64 ≠ 8 × 4）。
#src = #afir.tensor_desc<axis = [0, 1, 2],
                         repeats = [#afir.expr<"1">, #afir.expr<"1">, #afir.expr<"8">],
                         strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                         vectorized_axis = [0, 1, 2],
                         vectorized_strides = [#afir.expr<"0">, #afir.expr<"0">, #afir.expr<"1">],
                         role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1, 2],
                         repeats = [#afir.expr<"2">, #afir.expr<"4">, #afir.expr<"8">],
                         strides = [#afir.expr<"64">, #afir.expr<"8">, #afir.expr<"1">],
                         vectorized_axis = [0, 1, 2],
                         vectorized_strides = [#afir.expr<"64">, #afir.expr<"8">, #afir.expr<"1">],
                         role = <compute>>

module {
  func.func @broadcast_adjacent_brc(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#src],
      afir.destination_descs = [#dst],
      afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 收下了（此前这一格报 unsupported）。
// SELECT: ascend.implementation = {id = "broadcast.tensor.segmented", kind = "ascendc_api"}

// 两个广播段 → 两趟，由内而外，趟与趟之间一条 `PipeBarrier<PIPE_V>`。
// IR-LABEL: func.func @broadcast_adjacent_brc
// IR: ascendc.pipe_barrier
