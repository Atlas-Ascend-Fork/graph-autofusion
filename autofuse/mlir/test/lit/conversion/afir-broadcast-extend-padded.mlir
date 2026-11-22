// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// reg 侧分段广播，目的地带尾轴填充：`dst_shape` 的最尾一位传行距而不是行长。
//
// 这是 `Broadcast.cpp` 的 `classifyDenseSide` 要的那条正例。此前这一档拒收带填充的
// descriptor、整图回落 legacy（`af-capability-probe` 量到的 16 条就在这一类），而 legacy 的
// `GenParams` 一直是按行距生成的。
//
// 形状为什么这么挑：尾轴 5 个 f32 = 20 字节，不是 32 的倍数，所以 `kAligned` 给次内层的
// stride 取 `Align(5, 32/4) = 8`，而内段段长是 5。老判据「`stride[外]` 等于其内各段段长之积」
// 当场不成立（8 ≠ 5）；新判据问的是相邻两位 `stride[外] == stride[内] × 段长[内]`，并把那一位
// 的 shape 改为传 8。多读的那 3 个位置落在目的地的填充列上，本来就是 don't care。
//
// 两侧生成同一个 8：`BroadcastExtendImpl` 在非广播维上要求 `srcShape[i] == dstShape[i]`
// （`broadcast_extend_impl.h:76`），各生成自己的数就谁也不满足。
//
// 对照：`afir-broadcast-extend-reg-base.mlir` 是同一档的不带填充形状，那里尾位生成的是行长，
// 两个文件一起读就看出「行距 == 行长」只是这条判据的特例。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

// 广播轴在外、带填充的尾轴在内：`[1,5] -> [4,5]`，目的地行距 8。
#src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"5">],
                         strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1],
                         vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"5">],
                         strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                         vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module {
  func.func @broadcast_padded_tail(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.broadcast %src outs(%dst) {
      afir.operand_descs = [#src],
      afir.destination_descs = [#dst],
      afir.result_descs = [#dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 收下了（此前这一格报 unsupported）。
// SELECT: ascend.implementation = {id = "broadcast.tensor.extend_reg_base", kind = "ascendc_api"}

// IR-LABEL: func.func @broadcast_padded_tail
// IR: ascendc.broadcast_extend_reg_base %{{.*}}, %{{.*}}
// IR-NOT: scf.for

// 尾位是 8（行距），不是 5（行长），而且两个数组都是 8。 这一位生成 5 的话
// `BroadcastExtend` 按一个比实际行距小的跨度走地址，从第二行起全错。
// CPP-LABEL: void broadcast_padded_tail(
// CPP: const uint32_t dst_shape_[[SRC:v[0-9]+]]_brc_to_[[DST:v[0-9]+]][2] = {static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx)};
// CPP-NEXT: const uint32_t src_shape_[[SRC]]_brc_to_[[DST]][2] = {static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx)};
// CPP-NEXT: BroadcastExtend<float, 2>([[DST]], [[SRC]], dst_shape_[[SRC]]_brc_to_[[DST]], src_shape_[[SRC]]_brc_to_[[DST]]);
