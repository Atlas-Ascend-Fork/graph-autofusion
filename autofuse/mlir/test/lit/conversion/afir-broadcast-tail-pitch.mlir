// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 目的地尾轴有填充时，两段窗口那一位生成的是行距而不是行长。
//
// 片上 tile 的尾轴行距被补到 32B（f32 → 8 个元素，f16 → 16 个），行长更短时
// `vectorized_strides` 的次内一位与内段 extent 不等。此前 `classifySegments` 要求两者相等
// （「目的地在窗口里稠密」），于是所有尾轴不满一整档的形状都被拒、整图回落 legacy
// ——2026-10-12 实测 mem 侧 112 张图里 33 次拒收全部命中那一条，其余判据一次都没走到
// （`test/probe/CapabilityBoundaryProbe.cpp`）。
//
// 这一格的 descriptor 刻意让 `strides` 与 `vectorized_strides` 不一致：逻辑步长 8（稠密）、
// 真实行距 16。合轴用的是后者，所以它就是「行长 8、行距 16」那个形状。
//
// 为什么按行距传是对的：包装的 `dst_k` 身兼行长与行距两职
// （`BroadcastFirstDim` 的 `dst_inner_offset`），传行距之后行与行落在 `0, 16, 32, …`，正是
// 带填充的真实布局；每行多算的 8 个元素落进填充位，而填充位 don't care（`afir.store` 读这
// 块 tile 时 UB 侧按行距、GM 侧按行长，逐行跳过它们）。两侧一起换是因为包装要求
// `srcShape[i] == dstShape[i]`，只换一边一个重载都对不上。
//
// 另一条顺带满足的：行距恒是一整档，所以按行距传同时满足了包装对尾轴 32B 对齐的要求，
// 而按行长传不满足。
//
// 形态 B（最内那一段就是广播段）带填充也收了（2026-10-26），但换的是另一边——只换目的
// 地那一位，因为它的源那一位恒是字面量 1。正例在 `afir-broadcast-tail-pitch-inner-brc.mlir`，
// 两格配成一对。
//
// 变异验证（2026-10-12）：
//   - 判据改回 `provablyEqual(outer.strides[dst], inner.extent)` 的等值要求
//     → IR / CPP / SELECT 三条都红（能力层就报 `no supported`，根本没选中这一档）。
//   - `emitWindowCall` 里去掉 pitch 替换、仍传内段 extent
//     → IR / CPP 红，SELECT 仍绿。这正是要区分的那一半：判据没动、这一档照样被选中，
//       只是生成的数错了——能力层查不出这种错，只有钉实参的 CHECK 查得出。
//   - 只换 `dst_k` 不换 `src_k` → 同上，IR / CPP 红、SELECT 绿。
// 三条都实测过（2026-10-12，逐条改完重编 af-opt 再跑）。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// 形态 A `[brc 4][normal 8]`，目的地行距 16、行长 8。
#pad_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"8">],
                             strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1],
                             vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#pad_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                             strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                             vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">], role = <compute>>

module {
  func.func @broadcast_padded_tail(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                   %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#pad_src],
      afir.destination_descs = [#pad_dst],
      afir.result_descs = [#pad_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 这一档确实被选中了（不是回落到别的 implementation，也不是报不支持）。
// SELECT: id = "broadcast.tensor.segmented"

// IR-LABEL: func.func @broadcast_padded_tail
// `dst_m` 仍是外段 extent 4，而行长那一位两侧都是行距 16。
// IR: %[[K:.*]] = arith.constant 4 : index
// IR: %[[ZERO:.*]] = arith.constant 0 : index
// IR: %[[ONE:.*]] = arith.constant 1 : index
// IR: %[[PITCH:.*]] = arith.constant 16 : index
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[ONE]], %[[PITCH]], %[[ZERO]], %[[K]], %[[PITCH]], %[[ZERO]], %arg2
// 两段窗口一条调用收完，不建循环。
// IR-NOT: scf.for

// kernel 文本侧的同一件事。这一条比 IR 那几条结实：它不关心中间产生了哪些常量，
// 只要求传给包装的那两个 `k` 就是行距。
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c16_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c16_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});
