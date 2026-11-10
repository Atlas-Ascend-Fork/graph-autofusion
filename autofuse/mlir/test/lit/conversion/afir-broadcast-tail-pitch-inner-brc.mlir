// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 形态 B（`[normal][brc]`，最内那一段就是广播段）带填充：只换目的地那一边。
//
// 与 `afir-broadcast-tail-pitch.mlir`（形态 A）配成一对，两格的区别就是换哪一边：
//
//   - 形态 A：两侧的最内位是同一位，`sourceK` 与 `destinationK` 生成的是同一个值，所以行距
//     要两边一起换，否则包装的 `srcShape[i] == dstShape[i]` 对不上。
//   - 形态 B（本格）：源那一位恒是字面量 1，所以只换得了目的地那一边——也只需要换那
//     一边。包装走 `broadcast.h:304` 的 `src_k == 1 && src_m == dst_m` 一支，它对两侧 shape
//     的要求只落在非广播那一位（`m`）上。
//
// 这一格 2026-10-26 之前是负例（`afir-broadcast-unsupported.mlir` 的第二片），理由写的是
// 「两侧要对上的是另一对数，不是生成同一个行距能解决的」。那句话只对了一半：确实不是「传同
// 一个行距」，但也不需要——只换目的地一边就对。实测证据是 CPU 模拟器上的数值
// （`test/e2e/cpu/cases/load_broadcast_segmented_store` 的 `_33x1` 与 `_2x3x5x7x1` 两格，
// 哨兵填充、逐元素比对），以及探针 mem 侧 42 格从 `unsupported` 翻成 `ok`、0 格变差。
//
// 这一格靠着一条正确性不变量：行距恒是 32B 的整数倍（填充的定义），所以
// `AscendC::Broadcast<T, 2, 1>` 要的对齐天然满足。**codegen 这一侧没有检查它**，与形态 A
// 那一格同一个敞口（见 `docs/development/2026-09-23-afir-broadcast-design.md`）；真被破了
// 是静默算错，不是报错。
//
// 变异验证（2026-10-26，逐条改完重编 af-opt 再跑）：
//   - `classifySegments` 里把 `broadcastIsInner` 那条早退加回去
//     → SELECT / IR / CPP 三条都红（能力层直接报 `no supported`）。
//   - `emitWindowCall` 去掉 pitch 替换、仍传内段 extent
//     → IR / CPP 红，SELECT 仍绿：这一档照样被选中，只是 `dst_k` 生成了 8。能力层查不
//       出这种错，只有钉实参的 CHECK 查得出。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// 形态 B `[normal 4][brc 8]`：源 `(1, 0)`，目的地行距 16、行长 8。
// `strides` 与 `vectorized_strides` 刻意不一致——逻辑步长 8（稠密）、真实行距 16，合轴用后者。
#padb_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"1">],
                              strides = [#afir.expr<"1">, #afir.expr<"0">], vectorized_axis = [0, 1],
                              vectorized_strides = [#afir.expr<"1">, #afir.expr<"0">], role = <compute>>
#padb_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                              strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                              vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">], role = <compute>>

module {
  func.func @broadcast_padded_tail_inner_is_broadcast(%src: !ascendc.local_tensor<?xf32>,
                                                      %dst: !ascendc.local_tensor<?xf32>,
                                                      %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#padb_src],
      afir.destination_descs = [#padb_dst],
      afir.result_descs = [#padb_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 这一档确实被选中了（不是回落到别的 implementation，也不是报不支持）。
// SELECT: id = "broadcast.tensor.segmented"

// IR-LABEL: func.func @broadcast_padded_tail_inner_is_broadcast
// `src_k` 是字面量 1、`dst_k` 是行距 16 —— **这一对数就是本格的全部内容**：只换目的地一边。
// IR: %[[M:.*]] = arith.constant 4 : index
// IR: %[[ZERO:.*]] = arith.constant 0 : index
// IR: %[[ONE:.*]] = arith.constant 1 : index
// IR: %[[PITCH:.*]] = arith.constant 16 : index
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[M]], %[[ONE]], %[[ZERO]], %[[M]], %[[PITCH]], %[[ZERO]], %arg2
// 两段窗口一条调用收完，不建循环。
// IR-NOT: scf.for

// kernel 文本侧的同一件事。这一条比 IR 那几条结实：它不关心中间产生了哪些常量，只要求
// 传给包装的 `src_k` 是 1、`dst_k` 是行距。
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c16_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});
