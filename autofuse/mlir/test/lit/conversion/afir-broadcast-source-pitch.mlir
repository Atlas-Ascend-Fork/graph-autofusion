// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 形态 B [normal][brc] 的源有效元素按行距排列，支持非单位源步长。
// 2026-11-15 的 fuzz 复测中，22 个 Broadcast 失败中取得了 16 份布局样本，其中 15 份带有非单位
// 源步长。此前该布局因源步长不为 1 而被拒绝；现在通过第十个实参 last_dim_stride 传递源行距。
// 包装函数在该参数大于 1 时选择 BroadcastWithStride，取值与 legacy 的
// BroadcastGetLastDimStride 一致。src_k 仍须为 1，以选择正确分支。
//
// 源行距由 last_dim_stride 表示，目的地行距由 dst_k 表示。两者独立，本用例均取 8。
// WrapperPitchTest.cpp 的 SourcePitchReadsEveryRowStart 使用相同实参验证数值：
// Broadcast(dst, src, 4, 1, 0, 4, 8, 0, tmp, 8)。源有效元素位于下标 0、8、16、24，
// 目的地预填哨兵值后，检查每行两列是否等于对应源元素。
// 历史反向验证中，将最后一个实参改为 1 会错误地读取连续元素，数值断言随之失败。
// 现有 E2E 构图尚未生成该 UB 布局，因此数值测试直接调用包装函数。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// 形态 B `[normal 4][brc 2]`：源 `(8, 0)`——4 个有效元素落在 0、8、16、24；目的地 `(8, 1)`，
// 行距 8、行长 2。
#spitch_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"1">],
                                strides = [#afir.expr<"1">, #afir.expr<"0">], vectorized_axis = [0, 1],
                                vectorized_strides = [#afir.expr<"8">, #afir.expr<"0">], role = <compute>>
#spitch_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"2">],
                                strides = [#afir.expr<"2">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module {
  func.func @broadcast_source_pitch(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#spitch_src],
      afir.destination_descs = [#spitch_dst],
      afir.result_descs = [#spitch_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 确认选择分段广播实现。
// SELECT: id = "broadcast.tensor.segmented"

// IR-LABEL: func.func @broadcast_source_pitch
// IR: %[[M:.*]] = arith.constant 4 : index
// IR: %[[ZERO:.*]] = arith.constant 0 : index
// IR: %[[ONE:.*]] = arith.constant 1 : index
// IR: %[[PITCH:.*]] = arith.constant 8 : index
// `src_k` 是字面量 1、`dst_k` 是目的地行距 8、最后那个 `last_dim_stride` 是源行距 8。
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[M]], %[[ONE]], %[[ZERO]], %[[M]], %[[PITCH]], %[[ZERO]], %arg2, %[[PITCH]]
// 两段窗口由一次调用完成，无须生成循环。
// IR-NOT: scf.for

// 生成源码检查：第十个实参必须为 8。传 1 会选择连续读取分支，错误地读取源的前四个元素。
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}}, static_cast<uint32_t>(c8_idx));
