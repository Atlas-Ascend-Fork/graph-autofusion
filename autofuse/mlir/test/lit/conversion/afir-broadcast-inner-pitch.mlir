// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 形态 A [brc][normal] 的两侧内段步长均为 8，支持内段中的对齐填充。
// 上游合轴剔除 extent 为 1 的最内轴后，其 32B 对齐填充可能表现为保留轴上的非单位步长。
// fuzz 中源布局为 [1,2]、步长为 (0, 8) 的广播即属于此形态。
//
// 两侧内段步长相等时，窗口调用复制整个行距，保持有效元素的位置；填充位置的值不参与计算。
// legacy 的 GetLatterMergedSize 同样使用目的地前一组最内轴的步长作为行距。
// 两侧内段步长不等时仍不支持，负例见 afir-broadcast-unsupported.mlir 的第二个片段。
//
// WrapperPitchTest.cpp 的 InnerPitchKeepsPaddedElementPositions 使用相同实参验证数值：
// Broadcast(dst, src, 1, 16, 0, 4, 16, 0, tmp, 1)。源有效元素位于下标 0 和 8，
// 检查目的地四行的相应位置；填充位置不参与比对。
// 现有 E2E 构图尚未生成该 UB 布局，因此数值测试直接调用包装函数。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

// 形态 A `[brc 4][normal 2]`：源 `(0, 8)`、目的地 `(16, 8)`。一行两个有效元素落在 0 与 8，
// 行距 16。
#ipitch_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"2">],
                                strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                vectorized_strides = [#afir.expr<"0">, #afir.expr<"8">], role = <compute>>
#ipitch_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"2">],
                                strides = [#afir.expr<"2">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                vectorized_strides = [#afir.expr<"16">, #afir.expr<"8">], role = <compute>>

module {
  func.func @broadcast_inner_pitch(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                   %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#ipitch_src],
      afir.destination_descs = [#ipitch_dst],
      afir.result_descs = [#ipitch_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 确认选择分段广播实现。
// SELECT: id = "broadcast.tensor.segmented"

// IR-LABEL: func.func @broadcast_inner_pitch
// IR: %[[K:.*]] = arith.constant 4 : index
// IR: %[[ZERO:.*]] = arith.constant 0 : index
// IR: %[[ONE:.*]] = arith.constant 1 : index
// IR: %[[PITCH:.*]] = arith.constant 16 : index
// 两侧的行长位一起换成行距 16；`last_dim_stride` 传 1——源行距在形态 A 由同一位承载。
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[ONE]], %[[PITCH]], %[[ZERO]], %[[K]], %[[PITCH]], %[[ZERO]], %arg2, %[[ONE]]
// 两段窗口由一次调用完成，无须生成循环。
// IR-NOT: scf.for

// 生成源码检查：两侧的 `k` 都必须是 16。生成 2（内段 extent）会只复制一行里的头
// 两个元素，第二个有效元素落不到位。
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c16_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c16_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}}, static_cast<uint32_t>(c1_idx));
