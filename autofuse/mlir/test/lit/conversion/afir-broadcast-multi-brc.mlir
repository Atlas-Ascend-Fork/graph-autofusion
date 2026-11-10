// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP

// 四段 `(1,B,1,B) -> (A,B,A,B)`，A=3 B=8。legacy 的四轴形态认得这个 pattern，但它把外层
// 普通段留在 API 内，于是第二趟落 `BroadcastMiddleDim` 的 `Copy`；分解之后两趟都是
// `BroadcastFirstDim`。
#d_src = #afir.tensor_desc<axis = [0, 1, 2, 3],
                           repeats = [#afir.expr<"1">, #afir.expr<"8">, #afir.expr<"1">, #afir.expr<"8">],
                           strides = [#afir.expr<"0">, #afir.expr<"8">, #afir.expr<"0">, #afir.expr<"1">],
                           vectorized_axis = [0, 1, 2, 3],
                           vectorized_strides = [#afir.expr<"0">, #afir.expr<"8">, #afir.expr<"0">,
                                                 #afir.expr<"1">],
                           role = <compute>>
#d_dst = #afir.tensor_desc<axis = [0, 1, 2, 3],
                           repeats = [#afir.expr<"3">, #afir.expr<"8">, #afir.expr<"3">, #afir.expr<"8">],
                           strides = [#afir.expr<"192">, #afir.expr<"24">, #afir.expr<"8">, #afir.expr<"1">],
                           vectorized_axis = [0, 1, 2, 3],
                           vectorized_strides = [#afir.expr<"192">, #afir.expr<"24">, #afir.expr<"8">,
                                                 #afir.expr<"1">],
                           role = <compute>>

// 五段 `(M,1,N,1,K) -> (M,B1,N,B2,K)`，M=2 B1=3 N=4 B2=2 K=8。
// legacy 在这个形状上直接 `ASSERT(false)`——它只有三轴与四轴两个重载。
#e_src = #afir.tensor_desc<axis = [0, 1, 2, 3, 4],
                           repeats = [#afir.expr<"2">, #afir.expr<"1">, #afir.expr<"4">, #afir.expr<"1">,
                                      #afir.expr<"8">],
                           strides = [#afir.expr<"32">, #afir.expr<"0">, #afir.expr<"8">, #afir.expr<"0">,
                                      #afir.expr<"1">],
                           vectorized_axis = [0, 1, 2, 3, 4],
                           vectorized_strides = [#afir.expr<"32">, #afir.expr<"0">, #afir.expr<"8">,
                                                 #afir.expr<"0">, #afir.expr<"1">],
                           role = <compute>>
#e_dst = #afir.tensor_desc<axis = [0, 1, 2, 3, 4],
                           repeats = [#afir.expr<"2">, #afir.expr<"3">, #afir.expr<"4">, #afir.expr<"2">,
                                      #afir.expr<"8">],
                           strides = [#afir.expr<"192">, #afir.expr<"64">, #afir.expr<"16">, #afir.expr<"8">,
                                      #afir.expr<"1">],
                           vectorized_axis = [0, 1, 2, 3, 4],
                           vectorized_strides = [#afir.expr<"192">, #afir.expr<"64">, #afir.expr<"16">,
                                                 #afir.expr<"8">, #afir.expr<"1">],
                           role = <compute>>

module {
  func.func @broadcast_two_brc(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                               %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#d_src],
      afir.destination_descs = [#d_dst],
      afir.result_descs = [#d_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @broadcast_five_segment(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                    %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#e_src],
      afir.destination_descs = [#e_dst],
      afir.result_descs = [#e_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @broadcast_two_brc
// 四段 `[brc 3][normal 8][brc 3][normal 8]`：内侧那个广播段在循环体里只在第 0 趟执行，
// 循环之后一条 `PipeBarrier<PIPE_V>`，再生成外侧那个广播段的整块复制。
// 四段的 extent 按外→内依次物化：A、B、A、B。中间那段普通轴（B）是唯一变循环的一段。
// IR: %[[D_A0:.*]] = arith.constant 3 : index
// IR: %[[D_B0:.*]] = arith.constant 8 : index
// IR: %[[D_A2:.*]] = arith.constant 3 : index
// IR: %[[D_B2:.*]] = arith.constant 8 : index
// IR: %[[D_ZERO:.*]] = arith.constant 0 : index
// IR: %[[D_ONE:.*]] = arith.constant 1 : index
// IR: scf.for %{{.*}} = %[[D_ZERO]] to %[[D_B0]] step %[[D_ONE]] {
// 第 0 趟：最内那个广播段的两段窗口，形态 A。
// IR:   ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[D_ONE]], %[[D_B2]], %[[D_ZERO]], %[[D_A2]], %[[D_B2]], %[[D_ZERO]], %arg2
// IR: }
// IR: ascendc.pipe_barrier pipe_v
// 整块复制：`blk` 是外层广播段在目的地上的 stride（192 = 8 × 3 × 8），repeat 是 extent - 1。
// IR: %[[D_BLK:.*]] = arith.constant 192 : index
// IR: %[[D_RPT:.*]] = arith.subi %[[D_A0]], %[[D_ONE]] : index
// IR: %[[D_SHIFTED:.*]] = ascendc.local_tensor.subindex %[[D_DST:.*]][%[[D_BLK]]]
// 源与目的是同一块，目的偏移一整块——源区 [0, blk) 与写区 [blk, A × blk) 错开。
// IR: ascendc.broadcast_dim3 %[[D_SHIFTED]], %[[D_DST]], %[[D_ONE]], %[[D_BLK]], %[[D_ZERO]], %[[D_RPT]], %[[D_BLK]], %[[D_ZERO]], %arg2

// IR-LABEL: func.func @broadcast_five_segment
// 五段：两层 `scf.for` + 两条 `Broadcast` + 一条 `PipeBarrier`。**legacy 在这个形状上
// 直接 `ASSERT(false)`**，所以这一格就是「覆盖面严格超过 legacy 四轴」的那条断言。
// IR: %[[E_M:.*]] = arith.constant 2 : index
// IR: scf.for %{{.*}} = %{{.*}} to %[[E_M]]
// IR:   %[[E_B1:.*]] = arith.constant 3 : index
// IR:   %[[E_N:.*]] = arith.constant 4 : index
// IR:   %[[E_ZERO:.*]] = arith.constant 0 : index
// IR:   %[[E_ONE:.*]] = arith.constant 1 : index
// IR:   scf.for %{{.*}} = %[[E_ZERO]] to %[[E_N]] step %[[E_ONE]]
// IR:     ascendc.broadcast_dim3
// IR:   }
// barrier 与整块复制在内层循环之后、外层循环之内：内侧那一整块要等 N 跑完才算齐。
// IR:   ascendc.pipe_barrier pipe_v
// IR:   %[[E_BLK:.*]] = arith.constant 64 : index
// IR:   %[[E_RPT:.*]] = arith.subi %[[E_B1]], %[[E_ONE]] : index
// IR:   ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[E_ONE]], %[[E_BLK]], %[[E_ZERO]], %[[E_RPT]], %[[E_BLK]], %[[E_ZERO]], %arg2
// IR: }
// IR-NOT: ascendc.broadcast_dim3

// kernel 文本侧：每一条调用的第三个实参都是 `1`（两趟都落 `BroadcastFirstDim`），
// 所以整条路上没有 `Or`。
// CPP-LABEL: void broadcast_two_brc(
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c3_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});
// CPP: AscendC::PipeBarrier<PIPE_V>();
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c192_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(c192_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});

// CPP-LABEL: void broadcast_five_segment(
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c2_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});
// CPP: AscendC::PipeBarrier<PIPE_V>();
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c64_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(v{{[0-9]+}}), static_cast<uint32_t>(c64_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});
