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
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT

#a_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"8">],
                           strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1],
                           vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <compute>>
#a_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                           strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                           vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

#b_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"1">],
                           strides = [#afir.expr<"1">, #afir.expr<"0">], vectorized_axis = [0, 1],
                           vectorized_strides = [#afir.expr<"1">, #afir.expr<"0">], role = <compute>>
#b_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                           strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                           vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

#c_src = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"3">, #afir.expr<"1">, #afir.expr<"8">],
                           strides = [#afir.expr<"8">, #afir.expr<"0">, #afir.expr<"1">],
                           vectorized_axis = [0, 1, 2],
                           vectorized_strides = [#afir.expr<"8">, #afir.expr<"0">, #afir.expr<"1">],
                           role = <compute>>
#c_dst = #afir.tensor_desc<axis = [0, 1, 2], repeats = [#afir.expr<"3">, #afir.expr<"4">, #afir.expr<"8">],
                           strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                           vectorized_axis = [0, 1, 2],
                           vectorized_strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                           role = <compute>>

module {
  // 形态 A：最内两段是 `[brc 4][normal 8]`。
  func.func @broadcast_outer_dim(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                 %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#a_src],
      afir.destination_descs = [#a_dst],
      afir.result_descs = [#a_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 形态 B：最内两段是 `[normal 4][brc 8]`。
  func.func @broadcast_inner_dim(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                 %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#b_src],
      afir.destination_descs = [#b_dst],
      afir.result_descs = [#b_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }

  // 三段 `[normal 3][brc 4][normal 8]`：外层那一段变一层 `scf.for`。
  func.func @broadcast_outer_loop(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>,
                                  %tmp: !ascendc.local_tensor<?xui8>) {
    %result = afir.broadcast %src outs(%dst) tmp(%tmp) {
      afir.operand_descs = [#c_src],
      afir.destination_descs = [#c_dst],
      afir.result_descs = [#c_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xui8>
        -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 生成文本里重复出现的 `constexpr uint32_t c0_idx = 0;` 是这条短 pipeline 没跑 CSE 的结果，
// 不是本档多生成的——偏移按张量各建一个常量，真实路径（`--afir-aggregate-launch`）上 CSE
// 会把它们并掉，与 elementwise 接通那一步确认过的是同一件事。

// IR-LABEL: func.func @broadcast_outer_dim
// 形态 A `[brc 4][normal 8]`：零层循环、一条调用。
// 先锚到 `4` 上——它之前那两条 `arith.constant 0` 是两个张量的偏移。
// IR: %[[A_K:.*]] = arith.constant 4 : index
// IR: %[[A_Z:.*]] = arith.constant 8 : index
// IR: %[[A_ZERO:.*]] = arith.constant 0 : index
// IR: %[[A_ONE:.*]] = arith.constant 1 : index
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[A_ONE]], %[[A_Z]], %[[A_ZERO]], %[[A_K]], %[[A_Z]], %[[A_ZERO]], %arg2
// IR-NOT: scf.for

// IR-LABEL: func.func @broadcast_inner_dim
// 形态 B `[normal 4][brc 8]`：`src_m = M`、`src_k = 1`。
// IR: %[[B_M:.*]] = arith.constant 4 : index
// IR: %[[B_K:.*]] = arith.constant 8 : index
// IR: %[[B_ZERO:.*]] = arith.constant 0 : index
// IR: %[[B_ONE:.*]] = arith.constant 1 : index
// IR: ascendc.broadcast_dim3 %{{.*}}, %{{.*}}, %[[B_M]], %[[B_ONE]], %[[B_ZERO]], %[[B_M]], %[[B_K]], %[[B_ZERO]], %arg2
// IR-NOT: scf.for

// IR-LABEL: func.func @broadcast_outer_loop
// 三段：最外那一段变一层 `scf.for`，上界是它的 extent（3）。
// IR: %[[C_N:.*]] = arith.constant 3 : index
// IR: scf.for %[[IV:.*]] = %{{.*}} to %[[C_N]] step %{{.*}} {
// 偏移逐张量各取各的 stride：源 8、目的地 32。共用一个的话第二条会变成 `%[[IV]], %c8`。
// IR:   %[[SRC_OFF:.*]] = arith.muli %[[IV]], %{{.*}} : index
// IR:   ascendc.local_tensor.subindex %arg0[%[[SRC_OFF]]]
// IR:   %[[DST_OFF:.*]] = arith.muli %[[IV]], %{{.*}} : index
// IR:   ascendc.local_tensor.subindex %arg1[%[[DST_OFF]]]
// IR:   ascendc.broadcast_dim3
// 只有这一层循环。
// IR-NOT: scf.for

// 同一件事在 kernel 文本上。第三个实参恒为 `1`（形态 A）或第五个恒为 `0`（形态 B）——
// 这是「生成的调用落不到 `BroadcastMiddleDim` 的 `Or`」在编译期唯一钉得住的形式。
// CPP-LABEL: void broadcast_outer_dim(
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});

// CPP-LABEL: void broadcast_inner_dim(
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});

// CPP-LABEL: void broadcast_outer_loop(
// CPP: for (uint32_t
// CPP: Broadcast(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx), static_cast<uint32_t>(c0_idx), v{{[0-9]+}});

// 三个 func 各选中分段那一档，不是 `Duplicate` 那一档。
// SELECT: ascend.implementation = {id = "broadcast.tensor.segmented", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "broadcast.tensor.segmented", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "broadcast.tensor.segmented", kind = "ascendc_api"}

// reg 侧选不出：那一侧的广播是另一个 API（`BroadcastExtend<T, rank>`，吃整串向量化轴的
// shape 数组、不吃 tmp buffer），本轮不接。非常量源在 reg 侧照旧不支持、整图回落 legacy。
// RUN: not af-opt --afir-select-implementation --target=Ascend950DT_950x %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE
// WRONGSIDE: no supported
