// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// reg 侧的分段广播：`BroadcastExtend<T, rank>(dst, src, dst_shape, src_shape)`。
//
// 形状与 `afir-broadcast-segmented.mlir` 逐字相同，只少一个 tmp 操作数——reg 侧
// `BroadcastAscIrCodegenImplV2` 没有 `CalcTmpBufSize` 覆写，调度不给这块 buffer，
// `checkTmpBuffer` 拿块数严格对等。同一批 descriptor 在两侧生成两个不同的 API，这正是
// 这一档存在的理由。
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend950DT_950x %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s | FileCheck %s --check-prefix=SELECT

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
  // 广播段在外：`dst_shape = {4, 8}`、`src_shape = {1, 8}`。
  func.func @broadcast_outer_dim(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.broadcast %src outs(%dst) {
      afir.operand_descs = [#a_src],
      afir.destination_descs = [#a_dst],
      afir.result_descs = [#a_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  // 广播段在内：`dst_shape = {4, 8}`、`src_shape = {4, 1}`。
  func.func @broadcast_inner_dim(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.broadcast %src outs(%dst) {
      afir.operand_descs = [#b_src],
      afir.destination_descs = [#b_dst],
      afir.result_descs = [#b_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  // 三段 `[normal 3][brc 4][normal 8]`：mem 侧这一格生成一层 `scf.for`，reg 侧不生成——
  // 第三段直接进 shape 数组，循环分解是 `BroadcastExtend` 自己 tiling 里的事。
  func.func @broadcast_outer_loop(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.broadcast %src outs(%dst) {
      afir.operand_descs = [#c_src],
      afir.destination_descs = [#c_dst],
      afir.result_descs = [#c_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @broadcast_outer_dim
// IR: ascendc.broadcast_extend_reg_base %{{.*}}, %{{.*}}
// IR-NOT: scf.for

// IR-LABEL: func.func @broadcast_outer_loop
// 三段一样一层循环都不生成，这是与 mem 侧那一档最直观的差别。
// IR: ascendc.broadcast_extend_reg_base %{{.*}}, %{{.*}}
// IR-NOT: scf.for

// 同一件事在 kernel 文本上：两串 shape 落成两个局部 `const uint32_t[]`，`rank` 与数组长度
// 是同一个数，广播段那一位是 1。
// CPP-LABEL: void broadcast_outer_dim(
// CPP: const uint32_t dst_shape_[[A_SRC:v[0-9]+]]_brc_to_[[A_DST:v[0-9]+]][2] = {static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx)};
// CPP-NEXT: const uint32_t src_shape_[[A_SRC]]_brc_to_[[A_DST]][2] = {static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx)};
// CPP-NEXT: BroadcastExtend<float, 2>([[A_DST]], [[A_SRC]], dst_shape_[[A_SRC]]_brc_to_[[A_DST]], src_shape_[[A_SRC]]_brc_to_[[A_DST]]);

// CPP-LABEL: void broadcast_inner_dim(
// CPP: const uint32_t dst_shape_[[B_SRC:v[0-9]+]]_brc_to_[[B_DST:v[0-9]+]][2] = {static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx)};
// CPP-NEXT: const uint32_t src_shape_[[B_SRC]]_brc_to_[[B_DST]][2] = {static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c1_idx)};
// CPP-NEXT: BroadcastExtend<float, 2>([[B_DST]], [[B_SRC]], dst_shape_[[B_SRC]]_brc_to_[[B_DST]], src_shape_[[B_SRC]]_brc_to_[[B_DST]]);

// CPP-LABEL: void broadcast_outer_loop(
// CPP-NOT: for (
// CPP: const uint32_t dst_shape_[[C_SRC:v[0-9]+]]_brc_to_[[C_DST:v[0-9]+]][3] = {static_cast<uint32_t>(c3_idx), static_cast<uint32_t>(c4_idx), static_cast<uint32_t>(c8_idx)};
// CPP-NEXT: const uint32_t src_shape_[[C_SRC]]_brc_to_[[C_DST]][3] = {static_cast<uint32_t>(c3_idx), static_cast<uint32_t>(c1_idx), static_cast<uint32_t>(c8_idx)};
// CPP-NEXT: BroadcastExtend<float, 3>([[C_DST]], [[C_SRC]], dst_shape_[[C_SRC]]_brc_to_[[C_DST]], src_shape_[[C_SRC]]_brc_to_[[C_DST]]);

// 扩展头名与支持表的对应在注册期就查（`checkFamilyMatchesHeader`），对不上整张注册表建不
// 起来，所以这里不重复钉——这条短 pipeline 也不生成 include 块（那是 `--afir-aggregate-launch`
// 那一层的事）。

// 三个 func 各选中 reg 侧分段那一档。
// SELECT: ascend.implementation = {id = "broadcast.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "broadcast.tensor.extend_reg_base", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "broadcast.tensor.extend_reg_base", kind = "ascendc_api"}

// mem 侧选不出：那一侧的分段广播吃 tmp buffer，而这几格一个都没有。
// RUN: not af-opt --afir-select-implementation --target=Ascend910B1 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WRONGSIDE
// WRONGSIDE: no supported
