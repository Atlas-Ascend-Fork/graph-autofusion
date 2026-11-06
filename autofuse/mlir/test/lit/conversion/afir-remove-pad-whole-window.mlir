// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.remove_pad` 的第四种形态：整窗口逐元素收集（两段 gather）。前三种形态见
// `afir-remove-pad.mlir`；这一档单独成文件，因为它的 func 必须带 `afir.asc_graph`，
// 而那份文件的 CPP 路线（`ascir-translate`）不认输出里残留的 `afir.*` 属性。
//
// **形状从哪来**：transpose 尾块的对齐补白。上游按 32 B 对齐算完，行的有效长度是 4、
// 槽距是 8（f32 的 32 B 粒度），三行散在 3×32 的窗口里；目的地稠密（3 行 × 4 列）。
//
// **为什么两段没被合轴并掉**：名义布局上外段源步长 32 == 内段源步长 8 × 内段名义 4，
// 两端都无缝——纯几何判据会并成一段（那样就退化成逐元素档）。挡下来的是合轴护栏
// `canShrink`（`VectorSegmentation.cpp`）：内轴标着 TileInner、带着 split pair，运行期
// 可能截尾，「实际内轴缩短时行距仍按标称布局」，跨过这条边界会把下一行接到上一行
// 尾部。护栏只在真实图（`afir.asc_graph`）里有意义，没有 graph 时恒 false——所以本
// 档**必须**带 graph 才测得到。
//
// **发射**：`RemovePadExtend(dst, src, R, P, N)` = 从 src 取 N 行、行距 P、每行取前 R 个。
// 这档 R 恒 1（逐元素），P 是槽距 8，N 是「外段实际 3 × 内段名义 4」的乘积——尾块的
// 行间空槽要一并线性扫过、写进 tile 缓冲的名义空位，消费方按 ActualSize 前缀读。
// 与 legacy `GetOneAxisSize` 对非首 TileInner 轴取名义 `Size`、其余取 `ActualSize`
// 的口径逐字对齐（`remove_pad_api_call.cpp` 的 `is_last_axis_align` 分支）。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// CPP 路线与 `afir-remove-pad.mlir` 相同（ascir-translate），但要先剥掉 func 上的
// asc_graph——translate 不认输出里残留的 `afir.*` 属性。属性体内没有 `}`，一条 sed 够了。
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s \
// RUN:   | sed 's/attributes {afir.asc_graph = #afir.asc_graph<[^}]*>}//' \
// RUN:   | ascir-translate -mlir-to-ascendc | FileCheck %s --check-prefix=CPP
// 两侧共用同一个包装（矩阵 tier D1），reg 侧选中的是**同一个** implementation。
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s \
// RUN:   | FileCheck %s --check-prefix=SELECT

#whole_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"4">],
                              strides = [#afir.expr<"32">, #afir.expr<"8">], vectorized_axis = [0, 1],
                              vectorized_strides = [#afir.expr<"32">, #afir.expr<"8">], role = <compute>>
#whole_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"3">, #afir.expr<"4">],
                              strides = [#afir.expr<"4">, #afir.expr<"1">], vectorized_axis = [0, 1],
                              vectorized_strides = [#afir.expr<"4">, #afir.expr<"1">], role = <compute>>

module {
  func.func @remove_pad_whole_window(%src: !ascendc.local_tensor<?xf32>,
                                     %dst: !ascendc.local_tensor<?xf32>) attributes {afir.asc_graph =
      #afir.asc_graph<tiling_key = 0,
        axes = [<id = 0, name = "z", axis_type = Original, size = #afir.expr<"12">, align = #afir.expr<"-1">>,
                <id = 1, name = "zt", axis_type = TileInner, size = #afir.expr<"4">, align = #afir.expr<"1">, from = [0], split_pair_other_id = 2>,
                <id = 2, name = "zT", axis_type = TileOuter, size = #afir.expr<"3">, align = #afir.expr<"1">, from = [0], split_pair_other_id = 1>],
        type = Compute>} {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#whole_src],
      afir.destination_descs = [#whole_dst],
      afir.result_descs = [#whole_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 零层循环：整窗一条调用扫完（段数上限 2 是这一档的判据）。
// IR-LABEL: func.func @remove_pad_whole_window
// IR-NOT: scf.for
// 常量按建出的次序：外段 extent 3、内段 extent 4、行长 1，然后是名义内段 extent
// （4 的第二份，`materializeSegmentExtent` 对 scheduledAxis=-1 按名义建）、行距 8。
// IR: %[[W_OUTER:.*]] = arith.constant 3 : index
// IR: %{{.*}} = arith.constant 4 : index
// IR: %[[W_ONE:.*]] = arith.constant 1 : index
// 趟数是「外段实际 × 内段名义」的乘积，不是任何一段的 extent 单独值。
// IR: %[[W_N:.*]] = arith.muli %[[W_OUTER]], %{{[^,]*}} : index
// IR: %[[W_P:.*]] = arith.constant 8 : index
// IR: ascendc.remove_pad_extend %{{.*}}, %{{.*}}, %[[W_ONE]], %[[W_P]], %[[W_N]]

// CPP-LABEL: void remove_pad_whole_window(
// 趟数是运行期乘积（c3 × c4），不是常量折叠后的 12。
// CPP: v{{[0-9]+}} = c3_idx * c4_idx;
// CPP: RemovePadExtend<float>(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint64_t>(c1_idx), static_cast<uint64_t>(c8_idx), static_cast<uint64_t>(v{{[0-9]+}}));

// 两侧共用，所以 mem 与 reg 两个 target 上都是它。
// SELECT: ascend.implementation = {id = "remove_pad.tensor.extend", kind = "ascendc_api"}
