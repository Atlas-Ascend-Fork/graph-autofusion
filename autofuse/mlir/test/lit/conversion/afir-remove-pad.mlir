// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// `afir.remove_pad` 的三种形态，逐个落成一条 `RemovePadExtend(dst, src, R, P, N)`。
// 第四种形态（整窗口 gather）的 func 必须带 `afir.asc_graph` 才能触发合轴护栏，而本文件的
// CPP 路线（`ascir-translate`）不认输出里残留的 `afir.*` 属性，所以它住在单独的
// `afir-remove-pad-whole-window.mlir`。
//
// 三个实参的含义按包装内部那条 `GatherMask` 反推：从 src 取 N 行、行距 P 个元素、每行取
// 前 R 个，连续写到 dst。`dst` 侧没有跨度参数，所以目的地必须稠密——反例见
// `afir-remove-pad-unsupported.mlir`。
//
// 为什么这里是三个独立的 func 而不是一张端到端的图：`afir.remove_pad` 的两个真实上游
// （带填充的 tile、尾轴非连续的 Load）今天都下降不了——`materializeScheduledTileIndices`
// 要求局部侧合轴后只有一段、且那一段的 stride 字面量为 1
// （`dialect/AFIR/Analysis/TransferPlan.cpp:720`），带填充的 tile 是两段，非连续 Load 的那一段
// stride 不是 1。所以目前没有一条含 `RemovePad` 的图能走到执行那一步，前置是逐轴的
// `TransferPlan`，不是本档。见 `docs/development/2026-09-30-afir-remove-pad-design.md`。

// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | FileCheck %s --check-prefix=IR
// RUN: af-opt --afir-to-ascendc-pipeline --target=Ascend910B1 %s | ascir-translate -mlir-to-ascendc \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: af-opt --afir-select-implementation --target=Ascend910B1 %s | FileCheck %s --check-prefix=SELECT
// 两侧共用同一个包装（矩阵 tier D1），所以 reg 侧选中的是同一个 implementation。
// 这一条是 `ApiFamily::Shared` 与 `ApiFamily::MemBased` 的分水岭：写成 MemBased 的话这一行会红。
// RUN: af-opt --afir-select-implementation --target=Ascend950DT_950x %s \
// RUN:   | FileCheck %s --check-prefix=SELECT

// 形态一：两段。源的行距 16 个 f32（64 B，32 B 的整数倍），行长 8，共 4 行；目的地稠密。
// 这是非对齐模板里 `RemovePad` 的常态形状——上游按 32 B 对齐算完，行与行之间空着 8 个元素。
#two_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                             strides = [#afir.expr<"16">, #afir.expr<"1">], vectorized_axis = [0, 1],
                             vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">], role = <compute>>
#two_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                             strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                             vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

// 形态二：逐元素收集。每个元素单独占一个 32 B 块（f32 的行距 8），共 4 个。
// 这是 legacy `is_last_axis_align` 那条分支的形状——调度给尾轴的 stride 是
// `Align(1, 32/sizeof(T))`（`base_alignment_strategy.cpp:695` 的 `kDiscontinuous` 分支）。
#gather_src = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"4">],
                                strides = [#afir.expr<"8">], vectorized_axis = [0],
                                vectorized_strides = [#afir.expr<"8">], role = <compute>>
#gather_dst = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"4">],
                                strides = [#afir.expr<"1">], vectorized_axis = [0],
                                vectorized_strides = [#afir.expr<"1">], role = <compute>>

// 形态三：纯搬运。只有一段、源已经连续，趟数就是 1，行距那一位生成的是行长本身
// ——`repeatTimes == 1` 时 `src0RepeatStride` 读不到它（`removepad.h:22` 只在第 2..N 趟用它）。
#copy_desc = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"8">],
                                strides = [#afir.expr<"1">], vectorized_axis = [0],
                                vectorized_strides = [#afir.expr<"1">], role = <compute>>

module {
  func.func @remove_pad_two_segment(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#two_src],
      afir.destination_descs = [#two_dst],
      afir.result_descs = [#two_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @remove_pad_gather(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#gather_src],
      afir.destination_descs = [#gather_dst],
      afir.result_descs = [#gather_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }

  func.func @remove_pad_copy(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#copy_desc],
      afir.destination_descs = [#copy_desc],
      afir.result_descs = [#copy_desc]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// IR-LABEL: func.func @remove_pad_two_segment
// 零层循环：两段全部交给这一条调用（段数上限 2 是本档的判据）。
// IR-NOT: scf.for
// 三个常量按段的次序建（外段的趟数在前、内段的行长在后），行距最后建；调用按 C++
// 形参表的次序生成（行长、行距、趟数）。两个次序不同是对的，不要把其中一个改成另一个。
// IR: %[[N:.*]] = arith.constant 4 : index
// IR: %[[R:.*]] = arith.constant 8 : index
// IR: %[[P:.*]] = arith.constant 16 : index
// IR: ascendc.remove_pad_extend %{{.*}}, %{{.*}}, %[[R]], %[[P]], %[[N]]

// IR-LABEL: func.func @remove_pad_gather
// 行长恒为 1：每行只有一个元素。行距是那一段的源 stride（8），趟数是段的 extent（4）。
// IR: %[[G_N:.*]] = arith.constant 4 : index
// IR: %[[G_ONE:.*]] = arith.constant 1 : index
// IR: %[[G_P:.*]] = arith.constant 8 : index
// IR: ascendc.remove_pad_extend %{{.*}}, %{{.*}}, %[[G_ONE]], %[[G_P]], %[[G_N]]

// IR-LABEL: func.func @remove_pad_copy
// 趟数恒为 1，行距那一位生成的就是行长。
// IR: %[[C_R:.*]] = arith.constant 8 : index
// IR: %[[C_ONE:.*]] = arith.constant 1 : index
// IR: ascendc.remove_pad_extend %{{.*}}, %{{.*}}, %[[C_R]], %[[C_R]], %[[C_ONE]]

// kernel 文本侧：三个尺寸都 `static_cast<uint64_t>`（包装的形参是 `const uint64_t`），
// 调用不带 `AscendC::` 限定（包装住在全局作用域），模板实参是源的元素类型。
// CPP-LABEL: void remove_pad_two_segment(
// CPP: RemovePadExtend<float>(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint64_t>(c8_idx), static_cast<uint64_t>(c16_idx), static_cast<uint64_t>(c4_idx));

// CPP-LABEL: void remove_pad_gather(
// CPP: RemovePadExtend<float>(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint64_t>(c1_idx), static_cast<uint64_t>(c8_idx), static_cast<uint64_t>(c4_idx));

// CPP-LABEL: void remove_pad_copy(
// CPP: RemovePadExtend<float>(v{{[0-9]+}}, v{{[0-9]+}}, static_cast<uint64_t>(c8_idx), static_cast<uint64_t>(c8_idx), static_cast<uint64_t>(c1_idx));

// 三个 func 各选中同一个 implementation。两侧共用，所以 mem 与 reg 两个 target 上都是它。
// SELECT: ascend.implementation = {id = "remove_pad.tensor.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "remove_pad.tensor.extend", kind = "ascendc_api"}
// SELECT: ascend.implementation = {id = "remove_pad.tensor.extend", kind = "ascendc_api"}
