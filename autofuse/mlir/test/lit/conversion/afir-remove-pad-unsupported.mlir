// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 收不了的形态必须报不支持，不能顺手生成一条 `RemovePadExtend`。 四种形态各一片，
// 每一片的代价都是静默算错（不是编不过），所以一条都不能放行。
//
// `--split-input-file`：pass 在第一个报错的 op 上就停，一个文件里放四片只会测到第一片。
// 每片只违反一条判据——两条一起违反的话，删掉其中任一条都不会红。
// RUN: not af-opt --split-input-file --afir-select-implementation --target=Ascend910B1 %s 2>&1 \
// RUN:   | FileCheck %s

// 第一片：目的地不稠密（外段 stride 16，而内段只有 8 个元素）。
//
// `GatherMask` 的写是连续的，`RemovePadExtend` 的形参表里一个目的地跨度都没有。放行它就是
// 把第 2 行写到第 1 行的正后方，而目的地实际的第 2 行在 16 个元素处——静默写错位置。
// 源那一侧与正例 `remove_pad_two_segment` 逐字相同，所以这一片违反的只有稠密这一条。
#loose_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                              strides = [#afir.expr<"16">, #afir.expr<"1">], vectorized_axis = [0, 1],
                              vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">], role = <compute>>
#loose_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                              strides = [#afir.expr<"16">, #afir.expr<"1">], vectorized_axis = [0, 1],
                              vectorized_strides = [#afir.expr<"16">, #afir.expr<"1">], role = <compute>>

module {
  func.func @remove_pad_loose_destination(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#loose_src],
      afir.destination_descs = [#loose_dst],
      afir.result_descs = [#loose_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 第二片：三段。源在两处都不连续（128 ≠ 16 × 4、16 ≠ 1 × 8），目的地稠密。
//
// legacy 把所有向量化轴连乘进一次调用（`GetNonLastAxisMergeSize`），隐含假设「行距处处相同」；
// 段数 ≥ 3 恰好说明那个假设不成立，legacy 在这里是静默算错。本档不建外层循环也不猜，直接拒
// ——理由见 `Implementations/RemovePad.cpp` 的文件头。
#three_src = #afir.tensor_desc<axis = [0, 1, 2],
                               repeats = [#afir.expr<"3">, #afir.expr<"4">, #afir.expr<"8">],
                               strides = [#afir.expr<"128">, #afir.expr<"16">, #afir.expr<"1">],
                               vectorized_axis = [0, 1, 2],
                               vectorized_strides = [#afir.expr<"128">, #afir.expr<"16">, #afir.expr<"1">],
                               role = <compute>>
#three_dst = #afir.tensor_desc<axis = [0, 1, 2],
                               repeats = [#afir.expr<"3">, #afir.expr<"4">, #afir.expr<"8">],
                               strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                               vectorized_axis = [0, 1, 2],
                               vectorized_strides = [#afir.expr<"32">, #afir.expr<"8">, #afir.expr<"1">],
                               role = <compute>>

module {
  func.func @remove_pad_three_segments(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#three_src],
      afir.destination_descs = [#three_dst],
      afir.result_descs = [#three_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 第三片：行距不是 32 B 的整数倍（9 个 f32 = 36 B）。
//
// 包装里那一行是 `params.src0RepeatStride = last_axis_size * sizeof(T) / 32`——整除截断。
// 9 × 4 / 32 == 1，于是 repeat 前进 8 个元素而不是 9 个，零报错、结果错。机理与设计文档
// 探针结论 P3 那条 `dst_rpt_stride` 逐字相同。
// legacy 躲开它是因为它不读 stride、自己重算 `SizeAlign(行长, 32/sizeof(T))`；本档生成的是
// descriptor 里那条真实行距，所以要自己把这一格挡住。
#unaligned_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                                   strides = [#afir.expr<"9">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                   vectorized_strides = [#afir.expr<"9">, #afir.expr<"1">], role = <compute>>
#unaligned_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                                   strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                   vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module {
  func.func @remove_pad_unaligned_pitch(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#unaligned_src],
      afir.destination_descs = [#unaligned_dst],
      afir.result_descs = [#unaligned_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// -----

// 第四片：一行内部也有跨度（内段的源 stride 是 8，不是 1）。
//
// 两段时行长那一位说的是「从行首连续取几个」，`GatherMask` 的 `src0BlockStride` 才管行内跨度，
// 而包装把它写死成 1（`removepad.h:20`）。放行它会把行内每 8 个元素取一个当成连续 8 个。
// 目的地与行距（128，32 B 的整数倍）都合法，所以这一片违反的只有行内连续这一条。
#strided_row_src = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                                     strides = [#afir.expr<"128">, #afir.expr<"8">], vectorized_axis = [0, 1],
                                     vectorized_strides = [#afir.expr<"128">, #afir.expr<"8">], role = <compute>>
#strided_row_dst = #afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"4">, #afir.expr<"8">],
                                     strides = [#afir.expr<"8">, #afir.expr<"1">], vectorized_axis = [0, 1],
                                     vectorized_strides = [#afir.expr<"8">, #afir.expr<"1">], role = <compute>>

module {
  func.func @remove_pad_strided_row(%src: !ascendc.local_tensor<?xf32>, %dst: !ascendc.local_tensor<?xf32>) {
    %result = afir.remove_pad %src outs(%dst) {
      afir.operand_descs = [#strided_row_src],
      afir.destination_descs = [#strided_row_dst],
      afir.result_descs = [#strided_row_dst]
    } : !ascendc.local_tensor<?xf32>, !ascendc.local_tensor<?xf32> -> !ascendc.local_tensor<?xf32>
    return
  }
}

// 下面那一条是计数式的，不是单条匹配：`--split-input-file` 下每片各报一次诊断，只写单条
// 的话某一片被误放行时总退出码仍然非零、那一条也仍然匹配得到——四片就只剩三片在测。
// CHECK-COUNT-4: no supported
