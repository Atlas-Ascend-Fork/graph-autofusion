# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 形态 B（`[normal][brc]`，广播轴是内轴）的分段广播。这张图的 stage 有两个模板，计划选择
# 代不出常数，两个都留给运行期选；其中「广播轴排到 tile 外」那一份的搬运偏移要按非向量化
# 轴逐轴算出来（`TransferPlan.cpp` 的 `materializeTensorOffset`），算错是静默的。
#
# 两个形状都要：`[8,1]` 的 `size(z1T) > 1`，`[1024,1]` 的等于 1——后者会让「偏移 = 线性序号 ×
# 段长」这条旧式子碰巧也对，只留它就测不出逐轴式子的必要性。
#
# 形态 A（`[brc][normal]`，广播轴是外轴）的「广播轴排到 tile 外」那一份退化成 `Load → Store`，
# 走的是 `MTE2 → MTE3` 的流水线同步（队列的 enque / deque）。
add_mlir_e2e_cpu_case(load_broadcast_segmented_store_8x1
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[8, 1]"
)

add_mlir_e2e_cpu_case(load_broadcast_segmented_store_1024x1
    OUTPUT_SHAPE "[1024, 128]"
    INPUT_SHAPES "[1024, 1]"
)

add_mlir_e2e_cpu_case(load_broadcast_segmented_store_1x8
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[1, 8]"
)

# 多广播段：四段 `[brc][normal][brc][normal]`，两个广播段不相邻。分解成「每个广播段一趟、
# 由内而外」，趟与趟之间一条 `PipeBarrier<PIPE_V>`——覆盖面严格超过 legacy 的四轴形态。
#
# 形状里的 `[2, 8]` 是**上限**，不是随手取的：外层那一趟整块复制的块是内侧三段之积
# （`2 × 4 × 8 = 64` 个 f32），而 `BroadcastFirstDim` 在这一趟上按 counter 模式发 `Copy`，
# 一次 repeat 只覆盖 64 个 f32（8 block × 8）。再大一格（内侧 128 个）包装就自己按 2 次
# repeat 走地址，CPU 模拟器当场报 `dstLocal size in Copy when in counter mode`——那是
# `autofuse/ascendc/api/broadcast.h` 的遗留缺口，见设计文档「整块复制那一趟的上限」。
add_mlir_e2e_cpu_case(load_broadcast_segmented_store_1x2x1x8
    OUTPUT_SHAPE "[4, 2, 4, 8]"
    INPUT_SHAPES "[1, 2, 1, 8]"
)

# 尾轴**不满一整档**：5 个 f32 = 20B，片上行距被补到 32B（8 个元素）。此前
# `classifySegments` 要求「外段目的地 stride == 内段 extent」，这一格不成立、整图回落
# legacy；收下行距之后（见 `test/lit/conversion/afir-broadcast-tail-pitch.mlir`）它才第一次
# 真的走 MLIR。形状取自探针报 ok 的格（2026-10-12 实测 T1 `[1,5] -> [33,5]` f32）。
add_mlir_e2e_cpu_case(load_broadcast_segmented_store_1x5
    OUTPUT_SHAPE "[33, 5]"
    INPUT_SHAPES "[1, 5]"
)

# 形态 B（`[normal][brc]`，最内轴是广播轴）**带填充**。这两格 2026-10-26 之前整个被
# `classifySegments` 拒掉（探针 mem 侧 42 格命中那一条），因为形态 A 的收法是「行距两边
# 一起换」，而形态 B 的源那一位恒是字面量 1、只换得了目的地一边。放开之后发的是
# `Broadcast(dst, src, M, 1, 0, M, pitch, 0, tmp)` → 原生 `AscendC::Broadcast<T,2,1>`，
# 32B 对齐由「行距恒是一整档」天然满足。
#
# 两格差的是段数，都要：
#
#   - `[33,1] -> [33,5]`：两段，尾轴 5 × 4B = 20B，行距补到 32B（8 个 f32）。与同目录
#     `..._1x5`（形态 A、同一个尾轴）正好配成一对，两种形态换的是不同的那一边。
#   - `[2,3,5,7,1] -> [2,3,5,7,11]`：五轴、尾轴 11 × 4B = 44B（行距 48B），广播段是最内
#     段而外面还有三段普通段 —— 覆盖「带填充的窗口调用嵌在多层循环里」，只留两段的那一格
#     测不出外层偏移按行距算对没对。
add_mlir_e2e_cpu_case(load_broadcast_segmented_store_33x1
    OUTPUT_SHAPE "[33, 5]"
    INPUT_SHAPES "[33, 1]"
)

add_mlir_e2e_cpu_case(load_broadcast_segmented_store_2x3x5x7x1
    OUTPUT_SHAPE "[2, 3, 5, 7, 11]"
    INPUT_SHAPES "[2, 3, 5, 7, 1]"
)
