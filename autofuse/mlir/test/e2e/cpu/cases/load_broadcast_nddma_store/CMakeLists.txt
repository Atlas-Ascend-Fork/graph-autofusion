# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# `afir.nddma` 的数值证据。**图与 `load_broadcast_segmented_store` 逐字相同，只换了平台**：
# 同一个 `load -> brc -> store`，reg 侧的调度把 Broadcast 融进 Load 变成一个 `Nddma` 节点
# （`NddmaTemplate::GenNddmaNode`），于是 kernel 里发的是 `DataCopyNddma` 而不是片上广播 API。
# 两个用例对着读就能看出这一档收的是什么：同一批 descriptor，两侧发两条不同的指令。
#
# 形状各管一类，与 mem 侧那四格对齐，**不是随手挑的**：
#
#   `[8,1] -> [8,128]`    广播轴在内。GM 侧每趟前进 1、UB 侧前进 128——`input_stride` 与
#                         `output_stride` 两位都不等，抄错任何一位都算错地址。
#   `[1,8] -> [128,8]`    广播轴在外。GM 侧那一位是 0（每趟重读同一行），这一格是唯一
#                         测得出「广播段的 `input_stride` 必须发 0」的。
#   `[1024,1] -> [1024,128]`  `size(z1T) == 1`，让「偏移 = 线性序号 × 段长」这条旧式子碰巧
#                         也对；与上面那格一起才测得出逐轴式子的必要性（理由同 mem 侧）。
#                         **这一格也是 Store 的 GM 偏移那条判据唯一测得出来的**：69 个分块，
#                         偏移少乘一个 128 时 130982 / 131072 个元素错，而 `[8,128]` 只有一个
#                         分块、全绿（见 `TransferPlan.cpp` 的 `localStridesAreLayoutProxy`）。
#   `[1,2,1,8] -> [4,2,4,8]`  四段、两个广播段不相邻。维数 4 在包装的上限 5 之内，所以这一格
#                         走的是**一条** `DataCopyNddma`，而 mem 侧同一格要发三趟加
#                         `PipeBarrier`。维数再多一格就超限、报不支持回落 legacy——那条边界
#                         今天没有用例，补它要先有一张维数 ≥ 6 的真实图。
add_mlir_e2e_cpu_case(load_broadcast_nddma_store_8x1
    PLATFORM reg_base
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[8, 1]"
)

add_mlir_e2e_cpu_case(load_broadcast_nddma_store_1x8
    PLATFORM reg_base
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_nddma_store_1024x1
    PLATFORM reg_base
    OUTPUT_SHAPE "[1024, 128]"
    INPUT_SHAPES "[1024, 1]"
)

add_mlir_e2e_cpu_case(load_broadcast_nddma_store_1x2x1x8
    PLATFORM reg_base
    OUTPUT_SHAPE "[4, 2, 4, 8]"
    INPUT_SHAPES "[1, 2, 1, 8]"
)
