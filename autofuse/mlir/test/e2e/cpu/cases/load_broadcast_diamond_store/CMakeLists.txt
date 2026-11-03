# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 级联拓扑 T5（广播作分叉点）。两格，取探针报 ok 的两种形态（2026-10-12 实测）：
#
#   `[1,8] -> [128,8]`   形态 A：广播轴在外、最内轴不是广播轴，尾轴 32B 对齐。
#   `[8,1] -> [8,128]`   形态 B：**最内轴就是广播轴**，走的是另一条分段判据。
#
# 两种形态都要：分叉点的结果与 `add` 的结果共用一个非队列 `b<id>`，而两种形态的 plan
# 把广播排在不同位置（形态 A 的 B0Y2 把广播排在 `add` 之后），于是成员布局也不是同一张。
#
# 其余形状（`[1,5] -> [33,5]`、`[2,3,5,1] -> [2,3,5,7]`，以及 f16 的尾轴不足 32B 的那几格）
# 是 `afir.broadcast` 自己的能力缺口，报不支持、整图回落，写成 E2E 是假绿。
add_mlir_e2e_cpu_case(load_broadcast_diamond_store_1x8
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_diamond_store_8x1
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[8, 1]"
)
