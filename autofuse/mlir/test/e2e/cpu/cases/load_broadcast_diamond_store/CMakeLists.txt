# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 覆盖以广播为分叉点的 T5，注册两种已验证形态（2026-10-12 探针）。
#
# [1,8] -> [128,8]：外轴广播，尾轴按 32B 对齐。
# [8,1] -> [8,128]：最内轴广播，使用不同的分段条件。
#
# 两种形态的广播与 Add 调度位置不同，需分别检查共用 b<id> 的成员布局。
# 形态 A 的 B0Y2 将广播排在 Add 之后。
#
# 历史探针中的其他形状（[1,5] -> [33,5]、[2,3,5,1] -> [2,3,5,7] 及部分
# f16 短尾轴）不受支持；不能以回落 legacy 后数值正确作为 MLIR E2E 通过的证据。
add_mlir_e2e_cpu_case(load_broadcast_diamond_store_1x8
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_diamond_store_8x1
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[8, 1]"
)
