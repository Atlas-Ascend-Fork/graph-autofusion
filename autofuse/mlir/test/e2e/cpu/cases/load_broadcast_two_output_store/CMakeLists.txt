# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 级联拓扑 T7（一个 `brc` 的结果经两条 Store 写到两个输出）。两格都取自探针报 ok 的格
# （2026-10-12 实测），两种段方向各一格：
#
#   `[1,8] -> [128,8]`   形态 A（广播轴在外）
#   `[8,1] -> [8,128]`   形态 B（最内轴是广播轴）
#
# `OUTPUT_SHAPE` 给两个同形的形状，次序对应 `Output` 的 index。
add_mlir_e2e_cpu_case(load_broadcast_two_output_store_1x8
    OUTPUT_SHAPE "[128, 8]" "[128, 8]"
    INPUT_SHAPES "[1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_two_output_store_8x1
    OUTPUT_SHAPE "[8, 128]" "[8, 128]"
    INPUT_SHAPES "[8, 1]"
)
