# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 级联拓扑 T2（`load -> brc -> abs -> store`）。**三个形状都取自探针报 ok 的格**
# （2026-10-12 实测，见 Case.cpp 文件头），各管一类：
#
#   `[1,8] -> [128,8]`      形态 A（广播轴在外），尾轴 32B 对齐、最内轴不是广播轴——
#                           也就是探针那张签名表里唯一全通的那一象限。
#   `[1,1,8] -> [4,4,8]`    两个广播段且相邻（`[brc][brc][normal]`），下游算子要在
#                           **多段广播分解之后**那块片上张量上算。相邻广播段 2026-10-11
#                           才收下，在此之前只有一份 lit，这是它第一次有数值证据。
#   `[8,1] -> [8,128]`      形态 B（最内轴是广播轴）。这一格的 `B0Y1` 模板把广播轴切分、
#                           再与外轴合成一根块轴，调度于是把 `abs` 排到 `afir.broadcast`
#                           **之前**——它作用在广播前那块向量化轴 extent 全为 1 的 tile 上，
#                           本趟只有一个元素。这一格原先硬失败
#                           （`failed to legalize operation 'afir.abs'`），由
#                           `materializeVectorizedSegments` 的单元素窗口那一档收下。
add_mlir_e2e_cpu_case(load_broadcast_abs_store_1x8
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_abs_store_1x1x8
    OUTPUT_SHAPE "[4, 4, 8]"
    INPUT_SHAPES "[1, 1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_abs_store_8x1
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[8, 1]"
)
