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
#                           多段广播分解之后那块片上张量上算。相邻广播段 2026-10-11
#                           才收下，在此之前只有一份 lit，这是它第一次有数值证据。
#   `[8,1] -> [8,128]`      形态 B（最内轴是广播轴）。这一格的 `B0Y1` 模板把广播轴切分、
#                           再与外轴合成一根块轴，调度于是把 `abs` 排到 `afir.broadcast`
#                           之前——它作用在广播前那块向量化轴 extent 全为 1 的 tile 上，
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

#   `[1, 65536] -> [33, 65536]`  内轴超过 UB、必须切 tile 且广播轴与内轴先合轴再分块。
#                           H.1（2026-11-10 剩余工作清单）：这一档曾经「声称接管」但生成
#                           的 kernel 里没有任何 Broadcast/Duplicate 调用，输出大面积为
#                           0——`TransferPlan.cpp` 的 `getGlobalDescriptor` 把 Store 的
#                           GM 偏移错取成生产者 UB 侧的步长，广播轴那一项被当成 0 剔掉。
add_mlir_e2e_cpu_case(load_broadcast_abs_store_1x65536
    OUTPUT_SHAPE "[33, 65536]"
    INPUT_SHAPES "[1, 65536]"
)
