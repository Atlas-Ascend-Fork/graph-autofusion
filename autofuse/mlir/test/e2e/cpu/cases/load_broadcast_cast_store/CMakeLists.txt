# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 级联拓扑 T6（`load(f16) -> brc -> cast(f32) -> store`）。三格都取自探针报 ok 的格
# （2026-10-12 实测，见 Case.cpp 文件头），各管一类：
#
#   `[1,8] -> [128,8]`     形态 A。f16 下尾轴 16B（**不足一档**），cast 之后是 32B（刚好一档）
#                          ——同一串轴跨过 32B 边界的两侧，这正是这一格存在的理由。
#   `[1,1,8] -> [4,4,8]`   两个广播段且相邻，dtype 变化叠在多段分解上。
#   `[8,1] -> [8,128]`     形态 B（最内轴是广播轴）。调度在这一格把 `cast` 排到
#                          `afir.broadcast` **之前**，于是 dtype 变化发生在广播前那块
#                          向量化轴 extent 全为 1 的 tile 上，本趟只有一个元素。这一格
#                          原先硬失败（`failed to legalize operation 'afir.cast'`），
#                          由 `materializeVectorizedSegments` 的单元素窗口那一档收下。
#
# 反方向（f32 → f16）不在这里：探针实测整图回落，不是这条路生成的调用。
add_mlir_e2e_cpu_case(load_broadcast_cast_store_1x8
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_cast_store_1x1x8
    OUTPUT_SHAPE "[4, 4, 8]"
    INPUT_SHAPES "[1, 1, 8]"
)

add_mlir_e2e_cpu_case(load_broadcast_cast_store_8x1
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[8, 1]"
)
