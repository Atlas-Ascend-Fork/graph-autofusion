# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# reg 侧分段广播（`BroadcastExtend`）的 bf16 一档。这一位 2026-10-26 才开（ODS 的
# `AFIR_BroadcastElementTypes` 与 `kExtendElementTypes` 一起），开之前这张图失败在 verifier
# 上——`af-capability-probe` reg 侧 118 格硬失败里 43 格就是它。
#
# 两个形状都要，差的是尾轴够不够一整档：
#
#   - `[1, 128] -> [8, 128]`：尾轴 128 × 2B = 256B，32B 的整数倍；
#   - `[1, 5] -> [33, 5]`：尾轴 5 × 2B = 10B，片上行距要补到 32B。这条路（尾轴带填充）在
#     f32 / f16 上早就通着，只留对齐那一格的话，bf16 过不过这条路就没人量。
#
# 两个形状都取自探针报 ok 的格（2026-10-26 实测，reg 侧 bf16）。
add_mlir_e2e_cpu_case(load_broadcast_bf16_reg_base_store_1x128
    PLATFORM reg_base
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[1, 128]"
)

add_mlir_e2e_cpu_case(load_broadcast_bf16_reg_base_store_1x5
    PLATFORM reg_base
    OUTPUT_SHAPE "[33, 5]"
    INPUT_SHAPES "[1, 5]"
)
