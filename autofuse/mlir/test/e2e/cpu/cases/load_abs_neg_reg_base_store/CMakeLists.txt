# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 形状与 M1 那一格（`load_add_abs_reg_base_store`）取同一个 `[256, 128]`：这一格验的是算子表
# 新加的那一行，不是形状相关的判据（尾轴 128 × 4B = 512B 是 32B 的整数倍，所以对齐那一维
# 不掺进来）。元素数 32768 够一个块里跑多趟 lane 循环。
add_mlir_e2e_cpu_case(load_abs_neg_reg_base_store_256x128
    PLATFORM reg_base
    OUTPUT_SHAPE "[256, 128]"
    INPUT_SHAPES "[256, 128]"
)
