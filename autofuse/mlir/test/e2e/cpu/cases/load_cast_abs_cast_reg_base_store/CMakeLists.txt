# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 形状取 `[256, 130]`。两条都是故意的：
#
#   - 元素数 33280 够一个块里跑多趟 lane 循环，不至于只测到「一趟就算完」那种退化形状。
#     **跑不满多趟就逮不住 lane 宽度取错**：实测把宽度按 half 取（而不是按最宽的 float），
#     第一趟 lane 的 64 个元素仍然全对，从元素 64 起才开始错——只有一趟的形状会假绿。
#   - 尾轴 130 × 2B = 260B，不是 32B 的整数倍，所以这一格同时压着「元素数不被一段 lane
#     整除」那一侧（尾段靠 mask 挡住）。
add_mlir_e2e_cpu_case(load_cast_abs_cast_reg_base_store_256x130
    PLATFORM reg_base
    OUTPUT_SHAPE "[256, 130]"
    INPUT_SHAPES "[256, 130]"
)
