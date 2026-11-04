# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 形状与 `load_add_abs_reg_base_store` 同取 `[256, 128]`：尾轴 128 × 4B = 512B 是 32B 的
# 整数倍，所以这一格不掺进对齐那一维的判据；32768 个元素够跑多趟 lane 循环，只跑一趟的
# 形状会把「lane 宽度取错」这类错放过去（M2 实测过）。
add_mlir_e2e_cpu_case(load_eq_where_reg_base_store_256x128
    PLATFORM reg_base
    OUTPUT_SHAPE "[256, 128]"
    INPUT_SHAPES "[256, 128]" "[256, 128]" "[256, 128]" "[256, 128]"
)
