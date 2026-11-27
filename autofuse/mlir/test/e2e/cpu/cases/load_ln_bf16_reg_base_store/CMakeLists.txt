# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 形状取法与 `load_cast_abs_cast_reg_base_store` 同一条：33280 个元素够一个块里跑多趟 lane
# 循环；尾轴 130 × 2B = 260B 不是 32B 的整数倍，尾段靠 mask 挡住。
add_mlir_e2e_cpu_case(load_ln_bf16_reg_base_store_256x130
    PLATFORM reg_base
    OUTPUT_SHAPE "[256, 130]"
    INPUT_SHAPES "[256, 130]"
)
