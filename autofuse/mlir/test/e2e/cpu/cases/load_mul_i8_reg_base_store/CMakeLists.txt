# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 元素总数 33667 是奇数，无论怎么切块，至少有一轮的元素数不是一段 lane 的整数倍——尾段靠
# mask 挡住的那一侧一定走到。
add_mlir_e2e_cpu_case(load_mul_i8_reg_base_store_257x131
    PLATFORM reg_base
    OUTPUT_SHAPE "[257, 131]"
    INPUT_SHAPES "[257, 131]" "[257, 131]"
)
