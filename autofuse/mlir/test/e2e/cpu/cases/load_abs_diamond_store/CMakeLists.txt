# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 覆盖 unary 分叉的菱形图，采用 [128, 8]。
# 探针 T5e 在五种形状及 f32/f16 上结果一致，此处不重复注册所有组合。
add_mlir_e2e_cpu_case(load_abs_diamond_store_128x8
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[128, 8]" "[128, 8]" "[128, 8]"
)
