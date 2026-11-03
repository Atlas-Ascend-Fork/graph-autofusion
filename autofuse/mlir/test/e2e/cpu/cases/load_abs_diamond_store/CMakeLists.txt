# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 菱形，分叉点是 unary（探针 T5e）。形状与 dtype 都不起作用——五个形状格 × f32/f16
# 实测一致，所以这里只一格，取 `[128, 8]`。
add_mlir_e2e_cpu_case(load_abs_diamond_store_128x8
    OUTPUT_SHAPE "[128, 8]"
    INPUT_SHAPES "[128, 8]" "[128, 8]" "[128, 8]"
)
