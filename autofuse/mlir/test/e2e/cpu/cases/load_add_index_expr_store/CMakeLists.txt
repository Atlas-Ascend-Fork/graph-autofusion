# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# INPUT_SHAPES 只列 Data（index 0）。index 1 的 ScalarData 按值传、没有形状。
add_mlir_e2e_cpu_case(load_add_index_expr_store_104x16x8x1
    OUTPUT_SHAPE "[104, 16, 8, 1]"
    INPUT_SHAPES "[104, 16, 8, 1]"
)
