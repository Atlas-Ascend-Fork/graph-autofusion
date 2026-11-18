# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE for the full text of the License.

add_mlir_e2e_cpu_case(load_eq_i64_store_73x5x4
    OUTPUT_SHAPE "[73, 5, 4]"
    INPUT_SHAPES "[73, 5, 4]" "[73, 5, 4]"
)

# 小形状边界：单向量轴、tile 元素数 8/16/32 时节点级 tmp 公式（8×N×5）只算出
# 320/640/1280 字节，compare.h 包装（EQ 需 ≥2336B、GT 需 ≥1280B）在该区间内自身
# 无法推进。这三条用例钉的是调度层的兜底：common/ub_expr 的 kMinTmpBufferSize=8192
# 下限（MergeNodeTmpBuffers 对所有 tmp 统一 Max(需求,8192)）。若该下限被移除或调小，
# 这里会以死循环/除零的形式立刻红——它是这个下限与包装消耗之间契约的唯一回归。
add_mlir_e2e_cpu_case(load_eq_i64_store_1x8
    OUTPUT_SHAPE "[1, 8]"
    INPUT_SHAPES "[1, 8]" "[1, 8]"
)
add_mlir_e2e_cpu_case(load_eq_i64_store_1x16
    OUTPUT_SHAPE "[1, 16]"
    INPUT_SHAPES "[1, 16]" "[1, 16]"
)
add_mlir_e2e_cpu_case(load_eq_i64_store_1x32
    OUTPUT_SHAPE "[1, 32]"
    INPUT_SHAPES "[1, 32]" "[1, 32]"
)
