# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 动态多轴 broadcast：`[1, 1, 1, 1] -> [ks0, ks1, ks0, ks1]`。
# axis 0/2、axis 1/3 复用同一批 size var；调度后的表达式会生成
# `ks0 * ks0 * ks1 * ks1` 与 `Ceiling(Rational(1, 8) * ks1)`。
add_mlir_e2e_cpu_case(load_broadcast_multi_axis_dynamic_store_4x8x4x8
    OUTPUT_SHAPE "[4, 8, 4, 8]"
    INPUT_SHAPES "[1, 1, 1, 1]"
)
