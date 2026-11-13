# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 单轴。输入与输出同形——广播的源是标量，不是输入张量。
add_mlir_e2e_cpu_case(load_scalar_broadcast_store_131
    OUTPUT_SHAPE "[131]"
    INPUT_SHAPES "[131]"
)

# 两根轴，且每个 tile 装多于一行：`Duplicate` 的 count 取的是目的地那一段的物理跨度，
# 一个 tile 只装一行时跨度与元素数恰好相等，这一档探不到那个区别。
add_mlir_e2e_cpu_case(load_scalar_broadcast_store_40x128
    OUTPUT_SHAPE "[40, 128]"
    INPUT_SHAPES "[40, 128]"
)
