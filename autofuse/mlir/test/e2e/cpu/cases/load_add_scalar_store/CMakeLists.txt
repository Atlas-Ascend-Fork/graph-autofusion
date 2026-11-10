# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_mlir_e2e_cpu_case(load_add_scalar_store_202x40
    OUTPUT_SHAPE "[202, 40]"
    INPUT_SHAPES "[202, 40]"
)

# 每次 API 调用需处理超过一个 repeat（f32 为 64 个元素），见 Case.cpp 的 L-2。
# count 不超过一个 repeat 时，Adds 的 isSetMask 为 true 或 false 结果相同，无法检出错误。
add_mlir_e2e_cpu_case(load_add_scalar_store_128x23x5
    OUTPUT_SHAPE "[128, 23, 5]"
    INPUT_SHAPES "[128, 23, 5]"
)
