# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 使用 [256, 128]，与同族 VF 用例对照。尾轴 512B 满足 32B 对齐，
# 将测试重点限定在算子融合；32768 个元素可覆盖多轮 lane 循环。
add_mlir_e2e_cpu_case(load_add_abs_reg_base_store_256x128
    PLATFORM reg_base
    OUTPUT_SHAPE "[256, 128]"
    INPUT_SHAPES "[256, 128]" "[256, 128]"
)
