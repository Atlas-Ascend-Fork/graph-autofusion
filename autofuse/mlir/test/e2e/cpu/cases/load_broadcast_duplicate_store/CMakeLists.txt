# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 一根轴：源 tile 恒为 1 个元素，喂它的 Load 要从 GM 搬 4 个字节。这条路此前撞
# `DataCopy` 的 32B 对齐（2026-09-23 设计文档的 2.4 (c)），改成一律发 `DataCopyPadExtend`
# 之后通了。
add_mlir_e2e_cpu_case(load_broadcast_duplicate_store_128
    OUTPUT_SHAPE "[128]"
    INPUT_SHAPES "[1]"
)

# 两根轴：tile 是二维的，`Duplicate` 的判据仍然满足（源在两根轴上 repeat 都是 1）。
add_mlir_e2e_cpu_case(load_broadcast_duplicate_store_8x128
    OUTPUT_SHAPE "[8, 128]"
    INPUT_SHAPES "[1, 1]"
)

# Non-block-aligned output exercises Duplicate count and the final Store tail.
add_mlir_e2e_cpu_case(load_broadcast_duplicate_store_131
    OUTPUT_SHAPE "[131]"
    INPUT_SHAPES "[1]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_duplicate_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_duplicate_store/KernelTest.cpp
)
