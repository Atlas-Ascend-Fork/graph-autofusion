# Copyright (c) 2026 Huawei Technologies Co, Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# `afir.transpose` 的数值证据：`load -> transpose[0,2,1] -> store`，perm 落在内层轴、分块轴
# 两侧都在最外——调度保留转置的标准形态。
#
# 形状**不是随手挑的**（与 legacy 期望件 `tests/st/codegen/e2e/load_transpose_store_expect_code`
# 同一组）：`{2, 37, 23}` 的 37 与 23 都不对齐（`% 8`、`% 16` 都有余），尾块路径
# （`firstAxisRem` / `secondAxisRem`）与 MLIR 侧的段尾块修正都被踩到；小而对齐的形状对
# 这两条路全盲。
#
# mem 实例共用一张图（`Graph.cpp`），断言分家：mem 侧（默认 910B1）发
# `codegen::ConfusionTranspose` + host tiling，reg 侧（`PLATFORM reg_base`，950PR）发
# `TransposeExtend`——同一张图两个家族各走各的，这正是 tier D2 的「API 名 / 扩展头 /
# tmp / host tiling」四项差集。
add_mlir_e2e_cpu_case(load_transpose_store_2x37x23

    HOST_TILING_API TRUE
    OUTPUT_SHAPE "[2, 23, 37]"
    INPUT_SHAPES "[2, 37, 23]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/Graph.cpp
                       ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/CaseMem.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/KernelTest.cpp
)
# 对齐形状的对照格：`{4, 16, 32}` 全部对齐（`% 8 == 0`、`% 16 == 0`），尾块路径
# （`firstAxisRem` / `secondAxisRem`）不走——与非对齐格一起才把两条路都钉住。
add_mlir_e2e_cpu_case(load_transpose_store_4x16x32

    HOST_TILING_API TRUE
    OUTPUT_SHAPE "[4, 32, 16]"
    INPUT_SHAPES "[4, 16, 32]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/Graph.cpp
                       ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/CaseMem.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/KernelTest.cpp
)
# reg_base 与 mem 共用同一张图和非对齐形状，单独走 TransposeExtend 与 950/reg
# 的 kernel 数值仿真，不能只由 lit 的生成文本检查代替。
add_mlir_e2e_cpu_case(load_transpose_store_reg_base_2x37x23

    HOST_TILING_API TRUE
    PLATFORM reg_base REQUIRES_COMPACT_SDK
    OUTPUT_SHAPE "[2, 23, 37]"
    INPUT_SHAPES "[2, 37, 23]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/Graph.cpp
                       ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/CaseRegBase.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/KernelTest.cpp
)

add_mlir_e2e_cpu_case(load_transpose_store_reg_base_4x16x32

    HOST_TILING_API TRUE
    PLATFORM reg_base REQUIRES_COMPACT_SDK
    OUTPUT_SHAPE "[4, 32, 16]"
    INPUT_SHAPES "[4, 16, 32]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/Graph.cpp
                       ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/CaseRegBase.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store/KernelTest.cpp
)
