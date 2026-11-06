# Copyright (c) 2026 Huawei Technologies Co, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.

# f16 二维转置分别验证 mem 与 reg 数值仿真。
add_mlir_e2e_cpu_case(load_transpose_store_2d_64x64

    HOST_TILING_API TRUE
    OUTPUT_SHAPE "[64, 64]"
    INPUT_SHAPES "[64, 64]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store_2d/Graph.cpp
                       ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store_2d/CaseMem.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store_2d/KernelTest.cpp
)

add_mlir_e2e_cpu_case(load_transpose_store_2d_reg_base_64x64

    HOST_TILING_API TRUE
    PLATFORM reg_base REQUIRES_COMPACT_SDK
    OUTPUT_SHAPE "[64, 64]"
    INPUT_SHAPES "[64, 64]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store_2d/Graph.cpp
                       ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store_2d/CaseRegBase.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store_2d/KernelTest.cpp
)
