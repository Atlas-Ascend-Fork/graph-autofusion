# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.

# 归约之后融合一个一元算子，不切图的那一支（理由见 Case.cpp）。
#
# 不钉 plan：现场报在 RCore 二相图的归约节点上，钉了 plan 0 就编不到那个候选。形状按
# `A=8 < corenum` 取，让 RCore 成为候选。
#
# 不声明 TILE_ACCESSOR / COVERS：候选不止一个，切分形状随候选而变，观测不到单一的一组。
add_mlir_e2e_cpu_case(load_sum_relu_store_8x1024
    OUTPUT_SHAPE "[8, 1]"
    INPUT_SHAPES "[8, 1024]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_relu_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_relu_store/KernelTest.cpp
)
