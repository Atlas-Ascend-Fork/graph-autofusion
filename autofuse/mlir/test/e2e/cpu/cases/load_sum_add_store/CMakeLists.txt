# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.

# 归约**之后**融合一个二元算子，第二个操作数是归约后形状的张量（`sum(x, 1) + bias`，
# 理由见 Case.cpp）。
#
# 二元后融合**必然切图**（第二个操作数天然不属于归约的后继链），所以这条覆盖的是「切图后的
# 消费者子图带一个真实的外部张量输入」——`load_sum_mul_store` 的消费者子图只有 workspace 一个
# GM 输入，覆盖不到这条。
#
# 不声明 TILE_ACCESSOR / COVERS：切成两个子图后切分轴与 tiling 字段名随子图变，注册期声明不出
# 单一的一组 accessor（声明了会编不过：`AutofuseTilingData` 里根本没有 `get_z0t_size`）。
add_mlir_e2e_cpu_case(load_sum_add_store_16x37
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]" "[16, 1]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_add_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_add_store/KernelTest.cpp
)
