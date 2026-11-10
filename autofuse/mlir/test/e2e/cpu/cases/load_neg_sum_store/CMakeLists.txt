# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.

# The reduction axis is not a multiple of 32 bytes, and the tiled local view
# must be converted to one elementwise count.
add_mlir_e2e_cpu_case(load_neg_sum_store_16x37
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR graph0_result0_g0_tiling_data.get_z0t_size
    BLOCK_INNER_ACCESSOR graph0_result0_g0_tiling_data.get_z0Tb_size
    COVERS tail
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_neg_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_neg_sum_store/KernelTest.cpp
)

# ARAR 图在归约前加入 Neg。候选 3 的搬运有效视图有两项，elementwise
# 应按整块局部跨度调用一维 API，再执行 AR 归约。保留全部模板，由 ATT 选择 case 3。
add_mlir_e2e_cpu_case(load_neg_sum_store_arar_case3
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0;--force_tiling_case=3"
    TILE_ACCESSOR graph0_result0_g0_tiling_data.get_z2t_size
    BLOCK_INNER_ACCESSOR graph0_result0_g0_tiling_data.get_z0z2Tb_size
    COVERS none
    OUTPUT_SHAPE "[2, 1, 4, 1]"
    INPUT_SHAPES "[2, 3, 4, 5]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_neg_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_neg_sum_store/KernelTest.cpp
)
