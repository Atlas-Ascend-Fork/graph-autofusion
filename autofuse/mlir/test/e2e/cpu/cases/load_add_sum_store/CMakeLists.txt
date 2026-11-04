# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.

# 归约之前融合一个**二元**算子，两侧都是张量（理由见 Case.cpp）。
#
# 这是 reduce 融合语料里第一条两个张量输入的：其余几条是一元（`load_neg_sum_store` 的 Neg、
# `load_sum_relu_store` 的 Relu）或带 `Scalar`（`load_sum_mul_store`）。带 `Scalar` 那条的第二个
# 操作数不是张量、不参与 tile 寻址，所以覆盖不到「同一个有效视图上两份局部跨度」这件事。
#
# 形状与 `load_neg_sum_store_16x37` 同一组，便于一元 / 二元两支对账：R 轴 37 不对齐，Add 覆盖
# 到 padding 之后 `ReduceInit` 仍要把补出来的位置恢复成 sum 的单位元。
add_mlir_e2e_cpu_case(load_add_sum_store_16x37
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR graph0_result0_g0_tiling_data.get_z0t_size
    BLOCK_INNER_ACCESSOR graph0_result0_g0_tiling_data.get_z0Tb_size
    COVERS tail
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]" "[16, 37]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_sum_store/KernelTest.cpp
)
