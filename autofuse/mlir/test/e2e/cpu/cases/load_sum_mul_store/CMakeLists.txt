# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY IMPLIED WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.

# 归约之后接一个带外部输入的节点（这里是标量乘）。`ReducePartitionPostFusion` 会把图切成
# 两张，归约结果经 GM workspace 交接——`torch.mean` 走的就是这条路。
#
# **不声明 TILE_ACCESSOR / COVERS**：切图之后每个 group 有自己的 TilingData 结构体
# （`AscGraph0ScheduleResult0G0TilingData` / `...G1TilingData`），`z0t_size` 在它们里面各有
# 一份，而 `AF_E2E_TILING_SHAPE` 是按**扁平** `AutofuseTilingData` 取访问器的——写上去编不过。
# 于是这条用例照实不提覆盖声明（形状仍会打印，tile / block_inner 显示为 0 即「观测不到」），
# 它盯的是 workspace 交接那条路，不是搬运偏移的各项。
add_mlir_e2e_cpu_case(load_sum_mul_store_16x37
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_mul_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_mul_store/KernelTest.cpp
)

# 同一张图换成 `A=8 < corenum` 且**不钉 plan**：切图与 RCore 两相切分同时在场。
# 上面那条（`[16, 37]` 钉 plan 0）只覆盖了切图本身，覆盖不到「切图之后的那张图再被 RCore
# 切成两相」——真机上正是这一档挂的，而钉了 plan 的注册永远编不到那个候选。
add_mlir_e2e_cpu_case(load_sum_mul_store_8x1024
    OUTPUT_SHAPE "[8, 1]"
    INPUT_SHAPES "[8, 1024]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_mul_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_mul_store/KernelTest.cpp
)

# Execute the RCore writer followed by the fused workspace reader. Compile all
# candidates as usual, and use ATT's selection to verify this cross-stage path.
add_mlir_e2e_cpu_case(load_sum_mul_store_workspace_rcore
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=1"
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]"
    GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Case.cpp
    TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/KernelTest.cpp
)
