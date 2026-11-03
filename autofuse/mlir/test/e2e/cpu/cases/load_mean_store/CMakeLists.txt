# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# `Mean` 的数值用例。与 `load_sum_store` 同一张图、同一组模板，差别只有算子，所以这里**不**
# 重复覆盖 AR/RA 布局、尾块、AllLoad、RCore 那一整套——那些由求和那组承担。这里只挑能让
# 「多出来的那一次缩放」出错的形状：
#
#   - 分母错（取某一轮的 actual 而不是完整域）  -> 需要 R 被切成多轮
#   - 重复除（每轮都除一次）                    -> 需要 R 被切成多轮，且轮数 > 1
#   - 分母把 32B 对齐的填充算进去                -> 需要 R 不是 8 的倍数
#   - RA 下分母取错了边（拿 A 当 R）             -> 需要一条 RA
#
# 四条注册各对应其中一条，没有一条是「再跑一遍求和已经覆盖的东西」。
#
# 都钉 plan 0（RCommon 单相模板）：Mean 的候选集与 Sum 同构，钉 plan 0 才能保证跑到的是
# R 被切分的那个模板，而不是运行期择优选中的 AllLoad（那一档不跨轮，测不到末轮缩放）。

# 基准 AR：A / R 都对齐。R = 32 时切分出的轮数由 UB 预算定，可能只有一轮——此时末轮条件在
# 第 0 轮就成立，钉的是「单轮也必须除一次」。
add_mlir_e2e_cpu_case(load_mean_store_16x32
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 32]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/KernelTest.cpp
)

# R 不是 8 的倍数：`ReduceInit` 把 37..39 填成 0 让它们不进和里，而**分母必须是 37**。
# 取 `align(37) = 40` 当分母的错是静默偏小 7.5%，没有任何一条形状断言能发现它——只有这条
# 数值用例能。
add_mlir_e2e_cpu_case(load_mean_store_16x37
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/KernelTest.cpp
)

# R 大到一个 UB tile 装不下，**必然**多轮：这条是整组里唯一能把「每轮都除」和「只在末轮除」
# 分开的形状。每轮除一次会让结果塌成 `S / N^rounds`，差几个数量级。
#
# 为什么是 65536 而不是 1024：UB 是 196608 字节 = 49152 个 f32，`8 x 1024` 只有 8192 个元素，
# 整张图一轮就算完了——R 压根不被切，`accumulationAxes` 为空，末轮条件那段代码不发。
# **这件事是实测出来的**：把 `Muls` 改成每轮都发（反向对照），`8 x 1024` 与 `16 x 37` 照样
# 全绿，说明那两条对跨轮逻辑零区分度。`8 x 65536` 是 512K 个元素、2MB，必须切。
#
# 改这条形状之前先重做那个反向对照：它是本用例有效性的唯一凭据，形状一变就得重验。
add_mlir_e2e_cpu_case(load_mean_store_8x65536
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[8, 1]"
    INPUT_SHAPES "[8, 65536]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/KernelTest.cpp
)

# 多轮**且末轮长度不同**：65537 不是任何 tile 尺寸的整数倍，最后一轮必然短一截。
# 上面那条 65536 是 2 的幂，各轮等长，测不到「末轮是尾块」这一档——而尾块那一轮同时要
# 满足两件事：末轮条件照样只触发一次，分母照样是 65537 而不是 Σ 各轮 actual 或末轮 actual。
add_mlir_e2e_cpu_case(load_mean_store_8x65537
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[8, 1]"
    INPUT_SHAPES "[8, 65537]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/KernelTest.cpp
)

# `8 x 1024`：一轮算完的大 R。它与上面那条的差别只有「切不切」，所以两条一起才说明分母在
# 两种形态下取的是同一个 N。单独留着它不覆盖跨轮——见上面那段。
add_mlir_e2e_cpu_case(load_mean_store_8x1024
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[8, 1]"
    INPUT_SHAPES "[8, 1024]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/KernelTest.cpp
)

# RA：归约**首维**，R 是 tile 的物理外维且不是 8 的倍数。分母取成 A（16）而不是 R（37）
# 的错在 AR 下看不出来——AR 的分母恰好就是内维；这条是唯一能把两者分开的。
add_mlir_e2e_cpu_case(load_mean_store_ra_37x16
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z1Tb_size
    COVERS none
    OUTPUT_SHAPE "[1, 16]"
    INPUT_SHAPES "[37, 16]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store/KernelTest.cpp
)

# Four post-reduce operations force a workspace boundary; their composition is identity.
add_mlir_e2e_cpu_case(load_mean_partition
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    OUTPUT_SHAPE "[17, 1]" INPUT_SHAPES "[17, 37]"
    GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Case.cpp
    TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/KernelTest.cpp)
foreach(target MlirE2eCpu_load_mean_partition_generator MlirE2eCpu_load_mean_partition)
    target_compile_definitions(${target} PRIVATE AF_REDUCE_PARTITION=1)
endforeach()

# Exercise the stage-local extent binding on the executed RCore partition path.
add_mlir_e2e_cpu_case(load_mean_partition_workspace_rcore
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=1"
    OUTPUT_SHAPE "[17, 1]" INPUT_SHAPES "[17, 37]"
    GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Case.cpp
    TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/KernelTest.cpp)
foreach(target MlirE2eCpu_load_mean_partition_workspace_rcore_generator MlirE2eCpu_load_mean_partition_workspace_rcore)
    target_compile_definitions(${target} PRIVATE AF_REDUCE_PARTITION=1)
endforeach()
