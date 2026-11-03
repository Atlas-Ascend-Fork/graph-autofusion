# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reduce schedules yield several plans that the static selector cannot rank
# yet; plan 0 is the single-stage common template.
add_mlir_e2e_cpu_case(load_sum_store_16x32
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 32]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# 归约轴不按 32B 对齐：`ReduceInit` 要把补齐出来的位置填成 0，否则它们会被加进和里。
add_mlir_e2e_cpu_case(load_sum_store_16x37
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# 保留轴有尾块：最后一个 tile 的 A 维短一截，`first_actual` 要跟着变。
add_mlir_e2e_cpu_case(load_sum_store_17x32
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z0Tb_size
    COVERS none
    OUTPUT_SHAPE "[17, 1]"
    INPUT_SHAPES "[17, 32]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# RA：归约**首维**，tile 布局 `[R, A]`。与上面三条 AR 用例互为镜像，覆盖的是
# `ReduceInit` 不发、`srcShape` 两项对调、32B 对齐落在 A 上这三件 AR 下看不出来的事。
# 三档形状分别压 R 尾块、A 尾块和两者同时；R 被切成多个 tile，所以跨轮累加也走到。
add_mlir_e2e_cpu_case(load_sum_store_ra_37x16
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z1Tb_size
    COVERS none
    OUTPUT_SHAPE "[1, 16]"
    INPUT_SHAPES "[37, 16]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# A 维不按 32B 对齐：RA 下 A 是物理**内**维，对齐补出来的是额外结果槽，不会 Store 回 GM。
# 这条用例守的就是「那些槽位的垃圾值不污染有效输出」。
add_mlir_e2e_cpu_case(load_sum_store_ra_32x17
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z1Tb_size
    COVERS none
    OUTPUT_SHAPE "[1, 17]"
    INPUT_SHAPES "[32, 17]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

add_mlir_e2e_cpu_case(load_sum_store_ra_37x17
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    TILE_ACCESSOR get_z0t_size
    BLOCK_INNER_ACCESSOR get_z1Tb_size
    COVERS none
    OUTPUT_SHAPE "[1, 17]"
    INPUT_SHAPES "[37, 17]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# Real ARAR scheduler templates. Their transfer effective views contain 4/3/3/2
# extents respectively; all four local layouts are A...A,R...R and therefore
# merge to the same two-dimensional ReduceSum AR contract.
#
# 这一条**不钉模板**：四个模板全部下降进同一个 kernel，由入口按 stage 的 `tiling_key` 分派，
# ATT 在 host 侧择优写这个 key。原先是 `arar_case0..3` 四条各钉一个模板，那样四个模板的下降
# 都走到了，但分派这条路一条都没覆盖——而它恰恰是唯一会静默出错的地方（编号塌掉时多出来的
# 模板变成死代码，数值断言照样全过）。`Case.cpp` 的 `CheckTemplatesAreDispatched` 把这件事
# 变成断言。代价是只有胜出的那个模板会真的执行，另外三个只覆盖到下降，不覆盖数值。
#
# 不声明 TILE_ACCESSOR：切分轴随运行期选中的模板变（case0/1 是 z0t，case2/3 是 z2t），注册期
# 填哪个都会有一半时候在说假话。缺省降级成「观测不到」，那是照实说。
add_mlir_e2e_cpu_case(load_sum_store_arar
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    OUTPUT_SHAPE "[2, 1, 4, 1]"
    INPUT_SHAPES "[2, 3, 4, 5]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# ---------------------------------------------------------------------------
# AllLoad（全载）模板：R 轴整根一次进 UB，单相、不跨轮、不切图、不要 workspace。
#
# ATT 强制选择保证 AllLoad 在运行期被执行；仅把它编进 kernel 不能提供数值覆盖。
# 二维图的候选顺序为 RCommon、RCore、AllLoad，因此指定 plan 2；
# 三维 ARA 图当前为 General/RCore/AllLoad 三个候选，因此指定 plan 2。CPU wrapper 会核验执行的 plan 编号。
# 若调度器改变候选构成，还需通过 AFIR 的 is_reduce_mem_reuse 核验模板身份：
#
#   AUTOFUSE_DFX_FLAGS="--mlir_codegen_config=mlir" \
#   AUTOFUSE_DFX_FLAGS="--force_schedule_result=<n>;codegen_compile_debug=true;debug_dir=<dir>" \
#   LD_LIBRARY_PATH=<见该用例 build.make 里生成器那条命令> \
#     <build>/.../MlirE2eCpu_<case>_generator
#
# `is_reduce_mem_reuse = true` 是 AllLoad 的唯一标记（`optimize/optimize.cpp:1094` 只在
# `kAllLoad` 置真）。模板本身的回归不依赖这个下标——由
# `test/lit/conversion/afir-to-ascendc-reduce-allload-{ar,ra}.mlir` 在 IR 层独立钉住。
#
# 「至少一条不钉 plan 的 AllLoad 语料」由 `load_sum_mul_store_8x1024` 承担（它不钉 plan，
# 其 AllLoad 候选与其余候选一起编进同一个 kernel），这里不再重复一条。
#
# 不声明 TILE_ACCESSOR / COVERS：AllLoad 没有 R 切分位，切分轴字段与二维 RCommon 的注册对不上，
# 填上去会有一半时候在说假话。缺省降级成「观测不到」，那是照实说。

# 基准 AR：A / R 都对齐。
add_mlir_e2e_cpu_case(load_sum_store_allload_16x32
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=2"
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 32]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# R 尾块不对齐：整根 R 入 UB 时 37 不是 8 的倍数，srcShape 的内维要 align 到 40。
add_mlir_e2e_cpu_case(load_sum_store_allload_16x37
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=2"
    OUTPUT_SHAPE "[16, 1]"
    INPUT_SHAPES "[16, 37]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# A 尾块不对齐：17 行按核与 tile 切完末尾有残块，归约的 count 取的是 A 的 actual。
add_mlir_e2e_cpu_case(load_sum_store_allload_17x32
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=2"
    OUTPUT_SHAPE "[17, 1]"
    INPUT_SHAPES "[17, 32]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# 大 R：整根 1024 入 UB，盯的是「AllLoad 不切 R」这条前提在 UB 预算下仍然成立。
add_mlir_e2e_cpu_case(load_sum_store_allload_8x1024
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=2"
    OUTPUT_SHAPE "[8, 1]"
    INPUT_SHAPES "[8, 1024]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# ARA：AllLoad 下**唯一**能产出 RA 形态的语料（二维 RA 根本不生成 AllLoad 候选，见 `Case.cpp`
# 三维分支的注释），并且是 RA `srcShape` 公式里 **A 组连乘**项的首个用例。
add_mlir_e2e_cpu_case(load_sum_store_allload_ara_4x32x8
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=2"
    OUTPUT_SHAPE "[4, 1, 8]"
    INPUT_SHAPES "[4, 32, 8]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# ---------------------------------------------------------------------------
# RA 下 A 组多轴且最内 A 的 nominal 不是 32B 整数倍。
#
# RA 的 destination 镜像源的物理内维布局，槽位 `(a_outer, a_inner)` 落在
# `a_outer × align(A_inner_nominal) + a_inner`——有效槽位是**跨步**的。跨 tile 累加的 count
# 原先按 `align(Π A_actual)` 连续数，于是每个 tile 最后几行根本没被拷/累加进 destination。
# 两条形状是**回归**用例：修前分别在 A_outer = 2,5,8,… 和 4,9,14,… 整行错（每 tile 装 3 / 5
# 根 A_outer，count 24 / 32 对真实跨度 21 / 37），修后全绿。
#
# 为什么要两档：count 公式里有一项 `Σ (actual - 1) × stride`，单档只能钉住一个 A_outer
# 每-tile 数。`[4, 32, 5]` 这类小形状压不出来——UB 预算会把每 tile 的 A_outer 压成 1 行，
# 此时跨步与连续恰好等价，测不到东西。
#
# 不声明 TILE_ACCESSOR / COVERS：切分位随 UB 预算变，注册期填哪个都有一半时候在说假话。
add_mlir_e2e_cpu_case(load_sum_store_ra_agap_64x32x5
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    OUTPUT_SHAPE "[64, 1, 5]"
    INPUT_SHAPES "[64, 32, 5]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

add_mlir_e2e_cpu_case(load_sum_store_ra_agap_64x8x5
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    OUTPUT_SHAPE "[64, 1, 5]"
    INPUT_SHAPES "[64, 8, 5]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

# ---------------------------------------------------------------------------
# RCore（R 轴也参与分核）：两相，相间过 GM workspace。
#
# 这两条钉 plan 1。候选顺序是 General -> RCore -> AllLoad，二维图是 1 + 1 + 1，所以 RCore 是
# plan 1。身份判据不是下标而是产物形状：RCore 必须出 `stage0_template*` 与 `stage1_template*`
# 两个 stage，且 kernel 里出现 workspace。AR 输入还会**同时**出现
# `Pattern::Reduce::AR` 和 `::RA`——phase 1 在 `[A, Rm_t]` 上是 AR，phase 2 在 workspace 的
# `[Rm, A]` 上是 RA。这正是 `2026-09-21-afir-reduce-sum-ra-lowering-design.md` 3.3 说的
# 「RA 是 RCore phase 2 的硬前置」。
#
# 除 ATT 默认选择外，KernelTest 显式覆盖 (a,r,k)=(2,256,2)/(3,120,3)，检查两阶段
# 多轮 scratch 复用、A/R 尾块、workspace partial 和最终输出。CPU 模拟验证计算语义；
# 实机覆盖仍由 test/e2e/npu/reduce_template_matrix.py 的 rcore_ar/rcore_ra 承担。
#
# 不声明 TILE_ACCESSOR / COVERS：两个 stage 的切分位不同，注册期只能填一个，填哪个都在说假话。
add_mlir_e2e_cpu_case(load_sum_store_rcore_ar_8x1024
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=1"
    OUTPUT_SHAPE "[8, 1]"
    INPUT_SHAPES "[8, 1024]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

add_mlir_e2e_cpu_case(load_sum_store_rcore_ra_1024x8
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=1"
    OUTPUT_SHAPE "[1, 8]"
    INPUT_SHAPES "[1024, 8]"
    GENERATOR_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/Case.cpp
    TEST_SOURCES ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store/KernelTest.cpp
)

target_compile_definitions(MlirE2eCpu_load_sum_store_rcore_ar_8x1024 PRIVATE AF_RCORE_SCRATCH_TEST=1)
target_compile_definitions(MlirE2eCpu_load_sum_store_rcore_ra_1024x8 PRIVATE AF_RCORE_SCRATCH_TEST=1 AF_RCORE_RA=1)

# Four post-reduce operations force a workspace boundary; their composition is identity.
add_mlir_e2e_cpu_case(load_sum_partition
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0"
    OUTPUT_SHAPE "[17, 1]" INPUT_SHAPES "[17, 37]"
    GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Case.cpp
    TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/KernelTest.cpp)
foreach(target MlirE2eCpu_load_sum_partition_generator MlirE2eCpu_load_sum_partition)
    target_compile_definitions(${target} PRIVATE AF_REDUCE_PARTITION=1)
endforeach()

# Parallel graph branches, lowered into one AR implementation with shared API scratch.
add_mlir_e2e_cpu_case(load_sum_max_store_ar_8x1024
    ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=2"
    OUTPUT_SHAPE "[8, 1]" INPUT_SHAPES "[8, 1024]"
    GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Case.cpp
    TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/KernelTest.cpp)
foreach(target MlirE2eCpu_load_sum_max_store_ar_8x1024_generator MlirE2eCpu_load_sum_max_store_ar_8x1024)
    target_compile_definitions(${target} PRIVATE AF_SUM_MAX_TEST=1)
endforeach()
