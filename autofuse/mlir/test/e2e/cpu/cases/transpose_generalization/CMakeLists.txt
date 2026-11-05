# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# CPU 仿真只保留有稳定 CPU legacy/MLIR 对照意义的 Transpose→Add 形态。
# Cast、级联、广播和 Reduce 组合的有效证据来自 910B3 真机 NPU runner，见 test/e2e/npu/。
add_mlir_e2e_cpu_case(transpose_add_2x37x23 HOST_TILING_API TRUE
    INPUT_SHAPES "[2,37,23]" "[2,23,37]" OUTPUT_SHAPE "[2,23,37]")
add_mlir_e2e_cpu_case(transpose_add_4x16x32 HOST_TILING_API TRUE
    INPUT_SHAPES "[4,16,32]" "[4,32,16]" OUTPUT_SHAPE "[4,32,16]")
add_mlir_e2e_cpu_case(transpose_add_reg_base_2x37x23 HOST_TILING_API TRUE PLATFORM reg_base REQUIRES_COMPACT_SDK
    INPUT_SHAPES "[2,37,23]" "[2,23,37]" OUTPUT_SHAPE "[2,23,37]")
add_mlir_e2e_cpu_case(transpose_add_reg_base_4x16x32 HOST_TILING_API TRUE PLATFORM reg_base REQUIRES_COMPACT_SDK
    INPUT_SHAPES "[4,16,32]" "[4,32,16]" OUTPUT_SHAPE "[4,32,16]")
foreach(case_name transpose_add_2x37x23 transpose_add_4x16x32
                  transpose_add_reg_base_2x37x23 transpose_add_reg_base_4x16x32)
    foreach(target MlirE2eCpu_${case_name} MlirE2eCpu_${case_name}_generator)
        if(TARGET ${target})
            target_compile_definitions(${target} PRIVATE AF_TRANSPOSE_CASE=2)
        endif()
    endforeach()
endforeach()
