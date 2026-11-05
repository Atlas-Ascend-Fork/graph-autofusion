# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Legacy backend_e2e/sub_transpose_abs_test and its PGO variant share this graph.
add_mlir_e2e_cpu_case(sub_transpose_abs_32x64x512
    HOST_TILING_API TRUE
    OUTPUT_SHAPE "[64, 32, 512]"
    INPUT_SHAPES "[32, 64, 512]" "[32, 64, 512]"
)
add_mlir_e2e_cpu_case(sub_transpose_abs_5x37x23
    HOST_TILING_API TRUE
    OUTPUT_SHAPE "[37, 5, 23]"
    INPUT_SHAPES "[5, 37, 23]" "[5, 37, 23]"
)

add_mlir_e2e_cpu_case(sub_transpose_abs_reg_base_32x64x512
    HOST_TILING_API TRUE PLATFORM reg_base REQUIRES_COMPACT_SDK
    OUTPUT_SHAPE "[64, 32, 512]"
    INPUT_SHAPES "[32, 64, 512]" "[32, 64, 512]"
)
add_mlir_e2e_cpu_case(sub_transpose_abs_reg_base_5x37x23
    HOST_TILING_API TRUE PLATFORM reg_base REQUIRES_COMPACT_SDK
    OUTPUT_SHAPE "[37, 5, 23]"
    INPUT_SHAPES "[5, 37, 23]" "[5, 37, 23]"
)
