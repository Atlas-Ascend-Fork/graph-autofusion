# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_mlir_e2e_cpu_case(load_add_store_1x67x209
    OUTPUT_SHAPE "[1, 67, 209]"
    INPUT_SHAPES "[1, 67, 209]" "[1, 67, 209]"
)

add_mlir_e2e_cpu_case(load_add_store_128x23x5
    OUTPUT_SHAPE "[128, 23, 5]"
    INPUT_SHAPES "[128, 23, 5]" "[128, 23, 5]"
    TILE_ACCESSOR get_z0z1z2t_size
    BLOCK_INNER_ACCESSOR get_z0z1z2Tb_size
    # 3 核 x 每核 1 轮 x 尾块 4896：覆盖 blockIdx 项与尾块，不覆盖归纳变量项。
    COVERS multi_block tail
)

add_mlir_e2e_cpu_case(load_add_store_128x128x128
    OUTPUT_SHAPE "[128, 128, 128]"
    INPUT_SHAPES "[128, 128, 128]" "[128, 128, 128]"
    TILE_ACCESSOR get_z0z1z2t_size
    BLOCK_INNER_ACCESSOR get_z0z1z2Tb_size
    # 48 核 x 每核 6 轮 x 尾块 5496：本套件里唯一压住完整偏移式的注册。
    COVERS multi_block multi_round tail
)
