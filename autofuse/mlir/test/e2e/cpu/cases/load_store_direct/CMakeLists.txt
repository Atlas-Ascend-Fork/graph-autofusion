# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_mlir_e2e_cpu_case(load_store_direct_1x67x209
    OUTPUT_SHAPE "[1, 67, 209]"
    INPUT_SHAPES "[1, 67, 209]"
)

add_mlir_e2e_cpu_case(load_store_direct_128x128x128
    OUTPUT_SHAPE "[128, 128, 128]"
    INPUT_SHAPES "[128, 128, 128]"
)
