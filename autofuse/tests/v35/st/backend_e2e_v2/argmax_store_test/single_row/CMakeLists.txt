# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

backend_e2e_st_test(argmax_single_row_test
    CODEGEN ../argmax_store_backend_generator.cpp
    KERNEL_SRC
        argmax_store_test_kernel.cpp
        argmax_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC ../test_e2e_argmax_store_kernel.cpp)
target_compile_definitions(argmax_single_row_test_codegen_v2 PRIVATE ARGMAX_SINGLE_ROW)
target_compile_definitions(argmax_single_row_test_e2e_v2 PRIVATE ARGMAX_SINGLE_ROW)
