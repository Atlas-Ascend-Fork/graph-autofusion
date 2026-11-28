# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

foreach(width IN ITEMS 16 32)
    foreach(operation IN ITEMS Add Maximum Minimum)
        string(TOLOWER "${operation}" mnemonic)
        set(case_name "load_ui${width}_scalar_${mnemonic}_reg_base_store_100x221")
        add_mlir_e2e_cpu_case(${case_name} PLATFORM reg_base
            OUTPUT_SHAPE "[100, 221]" INPUT_SHAPES "[100, 221]")
        foreach(target IN ITEMS MlirE2eCpu_${case_name} MlirE2eCpu_${case_name}_generator)
            target_compile_definitions(${target} PRIVATE AFIR_SCALAR_WIDTH=${width} AFIR_SCALAR_${operation}=1)
        endforeach()
    endforeach()
endforeach()
