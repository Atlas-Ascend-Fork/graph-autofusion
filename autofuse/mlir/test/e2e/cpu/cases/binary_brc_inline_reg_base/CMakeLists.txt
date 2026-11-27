# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Post-inline inputs use aligned row strides, as required by legacy BrcInlineTemplateV2.
# Padded logical tails and alternating axes are covered by direct lowering lit cases.
set(_brc_ops Add Mul Minimum Maximum Sub Div TrueDiv)
set(_brc_apis "AscendC::Reg::Add(" "AscendC::Reg::Mul(" "AscendC::Reg::Min(" "AscendC::Reg::Max(" "AscendC::Reg::Sub(" "AscendC::Reg::Div(" "AscendC::Reg::Div(")
foreach(kind RANGE 0 6)
    list(GET _brc_ops ${kind} op)
    list(GET _brc_apis ${kind} api)
    string(TOLOWER "${op}" op_name)
    foreach(side lhs rhs)
        set(lhs "[1, 128]")
        set(rhs "[257, 128]")
        if(side STREQUAL "rhs")
            set(lhs "[257, 128]")
            set(rhs "[1, 128]")
        endif()
        set(name "binary_brc_inline_reg_base_${op_name}_${side}")
        add_mlir_e2e_cpu_case(${name}
            PLATFORM reg_base
            ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=0;--force_tiling_case=0"
            OUTPUT_SHAPE "[257, 128]"
            INPUT_SHAPES "${lhs}" "${rhs}"
        )
        target_compile_definitions(MlirE2eCpu_${name}_generator PRIVATE AF_BRC_OP=${op} AF_BRC_API="${api}")
        target_compile_definitions(MlirE2eCpu_${name} PRIVATE AF_BRC_KIND=${kind})
    endforeach()
endforeach()
