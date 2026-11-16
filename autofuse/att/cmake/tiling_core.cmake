# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
include_guard(GLOBAL)
# S1-J2J3-EXACT-02/05: compile the internal Program implementation for product and tests.
function(add_att_tiling_core target)
    add_library(${target} OBJECT
        ${CODE_ROOT_DIR}/att/serialization/aftp_v1.cpp
        ${CODE_ROOT_DIR}/att/serialization/tiling_stage_codec.cpp
        ${CODE_ROOT_DIR}/att/serialization/tiling_result_layout_codec.cpp
        ${CODE_ROOT_DIR}/att/serialization/axes_model_codec.cpp
        ${CODE_ROOT_DIR}/att/serialization/axes_callback_codec.cpp
        ${CODE_ROOT_DIR}/att/serialization/axes_evaluation_codec.cpp
        ${CODE_ROOT_DIR}/att/program/tiling_program_value.cpp
        ${CODE_ROOT_DIR}/att/program/tiling_expression.cpp
        ${CODE_ROOT_DIR}/att/program/tiling_expression_lowering.cpp
        ${CODE_ROOT_DIR}/att/program/tiling_stage.cpp
        ${CODE_ROOT_DIR}/att/program/tiling_result_writer.cpp
        ${CODE_ROOT_DIR}/att/program/axes_program_evaluation.cpp
        ${CODE_ROOT_DIR}/att/program/axes_callback_bindings.cpp
        ${CODE_ROOT_DIR}/att/program/axes_model_evaluation.cpp
    )
    target_include_directories(${target} PUBLIC ${CODE_ROOT_DIR}/att)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE
        -fno-common -Wfloat-equal -Wextra -Wall -Werror
        $<$<NOT:$<BOOL:${RUN_TEST}>>:-O2>
    )
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()
