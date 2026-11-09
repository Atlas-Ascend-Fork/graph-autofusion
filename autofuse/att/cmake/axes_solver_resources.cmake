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
set(ATT_AXES_CORE_DIR "${CODE_ROOT_DIR}/att/core/axes_reorder")
set(ATT_AXES_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/axes_solver_resources")
set(ATT_AXES_RESOURCE_HEADER "${ATT_AXES_RESOURCE_DIR}/axes_solver_resources.h")
add_custom_command(
    OUTPUT "${ATT_AXES_RESOURCE_HEADER}"
    COMMAND "${Python3_EXECUTABLE}" "${CODE_ROOT_DIR}/att/cmake/embed_axes_solver.py"
            "${ATT_AXES_CORE_DIR}" "${ATT_AXES_RESOURCE_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${ATT_AXES_RESOURCE_HEADER}"
    DEPENDS "${CODE_ROOT_DIR}/att/cmake/embed_axes_solver.py"
            "${ATT_AXES_CORE_DIR}/axes_reorder_head.inc"
            "${ATT_AXES_CORE_DIR}/axes_reorder_body.inc"
            "${ATT_AXES_CORE_DIR}/../scalar/expression_helpers.inc"
            "${ATT_AXES_CORE_DIR}/../scalar/scalar_helpers.inc"
            "${ATT_AXES_CORE_DIR}/../scalar/axes_ceil_div.inc"
    VERBATIM
)
add_custom_target(att_axes_solver_resource_text DEPENDS "${ATT_AXES_RESOURCE_HEADER}")
add_library(att_axes_solver_resources INTERFACE)
add_dependencies(att_axes_solver_resources att_axes_solver_resource_text)
target_include_directories(att_axes_solver_resources INTERFACE "${ATT_AXES_RESOURCE_DIR}")
