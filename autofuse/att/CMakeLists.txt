# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
include(${CODE_ROOT_DIR}/att/cmake/generate_raw_string_header.cmake)
generate_raw_string_header(
    ge_log_utils            # 目标名称
    ge_log_utils_src.h      # 输出文件名
    ${CODE_ROOT_DIR}/att/util/common_dfx.h  # 输入文件
    "#define UTILS_COMMON_DFX_H_"  # 起始标记
    "#endif"                 # 结束标记
)

file(GLOB_RECURSE ATT_SRCS CONFIGURE_DEPENDS
        "${CODE_ROOT_DIR}/att/generator/*.cpp"
        "${CODE_ROOT_DIR}/att/dtree/*.cpp"
        "${CODE_ROOT_DIR}/att/util/*.cpp"
        "${CODE_ROOT_DIR}/att/gen_model_info/*.cpp"
        "${CODE_ROOT_DIR}/att/gen_model_info/utils/*.cpp")

target_sources(aihac_codegen PRIVATE
    ${CODE_ROOT_DIR}/att/gen_tiling_impl.cpp
    ${ATT_SRCS}
)

target_include_directories(aihac_codegen PRIVATE
    ${CODE_ROOT_DIR}/att
    ${CODE_ROOT_DIR}/att/../common
    ${CODE_ROOT_DIR}/att/../inc/
    ${CODE_ROOT_DIR}/att/base
    ${CODE_ROOT_DIR}/att/util
    ${CODE_ROOT_DIR}/att/common_utils
    ${CODE_ROOT_DIR}/att/generator
    ${CODE_ROOT_DIR}/att/generator/preprocess
    ${CODE_ROOT_DIR}/att/generator/solver_pass_gen
    ${CODE_ROOT_DIR}/att/generator/cache
    ${CODE_ROOT_DIR}/att/ascir
    ${CODE_ROOT_DIR}/att/ascir/generator
    ${CODE_ROOT_DIR}/att/gen_model_info
    ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
)

include(${CODE_ROOT_DIR}/att/cmake/axes_solver_resources.cmake)
target_link_libraries(aihac_codegen PRIVATE att_axes_solver_resources)
