# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(AF_MLIR_E2E_CPU_SOC_VERSION "Ascend910B1" CACHE STRING "SoC version used by MLIR CPU E2E tests")
set(AF_MLIR_E2E_CPU_TIKICPULIB_TARGET "tikicpulib_ascend910B1" CACHE STRING
    "tikicpulib target used by MLIR CPU E2E tests")

# reg-base（NpuArch 3510）那一侧。用例按 `PLATFORM reg_base` 切过来，见 AddCpuCase.cmake。
set(AF_MLIR_E2E_CPU_REG_BASE_SOC_VERSION "Ascend950PR_9599" CACHE STRING
    "SoC version used by reg-base MLIR CPU E2E tests")
set(AF_MLIR_E2E_CPU_REG_BASE_TIKICPULIB_TARGET "tikicpulib_ascend950pr_9599" CACHE STRING
    "tikicpulib target used by reg-base MLIR CPU E2E tests")

# The SDK installation stays read-only. Correct Compact-mode CPU bounds checks
# in a build-local header tree; Normal mode and the DMA implementation are unchanged.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(AF_MLIR_E2E_CPU_REG_BASE_SDK "${CMAKE_CURRENT_BINARY_DIR}/reg_base_sdk")
set(AF_MLIR_E2E_CPU_REG_BASE_SDK_PROBED OFF)
# Ordinary reg-base cases use the original SDK, without requiring a patched copy.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_AF_MLIR_CPU_SDK_ARCH aarch64-linux)
else()
    set(_AF_MLIR_CPU_SDK_ARCH x86_64-linux)
endif()
set(AF_MLIR_E2E_CPU_REG_BASE_NATIVE_SDK "${ASCEND_INSTALL_PATH}/${_AF_MLIR_CPU_SDK_ARCH}/asc")
if(NOT IS_DIRECTORY "${AF_MLIR_E2E_CPU_REG_BASE_NATIVE_SDK}/include" AND
   IS_DIRECTORY "${ASCEND_INSTALL_PATH}/toolkit/${_AF_MLIR_CPU_SDK_ARCH}/asc/include")
    set(AF_MLIR_E2E_CPU_REG_BASE_NATIVE_SDK "${ASCEND_INSTALL_PATH}/toolkit/${_AF_MLIR_CPU_SDK_ARCH}/asc")
endif()


list(APPEND CMAKE_PREFIX_PATH "${ASCEND_INSTALL_PATH}/tools/tikicpulib/lib")
find_package(tikicpulib REQUIRED)
if(NOT TARGET ${AF_MLIR_E2E_CPU_TIKICPULIB_TARGET})
    message(FATAL_ERROR "tikicpulib target '${AF_MLIR_E2E_CPU_TIKICPULIB_TARGET}' is unavailable")
endif()

# Some CANN packages publish stale nested include paths. Keep only paths that
# exist so the imported target remains usable in local CPU environments.
foreach(_AF_MLIR_E2E_CPU_TARGET
        ${AF_MLIR_E2E_CPU_TIKICPULIB_TARGET} ${AF_MLIR_E2E_CPU_REG_BASE_TIKICPULIB_TARGET})
    if(NOT TARGET ${_AF_MLIR_E2E_CPU_TARGET})
        continue()
    endif()
    get_target_property(_AF_MLIR_E2E_CPU_TIKICPULIB_INCLUDES
        ${_AF_MLIR_E2E_CPU_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    if(_AF_MLIR_E2E_CPU_TIKICPULIB_INCLUDES)
        set(_AF_MLIR_E2E_CPU_VALID_INCLUDES "")
        foreach(_AF_MLIR_E2E_CPU_INCLUDE IN LISTS _AF_MLIR_E2E_CPU_TIKICPULIB_INCLUDES)
            if(IS_DIRECTORY "${_AF_MLIR_E2E_CPU_INCLUDE}")
                list(APPEND _AF_MLIR_E2E_CPU_VALID_INCLUDES "${_AF_MLIR_E2E_CPU_INCLUDE}")
            endif()
        endforeach()
        set_target_properties(${_AF_MLIR_E2E_CPU_TARGET} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_AF_MLIR_E2E_CPU_VALID_INCLUDES}")
    endif()
endforeach()

add_library(MlirE2eCpuTilingStub SHARED ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/support/TilingStub.cpp)

add_library(MlirE2eCpuGraphBuilder STATIC
    ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph/asc_graph_builder.cpp
)
target_include_directories(MlirE2eCpuGraphBuilder PUBLIC
    ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph
    ${MLIR_CODE_ROOT_DIR}
)
target_link_libraries(MlirE2eCpuGraphBuilder PUBLIC aihac_codegen metadef)

add_custom_target(check-e2e-cpu COMMENT "Running MLIR CPU end-to-end tests")
add_dependencies(check-e2e check-e2e-cpu)

include("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/AddCpuCase.cmake")
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_store" cases/load_add_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_store_direct" cases/load_store_direct)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/workspace_readback" cases/workspace_readback)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_duplicate_store" cases/load_broadcast_duplicate_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_segmented_store" cases/load_broadcast_segmented_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_abs_store" cases/load_broadcast_abs_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_add_store" cases/load_broadcast_add_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_chain_store" cases/load_broadcast_chain_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_cast_store" cases/load_broadcast_cast_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_two_output_store" cases/load_broadcast_two_output_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_diamond_store" cases/load_broadcast_diamond_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_multi_axis_dynamic_store" cases/load_broadcast_multi_axis_dynamic_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_nddma_store" cases/load_broadcast_nddma_store)
# `load_transpose_store` / `load_transpose_store_2d`：`afir.transpose` 的数值证据
# （`load -> transpose[0,2,1] -> store`）。3 档实例：mem 默认（910B，a2/a3）两形状
# 非对齐/对齐各一；`PLATFORM reg_base`（950，a5）一例；2d 为 64x64 对照格。
# 设计文档 `2026-10-19-afir-transpose-e2e-design.md`（2026-10-27 v2 迁移记录）。
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store" cases/load_transpose_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_transpose_store_2d" cases/load_transpose_store_2d)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/sub_transpose_abs" cases/sub_transpose_abs)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/transpose_generalization" cases/transpose_generalization)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_store" cases/load_sum_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mean_store" cases/load_mean_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_neg_sum_store" cases/load_neg_sum_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_sum_store" cases/load_add_sum_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_add_store" cases/load_sum_add_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_mul_store" cases/load_sum_mul_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum_relu_store" cases/load_sum_relu_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum0_mul_store" cases/load_sum0_mul_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sum0_store" cases/load_sum0_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_broadcast_bf16_reg_base_store" cases/load_broadcast_bf16_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add3_store" cases/load_add3_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_diamond_store" cases/load_add_diamond_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_abs_diamond_store" cases/load_abs_diamond_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_abs_store" cases/load_abs_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_exp_sqrt_store" cases/load_exp_sqrt_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_neg_store" cases/load_neg_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_neg_reg_base_store" cases/load_neg_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_abs_reg_base_store" cases/load_add_abs_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_cast_abs_cast_reg_base_store" cases/load_cast_abs_cast_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_abs_neg_reg_base_store" cases/load_abs_neg_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_eq_where_reg_base_store" cases/load_eq_where_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mul_abs_reg_base_store" cases/load_mul_abs_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_reciprocal_store" cases/load_reciprocal_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_abs_reciprocal_store" cases/load_abs_reciprocal_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_isnan_store" cases/load_isnan_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_isfinite_store" cases/load_isfinite_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_isinf_store" cases/load_isinf_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_select_store" cases/load_select_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_leaky_relu_store" cases/load_leaky_relu_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_clip_by_value_store" cases/load_clip_by_value_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_clip_by_value_scalar_store" cases/load_clip_by_value_scalar_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_bitwise_and_store" cases/load_bitwise_and_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_floor_div_store" cases/load_floor_div_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_remainder_store" cases/load_remainder_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sub_store" cases/load_sub_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_div_store" cases/load_div_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_gelu_store" cases/load_gelu_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sign_store" cases/load_sign_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sign_reg_base_store" cases/load_sign_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_logical_and_store" cases/load_logical_and_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_logical_or_store" cases/load_logical_or_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_axpy_store" cases/load_axpy_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_axpy_half_store" cases/load_axpy_half_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_truediv_store" cases/load_truediv_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_tanh_store" cases/load_tanh_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_sigmoid_store" cases/load_sigmoid_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_pow_store" cases/load_pow_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_cast_store" cases/load_cast_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_cast_reg_base_store" cases/load_cast_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_rsqrt_store" cases/load_rsqrt_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_logical_not_store" cases/load_logical_not_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_eq_store" cases/load_eq_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_eq_reg_base_store" cases/load_eq_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_eq_select_store" cases/load_eq_select_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_mask_select_store" cases/load_mask_select_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_scalar_store" cases/load_add_scalar_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_scalar_add_store" cases/load_scalar_add_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_scalar_sub_store" cases/load_scalar_sub_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_scalar_data_store" cases/load_add_scalar_data_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_index_expr_store" cases/load_add_index_expr_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_add_index_expr_dynamic_store" cases/load_add_index_expr_dynamic_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_scalar_div_store" cases/load_scalar_div_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_truediv_scalar_store" cases/load_truediv_scalar_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_scalar_pow_store" cases/load_scalar_pow_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_select_scalar_store" cases/load_select_scalar_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_where_scalar_store" cases/load_where_scalar_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_scalar_div_reg_base_store" cases/load_scalar_div_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_scalar_pow_reg_base_store" cases/load_scalar_pow_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_select_scalar_reg_base_store" cases/load_select_scalar_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_clip_by_value_scalar_reg_base_store" cases/load_clip_by_value_scalar_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_eq_scalar_reg_base_store" cases/load_eq_scalar_reg_base_store)
add_subdirectory("${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/cases/load_logical_and_scalar_reg_base_store" cases/load_logical_and_scalar_reg_base_store)

# Run both contracts independently; the legacy source/header remain byte-identical to upstream.
set(reduce_padding_header "${CMAKE_CURRENT_BINARY_DIR}/reduce_padding_generated.h")
add_custom_command(OUTPUT "${reduce_padding_header}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/probes/generate_reduce_padding.py"
        --af-opt $<TARGET_FILE:af-opt> --translate $<TARGET_FILE:ascir-translate>
        --output "${reduce_padding_header}" --target "${AF_MLIR_E2E_CPU_SOC_VERSION}"
    DEPENDS af-opt ascir-translate probes/generate_reduce_padding.py
    VERBATIM)
add_executable(MlirReduceInitTest probes/ReduceInitTest.cpp "${reduce_padding_header}")
target_include_directories(MlirReduceInitTest PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
add_executable(LegacyReduceInitTest ${CODE_ROOT_DIR}/tests/ut/ascendc/api/test_reduce_init.cpp)
foreach(target MlirReduceInitTest LegacyReduceInitTest)
    target_include_directories(${target} PRIVATE
        ${MLIR_CODE_ROOT_DIR}/api/ascendc/extensions
        ${CODE_ROOT_DIR}/ascendc/api
        ${CODE_ROOT_DIR}/tests/ut/ascendc/api)
    target_link_libraries(${target} PRIVATE
        ${AF_MLIR_E2E_CPU_TIKICPULIB_TARGET} GTest::gtest GTest::gtest_main pthread)
endforeach()
add_custom_target(check-e2e-cpu-reduce-init
    COMMAND $<TARGET_FILE:MlirReduceInitTest>
    COMMAND $<TARGET_FILE:LegacyReduceInitTest>
    DEPENDS MlirReduceInitTest LegacyReduceInitTest)
add_dependencies(check-e2e-cpu check-e2e-cpu-reduce-init)

add_custom_target(check-e2e-cpu-reduce-isolation DEPENDS
    check-e2e-cpu-load_sum_partition
    check-e2e-cpu-load_mean_partition
    check-e2e-cpu-reduce-init
    check-e2e-cpu-load_mean_store_16x32
    check-e2e-cpu-load_mean_store_16x37
    check-e2e-cpu-load_mean_store_8x65536
    check-e2e-cpu-load_mean_store_8x65537
    check-e2e-cpu-load_mean_store_8x1024
    check-e2e-cpu-load_mean_store_ra_37x16
    check-e2e-cpu-load_sum_store_16x37
    check-e2e-cpu-load_sum_store_arar
    check-e2e-cpu-load_sum_store_rcore_ar_8x1024
    check-e2e-cpu-load_sum_store_rcore_ra_1024x8)

# Shared addressing consumers: tails, block/round offsets, scalar operands,
# padded broadcast rows, and a direct transfer with output guards.
add_custom_target(check-e2e-cpu-nonreduce-addressing DEPENDS
    check-e2e-cpu-load_store_direct_1x67x209
    check-e2e-cpu-load_store_direct_128x128x128
    check-e2e-cpu-load_add_store_128x128x128
    check-e2e-cpu-load_abs_store_2x297x309x1
    check-e2e-cpu-load_add3_store_3x5x2x382
    check-e2e-cpu-load_scalar_add_store_201x39
    check-e2e-cpu-load_select_store_4x48x69x2
    check-e2e-cpu-load_broadcast_duplicate_store_131
    check-e2e-cpu-load_broadcast_segmented_store_1x5
    check-e2e-cpu-load_broadcast_segmented_store_8x1
    check-e2e-cpu-load_broadcast_chain_store_1x1x8)
