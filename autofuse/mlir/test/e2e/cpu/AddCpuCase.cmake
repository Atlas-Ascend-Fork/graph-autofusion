# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# One generator, output directory and executable per case. Generated tiling
# classes and exported kernel symbols must never be linked across cases.
function(parse_mlir_e2e_cpu_shape shape_text shape_values_var element_count_var)
    string(STRIP "${shape_text}" shape_text)
    string(LENGTH "${shape_text}" shape_text_length)
    if(shape_text_length LESS 2)
        message(FATAL_ERROR "CPU E2E shape must use the form [dim0, dim1, ...]")
    endif()

    string(SUBSTRING "${shape_text}" 0 1 shape_first_char)
    math(EXPR shape_last_index "${shape_text_length} - 1")
    string(SUBSTRING "${shape_text}" ${shape_last_index} 1 shape_last_char)
    if(NOT shape_first_char STREQUAL "[" OR NOT shape_last_char STREQUAL "]")
        message(FATAL_ERROR "CPU E2E shape must use the form [dim0, dim1, ...]")
    endif()

    math(EXPR shape_body_length "${shape_text_length} - 2")
    string(SUBSTRING "${shape_text}" 1 ${shape_body_length} shape_body)
    string(STRIP "${shape_body}" shape_body)
    if(shape_body STREQUAL "")
        message(FATAL_ERROR "CPU E2E shape must contain at least one dimension")
    endif()
    string(REPLACE "," ";" raw_shape_values "${shape_body}")
    set(shape_values "")
    set(element_count 1)
    set(max_element_count 4294967295)
    foreach(raw_dimension IN LISTS raw_shape_values)
        string(STRIP "${raw_dimension}" dimension)
        if(NOT dimension MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "CPU E2E shape values must be positive integers")
        endif()
        string(LENGTH "${dimension}" dimension_length)
        if(dimension_length GREATER 10 OR
           (dimension_length EQUAL 10 AND dimension STRGREATER "${max_element_count}"))
            message(FATAL_ERROR "CPU E2E shape element count must fit uint32_t")
        endif()
        list(APPEND shape_values "${dimension}")
        math(EXPR max_dimension "${max_element_count} / ${element_count}")
        if(dimension GREATER max_dimension)
            message(FATAL_ERROR "CPU E2E shape element count must fit uint32_t")
        endif()
        math(EXPR element_count "${element_count} * ${dimension}")
    endforeach()

    set(${shape_values_var} "${shape_values}" PARENT_SCOPE)
    set(${element_count_var} "${element_count}" PARENT_SCOPE)
endfunction()

# 用例默认跑在 `AF_MLIR_E2E_CPU_SOC_VERSION`（Ascend910B1）上，也就是 **mem-based 一侧**。
# 给 `PLATFORM` 传一个名字就能换到别的平台；今天只认 `reg_base`。
#
# **换平台要同时改三处，少一处就不是真的换了**：
#
#   1. `soc_version`（进 `CodegenOptions`）——MLIR 侧按它查 TargetProfile，决定
#      `RegBasedApiFamily` 是 Available 还是 Unavailable，也就是选哪一侧的 implementation；
#   2. **NpuArch 字符串**——legacy 那一半（optimizer / ATT / AscIR 的 codegen impl）全部按它
#      分流：`REG_ASC_IR(...).Impl(v1_soc_versions, ...)` 注册的键就是 `"2201"`，v2 那份是
#      `"3510"`。只改第 1 处的话，调度算的还是 v1 的 tmp buffer 尺寸，而 MLIR 选的是 reg 侧
#      的实现——两边对不上，`checkTmpBuffer` 当场判 unsupported；
#   3. tikicpulib 目标——CPU 模拟器按芯片分库，链错了编得过但跑的是另一颗芯片。
#
# NpuArch 那一处走 `PlatformContext::SetPlatformInfo`（见 Generator.cpp），不改共用的
# runtime stub：stub 是整个测试套件共用的，把它的返回值改成可配会牵连到所有用例。
function(add_mlir_e2e_cpu_case name)
    cmake_parse_arguments(PARSE_ARGV 1 CASE "REQUIRES_COMPACT_SDK" "PLATFORM;TILE_ACCESSOR;BLOCK_INNER_ACCESSOR;HOST_TILING_API"
        "GENERATOR_SOURCES;TEST_SOURCES;OUTPUT_SHAPE;INPUT_SHAPES;COVERS;ENVIRONMENT")
    if(CASE_REQUIRES_COMPACT_SDK)
        if(NOT CASE_PLATFORM STREQUAL "reg_base")
            message(FATAL_ERROR "${name}: REQUIRES_COMPACT_SDK requires PLATFORM reg_base")
        endif()
        if(NOT AF_MLIR_E2E_CPU_REG_BASE_SDK_PROBED)
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" "${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/support/prepare_reg_base_sdk.py"
                        "${AF_MLIR_E2E_CPU_REG_BASE_NATIVE_SDK}" "${AF_MLIR_E2E_CPU_REG_BASE_SDK}"
                RESULT_VARIABLE _AF_MLIR_E2E_CPU_REG_BASE_SDK_RESULT
            )
            set(AF_MLIR_E2E_CPU_REG_BASE_SDK_PROBED ON)
            set(AF_MLIR_E2E_CPU_REG_BASE_SDK_PROBED ON PARENT_SCOPE)
            if(NOT _AF_MLIR_E2E_CPU_REG_BASE_SDK_RESULT EQUAL 0 OR
               NOT EXISTS "${AF_MLIR_E2E_CPU_REG_BASE_SDK}/impl/basic_api/kernel_operator_data_copy_intf_impl.h")
                message(FATAL_ERROR "${name}: failed to prepare Compact DMA SDK from ${AF_MLIR_E2E_CPU_REG_BASE_NATIVE_SDK}")
            endif()
        endif()
    endif()
    # 两份源码缺省就是用例目录下那两个固定名字——这个函数只从 `cases/<dir>/CMakeLists.txt` 里
    # 调，所以 `CMAKE_CURRENT_SOURCE_DIR` 就是用例目录。显式传了就用传的那份（一个目录注册多个
    # 形状变体、或要多带一个源文件时需要），所以这只是去掉每个用例都要抄一遍、而且抄错了才发现
    # 的那两行路径。
    if(NOT CASE_GENERATOR_SOURCES)
        set(CASE_GENERATOR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Case.cpp")
    endif()
    if(NOT CASE_TEST_SOURCES)
        set(CASE_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/KernelTest.cpp")
    endif()
    foreach(source IN LISTS CASE_GENERATOR_SOURCES CASE_TEST_SOURCES)
        if(NOT EXISTS "${source}")
            message(FATAL_ERROR "${name}: source '${source}' does not exist")
        endif()
    endforeach()
    if(CASE_UNPARSED_ARGUMENTS OR CASE_KEYWORDS_MISSING_VALUES OR
       NOT CASE_OUTPUT_SHAPE OR NOT CASE_INPUT_SHAPES)
        message(FATAL_ERROR "${name}: expected OUTPUT_SHAPE and INPUT_SHAPES")
    endif()
    # The generated TilingData names its fields after the scheduled axes, so the
    # names differ per registration even when several share one test source
    # (z0t_size for [128] against z0z1z2t_size for [128, 23, 5]). Naming them here,
    # next to the shape that produces them, is what lets the test assert which
    # terms of the transfer offset it actually exercises instead of assuming.
    # 覆盖断言是**选填**的：给了就在运行期断言，没给就不断言。
    #
    # 原先三个参数都是必填。那条契约在用例形状由本仓自己定的时候成立，之后不再成立——
    # 形状被大量重塑（`load_abs_storex128 [128]` -> `load_abs_store_2x297x309x1`），而
    # 字段名（`get_z0t_size` 对 `get_z0z1z2z3t_size`）和覆盖项都是跟着形状走的。必填只会
    # 逼着新用例填一个没推导过的值，那比不填更糟：这套机制防的就是「声明与实际不符还能
    # 通过」，用猜来的值填满它等于把它拆了。
    #
    # 所以判据从「必须声明」改成「声明了就必须对」。消费端 `E2eTilingShape.h` 本来就是
    # `#if defined(...)` 保护的，不给宏自然降级成不断言。
    if((CASE_TILE_ACCESSOR AND NOT CASE_BLOCK_INNER_ACCESSOR) OR
       (CASE_BLOCK_INNER_ACCESSOR AND NOT CASE_TILE_ACCESSOR))
        message(FATAL_ERROR "${name}: TILE_ACCESSOR and BLOCK_INNER_ACCESSOR name the generated TilingData "
                            "getters for the vectorized axis extent and the tiles-per-core extent; give both or "
                            "neither")
    endif()
    if(CASE_COVERS AND NOT CASE_TILE_ACCESSOR)
        message(FATAL_ERROR "${name}: COVERS needs TILE_ACCESSOR and BLOCK_INNER_ACCESSOR to be asserted against")
    endif()

    # COVERS states which terms of the scheduled-transfer offset
    # `(blockIdx * block_inner + round) * tile` this registration's shape actually
    # reaches, and the test asserts it at run time. ATT decides the split, so a case
    # that only checks its output values can keep passing after it has stopped
    # exercising the term it was registered for. `none` is a deliberate choice, not a
    # default: it says the offset is identically zero here and the case is about
    # something else.
    set(known_coverage multi_block multi_round tail none)
    set(coverage_definitions "")
    foreach(coverage_token IN LISTS CASE_COVERS)
        if(NOT coverage_token IN_LIST known_coverage)
            message(FATAL_ERROR "${name}: unknown COVERS value '${coverage_token}'; expected one of ${known_coverage}")
        endif()
        if(coverage_token STREQUAL "multi_block")
            list(APPEND coverage_definitions AF_MLIR_E2E_CPU_COVERS_MULTI_BLOCK=1)
        elseif(coverage_token STREQUAL "multi_round")
            list(APPEND coverage_definitions AF_MLIR_E2E_CPU_COVERS_MULTI_ROUND=1)
        elseif(coverage_token STREQUAL "tail")
            list(APPEND coverage_definitions AF_MLIR_E2E_CPU_COVERS_TAIL=1)
        endif()
    endforeach()
    list(LENGTH CASE_COVERS coverage_count)
    if("none" IN_LIST CASE_COVERS AND coverage_count GREATER 1)
        message(FATAL_ERROR "${name}: COVERS 'none' cannot be combined with other values")
    endif()

    set(accessor_definitions "")
    if(CASE_TILE_ACCESSOR)
        list(APPEND accessor_definitions
            AF_MLIR_E2E_CPU_TILE_ACCESSOR=${CASE_TILE_ACCESSOR}
            AF_MLIR_E2E_CPU_BLOCK_INNER_ACCESSOR=${CASE_BLOCK_INNER_ACCESSOR})
    endif()

    set(case_soc_version "${AF_MLIR_E2E_CPU_SOC_VERSION}")
    set(case_tikicpulib "${AF_MLIR_E2E_CPU_TIKICPULIB_TARGET}")
    set(case_platform_defs "")

    # Merge the MLIR route with any case-specific DFX flags instead of assigning
    # AUTOFUSE_DFX_FLAGS twice; cmake -E env does not promise which duplicate wins.
    set(case_dfx_flags "AUTOFUSE_DFX_FLAGS=--mlir_codegen_config=mlir")
    set(case_environment "")
    foreach(env_entry IN LISTS CASE_ENVIRONMENT)
        if(env_entry MATCHES "^AUTOFUSE_DFX_FLAGS=(.*)$")
            string(APPEND case_dfx_flags ";${CMAKE_MATCH_1}")
        else()
            list(APPEND case_environment "${env_entry}")
        endif()
    endforeach()
    if(CASE_PLATFORM)
        if(NOT CASE_PLATFORM STREQUAL "reg_base")
            message(FATAL_ERROR "${name}: unknown PLATFORM '${CASE_PLATFORM}' (only 'reg_base')")
        endif()
        # 数值抄自 target/Ascend/Profiles/platform_config/Ascend950PR_9599.ini，
        # 与 MLIR 侧查的是同一份 profile，不要在这里另起一套。
        set(case_soc_version "${AF_MLIR_E2E_CPU_REG_BASE_SOC_VERSION}")
        set(case_tikicpulib "${AF_MLIR_E2E_CPU_REG_BASE_TIKICPULIB_TARGET}")
        set(case_platform_defs
            AF_MLIR_E2E_CPU_NPU_ARCH="3510"
            AF_MLIR_E2E_CPU_AIV_NUM=72
            AF_MLIR_E2E_CPU_UB_SIZE=253952
            AF_MLIR_E2E_CPU_L2_SIZE=134217728
        )
    endif()
    if(NOT TARGET ${case_tikicpulib})
        message(FATAL_ERROR "${name}: tikicpulib target '${case_tikicpulib}' is unavailable")
    endif()

    # OUTPUT_SHAPE and INPUT_SHAPES are ordered by output/Data index respectively.
    list(LENGTH CASE_OUTPUT_SHAPE output_shape_count)
    if(output_shape_count LESS 1)
        message(FATAL_ERROR "${name}: OUTPUT_SHAPE must contain at least one shape")
    endif()
    list(LENGTH CASE_INPUT_SHAPES input_shape_count)
    if(input_shape_count LESS 1)
        message(FATAL_ERROR "${name}: INPUT_SHAPES must contain at least one shape")
    endif()

    set(output_shape_configs "")
    foreach(output_shape_text IN LISTS CASE_OUTPUT_SHAPE)
        parse_mlir_e2e_cpu_shape("${output_shape_text}" output_shape output_element_count)
        string(JOIN ", " output_shape_values ${output_shape})
        list(APPEND output_shape_configs "{{${output_shape_values}}, ${output_element_count}U}")
    endforeach()

    set(input_shape_configs "")
    foreach(input_shape_text IN LISTS CASE_INPUT_SHAPES)
        parse_mlir_e2e_cpu_shape("${input_shape_text}" input_shape input_element_count)
        string(JOIN ", " input_shape_values ${input_shape})
        list(APPEND input_shape_configs "{{${input_shape_values}}, ${input_element_count}U}")
    endforeach()

    set(target "MlirE2eCpu_${name}")
    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${name}")
    set(kernel_source "${generated_dir}/kernel.cpp")
    set(tiling_header "${generated_dir}/autofuse_tiling_data.h")
    set(tiling_source "${generated_dir}/tiling.cpp")
    set(case_config_header "${generated_dir}/CaseConfig.h")
    string(JOIN ",\n    " output_shape_config_values ${output_shape_configs})
    string(JOIN ",\n    " input_shape_config_values ${input_shape_configs})
    set(case_config_content "#pragma once\n#include \"test/e2e/cpu/support/E2eCaseConfig.h\"\n\n")
    string(APPEND case_config_content
        "namespace mlir::afir::test_config {\n"
        "constexpr std::initializer_list<ShapeConfig> kOutputShapes = {\n"
        "    ${output_shape_config_values}\n"
        "};\n"
        "constexpr std::initializer_list<ShapeConfig> kInputShapes = {\n"
        "    ${input_shape_config_values}\n"
        "};\n"
        "}  // namespace mlir::afir::test_config\n")
    file(GENERATE OUTPUT "${case_config_header}" CONTENT "${case_config_content}")

    add_executable(${target}_generator
        ${CASE_GENERATOR_SOURCES}
        ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/support/Generator.cpp
        ${MLIR_CODE_ROOT_DIR}/test/e2e/cpu/support/KernelWriter.cpp
    )
    target_include_directories(${target}_generator PRIVATE
        ${CODE_ROOT_DIR}/optimize
        ${CODE_ROOT_DIR}/codegen
        "${generated_dir}"
    )
    # Optimizer platform queries must work on a host without an NPU.
    target_link_libraries(${target}_generator PRIVATE
        MlirE2eCpuGraphBuilder GTest::gtest GTest::gtest_main
        -Wl,--no-as-needed ${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET} -Wl,--as-needed
    )
    add_dependencies(${target}_generator MlirE2eCpuTilingStub)
    target_compile_definitions(${target}_generator PRIVATE
        AF_MLIR_E2E_CPU_TILING_LIBRARY="$<TARGET_FILE:MlirE2eCpuTilingStub>"
        AF_MLIR_E2E_CPU_SOC_VERSION="${case_soc_version}"
        ${case_platform_defs}
        AF_MLIR_E2E_CPU_KERNEL_SOURCE="${kernel_source}"
        AF_MLIR_E2E_CPU_TILING_HEADER="${tiling_header}"
        AF_MLIR_E2E_CPU_TILING_SOURCE="${tiling_source}"
        AF_MLIR_E2E_CPU_KERNEL_SYMBOL=mlir_${name}
    )

    add_custom_command(
        OUTPUT ${kernel_source} ${tiling_header} ${tiling_source}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
        COMMAND ${CMAKE_COMMAND} -E chdir "${generated_dir}" ${CMAKE_COMMAND} -E env
            ${case_environment}
            "${case_dfx_flags}"
            "LD_LIBRARY_PATH=${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
            $<TARGET_FILE:${target}_generator>
        DEPENDS ${target}_generator MlirE2eCpuTilingStub
        COMMENT "Generating MLIR CPU E2E sources: ${name}"
        VERBATIM
    )
    add_executable(${target} ${CASE_TEST_SOURCES} ${kernel_source} ${tiling_header} ${tiling_source})
    target_include_directories(${target} PRIVATE "${generated_dir}" ${MLIR_CODE_ROOT_DIR})
    # MLIR-only test builds do not enter tests/CMakeLists.txt, so runtime_headers
    # does not always expose the CANN logging headers used by generated tiling.
    target_include_directories(${target} SYSTEM PRIVATE
        "${ASCEND_INSTALL_PATH}/pkg_inc/base"
    )
    # reg 侧包装声明的 CANN 官方头（如 `TransposeExtend` 的
    # `basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h`，见
    # `OpVecTranspose.td` 的 `ApiExtensionHeader` 第三段）不在 tikicpulib 目标的传递
    # include 里，由这里补上——目录下恰好只有 `basic_api/` 这一层前缀。
    if(CASE_PLATFORM)
        set(case_sdk "${AF_MLIR_E2E_CPU_REG_BASE_NATIVE_SDK}")
        if(CASE_REQUIRES_COMPACT_SDK)
            set(case_sdk "${AF_MLIR_E2E_CPU_REG_BASE_SDK}")
        endif()
        target_include_directories(${target} SYSTEM BEFORE PRIVATE
            "${case_sdk}/include"
            "${case_sdk}/include/basic_api"
            "${case_sdk}/include/interface"
            "${case_sdk}/impl"
        )
    endif()
    target_compile_definitions(${target} PRIVATE
        AUTO_FUSE_DEVICE=1
        AF_MLIR_E2E_CPU_KERNEL_SYMBOL=mlir_${name}
        AF_MLIR_E2E_CPU_CASE_NAME="${name}"
        ${accessor_definitions}
        ${coverage_definitions}
    )
    target_link_libraries(${target} PRIVATE
        ${case_tikicpulib} GTest::gtest GTest::gtest_main pthread runtime_headers unified_dlog
    )
    # 走 host tiling API 的 plan（pad/transpose 一类），生成的 tiling.cpp 引用
    # `AscendC::PadTilingFunc`、`optiling::TilingDef::SaveToBuffer`、`ge::Shape` 等符号。
    # 库集合与真实交付流逐字一致（`ascendc_compile.py` 的 `HOST_LINK_LIBRARIES`）。这些
    # CANN 库之间存在交付流（`-shared`）才允许的悬空交叉引用——exe 链接显式放开，
    # 运行时由测试环境 LD_LIBRARY_PATH 里的 CANN lib64 解析。按用例 opt-in：纯算子的
    # tiling.cpp 没有这些引用，不改动它们的链接面。
    if(CASE_HOST_TILING_API)
        set(host_tiling_libs "")
        foreach(host_tiling_lib tiling_api platform graph_base register acl_rt)
            find_library(case_host_tiling_lib_${host_tiling_lib}
                NAMES ${host_tiling_lib}
                PATHS ${ASCEND_INSTALL_PATH}/lib64
                NO_DEFAULT_PATH)
            if(NOT case_host_tiling_lib_${host_tiling_lib})
                message(FATAL_ERROR
                    "${name}: host tiling library '${host_tiling_lib}' not found under ${ASCEND_INSTALL_PATH}/lib64")
            endif()
            list(APPEND host_tiling_libs "${case_host_tiling_lib_${host_tiling_lib}}")
        endforeach()
        target_link_options(${target} PRIVATE -Wl,--allow-shlib-undefined)
        target_link_libraries(${target} PRIVATE ${host_tiling_libs})
    endif()

    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
        LABELS "mlir_e2e;mlir_e2e_cpu"
        WORKING_DIRECTORY "${generated_dir}"
        ENVIRONMENT "LD_LIBRARY_PATH=${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
    )
    add_custom_target(check-e2e-cpu-${name}
        COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}"
            $<TARGET_FILE:${target}>
        DEPENDS ${target}
        COMMENT "Running MLIR CPU E2E case: ${name}"
        WORKING_DIRECTORY "${generated_dir}"
        USES_TERMINAL
    )
    add_dependencies(check-e2e-cpu check-e2e-cpu-${name})
endfunction()
