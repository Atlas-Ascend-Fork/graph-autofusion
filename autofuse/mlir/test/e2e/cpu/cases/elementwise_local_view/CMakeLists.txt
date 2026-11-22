# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Direct AFIR input is intentional: the scheduler normally gives equal UB views,
# whereas this regression exercises the lowering contract for independent strides.
set(generated "${CMAKE_CURRENT_BINARY_DIR}/local_view.cpp")
add_custom_command(OUTPUT "${generated}"
    COMMAND $<TARGET_FILE:af-opt> --afir-to-ascendc-pipeline --cse --target=${AF_MLIR_E2E_CPU_SOC_VERSION}
            "${CMAKE_CURRENT_SOURCE_DIR}/Kernel.mlir" -o "${CMAKE_CURRENT_BINARY_DIR}/local_view.mlir"
    COMMAND $<TARGET_FILE:ascir-translate> -mlir-to-ascendc
            "${CMAKE_CURRENT_BINARY_DIR}/local_view.mlir" -o "${generated}"
    DEPENDS af-opt ascir-translate "${CMAKE_CURRENT_SOURCE_DIR}/Kernel.mlir"
    VERBATIM)
add_executable(MlirE2eCpuLocalView KernelTest.cpp "${generated}")
# The generated functions are included in the device wrapper's translation unit.
set_source_files_properties("${generated}" PROPERTIES HEADER_FILE_ONLY TRUE)
target_include_directories(MlirE2eCpuLocalView PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${CODE_ROOT_DIR}")
target_compile_definitions(MlirE2eCpuLocalView PRIVATE AUTO_FUSE_DEVICE=1)
target_link_libraries(MlirE2eCpuLocalView PRIVATE ${AF_MLIR_E2E_CPU_TIKICPULIB_TARGET}
    GTest::gtest GTest::gtest_main pthread runtime_headers unified_dlog)
set(local_view_runtime "$<JOIN:$<TARGET_PROPERTY:${AF_MLIR_E2E_CPU_TIKICPULIB_TARGET},INTERFACE_LINK_DIRECTORIES>,:>:${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH}")
add_test(NAME MlirE2eCpuLocalView COMMAND MlirE2eCpuLocalView)
set_tests_properties(MlirE2eCpuLocalView PROPERTIES LABELS "mlir_e2e;mlir_e2e_cpu"
    ENVIRONMENT "LD_LIBRARY_PATH=${local_view_runtime}")
add_custom_target(check-e2e-cpu-local-view
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${local_view_runtime}" $<TARGET_FILE:MlirE2eCpuLocalView>
    DEPENDS MlirE2eCpuLocalView USES_TERMINAL)
add_dependencies(check-e2e-cpu check-e2e-cpu-local-view)
